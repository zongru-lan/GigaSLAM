import os
import time
import threading
import numpy as np
import torch
import torch.multiprocessing as mp
import cv2

from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from utils.camera_utils import Camera
from utils.logging_utils import Log, configure_logging
from utils.slam_utils import update_viewpoints_from_poses
from gaussian_splatting.gaussian_renderer import render

from utils.visual_odometry import ClassicTracking
from utils.slam_viz import draw_matches
from gui.gui_utils import GaussianPacket

from unidepth.models import UniDepthV2
from utils.railway_depth_model import RailwayDepthModel
from utils.rail_scale import GaugeFreeRailScaleStabilizer

import huggingface_hub


class FrontEnd(mp.Process):
    def __init__(self, config, dataset, save_dir = None):
        super().__init__()
        self.dataset = dataset
        self.save_dir = save_dir

        self.config = config
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None

        self.initialized = False
        self.kf_indices = []
        self.anchor_log = []
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []

        self.reset = True
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1
        self._gui_sync_count = 0

        self.gaussians = None
        self.cameras = dict()
        self.device = "cuda:0"
        self.pause = False

        cam_params = {}
        cam_params['fx'] = self.dataset.fx,
        cam_params['fy'] = self.dataset.fy,
        cam_params['cx'] = self.dataset.cx,
        cam_params['cy'] = self.dataset.cy,
        cam_params['width'] = self.dataset.width,
        cam_params['height'] = self.dataset.height,

        self.width = float(cam_params['width'][0])
        self.height = float(cam_params['height'][0])
        self.fx = float(cam_params['fx'][0])
        self.fy = float(cam_params['fy'][0])
        self.cx = float(cam_params['cx'][0])
        self.cy = float(cam_params['cy'][0])

        self.intrinsics = torch.tensor([
                [self.fx,   0,          self.cx ],
                [0,         self.fy,    self.cy ],
                [0,         0,          1       ]
            ], device='cuda')
        
        cam_params_ori = {}
        cam_params_ori['fx'] = self.dataset.fx_ori,
        cam_params_ori['fy'] = self.dataset.fy_ori,
        cam_params_ori['cx'] = self.dataset.cx_ori,
        cam_params_ori['cy'] = self.dataset.cy_ori,
        cam_params_ori['width'] = self.dataset.width_ori,
        cam_params_ori['height'] = self.dataset.height_ori,

        self.intrinsics_ori = torch.tensor([
                [self.dataset.fx_ori,   0,                      self.dataset.cx_ori ],
                [0,                     self.dataset.fy_ori,    self.dataset.cy_ori ],
                [0,                     0,                      1                   ]
            ], device='cuda')
        
        loop_enable = config['SLAM']['loop_closure']
        self.viz = config['SLAM']['viz']

        self.viz_frame_id_prev = 0
        self.viz_frame_id_curr = 0
        self.viz_frame_interval = 50

        self.classic_tracking = ClassicTracking(cam_params_ori, loop_enable, self.viz, config, save_dir = self.save_dir)

        self.motion_thresh = config['SLAM']['motion_thresh']

        self.keyframe_idx_list = []
        self.median_depth = 1.0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.depth_backend = config['DepthModel'].get('backend', 'unidepth')

        def _load_unidepth(cfg, dev):
            if cfg['from_huggingface']:
                return UniDepthV2.from_pretrained(
                    cfg['huggingface']['model_name'],
                    revision=cfg['huggingface']['commit_hash']).to(dev)
            return UniDepthV2.from_pretrained(cfg['local_snapshot_path']).to(dev)

        def _load_railway(cfg, dev):
            return RailwayDepthModel(
                ckpt_dir=cfg['railway_ckpt_dir'],
                model_name=cfg.get('railway_model_name', './Depth-Anything-V2-Small-hf'),
                device=str(dev))

        if self.depth_backend == 'railway':
            self.depth_model = _load_railway(config['DepthModel'], device)
        else:
            self.depth_model = _load_unidepth(config['DepthModel'], device)

        self.rail_scale = GaugeFreeRailScaleStabilizer(
            config.get('RailScale', {}),
            save_dir=self.save_dir,
        )

    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]

    def add_new_keyframe(self, cur_frame_idx, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        initial_depth = torch.tensor(viewpoint.depth, device='cuda').unsqueeze(0)
        initial_depth[~valid_rgb] = 0  # Ignore the invalid rgb pixels
        valid_depth = initial_depth[(initial_depth > 0) & valid_rgb]
        if valid_depth.numel() > 0:
            self.median_depth = float(torch.median(valid_depth).item())
        # record anchor count for growth curve
        if self.gaussians is not None and hasattr(self.gaussians, '_anchor'):
            self.anchor_log.append((cur_frame_idx, self.gaussians._anchor.shape[0]))
        return initial_depth[0].to(torch.float32)

    def initialize(self, cur_frame_idx, viewpoint):
        self.initialized = not self.monocular
        self.kf_indices = []
        self.anchor_log = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

        # Initialise the frame at the ground truth pose
        T = torch.eye(4, device=viewpoint.device)
        R_init = T[:3, :3]
        T_init = T[:3, 3]
        # viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
        viewpoint.update_RT(R_init, T_init)

        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.reset = False

    def tracking(self, cur_frame_idx, viewpoint):
        R, T = self.classic_tracking.get_pose()
        viewpoint.update_RT(R, T)


    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx
    ):
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]
        if last_keyframe_idx is None:
            return True

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        dist = float(torch.norm((pose_CW @ last_kf_WC)[0:3, 3]).item())
        median_depth = max(float(self.median_depth), 1e-6)
        dist_check = dist > kf_translation * median_depth
        dist_check2 = dist > kf_min_translation * median_depth
        interval_check = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval

        return interval_check or dist_check2 or dist_check

    def add_to_window(
        self, cur_frame_idx, window
    ):
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame

        removed_frame = None

        if len(window) > self.config["Training"]["window_size"]:


            removed_frame = window[-1]
            window.remove(removed_frame)

        return window, removed_frame

    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap):
        Log(f'Selected keyframe: frame {cur_frame_idx}')
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def reqeust_mapping(self, cur_frame_idx, viewpoint):
        msg = ["map", cur_frame_idx, viewpoint]
        self.backend_queue.put(msg)

    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        msg = ["init", cur_frame_idx, viewpoint, depth_map]
        self.backend_queue.put(msg)
        self.requested_init = True

    def request_loop_update(self, cur_frame_idx, poses_old, poses_update, kf_indices):
        msg = ["loop_update", cur_frame_idx, poses_old, poses_update, kf_indices]
        self.backend_queue.put(msg)

    def sync_backend(self, data):
        self.gaussians = data[1]
        occ_aware_visibility = data[2]
        keyframes = data[3]
        self.occ_aware_visibility = occ_aware_visibility

        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())

        if self.q_main2vis is not None and self.gaussians is not None:
            gaussians_ref = self.gaussians
            latest_cam = self.cameras.get(max(self.cameras.keys())) if self.cameras else None
            def _send():
                try:
                    packet = GaussianPacket(gaussians=gaussians_ref, current_frame=latest_cam)
                    self.q_main2vis.put(packet)
                    Log(f"GUI gaussian packet sent", tag="GUI")
                except BaseException as e:
                    import traceback
                    Log(f"GUI push error: {type(e).__name__}: {e}\n{traceback.format_exc()}", tag="GUI")
            threading.Thread(target=_send, daemon=True).start()

    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()

    def save_keyframe_viz(self, cur_frame_idx, viewpoint):
        if self.save_dir is None:
            return
        import os, imgviz
        save_dir = os.path.join(self.save_dir, "keyframe_viz")
        os.makedirs(save_dir, exist_ok=True)
        H, W = int(self.dataset.height_ori), int(self.dataset.width_ori)

        # 1. RGB
        if viewpoint.original_image is not None:
            rgb = (viewpoint.original_image.detach().cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
            rgb = cv2.resize(rgb, (W, H))
        else:
            rgb = np.zeros((H, W, 3), dtype=np.uint8)

        # 2. Depth
        if viewpoint.depth is not None:
            d = viewpoint.depth if isinstance(viewpoint.depth, np.ndarray) else viewpoint.depth.cpu().numpy()
            depth_color = imgviz.depth2rgb(d, min_value=0.1, max_value=float(np.max(d)) if np.max(d) > 0 else 255.0, colormap="jet")
            depth_color = cv2.resize(depth_color, (W, H))
        else:
            depth_color = np.zeros((H, W, 3), dtype=np.uint8)

        # 3. Gaussian render — use gaussians_main (has MLPs), update tensors from clone
        gaussians_main = getattr(self, 'gaussians_main', None)
        if gaussians_main is not None and self.gaussians is not None:
            try:
                for attr in ['_anchor', '_anchor_index', '_offset', '_anchor_feat',
                             '_scaling', '_rotation', '_opacity', '_level']:
                    val = getattr(self.gaussians, attr, None)
                    if val is not None:
                        setattr(gaussians_main, attr, val.cuda())
                if hasattr(self.gaussians, '_mlp_state_dicts'):
                    for attr, sd in self.gaussians._mlp_state_dicts.items():
                        module = getattr(gaussians_main, attr, None)
                        if module is None and attr == 'embedding_appearance':
                            from gaussian_splatting.scene.embedding import Embedding
                            num_cameras = next(iter(sd.values())).shape[0]
                            gaussians_main.embedding_appearance = Embedding(num_cameras, gaussians_main.appearance_dim).cuda()
                            module = gaussians_main.embedding_appearance
                        if module is not None:
                            module.load_state_dict({k: v.cuda() for k, v in sd.items()})
                n_anchors = gaussians_main._anchor.shape[0]
                Log(f"VIZ render: {n_anchors} anchors", tag="VIZ")
                with torch.no_grad():
                    render_pkg = render(viewpoint, gaussians_main, self.pipeline_params, self.background)
                rendering = render_pkg["render"].detach().cpu()
                rendering = torch.nn.functional.interpolate(
                    rendering.unsqueeze(0), size=(H, W), mode='bicubic', align_corners=False
                ).squeeze(0)
                gs_render = (rendering.permute(1,2,0).numpy() * 255).clip(0,255).astype(np.uint8)
            except Exception as e:
                Log(f"render error: {e}", tag="VIZ")
                gs_render = np.zeros((H, W, 3), dtype=np.uint8)
        else:
            gs_render = np.zeros((H, W, 3), dtype=np.uint8)

        # 4. Anchor point cloud projection (use original intrinsics)
        if self.gaussians is not None and hasattr(self.gaussians, '_anchor'):
            try:
                anchors = self.gaussians._anchor.detach().cuda()
                W2C = getWorld2View2(viewpoint.R, viewpoint.T)
                pts_h = torch.cat([anchors, torch.ones(anchors.shape[0],1,device=anchors.device)], dim=1)
                pts_cam = (W2C @ pts_h.T).T[:, :3]
                valid = pts_cam[:, 2] > 0.1
                pts_cam = pts_cam[valid]
                fx, fy = self.dataset.fx_ori, self.dataset.fy_ori
                cx, cy = self.dataset.cx_ori, self.dataset.cy_ori
                u = (pts_cam[:,0] / pts_cam[:,2] * fx + cx).long()
                v = (pts_cam[:,1] / pts_cam[:,2] * fy + cy).long()
                mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
                pcd_img = np.zeros((H, W, 3), dtype=np.uint8)
                u, v = u[mask].cpu().numpy(), v[mask].cpu().numpy()
                pcd_img[v, u] = [0, 255, 0]
                pcd_img = cv2.dilate(pcd_img, np.ones((3,3), np.uint8), iterations=1)
            except Exception as e:
                Log(f"pcd projection error: {e}", tag="VIZ")
                pcd_img = np.zeros((H, W, 3), dtype=np.uint8)
        else:
            pcd_img = np.zeros((H, W, 3), dtype=np.uint8)

        # 5. DISK keypoint distribution + vanishing point visualization
        kps_img = rgb.copy()
        last_kps = getattr(self.classic_tracking, 'last_kps', None)
        if last_kps is not None and len(last_kps) > 0:
            scale_x = W / self.dataset.width_ori
            scale_y = H / self.dataset.height_ori
            for x, y in last_kps:
                cx_k, cy_k = int(x * scale_x), int(y * scale_y)
                if 0 <= cx_k < W and 0 <= cy_k < H:
                    cv2.circle(kps_img, (cx_k, cy_k), 3, (0, 255, 0), -1)

        combined = np.hstack([rgb, depth_color, gs_render, pcd_img, kps_img])
        cv2.imwrite(os.path.join(save_dir, f"kf_{cur_frame_idx:05d}.png"), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

        # Save match visualization (reference + current frame with connecting lines)
        kp1 = getattr(self.classic_tracking, 'last_kps_ref', None)
        kp2 = getattr(self.classic_tracking, 'last_kps', None)
        ref_frame = getattr(self.classic_tracking, 'last_ref_frame', None)
        curr_frame = getattr(self.classic_tracking, 'last_curr_frame', None)
        if kp1 is not None and kp2 is not None and ref_frame is not None \
                and curr_frame is not None and len(kp1) > 0 \
                and len(self.kf_indices) % 20 == 1:
            match_dir = os.path.join(self.save_dir, "match_viz")
            draw_matches(ref_frame, kp1, curr_frame, kp2,
                         os.path.join(match_dir, f"match_{cur_frame_idx:05d}.png"))

    def run(self):
        configure_logging(self.config)
        cur_frame_idx = 0
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)

        while True:
            if self.q_vis2main.empty():
                if self.pause:
                    continue
            else:
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
                if self.pause:
                    self.backend_queue.put(["pause"])
                    continue
                else:
                    self.backend_queue.put(["unpause"])

            if self.frontend_queue.empty():
                tic.record()
                if cur_frame_idx >= len(self.dataset): 
                    break

                if self.requested_init:
                    time.sleep(0.01)
                    continue

                if self.single_thread and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )
                
                if cur_frame_idx > 2 and self.motion_thresh > 0:
                    flow = self.classic_tracking.motion_flow_keyframe(viewpoint.image_ori, prev_kf=False)
                    if flow < self.motion_thresh:
                        cur_frame_idx += 1

                        Log(f'Skip frame {cur_frame_idx}')
                        continue


                # unidepth take [C(rgb), H, W] as input 0~255 in uint8
                image_ori_for_depthmodel = torch.tensor(viewpoint.image_ori).cuda().to(torch.uint8)
                predictions = self.depth_model.infer(image_ori_for_depthmodel, self.intrinsics_ori) # return [1, 1, H, W]
                depth_pred = predictions["depth"].detach() # depth range: 0-255 in float

                depth_ori = depth_pred.squeeze().cpu().numpy() # shape: [H, W]
                rail_info = {"status": "disabled", "scale": 1.0, "confidence": 0.0}
                if self.rail_scale.enabled:
                    depth_ori, rail_info = self.rail_scale.apply(
                        cur_frame_idx,
                        viewpoint.image_ori,
                        depth_ori,
                        self.intrinsics_ori.cpu().numpy(),
                    )
                    if rail_info.get("status") == "corrected" and cur_frame_idx % 20 == 0:
                        Log(
                            f"RailScale scale={rail_info.get('scale', 1.0):.3f}, "
                            f"width={rail_info.get('width', 0.0):.3f}, "
                            f"conf={rail_info.get('confidence', 0.0):.2f}",
                            tag="RailScale",
                        )
                rail_info["image_height"] = int(depth_ori.shape[0])
                rail_info["image_width"] = int(depth_ori.shape[1])
                viewpoint.rail_info = rail_info
                depth_down = cv2.resize(depth_ori, (self.dataset.width, self.dataset.height))
                viewpoint.depth_ori = depth_ori
                viewpoint.depth =depth_down

                viewpoint.compute_grad_mask(self.config)

                self.classic_tracking.loop_kf(viewpoint.image_ori,  self.classic_tracking.kp_frame_id)

                loop_flag = self.classic_tracking.update(viewpoint.image_ori, viewpoint.depth_ori, self.classic_tracking.kp_frame_id)

                self.classic_tracking.kp_frame_id += 1

                self.classic_tracking.set_curr_keyframe(viewpoint.image_ori)

                if loop_flag:
                    # new keyframe has not been added
                    poses_old, poses_update = self.classic_tracking.get_loop_update()
                    poses_update = torch.tensor(poses_update)
                    poses_update = poses_update[:-1] # drop off the last pose (new keyframe)
                    poses_old = torch.tensor(poses_old)
                    poses_old = poses_old[:-1] # drop off the last pose (new keyframe)
                    update_viewpoints_from_poses(self.cameras, poses_update)
                    self.request_loop_update(cur_frame_idx, poses_old, poses_update, self.kf_indices)


                
                # Save RAM space, we dont need original resolution for mapping
                viewpoint.image_ori = None
                viewpoint.depth_ori = None

                self.cameras[cur_frame_idx] = viewpoint

                if self.reset:
                    self.initialize(cur_frame_idx, viewpoint)
                    self.current_window.append(cur_frame_idx)
                    self.keyframe_idx_list.append(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )

                self.tracking(cur_frame_idx, viewpoint)

                last_keyframe_idx = self.kf_indices[-1] if self.kf_indices else None
                is_backend_keyframe = (
                    not self.initialized
                    or self.is_keyframe(cur_frame_idx, last_keyframe_idx)
                )

                if is_backend_keyframe:
                    self.current_window, removed = self.add_to_window(
                        cur_frame_idx,
                        self.current_window,
                    )
                    if self.monocular and not self.initialized and removed is not None:
                        self.reset = True
                        Log(
                            "Keyframes lacks sufficient overlap to initialize the map, resetting."
                        )
                        continue

                    if self.viz and (cur_frame_idx - self.viz_frame_id_prev) > self.viz_frame_interval:
                        self.classic_tracking.viz_pose(self.classic_tracking.kp_frame_id)
                        self.viz_frame_id_prev = cur_frame_idx

                    depth_map = self.add_new_keyframe(
                        cur_frame_idx,
                        init=False,
                    )
                    self.keyframe_idx_list.append(cur_frame_idx)
                    self.request_keyframe(
                        cur_frame_idx, viewpoint, self.current_window, depth_map
                    )
                    self.save_keyframe_viz(cur_frame_idx, viewpoint) if len(self.kf_indices) % 10 == 1 else None

                if self.q_main2vis is not None:
                    if viewpoint.original_image is not None:
                        gtcolor = (viewpoint.original_image.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    else:
                        gtcolor = None
                    gtdepth = viewpoint.depth if isinstance(viewpoint.depth, np.ndarray) else (viewpoint.depth.cpu().numpy() if viewpoint.depth is not None else None)
                    self.q_main2vis.put(GaussianPacket(gtcolor=gtcolor, gtdepth=gtdepth))

                cur_frame_idx += 1

                toc.record()

            else:
                data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    self.sync_backend(data)

                elif data[0] == "keyframe":
                    self.sync_backend(data)
                    self.requested_keyframe -= 1

                elif data[0] == "init":
                    self.sync_backend(data)
                    self.requested_init = False

                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break
