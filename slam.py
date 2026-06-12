import os
import sys
import time
from argparse import ArgumentParser
from datetime import datetime

import torch
import torch.multiprocessing as mp
import yaml
from munch import munchify

from gaussian_splatting.scene.scaffold_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
from gui import gui_utils, slam_gui
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils import eval_rendering, plot_trajectory, plot_metrics_curve, plot_anchor_growth, save_render_video, save_flythrough_video, save_estimated_poses
from utils.logging_utils import Log, configure_logging
from utils.multiprocessing_utils import FakeQueue
from utils.slam_backend import BackEnd
from utils.slam_frontend import FrontEnd

import warnings
warnings.filterwarnings("ignore")

class SLAM:
    def __init__(self, config, save_dir=None):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        self.config = config
        self.save_dir = save_dir
        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )

        self.live_mode = self.config["Dataset"]["type"] == "realsense"
        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        self.use_gui = self.config["Results"]["use_gui"]
        if self.live_mode:
            self.use_gui = True
        self.eval_rendering = self.config["Results"]["eval_rendering"]

        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0
        self.dataset = load_dataset(
            model_params, model_params.source_path, config=config
        )
        intrinsics = torch.tensor([self.dataset.fx, 
                                   self.dataset.fy, 
                                   self.dataset.cx, 
                                   self.dataset.cy], device='cuda')


        # ===== Scaffold GS ===== #
        self.feat_dim = 32
        self.n_offsets = self.config['Hierarchical']['n_offsets'] # 16
        self.voxel_size =  0.005 # if voxel_size<=0, using 1nn dist

        # self.voxel_size_lis = [0.1, 0.25, 1, 5, 25]
        # self.distance_lis = [20, 40, 80, 160]
        self.voxel_size_lis = self.config['Hierarchical']['voxel_size_lis']
        self.distance_lis = self.config['Hierarchical']['distance_lis']

        self.max_level = len(self.voxel_size_lis) - 1

        self.update_depth = 3
        self.update_init_factor = 16
        self.update_hierachy_factor = 4

        self.use_feat_bank = False
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.lod = 0

        self.appearance_dim = self.config['Hierarchical']['appearance_dim']
        self.lowpoly = False
        self.ds = 1
        self.ratio = 1 # sampling the input point cloud
        self.undistorted = False 
        
        # In the Bungeenerf dataset, we propose to set the following three parameters to True,
        # Because there are enough dist variations.
        self.add_opacity_dist = False
        self.add_cov_dist = False
        self.add_color_dist = False

        self.gaussians = GaussianModel(self.feat_dim, 
                                    self.n_offsets, 
                                    self.voxel_size_lis, 
                                    self.distance_lis,
                                    self.update_depth, 
                                    self.update_init_factor, 
                                    self.update_hierachy_factor, 
                                    self.use_feat_bank, 
                                    self.appearance_dim, 
                                    self.ratio, 
                                    self.add_opacity_dist, 
                                    self.add_cov_dist, 
                                    self.add_color_dist,
                                    config = self.config, 
                                    intrinsics = intrinsics)
        self.gaussians.init_lr(6.0)

        # ===== Scaffold GS ===== #

        bg_color = [1, 1, 1]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        frontend_queue = mp.Queue()
        backend_queue = mp.Queue()

        q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        q_vis2main = mp.Queue() if self.use_gui else FakeQueue()

        self.config["Results"]["save_dir"] = save_dir
        self.config["Training"]["monocular"] = self.monocular

        self.frontend = FrontEnd(self.config, self.dataset, save_dir = self.save_dir)
        self.backend = BackEnd(self.config, self.dataset)

        # self.frontend.dataset = self.dataset
        self.frontend.background = self.background
        self.frontend.pipeline_params = self.pipeline_params
        self.frontend.gaussians_main = self.gaussians
        self.frontend.frontend_queue = frontend_queue
        self.frontend.backend_queue = backend_queue
        self.frontend.q_main2vis = q_main2vis
        self.frontend.q_vis2main = q_vis2main
        self.frontend.set_hyperparams()

        self.backend.gaussians = self.gaussians
        self.backend.background = self.background
        self.backend.cameras_extent = 6.0
        self.backend.pipeline_params = self.pipeline_params
        self.backend.opt_params = self.opt_params
        self.backend.frontend_queue = frontend_queue
        self.backend.backend_queue = backend_queue
        self.backend.live_mode = self.live_mode

        self.backend.set_hyperparams()

        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background,
            gaussians=self.gaussians,
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
        )

        backend_process = mp.Process(target=self.backend.run)
        if self.use_gui:
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            gui_process.start()
            time.sleep(5)

        backend_process.start()
        self.frontend.run()
        backend_queue.put(["pause"])

        end.record()
        torch.cuda.synchronize()
        # empty the frontend queue
        rendering_result = None

        if self.eval_rendering:
            self.gaussians = self.frontend.gaussians
            kf_indices = self.frontend.kf_indices

            # re-used the frontend queue to retrive the gaussians from the backend.
            while not frontend_queue.empty():
                frontend_queue.get()
            backend_queue.put(["color_refinement"])
            while True:
                if frontend_queue.empty():
                    time.sleep(0.01)
                    continue
                data = frontend_queue.get()
                if data[0] == "sync_backend" and frontend_queue.empty():
                    gaussians = data[1]
                    # Move CPU tensors back to CUDA for eval rendering
                    for attr in vars(gaussians):
                        val = getattr(gaussians, attr)
                        if isinstance(val, torch.Tensor) and not val.is_cuda:
                            setattr(gaussians, attr, val.cuda())
                    # Restore MLP modules from saved state dicts
                    if hasattr(gaussians, '_mlp_state_dicts'):
                        gaussians.create_mlps()
                        if gaussians.appearance_dim > 0:
                            from gaussian_splatting.scene.embedding import Embedding
                            sd = gaussians._mlp_state_dicts.get('embedding_appearance', {})
                            if sd:
                                num_cameras = next(iter(sd.values())).shape[0]
                                gaussians.embedding_appearance = Embedding(num_cameras, gaussians.appearance_dim).cuda()
                        for attr, sd in gaussians._mlp_state_dicts.items():
                            module = getattr(gaussians, attr, None)
                            if module is not None:
                                module.load_state_dict({k: v.cuda() for k, v in sd.items()})
                    self.gaussians = gaussians
                    break

            rendering_result = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                intrinsics=self.backend.intrinsics,
                height=self.backend.height,
                width=self.backend.width,
                kf_indices=kf_indices,
                iteration="after_opt",
                upsampling_method=config['Hierarchical']['upsampling_method'],
                rendered_depth_diag=config.get('Results', {}).get('rendered_depth_diag', {}),
                rendering_eval=config.get('Results', {}).get('rendering_eval', {}),
                
            )
        if self.save_dir is not None:
            self.gaussians.save_ckpt(self.save_dir)
            # Save estimated poses unconditionally so trajectory evaluation
            # (eval_ate.py) is available even when eval_rendering is False.
            save_estimated_poses(self.frontend.cameras, self.save_dir)
            plot_trajectory(self.frontend.cameras, self.frontend.kf_indices, self.save_dir)
            plot_anchor_growth(self.frontend.anchor_log, self.save_dir)
            self._auto_eval_trajectory()
            if self.eval_rendering and rendering_result is not None and rendering_result.get("psnr_array"):
                plot_metrics_curve(
                    rendering_result["psnr_array"],
                    rendering_result["ssim_array"],
                    rendering_result["eval_indices"],
                    self.save_dir,
                )
        backend_queue.put(["stop"])
        backend_process.join()
        Log("Backend stopped and joined the main thread")
        if self.use_gui:
            q_main2vis.put(gui_utils.GaussianPacket(finish=True))
            gui_process.join()
            Log("GUI Stopped and joined the main thread")

    def run(self):
        pass

    def _auto_eval_trajectory(self):
        results_cfg = self.config.get("Results", {})
        auto_eval = results_cfg.get("auto_eval_ate", False)
        auto_post = results_cfg.get("auto_postprocess_trajectory", False)
        if not auto_eval and not auto_post:
            return

        est_path = os.path.join(self.save_dir, "poses_est.txt")
        gt_path = self.config.get("Dataset", {}).get("pose_path", None)
        if not os.path.exists(est_path):
            Log(f"Skip automatic trajectory evaluation: {est_path} not found", tag="Eval")
            return

        plot_dir = os.path.join(self.save_dir, "plot")

        if auto_post:
            try:
                from scripts.postprocess_trajectory import run_postprocess

                gt_for_eval = gt_path if (auto_eval and gt_path and os.path.exists(gt_path)) else None
                result = run_postprocess(
                    est_path,
                    out_path=os.path.join(self.save_dir, "poses_est_post.txt"),
                    gt_path=gt_for_eval,
                    save_dir=plot_dir,
                    monocular=results_cfg.get("auto_eval_monocular", False),
                    est_convention=results_cfg.get("auto_eval_est_convention", "c2w"),
                )
                Log(f"Saved post-processed trajectory: {result['out_path']}", tag="Eval")
                if result["eval"] is not None:
                    Log(f"Automatic ATE original={result['eval']['ate_original']:.4f} m", tag="Eval")
                    Log(f"Automatic ATE post={result['eval']['ate_post']:.4f} m", tag="Eval")
                elif auto_eval:
                    Log(f"Skip automatic ATE: GT pose_path not found: {gt_path}", tag="Eval")
            except Exception as exc:
                Log(f"Automatic postprocess/eval failed: {exc}", tag="Eval")
            return

        if auto_eval:
            if not gt_path or not os.path.exists(gt_path):
                Log(f"Skip automatic ATE: GT pose_path not found: {gt_path}", tag="Eval")
                return
            try:
                from scripts.eval_ate import evaluate_saved_ate

                est_convention = results_cfg.get("auto_eval_est_convention", "c2w")
                monocular = results_cfg.get("auto_eval_monocular", False)
                use_pose_idx = results_cfg.get("auto_eval_use_pose_idx", True)
                ate = evaluate_saved_ate(
                    est_path,
                    gt_path,
                    plot_dir,
                    label="original",
                    monocular=monocular,
                    est_convention=est_convention,
                    use_pose_idx=use_pose_idx,
                )
                Log(
                    f"Automatic ATE original={ate:.4f} m "
                    f"(est={est_convention}, pose_idx={use_pose_idx})",
                    tag="Eval",
                )

                if results_cfg.get("auto_eval_compare_pose_idx", True):
                    ate_legacy = evaluate_saved_ate(
                        est_path,
                        gt_path,
                        plot_dir,
                        label="legacy_ordered",
                        monocular=monocular,
                        est_convention=est_convention,
                        use_pose_idx=False,
                    )
                    Log(f"Automatic ATE legacy_ordered={ate_legacy:.4f} m", tag="Eval")
            except Exception as exc:
                Log(f"Automatic ATE failed: {exc}", tag="Eval")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    parser.add_argument("--eval", action="store_true")

    args = parser.parse_args(sys.argv[1:])

    mp.set_start_method("spawn", force=True)

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)
    configure_logging(config)
    save_dir = None

    if args.eval:
        Log("Running MonoGS in Evaluation Mode")
        Log("Following config will be overriden")
        Log("\tsave_results=True")
        config["Results"]["save_results"] = True
        Log("\tuse_gui=False")
        config["Results"]["use_gui"] = False
        Log("\teval_rendering=True")
        config["Results"]["eval_rendering"] = True
        Log("\tuse_wandb=True")
        config["Results"]["use_wandb"] = True

    if config["Results"]["save_results"]:
        mkdir_p(config["Results"]["save_dir"])
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if config['SLAM']['loop_closure']:
            current_datetime += '-LC'
        else:
            current_datetime += '-No-LC'

        # path = config["Dataset"]["dataset_path"].split("/")
        path = config["Dataset"]["color_path"].split("/")
        save_dir = os.path.join(
            config["Results"]["save_dir"], path[-3] + "_" + path[-2] + "_" + path[-1], current_datetime
        )
        tmp = args.config
        tmp = tmp.split(".")[0]
        config["Results"]["save_dir"] = save_dir
        mkdir_p(save_dir)
        with open(os.path.join(save_dir, "config.yml"), "w") as file:
            documents = yaml.dump(config, file)
        Log("saving results in " + save_dir)

    slam = SLAM(config, save_dir=save_dir)

    slam.run()

    # All done
    Log("Done.")
