"""
Offline OpenGL flythrough video renderer for GigaSLAM.

Usage:
    # With Xvfb (headless server):
    Xvfb :99 -screen 0 1920x1080x24 &
    DISPLAY=:99 python scripts/render_flythrough_ogl.py \
        --ply_dir /root/GigaSLAM/PLY/gaussian_save \
        --poses_txt results/.../poses_est.txt \
        --config configs/rgb_12mp_middle.yaml \
        --output flythrough_ogl.mp4 \
        --width 1280 --height 720 --fps 10
"""

import argparse
import os
import sys

import cv2
import glfw
import numpy as np
import torch
import yaml
from OpenGL import GL as gl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaussian_splatting.gaussian_renderer import generate_neural_gaussians
from gaussian_splatting.scene.scaffold_model import GaussianModel
from gaussian_splatting.scene.embedding import Embedding
from gui.gl_render import util, util_gau
from gui.gl_render.render_ogl import OpenGLRenderer
from utils.camera_utils import Camera as SLAMCamera
from gaussian_splatting.utils.graphics_utils import focal2fov, getProjectionMatrix2


def load_gaussians(ply_dir, config):
    hier = config['Hierarchical']
    gaussians = GaussianModel(
        feat_dim=hier.get('feat_dim', 32),
        n_offsets=hier['n_offsets'],
        voxel_size_lis=hier['voxel_size_lis'],
        distance_lis=hier['distance_lis'],
        appearance_dim=hier.get('appearance_dim', 0),
        ratio=hier.get('point_ratio', 1),
    )
    pc_path = os.path.join(ply_dir, 'point_cloud', 'point_cloud.ply')
    gaussians.load_ply_sparse_gaussian(pc_path)
    gaussians.create_mlps()

    pt_dir = os.path.join(ply_dir, 'point_cloud')
    gaussians.mlp_opacity = torch.jit.load(os.path.join(pt_dir, 'opacity_mlp.pt')).cuda()
    gaussians.mlp_cov     = torch.jit.load(os.path.join(pt_dir, 'cov_mlp.pt')).cuda()
    gaussians.mlp_color   = torch.jit.load(os.path.join(pt_dir, 'color_mlp.pt')).cuda()

    emd_path = os.path.join(pt_dir, 'embedding_appearance.pt')
    if os.path.exists(emd_path) and gaussians.appearance_dim > 0:
        gaussians.embedding_appearance = torch.jit.load(emd_path).cuda()

    gaussians.eval()

    # _level is not saved in PLY; infer from anchor scaling (voxel size)
    voxel_sizes = torch.exp(gaussians._scaling[:, 0])
    level = torch.zeros(gaussians._anchor.shape[0], dtype=torch.int, device='cuda')
    for i, vs in enumerate(hier['voxel_size_lis']):
        level[voxel_sizes > vs * 0.5] = i
    gaussians._level = level

    return gaussians


def load_poses(poses_txt):
    """Returns list of 4x4 numpy arrays (as stored in poses_est.txt, i.e. C2W)."""
    poses = []
    with open(poses_txt) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) == 16:
                poses.append(np.array(vals).reshape(4, 4))
    return poses


def make_slam_camera(w2c, fx, fy, cx, cy, width, height, uid=0):
    R = torch.tensor(w2c[:3, :3], dtype=torch.float32)
    T = torch.tensor(w2c[:3, 3],  dtype=torch.float32)
    fovx = focal2fov(fx, width)
    fovy = focal2fov(fy, height)
    proj = getProjectionMatrix2(znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy,
                                W=width, H=height).transpose(0, 1).cuda()
    cam = SLAMCamera(
        uid=uid,
        color=None,
        depth=None,
        gt_T=None,
        projection_matrix=proj,
        fx=fx, fy=fy, cx=cx, cy=cy,
        fovx=fovx,
        fovy=fovy,
        image_height=height,
        image_width=width,
        device='cuda',
    )
    cam.R = R.cuda()
    cam.T = T.cuda()
    return cam


def w2c_to_gl_camera(w2c, fovy, width, height):
    """Convert W2C 4x4 to gui.gl_render.util.Camera."""
    c2w = np.linalg.inv(w2c)
    position = c2w[:3, 3].astype(np.float32)
    forward  = c2w[:3, 2].astype(np.float32)   # -Z in camera space = forward in world
    up       = -c2w[:3, 1].astype(np.float32)  # -Y in camera space = up in world
    target   = position + forward

    gl_cam = util.Camera(height, width)
    gl_cam.fovy     = fovy
    gl_cam.position = position
    gl_cam.target   = target
    gl_cam.up       = up
    return gl_cam


def infer_gaussians(gaussians, slam_cam):
    """Run MLP inference once to get fixed gaussian data for all frames."""
    with torch.no_grad():
        xyz, color, opacity, scaling, rot = generate_neural_gaussians(slam_cam, gaussians)
    SH_C0 = 0.28209479177387814
    sh = (color.detach().cpu().numpy() - 0.5) / SH_C0
    print(f"Inferred {xyz.shape[0]} gaussians, opacity range [{opacity.min():.3f}, {opacity.max():.3f}]")
    return util_gau.GaussianData(
        xyz=xyz.detach().cpu().numpy(),
        rot=rot.detach().cpu().numpy(),
        scale=scaling.detach().cpu().numpy(),
        opacity=opacity.detach().cpu().numpy(),
        sh=sh,
    )


def render_frame(renderer, gl_cam, gaus_data, width, height, window=None):
    """Upload gaussian data to OpenGL, render, read pixels."""
    renderer.update_gaussian_data(gaus_data)
    renderer.sort_and_update(gl_cam)
    renderer.update_camera_pose(gl_cam)
    renderer.update_camera_intrin(gl_cam)
    renderer.set_scale_modifier(1.0)
    renderer.set_render_mod(0)  # DC SH only

    gl.glClearColor(0, 0, 0, 1)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    renderer.draw()

    glfw.swap_buffers(window)
    glfw.poll_events()

    buf = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    img = np.frombuffer(buf, np.uint8).reshape(height, width, 3).copy()
    img = cv2.flip(img, 0)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ply_dir',   required=True, help='Path to gaussian_save/ dir')
    parser.add_argument('--poses_txt', required=True, help='Path to poses_est.txt')
    parser.add_argument('--config',    required=True, help='Path to yaml config')
    parser.add_argument('--output',    default='flythrough_ogl.mp4')
    parser.add_argument('--width',     type=int, default=1280)
    parser.add_argument('--height',    type=int, default=720)
    parser.add_argument('--fps',       type=int, default=10)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    # merge base config if needed
    if 'inherit_from' in config:
        base_path = os.path.join(os.path.dirname(args.config), '..', config['inherit_from'])
        with open(base_path) as f:
            base = yaml.safe_load(f)
        base.update(config)
        config = base

    calib = config['Dataset']['Calibration']
    fx, fy = calib['fx'], calib['fy']
    cx, cy = calib['cx'], calib['cy']
    fovy = 2 * np.arctan(calib['height'] / (2 * fy))  # use original image height

    # Init GLFW offscreen
    if not glfw.init():
        raise RuntimeError("GLFW init failed")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(args.width, args.height, "offscreen", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("GLFW window creation failed")
    glfw.make_context_current(window)

    renderer = OpenGLRenderer(args.width, args.height)

    gaussians = load_gaussians(args.ply_dir, config)
    poses = load_poses(args.poses_txt)
    print(f"Loaded {len(poses)} poses, {gaussians._anchor.shape[0]} anchors")

    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*'mp4v'),
        args.fps,
        (args.width, args.height)
    )

    for i, c2w in enumerate(poses):
        w2c = np.linalg.inv(c2w)
        slam_cam = make_slam_camera(w2c, fx, fy, cx, cy, args.width, args.height, uid=0)
        gaus_data = infer_gaussians(gaussians, slam_cam) if i == 0 or True else gaus_data
        gl_cam = w2c_to_gl_camera(w2c, fovy, args.width, args.height)
        img = render_frame(renderer, gl_cam, gaus_data, args.width, args.height, window=window)
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if i % 10 == 0:
            print(f"  frame {i}/{len(poses)}")

    writer.release()
    glfw.terminate()
    print(f"Saved {args.output}")


if __name__ == '__main__':
    main()
