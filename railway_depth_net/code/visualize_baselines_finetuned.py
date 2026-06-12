"""
================================================================================
Publication-ready qualitative visualization for Stage B (Metric Depth)
8-Column Layout: RGB | DepthPro | DA-V2 | ZoeDepth-NK | UniDepth | Ours | Error Map | LiDAR GT
================================================================================

python visualize_baselines_finetuned.py \
  --ours_full_path runs_Stage_B_ablation/full/best_metric_model \
  --zoe_ft_path checkpoints/zoedepth_sparse_uniform/best_model.pth \
  --uni_ft_path checkpoints/unidepth_sparse_uniform/best_model.pth \
  --dav2_ft_path checkpoints/dav2metric_sparse_uniform/best_model.pth \
  --depthpro_ft_path checkpoints/depthpro_sparse_uniform/best_model.pth \
  --selection_mode paper6

"""

import os
import sys
import gc
import json
import argparse
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from torch.utils.data import DataLoader
import _pickle # 用于处理加载异常

# 为了兼容新旧 timm，保留这个补丁
try:
    from timm.layers import DropPath
except ImportError:
    from timm.models.layers import DropPath

# Hugging Face 模型加载库 (DA-V2 和 DepthPro 均需要)
from transformers import AutoModelForDepthEstimation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # 复用数据加载逻辑
    import train_depth_anything_lora as base_impl
    from train_depth_anything_lora_metric_v4 import load_full_model, forward_branches
    # 用于加载 UniDepth (Baseline)
    from unidepth.models import UniDepthV2
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保此脚本与你的项目核心代码和 `unidepth` 库在同一目录下。")
    exit(1)

def get_args():
    p = argparse.ArgumentParser(description="Stage B 8-Column Auto-Visualization")
    p.add_argument("--data_root", type=str, default="OSDaR23/train_data")
    p.add_argument("--split", type=str, default="val")
    
    p.add_argument("--model_name", type=str, default="depth-anything/Depth-Anything-V2-Small-hf")
    p.add_argument("--ours_full_path", type=str, required=True, help="Ours (Stage A+B) checkpoint path")
    
    # baseline 的 fine-tuned 权重路径
    p.add_argument("--zoe_ft_path", type=str, required=True, help="Fine-tuned ZoeDepth path")
    p.add_argument("--uni_ft_path", type=str, required=True, help="Fine-tuned UniDepth path")
    p.add_argument("--dav2_ft_path", type=str, required=True, help="Fine-tuned DA-V2-Metric path")
    # 【新增】DepthPro 的权重路径
    p.add_argument("--depthpro_ft_path", type=str, required=True, help="Fine-tuned DepthPro path")
    
    p.add_argument("--output_dir", type=str, default="visualizations_baselines_finetuned")
    
    # 核心：自动选图配置
    p.add_argument("--selection_mode", type=str, default="paper6", choices=["paper6", "manual"])
    p.add_argument("--stems", type=str, default="", help="手动指定stems")
    
    p.add_argument("--col_width", type=int, default=512)
    p.add_argument("--col_height", type=int, default=360)
    
    # 度量深度可视化参数
    p.add_argument("--vis_max_depth", type=float, default=80.0, help="绝对深度最大渲染距离(米)")
    p.add_argument("--error_max", type=float, default=10.0, help="误差热图最大误差阈值(米)")
    p.add_argument("--gt_dilate", type=int, default=2, help="LiDAR点云膨胀尺寸")
    
    p.add_argument("--depth_cmap", type=str, default="turbo")
    p.add_argument("--error_cmap", type=str, default="magma")
    
    p.add_argument("--montage_row_height", type=int, default=360)
    p.add_argument("--jpeg_quality", type=int, default=95)
    return p.parse_args()

# ==============================================================================
# 工具与排版函数 (完全保留你原版代码)
# ==============================================================================
def ensure_dir(path: str): os.makedirs(path, exist_ok=True)

def create_text_tile(text: str, width: int, height: int, font_size: int = 55) -> np.ndarray:
    """创建一个纯白背景、使用加粗学术衬线字体且绝对居中的图块"""
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Ubuntu学术字体路径，根据需要微调
    font_path = '/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf'
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    try:
        left, top, right, bottom = font.getbbox(text)
        text_w = right - left
        text_h = bottom - top
    except AttributeError:
        text_w, text_h = draw.textsize(text, font=font)

    text_x = (width - text_w) // 2
    text_y = (height - text_h) // 2 - top if 'top' in locals() else (height - text_h) // 2
    draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)
    return np.array(img)

def apply_cmap_u8(gray_u8: np.ndarray, cmap_name: str) -> np.ndarray:
    cmap = {"turbo": cv2.COLORMAP_TURBO, "viridis": cv2.COLORMAP_VIRIDIS, 
            "magma": cv2.COLORMAP_MAGMA, "jet": cv2.COLORMAP_JET}[cmap_name]
    return cv2.applyColorMap(np.ascontiguousarray(gray_u8), cmap)

def normalize_metric_for_display(depth_m: np.ndarray, max_depth: float) -> np.ndarray:
    depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    depth_m[depth_m < 0] = 0.0
    norm = np.clip(depth_m / max(max_depth, 1e-6), 0.0, 1.0)
    return (norm * 255).astype(np.uint8)

def compute_error_map(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, error_max: float) -> np.ndarray:
    err = np.abs(pred - gt)
    norm_err = np.clip(err / max(error_max, 1e-6), 0.0, 1.0)
    err_u8 = (norm_err * 255).astype(np.uint8)
    err_u8[mask == 0] = 0
    return err_u8

def render_lidar_gt(gt: np.ndarray, mask: np.ndarray, max_depth: float, dilate: int) -> np.ndarray:
    gt_disp = normalize_metric_for_display(gt * (mask > 0), max_depth)
    if dilate > 0:
        kernel = np.ones((max(1, dilate)*2+1, max(1, dilate)*2+1), np.uint8)
        gt_disp = cv2.dilate(gt_disp, kernel)
    return gt_disp

def pad_to_width(img: np.ndarray, target_w: int, value: int = 255) -> np.ndarray:
    h, w = img.shape[:2]
    if w >= target_w: return img
    return np.concatenate([img, np.full((h, target_w - w, 3), value, dtype=np.uint8)], axis=1)

margin_val = 15 # 全局列/行白色间距，确保十字白框网格规整

def build_montage(panels: List[np.ndarray], margin: int = 15) -> np.ndarray:
    if not panels: return np.zeros((10, 10, 3), dtype=np.uint8)
    max_w = max(p.shape[1] for p in panels)
    final_rows = []
    for i, p in enumerate(panels):
        p_padded = pad_to_width(p, max_w, value=255)
        if i < len(panels) - 1:
            p_padded = np.concatenate([p_padded, np.full((margin, max_w, 3), 255, dtype=np.uint8)], axis=0)
        final_rows.append(p_padded)
    return np.concatenate(final_rows, axis=0)


# ==============================================================================
# 智能选图算法 (针对度量深度) (完全保留你原版代码)
# ==============================================================================
def compute_range_ratios(gt: np.ndarray, mask: np.ndarray) -> tuple:
    valid = mask > 0
    if valid.sum() < 10: return 0.0, 0.0, 0.0
    vals = gt[valid]
    return float((vals < 30).mean()), float(((vals >= 30) & (vals < 60)).mean()), float((vals >= 60).mean())

def choose_unique(rows: List[Dict], key_fn, used: set, reverse: bool = False) -> Dict:
    for row in sorted(rows, key=key_fn, reverse=reverse):
        if row["stem"] not in used:
            used.add(row["stem"])
            return row
    return None

def auto_select_stems(args, device) -> List[str]:
    print("\n[阶段 1/3] 正在启动智能选图算法 (Paper6 - Metric Depth)...")
    split_path = os.path.join(args.data_root, "splits", f"{args.split}.txt")
    ds = base_impl.OSDaR23DepthDataset(args.data_root, split_path, input_size=518)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2)
    
    base_model, metric_head = load_full_model(args.model_name, args.ours_full_path, device)
    base_model.eval()
    metric_head.eval()

    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="扫描验证集提取度量特征"):
            pv = batch["pixel_values" if "pixel_values" in batch else "image"].to(device)
            gt = batch["depth_gt"].numpy()
            mask = batch["mask"].numpy().astype(np.uint8)
            stems = batch["stem"]
            
            _, pred_abs, _ = forward_branches(base_model, metric_head, pv, gt.shape[-2:], compute_metric=True)
            pred_np = pred_abs.detach().float().cpu().numpy()
            if pred_np.ndim == 4: pred_np = pred_np[:, 0]
            elif pred_np.ndim == 2: pred_np = pred_np[np.newaxis, ...]
            
            for i in range(pv.shape[0]):
                valid_cnt = (mask[i] > 0).sum()
                if valid_cnt < 200: continue
                
                p, g = pred_np[i][mask[i]>0], gt[i][mask[i]>0]
                p, g = np.clip(p, 1e-6, None), np.clip(g, 1e-6, None)
                abs_rel = float(np.mean(np.abs(p - g) / g))
                
                near, mid, far = compute_range_ratios(gt[i], mask[i])
                entropy = -(near*np.log(max(near,1e-6)) + mid*np.log(max(mid,1e-6)) + far*np.log(max(far,1e-6)))
                
                rows.append({
                    "stem": stems[i], "abs_rel": abs_rel, 
                    "near": near, "far": far, "entropy": entropy
                })
                
    del base_model, metric_head
    gc.collect()
    torch.cuda.empty_cache()

    used = set()
    selected = []
    med = float(np.median([r["abs_rel"] for r in rows]))
    
    selected.append(choose_unique(rows, lambda r: r["abs_rel"], used, reverse=False)) # 最容易
    selected.append(choose_unique(rows, lambda r: abs(r["abs_rel"] - med), used, reverse=False)) # 典型
    selected.append(choose_unique(rows, lambda r: r["abs_rel"], used, reverse=True)) # 困难
    selected.append(choose_unique(rows, lambda r: (r["near"], -r["abs_rel"]), used, reverse=True)) # 近景
    selected.append(choose_unique(rows, lambda r: (r["far"], -r["abs_rel"]), used, reverse=True)) # 远景
    selected.append(choose_unique(rows, lambda r: (r["entropy"], -abs(r["abs_rel"] - med)), used, reverse=True)) # 混合

    final_stems = [s["stem"] for s in selected if s is not None]
    print(f"✅ 智能选图完成！选中 stems: {final_stems}")
    return final_stems

# ==============================================================================
# 核心流水线 (按模型逐一执行)
# ==============================================================================
def main():
    args = get_args()
    ensure_dir(args.output_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 混合精度环境定义
    try:
        from torch.amp import autocast
        def amp_autocast_fn(enabled=True): return autocast(device_type="cuda" if "cuda" in str(device) else "cpu", enabled=enabled)
    except ImportError:
        from torch.cuda.amp import autocast
        def amp_autocast_fn(enabled=True): return autocast(enabled=enabled)
    
    if args.selection_mode == "paper6":
        stems = auto_select_stems(args, device)
    else:
        stems = [s.strip() for s in args.stems.split(",") if s.strip()]

    print(f"\n[阶段 2/3] 准备对选中的 {len(stems)} 张图进行度量跨模型推理...")
    rgb_images, gt_data, mask_data, target_hw = {}, {}, {}, None
    for stem in stems:
        bgr = cv2.imread(os.path.join(args.data_root, "rgb", stem + ".png"))
        depth_u16 = cv2.imread(os.path.join(args.data_root, "depth", stem + ".png"), cv2.IMREAD_UNCHANGED)
        gt_m = depth_u16.astype(np.float32) / 100.0
        mask = (gt_m > 0).astype(np.uint8)
        
        rgb_images[stem] = bgr
        gt_data[stem] = gt_m
        mask_data[stem] = mask
        if target_hw is None: target_hw = bgr.shape[:2]
            
    predictions = {stem: {} for stem in stems}
    
    # 1. FT-ZoeDepth-NK (带环境免疫补丁)
    print("\n👉 运行 Fine-tuned ZoeDepth-NK...")
    import torch.nn as nn
    original_load = nn.Module.load_state_dict
    def safe_load(self, state_dict, strict=True, **kwargs):
        return original_load(self, state_dict, strict=False, **kwargs)
    try:
        nn.Module.load_state_dict = safe_load
        
        zoe = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=False, trust_repo=True)
        try:
            ckpt_zoe = torch.load(args.zoe_ft_path, map_location="cpu", weights_only=False)
        except (TypeError, _pickle.UnpicklingError):
            ckpt_zoe = torch.load(args.zoe_ft_path, map_location="cpu")
        zoe_state = ckpt_zoe["model_state_dict"] if "model_state_dict" in ckpt_zoe else ckpt_zoe
        zoe_state = {k: v for k, v in zoe_state.items() if "relative_position_index" not in k}
        zoe.load_state_dict(zoe_state, strict=False)
        zoe = zoe.to(device).eval()
        
        for m in zoe.modules():
            if hasattr(m, 'drop_path1') and not hasattr(m, 'drop_path'):
                m.drop_path = m.drop_path1
    finally:
        nn.Module.load_state_dict = original_load

    with torch.no_grad():
        for stem, bgr in rgb_images.items():
            pred = zoe.infer_pil(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            predictions[stem]["ZoeDepth"] = cv2.resize(pred, (target_hw[1], target_hw[0]))
    del zoe; gc.collect(); torch.cuda.empty_cache()

    # 2. FT-UniDepth
    print("👉 运行 Fine-tuned UniDepth...")
    from unidepth.models import UniDepthV2
    
    unidepth = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
    try:
        ckpt_uni = torch.load(args.uni_ft_path, map_location="cpu", weights_only=False)
    except (TypeError, _pickle.UnpicklingError):
        ckpt_uni = torch.load(args.uni_ft_path, map_location="cpu")
    unidepth.load_state_dict(ckpt_uni["model_state_dict"], strict=False)
    unidepth = unidepth.to(device).eval()
    
    with torch.no_grad():
        for stem, bgr in rgb_images.items():
            orig_h, orig_w = bgr.shape[:2]
            rgb_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(rgb_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            
            with amp_autocast_fn(enabled=True):
                predictions_uni = unidepth.infer(img_tensor.to(device))
            pred = predictions_uni["depth"].squeeze().cpu().numpy()
            predictions[stem]["UniDepth"] = cv2.resize(pred, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)
    del unidepth; gc.collect(); torch.cuda.empty_cache()

    # Define ImageNet normalization parameters (for DA-V2 and DepthPro)
    img_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    img_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # 3. FT-DA-V2-Metric 
    print("👉 运行 Fine-tuned DA-V2-Metric...")
    dav2_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf").to(device)
    
    try:
        ckpt_dav2 = torch.load(args.dav2_ft_path, map_location="cpu", weights_only=False)
    except (_pickle.UnpicklingError, TypeError):
        ckpt_dav2 = torch.load(args.dav2_ft_path, map_location="cpu")
        
    clean_state_dav2 = {k: v for k, v in ckpt_dav2["model_state_dict"].items() if "relative_position_index" not in k}
    dav2_model.load_state_dict(clean_state_dav2, strict=False)
    dav2_model.eval()

    with torch.no_grad():
        for stem, bgr in rgb_images.items():
            orig_h, orig_w = bgr.shape[:2]
            rgb_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            # 缩放至训练尺寸 (O V2 HF VIT需要)
            target_dav2_h, target_dav2_w = 378, 616
            rgb_resized = cv2.resize(rgb_rgb, (target_dav2_w, target_dav2_h), interpolation=cv2.INTER_LINEAR)
            img_tensor = torch.from_numpy(rgb_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            
            rgb_norm = (img_tensor.to(device) - img_mean) / img_std
            with amp_autocast_fn(enabled=True):
                outputs_dav2 = dav2_model(pixel_values=rgb_norm)
            
            # 强转 float32 并补通道，拉伸回 4K 原图分辨率
            pred_dav2 = outputs_dav2.predicted_depth.float().unsqueeze(1)
            pred_dav2 = F.interpolate(pred_dav2, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            
            pred_dav2_np = pred_dav2.squeeze().cpu().numpy()
            predictions[stem]["DAV2"] = cv2.resize(pred_dav2_np, (target_hw[1], target_hw[0]))
            
    del dav2_model; gc.collect(); torch.cuda.empty_cache()

    # ==========================================================================
    # 【修改 1】新增 3.5 FT-DepthPro 
    # ==========================================================================
    print("👉 运行 Fine-tuned DepthPro...")
    # 强制以 float32 加载，防止混合精度下的 OpenCV FP16 报错
    depthpro_model = AutoModelForDepthEstimation.from_pretrained(
        "apple/DepthPro-hf", 
        torch_dtype=torch.float32
    ).to(device)
    
    try:
        ckpt_dp = torch.load(args.depthpro_ft_path, map_location="cpu", weights_only=False)
    except (_pickle.UnpicklingError, TypeError):
        ckpt_dp = torch.load(args.depthpro_ft_path, map_location="cpu")
        
    clean_state_dp = {k: v for k, v in ckpt_dp["model_state_dict"].items() if "relative_position_index" not in k}
    depthpro_model.load_state_dict(clean_state_dp, strict=False)
    depthpro_model.eval()

    # DepthPro 强制要求最短边 >= 1536，在推理时临时使用大图
    target_dp_eval_h, target_dp_eval_w = 1536, 1536

    with torch.no_grad():
        for stem, bgr in tqdm(rgb_images.items(), desc="DepthPro 推理"):
            orig_h, orig_w = bgr.shape[:2]
            rgb_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            
            # 缩放至大图尺寸满足架构限制
            rgb_resized = cv2.resize(rgb_rgb, (target_dp_eval_w, target_dp_eval_h), interpolation=cv2.INTER_LINEAR)
            img_tensor = torch.from_numpy(rgb_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            
            # 归一化与 AMP 保护
            rgb_norm = (img_tensor.to(device) - img_mean) / img_std
            with amp_autocast_fn(enabled=True):
                outputs_dp = depthpro_model(pixel_values=rgb_norm)
            
            # 强转 float32 并补通道，拉伸回 4K 原图分辨率
            pred_dp = outputs_dp.predicted_depth.float().unsqueeze(1)
            pred_dp = F.interpolate(pred_dp, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            
            # 收集并缩放至排版尺寸
            pred_dp_np = pred_dp.squeeze().cpu().numpy()
            predictions[stem]["DepthPro"] = cv2.resize(pred_dp_np, (target_hw[1], target_hw[0]))
            
    del depthpro_model; gc.collect(); torch.cuda.empty_cache()
    # ==========================================================================

    # 4. Ours Full (Stage A+B)
    print("👉 运行 Ours Full...")
    ds = base_impl.OSDaR23DepthDataset(args.data_root, os.path.join(args.data_root, "splits", f"{args.split}.txt"), input_size=518)
    stem_to_idx = {stem: idx for idx, stem in enumerate([line.strip() for line in open(os.path.join(args.data_root, "splits", f"{args.split}.txt"))])}

    base_model, metric_head = load_full_model(args.model_name, args.ours_full_path, device)
    base_model.eval(); metric_head.eval()
    
    with torch.no_grad():
        for stem in stems:
            sample = ds[stem_to_idx[stem]]
            pv = sample["pixel_values" if "pixel_values" in sample else "image"].unsqueeze(0).to(device)
            _, pred_abs, _ = forward_branches(base_model, metric_head, pv, pv.shape[-2:], compute_metric=True)
            pred_np = pred_abs.detach().squeeze().float().cpu().numpy()
            predictions[stem]["Ours"] = cv2.resize(pred_np, (target_hw[1], target_hw[0]))
    del base_model, metric_head; gc.collect(); torch.cuda.empty_cache()

    # ==============================================================================
    # 渲染与拼图
    # ==============================================================================
    print("\n[阶段 3/3] 🎨 正在渲染出版级 8 列排版图...")
    montage_panels = []
    text_height = 100 
    
    # 【修改 2】更新图注为 8 列 (加入了 DepthPro)
    titles = ["RGB", "DepthPro (Apple)", "DA-V2-Metric", "ZoeDepth-NK", "UniDepth-V2", "Ours", "Error Map", "LiDAR GT"]
    
    for stem in stems:
        bgr = rgb_images[stem]
        gt = gt_data[stem]
        mask = mask_data[stem]
        preds = predictions[stem]
        
        # 度量深度渲染 (Magma)
        ours_color = apply_cmap_u8(normalize_metric_for_display(preds["Ours"], args.vis_max_depth), args.depth_cmap)
        # 【修改 3】渲染 DepthPro 彩色图
        dp_color   = apply_cmap_u8(normalize_metric_for_display(preds["DepthPro"], args.vis_max_depth), args.depth_cmap)
        dav2_color = apply_cmap_u8(normalize_metric_for_display(preds["DAV2"], args.vis_max_depth), args.depth_cmap)
        zoe_color  = apply_cmap_u8(normalize_metric_for_display(preds["ZoeDepth"], args.vis_max_depth), args.depth_cmap)
        uni_color  = apply_cmap_u8(normalize_metric_for_display(preds["UniDepth"], args.vis_max_depth), args.depth_cmap)
        
        # 误差图 (仅对 Ours)
        err_u8 = compute_error_map(preds["Ours"], gt, mask, args.error_max)
        err_color = apply_cmap_u8(err_u8, args.error_cmap)
        err_color[mask == 0] = 0 # 背景置黑
        
        # LiDAR GT (Turbo + Dilate)
        gt_disp = render_lidar_gt(gt, mask, args.vis_max_depth, args.gt_dilate)
        gt_color = apply_cmap_u8(gt_disp, args.depth_cmap)
        gt_color[gt_disp == 0] = 0
        
        # 【修改 4】将 DepthPro 加入到最终的图片列表中 (排在 RGB 后面)
        vis_list = [bgr, dp_color, dav2_color, zoe_color, uni_color, ours_color, err_color, gt_color]
        
        # 构建当前行 (插入十字白框网格间距)
        row_tiles = []
        margin_v = np.full((args.col_height, margin_val, 3), 255, dtype=np.uint8)
        
        for i, img in enumerate(vis_list):
            row_tiles.append(cv2.resize(img, (args.col_width, args.col_height)))
            if i < len(vis_list) - 1:
                row_tiles.append(margin_v) # 插入竖直白色间距
                
        panel = np.concatenate(row_tiles, axis=1)
        # 将单行保存为 _8col.jpg
        cv2.imwrite(os.path.join(args.output_dir, f"{stem}_8col.jpg"), panel, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        montage_panels.append(panel)

    # 构建底部居中图注行 (插入竖直间距对齐图片)
    if montage_panels:
        text_tiles = []
        margin_text = np.full((text_height, margin_val, 3), 255, dtype=np.uint8)
        
        for i, title in enumerate(titles):
            text_tiles.append(create_text_tile(title, args.col_width, text_height))
            if i < len(titles) - 1:
                text_tiles.append(margin_text) # 插入竖直间距对齐文字
                
        text_panel = np.concatenate(text_tiles, axis=1)
        montage_panels.append(text_panel) # 将文字行作为最后一行

        # 组合最终的大海报 (行间距 Margin=15，形成完美的十字白框网格)
        montage = build_montage(montage_panels, margin=margin_val)
        out_path = os.path.join(args.output_dir, "StageB_Auto_8col_montage.jpg")
        cv2.imwrite(out_path, montage, [cv2.IMWRITE_JPEG_QUALITY, 100])
        print(f"✅ 完美的 8 列可视化大图已保存至: {out_path}")

if __name__ == "__main__":
    main()