import os
import sys
import cv2
import torch
import numpy as np

# 确保能导入你同目录下的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # 完美复用你代码库中的核心组件
    from train_depth_anything_lora import OSDaR23DepthDataset, align_pred_to_gt
    from train_depth_anything_lora_metric_v4 import load_full_model, forward_branches
    from confidence_module import ConfidenceAwareSparseSupervision
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保此脚本与你的 train_depth_anything_lora*.py 和 confidence_module.py 在同一目录下。")
    exit(1)

# ==============================================================================
# 🎯 [配置区域] 填入你的模型权重和数据路径
# ==============================================================================
# 1. 你的数据集路径
DATA_ROOT = "OSDaR23/train_data"
SPLIT_FILE = "OSDaR23/train_data/splits/val.txt"  # 从验证集挑一张图

# 2. 你的模型权重路径 (填入你实际跑出来的 checkpoint 文件夹路径)
STAGE_A_CKPT = "runs_Stage_A/best_relative_model"                 # 包含 lora 和 decoder_head
STAGE_B_CKPT = "runs_Stage_B_ablation/full/run_20260321_143049/best_metric_model"     # 包含 lora, decoder_head 和 metric_head

# 3. 输出设置
OUTPUT_DIR = "Fig1_Paper_Stages_Uncropped"
MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"
VIS_MAX_DEPTH = 60.0  # 彩色深度图的最大截断距离(米)
LIDAR_DILATE = 5      # 考虑到 4K 分辨率极大，适当增大点云的膨胀核使其可见
FRAME_IDX = 0         # 选验证集里的第几张图来画？(0 就是第一张)

# ==============================================================================
# 🛠️ [渲染辅助函数] 适配超高分辨率的渲染逻辑
# ==============================================================================
def ensure_dir(path): os.makedirs(path, exist_ok=True)

def render_lidar_gt_fullres(depth_gt_orig, vis_max_depth, dilate_k=5):
    """稀疏雷达 GT 渲染 (Jet色系 + 膨胀，直接在 4K 原分辨率上操作)"""
    gt_u8 = np.clip(depth_gt_orig / vis_max_depth, 0, 1)
    gt_u8 = (gt_u8 * 255).astype(np.uint8)
    color = cv2.applyColorMap(gt_u8, cv2.COLORMAP_JET)
    
    # 仅保留有效点
    masked_color = np.zeros_like(color)
    valid = depth_gt_orig > 0
    masked_color[valid] = color[valid]
    
    # 膨胀处理，让单像素的点在 4K 图像上肉眼可见
    masked_color = cv2.dilate(masked_color, np.ones((dilate_k, dilate_k), np.uint8))
    return masked_color

def render_relative_grayscale(pred_rel_full):
    """相对深度渲染 (灰度图，越近越亮)"""
    p_inv = 1.0 / (pred_rel_full + 1e-6)  # invert
    p_min, p_max = p_inv.min(), p_inv.max()
    p_norm = (p_inv - p_min) / (p_max - p_min + 1e-6)
    gray_u8 = (p_norm * 255).astype(np.uint8)
    return cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)

def render_metric_color(pred_metric_full, vis_max_depth):
    """度量深度渲染 (Viridis色系)"""
    metric_u8 = (np.clip(pred_metric_full, 0, vis_max_depth) / vis_max_depth * 255).astype(np.uint8)
    return cv2.applyColorMap(metric_u8, cv2.COLORMAP_VIRIDIS)

def render_cass_total_fullres(c_total_518, depth_gt_orig, orig_shape, dilate_k=5):
    """CASS 权重图渲染：将 518x518 的 CASS 映射回 4K 原图并精准打点"""
    orig_w, orig_h = orig_shape
    # 将 CASS 分数插值回原分辨率 (使用最近邻防止边缘糊掉)
    c_total_full = cv2.resize(c_total_518, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    c_u8 = (np.clip(c_total_full, 0, 1) * 255).astype(np.uint8)
    color = cv2.applyColorMap(c_u8, cv2.COLORMAP_VIRIDIS)
    
    # 用最原始的 4K mask 去滤除无效区域，保证点极其锐利
    masked_color = np.zeros_like(color)
    valid = depth_gt_orig > 0
    masked_color[valid] = color[valid]
    
    masked_color = cv2.dilate(masked_color, np.ones((dilate_k, dilate_k), np.uint8))
    return masked_color

# ==============================================================================
# 🚀 [主流程] 加载模型 -> 推理 -> 恢复分辨率 -> 存图
# ==============================================================================
def main():
    ensure_dir(OUTPUT_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[1/5] 🖥️ 设备: {device}")

    # 1. 提取真实数据
    print(f"[2/5] 📖 加载数据集并提取第 {FRAME_IDX} 帧...")
    dataset = OSDaR23DepthDataset(DATA_ROOT, SPLIT_FILE, input_size=518, augment=False)
    sample = dataset[FRAME_IDX]
    stem = sample["stem"]
    
    pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
    depth_gt_t = sample["depth_gt"].unsqueeze(0).to(device)
    mask_t = sample["mask"].unsqueeze(0).to(device)
    target_hw = sample["depth_gt"].shape

    # 2. 读取最原始的超高清 RGB 和 Depth (解决形变/裁剪问题的关键)
    rgb_path = os.path.join(DATA_ROOT, "rgb", f"{stem}.png")
    depth_path = os.path.join(DATA_ROOT, "depth", f"{stem}.png")
    
    rgb_bgr = cv2.imread(rgb_path)
    orig_h, orig_w = rgb_bgr.shape[:2] # 比如 2504 x 4112
    
    depth_u16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth_gt_orig = depth_u16.astype(np.float32) / 100.0

    # 3. 加载 Stage A 模型并推理
    print(f"[3/5] 🧠 加载 Stage A 模型并推理...")
    model_a, _ = load_full_model(MODEL_NAME, STAGE_A_CKPT, device=device)
    model_a.eval()
    with torch.no_grad():
        with torch.autocast(device_type="cuda" if "cuda" in str(device) else "cpu"):
            pred_rel_a, _, _ = forward_branches(model_a, None, pixel_values, target_hw, compute_metric=False)
            # 对相对深度进行尺度对齐，使其更有意义
            pred_rel_aligned_t = align_pred_to_gt(pred_rel_a, depth_gt_t, mask_t)
            
    # 【修复处】：加上了 .float() 防止 float16 导致 cv2.resize 崩溃
    pred_rel_np = pred_rel_aligned_t.squeeze().float().cpu().numpy()
    del model_a; torch.cuda.empty_cache()

    # 4. 加载 Stage B 模型并推理
    print(f"[4/5] 🧠 加载 Stage B 模型并推理...")
    model_b, metric_head = load_full_model(MODEL_NAME, STAGE_B_CKPT, device=device)
    model_b.eval()
    metric_head.eval()
    with torch.no_grad():
        with torch.autocast(device_type="cuda" if "cuda" in str(device) else "cpu"):
            _, pred_metric_b, _ = forward_branches(model_b, metric_head, pixel_values, target_hw, compute_metric=True)
            
    # 【修复处】：加上了 .float()
    pred_metric_np = pred_metric_b.squeeze().float().cpu().numpy()

    # 计算 518 维度的 CASS 权重
    cass = ConfidenceAwareSparseSupervision()
    with torch.no_grad():
        # 【修复处】：加上了 .float()
        c_total_518 = cass(depth_gt_t, mask_t).squeeze().float().cpu().numpy()

    # 5. 渲染并保存所有 5 张高清原比例单图
    print(f"[5/5] 🎨 正在恢复高清分辨率并渲染 5 张独立单图至: {OUTPUT_DIR}")
    
    # 关键步骤：将模型输出的 float32 518x518 拉伸回原始宽高比例
    pred_rel_full = cv2.resize(pred_rel_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    pred_metric_full = cv2.resize(pred_metric_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    
    # (1) RGB 原图
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"Fig1_{stem}_1_rgb.jpg"), rgb_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])
    
    # (2) 稀疏雷达 GT (基于高清原图渲染)
    img_lidar = render_lidar_gt_fullres(depth_gt_orig, VIS_MAX_DEPTH, dilate_k=LIDAR_DILATE)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"Fig1_{stem}_2_lidar_gt.jpg"), img_lidar, [cv2.IMWRITE_JPEG_QUALITY, 100])
    
    # (3) CASS 权重结果 (基于高清原图掩码渲染)
    img_cass = render_cass_total_fullres(c_total_518, depth_gt_orig, (orig_w, orig_h), dilate_k=LIDAR_DILATE)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"Fig1_{stem}_3_cass_weight.jpg"), img_cass, [cv2.IMWRITE_JPEG_QUALITY, 100])
    
    # (4) 相对深度估计 (灰度图)
    img_rel = render_relative_grayscale(pred_rel_full)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"Fig1_{stem}_4_relative_depth.jpg"), img_rel, [cv2.IMWRITE_JPEG_QUALITY, 100])
    
    # (5) 度量深度估计 (彩色图)
    img_metric = render_metric_color(pred_metric_full, VIS_MAX_DEPTH)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"Fig1_{stem}_5_metric_depth.jpg"), img_metric, [cv2.IMWRITE_JPEG_QUALITY, 100])

    print(f"\n🎉 成功！所有图像现在都是宽屏的 4112×2504 高清原图尺寸，且不再有 resize 报错。")

if __name__ == "__main__":
    main()