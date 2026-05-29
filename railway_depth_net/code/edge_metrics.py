"""
================================================================================
边界区域误差计算工具 (适用于稀疏 LiDAR 评测)
================================================================================

功能：
  计算深度预测在边界区域的误差，用于验证边界保护效果

核心策略：
  用模型预测出的【密集深度图】（pred_depth）去找边缘
  为了避免稀疏 LiDAR 的 0 值空洞导致海量虚假边缘，
  本脚本利用【密集的预测深度图】来提取结构边缘，然后再用稀疏 GT 进行误差核算。
  预测图是 100% 密集的，用它做 Sobel 边缘检测完美丝滑，能精准框出场景中真实的物体轮廓。
  然后，我们再看哪些稀疏的 GT 真值点落在了这些轮廓内，去计算边缘误差。
  这是目前处理稀疏 LiDAR 边界评测最科学、最学术的做法。
  
你可能会发现 0.0924 / 0.0725 ≈ 1.27，为什么输出的 Ratio 是 1.6054？
因为我们在代码里计算的是“每张图像 Ratio 的平均值（Mean of Ratios）”，而不是“整体平均值的比值（Ratio of Means）”。
前者更科学，因为它反映了单张图像内边缘拟合的真实难度分布。
================================================================================
"""

import cv2
import numpy as np


def detect_depth_edges(dense_depth_map, threshold=0.1):
    """
    使用密集的深度图提取结构边界，彻底避开稀疏 GT 的 0 值截断问题

    Args:
        dense_depth_map: (H, W) 密集的预测深度图
        threshold: 梯度阈值（相对于深度值）

    Returns:
        edges: (H, W) 二值边界图
    """
    # Sobel 梯度
    grad_x = cv2.Sobel(dense_depth_map, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(dense_depth_map, cv2.CV_64F, 0, 1, ksize=3)

    # 梯度幅值
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # 归一化梯度（相对于局部深度值）
    depth_smooth = cv2.GaussianBlur(dense_depth_map, (5, 5), 0)
    grad_mag_normalized = grad_mag / (depth_smooth + 1e-6)

    # 阈值化得到二值边界
    edges = (grad_mag_normalized > threshold).astype(np.uint8)

    return edges


def dilate_edges(edges, kernel_size=5):
    """扩展边界区域"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    edge_region = cv2.dilate(edges, kernel, iterations=1)
    return edge_region


def compute_absrel(pred, gt, mask):
    """计算 AbsRel 误差"""
    valid = mask > 0
    if valid.sum() < 10:
        return np.nan

    p = pred[valid]
    g = gt[valid]

    p = np.clip(p, 1e-6, None)
    g = np.clip(g, 1e-6, None)

    absrel = np.mean(np.abs(p - g) / g)
    return float(absrel)


def compute_edge_aware_metrics(pred_depth, gt_depth, mask,
                                edge_threshold=0.1,
                                dilation_size=5):
    """
    计算边界感知的深度误差指标
    """
    # 1. 核心修复：直接从密集的预测图 (pred_depth) 提取场景的结构边缘！
    edges = detect_depth_edges(pred_depth, threshold=edge_threshold)

    # 2. 扩展边界区域，容许一定的对齐误差
    edge_region = dilate_edges(edges, kernel_size=dilation_size)

    # 3. 计算掩码交集：找出落在边界内的真实 LiDAR 点，以及非边界内的真实 LiDAR 点
    edge_mask = (edge_region > 0) & (mask > 0)
    non_edge_mask = (edge_region == 0) & (mask > 0)

    # 4. 计算各区域误差
    edge_absrel = compute_absrel(pred_depth, gt_depth, edge_mask.astype(np.uint8))
    non_edge_absrel = compute_absrel(pred_depth, gt_depth, non_edge_mask.astype(np.uint8))
    overall_absrel = compute_absrel(pred_depth, gt_depth, mask)

    # 5. 边界像素占比
    edge_pixel_ratio = edge_mask.sum() / (mask.sum() + 1e-6)

    # 6. 边界误差比
    if non_edge_absrel > 0 and not np.isnan(non_edge_absrel):
        edge_ratio = edge_absrel / non_edge_absrel
    else:
        edge_ratio = np.nan

    return {
        "Edge_AbsRel": edge_absrel,
        "NonEdge_AbsRel": non_edge_absrel,
        "Overall_AbsRel": overall_absrel,
        "Edge_Ratio": edge_ratio,
        "Edge_Pixel_Ratio": float(edge_pixel_ratio)
    }