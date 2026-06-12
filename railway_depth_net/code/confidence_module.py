"""
================================================================================
Confidence-Aware Sparse Supervision Module (CASS)
================================================================================

动机:
  极稀疏 LiDAR 投影点 (2.5% 覆盖率) 的质量是不均匀的:
    - 轨道表面、建筑立面 → 平坦规则表面，回波稳定，标定误差影响小 → 高置信
    - 植被边缘、反光金属 → 深度梯度剧烈，多路径反射 → 低置信
    - 远距离稀疏区域    → 测距噪声大，投影像素偏移 → 低置信
    - 孤立零散点        → 可能是噪声或动态物体残留 → 低置信

  传统做法对所有有效像素施加相同 loss 权重，噪声点会误导梯度方向。

核心思想:
  从稀疏深度 GT 本身的局部几何属性中提取 per-pixel 置信度:
    C(p) = C_density(p) × C_smooth(p) × C_distance(p)

  三个因子分别衡量:
    1. 局部支撑密度: 周围有多少有效邻居 → 密集区域更可靠
    2. 深度一致性:   局部深度变化是否平缓 → 平坦表面更可靠
    3. 距离衰减:     近处比远处更可靠 (已有，此处统一)

  最终置信度图与 loss 相乘，自适应地降低不可靠监督点的权重。

使用方式:
  # 在训练脚本中:
  from confidence_module import ConfidenceAwareSparseSupervision

  cass = ConfidenceAwareSparseSupervision()
  weight = cass(depth_gt, mask)  # (B, H, W) 置信度权重
  loss = criterion(pred, gt, mask, weight)

论文中的描述:
  "We propose a confidence-aware sparse supervision (CASS) module that
  adaptively weights each LiDAR supervision pixel based on its local
  geometric reliability. Specifically, we compute per-pixel confidence
  as the product of local support density, depth smoothness, and
  distance-dependent reliability..."

依赖: 仅 torch, 无额外依赖
================================================================================
"""

import torch
import torch.nn.functional as F


class ConfidenceAwareSparseSupervision(torch.nn.Module):
    """
    置信度感知的稀疏监督模块 (CASS)。

    从稀疏深度 GT 的局部几何属性中在线计算 per-pixel 置信度权重，
    无需额外标注、无需修改模型架构、无可学习参数（纯几何先验）。

    三个置信度因子:

    ┌─────────────────┬────────────────────────────────────────┬───────────┐
    │ 因子            │ 物理含义                                 │ 高置信条件 │
    ├─────────────────┼────────────────────────────────────────┼───────────┤
    │ C_density       │ 局部有效邻居数量                          │ 邻居多     │
    │ C_smooth        │ 局部深度变化幅度 (相对标准差)              │ 变化平缓   │
    │ C_distance      │ 目标距离 (已有距离衰减)                   │ 距离近     │
    └─────────────────┴────────────────────────────────────────┴───────────┘

    参数:
      density_kernel: 局部密度统计的窗口大小 (奇数)
      density_min:    最低密度置信度 (避免零权重)
      smooth_kernel:  深度一致性统计的窗口大小 (奇数)
      smooth_sigma:   深度变化的容忍度 (相对标准差，越大越宽容)
      smooth_min:     最低一致性置信度
      dist_start:     距离衰减起点 (米)
      dist_end:       距离衰减终点 (米)
      dist_min:       远距离最低权重
    """

    def __init__(
        self,
        density_kernel=7,
        density_min=0.1,
        smooth_kernel=5,
        smooth_sigma=0.15,
        smooth_min=0.1,
        dist_start=60.0,
        dist_end=100.0,
        dist_min=0.05,
        disable_density=False,
        disable_smooth=False,
        disable_distance=False,
    ):
        super().__init__()
        self.density_kernel = density_kernel
        self.density_min = density_min
        self.smooth_kernel = smooth_kernel
        self.smooth_sigma = smooth_sigma
        self.smooth_min = smooth_min
        self.dist_start = dist_start
        self.dist_end = dist_end
        self.dist_min = dist_min
        # 因子消融开关: True 表示禁用该因子 (设为恒 1)
        self.disable_density = disable_density
        self.disable_smooth = disable_smooth
        self.disable_distance = disable_distance

    @torch.no_grad()
    def forward(self, depth_gt, mask, **kwargs):
        """
        计算置信度权重图。

        参数:
          depth_gt: (B, H, W) 稀疏深度 GT (米), 0=无效
          mask:     (B, H, W) 有效像素掩码 (1=有效, 0=无效)

        返回:
          confidence: (B, H, W) 置信度权重, 范围 [min_val, 1.0]
                      仅在 mask>0 的位置有意义
        """
        c_density = self._compute_density_confidence(mask) if not self.disable_density else torch.ones_like(depth_gt)
        c_smooth = self._compute_smoothness_confidence(depth_gt, mask) if not self.disable_smooth else torch.ones_like(depth_gt)
        c_distance = self._compute_distance_confidence(depth_gt) if not self.disable_distance else torch.ones_like(depth_gt)

        # 三因子相乘
        confidence = c_density * c_smooth * c_distance

        # 归一化到 [0, 1]，保持整体权重量级稳定
        valid = mask > 0
        if valid.sum() > 0:
            max_val = confidence[valid].max()
            if max_val > 0:
                confidence = confidence / (max_val + 1e-8)

        return confidence

    def _compute_density_confidence(self, mask):
        """
        局部支撑密度置信度。

        原理:
          在每个有效像素的邻域内统计有效邻居的数量。
          孤立的零散点（可能是噪声/动态物体残留）邻居少 → 低置信度。
          密集连片区域（轨道表面、建筑墙面）邻居多 → 高置信度。

        实现:
          用 average pooling 对 mask 做卷积 = 局部密度。
          归一化到 [density_min, 1.0]。
        """
        k = self.density_kernel
        pad = k // 2

        # (B, H, W) → (B, 1, H, W) for pooling
        mask_4d = mask.unsqueeze(1).float()
        local_density = F.avg_pool2d(mask_4d, kernel_size=k, stride=1, padding=pad)
        local_density = local_density.squeeze(1)  # (B, H, W)

        # 归一化: density 范围 [0, 1]，映射到 [density_min, 1.0]
        c = local_density * (1.0 - self.density_min) + self.density_min
        return c.clamp(self.density_min, 1.0)

    def _compute_smoothness_confidence(self, depth_gt, mask):
        """
        深度一致性置信度。

        原理:
          在每个有效像素的邻域内计算深度的相对标准差 (CV = std/mean)。
          平坦表面 (轨道、路面): CV 小 → 高置信度
          深度边缘 (物体边界):  CV 大 → 低置信度 (投影可能跨越前后景)
          植被/不规则表面:      CV 中等 → 中置信度

        实现:
          用 local mean 和 local mean-of-squares 计算 local variance，
          然后转为相对标准差。注意只统计有效像素。
        """
        k = self.smooth_kernel
        pad = k // 2

        mask_4d = mask.unsqueeze(1).float()
        depth_4d = (depth_gt * mask).unsqueeze(1).float()  # 无效位置为 0

        # 局部有效像素计数
        count = F.avg_pool2d(mask_4d, k, 1, pad) * (k * k)
        count = count.clamp(min=1.0)  # 避免除零

        # 局部均值 (仅有效像素)
        local_sum = F.avg_pool2d(depth_4d, k, 1, pad) * (k * k)
        local_mean = local_sum / count

        # 局部方差
        depth_sq = (depth_gt * mask).unsqueeze(1).float() ** 2
        local_sum_sq = F.avg_pool2d(depth_sq, k, 1, pad) * (k * k)
        local_mean_sq = local_sum_sq / count
        local_var = (local_mean_sq - local_mean ** 2).clamp(min=0)

        # 相对标准差 (Coefficient of Variation)
        local_std = torch.sqrt(local_var + 1e-8)
        local_mean_safe = local_mean.clamp(min=0.1)
        cv = (local_std / local_mean_safe).squeeze(1)  # (B, H, W)

        # CV → 置信度: CV 越小越好
        # 使用高斯映射: c = exp(-(cv/sigma)^2)
        c = torch.exp(-(cv / self.smooth_sigma) ** 2)
        c = c * (1.0 - self.smooth_min) + self.smooth_min

        return c.clamp(self.smooth_min, 1.0)

    def _compute_distance_confidence(self, depth_gt):
        """
        距离衰减置信度 (与之前的 compute_distance_weight 统一)。

        近处 → 1.0, 远处 → dist_min, 中间线性衰减。
        """
        w = 1.0 - (depth_gt - self.dist_start) / (self.dist_end - self.dist_start + 1e-6)
        return w.clamp(min=self.dist_min, max=1.0)

    def get_config_dict(self):
        """返回配置字典，方便保存到 config.json。"""
        return {
            "density_kernel": self.density_kernel,
            "density_min": self.density_min,
            "smooth_kernel": self.smooth_kernel,
            "smooth_sigma": self.smooth_sigma,
            "smooth_min": self.smooth_min,
            "dist_start": self.dist_start,
            "dist_end": self.dist_end,
            "dist_min": self.dist_min,
            "disable_density": self.disable_density,
            "disable_smooth": self.disable_smooth,
            "disable_distance": self.disable_distance,
        }


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  独立测试 / 可视化                                                      ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def visualize_confidence(depth_gt, mask, save_path=None):
    """
    可视化各置信度因子，用于论文 figure 和调试。

    生成: [稀疏GT | 密度置信度 | 平滑置信度 | 距离置信度 | 综合置信度] 五联图。
    """
    import cv2
    import numpy as np

    cass = ConfidenceAwareSparseSupervision()

    if isinstance(depth_gt, np.ndarray):
        depth_gt = torch.from_numpy(depth_gt).unsqueeze(0).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()

    c_density = cass._compute_density_confidence(mask).squeeze(0).numpy()
    c_smooth = cass._compute_smoothness_confidence(depth_gt, mask).squeeze(0).numpy()
    c_distance = cass._compute_distance_confidence(depth_gt).squeeze(0).numpy()
    c_total = cass(depth_gt, mask).squeeze(0).numpy()
    mask_np = mask.squeeze(0).numpy()

    def to_heatmap(arr, mask_np, vmin=0, vmax=1):
        norm = np.clip((arr - vmin) / (vmax - vmin + 1e-8), 0, 1)
        u8 = (norm * 255).astype(np.uint8)
        color = cv2.applyColorMap(u8, cv2.COLORMAP_VIRIDIS)
        color[mask_np == 0] = [30, 30, 30]
        return color

    # 稀疏 GT 可视化
    gt_np = depth_gt.squeeze(0).numpy()
    gt_u8 = np.clip(gt_np / 80.0, 0, 1)
    gt_u8 = (gt_u8 * 255).astype(np.uint8)
    gt_u8 = cv2.dilate(gt_u8, np.ones((3, 3), np.uint8))
    gt_color = cv2.applyColorMap(gt_u8, cv2.COLORMAP_TURBO)
    gt_color[gt_u8 == 0] = 0

    panels = [
        gt_color,
        to_heatmap(c_density, mask_np),
        to_heatmap(c_smooth, mask_np),
        to_heatmap(c_distance, mask_np),
        to_heatmap(c_total, mask_np),
    ]

    labels = ["Sparse GT", "C_density", "C_smooth", "C_distance", "C_total (CASS)"]
    for img, label in zip(panels, labels):
        cv2.putText(img, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    result = np.concatenate(panels, axis=1)

    if save_path:
        cv2.imwrite(save_path, result, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"置信度可视化 → {save_path}")

    return result


if __name__ == "__main__":
    # 简单测试: 生成一个模拟稀疏深度图
    import numpy as np

    H, W = 518, 518
    depth = np.zeros((H, W), dtype=np.float32)
    mask = np.zeros((H, W), dtype=np.float32)

    # 模拟: 下半部分有稀疏点 (近处), 上半部分有零散点 (远处)
    np.random.seed(42)
    for _ in range(3000):
        y, x = np.random.randint(H // 2, H), np.random.randint(0, W)
        depth[y, x] = np.random.uniform(5, 40)
        mask[y, x] = 1.0
    for _ in range(500):
        y, x = np.random.randint(0, H // 2), np.random.randint(0, W)
        depth[y, x] = np.random.uniform(50, 120)
        mask[y, x] = 1.0

    # 测试模块
    cass = ConfidenceAwareSparseSupervision()
    depth_t = torch.from_numpy(depth).unsqueeze(0)
    mask_t = torch.from_numpy(mask).unsqueeze(0)
    conf = cass(depth_t, mask_t)
    print(f"Confidence shape: {conf.shape}")
    print(f"Confidence range: [{conf[mask_t > 0].min():.4f}, {conf[mask_t > 0].max():.4f}]")
    print(f"Confidence mean:  {conf[mask_t > 0].mean():.4f}")

    # 可视化
    visualize_confidence(depth, mask, save_path="cass_test.jpg")
