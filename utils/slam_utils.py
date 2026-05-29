import torch
import torch.nn.functional as F
import numpy as np


def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]


def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


def depth_reg(depth, gt_image, huber_eps=0.1, mask=None):
    mask_v, mask_h = image_gradient_mask(depth)
    gray_grad_v, gray_grad_h = image_gradient(gt_image.mean(dim=0, keepdim=True))
    depth_grad_v, depth_grad_h = image_gradient(depth)
    gray_grad_v, gray_grad_h = gray_grad_v[mask_v], gray_grad_h[mask_h]
    depth_grad_v, depth_grad_h = depth_grad_v[mask_v], depth_grad_h[mask_h]

    w_h = torch.exp(-10 * gray_grad_h**2)
    w_v = torch.exp(-10 * gray_grad_v**2)
    err = (w_h * torch.abs(depth_grad_h)).mean() + (
        w_v * torch.abs(depth_grad_v)
    ).mean()
    return err


def get_loss_tracking(config, image, depth, opacity, viewpoint, initialization=False):
    image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_tracking_rgb(config, image_ab, depth, opacity, viewpoint)
    return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint)


def get_loss_tracking_rgb(config, image, depth, opacity, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask
    l1 = opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    return l1.mean()


def get_loss_tracking_rgbd(
    config, image, depth, opacity, viewpoint, initialization=False
):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    l1_rgb = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint)
    depth_mask = depth_pixel_mask * opacity_mask
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)
    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()


def _as_depth_2d(depth):
    if depth is None:
        return None
    if isinstance(depth, np.ndarray):
        depth = torch.from_numpy(depth)
    if depth.dim() == 3 and depth.shape[0] == 1:
        depth = depth.squeeze(0)
    elif depth.dim() == 3:
        depth = depth.squeeze()
    return depth


def _resize_2d_like(tensor, shape, mode="bilinear"):
    if tensor is None or tuple(tensor.shape[-2:]) == tuple(shape):
        return tensor
    align_corners = False if mode in {"bilinear", "bicubic"} else None
    kwargs = {"mode": mode}
    if align_corners is not None:
        kwargs["align_corners"] = align_corners
    return F.interpolate(tensor[None, None].float(), size=shape, **kwargs).squeeze(0).squeeze(0)



def _finite_float(value, default=float("nan")):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _dgc_log_scale(config, render_depth, viewpoint, use_depth_scale):
    cfg = config.get("DepthGaussianConsistency", {}) or {}
    log_scale_delta = getattr(viewpoint, "depth_log_scale_delta", None)
    if use_depth_scale and log_scale_delta is not None and cfg.get("learn_depth_scale", False):
        clamp = float(cfg.get("depth_log_scale_clamp", 0.15))
        return torch.clamp(log_scale_delta, -clamp, clamp).view(())
    return render_depth.new_tensor(0.0)


def _dgc_scale_saturated(config, log_scale):
    cfg = config.get("DepthGaussianConsistency", {}) or {}
    clamp = float(cfg.get("depth_log_scale_clamp", 0.15))
    if clamp <= 0:
        return False
    return bool(torch.abs(log_scale.detach()).item() >= 0.98 * clamp)


def _dgc_zero_stats(config, viewpoint, depth, reason, active=False, extra=None):
    cfg = config.get("DepthGaussianConsistency", {}) or {}
    rail_info = getattr(viewpoint, "rail_info", {}) or {}
    log_scale_delta = getattr(viewpoint, "depth_log_scale_delta", None)
    if log_scale_delta is not None:
        clamp = float(cfg.get("depth_log_scale_clamp", 0.15))
        log_scale = torch.clamp(log_scale_delta.detach(), -clamp, clamp).view(())
        learned_scale = float(torch.exp(log_scale).item())
        scale_saturated = bool(abs(float(log_scale.item())) >= 0.98 * clamp) if clamp > 0 else False
    else:
        learned_scale = 1.0
        scale_saturated = False
    stats = {
        "mode": cfg.get("mode", "naive"),
        "active": bool(active),
        "skip_reason": reason,
        "frame_weight": 0.0,
        "dgc_loss": 0.0,
        "dgc_data_loss": 0.0,
        "dgc_prior_loss": 0.0,
        "metric_loss": 0.0,
        "shape_loss": 0.0,
        "valid_ratio": 0.0,
        "rail_mask_ratio": 0.0,
        "opacity_valid_ratio": 0.0,
        "learned_depth_scale": learned_scale,
        "scale_saturated": scale_saturated,
        "median_abs_residual": float("nan"),
        "median_log_residual": float("nan"),
        "normalization_offset_render": float("nan"),
        "normalization_offset_input": float("nan"),
        "rail_status": rail_info.get("status", ""),
        "rail_confidence": _finite_float(rail_info.get("confidence")),
        "rail_samples": _finite_float(rail_info.get("samples")),
        "rail_selected_ratio": _finite_float(rail_info.get("selected_ratio")),
        "rail_row_scale_raw_mad": _finite_float(rail_info.get("row_scale_raw_mad")),
        "rail_pixel_width_mad": _finite_float(rail_info.get("pixel_width_mad")),
    }
    if extra:
        stats.update(extra)
    return depth.sum() * 0.0, stats


def _dgc_robust_loss(residual, cfg):
    robust = str(cfg.get("robust_loss", cfg.get("loss_type", "log_l1"))).lower()
    if robust in {"huber", "huber_log", "smooth_l1"}:
        delta = float(cfg.get("huber_delta", 0.05))
        return F.huber_loss(residual, torch.zeros_like(residual), reduction="mean", delta=delta)
    if robust in {"l2", "mse"}:
        return (residual ** 2).mean()
    return residual.abs().mean()


def _dgc_get_nested(cfg, section, key, default):
    block = cfg.get(section, {}) or {}
    return block.get(key, default)


def _build_rail_corridor_mask(rail_info, shape, device, expand_width_frac=0.35):
    h, w = int(shape[0]), int(shape[1])
    mask = torch.zeros((h, w), dtype=torch.bool, device=device)
    points = rail_info.get("points") or []
    if not points:
        return mask

    src_h = _finite_float(rail_info.get("image_height"), h)
    src_w = _finite_float(rail_info.get("image_width"), w)
    if not np.isfinite(src_h) or src_h <= 0:
        src_h = h
    if not np.isfinite(src_w) or src_w <= 0:
        src_w = w
    sx = float(w) / float(src_w)
    sy = float(h) / float(src_h)

    scaled = []
    for point in points:
        if point is None or len(point) < 4:
            continue
        lx, y1, rx, y2 = [float(v) for v in point[:4]]
        y = 0.5 * (y1 + y2) * sy
        lx *= sx
        rx *= sx
        if rx < lx:
            lx, rx = rx, lx
        if not (np.isfinite(lx) and np.isfinite(rx) and np.isfinite(y)):
            continue
        if rx - lx < 1:
            continue
        scaled.append((lx, y, rx))
    if not scaled:
        return mask

    scaled.sort(key=lambda v: v[1])
    if len(scaled) == 1:
        lx, y, rx = scaled[0]
        margin = max((rx - lx) * expand_width_frac, 4.0)
        y0 = max(0, int(round(y - max(h * 0.02, 4))))
        y1 = min(h, int(round(y + max(h * 0.02, 4))) + 1)
        x0 = max(0, int(np.floor(lx - margin)))
        x1 = min(w, int(np.ceil(rx + margin)) + 1)
        if y1 > y0 and x1 > x0:
            mask[y0:y1, x0:x1] = True
        return mask

    for (lx0, y0f, rx0), (lx1, y1f, rx1) in zip(scaled[:-1], scaled[1:]):
        y0 = int(np.clip(round(y0f), 0, h - 1))
        y1 = int(np.clip(round(y1f), 0, h - 1))
        if y1 < y0:
            y0, y1 = y1, y0
            lx0, lx1 = lx1, lx0
            rx0, rx1 = rx1, rx0
        if y1 == y0:
            ys = [y0]
        else:
            ys = range(y0, y1 + 1)
        denom = max(float(y1 - y0), 1.0)
        for yy in ys:
            t = float(yy - y0) / denom
            lx = (1.0 - t) * lx0 + t * lx1
            rx = (1.0 - t) * rx0 + t * rx1
            width = max(rx - lx, 1.0)
            margin = max(width * expand_width_frac, 4.0)
            x0 = max(0, int(np.floor(lx - margin)))
            x1 = min(w, int(np.ceil(rx + margin)) + 1)
            if x1 > x0:
                mask[yy, x0:x1] = True
    return mask


def _dgc_rail_gate(config, viewpoint):
    cfg = config.get("DepthGaussianConsistency", {}) or {}
    gate_cfg = cfg.get("gating", {}) or {}
    rail_info = getattr(viewpoint, "rail_info", {}) or {}
    status = str(rail_info.get("status", ""))
    allowed = gate_cfg.get("allowed_rail_statuses", ["corrected"])
    if isinstance(allowed, str):
        allowed = [v.strip() for v in allowed.split(",") if v.strip()]
    if status not in set(allowed):
        return False, f"rail_status:{status or 'missing'}", 0.0

    confidence = _finite_float(rail_info.get("confidence"), 0.0)
    min_conf = float(gate_cfg.get("min_rail_confidence", 0.45))
    if confidence < min_conf:
        return False, "low_rail_confidence", 0.0

    row_scale_mad = _finite_float(rail_info.get("row_scale_raw_mad"))
    max_row_scale_mad = gate_cfg.get("max_row_scale_raw_mad", None)
    if max_row_scale_mad is not None and np.isfinite(row_scale_mad) and row_scale_mad > float(max_row_scale_mad):
        return False, "row_scale_raw_mad_high", 0.0

    pixel_width_mad = _finite_float(rail_info.get("pixel_width_mad"))
    max_pixel_width_mad = gate_cfg.get("max_pixel_width_mad", None)
    if max_pixel_width_mad is not None and np.isfinite(pixel_width_mad) and pixel_width_mad > float(max_pixel_width_mad):
        return False, "pixel_width_mad_high", 0.0

    denom = max(1.0 - min_conf, 1e-6)
    frame_weight = float(np.clip((confidence - min_conf) / denom, 0.0, 1.0))
    return True, "", frame_weight


def get_depth_gaussian_consistency_loss(config, depth, viewpoint, opacity, use_depth_scale=True):
    cfg = config.get("DepthGaussianConsistency", {}) or {}
    zero = depth.sum() * 0.0
    if not cfg.get("enabled", False) or viewpoint.depth is None:
        return zero, None

    mode = str(cfg.get("mode", "naive")).lower()
    render_depth = _as_depth_2d(depth).to(dtype=torch.float32, device=depth.device)
    input_depth = _as_depth_2d(viewpoint.depth).to(dtype=torch.float32, device=depth.device)
    input_depth = _resize_2d_like(input_depth, render_depth.shape, mode="bilinear")

    opacity_2d = _as_depth_2d(opacity)
    if opacity_2d is not None:
        opacity_2d = opacity_2d.to(dtype=torch.float32, device=depth.device)
        opacity_2d = _resize_2d_like(opacity_2d, render_depth.shape, mode="bilinear")

    gt_image = viewpoint.original_image.cuda()
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).to(device=depth.device)
    rgb_mask = _resize_2d_like(rgb_mask.float(), render_depth.shape, mode="nearest") > 0.5

    rail_mask = None
    if mode == "rail_selective_normalized" and bool(_dgc_get_nested(cfg, "rail_mask", "enabled", True)):
        expand_width_frac = float(_dgc_get_nested(cfg, "rail_mask", "expand_width_frac", 0.35))
        rail_mask = _build_rail_corridor_mask(
            getattr(viewpoint, "rail_info", {}) or {},
            render_depth.shape,
            depth.device,
            expand_width_frac=expand_width_frac,
        )

    downsample = max(int(cfg.get("downsample", 1)), 1)
    if downsample > 1:
        render_depth = render_depth[::downsample, ::downsample]
        input_depth = input_depth[::downsample, ::downsample]
        rgb_mask = rgb_mask[::downsample, ::downsample]
        if opacity_2d is not None:
            opacity_2d = opacity_2d[::downsample, ::downsample]
        if rail_mask is not None:
            rail_mask = rail_mask[::downsample, ::downsample]

    min_depth = float(cfg.get("min_depth", 0.1))
    max_depth = float(cfg.get("max_depth", 120.0))
    valid = (
        torch.isfinite(render_depth)
        & torch.isfinite(input_depth)
        & (render_depth > min_depth)
        & (input_depth > min_depth)
        & (render_depth < max_depth)
        & (input_depth < max_depth)
        & rgb_mask
    )

    opacity_valid_ratio = 0.0
    if opacity_2d is not None:
        opacity_mask = torch.isfinite(opacity_2d) & (opacity_2d > float(_dgc_get_nested(cfg, "gating", "opacity_threshold", cfg.get("opacity_threshold", 0.95))))
        opacity_valid_ratio = float(opacity_mask.float().mean().detach().item()) if opacity_mask.numel() else 0.0
        valid = valid & opacity_mask

    rail_mask_ratio = 0.0
    if rail_mask is not None:
        rail_mask_ratio = float(rail_mask.float().mean().detach().item()) if rail_mask.numel() else 0.0
        valid = valid & rail_mask

    active = True
    skip_reason = ""
    frame_weight = 1.0
    if mode == "rail_selective_normalized":
        gate_ok, skip_reason, frame_weight = _dgc_rail_gate(config, viewpoint)
        if not gate_ok:
            return _dgc_zero_stats(
                config,
                viewpoint,
                depth,
                skip_reason,
                active=False,
                extra={"rail_mask_ratio": rail_mask_ratio, "opacity_valid_ratio": opacity_valid_ratio},
            )
        if rail_mask is not None and not bool(rail_mask.any()):
            return _dgc_zero_stats(
                config,
                viewpoint,
                depth,
                "empty_rail_mask",
                active=False,
                extra={"rail_mask_ratio": rail_mask_ratio, "opacity_valid_ratio": opacity_valid_ratio},
            )

    valid_ratio = valid.float().mean().detach() if valid.numel() else torch.tensor(0.0, device=depth.device)
    min_valid_ratio = float(_dgc_get_nested(cfg, "gating", "min_valid_ratio", 0.0 if mode != "rail_selective_normalized" else 0.03))
    if (not bool(valid.any())) or float(valid_ratio.item()) < min_valid_ratio:
        return _dgc_zero_stats(
            config,
            viewpoint,
            depth,
            "low_valid_ratio" if bool(valid.numel()) else "empty_valid_mask",
            active=False,
            extra={
                "frame_weight": frame_weight,
                "valid_ratio": float(valid_ratio.item()) if valid.numel() else 0.0,
                "rail_mask_ratio": rail_mask_ratio,
                "opacity_valid_ratio": opacity_valid_ratio,
            },
        )

    log_scale = _dgc_log_scale(config, render_depth, viewpoint, use_depth_scale)
    target_depth = input_depth * torch.exp(log_scale)

    render_valid_depth = render_depth[valid].clamp(min=min_depth, max=max_depth)
    target_valid_depth = target_depth[valid].clamp(min=min_depth, max=max_depth)
    render_log = torch.log(render_valid_depth)
    target_log = torch.log(target_valid_depth)
    metric_residual = render_log - target_log

    if mode == "rail_selective_normalized":
        norm_cfg = cfg.get("normalized_loss", {}) or {}
        render_offset = torch.median(render_log.detach())
        target_offset = torch.median(target_log.detach())
        shape_residual = (render_log - render_offset) - (target_log - target_offset)
        metric_weight = float(norm_cfg.get("metric_weight", 0.25))
        shape_weight = float(norm_cfg.get("shape_weight", 0.75))
        metric_loss = _dgc_robust_loss(metric_residual, {**cfg, **norm_cfg})
        shape_loss = _dgc_robust_loss(shape_residual, {**cfg, **norm_cfg})
        data_loss = metric_weight * metric_loss + shape_weight * shape_loss
        active = True
    else:
        render_offset = render_log.new_tensor(float("nan"))
        target_offset = target_log.new_tensor(float("nan"))
        shape_loss = render_log.new_tensor(0.0)
        metric_loss = _dgc_robust_loss(metric_residual, cfg)
        data_loss = metric_loss

    prior_weight = float(cfg.get("depth_scale_prior_weight", 0.0))
    prior_loss = prior_weight * (log_scale ** 2)
    dgc_loss = float(frame_weight) * data_loss + prior_loss

    with torch.no_grad():
        metric_residual_m = (render_valid_depth - target_valid_depth).abs()
        abs_log_residual = metric_residual.abs()
        rail_info = getattr(viewpoint, "rail_info", {}) or {}
        stats = {
            "mode": mode,
            "active": bool(active),
            "skip_reason": skip_reason,
            "frame_weight": float(frame_weight),
            "dgc_loss": float(dgc_loss.detach().item()),
            "dgc_data_loss": float(data_loss.detach().item()),
            "dgc_prior_loss": float(prior_loss.detach().item()),
            "metric_loss": float(metric_loss.detach().item()),
            "shape_loss": float(shape_loss.detach().item()),
            "valid_ratio": float(valid_ratio.item()),
            "rail_mask_ratio": float(rail_mask_ratio),
            "opacity_valid_ratio": float(opacity_valid_ratio),
            "learned_depth_scale": float(torch.exp(log_scale.detach()).item()),
            "scale_saturated": _dgc_scale_saturated(config, log_scale),
            "median_abs_residual": float(torch.median(metric_residual_m).item()),
            "median_log_residual": float(torch.median(abs_log_residual).item()),
            "normalization_offset_render": float(render_offset.detach().item()),
            "normalization_offset_input": float(target_offset.detach().item()),
            "rail_status": rail_info.get("status", ""),
            "rail_confidence": _finite_float(rail_info.get("confidence")),
            "rail_samples": _finite_float(rail_info.get("samples")),
            "rail_selected_ratio": _finite_float(rail_info.get("selected_ratio")),
            "rail_row_scale_raw_mad": _finite_float(rail_info.get("row_scale_raw_mad")),
            "rail_pixel_width_mad": _finite_float(rail_info.get("pixel_width_mad")),
        }
    return dgc_loss, stats

def get_loss_mapping(config, image, depth, viewpoint, opacity, initialization=False, return_stats=False, use_depth_scale=True):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b

    if config["Training"].get("monocular", False):
        loss = get_loss_mapping_rgb(config, image_ab, depth, viewpoint)
    else:
        loss = get_loss_mapping_rgbd(config, image_ab, depth, viewpoint)

    stats = None
    dgc_cfg = config.get("DepthGaussianConsistency", {}) or {}
    if dgc_cfg.get("enabled", False) and not initialization:
        dgc_loss, stats = get_depth_gaussian_consistency_loss(
            config, depth, viewpoint, opacity, use_depth_scale=use_depth_scale
        )
        loss = loss + float(dgc_cfg.get("lambda_depth", 0.0)) * dgc_loss
    return (loss, stats) if return_stats else loss


def get_loss_mapping_rgb(config, image, depth, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)

    return l1_rgb.mean()


def get_loss_mapping_rgbd(config, image, depth, viewpoint, initialization=False):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    gt_image = viewpoint.original_image.cuda()

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)

    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)

    return alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean()


def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    depth = depth.detach().clone()
    opacity = opacity.detach().squeeze()
    valid = depth > 0
    if opacity is not None:
        valid = torch.logical_and(valid, opacity > 0.95)
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()

def get_matched_camera_points_vectorized(depth_map1: np.ndarray, feat_points1: np.ndarray, 
                                         depth_map2: np.ndarray, feat_points2: np.ndarray, K: np.ndarray):
    """
    Compute the matched 3D points from two depth maps in their respective camera coordinate systems.

    Parameters:
      depth_map1: Depth map of the first viewpoint, shape [H, W]
      feat_points1: Feature points from the first viewpoint, shape [n, 2], format (u, v)

      depth_map2: Depth map of the second viewpoint, shape [H, W]
      feat_points2: Feature points from the second viewpoint, shape [n, 2], format (u, v)

      K: Camera intrinsic matrix, shape [3, 3]

    Returns:
      pts_cam1: 3D points in the first camera's coordinate system, shape [n, 3]
      pts_cam2: 3D points in the second camera's coordinate system, shape [n, 3]
    """
    # Inverse of the intrinsic matrix
    K_inv = np.linalg.inv(K)

    # ------------------------------
    # First camera points
    # ------------------------------
    n = feat_points1.shape[0]
    pixels1 = np.hstack((feat_points1, np.ones((n, 1))))
    u1 = np.round(feat_points1[:, 0]).astype(int)
    v1 = np.round(feat_points1[:, 1]).astype(int)
    u1 = np.clip(u1, 0, depth_map1.shape[1]-1)
    v1 = np.clip(v1, 0, depth_map1.shape[0]-1)
    depth1 = depth_map1[v1, u1]
    pts_cam1 = depth1[:, np.newaxis] * (K_inv @ pixels1.T).T  # shape [n, 3]

    # ------------------------------
    # Second camera points
    # ------------------------------
    n2 = feat_points2.shape[0]
    pixels2 = np.hstack((feat_points2, np.ones((n2, 1))))
    u2 = np.round(feat_points2[:, 0]).astype(int)
    v2 = np.round(feat_points2[:, 1]).astype(int)
    u2 = np.clip(u2, 0, depth_map2.shape[1]-1)
    v2 = np.clip(v2, 0, depth_map2.shape[0]-1)
    depth2 = depth_map2[v2, u2]
    pts_cam2 = depth2[:, np.newaxis] * (K_inv @ pixels2.T).T

    return pts_cam1, pts_cam2



import open3d as o3d

def compute_sim3_open3d(pts1, pts2):
    """
    Compute the Sim(3) transformation between two point clouds using Open3D.
    
    Parameters:
      pts1: Source point cloud with shape [N, 3]
      pts2: Target point cloud with shape [N, 3]
    
    Returns:
      scale: Scaling factor
      R: Rotation matrix (3x3)
      t: Translation vector (3,)
    """
    pcd1 = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pts1)
    pcd2.points = o3d.utility.Vector3dVector(pts2)

    # Using ICP
    threshold = 1.0  # Maximum correspondence distance allowed
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd1, pcd2, threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )

    T = reg_p2p.transformation  # 4x4 transformation matrix
    R = T[:3, :3].copy()  # Copy R to make it writable
    t = T[:3, 3].copy()   # Copy t to make it writable
    scale = np.linalg.det(R) ** (1/3)  # Compute scale factor

    # Normalize R to make it an orthogonal matrix
    R /= scale

    return scale, R, t

def update_viewpoints_from_poses(viewpoints: dict, poses_update: torch.Tensor):
    """
    Efficiently update the R and T of Camera objects using batched matrix inversion.
    """
    assert len(viewpoints) == poses_update.shape[0], "Mismatched number of poses and viewpoints!"
    sorted_keys = sorted(viewpoints.keys())
    
    w2c_all = torch.linalg.inv(poses_update)

    R_all = w2c_all[:, :3, :3]
    T_all = w2c_all[:, :3, 3]

    for i, key in enumerate(sorted_keys):
        camera = viewpoints[key]
        camera.update_RT(R_all[i], T_all[i])


def update_point_cloud_vectorized(
    pcd: torch.Tensor,
    poses_old, # np.ndarray or torch.tensor
    poses_update, # np.ndarray or torch.tensor
    index_pose: torch.Tensor,
    keyframe_indices: list[int]  # e.g., [1, 4, 5, 9]
) -> torch.Tensor:
    """
    Update 3D point coordinates after pose optimization (e.g. loop closure),
    using a fully vectorized approach on the GPU.

    This function applies a rigid transformation to each point, based on
    the difference between old and updated camera poses. The transformation aligns
    point cloud data with the updated poses after global optimization.

    This function is designed for use before voxelization, so the updated points
    can then be discretized into a voxel grid.

    Args:
        pcd: Tensor of shape [N, 3], original point cloud in world coordinates
        poses_old: np.ndarray or torch.Tensor of shape [n, 4, 4], original camera poses (C2W)
        poses_update: np.ndarray or torch.Tensor of shape [n, 4, 4], updated camera poses (C2W)
        index_pose: Tensor of shape [N], pose index for each point in the point cloud
        keyframe_indices: [K], list mapping each pose to its keyframe index

    Returns:
        updated_pcd: Tensor of shape [N, 3], point cloud updated to the corrected poses
    """
    device = pcd.device

    if isinstance(poses_old, np.ndarray):
        poses_old = torch.from_numpy(poses_old)
    if isinstance(poses_update, np.ndarray):
        poses_update = torch.from_numpy(poses_update)

    poses_old = poses_old.float().to(device)
    poses_update = poses_update.float().to(device)

    # Build mapping from keyframe index to index in pose list
    kf_map = {kf_idx: i for i, kf_idx in enumerate(keyframe_indices)}
    pose_ids = torch.tensor([kf_map[int(i.item())] for i in index_pose], device=device)

    pose_old_batch = poses_old[pose_ids] # [N, 4, 4]
    pose_update_batch = poses_update[pose_ids] # [N, 4, 4]

    ones = torch.ones((pcd.shape[0], 1), device=device)
    pcd_h = torch.cat([pcd, ones], dim=1).unsqueeze(-1) # [N, 4, 1]

    pose_old_inv_batch = torch.linalg.inv(pose_old_batch) # [N, 4, 4]
    pcd_cam = torch.bmm(pose_old_inv_batch, pcd_h)
    pcd_updated_h = torch.bmm(pose_update_batch, pcd_cam)

    return pcd_updated_h[:, :3, 0]


def update_point_cloud_in_batches(
    pcd: torch.Tensor,
    poses_old, # np.ndarray or torch.tensor
    poses_update, # np.ndarray or torch.tensor
    index_pose: torch.Tensor,
    keyframe_indices: list[int],
    batch_size: int = 500_000
) -> torch.Tensor:
    """
    Update 3D point coordinates after pose graph optimization (e.g. loop closure),
    using batched processing to reduce GPU memory usage.

    This function performs the same rigid transformation as func update_point_cloud_vectorized(),
    but processes the point cloud in chunks to avoid out-of-memory errors on large-scale datasets.

    This function is designed for use before voxelization, so the updated points
    can then be discretized into a voxel grid.

    Args:
        pcd: Tensor of shape [N, 3], original point cloud in world coordinates
        poses_old: np.ndarray of shape [n, 4, 4], original camera poses (C2W)
        poses_update: np.ndarray of shape [n, 4, 4], updated camera poses (C2W)
        index_pose: Tensor of shape [N], pose index for each point in the point cloud
        keyframe_indices: [K], list mapping each pose to keyframe ID
        batch_size: Number of points to process per batch (adjust based on GPU memory)

    Returns:
        updated_pcd: Tensor of shape [N, 3], point cloud after applying updated poses

    A small exp for 10_000_000 pcds and 2000 poses below on a 4090 (torch.float32):
        Vectorized
            Time taken: 0.03 seconds
            Initial memory: 208.53 MB
            Final memory:   368.54 MB
            Peak memory:    3328.94 MB

        Batched (500k/batch)
            Time taken: 0.04 seconds
            Initial memory: 368.54 MB
            Final memory:   488.54 MB
            Peak memory:    677.25 MB
    """
    updated_pcd_list = []
    N = pcd.shape[0]

    for i in range(0, N, batch_size):
        pcd_batch = pcd[i:i + batch_size]
        index_batch = index_pose[i:i + batch_size]
        updated = update_point_cloud_vectorized(
            pcd_batch, poses_old, poses_update, index_batch, keyframe_indices
        )
        updated_pcd_list.append(updated)

    return torch.cat(updated_pcd_list, dim=0)