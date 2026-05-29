import csv
import os

import cv2
import numpy as np


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


class GaugeFreeRailScaleStabilizer:
    """Estimate a depth scale from the apparent 3D width of the rails.

    correction_mode="relative" keeps the original gauge-free behavior: calibrate
    an effective rail width from the first stable detections and keep it stable.
    correction_mode="metric_width" uses an explicit rail-head width prior.
    """

    def __init__(self, config, save_dir=None):
        self.enabled = bool(config.get("enabled", False))
        self.mode = str(config.get("mode", "hough")).lower()
        self.apply_correction = bool(config.get("apply_correction", True))
        self.correction_mode = str(config.get("correction_mode", "relative")).lower()
        if self.correction_mode not in {"relative", "metric_width"}:
            self.correction_mode = "relative"
        self.metric_width_m = _positive_float(config.get("metric_width_m", 0.0), 0.0)
        self.calibration_frames = int(config.get("calibration_frames", 30))
        self.min_confidence = float(config.get("min_confidence", 0.35))
        self.ema_alpha = float(config.get("ema_alpha", 0.35))
        self.min_scale = float(config.get("min_scale", 0.60))
        self.max_scale = float(config.get("max_scale", 1.40))
        self.max_scale_step = float(config.get("max_scale_step", 0.06))
        self.max_detect_width = int(config.get("max_detect_width", 1280))
        self.roi_y_min = float(config.get("roi_y_min", 0.45))
        self.roi_y_max = float(config.get("roi_y_max", 0.95))
        self.sample_y_fracs = config.get("sample_y_fracs", [0.58, 0.65, 0.72, 0.80, 0.88])
        self.min_samples = int(config.get("min_samples", 3))
        self.max_width_jump_ratio = float(config.get("max_width_jump_ratio", 0.12))
        self.log_diag = bool(config.get("log_diag", True))
        self.log_row_diag = bool(config.get("log_row_diag", True))
        self.save_debug = bool(config.get("save_debug", False))
        self.debug_every = int(config.get("debug_every", 20))
        self.metric_width_candidates_m = _positive_float_list(
            config.get("metric_width_candidates_m", [])
        )
        if self.metric_width_m > 0 and all(
            abs(self.metric_width_m - v) > 1e-6 for v in self.metric_width_candidates_m
        ):
            self.metric_width_candidates_m.append(self.metric_width_m)

        self.detector_checkpoint = config.get("detector_checkpoint", "")
        self.detector_model_name = config.get("detector_model_name", None)
        self.detector_image_size = tuple(config.get("detector_image_size", [768, 432]))
        self.detector_device = str(config.get("detector_device", "cuda"))
        self.detector_threshold = float(config.get("detector_threshold", 0.30))
        self.detector_min_peak = float(config.get("detector_min_peak", 0.25))
        self.detector_min_width_px = float(config.get("detector_min_width_px", 12.0))
        self.detector_center_x_frac = float(config.get("detector_center_x_frac", 0.50))
        self.detector_peak_topk = int(config.get("detector_peak_topk", 8))
        self.detector_peak_nms_px = int(config.get("detector_peak_nms_px", 24))
        self.detector_max_width_px_frac = float(config.get("detector_max_width_px_frac", 0.45))
        self.detector_max_center_jump_ratio = float(config.get("detector_max_center_jump_ratio", 0.18))
        self.detector_center_anchor_frames = max(int(config.get("detector_center_anchor_frames", 0)), 0)
        self.detector_max_anchor_center_shift_px = float(
            config.get("detector_max_anchor_center_shift_px", 0.0)
        )
        self.detector_max_anchor_center_shift_ratio = float(
            config.get("detector_max_anchor_center_shift_ratio", 0.0)
        )
        self.hold_last_scale_on_reject = bool(config.get("hold_last_scale_on_reject", False))
        hold_statuses = config.get(
            "hold_last_scale_statuses",
            [
                "width_jump",
                "center_jump",
                "center_anchor_shift",
                "row_inconsistent",
                "too_few_samples",
            ],
        )
        if isinstance(hold_statuses, str):
            hold_statuses = [s.strip() for s in hold_statuses.split(",")]
        self.hold_last_scale_statuses = {str(s) for s in hold_statuses}
        self.detector_row_consistency = bool(config.get("detector_row_consistency", False))
        self.detector_row_target_tracking = bool(
            config.get("detector_row_target_tracking", self.detector_row_consistency)
        )
        self.detector_max_row_center_shift_px = float(
            config.get("detector_max_row_center_shift_px", 0.0)
        )
        self.detector_max_row_center_shift_ratio = float(
            config.get("detector_max_row_center_shift_ratio", 0.0)
        )
        self.detector_overlay_alpha = float(config.get("detector_overlay_alpha", 0.75))
        self.detector_model = None
        self.detector_torch = None
        self.detector_loaded = False
        self.ego_path_guidance = None

        self.save_dir = save_dir
        self.ref_width = None
        self.width_history = []
        self.pixel_width_history = []
        self.last_good_width = None
        self.last_good_center = None
        self.detector_center_anchor_history = []
        self.detector_center_anchor = None
        self.scale_ema = 1.0
        self.scale_ema_initialized = False

        self.diag_path = os.path.join(save_dir, "rail_scale_diag.csv") if save_dir else None
        self.row_diag_path = os.path.join(save_dir, "rail_scale_rows.csv") if save_dir else None
        self.debug_dir = os.path.join(save_dir, "rail_scale_debug") if save_dir else None
        if self.save_debug and self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)

        ego_cfg = config.get("ego_path_guidance", {})
        if isinstance(ego_cfg, dict) and bool(ego_cfg.get("enabled", False)):
            try:
                from utils.ego_path_guidance import EgoPathGuidance

                self.ego_path_guidance = EgoPathGuidance(ego_cfg, save_dir=save_dir)
            except Exception:
                self.ego_path_guidance = None

    def apply(self, frame_idx, image_chw, depth, intrinsics):
        if not self.enabled:
            return depth, {"status": "disabled", "scale": 1.0}

        result = self._estimate_width(image_chw, depth, intrinsics)
        status = result.get("status", "unknown")
        confidence = float(result.get("confidence", 0.0))
        width = result.get("width", None)

        scale = 1.0
        applied_scale = 1.0
        raw_scale = 1.0
        scale_target_width = None
        if status == "ok" and width is not None and confidence >= self.min_confidence:
            self.width_history.append(float(width))
            pixel_width = result.get("pixel_width", None)
            if pixel_width is not None:
                self.pixel_width_history.append(float(pixel_width))

            if self.correction_mode == "metric_width":
                if self.metric_width_m > 0:
                    scale_target_width = self.metric_width_m
                else:
                    status = "missing_metric_width"
            else:
                if self.ref_width is None and len(self.width_history) >= self.calibration_frames:
                    recent = np.asarray(self.width_history[-self.calibration_frames:], dtype=np.float32)
                    self.ref_width = float(np.median(recent))
                scale_target_width = self.ref_width

            if scale_target_width is not None and width > 1e-6:
                raw_scale = float(np.clip(scale_target_width / width, self.min_scale, self.max_scale))
                if self.correction_mode == "metric_width" and not self.scale_ema_initialized:
                    self.scale_ema = raw_scale
                else:
                    raw_scale = self._limit_scale_step(raw_scale)
                    self.scale_ema = (
                        self.ema_alpha * raw_scale
                        + (1.0 - self.ema_alpha) * self.scale_ema
                    )
                self.scale_ema_initialized = True
                scale = float(np.clip(self.scale_ema, self.min_scale, self.max_scale))
                if self.apply_correction:
                    depth = depth * scale
                    applied_scale = scale
                    status = "corrected"
                else:
                    status = "shadow"
            elif status == "ok":
                status = "calibrating"
        elif self._should_hold_last_scale(status):
            if self.correction_mode == "metric_width" and self.metric_width_m > 0:
                scale_target_width = self.metric_width_m
            else:
                scale_target_width = self.ref_width
            scale = float(np.clip(self.scale_ema, self.min_scale, self.max_scale))
            raw_scale = scale
            depth = depth * scale
            applied_scale = scale
            status = f"{status}_hold"

        info = {
            "frame_id": frame_idx,
            "status": status,
            "width": width,
            "ref_width": self.ref_width,
            "scale_target_width": scale_target_width,
            "raw_scale": raw_scale,
            "scale": scale,
            "applied_scale": applied_scale,
            "apply_correction": self.apply_correction,
            "correction_mode": self.correction_mode,
            "metric_width_m": self.metric_width_m,
            "confidence": confidence,
            "samples": result.get("samples", 0),
            "mode": self.mode,
            "pixel_width": result.get("pixel_width"),
            "pixel_width_mad": result.get("pixel_width_mad"),
            "pixel_center": result.get("pixel_center"),
            "pixel_center_mad": result.get("pixel_center_mad"),
            "pixel_center_anchor": result.get("pixel_center_anchor", self.detector_center_anchor),
            "pixel_center_anchor_shift": result.get("pixel_center_anchor_shift"),
            "pixel_center_anchor_limit": result.get("pixel_center_anchor_limit"),
            "left_conf": result.get("left_conf"),
            "right_conf": result.get("right_conf"),
            "left_line": result.get("left_line"),
            "right_line": result.get("right_line"),
            "points": result.get("points"),
            "rows": result.get("rows"),
            "selected_ratio": result.get("selected_ratio"),
            "width_mad": result.get("width_mad"),
            "row_scale_raw_median": result.get("row_scale_raw_median"),
            "row_scale_raw_mad": result.get("row_scale_raw_mad"),
        }
        if self.ego_path_guidance is not None:
            ego_info = self.ego_path_guidance.observe(frame_idx, image_chw, info)
            info["ego_path_status"] = ego_info.get("status")
        self._log(info)
        self._log_rows(info)
        if self.save_debug and frame_idx % max(self.debug_every, 1) == 0:
            self._save_debug(frame_idx, image_chw, info)
        return depth, info

    def _estimate_width(self, image_chw, depth, intrinsics):
        if self.mode == "detector":
            return self._estimate_width_detector(image_chw, depth, intrinsics)
        return self._estimate_width_hough(image_chw, depth, intrinsics)

    def _estimate_width_hough(self, image_chw, depth, intrinsics):
        image = _chw_to_hwc_uint8(image_chw)
        if image is None or depth is None or depth.size == 0:
            return {"status": "invalid_input", "confidence": 0.0}

        h, w = depth.shape[:2]
        scale = min(float(self.max_detect_width) / max(w, 1), 1.0)
        det = cv2.resize(image, (int(w * scale), int(h * scale))) if scale < 1.0 else image
        gray = cv2.cvtColor(det, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        y0 = int(gray.shape[0] * self.roi_y_min)
        y1 = int(gray.shape[0] * self.roi_y_max)
        roi = gray[y0:y1]
        edges = cv2.Canny(roi, 60, 160)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=60,
            minLineLength=max(40, int(0.08 * gray.shape[1])),
            maxLineGap=max(20, int(0.03 * gray.shape[1])),
        )
        if lines is None:
            return {"status": "no_lines", "confidence": 0.0}

        left, right = self._select_line_pair(lines[:, 0], gray.shape[1], gray.shape[0], y0, scale)
        if left is None or right is None:
            return {"status": "no_pair", "confidence": 0.0}

        widths = []
        for yf in self.sample_y_fracs:
            y = int(h * float(yf))
            xl = _line_x_at_y(left, y)
            xr = _line_x_at_y(right, y)
            if xl is None or xr is None or not (0 <= xl < w and 0 <= xr < w) or xr <= xl:
                continue
            z_l = _sample_depth(depth, xl, y)
            z_r = _sample_depth(depth, xr, y)
            if z_l <= 0 or z_r <= 0:
                continue
            p_l = _backproject(xl, y, z_l, intrinsics)
            p_r = _backproject(xr, y, z_r, intrinsics)
            width_3d = float(np.linalg.norm(p_r - p_l))
            if np.isfinite(width_3d) and width_3d > 1e-6:
                widths.append(width_3d)

        if len(widths) < 3:
            return {
                "status": "too_few_samples",
                "confidence": 0.0,
                "samples": len(widths),
                "left_line": left,
                "right_line": right,
            }

        width = float(np.median(np.asarray(widths, dtype=np.float32)))
        confidence = self._confidence(left, right, w, h, len(widths))
        return {
            "status": "ok",
            "width": width,
            "confidence": confidence,
            "samples": len(widths),
            "left_line": left,
            "right_line": right,
        }

    def _estimate_width_detector(self, image_chw, depth, intrinsics):
        image = _chw_to_hwc_uint8(image_chw)
        if image is None or depth is None or depth.size == 0:
            return {"status": "invalid_input", "confidence": 0.0}
        if not self._ensure_detector_loaded():
            return {"status": "detector_unavailable", "confidence": 0.0}

        probs = self._infer_detector(image)
        if probs is None:
            return {"status": "detector_failed", "confidence": 0.0}

        h, w = depth.shape[:2]
        if probs.shape[-2:] != (h, w):
            probs = np.stack(
                [
                    cv2.resize(probs[0], (w, h), interpolation=cv2.INTER_LINEAR),
                    cv2.resize(probs[1], (w, h), interpolation=cv2.INTER_LINEAR),
                ],
                axis=0,
            )

        left = probs[0]
        right = probs[1]
        y0 = int(np.clip(self.roi_y_min, 0.0, 1.0) * h)
        y1 = int(np.clip(self.roi_y_max, 0.0, 1.0) * h)
        y1 = max(y1, y0 + 1)
        left_roi = left[y0:y1]
        right_roi = right[y0:y1]
        left_conf = float(left_roi.max(initial=0.0))
        right_conf = float(right_roi.max(initial=0.0))
        if left_conf < self.detector_min_peak or right_conf < self.detector_min_peak:
            return {
                "status": "low_detector_conf",
                "confidence": min(left_conf, right_conf),
                "left_conf": left_conf,
                "right_conf": right_conf,
            }

        widths_3d = []
        widths_px = []
        centers_px = []
        points = []
        rows = []
        row_target_center = self.last_good_center if self.detector_row_target_tracking else None
        row_reference_center = None
        row_reference_centers = []
        row_center_limit = self._detector_row_center_limit(w)
        # Pick rows from near to far. The near rail pair is less ambiguous and
        # gives a better center prior for multi-track scenes.
        for yf in sorted([float(v) for v in self.sample_y_fracs], reverse=True):
            y = int(np.clip(float(yf), 0.0, 1.0) * h)
            row = {
                "sample_y_frac": float(yf),
                "sample_y": y,
                "selected": False,
                "reject_reason": "",
            }
            y_top = max(0, y - 2)
            y_bottom = min(h, y + 3)
            lp = _smooth_1d(left[y_top:y_bottom].mean(axis=0))
            rp = _smooth_1d(right[y_top:y_bottom].mean(axis=0))
            pair = self._select_detector_pair_at_row(
                lp, rp, w, target_center=row_target_center
            )
            if pair is None:
                row["reject_reason"] = "no_valid_pair"
                rows.append(row)
                continue
            lx, rx, lpeak, rpeak = pair
            width_px = float(rx - lx)
            center_px = 0.5 * (float(lx) + float(rx))
            row.update(
                {
                    "lx": lx,
                    "rx": rx,
                    "left_peak": lpeak,
                    "right_peak": rpeak,
                    "pixel_width": width_px,
                    "pixel_center": center_px,
                    "row_reference_center": row_reference_center,
                    "row_center_limit": row_center_limit,
                }
            )
            if (
                lpeak < self.detector_min_peak
                or rpeak < self.detector_min_peak
                or width_px < self.detector_min_width_px
            ):
                row["reject_reason"] = "weak_or_narrow_pair"
                rows.append(row)
                continue
            if (
                self.detector_row_consistency
                and row_reference_center is not None
                and row_center_limit is not None
            ):
                center_shift = abs(center_px - row_reference_center)
                row["row_center_shift"] = center_shift
                if center_shift > row_center_limit:
                    row["reject_reason"] = "row_center_inconsistent"
                    rows.append(row)
                    continue

            z_l = _sample_depth(depth, lx, y)
            z_r = _sample_depth(depth, rx, y)
            row.update({"depth_left": z_l, "depth_right": z_r})
            if z_l <= 0 or z_r <= 0:
                row["reject_reason"] = "invalid_depth"
                rows.append(row)
                continue
            p_l = _backproject(lx, y, z_l, intrinsics)
            p_r = _backproject(rx, y, z_r, intrinsics)
            width_3d = float(np.linalg.norm(p_r - p_l))
            row["width_3d"] = width_3d
            if not np.isfinite(width_3d) or width_3d <= 1e-6:
                row["reject_reason"] = "invalid_width_3d"
                rows.append(row)
                continue
            widths_3d.append(width_3d)
            widths_px.append(width_px)
            centers_px.append(center_px)
            points.append((lx, y, rx, y))
            row["selected"] = True
            rows.append(row)
            if self.detector_row_consistency:
                row_reference_centers.append(center_px)
                row_reference_center = float(
                    np.median(np.asarray(row_reference_centers, dtype=np.float32))
                )
            if self.detector_row_target_tracking:
                row_target_center = row_reference_center if row_reference_center is not None else center_px

        if len(widths_3d) < self.min_samples:
            status = "row_inconsistent" if any(
                r.get("reject_reason") == "row_center_inconsistent" for r in rows
            ) else "too_few_samples"
            return {
                "status": status,
                "confidence": min(left_conf, right_conf),
                "samples": len(widths_3d),
                "left_conf": left_conf,
                "right_conf": right_conf,
                "points": points,
                "rows": rows,
            }

        pixel_width_arr = np.asarray(widths_px, dtype=np.float32)
        pixel_center_arr = np.asarray(centers_px, dtype=np.float32)
        width_arr = np.asarray(widths_3d, dtype=np.float32)
        pixel_width = float(np.median(pixel_width_arr))
        pixel_mad = float(np.median(np.abs(pixel_width_arr - pixel_width)))
        pixel_center = float(np.median(pixel_center_arr))
        pixel_center_mad = float(np.median(np.abs(pixel_center_arr - pixel_center)))
        width = float(np.median(width_arr))
        width_mad = float(np.median(np.abs(width_arr - width)))
        selected_ratio = float(len(widths_3d) / max(len(self.sample_y_fracs), 1))
        row_scale_raw_median = None
        row_scale_raw_mad = None
        scale_target_for_gate = None
        if self.correction_mode == "metric_width" and self.metric_width_m > 0:
            scale_target_for_gate = self.metric_width_m
        elif self.ref_width is not None:
            scale_target_for_gate = self.ref_width
        if scale_target_for_gate is not None:
            row_scale_arr = scale_target_for_gate / np.maximum(width_arr.astype(np.float32), 1e-9)
            row_scale_raw_median = float(np.median(row_scale_arr))
            row_scale_raw_mad = float(np.median(np.abs(row_scale_arr - row_scale_raw_median)))
        detector_info = {
            "confidence": min(left_conf, right_conf),
            "width": width,
            "width_mad": width_mad,
            "row_scale_raw_median": row_scale_raw_median,
            "row_scale_raw_mad": row_scale_raw_mad,
            "selected_ratio": selected_ratio,
            "pixel_width": pixel_width,
            "pixel_width_mad": pixel_mad,
            "pixel_center": pixel_center,
            "pixel_center_mad": pixel_center_mad,
            "pixel_center_anchor": self.detector_center_anchor,
            "samples": len(widths_3d),
            "left_conf": left_conf,
            "right_conf": right_conf,
            "points": points,
            "rows": rows,
        }
        if self.last_good_width is not None:
            rel_jump = abs(pixel_width - self.last_good_width) / max(self.last_good_width, 1.0)
            if rel_jump > self.max_width_jump_ratio:
                detector_info["status"] = "width_jump"
                return detector_info
        if self.last_good_center is not None:
            center_jump = abs(pixel_center - self.last_good_center) / max(w, 1.0)
            if center_jump > self.detector_max_center_jump_ratio:
                detector_info["status"] = "center_jump"
                return detector_info
        anchor_limit = self._detector_anchor_shift_limit(w)
        if anchor_limit is not None:
            anchor_shift = abs(pixel_center - self.detector_center_anchor)
            if anchor_shift > anchor_limit:
                detector_info.update(
                    {
                        "status": "center_anchor_shift",
                        "pixel_center_anchor_shift": anchor_shift,
                        "pixel_center_anchor_limit": anchor_limit,
                    }
                )
                return detector_info
        self.last_good_width = pixel_width
        self.last_good_center = pixel_center
        self._update_detector_center_anchor(pixel_center)

        sample_score = np.clip(len(widths_3d) / max(len(self.sample_y_fracs), 1), 0.0, 1.0)
        mad_score = 1.0 - np.clip(pixel_mad / max(pixel_width, 1.0), 0.0, 1.0)
        peak_score = min(left_conf, right_conf)
        confidence = float(0.45 * peak_score + 0.35 * sample_score + 0.20 * mad_score)
        detector_info["status"] = "ok"
        detector_info["confidence"] = confidence
        detector_info["pixel_center_anchor"] = self.detector_center_anchor
        return detector_info

    def _select_detector_pair_at_row(self, left_profile, right_profile, width, target_center=None):
        left_peaks = _top_profile_peaks(
            left_profile,
            threshold=self.detector_min_peak,
            topk=self.detector_peak_topk,
            nms_px=self.detector_peak_nms_px,
        )
        right_peaks = _top_profile_peaks(
            right_profile,
            threshold=self.detector_min_peak,
            topk=self.detector_peak_topk,
            nms_px=self.detector_peak_nms_px,
        )
        if not left_peaks or not right_peaks:
            return None

        if target_center is None:
            if self._detector_anchor_shift_limit(width) is not None:
                target_center = self.detector_center_anchor
        if target_center is None:
            target_center = self.last_good_center
        if target_center is None:
            target_center = float(width) * self.detector_center_x_frac
        width_prior = self.last_good_width
        best = None
        max_width = float(width) * self.detector_max_width_px_frac

        for lx, lpeak in left_peaks:
            for rx, rpeak in right_peaks:
                if rx <= lx:
                    continue
                pair_width = float(rx - lx)
                if pair_width < self.detector_min_width_px or pair_width > max_width:
                    continue
                center = 0.5 * (float(lx) + float(rx))
                center_err = abs(center - target_center) / max(float(width), 1.0)
                straddle_bonus = 0.25 if lx <= target_center <= rx else 0.0
                width_penalty = 0.0
                if width_prior is not None:
                    width_penalty = abs(pair_width - width_prior) / max(width_prior, 1.0)
                score = (
                    float(lpeak) + float(rpeak)
                    + straddle_bonus
                    - 1.8 * center_err
                    - 0.7 * width_penalty
                )
                if best is None or score > best[0]:
                    best = (score, int(lx), int(rx), float(lpeak), float(rpeak))

        if best is None:
            return None
        return best[1], best[2], best[3], best[4]

    def _detector_row_center_limit(self, image_width):
        limits = []
        if self.detector_max_row_center_shift_px > 0:
            limits.append(float(self.detector_max_row_center_shift_px))
        if self.detector_max_row_center_shift_ratio > 0:
            limits.append(float(self.detector_max_row_center_shift_ratio) * max(float(image_width), 1.0))
        if not limits:
            return None
        return min(limits)

    def _detector_anchor_shift_limit(self, image_width):
        if self.detector_center_anchor is None:
            return None
        limits = []
        if self.detector_max_anchor_center_shift_px > 0:
            limits.append(float(self.detector_max_anchor_center_shift_px))
        if self.detector_max_anchor_center_shift_ratio > 0:
            limits.append(float(self.detector_max_anchor_center_shift_ratio) * max(float(image_width), 1.0))
        if not limits:
            return None
        return min(limits)

    def _update_detector_center_anchor(self, pixel_center):
        if self.detector_center_anchor_frames <= 0 or self.detector_center_anchor is not None:
            return
        self.detector_center_anchor_history.append(float(pixel_center))
        if len(self.detector_center_anchor_history) >= self.detector_center_anchor_frames:
            recent = self.detector_center_anchor_history[-self.detector_center_anchor_frames:]
            self.detector_center_anchor = float(np.median(np.asarray(recent, dtype=np.float32)))

    def _should_hold_last_scale(self, status):
        return (
            self.hold_last_scale_on_reject
            and self.apply_correction
            and self.scale_ema_initialized
            and status in self.hold_last_scale_statuses
        )

    def _ensure_detector_loaded(self):
        if self.detector_loaded:
            return self.detector_model is not None
        self.detector_loaded = True
        if not self.detector_checkpoint:
            return False
        try:
            import torch

            from utils.rail_detector_model import build_segformer_rail_detector
        except Exception:
            return False

        if self.detector_device == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            device = torch.device(self.detector_device)

        try:
            try:
                checkpoint = torch.load(self.detector_checkpoint, map_location=device, weights_only=False)
            except TypeError:
                checkpoint = torch.load(self.detector_checkpoint, map_location=device)
            ckpt_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
            model_name = self.detector_model_name or ckpt_args.get("model_name") or "nvidia/segformer-b0-finetuned-ade-512-512"
            self.detector_image_size = tuple(ckpt_args.get("image_size") or self.detector_image_size)
            model = build_segformer_rail_detector(
                model_name=model_name,
                num_channels=2,
                pretrained=False,
            ).to(device)
            state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
            model.load_state_dict(state, strict=True)
            model.eval()
        except Exception:
            self.detector_model = None
            return False

        self.detector_torch = torch
        self.detector_device_obj = device
        self.detector_model = model
        return True

    def _infer_detector(self, image_rgb):
        torch = self.detector_torch
        out_w, out_h = [int(v) for v in self.detector_image_size]
        resized = cv2.resize(image_rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
        image = resized.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(self.detector_device_obj)
        with torch.no_grad():
            logits = self.detector_model(pixel_values=tensor).logits
            logits = torch.nn.functional.interpolate(
                logits,
                size=(out_h, out_w),
                mode="bilinear",
                align_corners=False,
            )
            probs = torch.sigmoid(logits)[0].detach().cpu().numpy()
        return probs.astype(np.float32)

    def _limit_scale_step(self, raw_scale):
        if self.max_scale_step <= 0:
            return raw_scale
        lo = self.scale_ema - self.max_scale_step
        hi = self.scale_ema + self.max_scale_step
        return float(np.clip(raw_scale, lo, hi))

    def _select_line_pair(self, lines, det_w, det_h, y_offset, resize_scale):
        left_candidates = []
        right_candidates = []
        center_x = det_w * 0.5
        eval_y = int(det_h * (0.5 * (self.roi_y_min + self.roi_y_max)))

        for x1, y1, x2, y2 in lines:
            y1 += y_offset
            y2 += y_offset
            dy = y2 - y1
            dx = x2 - x1
            if abs(dy) < 20 or abs(dx) < 5:
                continue
            slope = dx / float(dy)
            length = float(np.hypot(dx, dy))
            if abs(slope) < 0.05:
                continue
            line = _scale_line((x1, y1, x2, y2), resize_scale)
            x_eval = _line_x_at_y(line, int(eval_y / max(resize_scale, 1e-6)))
            if x_eval is None:
                continue
            score = length * min(abs(dy) / max(det_w, 1), 1.0)
            if slope < 0 and x_eval < center_x / max(resize_scale, 1e-6):
                left_candidates.append((score, line))
            elif slope > 0 and x_eval > center_x / max(resize_scale, 1e-6):
                right_candidates.append((score, line))

        best = None
        for _, left in sorted(left_candidates, reverse=True)[:10]:
            for _, right in sorted(right_candidates, reverse=True)[:10]:
                y_bottom = max(left[1], left[3], right[1], right[3])
                y_top = min(left[1], left[3], right[1], right[3])
                xl_b = _line_x_at_y(left, y_bottom)
                xr_b = _line_x_at_y(right, y_bottom)
                xl_t = _line_x_at_y(left, y_top)
                xr_t = _line_x_at_y(right, y_top)
                if None in (xl_b, xr_b, xl_t, xr_t):
                    continue
                bottom_width = xr_b - xl_b
                top_width = xr_t - xl_t
                if bottom_width <= 40 or top_width <= 5 or bottom_width <= top_width:
                    continue
                score = bottom_width + 0.5 * (bottom_width - top_width)
                if best is None or score > best[0]:
                    best = (score, left, right)
        if best is None:
            return None, None
        return best[1], best[2]

    def _confidence(self, left, right, w, h, n_samples):
        y_bottom = int(h * self.roi_y_max)
        y_top = int(h * self.roi_y_min)
        xl_b = _line_x_at_y(left, y_bottom)
        xr_b = _line_x_at_y(right, y_bottom)
        xl_t = _line_x_at_y(left, y_top)
        xr_t = _line_x_at_y(right, y_top)
        if None in (xl_b, xr_b, xl_t, xr_t):
            return 0.0
        bottom_width = max(xr_b - xl_b, 0.0)
        top_width = max(xr_t - xl_t, 0.0)
        width_score = np.clip(bottom_width / max(0.25 * w, 1.0), 0.0, 1.0)
        converge_score = np.clip((bottom_width - top_width) / max(bottom_width, 1.0), 0.0, 1.0)
        sample_score = np.clip(n_samples / max(len(self.sample_y_fracs), 1), 0.0, 1.0)
        return float(0.45 * width_score + 0.35 * converge_score + 0.20 * sample_score)

    def _log(self, info):
        if not self.log_diag or self.diag_path is None:
            return
        write_header = not os.path.exists(self.diag_path)
        with open(self.diag_path, "a", newline="") as f:
            fields = [
                "frame_id", "status", "width", "ref_width", "scale_target_width",
                "raw_scale", "scale", "applied_scale", "apply_correction",
                "correction_mode", "metric_width_m", "confidence",
                "samples", "mode", "selected_ratio", "width_mad",
                "row_scale_raw_median", "row_scale_raw_mad",
                "pixel_width", "pixel_width_mad",
                "pixel_center", "pixel_center_mad", "pixel_center_anchor",
                "pixel_center_anchor_shift", "pixel_center_anchor_limit",
                "left_conf", "right_conf",
                "left_line", "right_line", "points",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerow({k: info.get(k, "") for k in fields})

    def _log_rows(self, info):
        if not self.log_diag or not self.log_row_diag or self.row_diag_path is None:
            return
        rows = info.get("rows") or []
        if not rows:
            return

        write_header = not os.path.exists(self.row_diag_path)
        fields = [
            "frame_id", "frame_status", "mode", "apply_correction",
            "correction_mode", "metric_width_m", "ref_width", "scale_target_width",
            "frame_width", "frame_raw_scale", "frame_scale", "frame_applied_scale",
            "sample_y_frac", "sample_y", "selected", "reject_reason", "lx", "rx",
            "pixel_width", "pixel_center", "row_reference_center",
            "row_center_shift", "row_center_limit", "left_peak", "right_peak",
            "depth_left", "depth_right", "width_3d", "row_scale_raw",
            "row_scale_clipped",
        ]
        metric_fields = [
            _metric_scale_field_name(width_m)
            for width_m in self.metric_width_candidates_m
        ]
        fields.extend(metric_fields)
        with open(self.row_diag_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                writer.writeheader()
            for row in rows:
                width_3d = row.get("width_3d")
                row_scale_raw = ""
                row_scale_clipped = ""
                scale_target_width = info.get("scale_target_width")
                if scale_target_width is not None and width_3d not in (None, ""):
                    try:
                        row_scale_raw = float(scale_target_width) / max(float(width_3d), 1e-9)
                        row_scale_clipped = float(np.clip(row_scale_raw, self.min_scale, self.max_scale))
                    except Exception:
                        row_scale_raw = ""
                        row_scale_clipped = ""
                metric_scales = {field: "" for field in metric_fields}
                if width_3d not in (None, ""):
                    try:
                        width_3d_float = float(width_3d)
                        if np.isfinite(width_3d_float) and width_3d_float > 1e-9:
                            for width_m, field in zip(self.metric_width_candidates_m, metric_fields):
                                metric_scales[field] = float(width_m) / width_3d_float
                    except Exception:
                        metric_scales = {field: "" for field in metric_fields}
                out = {
                    "frame_id": info.get("frame_id"),
                    "frame_status": info.get("status"),
                    "mode": info.get("mode"),
                    "apply_correction": info.get("apply_correction"),
                    "correction_mode": info.get("correction_mode"),
                    "metric_width_m": info.get("metric_width_m"),
                    "ref_width": info.get("ref_width"),
                    "scale_target_width": info.get("scale_target_width"),
                    "frame_width": info.get("width"),
                    "frame_raw_scale": info.get("raw_scale"),
                    "frame_scale": info.get("scale"),
                    "frame_applied_scale": info.get("applied_scale"),
                    "sample_y_frac": row.get("sample_y_frac"),
                    "sample_y": row.get("sample_y"),
                    "selected": row.get("selected"),
                    "reject_reason": row.get("reject_reason"),
                    "lx": row.get("lx"),
                    "rx": row.get("rx"),
                    "pixel_width": row.get("pixel_width"),
                    "pixel_center": row.get("pixel_center"),
                    "row_reference_center": row.get("row_reference_center"),
                    "row_center_shift": row.get("row_center_shift"),
                    "row_center_limit": row.get("row_center_limit"),
                    "left_peak": row.get("left_peak"),
                    "right_peak": row.get("right_peak"),
                    "depth_left": row.get("depth_left"),
                    "depth_right": row.get("depth_right"),
                    "width_3d": width_3d,
                    "row_scale_raw": row_scale_raw,
                    "row_scale_clipped": row_scale_clipped,
                }
                out.update(metric_scales)
                writer.writerow(out)

    def _save_debug(self, frame_idx, image_chw, info):
        if not self.debug_dir:
            return
        image = _chw_to_hwc_uint8(image_chw)
        if image is None:
            return
        vis = image.copy()
        for key, color in (("left_line", (255, 0, 0)), ("right_line", (0, 255, 0))):
            line = info.get(key)
            if line is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in line]
            cv2.line(vis, (x1, y1), (x2, y2), color, 4)
        points = info.get("points") or []
        for lx, y1, rx, y2 in points:
            cv2.circle(vis, (int(lx), int(y1)), 5, (255, 255, 255), -1)
            cv2.circle(vis, (int(rx), int(y2)), 5, (255, 255, 255), -1)
            cv2.line(vis, (int(lx), int(y1)), (int(rx), int(y2)), (255, 255, 255), 2)
        cv2.imwrite(
            os.path.join(self.debug_dir, f"rail_scale_{frame_idx:05d}.png"),
            cv2.cvtColor(vis, cv2.COLOR_RGB2BGR),
        )


def _chw_to_hwc_uint8(image_chw):
    if image_chw is None:
        return None
    if hasattr(image_chw, "detach"):
        image_chw = image_chw.detach()
        if hasattr(image_chw, "cpu"):
            image_chw = image_chw.cpu()
        image = image_chw.numpy()
    else:
        image = np.asarray(image_chw)
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = image.transpose(1, 2, 0)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _scale_line(line, resize_scale):
    s = max(float(resize_scale), 1e-6)
    return tuple(float(v) / s for v in line)


def _line_x_at_y(line, y):
    x1, y1, x2, y2 = line
    if abs(y2 - y1) < 1e-6:
        return None
    t = (float(y) - y1) / (y2 - y1)
    return float(x1 + t * (x2 - x1))


def _sample_depth(depth, x, y):
    h, w = depth.shape[:2]
    xi = int(np.clip(round(x), 0, w - 1))
    yi = int(np.clip(round(y), 0, h - 1))
    return float(depth[yi, xi])


def _finite_float_or_none(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _positive_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(value) or value <= 0:
        return default
    return value


def _positive_float_list(values):
    out = []
    if values in (None, ""):
        return out
    if isinstance(values, (int, float, str)):
        values = [values]
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value > 0:
            out.append(value)
    return out


def _metric_scale_field_name(width_m):
    return f"metric_scale_m_{float(width_m):.3f}".replace(".", "_")


def _backproject(x, y, z, intrinsics):
    k = np.asarray(intrinsics, dtype=np.float64)
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[0, 2], k[1, 2]
    return np.asarray([(float(x) - cx) / fx * z, (float(y) - cy) / fy * z, z], dtype=np.float64)


def _smooth_1d(values, kernel=9):
    values = np.asarray(values, dtype=np.float32)
    if kernel <= 1 or values.size == 0:
        return values
    kernel = int(kernel)
    if kernel % 2 == 0:
        kernel += 1
    pad = kernel // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    filt = np.ones(kernel, dtype=np.float32) / kernel
    return np.convolve(padded, filt, mode="valid")


def _top_profile_peaks(profile, threshold=0.25, topk=8, nms_px=24):
    values = np.asarray(profile, dtype=np.float32)
    if values.size == 0:
        return []

    order = np.argsort(values)[::-1]
    peaks = []
    for idx in order:
        score = float(values[idx])
        if score < threshold:
            break
        if any(abs(int(idx) - int(prev_idx)) < nms_px for prev_idx, _ in peaks):
            continue
        peaks.append((int(idx), score))
        if len(peaks) >= topk:
            break
    return peaks
