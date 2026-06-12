"""Optional shadow diagnostics for a TEP-style ego-path model."""

import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class EgoPathGuidance:
    """Run a trained TEP regression detector in shadow mode.

    This class is intentionally diagnostic-first. It logs the predicted ego-path
    and compares it with the rail pair selected by RailScale, but it does not
    change RailScale candidate selection in v1.
    """

    def __init__(self, config, save_dir=None):
        self.enabled = bool(config.get("enabled", False))
        self.model_path = str(config.get("model_path", ""))
        self.tep_root = str(config.get("tep_root", "train-ego-path-detection"))
        self.device = str(config.get("device", config.get("detector_device", "cuda")))
        self.runtime = str(config.get("runtime", "pytorch"))
        self.crop = config.get("crop", "none")
        self.log_diag = bool(config.get("log_diag", True))
        self.save_debug = bool(config.get("save_debug", False))
        self.debug_every = int(config.get("debug_every", 20))
        self.sample_y_fracs = [float(v) for v in config.get("sample_y_fracs", [0.58, 0.65, 0.72, 0.80, 0.88])]

        self.detector = None
        self.load_attempted = False
        self.save_dir = save_dir
        self.diag_path = os.path.join(save_dir, "ego_path_diag.csv") if save_dir else None
        self.debug_dir = os.path.join(save_dir, "ego_path_debug") if save_dir else None
        if self.save_debug and self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)

    def observe(self, frame_idx, image_chw, rail_info=None):
        if not self.enabled:
            return {"status": "disabled"}
        image = _chw_to_hwc_uint8(image_chw)
        if image is None:
            out = {"frame_id": frame_idx, "status": "invalid_image"}
            self._log(out)
            return out
        if not self._ensure_loaded():
            out = {"frame_id": frame_idx, "status": "model_unavailable"}
            self._log(out)
            return out

        try:
            rails = self.detector.detect(Image.fromarray(image))
        except Exception as exc:
            out = {"frame_id": frame_idx, "status": "infer_failed", "error": str(exc)}
            self._log(out)
            return out

        out = self._summarize(frame_idx, rails, image.shape[:2], rail_info or {})
        self._log(out)
        if self.save_debug and frame_idx % max(self.debug_every, 1) == 0:
            self._save_debug(frame_idx, image, rails, rail_info or {}, out)
        return out

    def _ensure_loaded(self):
        if self.load_attempted:
            return self.detector is not None
        self.load_attempted = True
        if not self.model_path:
            return False
        try:
            tep_root = str(Path(self.tep_root).resolve())
            if tep_root not in sys.path:
                sys.path.insert(0, tep_root)
            from src.utils.interface import Detector
        except Exception:
            return False
        try:
            self.detector = Detector(
                model_path=self.model_path,
                crop_coords=self._crop_coords(),
                runtime=self.runtime,
                device=self.device,
            )
        except Exception:
            self.detector = None
        return self.detector is not None

    def _crop_coords(self):
        if self.crop in (None, "none", "None"):
            return None
        if self.crop == "auto":
            return "auto"
        if isinstance(self.crop, (list, tuple)) and len(self.crop) == 4:
            return tuple(int(v) for v in self.crop)
        if isinstance(self.crop, str):
            parts = [p.strip() for p in self.crop.split(",")]
            if len(parts) == 4:
                return tuple(int(v) for v in parts)
        return None

    def _summarize(self, frame_idx, rails, image_hw, rail_info):
        h, w = image_hw
        left, right = _rails_to_arrays(rails)
        out = {
            "frame_id": frame_idx,
            "status": "ok" if left is not None and right is not None else "empty_prediction",
            "ego_samples": 0,
            "ego_center_median": "",
            "ego_width_median": "",
            "rail_center_median": "",
            "rail_width_median": "",
            "center_delta_median": "",
            "width_delta_median": "",
            "rail_status": rail_info.get("status", ""),
            "model_path": self.model_path,
            "crop": self.crop,
        }
        if left is None or right is None:
            return out

        ego_centers = []
        ego_widths = []
        for yf in self.sample_y_fracs:
            y = float(yf) * float(max(h - 1, 1))
            lx = _x_at_y(left, y)
            rx = _x_at_y(right, y)
            if lx is None or rx is None or rx <= lx:
                continue
            ego_centers.append(0.5 * (lx + rx))
            ego_widths.append(rx - lx)
        if ego_centers:
            out["ego_samples"] = len(ego_centers)
            out["ego_center_median"] = float(np.median(np.asarray(ego_centers, dtype=np.float32)))
            out["ego_width_median"] = float(np.median(np.asarray(ego_widths, dtype=np.float32)))

        rail_centers = []
        rail_widths = []
        for row in rail_info.get("rows") or []:
            if not row.get("selected"):
                continue
            if row.get("pixel_center") in (None, "") or row.get("pixel_width") in (None, ""):
                continue
            rail_centers.append(float(row["pixel_center"]))
            rail_widths.append(float(row["pixel_width"]))
        if rail_centers:
            rail_center = float(np.median(np.asarray(rail_centers, dtype=np.float32)))
            rail_width = float(np.median(np.asarray(rail_widths, dtype=np.float32)))
            out["rail_center_median"] = rail_center
            out["rail_width_median"] = rail_width
            if ego_centers:
                out["center_delta_median"] = float(out["ego_center_median"] - rail_center)
                out["width_delta_median"] = float(out["ego_width_median"] - rail_width)
        return out

    def _log(self, info):
        if not self.log_diag or self.diag_path is None:
            return
        fields = [
            "frame_id", "status", "ego_samples", "ego_center_median", "ego_width_median",
            "rail_center_median", "rail_width_median", "center_delta_median",
            "width_delta_median", "rail_status", "model_path", "crop", "error",
        ]
        write_header = not os.path.exists(self.diag_path)
        with open(self.diag_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerow({key: info.get(key, "") for key in fields})

    def _save_debug(self, frame_idx, image, rails, rail_info, summary):
        if not self.debug_dir:
            return
        vis = image.copy()
        left, right = _rails_to_arrays(rails)
        if left is not None:
            cv2.polylines(vis, [np.round(left).astype(np.int32)], False, (255, 0, 0), 5, lineType=cv2.LINE_AA)
        if right is not None:
            cv2.polylines(vis, [np.round(right).astype(np.int32)], False, (255, 255, 0), 5, lineType=cv2.LINE_AA)
        for lx, y1, rx, y2 in rail_info.get("points") or []:
            cv2.line(vis, (int(lx), int(y1)), (int(rx), int(y2)), (255, 255, 255), 2)
            cv2.circle(vis, (int(lx), int(y1)), 4, (255, 255, 255), -1)
            cv2.circle(vis, (int(rx), int(y2)), 4, (255, 255, 255), -1)
        text = f"EgoPath {summary.get('status')} dcenter={summary.get('center_delta_median', '')}"
        cv2.putText(vis, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(vis, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imwrite(
            os.path.join(self.debug_dir, f"ego_path_{frame_idx:05d}.png"),
            cv2.cvtColor(vis, cv2.COLOR_RGB2BGR),
        )


def _rails_to_arrays(rails):
    try:
        if rails is None or len(rails) != 2:
            return None, None
        left = np.asarray(rails[0], dtype=np.float32)
        right = np.asarray(rails[1], dtype=np.float32)
        if left.ndim != 2 or right.ndim != 2 or left.shape[1] != 2 or right.shape[1] != 2:
            return None, None
        if left.shape[0] < 2 or right.shape[0] < 2:
            return None, None
        return left, right
    except Exception:
        return None, None


def _x_at_y(points, y):
    y = float(y)
    xs = []
    for p0, p1 in zip(points[:-1], points[1:]):
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])
        if abs(y1 - y0) < 1e-6:
            continue
        if min(y0, y1) - 1e-6 <= y <= max(y0, y1) + 1e-6:
            xs.append(x0 + (y - y0) * (x1 - x0) / (y1 - y0))
    if not xs:
        return None
    return float(np.median(np.asarray(xs, dtype=np.float32)))


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
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image
