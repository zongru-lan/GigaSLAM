#!/usr/bin/env python3
"""Analyze how rendering/GT comparison resolution affects PSNR/SSIM/LPIPS.

This script is intentionally evaluation-only. It uses saved rendered keyframe
PNGs and their original input images, then recomputes metrics at multiple
comparison resolutions by resizing both images to the same target width.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity




def psnr_metric(img1, img2):
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True).clamp_min(1e-12)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def _gaussian(window_size, sigma):
    vals = [np.exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2)) for x in range(window_size)]
    gauss = torch.tensor(vals, dtype=torch.float32)
    return gauss / gauss.sum()


def _create_window(window_size, channel, device, dtype):
    w1 = _gaussian(window_size, 1.5).to(device=device, dtype=dtype).unsqueeze(1)
    w2 = w1.mm(w1.t()).unsqueeze(0).unsqueeze(0)
    return w2.expand(channel, 1, window_size, window_size).contiguous()


def ssim_metric(img1, img2, window_size=11):
    channel = img1.size(-3)
    window = _create_window(window_size, channel, img1.device, img1.dtype)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()

def load_rgb_tensor(path):
    arr = np.array(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def resize_chw(image, target_width, mode):
    if target_width <= 0:
        return image
    _, h, w = image.shape
    if w == target_width:
        return image
    target_height = int(round(h * (target_width / float(w))))
    kwargs = {'mode': mode}
    if mode in {'bilinear', 'bicubic'}:
        kwargs['align_corners'] = False
    return F.interpolate(image.unsqueeze(0), size=(target_height, target_width), **kwargs).squeeze(0)


def metric_one(pred_chw, gt_chw, lpips_model, device):
    pred = pred_chw.clamp(0.0, 1.0).to(device)
    gt = gt_chw.clamp(0.0, 1.0).to(device)
    pred_hwc = pred.permute(1, 2, 0)
    gt_hwc = gt.permute(1, 2, 0)
    mask = gt_hwc > 0
    psnr_score = psnr_metric(pred_hwc[mask].unsqueeze(0), gt_hwc[mask].unsqueeze(0)).item()
    ssim_score = ssim_metric(pred_hwc.permute(2, 0, 1).unsqueeze(0), gt_hwc.permute(2, 0, 1).unsqueeze(0)).item()
    lpips_score = lpips_model(pred.unsqueeze(0), gt.unsqueeze(0)).item()
    return psnr_score, ssim_score, lpips_score


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    return {
        'mean': float(values.mean()),
        'median': float(np.median(values)),
        'p05': float(np.percentile(values, 5)),
        'p95': float(np.percentile(values, 95)),
        'min': float(values.min()),
        'max': float(values.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', required=True, help='Run directory containing img/rendered_keyframes.csv')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--widths', nargs='+', type=int, default=[0, 1280, 960, 640], help='0 means original saved resolution')
    parser.add_argument('--resize-mode', default='bicubic', choices=['nearest', 'bilinear', 'bicubic', 'area'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    manifest = result_dir / 'img' / 'rendered_keyframes.csv'
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(manifest.open(newline='', encoding='utf-8')))
    if args.limit > 0:
        rows = rows[:args.limit]

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    lpips_model = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)
    lpips_model.eval()

    metric_rows = []
    for i, row in enumerate(rows):
        frame_idx = int(row['frame_idx'])
        render_path = result_dir / 'img' / row['rendered_image']
        input_path = Path(row['input_image'])
        if not input_path.is_absolute():
            # rendered_keyframes.csv normally stores just a basename. Resolve it
            # from the run config's Dataset.color_path if possible.
            cfg_path = result_dir / 'config.yml'
            import yaml
            cfg = yaml.safe_load(cfg_path.read_text())
            input_path = Path(cfg['Dataset']['color_path']) / row['input_image']
        pred0 = load_rgb_tensor(render_path)
        gt0 = load_rgb_tensor(input_path)
        if pred0.shape[-2:] != gt0.shape[-2:]:
            gt0 = F.interpolate(gt0.unsqueeze(0), size=pred0.shape[-2:], mode='bicubic', align_corners=False).squeeze(0)

        for width in args.widths:
            pred = resize_chw(pred0, width, args.resize_mode)
            gt = resize_chw(gt0, width, args.resize_mode)
            with torch.no_grad():
                p, s, l = metric_one(pred, gt, lpips_model, device)
            metric_rows.append({
                'frame_idx': frame_idx,
                'rendered_image': row['rendered_image'],
                'input_image': str(input_path),
                'comparison_width': int(width if width > 0 else pred0.shape[-1]),
                'comparison_height': int(pred.shape[-2]),
                'resolution_label': 'saved_original' if width <= 0 else f'w{width}',
                'psnr': p,
                'ssim': s,
                'lpips': l,
            })
        if (i + 1) % 10 == 0:
            print(f'[resolution-effect] processed {i + 1}/{len(rows)} keyframes')

    csv_path = output_dir / 'resolution_metrics_by_keyframe.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)

    summary = {
        'result_dir': str(result_dir),
        'num_keyframes': len(rows),
        'resize_mode': args.resize_mode,
        'widths': args.widths,
        'metrics_by_resolution': {},
    }
    for label in sorted({r['resolution_label'] for r in metric_rows}):
        subset = [r for r in metric_rows if r['resolution_label'] == label]
        summary['metrics_by_resolution'][label] = {
            'comparison_width': int(subset[0]['comparison_width']),
            'comparison_height': int(subset[0]['comparison_height']),
            'psnr': summarize([r['psnr'] for r in subset]),
            'ssim': summarize([r['ssim'] for r in subset]),
            'lpips': summarize([r['lpips'] for r in subset]),
        }

    json_path = output_dir / 'resolution_metrics_summary.json'
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    md_lines = [
        '# Rendering Resolution Effect Diagnostics',
        '',
        f'- result_dir: `{result_dir}`',
        f'- keyframes: `{len(rows)}`',
        f'- resize_mode: `{args.resize_mode}`',
        '',
        '| resolution | PSNR mean | SSIM mean | LPIPS mean | LPIPS p95 |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    for label, item in sorted(summary['metrics_by_resolution'].items(), key=lambda kv: kv[1]['comparison_width'], reverse=True):
        md_lines.append(
            f"| {label} ({item['comparison_width']}x{item['comparison_height']}) | "
            f"{item['psnr']['mean']:.4f} | {item['ssim']['mean']:.4f} | "
            f"{item['lpips']['mean']:.4f} | {item['lpips']['p95']:.4f} |"
        )
    md_lines.extend([
        '',
        '## 解释',
        '',
        '`saved_original` 是当前保存的上采样渲染图与原始 GT 的比较口径；较低宽度是把渲染图和 GT 同时下采样后再比较。',
        '如果 LPIPS 在低分辨率显著下降，说明当前 4K 比较口径确实放大了铁路高频纹理/上采样差异。',
    ])
    (output_dir / 'README.md').write_text('\n'.join(md_lines) + '\n', encoding='utf-8')
    print(f'[resolution-effect] wrote {csv_path}')
    print(f'[resolution-effect] wrote {json_path}')


if __name__ == '__main__':
    main()
