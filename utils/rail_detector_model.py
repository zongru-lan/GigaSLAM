import torch
import torch.nn.functional as F


def build_segformer_rail_detector(
    model_name="nvidia/segformer-b0-finetuned-ade-512-512",
    num_channels=2,
    pretrained=True,
):
    """Build a SegFormer rail-edge detector.

    The model predicts two heatmap channels: left rail edge and right rail edge.
    We use the segmentation head as a dense heatmap head and train with
    BCE/Dice outside the HuggingFace built-in CE loss.
    """
    try:
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
    except Exception as exc:
        raise ImportError(
            "SegFormer rail detector requires transformers. Install on the cloud "
            "environment with: pip install transformers accelerate"
        ) from exc

    if pretrained:
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_channels,
            ignore_mismatched_sizes=True,
        )
    else:
        config = SegformerConfig.from_pretrained(model_name)
        config.num_labels = num_channels
        model = SegformerForSemanticSegmentation(config)
    return model


def rail_heatmap_loss(logits, targets, valid_mask=None, dice_weight=1.0):
    """BCEWithLogits + soft Dice for 2-channel rail heatmaps."""
    if logits.shape[-2:] != targets.shape[-2:]:
        logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)

    if valid_mask is None:
        valid_mask = torch.ones_like(targets[:, :1])
    if valid_mask.shape[-2:] != targets.shape[-2:]:
        valid_mask = F.interpolate(valid_mask, size=targets.shape[-2:], mode="nearest")
    valid = valid_mask.expand_as(targets)

    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp_min(1.0)

    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)
    intersection = (probs * targets * valid).sum(dims)
    denom = ((probs * valid).sum(dims) + (targets * valid).sum(dims)).clamp_min(1e-6)
    dice = 1.0 - (2.0 * intersection / denom).mean()
    return bce + dice_weight * dice, {"bce": float(bce.detach().cpu()), "dice": float(dice.detach().cpu())}


@torch.no_grad()
def rail_heatmap_metrics(logits, targets, valid_mask=None, threshold=0.3):
    if logits.shape[-2:] != targets.shape[-2:]:
        logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
    if valid_mask is None:
        valid_mask = torch.ones_like(targets[:, :1])
    if valid_mask.shape[-2:] != targets.shape[-2:]:
        valid_mask = F.interpolate(valid_mask, size=targets.shape[-2:], mode="nearest")
    valid = valid_mask.expand_as(targets) > 0.5

    pred = torch.sigmoid(logits) > threshold
    gt = targets > threshold
    inter = (pred & gt & valid).sum(dim=(0, 2, 3)).float()
    union = ((pred | gt) & valid).sum(dim=(0, 2, 3)).float().clamp_min(1.0)
    iou = inter / union
    return {
        "iou_left": float(iou[0].detach().cpu()) if iou.numel() > 0 else 0.0,
        "iou_right": float(iou[1].detach().cpu()) if iou.numel() > 1 else 0.0,
        "iou_mean": float(iou.mean().detach().cpu()) if iou.numel() else 0.0,
    }
