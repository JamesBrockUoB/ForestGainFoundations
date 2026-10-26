"""
Threshold and threshold-free metrics where every pixel counts with a weight.

weights = 1 everywhere  -> ordinary F1 / IoU / precision / recall / AP (what torchmetrics gave before).
weights = confidence    -> the 'soft' evaluation: a gain pixel counts as much as its confidence (0.5 ... 1.0), so missing a
                           95 %-confidence gain pixel costs more than missing a 55 % one. Background pixels count 1.
The weights are exactly the ones the soft loss uses (losses.pixel_weights).
"""

import torch


def _div(a, b):
    return float(a / b) if b > 0 else 0.0


def weighted_average_precision(p, t, w):
    """Step-wise average precision (same definition as sklearn / torchmetrics) with per-pixel weights."""
    p, t, w = p.double(), t.double(), w.double()
    total_pos = (t * w).sum()
    if total_pos <= 0:
        return 0.0
    order = torch.argsort(p, descending=True)
    p, t, w = p[order], t[order], w[order]
    tp = torch.cumsum(t * w, 0)
    fp = torch.cumsum((1 - t) * w, 0)
    last = torch.cat(
        [p[1:] != p[:-1], torch.ones(1, dtype=torch.bool)]
    )  # one point per distinct score
    tp, fp = tp[last], fp[last]
    precision = tp / (tp + fp).clamp(min=1e-12)
    recall = tp / tp[-1]
    prev = torch.cat([torch.zeros(1, dtype=recall.dtype), recall[:-1]])
    return float(((recall - prev) * precision).sum())


def summarise(samples, threshold, weighted):
    """samples: list of (probs, targets, weights) 1-D CPU tensors, one per tile (valid pixels only).
    Returns f1, iou, precision, recall (pooled over all tiles, probs > threshold), ap (pooled), map (mean of per-tile AP over
    tiles that contain gain) and prevalence (weighted share of gain = the AP of a random scorer).
    """
    keys = ("f1", "iou", "precision", "recall", "ap", "map", "prevalence")
    if not samples:
        return dict.fromkeys(keys, 0.0)
    p = torch.cat([s[0] for s in samples]).double()
    t = torch.cat([s[1] for s in samples]).double()
    w = torch.cat([s[2] for s in samples]).double() if weighted else torch.ones_like(p)
    pred = (p > threshold).double()
    tp = (w * t * pred).sum()
    fp = (w * (1 - t) * pred).sum()
    fn = (w * t * (1 - pred)).sum()
    tile_ap = [
        weighted_average_precision(sp, st, sw if weighted else torch.ones_like(sp))
        for sp, st, sw in samples
        if st.any()
    ]
    return {
        "f1": _div(2 * tp, 2 * tp + fp + fn),
        "iou": _div(tp, tp + fp + fn),
        "precision": _div(tp, tp + fp),
        "recall": _div(tp, tp + fn),
        "ap": weighted_average_precision(p, t, w),
        "map": sum(tile_ap) / len(tile_ap) if tile_ap else 0.0,
        "prevalence": _div((w * t).sum(), w.sum()),
    }
