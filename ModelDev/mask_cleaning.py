"""Label cleaning for the Forest Gain masks, used by datasets.py (training).

Labels are a binary ``mask`` (confidence > 0), a ``weight`` (confidence / 100, only used by the soft loss) and a
``valid`` map (pixels that count in loss and metrics). ``clean_labels`` applies, in this fixed order:

  1. confidence step, threshold T on the weight (conf_filter OR conf_harden, not both):
       conf_filter: gain pixels with weight < T become uncertain -> ignored (valid=0, mask=0, weight=0).
       conf_harden: same, and the retained gain pixels get weight 1.0 (binary labels).
       Pixels with no gain stay valid negatives. NOTE: with the HARD loss only ``mask`` is used, so hardening is
       then identical to filtering; hardening only differs from filtering with the soft loss.
  2. min_px  : 8-connected gain patches with fewer than min_px pixels are removed (they become valid negatives).
  3. fill_px : 4-connected background holes with fewer than fill_px pixels that do not touch the tile border are
               filled with gain (only valid pixels; ignored pixels stay ignored). Filled pixels get the mean weight
               of the kept gain pixels of the tile.
min_px / fill_px <= 1 disable a step (a size of 1 can never remove or fill anything).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def remove_small(mask: np.ndarray, size: int) -> np.ndarray:
    """Remove 8-connected True patches with fewer than ``size`` pixels."""
    if size <= 1:
        return mask
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    small = np.bincount(labels.ravel()) < size
    small[0] = False
    return mask & ~small[labels]


def fill_small(mask: np.ndarray, size: int) -> np.ndarray:
    """Fill 4-connected background holes with fewer than ``size`` pixels; border-touching background is exterior."""
    if size <= 1:
        return mask
    labels, _ = ndimage.label(~mask)
    small = np.bincount(labels.ravel()) < size
    small[0] = False
    border = np.unique(
        np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    )
    small[border] = False
    return mask | small[labels]


def clean_labels(
    mask: np.ndarray,
    weight: np.ndarray,
    valid: np.ndarray,
    *,
    conf_filter: float | None = None,
    conf_harden: float | None = None,
    min_px: int = 0,
    fill_px: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return new (mask float32, weight float32, valid bool) of shape (H, W); inputs are not modified."""
    if conf_filter is not None and conf_harden is not None:
        raise ValueError("conf_filter and conf_harden are mutually exclusive")
    mask = mask > 0.5
    weight = weight.astype(np.float32, copy=True)
    valid = valid.astype(bool, copy=True)

    thr = conf_harden if conf_harden is not None else conf_filter
    if thr is not None:
        uncertain = mask & (weight < thr)
        valid &= ~uncertain
        mask &= ~uncertain
        if conf_harden is not None:
            weight = np.where(mask, 1.0, weight).astype(np.float32)

    if min_px > 1 and mask.any():
        mask = remove_small(mask, min_px)

    if fill_px > 1 and mask.any():
        filled = fill_small(mask, fill_px) & ~mask & valid
        if filled.any():
            weight[filled] = float(weight[mask].mean())
            mask = mask | filled

    weight = np.where(mask, weight, 0.0).astype(np.float32)
    return mask.astype(np.float32), weight, valid


def describe(conf_filter=None, conf_harden=None, remove=0, fill=0) -> str:
    """Seed-independent variant label, e.g. 'harden0.70+rm5+fill5'; 'baseline' if nothing is applied."""
    parts = []
    if conf_filter is not None:
        parts.append(f"conf{conf_filter:.2f}")
    if conf_harden is not None:
        parts.append(f"harden{conf_harden:.2f}")
    if remove > 1:
        parts.append(f"rm{remove}")
    if fill > 1:
        parts.append(f"fill{fill}")
    return "+".join(parts) or "baseline"
