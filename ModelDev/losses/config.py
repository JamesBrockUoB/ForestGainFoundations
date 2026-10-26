"""Loss configuration: plain dataclasses, the single home of every loss default.

CLI overrides use dotted keys, e.g. ``--loss_opt tversky.alpha=0.7 hardness.c=0.01``.
"""

from __future__ import annotations

import ast
import dataclasses
from dataclasses import dataclass
from typing import Iterable

# Public loss name -> the terms that are summed. Names are kept so old run names / checkpoints still make sense.
RECIPES: dict[str, tuple[str, ...]] = {
    "bce": ("bce",),
    "dice": ("dice",),
    "focal_tversky": ("tversky",),
    "bce_focal_tversky": ("bce", "tversky"),
    "focal": ("focal",),
    "lovasz": ("lovasz",),
    "bce_lovasz": ("bce", "lovasz"),
}
LOSS_NAMES = tuple(RECIPES)

# How the BCE / focal term's per-pixel contributions are re-weighted. Only recipes containing "bce" or "focal" support ohem / hardness.
REWEIGHTS = ("none", "ohem", "hardness")


@dataclass(frozen=True)
class TverskyConfig:
    """alpha = beta = 0.5, gamma = 1 is plain soft Dice (nothing to tune); choose the precision/recall trade-off with
    the decision threshold on validation. alpha > beta penalises false negatives more; gamma < 1 focuses on hard examples.
    """

    alpha: float = 0.5
    beta: float = 0.5
    gamma: float = 1.0
    smooth: float = 1e-6


@dataclass(frozen=True)
class FocalConfig:
    """Binary focal loss -alpha_t (1 - p_t)^gamma log(p_t). gamma = 0 is BCE; alpha=None leaves classes unweighted."""

    gamma: float = 2.0
    alpha: float | None = None


@dataclass(frozen=True)
class OhemConfig:
    """Online hard example mining (mmsegmentation OHEMPixelSampler, binary case): keep pixels whose probability of the
    true class is below max(thresh, the value that keeps >= min_kept pixels per tile). thresh <= 0 keeps just the
    min_kept highest-loss pixels per tile. Defaults are mmseg's (min_kept 4096 of a 65536-px tile).
    """

    thresh: float = 0.7
    min_kept: int = 4096


@dataclass(frozen=True)
class HardnessConfig:
    """Learned pixel hardness, "Not All Pixels Are Equal" (arXiv 2305.08462).

    layer:  model module whose (B, C, h, w) output feeds the final classifier. Model specific, so there is no default;
            list candidates with [n for n, _ in model.named_modules()].
    head: Strategy for constructing the hardness head. ``"auto"`` selects
            a strategy based on the target module; explicit options depend on
            the implementation below.
    c:      hardness offset, hardest/easiest pixel weight ratio is at most (1 + c) / c. Paper value 0.1 (0.01 also tried).
    alpha:  scale of the hardness head's own loss. Paper value 0.01.
    """

    layer: str | None = None
    head: str = "auto"
    c: float = 0.1
    alpha: float = 0.01


@dataclass(frozen=True)
class LossConfig:
    name: str = "bce_focal_tversky"
    confidence: str = (
        "hard"  # 'soft': gain pixels count with their confidence weight (validation metrics too)
    )
    pos_weight: float | None = None  # BCE positive-class weight
    lovasz_per_image: bool = (
        False  # Lovasz hinge per tile instead of pooled over the batch
    )
    reweight: str = "none"  # none | ohem | hardness  (BCE term only)
    tversky: TverskyConfig = TverskyConfig()
    focal: FocalConfig = FocalConfig()
    ohem: OhemConfig = OhemConfig()
    hardness: HardnessConfig = HardnessConfig()

    def __post_init__(self):
        if self.name not in RECIPES:
            raise ValueError(
                f"Unsupported loss '{self.name}'. Choose from {LOSS_NAMES}."
            )
        if self.confidence not in ("hard", "soft"):
            raise ValueError(
                f"Invalid loss_type '{self.confidence}'. Use 'hard' or 'soft'."
            )
        if self.reweight not in REWEIGHTS:
            raise ValueError(
                f"Invalid reweight '{self.reweight}'. Choose from {REWEIGHTS}."
            )
        per_pixel = {
            "bce",
            "focal",
        }  # the terms that hand per-pixel losses to the reweighter
        if self.reweight != "none" and not per_pixel & set(self.terms):
            raise ValueError(
                f"reweight='{self.reweight}' re-weights the BCE / focal term; use one of "
                f"{[n for n, t in RECIPES.items() if per_pixel & set(t)]}."
            )
        if self.reweight == "hardness" and not self.hardness.layer:
            raise ValueError(
                "reweight='hardness' needs hardness.layer (the module feeding the classifier)."
            )

    @property
    def terms(self) -> tuple[str, ...]:
        return RECIPES[self.name]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LossConfig":
        d = dict(d)
        for key, sub in _SECTIONS.items():
            if isinstance(d.get(key), dict):
                d[key] = sub(**d[key])
        return cls(**d)

    def with_overrides(self, opts: Iterable[str] | None) -> "LossConfig":
        """Apply ['tversky.alpha=0.7', 'lovasz_per_image=True', ...]."""
        cfg = self
        for opt in opts or ():
            key, eq, raw = opt.partition("=")
            if not eq:
                raise ValueError(f"loss option '{opt}' must look like key=value")
            cfg = _set(cfg, key.split("."), raw)
        return cfg

    @classmethod
    def from_legacy(cls, kwargs: dict) -> tuple["LossConfig", dict]:
        """Build a config from the flat hyperparameters of checkpoints saved before this refactor.
        Returns (config, the remaining kwargs)."""
        kw = dict(kwargs)
        take = lambda k, default=None: kw.pop(k, default)  # noqa: E731
        reweight = (
            "hardness"
            if take("hardness", False)
            else "ohem" if take("ohem", False) else "none"
        )
        cfg = cls(
            name=take("loss_name", "bce_focal_tversky"),
            confidence=take("loss_type", "hard"),
            pos_weight=take("pos_weight"),
            lovasz_per_image=take("lovasz_per_image", False),
            reweight=reweight,
            tversky=TverskyConfig(
                alpha=take("tversky_alpha", 0.5),
                beta=take("tversky_beta", 0.5),
                gamma=take("tversky_gamma", 1.0),
            ),
            focal=FocalConfig(
                gamma=take("focal_gamma", 2.0), alpha=take("focal_alpha")
            ),
            ohem=OhemConfig(
                thresh=take("ohem_thresh", 0.7), min_kept=take("ohem_min_kept", 4096)
            ),
            hardness=HardnessConfig(
                layer=take("hl_layer"),
                c=take("hl_c", 0.1),
                alpha=take("hl_alpha", 0.01),
            ),
        )
        kw.pop("hl_channels", None)
        kw.pop("ohem", None)
        kw.pop("hardness", None)
        return cfg, kw


_SECTIONS = {
    "tversky": TverskyConfig,
    "focal": FocalConfig,
    "ohem": OhemConfig,
    "hardness": HardnessConfig,
}


def parse_value(raw: str):
    """'0.7' -> 0.7, 'True' -> True, 'None' -> None, anything else stays a string."""
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def _set(obj, path: list[str], raw: str):
    head, *rest = path
    names = {f.name for f in dataclasses.fields(obj)}
    if head not in names:
        raise ValueError(
            f"unknown loss option '{head}' in {type(obj).__name__}; choose from {sorted(names)}"
        )
    current = getattr(obj, head)
    if rest:
        if not dataclasses.is_dataclass(current):
            raise ValueError(f"'{head}' has no sub-options")
        value = _set(current, rest, raw)
    else:
        if dataclasses.is_dataclass(current):
            raise ValueError(
                f"'{head}' is a group; set one of its fields, e.g. {head}.{dataclasses.fields(current)[0].name}=..."
            )
        value = parse_value(raw)
    return dataclasses.replace(obj, **{head: value})
