"""Experiment plan registry. Each suite owns its own deterministic builder."""
from .augmentation import build_augmentation
from .benchmark import build_benchmark
from .data_sources import build_data_sources
from .hardness import build_hardness
from .losses import build_default_loss_manifest, build_tune_manifest
from .maskclean import build_maskclean
from .temporal import build_temporal

SUITES = ("benchmark", "data-sources", "losses", "hardness", "temporal", "augmentation", "maskclean")


def build_plan(suite: str, **kwargs):
    if suite == "losses":
        stage = kwargs.pop("stage", "defaults")
        if stage == "defaults":
            return build_default_loss_manifest(**kwargs)
        if stage == "tune":
            return build_tune_manifest(**kwargs)
        raise ValueError("losses stage must be defaults or tune")
    builders = {
        "benchmark": build_benchmark,
        "data-sources": build_data_sources,
        "hardness": build_hardness,
        "temporal": build_temporal,
        "augmentation": build_augmentation,
        "maskclean": build_maskclean,
    }
    try:
        return builders[suite](**kwargs)
    except KeyError as exc:
        raise ValueError(f"Unknown experiment suite {suite!r}; expected one of {SUITES}") from exc
