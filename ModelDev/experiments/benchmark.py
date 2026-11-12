"""Reproducible benchmark candidates using a seeded random search plan."""
from __future__ import annotations
import random
from .common import build_manifest


def build_benchmark(model="tsvit", group="benchmark", seeds=(0, 1, 2), trials=25, search_seed=0,
                    data_dir="../DataCollection/data/forest_gain_tiles", holdout_dir="../DataCollection/data/holdout_tiles",
                    split_seed=0, epochs=100, patience=20):
    if model not in ("tsvit", "sits_scd", "unet_lstm"):
        raise ValueError(f"Unknown model: {model}")
    if trials < 1:
        raise ValueError("trials must be >= 1")
    rng = random.Random(search_seed)
    rows = []
    for trial in range(trials):
        lr = 10 ** rng.uniform(-4.522879, -3.0)  # log-uniform 3e-5 .. 1e-3
        wd = 10 ** rng.uniform(-5.0, -1.30103)
        dropout = rng.uniform(0.0, 0.4)
        if model == "tsvit":
            dim = rng.choice([96, 128, 160, 192]); heads = rng.choice([h for h in (2, 4, 8) if dim % h == 0])
            opts = [f"embed_dim={dim}", f"num_heads={heads}", f"depth_temporal={rng.randint(1,4)}",
                    f"depth_spatial={rng.choice([2,4,6])}", f"patch_size={rng.choice([8,16,32])}",
                    f"dropout={dropout:.6f}", f"skip_channels={rng.choice([16,24,32,48])}"]
        elif model == "sits_scd":
            width = rng.choice([24,32,48,64]); heads = rng.choice([h for h in (2,4,8) if width % h == 0])
            opts = [f"base_features={width}", f"num_heads={heads}", f"dropout={dropout:.6f}",
                    f"depth_temporal={rng.randint(1,3)}", f"convs_per_block={rng.randint(1,2)}"]
        else:
            opts = [f"base_features={rng.choice([24,32,48,64])}", f"dropout={dropout:.6f}",
                    f"depth_temporal={rng.randint(1,3)}", f"lstm_kernel={rng.choice([3,5])}",
                    f"convs_per_block={rng.randint(1,2)}"]
        for seed in seeds:
            rows.append({"suite":"benchmark", "model_type":model, "sources":["s1","s2"], "seed":seed,
                         "learning_rate":lr, "weight_decay":wd, "model_opt":opts, "batch_size":rng.choice([2,4,8]),
                         "loss_name":"bce_focal_tversky", "epochs":epochs, "patience":patience,
                         "data_dir":data_dir, "holdout_dir":holdout_dir, "split_seed":split_seed,
                         "skip_test_eval":True, "val_metrics":True, "wandb_group":group,
                         "wandb_tags":["benchmark",model], "variant":f"trial-{trial+1:03d}"})
    return build_manifest(rows, "benchmark", group, {"search":"seeded-random", "search_seed":search_seed,
                          "trials":trials, "training_seeds":list(seeds), "selection_metric":"mean val_ap"})
