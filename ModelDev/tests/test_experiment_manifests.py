from experiments.common import build_manifest, select_top_k, train_command
from experiments.losses import build_default_loss_manifest
from experiments.maskclean import build_maskclean


def test_training_arguments_keep_train_py_underscore_names():
    command = train_command({"model_type": "tsvit", "loss_name": "bce", "skip_test_eval": True})
    assert "--model_type" in command
    assert "--loss" in command
    assert "--skip_test_eval" in command
    assert "--model-type" not in command


def test_default_loss_screen_has_six_recipes_and_three_seeds():
    manifest = build_default_loss_manifest()
    assert len(manifest["tasks"]) == 18
    assert {t["config"]["variant"] for t in manifest["tasks"]} == {
        "default-bce", "default-dice", "default-focal", "default-bce_focal",
        "default-focal_tversky", "default-bce_focal_tversky",
    }


def test_selection_requires_all_unique_seeds():
    runs = []
    for seed, value in enumerate((.4, .5, .6)):
        runs.append({"state":"finished","config":{"variant":"x","seed":seed},"summary":{"val_ap":value}})
    assert select_top_k(runs, required_seeds=(0,1,2))[0]["mean"] == .5
    assert select_top_k(runs[:2], required_seeds=(0,1,2)) == []
    assert select_top_k(runs + [runs[0]], required_seeds=(0,1,2)) == []


def test_maskclean_manifest_is_stable_and_carries_split_settings():
    manifest = build_maskclean(seeds=(0,), stage="singles", sizes=(2,), confs=(.6,))
    configs = [task["config"] for task in manifest["tasks"]]
    assert all(c["split_seed"] == 0 and c["skip_test_eval"] for c in configs)
    assert any(c["variant"] == "fill2" for c in configs)
