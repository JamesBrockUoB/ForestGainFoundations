"""Optuna TPE benchmark search, preserving the original adaptive workflow."""
from __future__ import annotations

import json
import re
import statistics
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path

from .common import build_manifest, train_command, write_manifest

BENCHMARK_MODELS = ("tsvit", "sits_scd", "unet_lstm")
BENCHMARK_LOSS = "bce_focal_tversky"


def _storage(path: str):
    import optuna
    try:  # Optuna 4+
        from optuna.storages.journal import JournalFileBackend
        backend = JournalFileBackend(path)
    except ImportError:  # Optuna 3
        from optuna.storages import JournalFileStorage
        backend = JournalFileStorage(path)
    return optuna.storages.JournalStorage(backend)


def _suggest(trial, model: str) -> dict:
    lr = trial.suggest_float("learning_rate", 3e-5, 1e-3, log=True)
    wd = trial.suggest_float("weight_decay", 1e-5, 5e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    if model == "tsvit":
        dim = trial.suggest_categorical("embed_dim", [96, 128, 160, 192])
        heads = trial.suggest_categorical("num_heads", [h for h in (2, 4, 8) if dim % h == 0])
        structure = {"embed_dim":dim,"num_heads":heads,"depth_temporal":trial.suggest_int("depth_temporal",1,4),
            "depth_spatial":trial.suggest_categorical("depth_spatial",[2,4,6]),
            "patch_size":trial.suggest_categorical("patch_size",[8,16,32]),"dropout":dropout,
            "skip_channels":trial.suggest_categorical("skip_channels",[16,24,32,48])}
    elif model == "sits_scd":
        width=trial.suggest_categorical("base_features",[24,32,48,64])
        heads=trial.suggest_categorical("num_heads",[h for h in (2,4,8) if width % h == 0])
        structure={"base_features":width,"num_heads":heads,"dropout":dropout,
            "depth_temporal":trial.suggest_int("depth_temporal",1,3),
            "convs_per_block":trial.suggest_int("convs_per_block",1,2)}
    elif model == "unet_lstm":
        structure={"base_features":trial.suggest_categorical("base_features",[24,32,48,64]),
            "dropout":dropout,"depth_temporal":trial.suggest_int("depth_temporal",1,3),
            "lstm_kernel":trial.suggest_categorical("lstm_kernel",[3,5]),
            "convs_per_block":trial.suggest_int("convs_per_block",1,2)}
    else:
        raise ValueError(f"Unsupported benchmark model: {model}")
    batch = trial.suggest_categorical("batch_size",[2,4,8])
    return {"learning_rate":lr,"weight_decay":wd,"batch_size":batch,
            "model_opt":[f"{k}={v}" for k,v in structure.items()]}


def run_tpe_benchmark(*, model="tsvit", trials=25, seeds=(0,1,2), study_name=None,
                      storage_path=None, worker_id=0, sampler_seed=0, startup_trials=None,
                      epochs=100, patience=20, batch_size=8,
                      data_dir="../DataCollection/data/forest_gain_tiles",
                      holdout_dir="../DataCollection/data/holdout_tiles", split_seed=0,
                      wandb_project="Forest-Gain-CD", output_dir="experiment_manifests",
                      modeldev_dir="."):
    """Run/continue one shared Optuna TPE study; objective is mean val AP over seeds.

    Parallel workers share the same study name and journal path and use distinct
    worker IDs/seeds. Each trial is only complete if every requested seed trains.
    """
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit("The benchmark requires Optuna: install optuna") from exc
    if model not in BENCHMARK_MODELS or trials < 1 or not seeds:
        raise ValueError("Invalid model, trial count, or training seeds")
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_name = study_name or f"forestgain_{model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    startup = startup_trials if startup_trials is not None else min(24,max(5,trials//3))
    study = optuna.create_study(direction="maximize", study_name=study_name,
        storage=_storage(storage_path) if storage_path else None, load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=sampler_seed+worker_id,
            n_startup_trials=min(startup,trials), multivariate=True, constant_liar=True))
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    cwd=Path(modeldev_dir).resolve()
    results=[]
    while True:
        if len(study.get_trials(deepcopy=False)) >= trials:
            break
        trial=study.ask()
        if trial.number >= trials:
            study.tell(trial,state=optuna.trial.TrialState.PRUNED)
            break
        number=trial.number+1
        cfg={"suite":"benchmark","model_type":model,"sources":["s1","s2"],
             "loss_name":BENCHMARK_LOSS,"epochs":epochs,"patience":patience,
             "batch_size":batch_size,"data_dir":data_dir,"holdout_dir":holdout_dir,
             "split_seed":split_seed,"skip_test_eval":True,"val_metrics":True,
             "wandb_project":wandb_project,"wandb_group":study_name,
             "wandb_tags":["benchmark",model],"variant":f"trial{number:03d}"} | _suggest(trial,model)
        trial.set_user_attr("config",cfg); trial.set_user_attr("seeds",list(seeds))
        per_seed=[]; failed=None
        for seed in seeds:
            seed_cfg=dict(cfg,seed=int(seed))
            command=train_command(seed_cfg)
            print(f"\n[{model} trial {number}/{trials} seed {seed} worker {worker_id}] {' '.join(command)}",flush=True)
            proc=subprocess.run(command,cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            print(proc.stdout,end="" if proc.stdout.endswith("\n") else "\n",flush=True)
            match=re.search(r"best checkpoint:.*?val_ap:\s*([0-9]*\.?[0-9]+)",proc.stdout)
            ap=float(match.group(1)) if match else None
            per_seed.append({"seed":int(seed),"command":command,"returncode":proc.returncode,"val_ap":ap})
            if proc.returncode or ap is None:
                failed=(seed,proc.returncode); break
        aps=[x["val_ap"] for x in per_seed if x["val_ap"] is not None]
        if failed:
            study.tell(trial,state=optuna.trial.TrialState.FAIL)
            outcome={"trial":number,"config":cfg,"seeds":per_seed,"val_ap":None,
                     "error":f"seed {failed[0]} failed or val_ap was not emitted"}
            print(f"Trial {number} failed: {outcome['error']}",file=sys.stderr)
        else:
            value=statistics.fmean(aps); study.tell(trial,value)
            trial.set_user_attr("val_ap_per_seed",aps)
            outcome={"trial":number,"config":cfg,"seeds":per_seed,"val_ap":value,
                     "val_ap_std":statistics.pstdev(aps) if len(aps)>1 else 0.0}
            print(f"Trial {number}: mean val_ap={value:.6f} over seeds {list(seeds)}",flush=True)
        results.append(outcome)
        path=out/f"benchmark_{model}_worker{worker_id:02d}.json"
        path.write_text(json.dumps({"study":study_name,"sampler":"Optuna TPE (multivariate, constant liar)",
            "requested_trials":trials,"seeds":list(seeds),"worker_id":worker_id,"results":results},indent=2)+"\n")
    complete=study.get_trials(deepcopy=False,states=[optuna.trial.TrialState.COMPLETE])
    if complete:
        print(f"BEST {model}: trial {study.best_trial.number+1}, val_ap={study.best_value:.6f}, params={study.best_trial.params}")
    return study


def build_benchmark(*args, **kwargs):
    """TPE is adaptive and cannot be represented as a fixed task-array manifest."""
    raise ValueError("Benchmark uses adaptive Optuna TPE; run `python -m experiments benchmark ...` instead of `plan benchmark`.")
