"""Focused default-loss screen and small follow-up tuning plans."""
from __future__ import annotations
from typing import Any
from .common import build_manifest, select_top_k
DEFAULT_LOSSES=("bce","dice","focal","bce_focal","focal_tversky","bce_focal_tversky")
def build_default_loss_manifest(model="tsvit",sources=("s1","s2"),group="losses_defaults",seeds=(0,1,2),base_config=None):
 rows=[]
 for loss in DEFAULT_LOSSES:
  for seed in seeds:
   rows.append({**(base_config or {}),"suite":"losses","model_type":model,"sources":list(sources),"seed":seed,
    "loss_name":loss,"variant":f"default-{loss}","reweight":"none","wandb_group":group,"wandb_tags":["losses",model],
    "skip_test_eval":True,"val_metrics":True})
 return build_manifest(rows,"losses",group,{"stage":"defaults","recipes":list(DEFAULT_LOSSES),"seeds":list(seeds)})
def build_tune_manifest(model="tsvit",sources=("s1","s2"),group="losses_tune",seeds=(0,1,2),selected_loss="bce_focal_tversky",base_config=None):
 if selected_loss=="bce": points=[("pos-weight-auto",{"pos_weight":"auto"})]
 elif selected_loss in ("focal","bce_focal"):
  points=[(f"gamma{g}",{"loss_opt":[f"focal.gamma={g}.0"]}) for g in (1,3)]+[(f"alpha{a}",{"loss_opt":[f"focal.alpha={a}"]}) for a in (.25,.75)]
 elif selected_loss in ("focal_tversky","bce_focal_tversky"):
  points=[("alpha0.4",{"loss_opt":["tversky.alpha=0.4","tversky.beta=0.6"]}),("alpha0.6",{"loss_opt":["tversky.alpha=0.6","tversky.beta=0.4"]}),
          ("gamma0.75",{"loss_opt":["tversky.gamma=0.75"]}),("gamma1.33",{"loss_opt":["tversky.gamma=1.33"]})]
 elif selected_loss=="dice": points=[]
 else: raise ValueError(f"Unsupported tuning target: {selected_loss}")
 rows=[]
 for label,options in points:
  for seed in seeds:
   rows.append({**(base_config or {}),"suite":"losses","model_type":model,"sources":list(sources),"seed":seed,
    "loss_name":selected_loss,"variant":f"tune-{selected_loss}-{label}","reweight":"none","wandb_group":group,
    "wandb_tags":["losses",model],"skip_test_eval":True,"val_metrics":True,**options})
 return build_manifest(rows,"losses",group,{"stage":"tune","selected_loss":selected_loss,"seeds":list(seeds)})
def select_default_loss_winner(runs:list[dict[str,Any]],seeds=(0,1,2)):
 selected=select_top_k(runs,"val_ap",len(DEFAULT_LOSSES),seeds)
 eligible=[x for x in selected if x["variant"].startswith("default-")]
 if not eligible: raise ValueError("No complete default-loss variants found")
 return eligible[0]["variant"].removeprefix("default-")
