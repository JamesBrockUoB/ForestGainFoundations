"""Manifest-driven experiment CLI. Run from ModelDev with python -m experiments."""
from __future__ import annotations
import argparse, csv, json, subprocess, sys
from pathlib import Path
from .common import build_manifest, read_manifest, run_task, select_top_k, wandb_runs, write_manifest
from .suites import SUITES, build_plan

TRAIN_CONFIG_KEYS={"model_type","model_opt","loss_name","loss_type","pos_weight","reweight","hl_layer","loss_opt","batch_size","learning_rate","weight_decay","warmup_frac","epochs","patience","seed","data_dir","holdout_dir","val_frac","split_seed","sources","years","crop_size","repeats","min_gain_px","fill_holes_px","conf_filter","conf_harden","clean_val","no_augment","no_geo_aug","gain_sigma","frame_sigma","frame_drop_p","label_sigma","val_metrics","skip_test_eval","wandb_project","wandb_name","wandb_group","wandb_tags","suite","variant"}

def plan_cmd(a):
 kwargs={"model":a.model,"group":a.group or f"{a.suite}-{a.stage}-{a.model}","seeds":tuple(a.seeds),
  "sources":tuple(a.sources),"epochs":a.epochs,"patience":a.patience,"batch_size":a.batch_size,
  "learning_rate":a.learning_rate,"weight_decay":a.weight_decay,"data_dir":a.data_dir,"holdout_dir":a.holdout_dir,
  "split_seed":a.split_seed,"model_opt":a.model_opt,"stage":a.stage,"loss_name":a.loss,"loss_type":a.loss_type}
 if a.suite=="losses": kwargs.update({"selected_loss":a.selected_loss or "bce_focal_tversky","base_config":{k:v for k,v in {
  "epochs":a.epochs,"patience":a.patience,"batch_size":a.batch_size,"learning_rate":a.learning_rate,"weight_decay":a.weight_decay,
  "data_dir":a.data_dir,"holdout_dir":a.holdout_dir,"split_seed":a.split_seed,"model_opt":a.model_opt}.items() if v is not None}})
 if a.suite=="maskclean": kwargs.update({"sizes":tuple(a.mask_sizes),"confs":tuple(a.mask_confs),"combo_confs":tuple(a.combo_confs),"combo_morph":tuple(tuple(x) for x in a.combo_morph),"clean_val":a.clean_val})
 if a.suite=="benchmark": kwargs.update({"trials":a.trials,"search_seed":a.search_seed})
 if a.suite=="temporal" and a.year_sets: kwargs["year_sets"]=tuple(tuple(x) for x in a.year_sets)
 manifest=build_plan(a.suite,**kwargs)
 write_manifest(manifest,a.output)
 print(f"Plan {a.suite}/{a.stage}: {len(manifest['tasks'])} tasks; manifest={a.output}; sha256={manifest['manifest_sha256']}")
 for task in manifest["tasks"]: print(f"{task['task_id']:05d}: {' '.join(task['command'])}")

def run_cmd(a):
 manifest=read_manifest(a.manifest); indexes=[a.task] if a.task is not None else range(len(manifest["tasks"]))
 for index in indexes:
  code=run_task(a.manifest,index,a.cwd,a.log_dir)
  if code: raise SystemExit(code)

def select_cmd(a):
 rows=wandb_runs(a.project,a.group); selected=select_top_k(rows,a.metric,a.top_k,a.seeds)
 if not selected: raise SystemExit("No variants have completed results for every required seed (duplicates excluded)")
 result={"project":a.project,"source_group":a.group,"metric":a.metric,"required_seeds":a.seeds,
  "selected":[{"rank":i,"variant":x["variant"],"mean":x["mean"],"per_seed":x["per_seed"],
   "runs":[{"id":r["id"],"url":r["url"],"config":r["config"]} for r in x["runs"]]} for i,x in enumerate(selected,1)]}
 Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(result,indent=2)+"\n")
 for x in result["selected"]: print(f"{x['rank']}. {x['variant']}: mean {a.metric}={x['mean']:.6f}; {x['per_seed']}")
 print(f"Selection saved: {a.output}")

def confirm_cmd(a):
 selection=json.loads(Path(a.selection).read_text()); rows=[]
 for chosen in selection["selected"]:
  source=chosen["runs"][0]["config"]
  cfg={k:v for k,v in source.items() if k in TRAIN_CONFIG_KEYS}
  cfg.update({"suite":source.get("suite","confirmation"),"variant":f"confirm-{chosen['variant']}",
   "wandb_group":a.group,"wandb_tags":["confirmation",source.get("model_type","tsvit")],
   "skip_test_eval":False,"val_metrics":True})
  for seed in a.seeds: rows.append({**cfg,"seed":seed})
 manifest=build_manifest(rows,"confirmation",a.group,{"source_group":selection["source_group"],"selection_file":str(Path(a.selection)),
  "metric_for_selection":selection["metric"],"selected_variants":[x["variant"] for x in selection["selected"]],
  "confirmation_seeds":a.seeds,"holdout_evaluation":True})
 write_manifest(manifest,a.output); print(f"Wrote {len(rows)} retrain-and-holdout tasks to {a.output}")

def evaluate_cmd(a):
 cmd=[sys.executable,"eval.py",str(Path(a.checkpoint).resolve()),"--test_dir",a.test_dir]
 for key,value in (("threshold",a.threshold),("min_gain_px",a.min_gain_px),("fill_holes_px",a.fill_holes_px),
                   ("conf_filter",a.conf_filter),("conf_harden",a.conf_harden)):
  if value is not None and value != 0: cmd += ["--"+key,str(value)]
 subprocess.run(cmd,cwd=a.cwd,check=True)

def report_cmd(a):
 rows=wandb_runs(a.project,a.group); fields=["id","name","state","variant","seed","val_ap","val_eval_ap","val_eval_prevalence","val_eval_clean_ap","test_ap","test_prevalence","test_clean_ap"]
 out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
 with out.open("w",newline="",encoding="utf-8") as f:
  writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
  for r in rows:
   cfg,s=r["config"],r["summary"]; writer.writerow({"id":r["id"],"name":r["name"],"state":r["state"],"variant":cfg.get("variant"),"seed":cfg.get("seed"),
    **{k:s.get(k,s.get(k+".max")) for k in fields[5:]}})
 print(f"Exported {len(rows)} runs to {out}")

def main(argv=None):
 p=argparse.ArgumentParser(prog="python -m experiments"); sub=p.add_subparsers(dest="action",required=True)
 plan=sub.add_parser("plan"); plan.add_argument("suite",choices=SUITES); plan.add_argument("--stage",default="all",choices=("defaults","tune","singles","combos","all")); plan.add_argument("--model",default="tsvit",choices=("tsvit","sits_scd","unet_lstm"))
 plan.add_argument("--sources",nargs="+",default=["s1","s2"]); plan.add_argument("--seeds",nargs="+",type=int,default=[0,1,2]); plan.add_argument("--group"); plan.add_argument("--output",required=True)
 plan.add_argument("--epochs",type=int,default=100); plan.add_argument("--patience",type=int,default=20); plan.add_argument("--batch-size",type=int,default=8); plan.add_argument("--learning-rate",type=float,default=3e-4); plan.add_argument("--weight-decay",type=float,default=1e-2)
 plan.add_argument("--data-dir",default="../DataCollection/data/forest_gain_tiles"); plan.add_argument("--holdout-dir",default="../DataCollection/data/holdout_tiles"); plan.add_argument("--split-seed",type=int,default=0); plan.add_argument("--model-opt",nargs="+"); plan.add_argument("--loss",default="bce_focal_tversky"); plan.add_argument("--loss-type",choices=("hard","soft"),default="hard"); plan.add_argument("--selected-loss")
 plan.add_argument("--trials",type=int,default=25); plan.add_argument("--search-seed",type=int,default=0); plan.add_argument("--mask-sizes",nargs="+",type=int,default=[2,5,10,25]); plan.add_argument("--mask-confs",nargs="+",type=float,default=[round(.5+.05*i,2) for i in range(9)]); plan.add_argument("--combo-confs",nargs="+",type=float,default=[.6,.7,.8]); plan.add_argument("--combo-morph",nargs="+",type=lambda x:tuple(map(int,x.split(","))),default=[(5,5),(10,10)]); plan.add_argument("--clean-val",action="store_true"); plan.add_argument("--year-sets",nargs="+",type=lambda x:tuple(map(int,x.split(",")))); plan.set_defaults(func=plan_cmd)
 run=sub.add_parser("run"); run.add_argument("manifest"); run.add_argument("--task",type=int); run.add_argument("--cwd",default="."); run.add_argument("--log-dir",default="logs"); run.set_defaults(func=run_cmd)
 sel=sub.add_parser("select"); sel.add_argument("--project",required=True); sel.add_argument("--group",required=True); sel.add_argument("--metric",default="val_ap"); sel.add_argument("--top-k",type=int,default=3); sel.add_argument("--seeds",nargs="+",type=int,default=[0,1,2]); sel.add_argument("--output",required=True); sel.set_defaults(func=select_cmd)
 con=sub.add_parser("confirm"); con.add_argument("selection"); con.add_argument("--seeds",nargs="+",type=int,default=[0,1,2]); con.add_argument("--group",required=True); con.add_argument("--output",required=True); con.set_defaults(func=confirm_cmd)
 ev=sub.add_parser("evaluate"); ev.add_argument("--checkpoint",required=True); ev.add_argument("--test-dir",default="../DataCollection/data/holdout_tiles"); ev.add_argument("--cwd",default="."); ev.add_argument("--threshold",type=float,default=.5); ev.add_argument("--min-gain-px",type=int,default=0); ev.add_argument("--fill-holes-px",type=int,default=0); ev.add_argument("--conf-filter",type=float); ev.add_argument("--conf-harden",type=float); ev.set_defaults(func=evaluate_cmd)
 rep=sub.add_parser("report"); rep.add_argument("--project",required=True); rep.add_argument("--group",required=True); rep.add_argument("--output",required=True); rep.set_defaults(func=report_cmd)
 a=p.parse_args(argv); a.func(a)
if __name__=="__main__": main()
