"""Temporal input-frame ablation plans."""
from .common import build_manifest
YEAR_SETS=((2017,2024),(2017,2021,2024),(2017,2020,2022,2024),None)
def build_temporal(model="tsvit",group="temporal",seeds=(0,1,2),sources=("s1","s2"),year_sets=YEAR_SETS,
                   loss_name="bce_focal_tversky",learning_rate=3e-4,weight_decay=1e-2,model_opt=None,batch_size=8,
                   epochs=100,patience=20,data_dir="../DataCollection/data/forest_gain_tiles",
                   holdout_dir="../DataCollection/data/holdout_tiles",split_seed=0):
 rows=[]
 for years in year_sets:
  for seed in seeds:
   rows.append({"suite":"temporal","model_type":model,"sources":list(sources),"years":list(years) if years else None,
    "seed":seed,"loss_name":loss_name,"learning_rate":learning_rate,"weight_decay":weight_decay,"model_opt":model_opt,
    "batch_size":batch_size,"epochs":epochs,"patience":patience,"data_dir":data_dir,"holdout_dir":holdout_dir,
    "split_seed":split_seed,"skip_test_eval":True,"val_metrics":True,"wandb_group":group,"wandb_tags":["temporal",model],
    "variant":"years-"+("all" if years is None else "-".join(map(str,years)))})
 return build_manifest(rows,"temporal",group,{"year_sets":[list(x) if x else "all" for x in year_sets]})
