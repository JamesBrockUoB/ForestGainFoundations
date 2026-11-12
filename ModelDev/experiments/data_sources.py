"""Input-source ablation plans."""
from .common import build_manifest
SOURCE_SETS = (("s1",),("s2",),("s1","s2"),("alphaearth",),("tessera",),
 ("s1","alphaearth"),("s2","alphaearth"),("s1","s2","alphaearth"),("s1","tessera"),
 ("s2","tessera"),("s1","s2","tessera"),("s1","s2","alphaearth","tessera"),
 ("alphaearth","tessera"),("s1","alphaearth","tessera"),("s2","alphaearth","tessera"))

def build_data_sources(model="tsvit", group="data-sources", seeds=(0,1,2), sources=None,
                       loss_name="bce_focal_tversky", learning_rate=3e-4, weight_decay=1e-2,
                       model_opt=None, batch_size=8, epochs=100, patience=20,
                       data_dir="../DataCollection/data/forest_gain_tiles", holdout_dir="../DataCollection/data/holdout_tiles", split_seed=0):
    sets = [tuple(sources)] if sources else SOURCE_SETS
    rows=[]
    for source_set in sets:
      for seed in seeds:
       rows.append({"suite":"data-sources","model_type":model,"sources":list(source_set),"seed":seed,
        "loss_name":loss_name,"learning_rate":learning_rate,"weight_decay":weight_decay,"model_opt":model_opt,
        "batch_size":batch_size,"epochs":epochs,"patience":patience,"data_dir":data_dir,"holdout_dir":holdout_dir,
        "split_seed":split_seed,"skip_test_eval":True,"val_metrics":True,"wandb_group":group,
        "wandb_tags":["data-sources",model],"variant":"sources-"+"+".join(source_set)})
    return build_manifest(rows,"data-sources",group,{"source_sets":[list(x) for x in sets],"seeds":list(seeds)})
