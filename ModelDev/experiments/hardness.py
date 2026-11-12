"""Matched BCE-based controls for OHEM and learned pixel hardness."""
from .common import build_manifest

def build_hardness(model="tsvit", group="hardness", seeds=(0,1,2), sources=("s1","s2"),
                   loss_names=("bce","bce_focal"), layers=None, epochs=100, patience=20,
                   batch_size=8, learning_rate=3e-4, weight_decay=1e-2,
                   data_dir="../DataCollection/data/forest_gain_tiles",holdout_dir="../DataCollection/data/holdout_tiles",split_seed=0):
    defaults={"tsvit":("decoder","decoder.final_head"),"sits_scd":("final_head",),"unet_lstm":("out_head",)}
    layers=tuple(layers or defaults[model]); rows=[]
    for loss in loss_names:
      conditions=[("none",None,None)] + [("ohem",None,t) for t in (.7,.9)] + [("hardness",layer,c) for layer in layers for c in (.01,.1)]
      for mode,layer,c in conditions:
       for seed in seeds:
        opts=[]
        if mode=="ohem": opts=["ohem.thresh="+str(t)]
        if mode=="hardness": opts=["hardness.c="+str(c)]
        rows.append({"suite":"hardness","model_type":model,"sources":list(sources),"seed":seed,"loss_name":loss,
         "reweight":mode,"hl_layer":layer,"loss_opt":opts or None,"epochs":epochs,"patience":patience,"batch_size":batch_size,
         "learning_rate":learning_rate,"weight_decay":weight_decay,"data_dir":data_dir,"holdout_dir":holdout_dir,
         "split_seed":split_seed,"skip_test_eval":True,"val_metrics":True,"wandb_group":group,"wandb_tags":["hardness",model],
         "variant":f"{loss}-{mode}"+(f"-{layer}-c{c}" if mode=="hardness" else f"-t{t}" if mode=="ohem" else "")})
    return build_manifest(rows,"hardness",group,{"matched_controls":True,"seeds":list(seeds)})
