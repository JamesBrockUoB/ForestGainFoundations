"""Augmentation and sampling comparison plans."""
from .common import build_manifest

def build_augmentation(model="tsvit",group="augmentation",seeds=(0,1,2),sources=("s1","s2"),loss_name="bce_focal_tversky",
                       loss_type="hard",learning_rate=3e-4,weight_decay=1e-2,model_opt=None,batch_size=8,epochs=100,patience=20,
                       data_dir="../DataCollection/data/forest_gain_tiles",holdout_dir="../DataCollection/data/holdout_tiles",split_seed=0,
                       match_steps=True):
 points=[("baseline",{}),("aug-off",{"no_augment":True}),("geo-only",{"gain_sigma":0.,"frame_sigma":0.}),
 ("spectral-only",{"no_geo_aug":True}),("spectral-g0.1-f0.03",{"gain_sigma":.1,"frame_sigma":.03}),
 ("spectral-g0.2-f0.05",{"gain_sigma":.2,"frame_sigma":.05}),("framedrop0.1",{"frame_drop_p":.1}),
 ("framedrop0.25",{"frame_drop_p":.25})]
 points += [(f"repeats{n}",{"repeats":n}) for n in (2,4,8)]
 points += [(f"crop{c}-repeats4",{"crop_size":c,"repeats":4}) for c in (128,192)]
 if loss_type=="soft": points += [(f"label-sigma{s}",{"label_sigma":s}) for s in (0.,.05,.1,.2)]
 rows=[]
 for name,extra in points:
  repeats=extra.get("repeats",1); divisor=repeats if match_steps else 1
  for seed in seeds:
   rows.append({"suite":"augmentation","model_type":model,"sources":list(sources),"seed":seed,"loss_name":loss_name,
    "loss_type":loss_type,"learning_rate":learning_rate,"weight_decay":weight_decay,"model_opt":model_opt,"batch_size":batch_size,
    "epochs":max(1,round(epochs/divisor)),"patience":max(1,round(patience/divisor)) if patience else 0,
    "data_dir":data_dir,"holdout_dir":holdout_dir,"split_seed":split_seed,"skip_test_eval":True,"val_metrics":True,
    "wandb_group":group,"wandb_tags":["augmentation",model],"variant":name,"repeats":repeats,**extra})
 return build_manifest(rows,"augmentation",group,{"match_optimizer_steps":match_steps,"variants":[n for n,_ in points]})
