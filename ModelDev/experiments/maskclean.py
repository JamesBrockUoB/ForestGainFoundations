"""Training-label cleaning ablations: singles and selected combinations."""
from .common import build_manifest

def build_maskclean(model="tsvit",group="maskclean",seeds=(0,1,2),sources=("s1","s2"),loss_name="bce_focal_tversky",
                    loss_type="soft",learning_rate=3e-4,weight_decay=1e-2,model_opt=None,batch_size=8,epochs=100,patience=20,
                    data_dir="../DataCollection/data/forest_gain_tiles",holdout_dir="../DataCollection/data/holdout_tiles",split_seed=0,
                    stage="all",sizes=(2,5,10,25),confs=tuple(round(.5+.05*i,2) for i in range(9)),
                    combo_confs=(.6,.7,.8),combo_morph=((5,5),(10,10)),clean_val=False):
 if stage not in ("singles","combos","all"): raise ValueError("maskclean stage must be singles, combos, or all")
 variants=[("baseline",{})]
 if stage in ("singles","all"):
  variants += [(f"rm{n}",{"min_gain_px":n}) for n in sizes if n>1]
  variants += [(f"fill{n}",{"fill_holes_px":n}) for n in sizes if n>1]
  variants += [(f"conf{c:.2f}",{"conf_filter":c}) for c in confs]
  if loss_type=="soft": variants += [(f"harden{c:.2f}",{"conf_harden":c}) for c in confs]
 if stage in ("combos","all"):
  for rm,fill in combo_morph: variants.append((f"rm{rm}-fill{fill}",{"min_gain_px":rm,"fill_holes_px":fill}))
  for c in combo_confs:
   for rm,fill in combo_morph:
    variants.append((f"conf{c:.2f}-rm{rm}-fill{fill}",{"conf_filter":c,"min_gain_px":rm,"fill_holes_px":fill}))
    if loss_type=="soft": variants.append((f"harden{c:.2f}-rm{rm}-fill{fill}",{"conf_harden":c,"min_gain_px":rm,"fill_holes_px":fill}))
 rows=[]
 for name,extra in variants:
  for seed in seeds:
   rows.append({"suite":"maskclean","model_type":model,"sources":list(sources),"seed":seed,"loss_name":loss_name,
    "loss_type":loss_type,"learning_rate":learning_rate,"weight_decay":weight_decay,"model_opt":model_opt,"batch_size":batch_size,
    "epochs":epochs,"patience":patience,"data_dir":data_dir,"holdout_dir":holdout_dir,"split_seed":split_seed,
    "skip_test_eval":True,"val_metrics":True,"wandb_group":group,"wandb_tags":["maskclean",model],"variant":name,
    "clean_val":clean_val,**extra})
 return build_manifest(rows,"maskclean",group,{"stage":stage,"clean_validation_labels":clean_val,"variants":[n for n,_ in variants]})
