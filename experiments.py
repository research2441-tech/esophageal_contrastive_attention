from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import yaml, pandas as pd
from train import train_main
from evaluate import eval_checkpoint

EXPERIMENTS = {
 "resnet50_baseline": {"backbone":"resnet50","attention":False,"contrastive_loss_weight":0.0,"augmentation":True},
 "resnet50_attention": {"backbone":"resnet50","attention":True,"contrastive_loss_weight":0.0,"augmentation":True},
 "resnet50_contrastive": {"backbone":"resnet50","attention":False,"contrastive_loss_weight":0.2,"augmentation":True},
 "proposed": {"backbone":"resnet50","attention":True,"contrastive_loss_weight":0.2,"augmentation":True},
 "proposed_no_augmentation": {"backbone":"resnet50","attention":True,"contrastive_loss_weight":0.2,"augmentation":False},
 "densenet121": {"backbone":"densenet121","attention":False,"contrastive_loss_weight":0.0,"augmentation":True},
 "efficientnet_b0": {"backbone":"efficientnet_b0","attention":False,"contrastive_loss_weight":0.0,"augmentation":True},
 "vit_base": {"backbone":"vit_base_patch16_224","attention":False,"contrastive_loss_weight":0.0,"augmentation":True},
}

def main(config):
    with open(config) as f: base=yaml.safe_load(f)
    rows=[]
    for name,s in EXPERIMENTS.items():
        cfg=copy.deepcopy(base)
        cfg["model"]["backbone"]=s["backbone"]; cfg["model"]["attention"]=s["attention"]
        cfg["training"]["contrastive_loss_weight"]=s["contrastive_loss_weight"]
        cfg["augmentation"]["enabled"]=s["augmentation"]
        cfg["output"]["directory"]=str(Path(base["output"]["directory"])/name)
        Path(cfg["output"]["directory"]).mkdir(parents=True,exist_ok=True)
        cp=Path(cfg["output"]["directory"])/"run_config.yaml"
        cp.write_text(yaml.safe_dump(cfg,sort_keys=False))
        ck=train_main(cp)
        metrics=eval_checkpoint(cp,ck); rows.append({"experiment":name,**metrics})
    out=Path(base["output"]["directory"]); out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out/"experiment_summary.csv",index=False)

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config",default="config.yaml"); a=p.parse_args(); main(a.config)
