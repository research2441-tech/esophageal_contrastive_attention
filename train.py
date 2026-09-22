from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml, torch, numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score
from data_pipeline import prepare, loaders, class_weights, seed_everything
from model import build_model, supervised_contrastive_loss

def choose_device(name):
    if name!="auto": return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_epoch(model, loader, opt, scaler, device, ce, cfg, train=True):
    model.train(train); total=0.; ys=[]; ps=[]
    for x,y,_ in loader:
        x=x.to(device); y=y.to(device); opt.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=(cfg["training"]["mixed_precision"] and device.type=="cuda")):
                o=model(x)
                l1=ce(o["logits"],y)
                l2=supervised_contrastive_loss(o["projection"],y,cfg["training"]["contrastive_temperature"])
                loss=cfg["training"]["supervised_loss_weight"]*l1+cfg["training"]["contrastive_loss_weight"]*l2
            if train:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        total+=loss.item()*len(y); ys.extend(y.detach().cpu().tolist()); ps.extend(o["logits"].argmax(1).detach().cpu().tolist())
    return total/len(loader.dataset), balanced_accuracy_score(ys,ps)

def train_main(cfg_path):
    with open(cfg_path) as f: cfg=yaml.safe_load(f)
    seed_everything(cfg["seed"]); out=Path(cfg["output"]["directory"]); out.mkdir(exist_ok=True,parents=True)
    manifest=prepare(cfg,out); L=loaders(manifest,cfg); device=choose_device(cfg["device"])
    model=build_model(cfg).to(device)
    w=class_weights(manifest[manifest.split=="train"],cfg["model"]["num_classes"]).to(device) if cfg["training"]["use_class_weights"] else None
    ce=torch.nn.CrossEntropyLoss(weight=w)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg["training"]["learning_rate"],weight_decay=cfg["training"]["weight_decay"])
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg["training"]["epochs"])
    scaler=torch.cuda.amp.GradScaler(enabled=(cfg["training"]["mixed_precision"] and device.type=="cuda"))
    best=-1; bad=0; hist=[]
    ckpt=out/cfg["output"]["checkpoint_name"]
    for ep in range(1,cfg["training"]["epochs"]+1):
        trl,trb=run_epoch(model,L["train"],opt,scaler,device,ce,cfg,True)
        val,vab=run_epoch(model,L["val"],opt,scaler,device,ce,cfg,False); sched.step()
        hist.append({"epoch":ep,"train_loss":trl,"train_balanced_accuracy":trb,"val_loss":val,"val_balanced_accuracy":vab})
        pd.DataFrame(hist).to_csv(out/"training_history.csv",index=False)
        if vab>best:
            best=vab; bad=0
            torch.save({"model":model.state_dict(),"config":cfg,"best_val_balanced_accuracy":best},ckpt)
        else:
            bad+=1
            if bad>=cfg["training"]["early_stopping_patience"]: break
    return str(ckpt)

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config",default="config.yaml"); a=p.parse_args()
    print(train_main(a.config))
