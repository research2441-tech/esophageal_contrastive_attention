from __future__ import annotations
import hashlib, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

IMG_EXT = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}

def seed_everything(seed:int=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20), b""): h.update(b)
    return h.hexdigest()

def discover(root, class_names):
    rows=[]
    root=Path(root)
    for y,name in enumerate(class_names):
        d=root/name
        if not d.exists(): continue
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXT:
                rows.append({"path":str(p.resolve()),"label":y,"class_name":name})
    if not rows:
        raise FileNotFoundError(f"No images found below {root}")
    df=pd.DataFrame(rows)
    df["sha256"]=[sha256(p) for p in df.path]
    return df

def split_manifest(df, train_ratio=.7, val_ratio=.1, test_ratio=.2, seed=42):
    if abs(train_ratio+val_ratio+test_ratio-1)>1e-8:
        raise ValueError("split ratios must sum to 1")
    tr, temp = train_test_split(df, test_size=1-train_ratio, stratify=df.label, random_state=seed)
    rel_test=test_ratio/(val_ratio+test_ratio)
    va, te = train_test_split(temp, test_size=rel_test, stratify=temp.label, random_state=seed)
    tr=tr.copy(); va=va.copy(); te=te.copy()
    tr["split"]="train"; va["split"]="val"; te["split"]="test"
    out=pd.concat([tr,va,te], ignore_index=True)
    # exact duplicate leakage audit
    leak=out.groupby("sha256").split.nunique()
    bad=leak[leak>1]
    if len(bad):
        raise RuntimeError(f"Exact duplicate leakage across splits detected for {len(bad)} hashes.")
    return out

def build_transforms(size, aug):
    norm=transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    train_ops=[transforms.Resize((size,size))]
    if aug.get("enabled",True):
        train_ops += [
            transforms.RandomHorizontalFlip(aug.get("horizontal_flip_p",.5)),
            transforms.RandomVerticalFlip(aug.get("vertical_flip_p",0.0)),
            transforms.RandomRotation(aug.get("rotation_degrees",10)),
            transforms.ColorJitter(
                brightness=aug.get("brightness",.15),
                contrast=aug.get("contrast",.15),
                saturation=aug.get("saturation",.10),
                hue=aug.get("hue",.02)),
        ]
    train_ops += [transforms.ToTensor(), norm]
    eval_ops=[transforms.Resize((size,size)), transforms.ToTensor(), norm]
    return transforms.Compose(train_ops), transforms.Compose(eval_ops)

class ImageDataset(Dataset):
    def __init__(self, df, tfm):
        self.df=df.reset_index(drop=True); self.tfm=tfm
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]
        x=Image.open(r.path).convert("RGB")
        return self.tfm(x), int(r.label), str(r.path)

def loaders(manifest, cfg):
    tr_tfm, ev_tfm=build_transforms(cfg["data"]["image_size"], cfg["augmentation"])
    bs=cfg["training"]["batch_size"]; nw=cfg["data"]["num_workers"]
    out={}
    for split,tfm,shuffle in [("train",tr_tfm,True),("val",ev_tfm,False),("test",ev_tfm,False)]:
        ds=ImageDataset(manifest[manifest.split==split], tfm)
        out[split]=DataLoader(ds,batch_size=bs,shuffle=shuffle,num_workers=nw,pin_memory=True)
    return out

def class_weights(train_df, n_classes=2):
    c=np.bincount(train_df.label.to_numpy(), minlength=n_classes).astype(float)
    w=c.sum()/(n_classes*np.maximum(c,1))
    return torch.tensor(w,dtype=torch.float32)

def prepare(cfg, outdir):
    seed_everything(cfg.get("seed",42))
    df=discover(cfg["data"]["root"], cfg["data"]["class_names"])
    manifest=split_manifest(df,cfg["data"]["train_ratio"],cfg["data"]["val_ratio"],
                            cfg["data"]["test_ratio"],cfg.get("seed",42))
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    manifest.to_csv(outdir/"split_manifest.csv",index=False)
    dup=df[df.duplicated("sha256",keep=False)].sort_values("sha256")
    dup.to_csv(outdir/"duplicate_report.csv",index=False)
    summary=manifest.groupby(["split","class_name"]).size().unstack(fill_value=0)
    summary.to_csv(outdir/"split_counts.csv")
    return manifest
