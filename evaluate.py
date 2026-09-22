from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml, torch, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score,balanced_accuracy_score,precision_score,recall_score,f1_score,
roc_auc_score,average_precision_score,matthews_corrcoef,confusion_matrix,roc_curve,precision_recall_curve)
from data_pipeline import loaders
from model import build_model

def specificity_npv(y,p):
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    spec=tn/(tn+fp) if tn+fp else float("nan")
    npv=tn/(tn+fn) if tn+fn else float("nan")
    return spec,npv,tn,fp,fn,tp

def eval_checkpoint(cfg_path, checkpoint):
    with open(cfg_path) as f: cfg=yaml.safe_load(f)
    out=Path(cfg["output"]["directory"]); manifest=pd.read_csv(out/"split_manifest.csv")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=build_model(cfg).to(device)
    state=torch.load(checkpoint,map_location=device); model.load_state_dict(state["model"]); model.eval()
    ys=[]; probs=[]; paths=[]
    with torch.no_grad():
        for x,y,p in loaders(manifest,cfg)["test"]:
            q=torch.softmax(model(x.to(device))["logits"],1)[:,1].cpu().numpy()
            ys.extend(y.numpy().tolist()); probs.extend(q.tolist()); paths.extend(p)
    y=np.asarray(ys); prob=np.asarray(probs); pred=(prob>=.5).astype(int)
    spec,npv,tn,fp,fn,tp=specificity_npv(y,pred)
    m={
      "accuracy":accuracy_score(y,pred),
      "balanced_accuracy":balanced_accuracy_score(y,pred),
      "precision_ppv":precision_score(y,pred,zero_division=0),
      "recall_sensitivity":recall_score(y,pred,zero_division=0),
      "specificity":spec,
      "npv":npv,
      "f1":f1_score(y,pred,zero_division=0),
      "roc_auc":roc_auc_score(y,prob),
      "pr_auc":average_precision_score(y,prob),
      "mcc":matthews_corrcoef(y,pred),
      "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)
    }
    (out/"metrics.json").write_text(json.dumps(m,indent=2))
    pd.DataFrame({"path":paths,"y_true":y,"prob_class_1":prob,"y_pred":pred}).to_csv(out/"test_predictions.csv",index=False)

    fpr,tpr,_=roc_curve(y,prob); plt.figure(); plt.plot(fpr,tpr); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve"); plt.tight_layout(); plt.savefig(out/"roc_curve.png",dpi=200); plt.close()
    pr,re,_=precision_recall_curve(y,prob); plt.figure(); plt.plot(re,pr); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve"); plt.tight_layout(); plt.savefig(out/"pr_curve.png",dpi=200); plt.close()
    cm=confusion_matrix(y,pred,labels=[0,1]); plt.figure(); plt.imshow(cm); plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    for i in range(2):
        for j in range(2): plt.text(j,i,str(cm[i,j]),ha="center",va="center")
    plt.tight_layout(); plt.savefig(out/"confusion_matrix.png",dpi=200); plt.close()
    return m

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config",default="config.yaml"); p.add_argument("--checkpoint",required=True); a=p.parse_args()
    print(json.dumps(eval_checkpoint(a.config,a.checkpoint),indent=2))
