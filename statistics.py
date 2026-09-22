from __future__ import annotations
import argparse, json
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from scipy.stats import binomtest

def bootstrap_ci(y,p,metric,n=2000,level=.95,seed=42):
    rng=np.random.default_rng(seed); vals=[]; N=len(y)
    for _ in range(n):
        idx=rng.integers(0,N,N)
        vals.append(metric(y[idx],p[idx]))
    a=(1-level)/2
    return float(np.quantile(vals,a)), float(np.quantile(vals,1-a))

def mcnemar_exact(y,pa,pb):
    a=(pa==y); b=(pb==y)
    n01=int(np.sum(a & ~b)); n10=int(np.sum(~a & b))
    n=n01+n10
    p=1.0 if n==0 else binomtest(min(n01,n10),n,0.5,alternative="two-sided").pvalue
    return {"a_correct_b_wrong":n01,"a_wrong_b_correct":n10,"p_value":float(p)}

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--predictions",required=True); args=ap.parse_args()
    df=pd.read_csv(args.predictions); y=df.y_true.to_numpy(); p=df.y_pred.to_numpy()
    out={
      "accuracy_ci95":bootstrap_ci(y,p,accuracy_score),
      "balanced_accuracy_ci95":bootstrap_ci(y,p,balanced_accuracy_score)
    }
    print(json.dumps(out,indent=2))
