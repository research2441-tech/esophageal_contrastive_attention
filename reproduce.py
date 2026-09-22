from __future__ import annotations
import argparse, json
from train import train_main
from evaluate import eval_checkpoint

def main(config):
    ck=train_main(config)
    metrics=eval_checkpoint(config,ck)
    print("Checkpoint:",ck)
    print(json.dumps(metrics,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config",default="config.yaml"); a=p.parse_args(); main(a.config)
