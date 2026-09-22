from __future__ import annotations
import argparse, json, time
from pathlib import Path
import yaml, torch, psutil
from model import build_model

def main(config,checkpoint=None):
    with open(config) as f: cfg=yaml.safe_load(f)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=build_model(cfg).to(device)
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint,map_location=device)["model"])
    model.eval()
    n=sum(p.numel() for p in model.parameters())
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_mb=sum(p.numel()*p.element_size() for p in model.parameters())/1024**2
    x=torch.randn(1,3,cfg["data"]["image_size"],cfg["data"]["image_size"],device=device)
    with torch.no_grad():
        for _ in range(10): model(x)
        if device.type=="cuda": torch.cuda.synchronize()
        t=time.perf_counter()
        for _ in range(100): model(x)
        if device.type=="cuda": torch.cuda.synchronize()
    latency_ms=(time.perf_counter()-t)*1000/100
    flops=None
    try:
        from thop import profile
        macs,_=profile(model,inputs=(x,),verbose=False); flops=float(2*macs)
    except Exception: pass
    r={"parameters":int(n),"trainable_parameters":int(trainable),"model_size_mb":model_mb,
       "latency_ms_per_image":latency_ms,"estimated_flops":flops,
       "device":str(device),"cpu_ram_gb":psutil.virtual_memory().total/1024**3}
    if device.type=="cuda":
        r["gpu_name"]=torch.cuda.get_device_name(0)
        r["peak_gpu_memory_mb"]=torch.cuda.max_memory_allocated()/1024**2
    out=Path(cfg["output"]["directory"]); out.mkdir(parents=True,exist_ok=True)
    (out/"complexity.json").write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config",default="config.yaml"); p.add_argument("--checkpoint"); a=p.parse_args(); main(a.config,a.checkpoint)
