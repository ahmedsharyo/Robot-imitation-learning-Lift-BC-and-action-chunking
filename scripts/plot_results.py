#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser(); p.add_argument("metrics",nargs="+",help="run metrics.json files"); p.add_argument("--output",default="results/figures/training_curves.png"); a=p.parse_args()
 fig,ax=plt.subplots()
 for filename in a.metrics:
  d=json.loads(Path(filename).read_text()); h=d["history"]; name=d["metadata"]["model"]+" H="+str(d["metadata"]["horizon"])
  ax.plot([x["epoch"] for x in h],[x["train_mse"] for x in h],label=name+" train")
  ax.plot([x["epoch"] for x in h],[x["val_mse"] for x in h],linestyle="--",label=name+" val")
 ax.set(xlabel="Epoch",ylabel="MSE",title="Training and validation action MSE"); ax.legend(); fig.tight_layout()
 Path(a.output).parent.mkdir(parents=True,exist_ok=True); fig.savefig(a.output,dpi=160)
if __name__=="__main__":main()
