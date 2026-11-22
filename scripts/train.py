#!/usr/bin/env python3
import argparse, json, platform
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.robot_il.data.dataset import load_trajectories, split_trajectories, ObservationNormalizer, TransitionDataset, ChunkDataset
from src.robot_il.models.mlp_bc import MLPBC
from src.robot_il.models.chunk_bc import ChunkBC
from src.robot_il.training.trainer import train, save_checkpoint
from src.robot_il.utils.seeding import seed_everything, choose_device

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",default="data/lift.hdf5");
    p.add_argument("--model",choices=["bc","chunk"],default="bc")
    p.add_argument("--horizon",type=int,default=4); p.add_argument("--seed",type=int,default=1);
    p.add_argument("--epochs",type=int,default=100)
    p.add_argument("--device",default="auto"); p.add_argument("--output",default="results/run")
    a=p.parse_args(); seed_everything(a.seed); device=choose_device(a.device)
    trajectories=load_trajectories(a.data); train_ids,val_ids,test_ids=split_trajectories(trajectories,a.seed)
    norm=ObservationNormalizer().fit(trajectories[k].obs for k in train_ids)
    train_ds=TransitionDataset(trajectories,train_ids,norm) if a.model=="bc" else ChunkDataset(trajectories,train_ids,a.horizon,norm)
    val_ds=TransitionDataset(trajectories,val_ids,norm) if a.model=="bc" else ChunkDataset(trajectories,val_ids,a.horizon,norm)
    od,ad=next(iter(trajectories.values())).obs.shape[1],next(iter(trajectories.values())).actions.shape[1]
    model=MLPBC(od,ad) if a.model=="bc" else ChunkBC(od,ad,a.horizon)
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    result=train(model,train_ds,val_ds,device=device,epochs=a.epochs,checkpoint=out/"best.pt",seed=a.seed)
    metadata={"model":a.model,"horizon":a.horizon if a.model=="chunk" else 1,"seed":a.seed,"device":str(device),
              "python":platform.python_version(),"torch":torch.__version__,
              "demo_counts":{"train":len(train_ids),"val":len(val_ids),
                              "test":len(test_ids)},"trajectory_ids":{"train":train_ids,"val":val_ids,"test":test_ids}}
    save_checkpoint(out/"checkpoint.pt",model,norm,metadata)
    (out/"metrics.json").write_text(json.dumps({"metadata":metadata,**result},indent=2))
    print(json.dumps({"best_val_mse":result["best_val_mse"],"device":str(device),"output":str(out)}))
if __name__=="__main__":main()
