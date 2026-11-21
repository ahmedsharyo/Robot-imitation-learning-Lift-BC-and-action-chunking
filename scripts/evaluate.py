#!/usr/bin/env python3
import argparse, json
from torch.utils.data import DataLoader
from robot_il.data.dataset import load_trajectories, split_trajectories, ObservationNormalizer, TransitionDataset, ChunkDataset
from robot_il.models.mlp_bc import MLPBC
from robot_il.models.chunk_bc import ChunkBC
from robot_il.training.trainer import load_checkpoint
from robot_il.evaluation.offline import evaluate
from robot_il.utils.seeding import choose_device

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",default="data/lift.hdf5"); p.add_argument("--checkpoint",required=True); p.add_argument("--device",default="auto")
    a=p.parse_args(); state=torch_load=None
    import torch
    raw=torch.load(a.checkpoint,map_location="cpu",weights_only=False); meta=raw["metadata"]
    tr=load_trajectories(a.data); ids=meta["trajectory_ids"]["test"]
    norm=ObservationNormalizer.from_state_dict(raw["normalizer"])
    od,ad=next(iter(tr.values())).obs.shape[1],next(iter(tr.values())).actions.shape[1]
    if meta["model"]=="chunk": model=ChunkBC(od,ad,meta["horizon"]); ds=ChunkDataset(tr,ids,meta["horizon"],norm)
    else: model=MLPBC(od,ad); ds=TransitionDataset(tr,ids,norm)
    device=choose_device(a.device); load_checkpoint(a.checkpoint,model,device); model.to(device)
    print(json.dumps({"offline_test":evaluate(model,DataLoader(ds,batch_size=512),device),"closed_loop":"NOT EVALUATED by this command"},indent=2))
if __name__=="__main__":main()
