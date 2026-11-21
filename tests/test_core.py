import h5py
import numpy as np
import pytest
import torch
from robot_il.data.dataset import (Trajectory, load_trajectories, split_trajectories,
    ObservationNormalizer, TransitionDataset, ChunkDataset)
from robot_il.models.mlp_bc import MLPBC
from robot_il.models.chunk_bc import ChunkBC
from robot_il.training.losses import masked_mse
from robot_il.training.trainer import save_checkpoint, load_checkpoint
from robot_il.evaluation.rollout import execute_chunk


def synthetic(n=5):
    return {f"demo_{i}":Trajectory(np.full((i+2,3),i,dtype=np.float32),np.full((i+2,2),i,dtype=np.float32)) for i in range(n)}


def test_hdf5_loading_and_alignment(tmp_path):
    fpath=tmp_path/"x.hdf5"
    with h5py.File(fpath,"w") as f:
        d=f.create_group("data").create_group("demo_0"); d.create_dataset("actions",data=np.ones((3,2)))
        o=d.create_group("obs"); o.create_dataset("a",data=np.ones((4,2))); o.create_dataset("b",data=np.ones((4,1))*2)
    loaded=load_trajectories(fpath)
    assert loaded["demo_0"].obs.shape==(3,3) and loaded["demo_0"].actions.shape==(3,2)


def test_trajectory_split_disjoint_and_repeatable():
    x=synthetic(); a=split_trajectories(x,7); b=split_trajectories(x,7)
    assert a==b and not (set(a[0])&set(a[1]) or set(a[1])&set(a[2]) or set(a[0])&set(a[2]))


def test_normalizer_fit_and_roundtrip():
    n=ObservationNormalizer().fit([np.array([[1,5],[3,5]],np.float32)])
    np.testing.assert_allclose(n.transform([[2,5]]),[[0,0]])
    assert ObservationNormalizer.from_state_dict(n.state_dict()).mean.tolist()==n.mean.tolist()


def test_chunk_masks_and_no_crossing():
    ds=ChunkDataset(synthetic(1),["demo_0"],4)
    _,a,m=ds[-1]
    assert m.tolist()==[1,0,0,0] and a[0].tolist()==[0,0]
    assert len(ds)==2


def test_models_loss_and_backward():
    x=torch.randn(4,3); bc=MLPBC(3,2); assert bc(x).shape==(4,2)
    chunk=ChunkBC(3,2,4); y=chunk(x); assert y.shape==(4,4,2)
    target=torch.randn_like(y); mask=torch.tensor([[1.,1,0,0]]*4)
    loss=masked_mse(y,target,mask); loss.backward(); assert torch.isfinite(loss)
    expected=((y.detach()[:,:2]-target[:,:2])**2).mean()
    assert torch.allclose(loss.detach(),expected)


def test_checkpoint_roundtrip(tmp_path):
    model=MLPBC(3,2); norm=ObservationNormalizer().fit([np.ones((2,3),np.float32)])
    path=tmp_path/"m.pt"; save_checkpoint(path,model,norm,{"x":1})
    restored=MLPBC(3,2); state=load_checkpoint(path,restored)
    assert state["metadata"]=={"x":1}
    for a,b in zip(model.parameters(),restored.parameters()): assert torch.equal(a,b)


def test_chunk_execution_bounds():
    class E:
        def __init__(self): self.n=0
        def step(self,a): self.n+=1; return self.n
    e=E(); assert execute_chunk(e,np.zeros((4,2)),2)==[1,2]
    with pytest.raises(ValueError): execute_chunk(e,np.zeros((4,2)),5)
