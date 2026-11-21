from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _flatten_obs(group: h5py.Group, keys: list[str] | None = None) -> np.ndarray:
    names = keys or sorted(group.keys())
    parts = [np.asarray(group[k], dtype=np.float32).reshape(len(group[k]), -1) for k in names]
    if not parts:
        raise ValueError("No observation arrays found")
    return np.concatenate(parts, axis=1)


@dataclass
class Trajectory:
    obs: np.ndarray
    actions: np.ndarray
    success: bool | None = None


def load_trajectories(path: str | Path, obs_keys: list[str] | None = None) -> dict[str, Trajectory]:
    """Load robomimic HDF5 demo group, preserving trajectory boundaries."""
    result = {}
    with h5py.File(path, "r") as f:
        demos = f["data"] if "data" in f else f
        for name in sorted(k for k in demos.keys() if k.startswith("demo_")):
            ep = demos[name]
            obs_group = ep["obs"]
            obs = _flatten_obs(obs_group, obs_keys)
            actions = np.asarray(ep["actions"], dtype=np.float32)
            # robomimic commonly stores T actions and T+1 observations.
            obs = obs[:len(actions)]
            success = None
            if "success" in ep.attrs:
                success = bool(ep.attrs["success"])
            elif "successful" in ep.attrs:
                success = bool(ep.attrs["successful"])
            result[name] = Trajectory(obs, actions, success)
    if not result:
        raise ValueError(f"No demo_* trajectories found in {path}")
    dims = {(t.obs.shape[1], t.actions.shape[1]) for t in result.values()}
    if len(dims) != 1:
        raise ValueError(f"Inconsistent dimensions across trajectories: {dims}")
    return result


def split_trajectories(trajectories: dict[str, Trajectory], seed: int = 0,
                       fractions=(0.8, 0.1, 0.1)) -> tuple[list[str], list[str], list[str]]:
    if len(fractions) != 3 or not np.isclose(sum(fractions), 1):
        raise ValueError("fractions must contain three values summing to 1")
    ids = np.array(sorted(trajectories))
    if len(ids) < 3:
        raise ValueError("At least three trajectories are required for train/val/test splits")
    np.random.default_rng(seed).shuffle(ids)
    n = len(ids)
    n_train = max(1, int(n * fractions[0]))
    n_val = max(1, int(n * fractions[1]))
    if n_train + n_val >= n:
        n_train, n_val = n - 2, 1
    return list(ids[:n_train]), list(ids[n_train:n_train+n_val]), list(ids[n_train+n_val:])


class ObservationNormalizer:
    def fit(self, arrays: Iterable[np.ndarray]) -> "ObservationNormalizer":
        x = np.concatenate(list(arrays), axis=0).astype(np.float64)
        self.mean = x.mean(axis=0).astype(np.float32)
        self.std = x.std(axis=0).astype(np.float32)
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, x):
        return (np.asarray(x, dtype=np.float32) - self.mean) / self.std

    def state_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state_dict(cls, state):
        obj = cls()
        obj.mean = np.asarray(state["mean"], np.float32)
        obj.std = np.asarray(state["std"], np.float32)
        return obj


class TransitionDataset(Dataset):
    def __init__(self, trajectories: dict[str, Trajectory], ids: list[str], normalizer=None):
        self.items = []
        for tid in ids:
            t = trajectories[tid]
            obs = normalizer.transform(t.obs) if normalizer else t.obs
            self.items.append((torch.from_numpy(obs), torch.from_numpy(t.actions)))
        self.index = [(i, j) for i, (_, a) in enumerate(self.items) for j in range(len(a))]

    def __len__(self): return len(self.index)

    def __getitem__(self, idx):
        i, j = self.index[idx]
        obs, actions = self.items[i]
        return obs[j], actions[j]


class ChunkDataset(Dataset):
    """Right-padded future action sequences; mask is 1 for valid targets."""
    def __init__(self, trajectories, ids, horizon, normalizer=None):
        if horizon < 1: raise ValueError("horizon must be positive")
        self.samples = []
        for tid in ids:
            t = trajectories[tid]
            obs = normalizer.transform(t.obs) if normalizer else t.obs
            for start in range(len(t.actions)):
                stop = min(start + horizon, len(t.actions))
                y = np.zeros((horizon, t.actions.shape[1]), np.float32)
                mask = np.zeros(horizon, np.float32)
                y[:stop-start] = t.actions[start:stop]
                mask[:stop-start] = 1
                self.samples.append((obs[start], y, mask))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        o, a, m = self.samples[idx]
        return torch.from_numpy(o), torch.from_numpy(a), torch.from_numpy(m)
