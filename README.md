# Robot imitation learning: Lift BC and action chunking

This repository implements low dimensional behavior cloning for the robomimic Lift demonstration dataset. It contains an explicit MLP one-step policy, an MLP action-chunk policy, trajectory-isolated data splits, masked sequence loss, checkpoints, offline held-out evaluation, and an optional robosuite rollout adapter.

## Environment inspected

At project setup, the available interpreter is Python 3.10.0 (Apple Silicon/macOS per the machine description). None of PyTorch, NumPy, h5py, robomimic, robosuite, MuJoCo, matplotlib, pandas, PyYAML, or pytest are installed in that interpreter. Therefore dependency compatibility, MPS kernel coverage, dataset download, training on real demos, and simulation launch have **not been empirically verified on this machine**. This project supports CPU first and selects MPS when PyTorch reports it available. Retry an unsupported MPS operation on CPU if needed. We make no rollout or performance claims.

The pinned core uses Python >=3.10, NumPy 1.26.4, PyTorch 2.3.1, and h5py 3.11.0; the optional simulation extra pins robomimic 0.3.0, robosuite 1.4.1, and MuJoCo 3.1.6. These pins are a reproducible starting point, not a compatibility claim: validate them in a clean environment on your specific macOS/Python version. If native simulation fails, use Linux with the same lock inputs/container or a documented cloud Linux VM, and keep simulator metrics separate from offline MSE.

## Install

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
# Optional simulator integration:
python -m pip install -e '.[sim,dev]'
```

For no-GPU systems, use `--device cpu`; default `--device auto` selects MPS if available and otherwise CPU. PyTorch MPS support depends on the installed wheel and macOS; no CUDA is assumed.

## Dataset

The expected input is a robomimic HDF5 file with a `data/demo_*` layout, `actions`, and low-dimensional `obs` datasets. HDF5 episode boundaries are preserved. The loader concatenates sorted observation keys by default and aligns `T+1` observation sequences to the `T` action transitions by dropping the terminal observation. Use `obs_keys` explicitly if a particular fixed key ordering is required.

```bash
python scripts/download_data.py --output data/lift.hdf5 --dataset lift/mg
```

The download helper copies a locally cached official registry dataset. If it is not present, follow the dataset download instructions and official download utility in the [robomimic repository](https://github.com/ARISE-Initiative/robomimic), then rerun the helper or place the HDF5 file at `data/lift.hdf5`. Check source license/terms and available metadata before redistribution. `load_trajectories` exposes any episode success attribute it can find but does not infer success from the final state.

## Train and evaluate

```bash
python scripts/train.py --data data/lift.hdf5 --model bc --seed 1 --device auto --output results/bc_s1
python scripts/train.py --data data/lift.hdf5 --model chunk --horizon 8 --seed 1 --device auto --output results/chunk8_s1
python scripts/evaluate.py --data data/lift.hdf5 --checkpoint results/bc_s1/checkpoint.pt
python scripts/plot_results.py results/bc_s1/metrics.json results/chunk8_s1/metrics.json
```

The split is made by trajectory with a seeded 80/10/10 split and saved in checkpoint metadata; normalization statistics are fit on training demonstrations only. Chunk targets stop at trajectory ends and are right-padded with masks. `H` (prediction horizon) is distinct from `K` (execution horizon): use `rollout(..., horizon=H, execute_horizon=K)`. `K=H` is open loop for that chunk; `K=1` replans every step.

The offline evaluator reports held-out action MSE, which is not task success. `src.robot_il.evaluation.rollout` provides a robosuite environment/step adapter, but observation conversion is environment-specific and must be supplied as `obs_fn`; validate action semantics and limits for the chosen controller before interpreting success. The rollout helper does not currently aggregate multi-seed success statistics or write videos.

## Tests and experiment scope

```bash
pytest
```

Synthetic unit tests cover HDF5 extraction, disjoint deterministic splits, normalization, padding/masks, model shapes and gradients, checkpoint roundtrip, and chunk execution. Current Python environment lacks pytest and the ML dependencies, so tests were not run here. No Lift training, comparison plots, rollout videos, or closed-loop results are present. Generate them only after running the commands; do not treat example or absent results as experimental evidence.

For primary comparisons, repeat at least three seeds, keep held-out trajectory IDs fixed across model/data-size conditions, and record each run's device, versions, training time, validation/test MSE, and rollout configuration. Dataset-size studies should use nested subsets of training trajectory IDs. Distribution-shift perturbations must remain within valid simulator ranges and use separate evaluation labels. Confidence intervals for binary rollout success can use the Wilson interval utility.

## Layout

`src/src.robot_il/data` handles HDF5 trajectories and sequence sampling; `models` contains the two policies; `training` implements masked loss and early-stopped training; `evaluation` separates offline metrics from optional simulation; `scripts` are command-line entry points; `reports` and `results` hold generated artifacts.
