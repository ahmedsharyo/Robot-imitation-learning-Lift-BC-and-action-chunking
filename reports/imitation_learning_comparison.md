# Imitation learning comparison: Lift

## Executive Summary

Implemented trajectory-preserving HDF5 loading, train-only observation normalization, one-step MLP behavior cloning, masked MLP action chunks, checkpointed training, held-out offline MSE evaluation, and an optional robosuite adapter. No real-dataset training or rollout was run: the inspected Python 3.10 environment has none of the required ML/simulation packages installed and no dataset was present. Results below are intentionally marked not evaluated.

## Methodology

The project expects robomimic Lift low-dimensional HDF5 demonstrations. Splits are made by whole demonstration, never timestep. Inputs concatenate selected observation arrays; action trajectories remain separate. Chunk targets have configurable horizon and are padded only at trajectory ends with a validity mask. Both models use two 256-unit ReLU hidden layers and Adam/MSE training with gradient clipping and early stopping. Evaluation separates offline action error from closed-loop success.

## Experimental Results

| Experiment | Status | Result |
|---|---|---|
| Single-action BC vs H=4/8/16 | Not evaluated | Dependencies and dataset unavailable |
| Demonstration count curve | Not evaluated | No trained runs |
| Distribution shift | Not evaluated | Simulator unavailable |
| Closed-loop Lift success | Not evaluated | No rollout executed |

No plots or performance values are reported because none were measured.

## Offline vs Closed-Loop Performance

Held-out MSE measures action agreement on expert-distributed observations. It does not show that the policy reaches those observations when acting. A small action error can move the robot into a state missing from the demonstrations; subsequent predictions then extrapolate, and errors may accumulate. Only closed-loop rollouts can establish manipulation success.

## Action Chunking Analysis

Chunk prediction provides temporally coordinated targets and can reduce replanning overhead. Longer horizons also condition more future actions on the current observation; open-loop execution can apply stale actions after contact or object motion. Receding-horizon execution predicts H actions but applies K≤H before observing again, allowing feedback at a chosen cadence. No empirical benefit or drawback was measured here.

## Demonstration Efficiency

Not evaluated. The experiment protocol should use nested, trajectory-level subsets (for example 10/25/50/100/all, clipped to available training demonstrations), fixed validation/test IDs, and repeated seeds. Report offline MSE and rollout success separately.

## Distribution Shift

Behavior cloning learns under expert state distribution p_E(s), while acting induces p_pi(s). The distributions can diverge as action errors alter subsequent states. Perturbed initial conditions, observation noise, and controlled action perturbations are planned evaluation axes, but none were run; no causal conclusion is warranted.

## Failure Analysis

No rollouts were recorded, so there are no observed failures to classify. Once available, retain per-step states/actions and evidence for misses, early/late grasp closure, slips, oscillation, recovery failures, and stale chunk execution. Do not label hypothetical examples as observed failures.

## Limitations

The dependency pins have not been installed on the target Mac, and native robosuite/MuJoCo behavior on Apple Silicon has not been checked in this environment. Observation key selection and action semantics depend on the dataset/controller configuration. Synthetic tests are authored but unrun. The implementation does not yet provide automated multi-seed experiment orchestration, rollout video recording, detailed failure classification, or real-data plots.

## Future Work

First validate dataset metadata and simulator/action compatibility on macOS or Linux, then run at least three seeds and 50 rollouts/configuration where compute permits. Add robust comparison orchestration and saved rollout evidence. Later extensions can investigate DAgger, Diffusion Policy, ACT-style architectures, visual imitation, offline RL, and multitask manipulation.
