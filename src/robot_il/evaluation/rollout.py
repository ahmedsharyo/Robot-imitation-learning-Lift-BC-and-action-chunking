"""Optional robosuite Lift rollout adapter. Requires `pip install .[sim]`."""
import time
import numpy as np


def execute_chunk(env, actions, execute_horizon):
    if execute_horizon < 1 or execute_horizon > len(actions):
        raise ValueError("execution horizon must be in [1, predicted chunk length]")
    results = []
    for action in actions[:execute_horizon]:
        results.append(env.step(np.asarray(action, dtype=np.float32)))
    return results


def make_lift_env(render=False, seed=None):
    try:
        import robosuite as suite
    except Exception as exc:
        raise RuntimeError("robosuite/MuJoCo unavailable; install the sim extra") from exc
    env = suite.make("Lift", robots="Panda", use_camera_obs=False,
                     use_object_obs=True, has_renderer=render, has_offscreen_renderer=False,
                     ignore_done=True)
    if seed is not None and hasattr(env, "seed"):
        env.seed(seed)
    return env


def rollout(env, policy, normalizer, obs_fn, *, horizon=1, execute_horizon=1,
            max_steps=500, action_low=-1.0, action_high=1.0):
    if execute_horizon > horizon: raise ValueError("execute_horizon cannot exceed horizon")
    obs = env.reset(); done = False; steps = 0; success = False; latencies = []
    while steps < max_steps and not done:
        vector = normalizer.transform(obs_fn(obs))
        import torch
        started = time.perf_counter()
        with torch.no_grad():
            pred = policy(torch.as_tensor(vector, dtype=torch.float32).unsqueeze(0)).cpu().numpy()[0]
        latencies.append(time.perf_counter()-started)
        acts = pred if pred.ndim == 2 else pred[None, :]
        for action in acts[:execute_horizon]:
            obs, reward, done, info = env.step(np.clip(action, action_low, action_high))
            steps += 1
            success = success or bool(info.get("success", False))
            if done or steps >= max_steps: break
    return {"success": success, "steps": steps, "inference_latency_seconds": float(np.mean(latencies)) if latencies else None}
