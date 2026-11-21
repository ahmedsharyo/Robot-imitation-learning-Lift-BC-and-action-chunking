import torch


def masked_mse(pred, target, mask):
    if pred.shape != target.shape or pred.shape[:-1] != mask.shape:
        raise ValueError("pred/target/mask shapes are incompatible")
    weights = mask.unsqueeze(-1).expand_as(pred)
    denom = weights.sum().clamp_min(1.0)
    return ((pred - target).square() * weights).sum() / denom
