import torch


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); chunked = model.__class__.__name__ == "ChunkBC"
    sum_sq = 0.0; count = 0
    for batch in loader:
        obs, target = batch[0].to(device), batch[1].to(device)
        pred = model(obs)
        if chunked:
            mask = batch[2].to(device).unsqueeze(-1).expand_as(pred)
            sum_sq += ((pred-target).square()*mask).sum().item(); count += mask.sum().item()
        else:
            sum_sq += (pred-target).square().sum().item(); count += target.numel()
    return {"mse": sum_sq / max(count, 1), "elements": count}
