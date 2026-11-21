from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from .losses import masked_mse


def train(model, train_ds, val_ds, *, device, epochs=100, batch_size=256,
          lr=1e-3, patience=12, grad_clip=1.0, checkpoint=None, seed=0):
    torch.manual_seed(seed)
    model.to(device)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    chunked = model.__class__.__name__ == "ChunkBC"
    best, stale, history = float("inf"), 0, []
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train(); total = count = 0
        for batch in train_loader:
            obs, actions = batch[0].to(device), batch[1].to(device)
            mask = batch[2].to(device) if chunked else None
            optimizer.zero_grad(set_to_none=True)
            pred = model(obs)
            loss = masked_mse(pred, actions, mask) if chunked else torch.nn.functional.mse_loss(pred, actions)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip); optimizer.step()
            total += loss.item() * len(obs); count += len(obs)
        train_loss = total / max(count, 1)
        model.eval(); total = count = 0
        with torch.no_grad():
            for batch in val_loader:
                obs, actions = batch[0].to(device), batch[1].to(device)
                mask = batch[2].to(device) if chunked else None
                pred = model(obs)
                loss = masked_mse(pred, actions, mask) if chunked else torch.nn.functional.mse_loss(pred, actions)
                total += loss.item() * len(obs); count += len(obs)
        val_loss = total / max(count, 1)
        history.append({"epoch": epoch + 1, "train_mse": train_loss, "val_mse": val_loss})
        if val_loss < best:
            best, stale = val_loss, 0
            if checkpoint:
                Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "epoch": epoch + 1, "val_mse": best}, checkpoint)
        else:
            stale += 1
            if stale >= patience: break
    if checkpoint and Path(checkpoint).exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
    return {"history": history, "best_val_mse": best, "epochs": len(history),
            "training_seconds": time.perf_counter() - started}


def save_checkpoint(path, model, normalizer, metadata):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "normalizer": normalizer.state_dict(), "metadata": metadata}, path)


def load_checkpoint(path, model, device="cpu"):
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    return state
