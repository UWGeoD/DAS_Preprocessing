"""
Usage:
    python train.py --task denoising
    python train.py --task detection
    python train.py --task weight
    python train.py --task denoising --config configs/denoising.yaml --epochs 50
"""

import warnings
warnings.filterwarnings("ignore", message="Pandas requires version", category=UserWarning)

import argparse
import copy
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Subset

from dataset import DASSampleDataset, DASCountDataset, DASWeightDataset, IDX_TO_TYPE
from models.detection_transformer import DASCountTransformer
from models.unet_v2 import UNetV2
from models.weight_cnn import DASWeightCNN
from preprocessing import make_preprocess


def load_config(task, config_path=None):
    path = config_path or f"configs/{task}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg, args):
    """Let CLI flags override YAML values."""
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.data_dir is not None:
        cfg["data_dir"] = args.data_dir
    return cfg


def _save_splits(save_dir, train_ids, val_ids, test_ids):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "splits.json")
    with open(path, "w") as f:
        json.dump({"train": train_ids, "val": val_ids, "test": test_ids}, f, indent=2)
    print(f"Splits saved → {path}")


def _save_loss_plot(history, save_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history["train_loss"], label="Train")
    ax.plot(epochs, history["val_loss"],   label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(save_dir, "losses.png")
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"Loss plot → {out}")


def _correlation_loss(pred, target, eps=1e-8):
    """1 − Pearson correlation, averaged over the batch.
    A constant prediction gives correlation=0 → loss=1, so the model
    cannot minimise this by predicting the target mean everywhere."""
    p = pred.view(pred.size(0), -1)
    t = target.view(target.size(0), -1)
    p = p - p.mean(dim=1, keepdim=True)
    t = t - t.mean(dim=1, keepdim=True)
    corr = (p * t).sum(dim=1) / (p.norm(dim=1) * t.norm(dim=1) + eps)
    return 1.0 - corr.mean()


# ---------------------------------------------------------------------------
# Denoising
# ---------------------------------------------------------------------------

def train_denoising(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    save_dir = cfg.get("save_dir", "results/denoising")
    data_dir = cfg["data_dir"]
    df = pd.read_csv(os.path.join(data_dir, cfg["labels_csv"]))
    df_pos = df[df["count"] > 0].sort_values("sample_id").reset_index(drop=True)

    split_mode = cfg.get("split_mode", "temporal")
    overlap_gap = cfg.get("overlap_gap", 0)
    n_total = len(df_pos)
    if "train_frac" in cfg:
        n_train = min(int(n_total * cfg["train_frac"]), n_total)
    else:
        n_train = min(cfg.get("n_train", 40), n_total)

    if split_mode == "random":
        rng = np.random.default_rng(cfg.get("split_seed", 42))
        idx = rng.permutation(n_total)
        train_idx = sorted(idx[:n_train].tolist())
        remaining = idx[n_train:]
        n_val = len(remaining) // 2
        val_idx  = sorted(remaining[:n_val].tolist())
        test_idx = sorted(remaining[n_val:].tolist())
        train_df = df_pos.iloc[train_idx].copy()
        val_df   = df_pos.iloc[val_idx]
        test_df  = df_pos.iloc[test_idx]
        print(f"Split mode: random (seed={cfg.get('split_seed', 42)})")
    else:
        n_pool = n_total - n_train - 2 * overlap_gap
        n_val  = max(0, n_pool // 2)
        train_df = df_pos.iloc[:n_train].copy()
        val_df   = df_pos.iloc[n_train + overlap_gap : n_train + overlap_gap + n_val]
        test_df  = df_pos.iloc[n_train + overlap_gap + n_val + overlap_gap :]
        print(f"Split mode: temporal (overlap_gap={overlap_gap})")

    # Optionally add background (count == 0) windows to training only.
    if cfg.get("include_background", False):
        df_bg = df[df["count"] == 0].sort_values("sample_id").reset_index(drop=True)
        train_df = pd.concat([train_df, df_bg], ignore_index=True)
        print(f"  + {len(df_bg)} background samples → training total: {len(train_df)}")

    print(f"Split — Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    _save_splits(
        save_dir,
        train_ids=train_df["sample_id"].tolist(),
        val_ids=val_df["sample_id"].tolist(),
        test_ids=test_df["sample_id"].tolist(),
    )

    steps = [(s["name"], {k: v for k, v in s.items() if k != "name"})
             for s in cfg["steps"]]
    pp = make_preprocess(steps=steps, dx=cfg.get("dx"), dt=cfg.get("dt"))

    fs = cfg["fs"]
    train_ds = DASSampleDataset(train_df, preprocess=pp, fs_das=fs, augment=True)
    val_ds   = DASSampleDataset(val_df,   preprocess=pp, fs_das=fs, augment=False)

    batch = cfg.get("batch_size", 8)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False)

    model     = UNetV2(in_channels=1, out_channels=1).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        patience=cfg.get("lr_patience", 20),
        factor=0.5,
        min_lr=1e-6,
    )
    corr_w = cfg.get("correlation_weight", 0.0)

    best_val, best_state = float("inf"), None
    log_interval = cfg.get("log_interval", 10)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(cfg["epochs"]):
        model.train()
        train_loss = 0.0
        for raw_b, clean_b in train_loader:
            raw_b, clean_b = raw_b.to(device), clean_b.to(device)
            optimizer.zero_grad()
            pred = model(raw_b)
            loss = criterion(pred, clean_b) + corr_w * _correlation_loss(pred, clean_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * raw_b.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for raw_b, clean_b in val_loader:
                raw_b, clean_b = raw_b.to(device), clean_b.to(device)
                pred = model(raw_b)
                val_loss += (
                    criterion(pred, clean_b) + corr_w * _correlation_loss(pred, clean_b)
                ).item() * raw_b.size(0)
        val_loss /= len(val_ds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % log_interval == 0 or epoch == cfg["epochs"] - 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1}/{cfg['epochs']} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr_now:.2e}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Best val loss: {best_val:.6f}")

    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
    print(f"Saved → {save_dir}/best_model.pt")

    losses_path = os.path.join(save_dir, "losses.json")
    with open(losses_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Losses saved → {losses_path}")
    _save_loss_plot(history, save_dir)
    print("Run `python predict.py --task denoising` to generate data/denoised/.")


# ---------------------------------------------------------------------------
# Count / type prediction
# ---------------------------------------------------------------------------

def _enforce_business_logic(raw_count, type_logits):
    """count=0 → background, count>1 → mixed, count=1 → single-vehicle type."""
    count = int(round(raw_count))
    if count == 0:
        return count, "background"
    if count > 1:
        return count, "mixed"
    best = int(np.argmax(type_logits[2:6])) + 2
    return count, IDX_TO_TYPE[best]


def train_detection(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    save_dir = cfg.get("save_dir", "results/detection")
    data_dir = cfg["data_dir"]
    df = pd.read_csv(os.path.join(data_dir, cfg["labels_csv"]))
    df["data_path"] = df["sample_id"].apply(
        lambda x: os.path.join(data_dir, "denoised", f"denoised_sample_{str(x).zfill(6)}.npy")
    )

    df = df.sort_values("sample_id").reset_index(drop=True)
    dataset = DASCountDataset(df)
    overlap_gap = cfg.get("overlap_gap", 4)
    n_total = len(df)
    n_pool  = n_total - 2 * overlap_gap
    n_train = int(0.7 * n_pool)
    n_val   = int(0.15 * n_pool)

    train_idx = list(range(n_train))
    val_idx   = list(range(n_train + overlap_gap, n_train + overlap_gap + n_val))
    test_idx  = list(range(n_train + overlap_gap + n_val + overlap_gap, n_total))

    train_ds = Subset(dataset, train_idx)
    val_ds   = Subset(dataset, val_idx)
    test_ds  = Subset(dataset, test_idx)
    print(f"Split — Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    _save_splits(
        save_dir,
        train_ids=[f"{int(dataset.df.iloc[i]['sample_id']):06d}" for i in train_idx],
        val_ids=[f"{int(dataset.df.iloc[i]['sample_id']):06d}" for i in val_idx],
        test_ids=[f"{int(dataset.df.iloc[i]['sample_id']):06d}" for i in test_idx],
    )

    batch = cfg.get("batch_size", 8)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch, shuffle=False)

    spatial_ch = train_ds[0][0].shape[1]
    model = DASCountTransformer(
        spatial_channels=spatial_ch,
        d_model=cfg.get("d_model", 64),
        nhead=cfg.get("nhead", 4),
        num_layers=cfg.get("num_layers", 2),
    ).to(device)

    count_crit = nn.MSELoss()
    type_crit  = nn.CrossEntropyLoss()
    optimizer  = optim.Adam(model.parameters(), lr=cfg["lr"])
    w_count, w_type = cfg.get("weight_count", 1.0), cfg.get("weight_type", 0.5)

    history = {"train_loss": [], "val_loss": []}
    for epoch in range(cfg["epochs"]):
        model.train()
        running = 0.0
        for x, y_count, y_type in train_loader:
            x, y_count, y_type = x.to(device), y_count.to(device), y_type.to(device)
            optimizer.zero_grad()
            p_count, p_type = model(x)
            loss = w_count * count_crit(p_count, y_count) + w_type * type_crit(p_type, y_type)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y_count, y_type in val_loader:
                x, y_count, y_type = x.to(device), y_count.to(device), y_type.to(device)
                p_count, p_type = model(x)
                val_loss += (w_count * count_crit(p_count, y_count)
                             + w_type * type_crit(p_type, y_type)).item() * x.size(0)

        history["train_loss"].append(running / len(train_ds))
        history["val_loss"].append(val_loss / len(val_ds))
        if (epoch + 1) % 5 == 0 or epoch == cfg["epochs"] - 1:
            print(f"Epoch {epoch+1}/{cfg['epochs']} | "
                  f"Train: {running/len(train_ds):.4f} | Val: {val_loss/len(val_ds):.4f}")

    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
    print(f"Saved → {save_dir}/best_model.pt")

    losses_path = os.path.join(save_dir, "losses.json")
    with open(losses_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Losses saved → {losses_path}")
    _save_loss_plot(history, save_dir)

    _eval_detection(model, device, test_loader)


def _eval_detection(model, device, test_loader):
    from sklearn.metrics import mean_absolute_error, accuracy_score

    model.eval()
    act_counts, pred_counts, act_types, pred_types = [], [], [], []
    with torch.no_grad():
        for x, y_count, y_type in test_loader:
            p_counts, p_types = model(x.to(device))
            p_counts = p_counts.cpu().numpy().flatten()
            p_types  = p_types.cpu().numpy()
            for i in range(len(x)):
                act_counts.append(y_count[i].item())
                act_types.append(IDX_TO_TYPE[y_type[i].item()])
                c, t = _enforce_business_logic(p_counts[i], p_types[i])
                pred_counts.append(c)
                pred_types.append(t)

    act_arr  = np.array(act_counts)
    pred_arr = np.array(pred_counts)
    print("\n--- Test Metrics ---")
    print(f"Type accuracy:    {accuracy_score(act_types, pred_types)*100:.1f}%")
    print(f"Count exact match:{np.mean(act_arr == pred_arr)*100:.1f}%")
    print(f"Count off-by-one: {np.mean(np.abs(act_arr - pred_arr) <= 1)*100:.1f}%")
    print(f"Count MAE:        {mean_absolute_error(act_arr, pred_arr):.2f}")


# ---------------------------------------------------------------------------
# Weight regression
# ---------------------------------------------------------------------------

def train_weight(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    save_dir = cfg.get("save_dir", "results/weight")
    data_dir = cfg["data_dir"]
    df = pd.read_csv(os.path.join(data_dir, cfg["labels_csv"]))
    df["data_path"] = df["sample_id"].apply(
        lambda x: os.path.join(data_dir, "denoised", f"denoised_sample_{str(x).zfill(6)}.npy")
    )

    df = df.sort_values("sample_id").reset_index(drop=True)
    dataset = DASWeightDataset(df)
    overlap_gap = cfg.get("overlap_gap", 4)
    n_total = len(df)
    n_pool  = n_total - 2 * overlap_gap
    n_train = int(0.7 * n_pool)
    n_val   = int(0.15 * n_pool)

    train_idx = list(range(n_train))
    val_idx   = list(range(n_train + overlap_gap, n_train + overlap_gap + n_val))
    test_idx  = list(range(n_train + overlap_gap + n_val + overlap_gap, n_total))

    train_ds = Subset(dataset, train_idx)
    val_ds   = Subset(dataset, val_idx)
    test_ds  = Subset(dataset, test_idx)
    print(f"Split — Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    _save_splits(
        save_dir,
        train_ids=[f"{int(dataset.df.iloc[i]['sample_id']):06d}" for i in train_idx],
        val_ids=[f"{int(dataset.df.iloc[i]['sample_id']):06d}" for i in val_idx],
        test_ids=[f"{int(dataset.df.iloc[i]['sample_id']):06d}" for i in test_idx],
    )

    batch = cfg.get("batch_size", 4)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch, shuffle=False)

    spatial_ch = train_ds[0][0].shape[1]
    model     = DASWeightCNN(spatial_channels=spatial_ch).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    history = {"train_loss": [], "val_loss": []}
    for epoch in range(cfg["epochs"]):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item() * x.size(0)

        history["train_loss"].append(running / len(train_ds))
        history["val_loss"].append(val_loss / len(val_ds))
        if (epoch + 1) % 10 == 0 or epoch == cfg["epochs"] - 1:
            print(f"Epoch {epoch+1}/{cfg['epochs']} | "
                  f"Train: {running/len(train_ds):.4f} | Val: {val_loss/len(val_ds):.4f}")

    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
    print(f"Saved → {save_dir}/best_model.pt")

    losses_path = os.path.join(save_dir, "losses.json")
    with open(losses_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Losses saved → {losses_path}")
    _save_loss_plot(history, save_dir)

    _eval_weight(model, device, test_loader)


def _eval_weight(model, device, test_loader):
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    model.eval()
    actuals, preds = [], []
    with torch.no_grad():
        for x, y in test_loader:
            p = model(x.to(device)).cpu().numpy().flatten() * 1000.0
            actuals.extend((y.numpy().flatten() * 1000.0).tolist())
            preds.extend(np.clip(p, 0, None).tolist())

    act_arr  = np.array(actuals)
    pred_arr = np.array(preds)
    print("\n--- Test Metrics ---")
    print(f"MAE:  {mean_absolute_error(act_arr, pred_arr):.0f} lbs")
    print(f"RMSE: {np.sqrt(mean_squared_error(act_arr, pred_arr)):.0f} lbs")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train DAS models.")
    parser.add_argument("--task", required=True, choices=["denoising", "detection", "weight"])
    parser.add_argument("--config", default=None, help="Path to YAML config (default: configs/<task>.yaml)")
    parser.add_argument("--epochs",   type=int,   default=None, help="Override epochs")
    parser.add_argument("--lr",       type=float, default=None, help="Override learning rate")
    parser.add_argument("--data_dir", type=str,   default=None, help="Override data directory")
    args = parser.parse_args()

    cfg = load_config(args.task, args.config)
    apply_overrides(cfg, args)

    if args.task == "denoising":
        train_denoising(cfg)
    elif args.task == "detection":
        train_detection(cfg)
    elif args.task == "weight":
        train_weight(cfg)


if __name__ == "__main__":
    main()
