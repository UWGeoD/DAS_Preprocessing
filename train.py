"""
Usage:
    python train.py --task denoising
    python train.py --task detection
    python train.py --task weight
    python train.py --task denoising --config configs/denoising.yaml --epochs 50
"""

import argparse
import copy
import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, random_split

from dataset import DASSampleDataset, DASCountDataset, DASWeightDataset, IDX_TO_TYPE
from models.detection_transformer import DASCountTransformer
from models.unet import UNet
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


# ---------------------------------------------------------------------------
# Denoising
# ---------------------------------------------------------------------------

def train_denoising(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = cfg["data_dir"]
    df = pd.read_csv(os.path.join(data_dir, cfg["labels_csv"]))
    df_pos = df[df["count"] > 0].sample(frac=1.0, random_state=42).reset_index(drop=True)

    n_train = min(cfg.get("n_train", 40), len(df_pos))
    n_remaining = len(df_pos) - n_train
    n_val = n_remaining // 2
    n_test = n_remaining - n_val

    train_df = df_pos.iloc[:n_train]
    val_df   = df_pos.iloc[n_train:n_train + n_val]
    test_df  = df_pos.iloc[n_train + n_val:]
    print(f"Split — Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    steps = [(s["name"], {k: v for k, v in s.items() if k != "name"})
             for s in cfg["steps"]]
    pp = make_preprocess(steps=steps, dx=cfg.get("dx"), dt=cfg.get("dt"))

    fs = cfg["fs"]
    train_ds = DASSampleDataset(train_df, preprocess=pp, fs_das=fs)
    val_ds   = DASSampleDataset(val_df,   preprocess=pp, fs_das=fs)

    batch = cfg.get("batch_size", 2)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False)

    model     = UNet(in_channels=1, out_channels=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    best_val, best_state = float("inf"), None

    for epoch in range(cfg["epochs"]):
        model.train()
        train_loss = 0.0
        for raw_b, clean_b in train_loader:
            raw_b, clean_b = raw_b.to(device), clean_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(raw_b), clean_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * raw_b.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for raw_b, clean_b in val_loader:
                raw_b, clean_b = raw_b.to(device), clean_b.to(device)
                val_loss += criterion(model(raw_b), clean_b).item() * raw_b.size(0)
        val_loss /= len(val_ds)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch+1}/{cfg['epochs']} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Best val loss: {best_val:.6f}")

    save_dir = cfg.get("save_dir", "results/denoising")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
    print(f"Saved → {save_dir}/best_model.pt")

    if cfg.get("save_denoised", True):
        _run_denoising_inference(model, device, data_dir, fs)


def _run_denoising_inference(model, device, data_dir, fs):
    out_dir = os.path.join(data_dir, "denoised")
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(data_dir, "raw", "sample_*.npy")))
    print(f"Saving denoised outputs for {len(files)} samples → {out_dir}/")
    model.eval()
    with torch.no_grad():
        for fpath in files:
            raw = np.load(fpath).astype(np.float32)
            raw_t = torch.from_numpy(raw).unsqueeze(0).unsqueeze(0).to(device)
            pred = model(raw_t).squeeze().cpu().numpy()
            np.save(os.path.join(out_dir, f"denoised_{os.path.basename(fpath)}"), pred)
    print("Done.")


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

    data_dir = cfg["data_dir"]
    df = pd.read_csv(os.path.join(data_dir, cfg["labels_csv"]))
    df["data_path"] = df["sample_id"].apply(
        lambda x: os.path.join(data_dir, "denoised", f"denoised_sample_{str(x).zfill(6)}.npy")
    )

    dataset = DASCountDataset(df)
    total = len(dataset)
    n_train = int(0.7 * total)
    n_val   = int(0.15 * total)
    n_test  = total - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42)
    )
    print(f"Split — Train: {n_train}  Val: {n_val}  Test: {n_test}")

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

        if (epoch + 1) % 5 == 0 or epoch == cfg["epochs"] - 1:
            print(f"Epoch {epoch+1}/{cfg['epochs']} | "
                  f"Train: {running/len(train_ds):.4f} | Val: {val_loss/len(val_ds):.4f}")

    save_dir = cfg.get("save_dir", "results/detection")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
    print(f"Saved → {save_dir}/best_model.pt")

    _eval_detection(model, device, test_loader)


def _eval_detection(model, device, test_loader):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score

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

    data_dir = cfg["data_dir"]
    df = pd.read_csv(os.path.join(data_dir, cfg["labels_csv"]))
    df["data_path"] = df["sample_id"].apply(
        lambda x: os.path.join(data_dir, "denoised", f"denoised_sample_{str(x).zfill(6)}.npy")
    )

    dataset = DASWeightDataset(df)
    total = len(dataset)
    n_train = int(0.7 * total)
    n_val   = int(0.15 * total)
    n_test  = total - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42)
    )
    print(f"Split — Train: {n_train}  Val: {n_val}  Test: {n_test}")

    batch = cfg.get("batch_size", 4)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch, shuffle=False)

    spatial_ch = train_ds[0][0].shape[1]
    model     = DASWeightCNN(spatial_channels=spatial_ch).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

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

        if (epoch + 1) % 10 == 0 or epoch == cfg["epochs"] - 1:
            print(f"Epoch {epoch+1}/{cfg['epochs']} | "
                  f"Train: {running/len(train_ds):.4f} | Val: {val_loss/len(val_ds):.4f}")

    save_dir = cfg.get("save_dir", "results/weight")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
    print(f"Saved → {save_dir}/best_model.pt")

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
