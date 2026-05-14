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
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Subset

from dataset import DASSampleDataset, DASCountDataset, DASWeightDataset, IDX_TO_TYPE
from models.detection_transformer import DASCountTransformer
from models.unet import UNet
from models.unet_v2 import UNetV2
from models.weight_cnn import DASWeightCNN
from preprocessing import make_preprocess


def load_config(task, config_path=None, dataset=None):
    """Load task YAML, then overlay dataset config values for sensor/data params.

    Dataset config is authoritative for: data_dir, dx, dt, fs (sensor/data-specific).
    Task YAML wins for everything else (model hyperparams, preprocessing steps, labels_csv).
    """
    from config import load_dataset_config
    path = config_path or f"configs/{task}.yaml"
    with open(path) as f:
        task_cfg = yaml.safe_load(f)

    # Merge loss-specific defaults from losses.yaml (task_cfg wins on conflict)
    losses_path = "configs/losses.yaml"
    if task == "denoising" and os.path.exists(losses_path):
        with open(losses_path) as f:
            losses_cfg = yaml.safe_load(f)
        loss_name = task_cfg.get("loss", "mse")
        if loss_name in losses_cfg:
            task_cfg = {**losses_cfg[loss_name], **task_cfg}

    ds_cfg = load_dataset_config(dataset)
    task_cfg["data_dir"] = ds_cfg["data_dir"]
    # fs_das in dataset config → fs key used by training code
    task_cfg["fs"] = ds_cfg["fs_das"]
    for key in ("dx", "dt"):
        if key in ds_cfg:
            task_cfg[key] = ds_cfg[key]
    return task_cfg


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

    def _plot(ax, keys_labels, title, hline=None):
        for key, label, color in keys_labels:
            if key in history:
                ax.plot(epochs, history[key], label=label, color=color)
        if hline is not None:
            ax.axhline(hline, ls="--", color="gray", alpha=0.5, label=f"equilibrium {hline}")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    if "train_gan" in history and "train_recon" in history:
        # adv_nce style: recon is part of the loss
        fig, axes = plt.subplots(1, 5, figsize=(22, 4))
        _plot(axes[0], [("train_recon","Train Recon","tab:blue"),
                        ("val_loss",  "Val Recon",  "tab:orange")], "Reconstruction")
        _plot(axes[1], [("train_gan", "GAN (G)", "tab:red")],
              "GAN Loss (G)", hline=0.25)
        _plot(axes[2], [("train_nce", "NCE", "tab:purple")], "NCE Loss")
        _plot(axes[3], [("train_d",   "D Loss", "tab:green")],
              "Discriminator Loss", hline=0.25)
        _plot(axes[4], [("train_loss","L_G Total","tab:blue"),
                        ("val_loss", "Val Recon", "tab:orange")], "Total L_G vs Val")
    elif "train_gan" in history:
        # gan_nce style: no recon in loss — val is the reconstruction monitor
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        _plot(axes[0], [("train_gan", "GAN (G)", "tab:red")],
              "GAN Loss (G)", hline=0.25)
        _plot(axes[1], [("train_nce", "NCE", "tab:purple")], "NCE Loss")
        _plot(axes[2], [("train_d",   "D Loss", "tab:green")],
              "Discriminator Loss", hline=0.25)
        _plot(axes[3], [("val_loss",  "Val Recon", "tab:orange")], "Val Recon (monitor)")
    elif "train_reg" in history:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        _plot(axes[0], [("train_loss", "Train Total", "tab:blue"),
                        ("val_loss",   "Val",         "tab:orange")], "Total Loss")
        _plot(axes[1], [("train_recon","Train Recon", "tab:blue"),
                        ("val_loss",   "Val",         "tab:orange")], "Reconstruction")
        _plot(axes[2], [("train_reg",  "Reg",         "tab:red")],   "Pred Regularization")
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        _plot(ax, [("train_loss","Train","tab:blue"),
                   ("val_loss",  "Val",  "tab:orange")], "Loss")

    fig.tight_layout()
    out = os.path.join(save_dir, "losses.png")
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"Loss plot → {out}")


def _pred_reg_loss(pred, reg_type="l1"):
    """Penalizes non-zero output to suppress background noise in the prediction.
    l1: sparse prior — threshold-like, preserves high-amplitude signal.
    l2: quadratic — over-penalizes signal peaks, use with very small weights only."""
    if reg_type == "l2":
        return pred.pow(2).mean()
    return pred.abs().mean()


# ---------------------------------------------------------------------------
# Denoising
# ---------------------------------------------------------------------------

def train_denoising_core(df_pos, cfg, save_dir, device=None, df_bg=None):
    """
    Core denoising training loop.

    Parameters
    ----------
    df_pos   : DataFrame of positive-count samples, already filtered and sliced.
    cfg      : config dict with training hyperparameters, 'steps', 'fs', 'dx', 'dt'.
    save_dir : directory to write splits.json (caller is responsible for saving best_model.pt).
    device   : torch.device; defaults to CUDA if available.
    df_bg    : optional background-only DataFrame appended to the train split only.

    Returns
    -------
    model      : UNetV2 (or UNet) with best-val weights loaded.
    history    : {"train_loss": [...], "val_loss": [...]}.
    train_ids  : list of sample_id values in the training split.
    val_ids    : list of sample_id values in the validation split.
    test_ids   : list of sample_id values in the test split.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_mode  = cfg.get("split_mode", "temporal")
    overlap_gap = cfg.get("overlap_gap", 0)
    n_total     = len(df_pos)

    if "train_frac" in cfg:
        n_train = min(int(n_total * cfg["train_frac"]), n_total)
    else:
        n_train = min(cfg.get("n_train", 40), n_total)

    if split_mode == "random":
        rng       = np.random.default_rng(cfg.get("split_seed", 42))
        idx       = rng.permutation(n_total)
        train_idx = sorted(idx[:n_train].tolist())
        remaining = idx[n_train:]
        n_val     = len(remaining) // 2
        val_idx   = sorted(remaining[:n_val].tolist())
        test_idx  = sorted(remaining[n_val:].tolist())
        train_df  = df_pos.iloc[train_idx].copy()
        val_df    = df_pos.iloc[val_idx]
        test_df   = df_pos.iloc[test_idx]
        print(f"Split mode: random (seed={cfg.get('split_seed', 42)})")
    else:
        n_pool   = n_total - n_train - 2 * overlap_gap
        n_val    = max(0, n_pool // 2)
        train_df = df_pos.iloc[:n_train].copy()
        val_df   = df_pos.iloc[n_train + overlap_gap : n_train + overlap_gap + n_val]
        test_df  = df_pos.iloc[n_train + overlap_gap + n_val + overlap_gap :]
        print(f"Split mode: temporal (overlap_gap={overlap_gap})")

    if len(val_df) < 5:
        print(f"  WARNING: val set has only {len(val_df)} samples — early stopping may be unreliable")

    if df_bg is not None and len(df_bg):
        train_df = pd.concat([train_df, df_bg], ignore_index=True)
        print(f"  + {len(df_bg)} background samples → training total: {len(train_df)}")

    print(f"Split — Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    train_ids = train_df["sample_id"].tolist()
    val_ids   = val_df["sample_id"].tolist()
    test_ids  = test_df["sample_id"].tolist()
    _save_splits(save_dir, train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)

    steps = [(s["name"], {k: v for k, v in s.items() if k != "name"})
             for s in cfg["steps"]]
    pp = make_preprocess(steps=steps, dx=cfg.get("dx"), dt=cfg.get("dt"))

    fs           = cfg["fs"]
    batch        = cfg.get("batch_size", 8)
    train_ds     = DASSampleDataset(train_df, preprocess=pp, fs_das=fs, augment=True)
    val_ds       = DASSampleDataset(val_df,   preprocess=pp, fs_das=fs, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False)

    model_name = cfg.get("model", "unet_v2")
    if model_name == "unet":
        model = UNet(in_channels=1, out_channels=1).to(device)
    else:
        model = UNetV2(in_channels=1, out_channels=1).to(device)
    print(f"Model: {model_name}  |  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    loss_name     = cfg.get("loss", "mse")
    recon_type    = cfg.get("recon", "mse")   # "mse" or "l1"; default mse for val monitoring
    criterion     = nn.MSELoss() if recon_type == "mse" else nn.L1Loss()
    pred_reg_w    = cfg.get("pred_reg_weight", 0.0)
    pred_reg_type = cfg.get("pred_reg_type", "l1")
    print(f"Loss: {loss_name}")

    use_gan_nce = (loss_name == "gan_nce")
    if use_gan_nce:
        from models.discriminator import PatchGAN
        from models.patchnce import PatchSampleF, PatchNCELoss as NCELoss
        d_depth         = cfg.get("d_depth", 3)
        discriminator   = PatchGAN(in_channels=1, n_layers=d_depth).to(device)
        lambda_NCE      = cfg.get("lambda_NCE", 1.0)
        nce_tau         = cfg.get("nce_tau", 0.07)
        nce_num_patches = cfg.get("nce_num_patches", 256)
        nce_layers      = cfg.get("nce_layers", list(range(len(model.feat_channels))))
        feat_ch_nce     = [model.feat_channels[i] for i in nce_layers]
        patch_sampler   = PatchSampleF(feat_ch_nce, embed_dim=256).to(device)
        nce_loss_fn     = NCELoss(tau=nce_tau).to(device)
        optimizer_G = optim.Adam(
            list(model.parameters()) + list(patch_sampler.parameters()),
            lr=cfg["lr"], weight_decay=1e-4,
        )
        optimizer_D = optim.Adam(discriminator.parameters(), lr=cfg["lr"], betas=(0.5, 0.999))
    else:
        optimizer_G = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_G, mode="min", patience=cfg.get("lr_patience", 20), factor=0.5, min_lr=1e-6,
    )

    best_val, best_state = float("inf"), None
    log_interval         = cfg.get("log_interval", 10)
    if use_gan_nce:
        history = {"train_loss": [], "val_loss": [], "train_gan": [], "train_nce": [], "train_d": []}
    else:
        history = {"train_loss": [], "val_loss": [], "train_recon": [], "train_reg": []}

    for epoch in range(cfg["epochs"]):
        model.train()
        train_loss = 0.0
        sum_recon  = 0.0
        sum_reg    = 0.0

        if use_gan_nce:
            discriminator.train()
            sum_gan = sum_nce = sum_d = 0.0

        for raw_b, clean_b in train_loader:
            raw_b, clean_b = raw_b.to(device), clean_b.to(device)

            if use_gan_nce:
                # ── D update ──────────────────────────────────────────────────
                discriminator.requires_grad_(True)
                optimizer_D.zero_grad()
                with torch.no_grad():
                    fake = model(raw_b)
                d_real  = discriminator(clean_b)
                d_fake  = discriminator(fake)
                loss_D  = 0.5 * (F.mse_loss(d_real, torch.ones_like(d_real))
                                 + F.mse_loss(d_fake, torch.zeros_like(d_fake)))
                loss_D.backward()
                optimizer_D.step()

                # ── G update: L_G = L_GAN + λ_NCE · L_NCE ────────────────────
                discriminator.requires_grad_(False)
                optimizer_G.zero_grad()

                pred, feats_raw  = model(raw_b, return_features=True)
                feats_raw        = [f.detach() for f in feats_raw]

                d_fake_g  = discriminator(pred)
                loss_GAN  = F.mse_loss(d_fake_g, torch.ones_like(d_fake_g))

                _, feats_pred    = model(pred, return_features=True)
                feats_pred_nce   = [feats_pred[i] for i in nce_layers]
                feats_raw_nce    = [feats_raw[i]  for i in nce_layers]
                proj_q, ids      = patch_sampler(feats_pred_nce, nce_num_patches)
                proj_k, _        = patch_sampler(feats_raw_nce,  nce_num_patches, patch_ids=ids)
                loss_NCE = sum(nce_loss_fn(q, k) for q, k in zip(proj_q, proj_k)) / len(proj_q)

                loss_G = loss_GAN + lambda_NCE * loss_NCE
                loss_G.backward()
                optimizer_G.step()

                n = raw_b.size(0)
                sum_gan    += loss_GAN.item()   * n
                sum_nce    += loss_NCE.item()   * n
                sum_d      += loss_D.item()     * n
                train_loss += loss_G.item()     * n
            else:
                optimizer_G.zero_grad()
                pred    = model(raw_b)
                recon_l = criterion(pred, clean_b)
                loss    = recon_l + pred_reg_w * _pred_reg_loss(pred, pred_reg_type)
                loss.backward()
                optimizer_G.step()
                n = raw_b.size(0)
                sum_recon  += recon_l.item() * n
                sum_reg    += (loss.item() - recon_l.item()) * n
                train_loss += loss.item() * n

        train_loss /= len(train_ds)

        model.eval()
        if use_gan_nce:
            discriminator.eval()
        val_loss = 0.0
        with torch.no_grad():
            for raw_b, clean_b in val_loader:
                raw_b, clean_b = raw_b.to(device), clean_b.to(device)
                pred     = model(raw_b)
                recon    = criterion(pred, clean_b)
                if pred_reg_w > 0:
                    recon = recon + pred_reg_w * _pred_reg_loss(pred, pred_reg_type)
                val_loss += recon.item() * raw_b.size(0)
        val_loss /= len(val_ds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        n = len(train_ds)
        if use_gan_nce:
            history["train_gan"].append(sum_gan / n)
            history["train_nce"].append(sum_nce / n)
            history["train_d"].append(sum_d / n)
        else:
            history["train_recon"].append(sum_recon / n)
            history["train_reg"].append(sum_reg / n)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % log_interval == 0 or epoch == cfg["epochs"] - 1:
            lr_now = optimizer_G.param_groups[0]["lr"]
            if use_gan_nce:
                n = len(train_ds)
                print(f"Epoch {epoch+1}/{cfg['epochs']} | "
                      f"GAN: {sum_gan/n:.4f} | NCE: {sum_nce/n:.4f} | "
                      f"D: {sum_d/n:.4f} | Val: {val_loss:.6f} | LR: {lr_now:.2e}")
            else:
                print(f"Epoch {epoch+1}/{cfg['epochs']} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr_now:.2e}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Best val loss: {best_val:.6f}")

    if use_gan_nce:
        torch.save(discriminator.state_dict(), os.path.join(save_dir, "discriminator.pt"))
        torch.save(patch_sampler.state_dict(), os.path.join(save_dir, "patch_sampler.pt"))
        print(f"Saved → {save_dir}/discriminator.pt, patch_sampler.pt")

    return model, history, train_ids, val_ids, test_ids


def train_denoising(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    save_dir = cfg.get("save_dir", "results/denoising")
    data_dir = cfg["data_dir"]
    df       = pd.read_csv(os.path.join(data_dir, cfg["labels_csv"]))
    df_pos   = df[df["count"] > 0].sort_values("sample_id").reset_index(drop=True)

    # Optional cap on total positive samples used (set by run_sweep.py; no-op if absent).
    sample_size = cfg.get("sample_size")
    if sample_size is not None:
        df_pos = df_pos.iloc[:sample_size].copy()

    df_bg = None
    if cfg.get("include_background", False):
        df_bg = df[df["count"] == 0].sort_values("sample_id").reset_index(drop=True)

    os.makedirs(save_dir, exist_ok=True)
    model, history, _, _, _ = train_denoising_core(df_pos, cfg, save_dir, device=device, df_bg=df_bg)

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
    parser.add_argument("--dataset", default=None,
                        help="Dataset profile name or path (default: ACTIVE_DATASET in config.py)")
    parser.add_argument("--epochs",   type=int,   default=None, help="Override epochs")
    parser.add_argument("--lr",       type=float, default=None, help="Override learning rate")
    parser.add_argument("--data_dir", type=str,   default=None, help="Override data directory")
    args = parser.parse_args()

    cfg = load_config(args.task, args.config, dataset=args.dataset)
    apply_overrides(cfg, args)

    if args.task == "denoising":
        train_denoising(cfg)
    elif args.task == "detection":
        train_detection(cfg)
    elif args.task == "weight":
        train_weight(cfg)


if __name__ == "__main__":
    main()
