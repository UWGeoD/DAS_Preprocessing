"""
Usage:
    python predict.py --task denoising                          # denoise all data/raw/*.npy → data/denoised/
    python predict.py --task detection --input sample.npy       # predict vehicle count + type
    python predict.py --task weight    --input sample.npy       # predict vehicle weight
"""

import warnings
warnings.filterwarnings("ignore", message="Pandas requires version", category=UserWarning)

import argparse
import glob
import os

import numpy as np
import torch

from dataset import IDX_TO_TYPE
from models.detection_transformer import DASCountTransformer
from models.unet_v2 import UNetV2
from models.weight_cnn import DASWeightCNN
from train import load_config


def _enforce_business_logic(raw_count, type_logits):
    """count=0 → background, count>1 → mixed, count=1 → single-vehicle type."""
    count = int(round(raw_count))
    if count == 0:
        return count, "background"
    if count > 1:
        return count, "mixed"
    best = int(np.argmax(type_logits[2:6])) + 2
    return count, IDX_TO_TYPE[best]


def predict_denoising(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = cfg.get("save_dir", "results/denoising")
    data_dir = cfg["data_dir"]

    model = UNetV2(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pt"), map_location=device))
    model.eval()

    in_dir  = os.path.join(data_dir, "raw")
    out_dir = os.path.join(data_dir, "denoised")
    os.makedirs(out_dir, exist_ok=True)

    stale = glob.glob(os.path.join(out_dir, "denoised_*.npy"))
    if stale:
        for f in stale:
            os.remove(f)
        print(f"Cleared {len(stale)} stale denoised file(s) from {out_dir}/")

    files = sorted(glob.glob(os.path.join(in_dir, "sample_*.npy")))
    print(f"Denoising {len(files)} files → {out_dir}/")
    with torch.no_grad():
        for fpath in files:
            raw = np.load(fpath).astype(np.float32)
            mean, std = raw.mean(), raw.std() + 1e-8
            raw_norm = (raw - mean) / std
            t = torch.from_numpy(raw_norm).unsqueeze(0).unsqueeze(0).to(device)
            pred_norm = model(t).squeeze().cpu().numpy()
            pred = pred_norm * std + mean  # restore original signal units
            np.save(os.path.join(out_dir, f"denoised_{os.path.basename(fpath)}"), pred)
    print("Done.")


def predict_detection(cfg, input_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = cfg.get("save_dir", "results/detection")

    sample = np.load(input_path).astype(np.float32)
    spatial_ch = sample.shape[0]

    model = DASCountTransformer(
        spatial_channels=spatial_ch,
        d_model=cfg.get("d_model", 64),
        nhead=cfg.get("nhead", 4),
        num_layers=cfg.get("num_layers", 2),
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pt"), map_location=device))
    model.eval()

    t = torch.from_numpy(sample).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        p_count, p_type = model(t)

    count, vtype = _enforce_business_logic(p_count.item(), p_type.cpu().numpy().flatten())
    print(f"File: {input_path}")
    print(f"Count: {count} | Type: {vtype}")


def predict_weight(cfg, input_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = cfg.get("save_dir", "results/weight")

    sample = np.load(input_path).astype(np.float32)
    spatial_ch = sample.shape[0]

    model = DASWeightCNN(spatial_channels=spatial_ch).to(device)
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pt"), map_location=device))
    model.eval()

    t = torch.from_numpy(sample).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(t)

    weight_lbs = max(0.0, pred.item() * 1000.0)
    print(f"File: {input_path}")
    print(f"Predicted weight: {weight_lbs:,.0f} lbs")


def main():
    parser = argparse.ArgumentParser(description="Run DAS model inference.")
    parser.add_argument("--task", required=True, choices=["denoising", "detection", "weight"])
    parser.add_argument("--config", default=None, help="Path to YAML config (default: configs/<task>.yaml)")
    parser.add_argument("--input", default=None, help="Path to .npy input file (detection/weight tasks)")
    args = parser.parse_args()

    cfg = load_config(args.task, args.config)

    if args.task == "denoising":
        predict_denoising(cfg)
    elif args.task == "detection":
        if not args.input:
            parser.error("--input is required for detection task")
        predict_detection(cfg, args.input)
    elif args.task == "weight":
        if not args.input:
            parser.error("--input is required for weight task")
        predict_weight(cfg, args.input)


if __name__ == "__main__":
    main()
