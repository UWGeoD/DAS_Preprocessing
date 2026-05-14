import os
import numpy as np
import torch
from torch.utils.data import Dataset

TYPE_MAP = {'background': 0, 'mixed': 1, 'suv': 2, 'van': 3, 'sedan': 4, 'truck': 5}
IDX_TO_TYPE = {v: k for k, v in TYPE_MAP.items()}


class DASSampleDataset(Dataset):
    """
    Dataset driven by labels.csv rows.

    Each row has:
        sample_id, data_path, count, start_frame, end_frame, vehicle_type

    We use:
        X = raw np.load(data_path)
        Y = preprocess(X, fs, dx=..., dt=...)   # dt optional; defaults to 1/fs
    """
    def __init__(self, df, preprocess, fs_das, *, dx=None, root_dir=".", augment=False):
        """
        df        : pandas DataFrame subset of labels.csv (already filtered)
        preprocess: callable preprocess(x, fs, **ctx) from make_preprocess(...)
        fs_das    : sampling rate (Hz)
        dx        : channel spacing (m). Only needed if you include fk_filter in the pipeline.
        root_dir  : base directory for relative paths in df["data_path"]
        """
        self.df = df.reset_index(drop=True)
        self.preprocess = preprocess
        self.fs_das = float(fs_das)
        self.dx = dx
        self.root_dir = root_dir
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["data_path"]

        if not os.path.isabs(path):
            path = os.path.join(self.root_dir, path)

        raw = np.load(path).astype(np.float32)  # (n_channels, n_time)

        if self.dx is None:
            clean = self.preprocess(raw, self.fs_das).astype(np.float32)
        else:
            clean = self.preprocess(raw, self.fs_das, dx=self.dx).astype(np.float32)

        if self.augment:
            # Channel flip (p=0.5): reverse spatial order on both arrays.
            # Physically valid — DAS measurement is symmetric along the fiber.
            if np.random.random() < 0.5:
                raw   = raw[::-1, :].copy()
                clean = clean[::-1, :].copy()
            # Time reversal (p=0.5): reverse temporal order on both arrays.
            # Physically valid — bandpass/fk_filter are applied before this flip,
            # and raw→clean relationship is preserved under time reversal.
            if np.random.random() < 0.5:
                raw   = raw[:, ::-1].copy()
                clean = clean[:, ::-1].copy()

        # Per-channel mean removal: each channel has its own DC offset (absolute fiber strain
        # level), so we center each channel independently. std is then computed on the
        # centered signal and reflects only the within-channel noise amplitude — not the
        # inter-channel DC spread, which would inflate std and make clean_norm near-zero.
        chan_mean = raw.mean(axis=1, keepdims=True)  # (n_channels, 1)
        std       = (raw - chan_mean).std() + 1e-8
        raw_norm   = (raw - chan_mean) / std
        clean_norm = clean / std

        raw_t   = torch.from_numpy(raw_norm[None, ...])
        clean_t = torch.from_numpy(clean_norm[None, ...])

        return raw_t, clean_t


class DASCountDataset(Dataset):
    """Loads denoised samples for vehicle count + type prediction."""

    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = torch.from_numpy(np.load(row['data_path']).astype(np.float32)).unsqueeze(0)
        y_count = torch.tensor([row['count']], dtype=torch.float32)
        y_type = torch.tensor(TYPE_MAP[row['vehicle_type'].lower()], dtype=torch.long)
        return x, y_count, y_type


class DASWeightDataset(Dataset):
    """Loads denoised samples for vehicle weight regression."""

    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = torch.from_numpy(np.load(row['data_path']).astype(np.float32)).unsqueeze(0)
        y = torch.tensor([row['weight'] / 1000.0], dtype=torch.float32)
        return x, y
