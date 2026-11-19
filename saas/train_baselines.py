import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

DEPLOYMENT_ROOT = Path(__file__).resolve().parent
BASE_DIR = DEPLOYMENT_ROOT.parent if (DEPLOYMENT_ROOT.parent / "tensor_cache").exists() else DEPLOYMENT_ROOT
MODEL_DIR = Path(__file__).resolve().parent / "trained_baselines"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Cached2DCNNDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = torch.load(
            BASE_DIR / "tensor_cache" / f"point_{int(row['point_id'])}.pt",
            map_location="cpu",
            weights_only=False,
        )
        img = torch.from_numpy(np.asarray(data["img"], dtype=np.float32))
        coords = torch.from_numpy(np.asarray(data["coords"], dtype=np.float32))
        env = torch.from_numpy(np.asarray(data["env"], dtype=np.float32)[:4].copy())
        target = torch.from_numpy(np.asarray(data["target"], dtype=np.float32)).view(-1)
        return img, coords, env, target


class SpatialFourierEmbedding(nn.Module):
    def __init__(self, num_bands=4):
        super().__init__()
        self.register_buffer("frequencies", 2.0 ** torch.arange(num_bands))

    def forward(self, coords):
        scaled = coords.unsqueeze(-1) * self.frequencies * np.pi
        return torch.cat([
            torch.sin(scaled).reshape(coords.shape[0], -1),
            torch.cos(scaled).reshape(coords.shape[0], -1),
        ], dim=1)


class TwoDCNNBaseline(nn.Module):
    def __init__(self, target_mean=29.6221):
        super().__init__()
        self.norm = nn.BatchNorm2d(12)
        self.fourier = SpatialFourierEmbedding(4)
        self.features = nn.Sequential(
            nn.Conv2d(12, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(4),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 + 16 + 4, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        nn.init.constant_(self.head[-1].bias, np.log1p(target_mean))

    def forward(self, image, coords, env):
        image = self.norm(torch.nan_to_num(image, nan=0.0))
        features = self.features(image)
        return torch.nan_to_num(
            self.head(torch.cat([features, self.fourier(coords), env], dim=1)),
            nan=0.0,
        )


def extract_tabular_features(df):
    rows = []
    targets = []
    for _, row in df.iterrows():
        data = torch.load(
            BASE_DIR / "tensor_cache" / f"point_{int(row['point_id'])}.pt",
            map_location="cpu",
            weights_only=False,
        )
        image = np.asarray(data["img"], dtype=np.float32)
        env = np.asarray(data["env"], dtype=np.float32)[:4]
        features = np.concatenate([
            np.nanmean(image, axis=(1, 2)),
            np.nanstd(image, axis=(1, 2)),
            np.nanmin(image, axis=(1, 2)),
            np.nanmax(image, axis=(1, 2)),
            env,
        ])
        rows.append(np.nan_to_num(features, nan=0.0))
        targets.append(float(row["oc_g_kg"]))
    return np.asarray(rows, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def pooled_metrics(y_true, y_pred):
    y_pred = np.clip(np.nan_to_num(y_pred, nan=float(np.mean(y_true))), 0.0, 200.0)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse,
        "RPD": float(np.std(y_true) / rmse),
    }


def train_tabular_baselines(df):
    X, y = extract_tabular_features(df)
    splits = [
        (np.where(df["spatial_cluster"].to_numpy() != fold)[0],
         np.where(df["spatial_cluster"].to_numpy() == fold)[0])
        for fold in range(5)
    ]

    specifications = {
        "Random Forest": lambda: RandomForestRegressor(
            n_estimators=150, max_depth=14, random_state=42, n_jobs=-1
        ),
        "XGBoost / GBDT": lambda: GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=6, random_state=42
        ),
    }
    metrics = {}

    for name, factory in specifications.items():
        oof_log = np.zeros(len(df), dtype=np.float64)
        for train_idx, val_idx in splits:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_val = scaler.transform(X[val_idx])
            model = factory()
            model.fit(X_train, np.log1p(y[train_idx]))
            oof_log[val_idx] = model.predict(X_val)
        metrics[name] = pooled_metrics(y, np.expm1(oof_log))

        full_scaler = StandardScaler()
        X_full = full_scaler.fit_transform(X)
        full_model = factory()
        full_model.fit(X_full, np.log1p(y))
        joblib.dump(full_scaler, MODEL_DIR / f"{name.replace(' ', '_').replace('/', '_')}_scaler.joblib")
        joblib.dump(full_model, MODEL_DIR / f"{name.replace(' ', '_').replace('/', '_')}.joblib")

    return metrics


def train_2dcnn(df, epochs=8):
    y_true = df["oc_g_kg"].to_numpy(dtype=np.float64)
    target_mean = float(y_true.mean())
    oof_log = np.zeros(len(df), dtype=np.float32)

    splits = [
        (np.where(df["spatial_cluster"].to_numpy() != fold)[0],
         np.where(df["spatial_cluster"].to_numpy() == fold)[0])
        for fold in range(5)
    ]

    for fold, (train_idx, val_idx) in enumerate(splits):
        model = TwoDCNNBaseline(target_mean).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.005)
        train_loader = DataLoader(
            Cached2DCNNDataset(df.iloc[train_idx]),
            batch_size=16,
            shuffle=True,
            num_workers=2,
            pin_memory=DEVICE.type == "cuda",
        )
        val_loader = DataLoader(
            Cached2DCNNDataset(df.iloc[val_idx]),
            batch_size=16,
            shuffle=False,
            num_workers=2,
            pin_memory=DEVICE.type == "cuda",
        )
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            model.train()
            for image, coords, env, target in train_loader:
                image = image.to(DEVICE)
                coords = coords.to(DEVICE)
                env = env.to(DEVICE)
                target = target.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                pred = model(image, coords, env)
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()

        model.eval()
        predictions = []
        with torch.inference_mode():
            for image, coords, env, _ in val_loader:
                image = image.to(DEVICE)
                coords = coords.to(DEVICE)
                env = env.to(DEVICE)
                predictions.append(model(image, coords, env).cpu().numpy().reshape(-1))
        oof_log[val_idx] = np.concatenate(predictions)

    model = TwoDCNNBaseline(target_mean).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.005)
    full_loader = DataLoader(
        Cached2DCNNDataset(df),
        batch_size=16,
        shuffle=True,
        num_workers=2,
        pin_memory=DEVICE.type == "cuda",
    )
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        for image, coords, env, target in full_loader:
            image = image.to(DEVICE)
            coords = coords.to(DEVICE)
            env = env.to(DEVICE)
            target = target.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(image, coords, env), target)
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), MODEL_DIR / "2D_CNN.pt")
    return pooled_metrics(y_true, np.expm1(oof_log))


if __name__ == "__main__":
    df = pd.read_csv(BASE_DIR / "production_multimodal_master_index.csv")
    tabular_metrics = train_tabular_baselines(df)
    cnn_metrics = train_2dcnn(df, epochs=8)
    all_metrics = {
        "Random Forest": tabular_metrics["Random Forest"],
        "XGBoost / GBDT": tabular_metrics["XGBoost / GBDT"],
        "2D-CNN": cnn_metrics,
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(json.dumps(all_metrics, indent=2))
