import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from train_baselines import TwoDCNNBaseline

DEPLOYMENT_ROOT = Path(__file__).resolve().parent
BASE_DIR = DEPLOYMENT_ROOT.parent if (DEPLOYMENT_ROOT.parent / "tensor_cache").exists() else DEPLOYMENT_ROOT
MODEL_DIR = Path(__file__).resolve().parent / "trained_baselines"


EUROPE_BOUNDS = {
    "lat_min": 34.5,
    "lat_max": 71.0,
    "lon_min": -10.5,
    "lon_max": 40.0,
}


PRESETS = [
    {"name": "Central cropland", "land": "Cropland", "lat": 47.15023795, "lon": 16.13421178},
    {"name": "French grassland", "land": "Grassland", "lat": 44.10761653, "lon": 0.436465825},
    {"name": "Northern peatland", "land": "Peatland", "lat": 54.6872, "lon": 25.2797},
]


def in_europe(lat, lon):
    return (
        EUROPE_BOUNDS["lat_min"] <= float(lat) <= EUROPE_BOUNDS["lat_max"]
        and EUROPE_BOUNDS["lon_min"] <= float(lon) <= EUROPE_BOUNDS["lon_max"]
    )


def extract_tabular_features(image, env4):
    image = np.asarray(image, dtype=np.float32)
    vector = np.concatenate([
        np.nanmean(image, axis=(1, 2)),
        np.nanstd(image, axis=(1, 2)),
        np.nanmin(image, axis=(1, 2)),
        np.nanmax(image, axis=(1, 2)),
        np.asarray(env4, dtype=np.float32).reshape(-1),
    ])
    return np.nan_to_num(vector, nan=0.0).reshape(1, -1).astype(np.float32)


class BaselineBackend:
    def __init__(self, device):
        self.device = device
        self.rf = None
        self.rf_scaler = None
        self.gbdt = None
        self.gbdt_scaler = None
        self.cnn = None
        self.metrics = {}
        self.available = False
        self.load()

    def load(self):
        metrics_path = MODEL_DIR / "metrics.json"
        if metrics_path.exists():
            self.metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        rf_scaler_path = next(MODEL_DIR.glob("Random_Forest_scaler.joblib"), None)
        rf_path = next(MODEL_DIR.glob("Random_Forest.joblib"), None)
        gbdt_scaler_path = next(MODEL_DIR.glob("*GBDT*scaler.joblib"), None)
        gbdt_path = next(
            (path for path in MODEL_DIR.glob("*GBDT*.joblib") if "scaler" not in path.name),
            None,
        )
        cnn_path = MODEL_DIR / "2D_CNN.pt"

        if rf_path is not None and rf_scaler_path is not None:
            self.rf_scaler = joblib.load(rf_scaler_path)
            self.rf = joblib.load(rf_path)
        if gbdt_path is not None and gbdt_scaler_path is not None:
            self.gbdt_scaler = joblib.load(gbdt_scaler_path)
            self.gbdt = joblib.load(gbdt_path)
        if cnn_path.exists():
            self.cnn = TwoDCNNBaseline()
            self.cnn.load_state_dict(torch.load(cnn_path, map_location="cpu", weights_only=True))
            self.cnn.eval()
            self.cnn.to(self.device)
        self.available = bool(self.rf and self.gbdt)

    def predict_tabular(self, model_name, image, env4):
        features = extract_tabular_features(image, env4)
        if model_name == "Random Forest":
            scaled = self.rf_scaler.transform(features)
            return float(np.expm1(self.rf.predict(scaled)[0]))
        if model_name == "XGBoost / GBDT":
            scaled = self.gbdt_scaler.transform(features)
            return float(np.expm1(self.gbdt.predict(scaled)[0]))
        raise ValueError(f"Unknown baseline model: {model_name}")

    @torch.inference_mode()
    def predict_2dcnn(self, image, coords, env4):
        if self.cnn is None:
            raise RuntimeError("The trained 2D-CNN artifact is not available yet.")
        tensor = torch.from_numpy(np.asarray(image, dtype=np.float32)).unsqueeze(0).to(self.device)
        coord_tensor = torch.from_numpy(np.asarray(coords, dtype=np.float32)).unsqueeze(0).to(self.device)
        env_tensor = torch.from_numpy(np.asarray(env4, dtype=np.float32)).unsqueeze(0).to(self.device)
        return float(np.expm1(self.cnn(tensor, coord_tensor, env_tensor).cpu().numpy().reshape(-1)[0]))

    def predict_map(self, model_name, image, coords, env4, window=256, stride=128):
        h, w = image.shape[1], image.shape[2]
        output = np.zeros((h, w), dtype=np.float32)
        counts = np.zeros((h, w), dtype=np.float32)
        if h < window or w < window:
            patch = np.zeros((12, window, window), dtype=np.float32)
            patch[:, :h, :w] = image[:, :h, :w]
            prediction = (
                self.predict_tabular(model_name, patch, env4)
                if model_name != "2D-CNN"
                else self.predict_2dcnn(patch, coords, env4)
            )
            return np.full((h, w), prediction, dtype=np.float32)

        for y in range(0, h - window + 1, stride):
            for x in range(0, w - window + 1, stride):
                patch = image[:, y:y + window, x:x + window]
                prediction = (
                    self.predict_tabular(model_name, patch, env4)
                    if model_name != "2D-CNN"
                    else self.predict_2dcnn(patch, coords, env4)
                )
                output[y:y + window, x:x + window] += prediction
                counts[y:y + window, x:x + window] += 1

        edge_y = h - window
        edge_x = w - window
        if edge_y % stride:
            y = edge_y
            for x in range(0, w - window + 1, stride):
                patch = image[:, y:y + window, x:x + window]
                prediction = (
                    self.predict_tabular(model_name, patch, env4)
                    if model_name != "2D-CNN"
                    else self.predict_2dcnn(patch, coords, env4)
                )
                output[y:y + window, x:x + window] += prediction
                counts[y:y + window, x:x + window] += 1
        if edge_x % stride:
            x = edge_x
            for y in range(0, h - window + 1, stride):
                patch = image[:, y:y + window, x:x + window]
                prediction = (
                    self.predict_tabular(model_name, patch, env4)
                    if model_name != "2D-CNN"
                    else self.predict_2dcnn(patch, coords, env4)
                )
                output[y:y + window, x:x + window] += prediction
                counts[y:y + window, x:x + window] += 1
        counts = np.maximum(counts, 1)
        return output / counts


def benchmark_metrics_from_repository():
    table = BASE_DIR / "results" / "table4_controlled_in_dataset_benchmark.csv"
    if not table.exists():
        return pd.DataFrame()
    frame = pd.read_csv(table)
    renamed = pd.DataFrame({
        "Model": frame["Model / Paper Architecture"].str.replace(r"^[0-9]+\\.\\s*", "", regex=True),
        "R2": frame["SBCV R²"],
        "MAE": frame["MAE (g/kg)"],
        "RMSE": frame["RMSE (g/kg)"],
        "RPD": frame["RPD"],
    })
    return renamed
