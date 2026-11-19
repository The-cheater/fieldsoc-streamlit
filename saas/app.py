import io
import sys
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import rasterio
import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from streamlit_folium import st_folium
from rasterio.transform import from_origin
from rasterio.io import MemoryFile
from baseline_backend import (
    BaselineBackend,
    EUROPE_BOUNDS,
    PRESETS,
    benchmark_metrics_from_repository,
    in_europe,
)

DEPLOYMENT_ROOT = Path(__file__).resolve().parent
BASE_DIR = DEPLOYMENT_ROOT.parent if (DEPLOYMENT_ROOT.parent / "tensor_cache").exists() else DEPLOYMENT_ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import OptimizedMultimodalSOCNet

st.set_page_config(
    page_title="FieldSOC Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #17251f;
        --muted: #69766f;
        --green: #1f6a4d;
        --green-soft: #eaf3ee;
        --line: #dde5df;
        --canvas: #f4f7f5;
    }
    html, body, [class*="css"] {font-family: Inter, "Segoe UI", sans-serif; color: var(--ink) !important;}
    .stApp {background: var(--canvas) !important; color: var(--ink) !important;}
    .stApp p,
    .stApp label,
    .stApp span,
    .stApp div,
    .stApp small,
    .stApp [data-testid="stMarkdownContainer"] {
        color: var(--ink);
    }
    .block-container {max-width: 1500px; padding: 1.35rem 2.1rem 3rem;}
    section[data-testid="stSidebar"] {
        background: #fbfcfb;
        border-right: 1px solid var(--line);
        min-width: 286px;
    }
    section[data-testid="stSidebar"] > div {padding-top: 1.15rem;}
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {line-height: 1.4;}
    header[data-testid="stHeader"] {background: transparent;}
    [data-testid="stToolbar"], [data-testid="stDecoration"] {display: none;}
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 15px 16px 13px;
        box-shadow: 0 5px 18px rgba(31, 65, 48, 0.035);
    }
    [data-testid="stMetricLabel"] {color: var(--muted); font-size: 0.78rem; font-weight: 700;}
    [data-testid="stMetricValue"] {color: var(--ink); font-size: 1.55rem; font-weight: 760;}
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricValue"] p,
    [data-testid="stCaptionContainer"] p {color: var(--muted) !important; opacity: 1 !important;}
    [data-testid="stCaptionContainer"] {color: var(--muted) !important;}
    .metric-caption {color: var(--muted); font-size: 0.76rem; margin-top: -5px;}
    .helper-text {color: var(--muted); font-size: 0.78rem; line-height: 1.42; margin: -4px 0 12px;}
    .plain-card {
        background: #fff;
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 14px 16px;
        color: var(--ink);
        box-shadow: 0 5px 16px rgba(31,65,48,.035);
    }
    .plain-card strong {display:block; margin-bottom:3px;}
    .plain-card span {color: var(--muted); font-size:.8rem;}
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 1px solid var(--line);
        gap: 28px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        color: var(--muted);
        font-weight: 700;
        padding: 0 4px;
    }
    .stTabs [aria-selected="true"] {color: var(--green); border-bottom-color: var(--green) !important;}
    [data-testid="stVerticalBlock"] {gap: 0.8rem;}
    [data-testid="stDataFrame"], [data-testid="stPlotlyChart"] {
        background: #fff;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 8px;
        color: var(--ink);
    }
    .main-header {
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 1.55rem;
        font-weight: 780;
        letter-spacing: -0.025em;
        color: var(--ink);
    }
    .brand-mark {
        display: grid;
        place-items: center;
        width: 38px;
        height: 38px;
        border-radius: 11px;
        background: var(--green);
        color: #fff;
        font-size: 1.05rem;
        box-shadow: 0 7px 16px rgba(31, 106, 77, 0.18);
    }
    .sub-header {color: var(--muted); margin: 2px 0 18px 50px; font-size: 0.9rem;}
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 18px;
        color: var(--ink);
        font-weight: 780;
        font-size: 1.02rem;
    }
    .sidebar-brand span {
        display: grid;
        place-items: center;
        width: 32px;
        height: 32px;
        border-radius: 9px;
        background: var(--green);
        color: #fff;
    }
    .section-label {
        color: var(--muted);
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        margin: 4px 0 8px;
    }
    .status-card {
        border: 1px solid var(--line);
        background: #fff;
        border-radius: 14px;
        padding: 22px;
        min-height: 220px;
        box-shadow: 0 8px 24px rgba(31, 65, 48, 0.04);
    }
    .status-card h3 {margin: 8px 0 6px; color: var(--ink);}
    .status-card p {color: var(--muted); max-width: 520px;}
    .idle-icon {
        display: grid;
        place-items: center;
        width: 44px;
        height: 44px;
        border-radius: 12px;
        background: var(--green-soft);
        color: var(--green);
        font-size: 1.2rem;
    }
    .map-shell {
        border: 1px solid var(--line);
        border-radius: 14px;
        overflow: hidden;
        background: #fff;
        box-shadow: 0 8px 24px rgba(31, 65, 48, 0.04);
    }
    div[data-testid="stButton"] button {
        border: 0;
        border-radius: 9px;
        background: var(--green);
        color: #fff;
        min-height: 44px;
        font-weight: 750;
        box-shadow: 0 8px 18px rgba(31, 106, 77, 0.14);
    }
    div[data-testid="stButton"] button:hover {background: #18583f; color: #fff; border: 0;}
    [data-testid="stSidebar"] hr {margin: 14px 0; border-color: var(--line);}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoints = sorted((BASE_DIR / "checkpoints").glob("*.pt"))
    checkpoints += sorted((BASE_DIR / "checkpoints").glob("*.pth"))
    checkpoints = list(dict.fromkeys(checkpoints))
    models = []
    for checkpoint in checkpoints[:5]:
        model = OptimizedMultimodalSOCNet(in_channels=12, env_dim=5)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    if not models:
        raise RuntimeError("No trained checkpoints found in e:\\sat\\checkpoints")
    return models, device


@st.cache_data
def load_index():
    path = BASE_DIR / "production_multimodal_master_index.csv"
    frame = pd.read_csv(path) if path.exists() else pd.DataFrame()
    manifest = BASE_DIR / "available_tensors.json"
    if not frame.empty and manifest.exists():
        import json
        available = json.loads(manifest.read_text(encoding="utf-8"))
        available_ids = set(int(value) for value in available.get("point_ids", []))
        if available_ids:
            frame = frame[frame["point_id"].astype(int).isin(available_ids)].copy()
    return frame


@st.cache_resource
def download_lucas_tile(point_id):
    try:
        repo_id = st.secrets.get("HF_DATASET_REPO", "")
    except Exception:
        repo_id = ""
    if not repo_id:
        repo_id = "dexter-io/lucas-soc-tensors"
    if not repo_id:
        return None
    try:
        token = st.secrets.get("HF_TOKEN", None)
    except Exception:
        token = None
    relative_path = f"tensor_cache/point_{int(point_id)}.pt"
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=relative_path,
            token=token,
        )
        return Path(local_path)
    except Exception as exc:
        return None


@st.cache_data
def load_benchmark_results():
    repository_metrics = benchmark_metrics_from_repository()
    metrics_path = Path(__file__).resolve().parent / "trained_baselines" / "metrics.json"
    trained_metrics = {}
    if metrics_path.exists():
        import json
        trained_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    rows = []
    for _, row in repository_metrics.iterrows():
        model = str(row["Model"])
        if model.startswith("Our Proposed") or model.startswith("Proposed"):
            model = "ResUNet-a + S2AP (Ours)"
        elif model.startswith("Random Forest"):
            model = "Random Forest"
        elif model.startswith("XGBoost"):
            model = "XGBoost / GBDT"
        elif model.startswith("Custom ResUNet"):
            model = "ResUNet-a + S2AP (Ours)"
        else:
            continue
        if model not in {"Random Forest", "XGBoost / GBDT", "ResUNet-a + S2AP (Ours)"}:
            continue
        rows.append({
            "Model": model,
            "R2": float(row["R2"]),
            "MAE g/kg": float(row["MAE"]),
            "RMSE g/kg": float(row["RMSE"]),
            "RPD": float(row["RPD"]),
        })

    for model, values in trained_metrics.items():
        matching = next((item for item in rows if item["Model"] == model), None)
        if matching is not None:
            matching.update({
                "R2": float(values["R2"]),
                "MAE g/kg": float(values["MAE"]),
                "RMSE g/kg": float(values["RMSE"]),
                "RPD": float(values["RPD"]),
            })
        else:
            rows.append({
                "Model": model,
                "R2": float(values["R2"]),
                "MAE g/kg": float(values["MAE"]),
                "RMSE g/kg": float(values["RMSE"]),
                "RPD": float(values["RPD"]),
            })

    s2ap_pooled_path = BASE_DIR / "results" / "table1_model_performance_summary.csv"
    s2ap_row = None
    if s2ap_pooled_path.exists():
        pooled = pd.read_csv(s2ap_pooled_path)
        sbcv = pooled[pooled["Validation Scheme"].str.contains("Spatially Blocked", case=False, na=False)]
        if not sbcv.empty:
            record = sbcv.iloc[0]
            s2ap_row = {
                "Model": "ResUNet-a + S2AP (Ours)",
                "R2": float(record["R2 Score"]),
                "MAE g/kg": float(record["MAE (g/kg)"]),
                "RMSE g/kg": float(record["RMSE (g/kg)"]),
                "RPD": float(record["RPD"]),
            }
    if s2ap_row is None:
        raise RuntimeError(
            "The S2AP pooled SBCV metrics were not found in "
            "results/table1_model_performance_summary.csv"
        )

    allowed = ["Random Forest", "XGBoost / GBDT", "2D-CNN", "ResUNet-a + S2AP (Ours)"]
    rows_by_name = {row["Model"]: row for row in rows}
    rows_by_name["ResUNet-a + S2AP (Ours)"] = s2ap_row
    result = [rows_by_name[name] for name in allowed if name in rows_by_name]
    return pd.DataFrame(result)


def normalize_upload(image):
    image = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if image.shape[0] != 12:
        raise ValueError(f"Expected 12 bands, received {image.shape[0]}.")
    normalized = np.empty_like(image)
    for band in range(12):
        std = float(np.std(image[band]))
        normalized[band] = (image[band] - float(np.mean(image[band]))) / std if std > 0 else 0.0
    return normalized


def default_env(row=None):
    if row is not None:
        return np.array([
            row["elevation"] / 1000.0,
            row["slope"] / 45.0,
            row["temp"] / 40.0,
            row["precip"] / 2000.0,
            np.log1p(row["soilgrids_prior"]) / 5.0,
        ], dtype=np.float32)
    return np.array([0.35, 0.22, 0.45, 0.50, np.log1p(20.0) / 5.0], dtype=np.float32)


def pad_to_256(image):
    _, h, w = image.shape
    if h >= 256 and w >= 256:
        return image[:, :256, :256]
    out = np.zeros((12, max(h, 256), max(w, 256)), dtype=np.float32)
    out[:, :h, :w] = image
    if h < 256:
        out[:, h:, :w] = image[:, -1:, :] if h else 0
    if w < 256:
        out[:, :h, w:] = out[:, :h, max(w - 1, 0):w]
    if h < 256 and w < 256:
        out[:, h:, w:] = out[:, h - 1:h, w - 1:w]
    return out[:, :256, :256]


@torch.inference_mode()
def point_inference(models, device, image, coords, env, mc_passes=12):
    image_tensor = torch.from_numpy(image.copy()).unsqueeze(0).to(device)
    coords_tensor = torch.from_numpy(np.asarray(coords, dtype=np.float32)).unsqueeze(0).to(device)
    env_tensor = torch.from_numpy(np.asarray(env, dtype=np.float32)).unsqueeze(0).to(device)
    soc_samples = []
    seg_samples = []
    for model in models:
        model.to(device)
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()
        for _ in range(mc_passes):
            logits, soc_log = model(image_tensor, coords_tensor, env_tensor)
            soc_samples.append(np.expm1(soc_log.detach().cpu().numpy()).reshape(-1)[0])
            seg_samples.append(torch.sigmoid(logits).detach().cpu().numpy()[0, 0])
    soc_mean = float(np.mean(soc_samples))
    soc_std = float(np.std(soc_samples))
    seg_mean = np.mean(seg_samples, axis=0)
    seg_mean = seg_mean / (seg_mean.max() + 1e-6)
    return soc_mean, soc_std, seg_mean


def sliding_inference(models, device, image, coords, env, mc_passes=5, window=256, stride=128):
    h, w = image.shape[1], image.shape[2]
    if h < window or w < window:
        patch = pad_to_256(image)
        soc, uncertainty, mask = point_inference(models, device, patch, coords, env, mc_passes)
        return (
            np.full((h, w), soc, dtype=np.float32),
            np.full((h, w), uncertainty, dtype=np.float32),
            np.array(mask[:h, :w]),
            soc,
        )

    soc_sum = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    uncertainty_sum = np.zeros((h, w), dtype=np.float32)
    mask_sum = np.zeros((h, w), dtype=np.float32)
    for y in range(0, h - window + 1, stride):
        for x in range(0, w - window + 1, stride):
            patch = image[:, y:y + window, x:x + window]
            soc, uncertainty, mask = point_inference(models, device, patch, coords, env, mc_passes)
            soc_sum[y:y + window, x:x + window] += soc
            uncertainty_sum[y:y + window, x:x + window] += uncertainty
            mask_sum[y:y + window, x:x + window] += mask
            weight[y:y + window, x:x + window] += 1
    edge_y, edge_x = h - window, w - window
    if edge_y % stride:
        y = edge_y
    else:
        y = None
    if edge_x % stride:
        x = edge_x
    else:
        x = None
    if y is not None:
        for x0 in range(0, w - window + 1, stride):
            patch = image[:, y:y + window, x0:x0 + window]
            soc, uncertainty, mask = point_inference(models, device, patch, coords, env, mc_passes)
            soc_sum[y:y + window, x0:x0 + window] += soc
            uncertainty_sum[y:y + window, x0:x0 + window] += uncertainty
            mask_sum[y:y + window, x0:x0 + window] += mask
            weight[y:y + window, x0:x0 + window] += 1
    if x is not None:
        for y0 in range(0, h - window + 1, stride):
            patch = image[:, y0:y0 + window, x:x + window]
            soc, uncertainty, mask = point_inference(models, device, patch, coords, env, mc_passes)
            soc_sum[y0:y0 + window, x:x + window] += soc
            uncertainty_sum[y0:y0 + window, x:x + window] += uncertainty
            mask_sum[y0:y0 + window, x:x + window] += mask
            weight[y0:y0 + window, x:x + window] += 1
    weight = np.maximum(weight, 1)
    soc_map = soc_sum / weight
    uncertainty_map = uncertainty_sum / weight
    mask_map = mask_sum / weight
    mask_map = mask_map / (mask_map.max() + 1e-6)
    return soc_map, uncertainty_map, mask_map, float(np.mean(soc_map))


def make_rgb(image):
    if image.shape[0] < 3:
        return np.zeros((image.shape[1], image.shape[2], 3), dtype=np.uint8)
    red = np.nan_to_num(image[2])
    green = np.nan_to_num(image[1])
    blue = np.nan_to_num(image[0])
    rgb = np.stack([red, green, blue], axis=-1)
    for channel in range(3):
        values = rgb[:, :, channel]
        lo, hi = np.percentile(values, (2, 98))
        rgb[:, :, channel] = np.clip((values - lo) / (hi - lo + 1e-6), 0, 1)
    return (rgb * 255).astype(np.uint8)


def image_overlay(data, colormap="inferno", alpha=0.72):
    data = np.nan_to_num(data.astype(np.float32), nan=0.0)
    lo, hi = np.percentile(data, (2, 98))
    normalized = np.clip((data - lo) / (hi - lo + 1e-6), 0, 1)
    cmap = px.colors.sample_colorscale(colormap, normalized.flatten())
    rgb = np.array([px.colors.unhex(p) if p.startswith("#") else (255, 255, 255) for p in cmap])
    # px.colors.sample_colorscale returns CSS rgb strings for some versions.
    if rgb.shape[-1] != 3:
        import matplotlib.cm as cm
        rgb = (cm.inferno(normalized)[..., :3] * 255).astype(np.uint8)
    else:
        rgb = rgb.reshape(normalized.shape[0], normalized.shape[1], 3).astype(np.uint8)
    return rgb


def safe_overlay(data, cmap_name="turbo", alpha=0.72):
    data = np.nan_to_num(data.astype(np.float32), nan=0.0)
    lo, hi = np.percentile(data, (2, 98))
    norm = np.clip((data - lo) / (hi - lo + 1e-6), 0, 1)
    try:
        import matplotlib.cm as cm
        return (cm.get_cmap(cmap_name)(norm)[..., :3] * 255).astype(np.uint8)
    except Exception:
        return (norm[..., None] * np.array([255, 120, 40], dtype=np.float32)).astype(np.uint8)


def geojson_mask_overlay(mask):
    return (np.clip(mask, 0, 1) * 255).astype(np.uint8)


def geotiff_bytes(soc, baseline, residual, bare_mask, uncertainty, transform):
    layers = np.stack([
        soc,
        baseline,
        residual,
        bare_mask,
        uncertainty,
        np.clip(soc - 1.96 * uncertainty, 0, None),
        soc + 1.96 * uncertainty,
    ]).astype(np.float32)
    buffer = io.BytesIO()
    with rasterio.Env():
        with rasterio.open(
            buffer,
            "w",
            driver="GTiff",
            height=layers.shape[1],
            width=layers.shape[2],
            count=layers.shape[0],
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(layers)
            descriptions = [
                "S2AP SOC (g/kg)",
                "Baseline SOC (g/kg)",
                "S2AP minus baseline (g/kg)",
                "Bare-soil probability",
                "S2AP uncertainty (g/kg)",
                "Lower 95% SOC bound (g/kg)",
                "Upper 95% SOC bound (g/kg)",
            ]
            for band, description in enumerate(descriptions, 1):
                dst.set_band_description(band, description)
    return buffer.getvalue()


index = load_index()
benchmark = load_benchmark_results()
models, device = load_models()
baseline_backend = BaselineBackend(device)

st.markdown(
    '<div class="main-header"><span class="brand-mark">F</span> FieldSOC Intelligence</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Estimate soil health and carbon storage from satellite imagery</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="sidebar-brand"><span>FS</span> FieldSOC</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Analysis setup</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="helper-text">Choose the field you want to assess. The app compares its satellite conditions with trained soil patterns.</div>',
        unsafe_allow_html=True,
    )
    source = st.radio("Input source", ["Interactive map", "LUCAS tile", "Upload GeoTIFF"])
    st.markdown('<div class="section-label">European validation presets</div>', unsafe_allow_html=True)
    preset_cols = st.columns(3)
    for col, preset in zip(preset_cols, PRESETS):
        if col.button(preset["land"], key=f"preset_{preset['land']}"):
            st.session_state["preset_center"] = [preset["lat"], preset["lon"]]
            st.session_state["preset_label"] = preset["name"]

    selected_row = None
    uploaded_image = None
    uploaded_transform = None
    center = [48.5, 10.0]
    if "preset_center" in st.session_state:
        center = st.session_state["preset_center"]

    if source == "LUCAS tile" and not index.empty:
        point_ids = index["point_id"].astype(str).tolist()
        chosen = st.selectbox("Cached validation tile", point_ids)
        selected_row = index.loc[index["point_id"].astype(str) == chosen].iloc[0]
        center = [float(selected_row["latitude"]), float(selected_row["longitude"])]
    elif source == "Upload GeoTIFF":
        upload = st.file_uploader("12-channel Sentinel-1/2 patch", type=["tif", "tiff"])
        if upload:
            with rasterio.open(io.BytesIO(upload.getvalue())) as src:
                uploaded_image = src.read().astype(np.float32)
                uploaded_transform = src.transform
                bounds = src.bounds
                center = [(bounds.bottom + bounds.top) / 2, (bounds.left + bounds.right) / 2]
            if not in_europe(center[0], center[1]):
                st.error("Out of European LUCAS domain. Use coordinates inside 34.5°N–71.0°N and 10.5°W–40.0°E.")
                uploaded_image = None
            if uploaded_image is not None:
                st.caption(f"{uploaded_image.shape[0]} bands x {uploaded_image.shape[1]} x {uploaded_image.shape[2]} pixels")
    else:
        lat = st.number_input("Latitude", value=48.5, format="%.5f")
        lon = st.number_input("Longitude", value=10.0, format="%.5f")
        center = [lat, lon]
        if "preset_center" in st.session_state:
            center = st.session_state["preset_center"]
        if not in_europe(center[0], center[1]):
            st.error("Out of European LUCAS domain.")
        pick_map = folium.Map(
            location=center,
            zoom_start=5,
            tiles="Esri WorldImagery",
            max_bounds=True,
            min_zoom=4,
        )
        pick_map.fit_bounds([
            [EUROPE_BOUNDS["lat_min"], EUROPE_BOUNDS["lon_min"]],
            [EUROPE_BOUNDS["lat_max"], EUROPE_BOUNDS["lon_max"]],
        ])
        pick_map.add_child(folium.LatLngPopup())
        picked = st_folium(pick_map, width=300, height=220, key="coordinate_picker")
        if picked and picked.get("last_clicked"):
            clicked_lat = float(picked["last_clicked"]["lat"])
            clicked_lon = float(picked["last_clicked"]["lng"])
            if in_europe(clicked_lat, clicked_lon):
                center = [clicked_lat, clicked_lon]
                st.caption(f"Selected: {center[0]:.5f}, {center[1]:.5f}")
            else:
                st.error("Selected point is outside the European LUCAS domain.")

    mc_passes = st.slider("Monte Carlo passes", 3, 20, 8)
    st.caption("More passes give a more cautious confidence estimate, but take longer.")
    window_size = st.select_slider(
        "Area detail",
        options=[128, 256, 384],
        value=256,
        help="Smaller areas detect more local variation; larger areas provide a broader field view.",
    )
    st.caption("Controls the size of each satellite area analyzed by the model.")
    st.markdown('<div class="section-label">Baseline comparison</div>', unsafe_allow_html=True)
    baseline_model = st.selectbox(
        "Baseline model",
        ["Random Forest", "XGBoost / GBDT", "2D-CNN"],
        help="The selected baseline is compared with the trained S2AP model on the same satellite patch.",
    )
    st.markdown("---")
    run = st.button("Run SOC analysis", type="primary", use_container_width=True)

if run:
    try:
        if not in_europe(center[0], center[1]):
            st.error(
                "Out-of-distribution warning: this dashboard is restricted to "
                "34.5°N–71.0°N and 10.5°W–40.0°E."
            )
            st.stop()
        if source == "LUCAS tile" and selected_row is not None:
            point_id = int(selected_row["point_id"])
            cache_path = BASE_DIR / "tensor_cache" / f"point_{point_id}.pt"
            if not cache_path.exists():
                hf_path = download_lucas_tile(point_id)
                cache_path = hf_path if hf_path is not None else cache_path
            if not cache_path.exists():
                st.error(f"Cached tile is missing: {cache_path}")
                st.stop()
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            image = np.asarray(cached["img"], dtype=np.float32)
            coords = np.asarray(cached["coords"], dtype=np.float32)
            env = np.asarray(cached["env"], dtype=np.float32)
            mask_source = np.asarray(cached["mask"], dtype=np.float32)
            transform = from_origin(center[1] - 0.02, center[0] + 0.02, 0.00015625, 0.00015625)
        elif source == "Upload GeoTIFF" and uploaded_image is not None:
            image = normalize_upload(uploaded_image)
            coords = np.array([center[0] / 90.0, center[1] / 180.0], dtype=np.float32)
            env = default_env(selected_row)
            mask_source = np.clip(
                ((image[3] - image[2]) / (image[3] + image[2] + 1e-6) < 0.25),
                0,
                1,
            )
            transform = uploaded_transform
        elif not index.empty:
            lat, lon = center
            distances = (index["latitude"] - lat) ** 2 + (index["longitude"] - lon) ** 2
            selected_row = index.loc[distances.idxmin()]
            point_id = int(selected_row["point_id"])
            cache_path = BASE_DIR / "tensor_cache" / f"point_{point_id}.pt"
            if not cache_path.exists():
                hf_path = download_lucas_tile(point_id)
                cache_path = hf_path if hf_path is not None else cache_path
            if not cache_path.exists():
                st.error("No cached tensor exists for the nearest LUCAS location.")
                st.stop()
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            image = np.asarray(cached["img"], dtype=np.float32)
            coords = np.asarray(cached["coords"], dtype=np.float32)
            env = np.asarray(cached["env"], dtype=np.float32)
            mask_source = np.asarray(cached["mask"], dtype=np.float32)
            transform = from_origin(lon - 0.02, lat + 0.02, 0.00015625, 0.00015625)
        else:
            st.error("No data source is available.")
            st.stop()

        soc_map, uncertainty_map, seg_map, field_mean = sliding_inference(
            models, device, image, coords, env, mc_passes=mc_passes, window=window_size, stride=max(window_size // 2, 64)
        )
        env4 = np.asarray(env, dtype=np.float32).reshape(-1)[:4]
        baseline_map = baseline_backend.predict_map(
            baseline_model,
            image,
            coords,
            env4,
            window=window_size,
            stride=max(window_size // 2, 64),
        )
        baseline_map = np.asarray(baseline_map, dtype=np.float32)
        if baseline_map.shape != soc_map.shape:
            baseline_map = np.resize(baseline_map, soc_map.shape)
        residual_map = soc_map - baseline_map
        st.session_state["analysis"] = {
            "image": image,
            "soc": soc_map,
            "uncertainty": uncertainty_map,
            "seg": seg_map,
            "baseline": baseline_map,
            "residual": residual_map,
            "baseline_model": baseline_model,
            "mask_source": mask_source,
            "transform": transform,
            "center": center,
            "mean": field_mean,
        }
        st.success("SOC analysis complete")
    except Exception as exc:
        st.exception(exc)

analysis = st.session_state.get("analysis")
if not isinstance(analysis, dict):
    st.markdown(
        """
        <div class="status-card">
          <div class="idle-icon">◉</div>
          <h3>Ready for a field assessment</h3>
          <p>Select a coordinate, a cached LUCAS validation tile, or upload a 12-band GeoTIFF,
          then run the multimodal SOC analysis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

image = analysis["image"]
soc_map = analysis["soc"]
uncertainty = analysis["uncertainty"]
seg_map = analysis["seg"]
center = analysis["center"]
baseline_map = analysis["baseline"]
residual_map = analysis["residual"]
baseline_model = analysis["baseline_model"]

kpis = st.columns(4)
kpis[0].metric("Mean SOC", f"{np.mean(soc_map):.2f} g/kg")
kpis[0].caption("Average soil carbon in the analyzed field")
area_ha = soc_map.size * 0.01
carbon_stock = np.mean(soc_map) * 1.3 * 0.30 * 10
kpis[1].metric("Carbon stock", f"{carbon_stock:.2f} t C/ha")
kpis[1].caption("Estimated carbon stored per hectare")
kpis[2].metric("Estimated area", f"{area_ha:.1f} ha")
kpis[2].caption("Approximate size of this analysis")
kpis[3].metric("Mean uncertainty", f"{np.mean(uncertainty):.2f} g/kg")
kpis[3].caption("How much the estimate may vary")

s2ap_status = st.columns(1)[0]
s2ap_status.caption(
    "Primary model: trained ResUNet-a + S2AP checkpoints — compared against "
    f"{baseline_model} on the same 12-channel patch."
)

map_tab, analytics_tab, export_tab, comparison_tab = st.tabs(
    ["Spatial layers", "Analytics", "Export", "Model comparison"]
)

with map_tab:
    st.markdown('<div class="section-label">Map layers</div>', unsafe_allow_html=True)
    st.caption("Turn layers on or off to compare the satellite view, bare-soil areas, and predicted soil carbon.")
    controls = st.columns([1, 1, 1, 1])
    show_rgb = controls[0].checkbox("Sentinel-2 RGB", True)
    show_radar = controls[1].checkbox("Sentinel-1 VV/VH", True)
    show_mask = controls[2].checkbox("Bare-soil mask", True)
    show_soc = controls[3].checkbox("SOC heatmap", True)

    rgb = make_rgb(image)
    bare_mask = (image[3] - image[2]) / (image[3] + image[2] + 1e-6)
    bare_mask = ((bare_mask < 0.25) & ((image[4] - image[5]) / (image[4] + image[5] + 1e-6) < 0.10))

    nearby = index.copy()
    distance = ((nearby["latitude"] - center[0]) ** 2 + (nearby["longitude"] - center[1]) ** 2) ** 0.5
    nearby = nearby.assign(_distance=distance).sort_values("_distance").head(6)
    pin_x = []
    pin_y = []
    pin_text = []
    h, w = soc_map.shape
    for _, pin in nearby.iterrows():
        px_coord = int(np.clip((float(pin["longitude"]) - center[1] + 0.03) / 0.06 * w, 0, w - 1))
        py_coord = int(np.clip((0.02 - (float(pin["latitude"]) - center[0])) / 0.04 * h, 0, h - 1))
        pin_x.append(px_coord)
        pin_y.append(py_coord)
        pin_text.append(
            f"{int(pin['point_id'])}: Observed SOC {float(pin['oc_g_kg']):.2f} g/kg"
        )

    fig = make_subplots(
        rows=2,
        cols=2,
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.035,
        vertical_spacing=0.055,
        subplot_titles=[
            "Sentinel-2 RGB + bare-soil mask",
            "ResUNet-a + S2AP",
            f"{baseline_model} baseline",
            "S2AP minus baseline",
        ],
    )
    fig.add_trace(go.Image(z=rgb), row=1, col=1)
    if show_mask:
        mask_rgba = np.zeros((*bare_mask.shape, 4), dtype=np.uint8)
        mask_rgba[..., 1] = (bare_mask * 255).astype(np.uint8)
        mask_rgba[..., 3] = (bare_mask * 150).astype(np.uint8)
        fig.add_trace(go.Image(z=mask_rgba, name="Bare-soil mask"), row=1, col=1)
    fig.add_trace(go.Heatmap(
        z=soc_map,
        zmin=np.nanpercentile(soc_map, 2),
        zmax=np.nanpercentile(soc_map, 98),
        colorscale="Viridis",
        showscale=True,
        colorbar=dict(title="g/kg", len=0.46, y=0.78),
    ), row=1, col=2)
    fig.add_trace(go.Heatmap(
        z=baseline_map,
        zmin=np.nanpercentile(soc_map, 2),
        zmax=np.nanpercentile(soc_map, 98),
        colorscale="Viridis",
        showscale=True,
        colorbar=dict(title="g/kg", len=0.46, y=0.78),
    ), row=2, col=1)
    residual_limit = max(float(np.nanpercentile(np.abs(residual_map), 98)), 1.0)
    fig.add_trace(go.Heatmap(
        z=residual_map,
        zmin=-residual_limit,
        zmax=residual_limit,
        colorscale="RdBu",
        showscale=True,
        colorbar=dict(title="Δ g/kg", len=0.46, y=0.22),
    ), row=2, col=2)
    for row, col in [(1, 1), (1, 2), (2, 1), (2, 2)]:
        fig.add_trace(go.Scatter(
            x=pin_x,
            y=pin_y,
            mode="markers",
            marker=dict(size=9, color="#f4c542", line=dict(width=1.5, color="#17251f")),
            text=pin_text,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ), row=row, col=col)
    fig.update_layout(
        height=720,
        hovermode="x unified",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#17251f", family="Inter, Segoe UI, sans-serif"),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

    bounds = [
        [center[0] - 0.02, center[1] - 0.03],
        [center[0] + 0.02, center[1] + 0.03],
    ]
    fmap = folium.Map(location=center, zoom_start=14, tiles="Esri WorldImagery")

    if show_rgb:
        rgb = make_rgb(image)
        folium.raster_layers.ImageOverlay(
            rgb, bounds=bounds, opacity=0.82, name="Sentinel-2 RGB"
        ).add_to(fmap)
    if show_radar:
        radar = np.stack([np.clip(image[6], 0, 1), np.clip(image[7], 0, 1)], axis=-1)
        radar = np.concatenate([radar, np.ones((*radar.shape[:2], 1), dtype=np.float32)], axis=-1)
        radar = (radar * 255).astype(np.uint8)
        folium.raster_layers.ImageOverlay(
            radar, bounds=bounds, opacity=0.58, name="Sentinel-1 VV/VH"
        ).add_to(fmap)
    if show_mask:
        mask_rgb = np.zeros((*seg_map.shape, 4), dtype=np.uint8)
        mask_rgb[..., 1] = (seg_map * 255).astype(np.uint8)
        mask_rgb[..., 3] = (seg_map * 180).astype(np.uint8)
        folium.raster_layers.ImageOverlay(
            mask_rgb, bounds=bounds, opacity=0.65, name="Predicted bare-soil mask"
        ).add_to(fmap)
    if show_soc:
        soc_rgb = safe_overlay(soc_map, "turbo")
        folium.raster_layers.ImageOverlay(
            soc_rgb, bounds=bounds, opacity=0.78, name="SOC heatmap (g/kg)"
        ).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    st.markdown('<div class="map-shell">', unsafe_allow_html=True)
    st_folium(fmap, width=None, height=520)
    st.markdown("</div>", unsafe_allow_html=True)

with analytics_tab:
    st.markdown('<div class="section-label">SOC distribution</div>', unsafe_allow_html=True)
    st.caption("SOC is soil organic carbon in grams per kilogram. Higher values generally indicate healthier, more carbon-rich soil.")
    c1, c2 = st.columns([1, 1])
    with c1:
        low = float(np.sum(soc_map < 10))
        medium = float(np.sum((soc_map >= 10) & (soc_map < 20)))
        high = float(np.sum(soc_map >= 20))
        total = max(low + medium + high, 1)
        zone = pd.DataFrame({
            "Zone": ["Low <10", "Medium 10-20", "High >=20"],
            "Share": [low / total * 100, medium / total * 100, high / total * 100],
        })
        fig = px.bar(zone, x="Zone", y="Share", color="Zone", text_auto=".1f")
        fig.update_layout(
            showlegend=False,
            yaxis_title="Pixel share (%)",
            height=300,
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#17251f", family="Inter, Segoe UI, sans-serif"),
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        values = soc_map.ravel()
        fig = px.histogram(values, nbins=30, labels={"value": "SOC (g/kg)"}, opacity=0.88)
        fig.update_layout(
            showlegend=False,
            height=300,
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#17251f", family="Inter, Segoe UI, sans-serif"),
        )
        st.plotly_chart(fig, use_container_width=True)

    u1, u2 = st.columns([1, 1])
    with u1:
        y = np.mean(soc_map)
        y0 = np.clip(y - 1.96 * np.mean(uncertainty), 0, None)
        y1 = y + 1.96 * np.mean(uncertainty)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=["95% uncertainty interval"],
            y=[y],
            mode="markers",
            error_y=dict(type="data", array=[y1 - y], arrayminus=[y - y0]),
            marker=dict(size=12, color="#2d6a4f"),
        ))
        fig.update_layout(
            yaxis_title="SOC (g/kg)",
            height=280,
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#17251f", family="Inter, Segoe UI, sans-serif"),
        )
        st.plotly_chart(fig, use_container_width=True)
    with u2:
        st.write("**Soil health interpretation**")
        mean_soc = float(np.mean(soc_map))
        if mean_soc < 10:
            st.error("Low SOC zone: prioritize cover crops and organic amendments.")
        elif mean_soc < 20:
            st.warning("Moderate SOC: maintain residue cover and optimize rotations.")
        else:
            st.success("High SOC: current soil carbon appears favorable.")

with export_tab:
    st.markdown('<div class="section-label">One-click reports</div>', unsafe_allow_html=True)
    east, west = st.columns(2)
    transform = analysis["transform"]
    east.download_button(
        "Download SOC GeoTIFF",
        data=geotiff_bytes(
            soc_map.astype(np.float32),
            baseline_map.astype(np.float32),
            residual_map.astype(np.float32),
            seg_map.astype(np.float32),
            uncertainty.astype(np.float32),
            transform,
        ),
        file_name="predicted_soc.tif",
        mime="image/tiff",
        use_container_width=True,
    )

    rows, cols = soc_map.shape
    ys, xs = np.indices((rows, cols))
    csv_df = pd.DataFrame({
        "row": ys.ravel(),
        "column": xs.ravel(),
        "s2ap_soc_g_kg": soc_map.ravel(),
        "baseline_soc_g_kg": baseline_map.ravel(),
        "delta_soc_g_kg": residual_map.ravel(),
        "uncertainty_g_kg": uncertainty.ravel(),
        "bare_soil_probability": seg_map.ravel(),
        "lower_95_g_kg": np.clip(soc_map.ravel() - 1.96 * uncertainty.ravel(), 0, None),
        "upper_95_g_kg": soc_map.ravel() + 1.96 * uncertainty.ravel(),
    })
    west.download_button(
        "Download soil health CSV",
        data=csv_df.to_csv(index=False).encode("utf-8"),
        file_name="soil_health_report.csv",
        mime="text/csv",
        use_container_width=True,
    )

with comparison_tab:
    st.markdown('<div class="section-label">Model benchmark comparison</div>', unsafe_allow_html=True)
    st.caption(
        "Exact four-model SBCV comparison. Random Forest and XGBoost use trained leak-free baseline artifacts; "
        "2D-CNN uses its trained folded model; S2AP uses pooled SBCV laboratory benchmark metrics."
    )

    comparison = benchmark.copy()
    comparison["Rank"] = comparison["R2"].rank(ascending=False, method="min").astype(int)
    comparison = comparison.sort_values(["R2"], ascending=False)

    metric1, metric2, metric3 = st.columns(3)
    winner = comparison.iloc[0]
    metric1.metric("Highest R²", winner["Model"], f"{winner['R2']:.4f}")
    metric2.metric("Lowest RMSE", comparison.sort_values("RMSE g/kg").iloc[0]["Model"],
                   f"{comparison['RMSE g/kg'].min():.2f} g/kg")
    metric3.metric("Lowest MAE", comparison.sort_values("MAE g/kg").iloc[0]["Model"],
                   f"{comparison['MAE g/kg'].min():.2f} g/kg")

    c1, c2 = st.columns([1.55, 1])
    with c1:
        fig = px.bar(
            comparison,
            x="R2",
            y="Model",
            orientation="h",
            color="R2",
            color_continuous_scale=["#d7e8de", "#1f6a4d"],
            text="R2",
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(
            height=560,
            yaxis={"categoryorder": "total ascending"},
            xaxis_title="R² (higher is better)",
            coloraxis_showscale=False,
            margin=dict(l=10, r=30, t=10, b=40),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#17251f", family="Inter, Segoe UI, sans-serif"),
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="RMSE",
            x=comparison["Model"],
            y=comparison["RMSE g/kg"],
            marker_color="#c88b42",
        ))
        fig.add_trace(go.Bar(
            name="MAE",
            x=comparison["Model"],
            y=comparison["MAE g/kg"],
            marker_color="#1f6a4d",
        ))
        fig.update_layout(
            barmode="group",
            height=560,
            xaxis_tickangle=-55,
            yaxis_title="Error (g/kg)",
            legend=dict(orientation="h", y=1.02),
            margin=dict(l=10, r=10, t=50, b=120),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#17251f", family="Inter, Segoe UI, sans-serif"),
        )
        st.plotly_chart(fig, use_container_width=True)

    display = comparison[
        ["Rank", "Model", "R2", "MAE g/kg", "RMSE g/kg", "RPD"]
    ].rename(columns={"R2": "R²"})
    st.markdown('<div class="section-label">Complete benchmark table</div>', unsafe_allow_html=True)
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn("#", format="%d"),
            "R²": st.column_config.NumberColumn(format="%.4f"),
            "MAE g/kg": st.column_config.NumberColumn(format="%.2f"),
            "RMSE g/kg": st.column_config.NumberColumn(format="%.2f"),
            "RPD": st.column_config.NumberColumn(format="%.2f"),
        },
    )
