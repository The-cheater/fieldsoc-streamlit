# FieldSOC Intelligence SaaS

Streamlit dashboard for SOC inference with the repository's cached LUCAS tensors and trained
`OptimizedMultimodalSOCNet` checkpoints.

## Run

```powershell
cd E:\sat\saas
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app uses `checkpoints/` for model weights, `tensor_cache/` for validation tiles, and
`production_multimodal_master_index.csv` for location metadata.
