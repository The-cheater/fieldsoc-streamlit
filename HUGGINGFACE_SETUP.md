# Hugging Face setup for LUCAS tensor data

The app downloads one tensor only when a point is selected.

## 1. Create a Hugging Face account

1. Open https://huggingface.co/join
2. Create an account and sign in.
3. Open https://huggingface.co/settings/tokens
4. Create a read token.

If the dataset will be public, the app does not need the token. Keeping the dataset public is simplest.

## 2. Create a dataset repository

1. Open https://huggingface.co/new-dataset
2. Enter a repository name, for example `lucas-soc-tensors`.
3. Set visibility to Public or Private.
4. Click **Create dataset**.

## 3. Upload the folder structure

Upload all 17,823 files under this exact structure:

```text
tensor_cache/
  point_0.pt
  point_1.pt
  point_2.pt
  ...
  point_17822.pt
```

Do not upload `SENTNEL1.PY`, `satellite.docx`, raster directories, images, or checkpoints to the data repository.

The only required Hugging Face data file for each selected point is:

```text
tensor_cache/point_<point_id>.pt
```

Each file must contain the dictionary keys:

```text
img
mask
coords
env
target
```

The `img` tensor must have shape `[12, 256, 256]`.

## 4. Configure the Streamlit app

Open the Streamlit app settings, select **Secrets**, and add:

```toml
HF_DATASET_REPO = "YOUR_HUGGINGFACE_USERNAME/lucas-soc-tensors"
```

For a private dataset, also add:

```toml
HF_TOKEN = "hf_your_read_token"
```

Do not put the token in GitHub or in `app.py`.

## 5. Upload data from Windows

Install the Hugging Face CLI and upload the folder:

```powershell
cd /d E:\sat
.\.venv\Scripts\python.exe -m pip install huggingface_hub
$env:HF_TOKEN="hf_your_read_token"
hf auth login
hf upload YOUR_HUGGINGFACE_USERNAME/lucas-soc-tensors tensor_cache tensor_cache --repo-type dataset
```

The upload command may take several hours because the cache contains thousands of 5 MB tensors.

## 6. Alternative: smaller demonstration subset

Upload only the numbered files already included in `deploy/tensor_cache/`, but the LUCAS tile selector may then show points that are missing. Either restrict the CSV to matching IDs or upload the complete cache.

## 7. Redeploy

Push the updated app and dependency files to GitHub, then reboot the Streamlit app. The app will download each missing `point_<id>.pt` from Hugging Face automatically.
