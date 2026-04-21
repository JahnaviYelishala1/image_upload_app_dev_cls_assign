# Image Compression & Enhancement

A modular Streamlit app for image compression and enhancement with OpenCV, metrics (PSNR/SSIM), and optional Real-ESRGAN super-resolution.

## Project Structure

```text
image_upload/
  main.py
  app/
    __init__.py
    ui.py
    processing.py
  utils/
    __init__.py
    image_utils.py
    metrics.py
  models/
    __init__.py
    super_resolution.py
  .streamlit/
    config.toml
  requirements.txt
  app.py
```

## Run Locally

1. Create and activate virtual environment (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run app:

```powershell
streamlit run main.py
```

## Streamlit Cloud Notes

- Entry point: `main.py`
- Dependencies are listed in `requirements.txt`
- Uses `opencv-python-headless` for cloud compatibility
- Real-ESRGAN is loaded lazily and cached; if unavailable, enhancement falls back gracefully
