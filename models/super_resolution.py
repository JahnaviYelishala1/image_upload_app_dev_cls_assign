from pathlib import Path
import importlib

import numpy as np
import streamlit as st

from utils.image_utils import ensure_same_size


MODEL_FILENAME = "RealESRGAN_x4plus.pth"
WEIGHTS_PATH = Path("models") / MODEL_FILENAME
MODEL_URLS = [
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4plus.pth",
]


def _has_valid_weights(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def download_weights(weights_path: Path = WEIGHTS_PATH) -> Path:
    # Import downloader dependencies lazily to avoid import-time crashes.
    try:
        requests_module = importlib.import_module("requests")
    except Exception as exc:
        raise RuntimeError("Missing dependency: requests. Install it with pip.") from exc

    try:
        tqdm_cls = getattr(importlib.import_module("tqdm"), "tqdm")
    except Exception:
        tqdm_cls = None

    # Ensure models directory exists before any download attempt.
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    # Use local file if already present and non-empty.
    if _has_valid_weights(weights_path):
        return weights_path

    last_error: Exception | None = None

    # Retry once on failure using a secondary mirror URL.
    for attempt in range(2):
        url = MODEL_URLS[min(attempt, len(MODEL_URLS) - 1)]
        tmp_path = weights_path.with_suffix(".tmp")
        progress_bar = st.progress(0.0, text=f"Downloading {MODEL_FILENAME} (attempt {attempt + 1}/2)")

        try:
            with requests_module.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0

                with open(tmp_path, "wb") as output_file:
                    if tqdm_cls is not None:
                        progress_context = tqdm_cls(
                            total=total_size if total_size > 0 else None,
                            unit="B",
                            unit_scale=True,
                            desc=f"Real-ESRGAN attempt {attempt + 1}",
                        )
                    else:
                        progress_context = None

                    if progress_context is not None:
                        progress_iter = progress_context
                    else:
                        class _NoOpProgress:
                            def __enter__(self):
                                return None

                            def __exit__(self, exc_type, exc, tb):
                                return False

                        progress_iter = _NoOpProgress()

                    with progress_iter as progress_console:
                        for chunk in response.iter_content(chunk_size=8192):
                            if not chunk:
                                continue
                            output_file.write(chunk)
                            downloaded += len(chunk)
                            if progress_console is not None:
                                progress_console.update(len(chunk))

                            if total_size > 0:
                                progress_bar.progress(
                                    min(downloaded / total_size, 1.0),
                                    text=f"Downloading {MODEL_FILENAME} (attempt {attempt + 1}/2)",
                                )

            tmp_path.replace(weights_path)

            # Basic integrity check required by the pipeline.
            if not _has_valid_weights(weights_path):
                raise RuntimeError("Downloaded model file is invalid or empty.")

            progress_bar.progress(1.0, text=f"Downloaded {MODEL_FILENAME}")
            return weights_path
        except Exception as exc:
            last_error = exc
            progress_bar.empty()
            if tmp_path.exists():
                tmp_path.unlink()

    raise RuntimeError(
        f"Failed to download {MODEL_FILENAME} after 2 attempts. "
        f"Last error: {last_error}"
    )


@st.cache_resource(show_spinner=False)
def load_realesrgan_model():
    # Import dependencies lazily so the app can still start without optional packages.
    try:
        rrdb_module = importlib.import_module("basicsr.archs.rrdbnet_arch")
        realesrgan_module = importlib.import_module("realesrgan")
        torch_module = importlib.import_module("torch")

        RRDBNet = getattr(rrdb_module, "RRDBNet")
        RealESRGANer = getattr(realesrgan_module, "RealESRGANer")
    except Exception as exc:
        raise RuntimeError(f"Real-ESRGAN dependencies are unavailable: {exc}") from exc

    # Load local weights only; download happens only if file is missing/invalid.
    weights_path = download_weights(WEIGHTS_PATH)

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    )

    # Use CPU by default and GPU only when available.
    use_gpu = bool(torch_module.cuda.is_available())

    try:
        upsampler = RealESRGANer(
            scale=4,
            model_path=str(weights_path),
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=use_gpu,
            gpu_id=0 if use_gpu else None,
        )
        return upsampler
    except Exception as exc:
        raise RuntimeError(f"Failed to load Real-ESRGAN model: {exc}") from exc


def apply_realesrgan(image: np.ndarray) -> np.ndarray:
    try:
        upsampler = load_realesrgan_model()
        output, _ = upsampler.enhance(image, outscale=2)
        return ensure_same_size(image, output)
    except Exception as exc:
        st.error(f"Enhancement fallback: {exc}")
        return image
