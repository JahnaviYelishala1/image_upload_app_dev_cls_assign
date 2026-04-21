from pathlib import Path
import io

import cv2
import numpy as np
from PIL import Image


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    rgb = np.array(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv2_to_pil(image: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def clamp_quality(quality: int) -> int:
    return int(np.clip(quality, 0, 100))


def image_format_from_suffix(suffix: str) -> str:
    suffix = suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "JPEG"
    return "PNG"


def normalize_image_array(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a numpy array.")

    if image.size == 0:
        raise ValueError("Input image is empty.")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        return image

    if image.ndim != 3:
        raise ValueError("Input image must be 2D (grayscale) or 3D (color).")

    channels = image.shape[2]
    if channels == 1:
        return image[:, :, 0]
    if channels == 3:
        return image
    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    raise ValueError("Unsupported channel count. Expected 1, 3, or 4 channels.")


def ensure_same_size(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    src_h, src_w = source.shape[:2]
    tgt_h, tgt_w = target.shape[:2]
    if (src_h, src_w) == (tgt_h, tgt_w):
        return target
    return cv2.resize(target, (src_w, src_h), interpolation=cv2.INTER_CUBIC)


def size_kb_from_bytes(data: bytes) -> float:
    return len(data) / 1024.0


def output_filename(original_name: str, fmt: str) -> str:
    stem = Path(original_name).stem or "processed_image"
    extension = ".jpg" if fmt == "JPEG" else ".png"
    return f"{stem}_processed{extension}"


def output_mime(fmt: str) -> str:
    return "image/jpeg" if fmt == "JPEG" else "image/png"


def save_image_to_bytes(
    image: np.ndarray,
    quality: int = 70,
    fmt: str = "JPEG",
    target_max_bytes: int | None = None,
) -> bytes:
    safe_image = normalize_image_array(image)
    buffer = io.BytesIO()

    if fmt == "JPEG":
        # Use PIL save with explicit quality and optimize to ensure JPEG compression is applied.
        result_pil = cv2_to_pil(safe_image).convert("RGB")
        current_quality = int(np.clip(quality, 40, 80))

        while True:
            buffer.seek(0)
            buffer.truncate(0)
            result_pil.save(buffer, format="JPEG", quality=current_quality, optimize=True)
            compressed = buffer.getvalue()

            # Best effort to keep compressed output smaller when compression is selected.
            if target_max_bytes is None or len(compressed) < target_max_bytes or current_quality <= 40:
                return compressed

            current_quality = max(40, current_quality - 5)
    else:
        result_pil = cv2_to_pil(safe_image)
        result_pil.save(buffer, format="PNG", optimize=True, compress_level=9)
        return buffer.getvalue()
