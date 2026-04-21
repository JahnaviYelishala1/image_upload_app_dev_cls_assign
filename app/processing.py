import cv2
import numpy as np

from models.super_resolution import apply_realesrgan
from utils.image_utils import clamp_quality, normalize_image_array


MODES = ["Compress", "Enhance", "Compress + Enhance"]


def detect_mode_from_prompt(user_text: str, selected_mode: str) -> str:
    prompt = (user_text or "").strip().lower()

    if "both" in prompt:
        return "Compress + Enhance"
    if "compress" in prompt:
        return "Compress"
    if "enhance" in prompt:
        return "Enhance"

    return selected_mode


def compress_image(image: np.ndarray, quality: int) -> np.ndarray:
    safe_image = normalize_image_array(image)
    # Keep quality in a practical range to make compression effect visible.
    safe_quality = int(np.clip(clamp_quality(quality), 40, 80))

    success, encoded = cv2.imencode(
        ".jpg",
        safe_image,
        [int(cv2.IMWRITE_JPEG_QUALITY), safe_quality, int(cv2.IMWRITE_JPEG_OPTIMIZE), 1],
    )
    if not success or encoded is None:
        raise ValueError("JPEG encoding failed.")

    decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise ValueError("JPEG decoding failed.")

    if safe_image.ndim == 3 and safe_image.shape[2] == 3 and decoded.ndim == 2:
        decoded = cv2.cvtColor(decoded, cv2.COLOR_GRAY2BGR)

    return decoded


def enhance_image_clahe_denoise(image: np.ndarray) -> np.ndarray:
    safe_image = normalize_image_array(image)

    if safe_image.ndim == 2:
        bgr = cv2.cvtColor(safe_image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = safe_image

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    merged = cv2.merge((l_enhanced, a_channel, b_channel))
    contrast_enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    denoised = cv2.fastNlMeansDenoisingColored(
        contrast_enhanced,
        None,
        h=5,
        hColor=5,
        templateWindowSize=7,
        searchWindowSize=21,
    )
    return denoised


def enhance_image(image: np.ndarray, use_super_resolution: bool = False) -> np.ndarray:
    base_enhanced = enhance_image_clahe_denoise(image)
    if use_super_resolution:
        return apply_realesrgan(base_enhanced)
    return base_enhanced


def process_image(image: np.ndarray, action: str, quality: int) -> np.ndarray:
    if action == "Compress":
        return compress_image(image, quality)
    if action == "Enhance":
        return enhance_image(image, use_super_resolution=True)

    compressed = compress_image(image, quality)
    return enhance_image(compressed, use_super_resolution=False)
