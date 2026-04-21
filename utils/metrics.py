import cv2
import numpy as np

from utils.image_utils import ensure_same_size


def compute_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    aligned = ensure_same_size(original, processed)
    return float(cv2.PSNR(original, aligned))


def compute_ssim(original: np.ndarray, processed: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity as ssim

        aligned = ensure_same_size(original, processed)
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        processed_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        return float(ssim(original_gray, processed_gray))
    except ImportError as exc:
        raise RuntimeError(
            "SSIM is unavailable because scikit-image is not installed in this environment. "
            "Install it with: pip install scikit-image"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"SSIM computation failed: {exc}") from exc
