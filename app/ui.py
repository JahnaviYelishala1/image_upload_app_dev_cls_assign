from pathlib import Path
import io

import streamlit as st
from PIL import Image

from app.processing import MODES, detect_mode_from_prompt, process_image
from utils.image_utils import (
    cv2_to_pil,
    image_format_from_suffix,
    output_filename,
    output_mime,
    pil_to_cv2,
    save_image_to_bytes,
    size_kb_from_bytes,
)
from utils.metrics import compute_psnr, compute_ssim


def render_app() -> None:
    st.set_page_config(page_title="Image Compression & Enhancement", layout="wide")
    st.title("Image Compression & Enhancement")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        return

    original_file_bytes = uploaded_file.getvalue()
    original_image = Image.open(uploaded_file).convert("RGB")
    original_bgr = pil_to_cv2(original_image)
    file_format = image_format_from_suffix(Path(uploaded_file.name).suffix)

    user_prompt = st.text_input("What do you want to do with the image?")
    selected_mode = st.selectbox("Choose processing option", MODES)

    detected_mode = detect_mode_from_prompt(user_prompt, selected_mode)
    st.info(f"Detected mode: {detected_mode}")

    quality = st.slider("Compression quality", min_value=40, max_value=80, value=70)

    if st.button("Process Image", width="stretch"):
        processed_image = process_image(original_bgr, detected_mode, quality)

        # Force JPEG output for compression modes so quality and optimize are always applied.
        output_format = "JPEG" if detected_mode in {"Compress", "Compress + Enhance"} else file_format
        processed_bytes = save_image_to_bytes(
            processed_image,
            quality=quality,
            fmt=output_format,
            target_max_bytes=len(original_file_bytes) if detected_mode in {"Compress", "Compress + Enhance"} else None,
        )
        # Decode from final bytes so preview and metrics match the exact downloadable file.
        processed_image_pil = Image.open(io.BytesIO(processed_bytes)).convert("RGB")
        processed_bgr_for_metrics = pil_to_cv2(processed_image_pil)

        left_col, right_col = st.columns(2)
        with left_col:
            st.image(original_image, caption="Original", width="stretch")
        with right_col:
            st.image(processed_image_pil, caption="Processed", width="stretch")

        original_size_kb = size_kb_from_bytes(original_file_bytes)
        processed_size_kb = size_kb_from_bytes(processed_bytes)
        psnr_score = compute_psnr(original_bgr, processed_bgr_for_metrics)

        try:
            ssim_score = compute_ssim(original_bgr, processed_bgr_for_metrics)
        except Exception as exc:
            st.warning(str(exc))
            ssim_score = 0.0

        compression_ratio = (processed_size_kb / original_size_kb) if original_size_kb > 0 else float("inf")

        st.subheader("Debug Output")
        st.write(f"Original file size: {original_size_kb:.2f} KB")
        st.write(f"Processed file size: {processed_size_kb:.2f} KB")
        st.write(f"Compression ratio (processed/original): {compression_ratio:.3f}")
        if detected_mode in {"Compress", "Compress + Enhance"} and processed_size_kb >= original_size_kb:
            st.warning("Compressed file is not smaller than original for this image. Try lowering quality.")

        st.subheader("Quality Comparison")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("Original Size (KB)", f"{original_size_kb:.2f}")
        metric_col2.metric("Processed Size (KB)", f"{processed_size_kb:.2f}")
        metric_col3.metric("PSNR", f"{psnr_score:.2f} dB")
        metric_col4.metric("SSIM", f"{ssim_score:.4f}")

        st.download_button(
            label="Download Processed Image",
            data=processed_bytes,
            file_name=output_filename(uploaded_file.name, output_format),
            mime=output_mime(output_format),
            width="stretch",
        )
