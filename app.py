import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
import os
import time
import urllib.request
import json
from pathlib import Path
from typing import Optional

from utils import load_cyclegan, translate_image, create_sketch_from_image

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
DEFAULT_CHECKPOINT = MODELS_DIR / "best_model.pt"

# GitHub repo details for LFS download
GITHUB_REPO = "Huzzi122201/GenAI_Q2"
LFS_FILE_PATH = "models/best_model.pt"


def _is_lfs_pointer(filepath: Path) -> bool:
    """Check if a file is an LFS pointer (small text file) instead of the real binary."""
    if not filepath.is_file():
        return False
    if filepath.stat().st_size > 1024:
        return False
    try:
        text = filepath.read_text(encoding="utf-8", errors="ignore")
        return text.startswith("version https://git-lfs.github.com")
    except Exception:
        return False


def _parse_lfs_pointer(filepath: Path) -> Optional[str]:
    """Extract the OID from an LFS pointer file."""
    try:
        text = filepath.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("oid sha256:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


def _download_from_github_lfs(oid: str, dest: Path, progress_bar=None) -> Path:
    """Download the real file from GitHub LFS using the batch API."""
    batch_url = f"https://github.com/{GITHUB_REPO}.git/info/lfs/objects/batch"
    payload = json.dumps({
        "operation": "download",
        "transfer": ["basic"],
        "objects": [{"oid": oid, "size": 0}],
    }).encode("utf-8")

    req = urllib.request.Request(
        batch_url,
        data=payload,
        headers={
            "Content-Type": "application/vnd.git-lfs+json",
            "Accept": "application/vnd.git-lfs+json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    obj = data["objects"][0]
    if "error" in obj:
        raise RuntimeError(f"LFS error: {obj['error']['message']}")

    download_url = obj["actions"]["download"]["href"]
    headers = obj["actions"]["download"].get("header", {})

    dl_req = urllib.request.Request(download_url)
    for k, v in headers.items():
        dl_req.add_header(k, v)

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".pt.downloading")

    try:
        with urllib.request.urlopen(dl_req, timeout=300) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 256
            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_bar and total > 0:
                        progress_bar.progress(
                            min(downloaded / total, 1.0),
                            text=f"Downloading model… {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB",
                        )
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)

    return dest


def _download_from_url(url: str, dest: Path, progress_bar=None) -> Path:
    """Download checkpoint from a direct HTTPS URL."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".pt.downloading")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CycleGAN-Streamlit/1.0"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 256
            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_bar and total > 0:
                        progress_bar.progress(
                            min(downloaded / total, 1.0),
                            text=f"Downloading model… {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB",
                        )
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return dest


def ensure_checkpoint(progress_bar=None) -> Path:
    """
    Make sure the real checkpoint binary exists at DEFAULT_CHECKPOINT.
    Handles three scenarios:
      1. Real .pt file already present -> return it
      2. LFS pointer file present -> download real binary from GitHub LFS
      3. Neither present -> try CHECKPOINT_URL from secrets/env
    """
    if DEFAULT_CHECKPOINT.is_file() and not _is_lfs_pointer(DEFAULT_CHECKPOINT):
        return DEFAULT_CHECKPOINT

    if _is_lfs_pointer(DEFAULT_CHECKPOINT):
        oid = _parse_lfs_pointer(DEFAULT_CHECKPOINT)
        if oid:
            real_dest = MODELS_DIR / "best_model_real.pt"
            if real_dest.is_file() and real_dest.stat().st_size > 1024:
                return real_dest
            _download_from_github_lfs(oid, real_dest, progress_bar)
            return real_dest

    url = None
    try:
        url = st.secrets.get("CHECKPOINT_URL") or st.secrets.get("CYCLEGAN_CHECKPOINT_URL")
    except Exception:
        pass
    if not url:
        url = os.environ.get("CHECKPOINT_URL") or os.environ.get("CYCLEGAN_CHECKPOINT_URL")

    if url:
        real_dest = MODELS_DIR / "best_model_real.pt"
        if real_dest.is_file() and real_dest.stat().st_size > 1024:
            return real_dest
        _download_from_url(url, real_dest, progress_bar)
        return real_dest

    raise FileNotFoundError(
        f"Checkpoint not found at {DEFAULT_CHECKPOINT} and no CHECKPOINT_URL configured.\n"
        f"On Streamlit Cloud, set CHECKPOINT_URL in App Settings → Secrets."
    )


# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CycleGAN: Sketch-Photo Translation",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        transition: 0.3s;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-header">'
    '<h1>🔄 CycleGAN: Sketch ↔ Photo Translation</h1>'
    '<p>Translate between sketches and photos using unpaired image-to-image translation</p>'
    '</div>',
    unsafe_allow_html=True,
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📁 Model Configuration")

    device_option = st.selectbox("Device", ["Auto", "CPU", "CUDA"])
    if device_option == "Auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_option.lower()
    st.info(f"🖥️ Using device: **{device.upper()}**")

    if st.button("🚀 Load Model", use_container_width=True):
        try:
            progress = st.progress(0, text="Checking checkpoint…")
            ckpt_path = ensure_checkpoint(progress_bar=progress)
            progress.progress(1.0, text="Loading into PyTorch…")

            g_ab, g_ba = load_cyclegan(str(ckpt_path), device)
            st.session_state["g_ab"] = g_ab
            st.session_state["g_ba"] = g_ba
            st.session_state["model_loaded"] = True

            progress.empty()
            if g_ba is None:
                st.success("✅ Loaded g_ab only (Sketch→Photo). Reverse direction unavailable.")
            else:
                st.success("✅ Both generators loaded (Sketch→Photo & Photo→Sketch)")
        except FileNotFoundError as e:
            st.error(f"❌ {e}")
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")

    st.markdown("---")
    st.markdown("### 📖 Instructions")
    st.markdown("""
    1. **Load Model** – click the button above
    2. **Choose Direction** – Sketch→Photo or Photo→Sketch
    3. **Upload Image** – upload your input
    4. **Translate** – click the translate button
    5. **Download** – save the result
    """)

    st.markdown("---")
    st.markdown("### 🎯 Features")
    st.markdown("""
    - 🔄 Bidirectional translation
    - 🖌️ Sketch from photo converter
    - 📊 Real-time inference
    - 💾 Download results
    - 🚀 GPU acceleration
    """)

# ── Session state defaults ────────────────────────────────────────────────────
for key, default in [
    ("model_loaded", False),
    ("g_ab", None),
    ("g_ba", None),
    ("generated_image", None),
    ("generation_time", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Translation direction ─────────────────────────────────────────────────────
direction = st.radio(
    "Translation Direction",
    ["Sketch → Photo", "Photo → Sketch"],
    horizontal=True,
)

# ── Main area ─────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("## 📤 Input")

    input_method = st.radio(
        "Choose input method:",
        ["Upload Image", "Draw Sketch"],
        horizontal=True,
    )

    input_image = None

    if input_method == "Upload Image":
        uploaded_file = st.file_uploader(
            "Upload a sketch or photo",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
        )
        if uploaded_file is not None:
            input_image = np.array(Image.open(uploaded_file).convert("RGB"))
            st.image(input_image, caption="Uploaded Image", use_container_width=True)

    elif input_method == "Draw Sketch":
        st.info("Draw your sketch below")
        try:
            from streamlit_drawable_canvas import st_canvas

            drawing_mode = st.sidebar.selectbox(
                "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
            )
            stroke_width = st.sidebar.slider("Stroke width:", 1, 25, 3)
            stroke_color = st.sidebar.color_picker("Stroke color:", "#000000")
            bg_color = st.sidebar.color_picker("Background color:", "#FFFFFF")

            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 1)",
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color=bg_color,
                update_streamlit=True,
                height=400,
                width=400,
                drawing_mode=drawing_mode,
                key="canvas",
            )
            if canvas_result.image_data is not None:
                input_image = canvas_result.image_data.astype(np.uint8)
                if input_image.shape[2] == 4:
                    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGBA2RGB)
                st.image(input_image, caption="Your Drawing", use_container_width=True)
        except ImportError:
            st.warning("Install `streamlit-drawable-canvas` for drawing support.")

    if input_method == "Upload Image" and input_image is not None:
        if direction == "Sketch → Photo":
            with st.expander("🛠️ Don't have a sketch? (Photo-to-Sketch helper)"):
                st.caption("If you uploaded a **real photo** but want to test the Sketch→Photo model, you can extract a basic sketch from it first using this utility.")
                if st.button("🖌️ Extract Sketch from Photo"):
                    with st.spinner("Converting to sketch..."):
                        input_image = create_sketch_from_image(input_image)
                        st.image(input_image, caption="Converted Sketch", use_container_width=True)

    st.markdown("---")
    if st.button("🔄 Translate Image", use_container_width=True, type="primary"):
        if not st.session_state["model_loaded"]:
            st.error("❌ Please load a model first!")
        elif input_image is None:
            st.error("❌ Please provide an input image!")
        else:
            if direction == "Sketch → Photo":
                gen = st.session_state["g_ab"]
            else:
                gen = st.session_state["g_ba"]

            if gen is None:
                st.error("❌ Generator for this direction was not found in the checkpoint.")
            else:
                with st.spinner("Translating..."):
                    try:
                        start = time.time()
                        output = translate_image(gen, input_image, device)
                        elapsed = time.time() - start
                        st.session_state["generated_image"] = output
                        st.session_state["generation_time"] = elapsed
                        st.success(f"✅ Done in {elapsed:.2f}s")
                    except Exception as e:
                        st.error(f"❌ Translation failed: {e}")

with col2:
    st.markdown("## 🎨 Output")

    if st.session_state["generated_image"] is not None:
        st.image(
            st.session_state["generated_image"],
            caption=f"Translated Output ({direction})",
            use_container_width=True,
        )

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Inference Time", f"{st.session_state['generation_time']:.2f}s")
        with m2:
            h, w = st.session_state["generated_image"].shape[:2]
            st.metric("Output Size", f"{h}x{w}")

        from io import BytesIO
        buf = BytesIO()
        Image.fromarray(st.session_state["generated_image"]).save(buf, format="PNG")
        st.download_button(
            label="📥 Download Result",
            data=buf.getvalue(),
            file_name="cyclegan_output.png",
            mime="image/png",
            use_container_width=True,
        )

        if input_image is not None:
            st.markdown("---")
            st.markdown("### 🔍 Side-by-Side Comparison")
            original_display = cv2.resize(input_image, (128, 128))
            generated_display = st.session_state["generated_image"]

            if len(original_display.shape) == 2:
                original_display = cv2.cvtColor(original_display, cv2.COLOR_GRAY2RGB)
            elif original_display.shape[2] == 4:
                original_display = cv2.cvtColor(original_display, cv2.COLOR_RGBA2RGB)

            comparison = np.hstack([original_display, generated_display])
            st.image(comparison, caption="Input | Output", use_container_width=True)
    else:
        st.info("👈 Upload an image and click 'Translate Image' to see results here")
        st.markdown("""
        **Tips for best results:**
        - Use clear, high-contrast sketches for Sketch→Photo
        - Use well-lit photos for Photo→Sketch
        - Model was trained on 128×128 images
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Powered by CycleGAN | Sketch ↔ Photo Translation | Made with Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
