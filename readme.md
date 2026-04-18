# 🔄 CycleGAN: Sketch ↔ Photo Translation

Translate between sketches and photos in real-time using a trained CycleGAN model, powered by Streamlit.

## ✨ Features

- 🔄 **Bidirectional translation** — Sketch→Photo and Photo→Sketch
- 🖼️ Upload any sketch or photo
- ✏️ Draw sketches directly in the app
- 🖌️ Convert photos to sketches
- 💾 Download generated results
- 🚀 GPU acceleration support

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPU (optional, for faster inference)

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place your trained checkpoint

Copy your CycleGAN checkpoint (`.pt` file with `g_ab` and `g_ba` keys) into:

```
models/best_model.pt
```

The checkpoint should have this structure (e.g. saved after CycleGAN training):
```python
{"g_ab": g_ab.state_dict(), "g_ba": g_ba.state_dict(), ...}
```

### 3. Run the app

```bash
streamlit run app.py
```

## ☁️ Streamlit Cloud / remote deploy

Deployed apps only see files that are **in your Git repository**. `best_model.pt` is usually too large for GitHub (or is listed in `.gitignore`), so the default path `models/best_model.pt` will **not exist** on the server unless you ship it some other way.

**Option A — Secrets + hosted file (recommended)**  
Upload `best_model.pt` to any HTTPS host that supports **direct download** (your own server, S3 public URL, Hugging Face file URL, etc.). In Streamlit: **App settings → Secrets**, add:

```toml
CYCLEGAN_CHECKPOINT_URL = "https://example.com/path/best_model.pt"
```

Redeploy, open the app, and click **Load Model**. The first run downloads the file into `.cache/` beside the app.

**Option B — Commit the checkpoint**  
Only if the file is small enough for your host (GitHub warns above ~50 MB; hard limit 100 MB). Remove `models/*.pt` from `.gitignore` if you added it, commit `models/best_model.pt`, and redeploy.

**Option C — Git LFS**  
Track the `.pt` with [Git LFS](https://git-lfs.com/) so clones (and Streamlit Cloud) pull the real file instead of a pointer.

You can still override the default text field with the env var `CYCLEGAN_CHECKPOINT` on your own server or Docker image.

### 4. Use the app

1. Click **Load Model** in the sidebar
2. Choose translation direction: **Sketch→Photo** or **Photo→Sketch**
3. Upload an image or draw a sketch
4. Click **Translate Image**
5. Download the result

## 🏗️ Model Architecture

- **Generator**: ResNet-based (6 residual blocks, 128×128 input)
- **Discriminator**: PatchGAN with InstanceNorm
- **Training**: CycleGAN with adversarial + cycle consistency + identity losses

## 📁 Project Structure

```
├── app.py              # Streamlit web application
├── model.py            # CycleGAN Generator & Discriminator
├── utils.py            # Model loading, preprocessing, inference
├── requirements.txt    # Python dependencies
├── readme.md           # This file
└── models/             # Place trained checkpoints here
    └── best_model.pt
```
