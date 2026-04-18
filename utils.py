import torch
import numpy as np
from PIL import Image
import cv2

from model import Generator


def load_cyclegan(checkpoint_path, device="cpu"):
    """
    Load a CycleGAN checkpoint that contains g_ab and g_ba state dicts.
    Returns (g_ab, g_ba) — both in eval mode on the given device.

    Accepted checkpoint formats:
      - {"g_ab": state_dict, "g_ba": state_dict, ...}
      - A raw state_dict (loaded as g_ab only; g_ba will be None)
    """
    g_ab = Generator(in_ch=3, out_ch=3, nf=64, n_res=6)
    g_ba = Generator(in_ch=3, out_ch=3, nf=64, n_res=6)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "g_ab" in ckpt:
        g_ab.load_state_dict(_strip_module_prefix(ckpt["g_ab"]))
        g_ba.load_state_dict(_strip_module_prefix(ckpt["g_ba"]))
    else:
        sd = ckpt if not isinstance(ckpt, dict) else ckpt.get("state_dict", ckpt)
        g_ab.load_state_dict(_strip_module_prefix(sd))
        g_ba = None

    g_ab = g_ab.to(device).eval()
    if g_ba is not None:
        g_ba = g_ba.to(device).eval()

    return g_ab, g_ba


def _strip_module_prefix(state_dict):
    """Remove 'module.' prefix added by DataParallel."""
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace("module.", "")] = v
    return cleaned


def preprocess_image(image, target_size=(128, 128)):
    """Preprocess input image: resize, normalize to [-1, 1], return tensor."""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 127.5 - 1.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return image


def postprocess_image(tensor):
    """Convert model output tensor [-1, 1] to displayable uint8 numpy array."""
    image = tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    image = (image + 1) / 2.0
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    return image


def translate_image(generator, input_image, device="cpu"):
    """Run a single generator on an input image and return the output as numpy."""
    with torch.no_grad():
        tensor = preprocess_image(input_image).to(device)
        output = generator(tensor)
    return postprocess_image(output)


def create_sketch_from_image(image):
    """Convert an image to sketch-like appearance using dodge-blend."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blurred, scale=256)
    sketch = np.clip(sketch, 0, 255).astype(np.uint8)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
