import os
import glob
import requests
from pathlib import Path
from typing import List, Union

from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


# ---------- Paths / constants ----------
BLIP_CKPT_URL = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth"
BLIP_CKPT_PATH = "checkpoints/model_large_caption.pth"
IMAGE_SIZE = 384
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


# ---------- Filesystem helpers ----------
def create_dir(directory_path: Union[str, Path]) -> str:
    """Create a directory if missing and return its name (kept for backward compat)."""
    p = Path(directory_path)
    p.mkdir(parents=True, exist_ok=True)
    return p.stem


# ---------- Checkpoint download ----------
def ensure_checkpoint(
    ckpt_path: Union[str, Path] = BLIP_CKPT_PATH,
    url: str = BLIP_CKPT_URL
) -> Path:
    """
    Ensure BLIP checkpoint exists at ckpt_path. If missing, download with progress.
    Returns the resolved Path.
    """
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    if ckpt_path.exists() and ckpt_path.stat().st_size > 0:
        return ckpt_path

    print(f"Downloading checkpoint to: {ckpt_path}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(ckpt_path, "wb") as f, tqdm(total=total, unit="iB", unit_scale=True) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    print("Checkpoint downloaded!")
    return ckpt_path


# Backward-compat function name (your original)
def download_checkpoint():
    """Deprecated wrapper. Keeps your old name working by calling ensure_checkpoint()."""
    return ensure_checkpoint()


# ---------- Image I/O ----------
def read_images_from_directory(image_directory: str) -> list:
    """Return a list of image file paths in a directory (gif/png/jpg/jpeg)."""
    exts = ("*.gif", "*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(image_directory, ext)))
    print(f"Images found: {len(paths)}")
    return sorted(paths)


def read_with_pil(list_of_images: List[Union[str, Path]], resize: bool = False) -> List[Image.Image]:
    """Open images as RGB PIL. Optional thumbnail for quick preview."""
    pil_images = []
    for p in list_of_images:
        img = Image.open(p).convert("RGB")
        if resize:
            img.thumbnail((512, 512))
        pil_images.append(img)
    return pil_images


# ---------- Preprocessing ----------
_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(_CLIP_MEAN, _CLIP_STD),
    ]
)

def preprocess_image(img: Image.Image) -> torch.Tensor:
    """Convert a single PIL image to a normalized tensor (C,H,W) at 384x384."""
    return _transform(img)

def preprocess_batch(pil_images: List[Image.Image], device: torch.device) -> torch.Tensor:
    """Convert a list of PIL images into a single batched tensor (B,3,384,384) on device."""
    tensors = [preprocess_image(im) for im in pil_images]
    batch = torch.stack(tensors, dim=0).to(device, non_blocking=True)
    return batch


# Backward-compat (your original function that returned a list of 1x tensors)
def prep_images(pil_images: List[Image.Image], device) -> list:
    """Legacy: returns a list of 1x tensors on device. New code prefers preprocess_batch()."""
    return [preprocess_image(img).unsqueeze(0).to(device) for img in pil_images]
