import requests
from pathlib import Path
from typing import List

from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


BLIP_CKPT_URL = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth"
BLIP_CKPT_PATH = "checkpoints/model_large_caption.pth"

IMAGE_SIZE = 384
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# ---------- Checkpoint ----------
def ensure_checkpoint(ckpt_path=BLIP_CKPT_PATH, url=BLIP_CKPT_URL) -> Path:
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    if ckpt_path.exists() and ckpt_path.stat().st_size > 0:
        return ckpt_path

    print(f"Downloading BLIP checkpoint → {ckpt_path}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(ckpt_path, "wb") as f, tqdm(total=total, unit="iB", unit_scale=True) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

    print("Checkpoint ready.")
    return ckpt_path


# ---------- Preprocessing ----------
_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(_CLIP_MEAN, _CLIP_STD),
])

def preprocess_image(img: Image.Image) -> torch.Tensor:
    return _transform(img)

def preprocess_batch(pil_images: List[Image.Image], device: torch.device) -> torch.Tensor:
    tensors = [preprocess_image(im) for im in pil_images]
    return torch.stack(tensors).to(device, non_blocking=True)
