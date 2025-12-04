# =======================
# AutoCaption · Social-only Inference App
# =======================
import os
from typing import List

import torch
import gradio as gr
from PIL import Image

from utils import (
    ensure_checkpoint,           # robust checkpoint downloader
    preprocess_batch,            # (B,3,384,384) tensor
    BLIP_CKPT_PATH, BLIP_CKPT_URL
)

MED_CFG_PATH = "configs/med_config.json"
COLUMNS = ["filename", "caption", "hashtags"]


# ---------- BLIP thin loader (models/ layout) ----------
def load_blip_decoder(med_config: str, image_size: int, vit: str, checkpoint: str, device: torch.device):
    """
    Instantiate BLIP_Decoder and then load weights from checkpoint.
    """
    from models.blip import BLIP_Decoder

    model = BLIP_Decoder(
        med_config=med_config,
        image_size=image_size,
        vit=vit
    )

    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt)  # some checkpoints use {'model': state_dict}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[BLIP] loaded with missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    return model.to(device)


# ---------- Model init ----------
def load_model(device: torch.device):
    ensure_checkpoint(BLIP_CKPT_PATH, BLIP_CKPT_URL)  # download if missing
    model = load_blip_decoder(
        med_config=MED_CFG_PATH,
        image_size=384,
        vit="large",
        checkpoint=BLIP_CKPT_PATH,
        device=device
    )
    model.eval()
    return model


# ---------- Social helpers ----------
@torch.no_grad()
def caption_batch(pil_images: List[Image.Image], model, device, num_beams=3, max_len=30) -> List[str]:
    batch = preprocess_batch(pil_images, device)  # (B,3,384,384)
    captions = model.generate(
        batch,
        sample=False,
        num_beams=int(num_beams),
        max_length=int(max_len),
        min_length=5,
        repetition_penalty=1.1,
    )
    return [c.strip().rstrip(".") for c in captions]

def social_caption(base: str, tone: str) -> str:
    """Return ONE caption styled for social."""
    base_clean = base[:1].upper() + base[1:]
    if tone == "Fun":
        return f"{base_clean}. Vibes = immaculate ✨"
    if tone == "Luxury":
        return f"{base_clean}. Subtle. Refined. Timeless."
    return base_clean + "."

_STOP = {
    "a","an","the","on","in","of","for","with","and","to","at","by","from","over","under","into","is","are","this","that",
    "near","front","back","side","view","photo","image","picture"
}

def hashtags_from_caption(caption: str, cap: int = 10) -> List[str]:
    words = [w.strip(".,!?:;()\"'").lower() for w in caption.split()]
    keywords = [w for w in words if w.isalpha() and w not in _STOP and len(w) > 2]
    bank = ["photography", "instagood", "nature", "travel", "daily", "love", "life"]
    uniq = []
    for w in keywords + bank:
        tag = f"#{w.replace(' ', '')}"
        if tag not in uniq:
            uniq.append(tag)
        if len(uniq) >= cap:
            break
    return uniq[:cap]

def run_social(files, tone, beams, maxlen, device, model):
    if not files:
        return []

    pil_images, names = [], []
    for f in (files if isinstance(files, list) else [files]):
        path = getattr(f, "name", None) or f  # gr.Files gives temp object with .name
        img = Image.open(path).convert("RGB")
        pil_images.append(img)
        names.append(os.path.basename(path) if isinstance(path, str) else os.path.basename(f.name))

    bases = caption_batch(pil_images, model, device, num_beams=beams, max_len=maxlen)

    # 2-D rows for the Dataframe: [filename, caption, hashtags]
    table_rows = []
    for name, base in zip(names, bases):
        cap = social_caption(base, tone)
        tags = " ".join(hashtags_from_caption(base))
        table_rows.append([name, cap, tags])
    return table_rows


# ---------- Gradio UI ----------
def build_app():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    with gr.Blocks(title="AutoCaption Social") as demo:
        gr.Markdown("# AutoCaption · Social Media Generator")
        gr.Markdown("Upload images → get a single social caption + 10 hashtags.")

        with gr.Row():
            files = gr.Files(file_types=["image"], label="Upload one or more images")
        with gr.Row():
            tone = gr.Radio(choices=["Neutral", "Fun", "Luxury"], value="Neutral", label="Tone")
            beams = gr.Slider(1, 5, value=3, step=1, label="Beam search (quality)")
            maxlen = gr.Slider(10, 40, value=30, step=1, label="Max caption length")

        run_btn = gr.Button("Generate")
        out = gr.Dataframe(
            headers=COLUMNS,
            datatype=["str"] * len(COLUMNS),
            row_count=(1, "dynamic"),
            col_count=(len(COLUMNS), "fixed"),
            wrap=True,
            label="Results"
        )

        def _run(files, tone, beams, maxlen):
            return run_social(files, tone, beams, maxlen, device, model)

        run_btn.click(_run, inputs=[files, tone, beams, maxlen], outputs=[out])

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(share=True)
