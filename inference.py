# =======================
# AutoCaption · Social Media Caption Generator
# =======================
import os
from typing import List
import random

import torch
import gradio as gr
from PIL import Image

from utils import (
    ensure_checkpoint,
    preprocess_batch,
    BLIP_CKPT_PATH,
    BLIP_CKPT_URL
)

MED_CFG_PATH = "configs/med_config.json"


# ---------- BLIP loader ----------
def load_blip_decoder(med_config, image_size, vit, checkpoint, device):
    from models.blip import BLIP_Decoder
    model = BLIP_Decoder(
        med_config=med_config,
        image_size=image_size,
        vit=vit
    )
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    return model.to(device)


def load_model(device):
    ensure_checkpoint(BLIP_CKPT_PATH, BLIP_CKPT_URL)
    model = load_blip_decoder(
        MED_CFG_PATH, 384, "large", BLIP_CKPT_PATH, device
    )
    model.eval()
    return model


# ---------- Caption generation ----------
@torch.no_grad()
def caption_batch(images, model, device, beams, max_len):
    batch = preprocess_batch(images, device)
    out = model.generate(
        batch,
        sample=False,
        num_beams=beams,
        max_length=max_len,
        min_length=5,
    )
    return [c.strip().rstrip(".") for c in out]


# ---------- Context-aware creative logic ----------
def detect_context(caption: str):
    c = caption.lower()
    if any(w in c for w in ["blood", "knife", "sword", "weapon", "gun"]):
        return "dark"
    if any(w in c for w in ["car", "watch", "luxury", "building", "city"]):
        return "luxury"
    if any(w in c for w in ["man", "woman", "person", "face"]):
        return "portrait"
    if any(w in c for w in ["tree", "sky", "mountain", "nature"]):
        return "nature"
    return "general"


FUNNY_MAP = {
    "dark": [
        "{c}. Not your average Monday 😬",
        "{c}. Chaos, but make it aesthetic 😅"
    ],
    "portrait": [
        "{c}. Main character energy 😎",
        "{c}. This face says everything 😂"
    ],
    "nature": [
        "{c}. Nature showing off again 🌿",
        "{c}. Proof that Earth is undefeated 🌍"
    ],
    "general": [
        "{c}. Internet, you’re welcome 😌",
        "{c}. This didn’t have to go this hard 🔥"
    ]
}

LUXURY_MAP = {
    "dark": [
        "{c}. Power. Control. Presence.",
    ],
    "luxury": [
        "{c}. Quiet luxury in focus.",
        "{c}. Crafted for those who notice details."
    ],
    "portrait": [
        "{c}. Elegance in expression.",
        "{c}. Confidence, perfectly framed."
    ],
    "general": [
        "{c}. Minimal. Refined. Timeless.",
        "{c}. Designed beyond trends."
    ]
}


def style_caption(base, tone, creative):
    base = base[:1].upper() + base[1:]
    if not creative or tone == "Neutral":
        return base + "."

    context = detect_context(base)

    if tone == "Fun":
        return random.choice(FUNNY_MAP.get(context, FUNNY_MAP["general"])).format(c=base)

    if tone == "Luxury":
        return random.choice(LUXURY_MAP.get(context, LUXURY_MAP["general"])).format(c=base)

    return base + "."


def hashtags_from_caption(caption):
    words = [
        w.lower().strip(".,!?")
        for w in caption.split()
        if w.isalpha() and len(w) > 3
    ]
    base = words[:5] + ["photography", "instadaily", "explore"]
    tags = []
    for w in base:
        tag = f"#{w}"
        if tag not in tags:
            tags.append(tag)
    return " ".join(tags)


# ---------- HTML Results ----------
def render_results(names, captions, hashtags):
    rows = ""
    for n, c, h in zip(names, captions, hashtags):
        rows += f"""
        <tr>
          <td>{n}</td>
          <td>
            <div class="cell">
              <span>{c}</span>
              <button onclick="navigator.clipboard.writeText(`{c}`)">Copy</button>
            </div>
          </td>
          <td>
            <div class="cell">
              <span>{h}</span>
              <button onclick="navigator.clipboard.writeText(`{h}`)">Copy</button>
            </div>
          </td>
        </tr>
        """

    return f"""
    <style>
    table {{
        width:100%;
        border-collapse: collapse;
    }}
    th, td {{
        border:1px solid #333;
        padding:10px;
        vertical-align: top;
    }}
    .cell {{
        display:flex;
        justify-content: space-between;
        gap:10px;
    }}
    button {{
        padding:4px 10px;
        border-radius:6px;
        border:none;
        cursor:pointer;
        background:#2d2d2d;
        color:#fff;
        font-size:12px;
    }}
    button:hover {{
        background:#444;
    }}
    </style>
    <table>
      <tr>
        <th>Filename</th>
        <th>Caption</th>
        <th>Hashtags</th>
      </tr>
      {rows}
    </table>
    """


# ---------- Pipeline ----------
def run_social(files, tone, beams, maxlen, creative, device, model):
    images, names = [], []
    for f in files:
        img = Image.open(f.name).convert("RGB")
        images.append(img)
        names.append(os.path.basename(f.name))

    base_caps = caption_batch(images, model, device, beams, maxlen)
    final_caps = [style_caption(c, tone, creative) for c in base_caps]
    tags = [hashtags_from_caption(c) for c in base_caps]

    return images, render_results(names, final_caps, tags)


# ---------- UI ----------
def build_app():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    with gr.Blocks(title="AutoCaption Social") as demo:
        gr.Markdown("## AutoCaption · Social Media Caption Generator")

        files = gr.Files(file_types=["image"], label="Upload images")

        preview = gr.Gallery(
            label="Preview",
            columns=3,
            height=340,
            object_fit="contain"
        )

        with gr.Row():
            tone = gr.Radio(["Neutral", "Fun", "Luxury"], value="Fun")
            creative = gr.Checkbox(label="Creative mode (context-aware)")

        with gr.Row():
            beams = gr.Slider(1, 5, value=3, label="Beam search")
            maxlen = gr.Slider(10, 40, value=30, label="Max length")

        btn = gr.Button("Generate")
        result = gr.HTML(label="Results")

        btn.click(
            fn=lambda f, t, b, m, c: run_social(f, t, b, m, c, device, model),
            inputs=[files, tone, beams, maxlen, creative],
            outputs=[preview, result]
        )

    return demo


if __name__ == "__main__":
    build_app().launch(share=True)
