# AutoCaption – Social Media Caption Generator

<img src="./ui_preview.png" width="100%">

AutoCaption is a **real-world AI powered Social Media Caption Generator** built using **BLIP Vision-Language Model**.  
It converts any image into a ready-to-post, social-platform-optimized caption with intelligent hashtag suggestions.

This tool is designed for modern digital content workflows — creators, ecommerce product owners, marketing teams, agencies, meme pages, photographers, advertisers and influencers.

### Why this project matters?

Every single major platform today is image-first:

- Instagram  
- Facebook  
- Twitter (X)  
- Pinterest  
- LinkedIn  

Yet – writing captions and finding relevant hashtags is still a manual effort and wastes time.

AutoCaption solves this by auto-generating:

- 1 unique well-structured caption per image
- Tone control (Neutral / Fun / Luxury)
- Platform friendly hashtags (10+ curated tags)
- Multi-image processing at once

This makes AutoCaption directly usable in the **real world**.

---

## Key Features

| Feature | Description |
|--------|-------------|
| AI Image Captioning | Uses BLIP Vision-Language AI model to generate context aware captions |
| Platform Optimized | Captions suitable for insta reels, product photos, artworks, photography posts, travel shots |
| Hashtag Auto Generation | Generates keyword based trendy hashtags |
| Multi Upload Support | Upload multiple images and get multiple social caption rows instantly |
| Social Tone | Neutral / Fun / Luxury styles |
| Gradio UI | Clean, fast, responsive UI for demo / deployment |

---

## Tech Stack

| Component | Technology |
|----------|------------|
| Model | BLIP Base Vision Language Model |
| Backend | Python + PyTorch |
| Preprocessing | torchvision |
| Frontend UI | Gradio |
| Environment | Local / Cloud / HF Spaces |

---

## Output Format Example

| filename | caption | hashtags |
|---------|---------|----------|
| image_09.png | A statue of an angel holding a bird. | #statue #angel #bird #photography #instagood ... |

---

## Demo Preview

<img src="./ui_preview.png" width="100%">

---

## Installation & Setup

```bash
git clone https://github.com/Prasannaff/AutoCaption
cd AutoCaption
pip install -r requirements.txt
python inference.py
