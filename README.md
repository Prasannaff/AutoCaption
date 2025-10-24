# 🖼️ AutoCaption — Image Captioning using BLIP

AutoCaption is an **image-to-text (img2txt)** model that automatically generates **descriptive captions** for images using the **BLIP (Bootstrapped Language Image Pretraining)** model.

---

## 🚀 Features
- Generates natural language captions from input images.
- Supports **batch processing** for multiple images at once.
- Automatically downloads the pretrained **BLIP-Large** model if not available.
- GPU acceleration supported for faster inference.

---

## 📦 Model Checkpoints (Required)

If there is no `checkpoints` folder, the script will automatically create one and download the model file.  
You can also download it manually and place it inside the folder.

**Download Link:**
- [BLIP-Large](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth)

---

## 🧩 Folder Structure
```bash
AutoCaption/
 ├── checkpoints/
 ├── images/
 ├── captions/
 │    ├── 0_captions.txt
 │    ├── 1_captions.txt
 │    └── ...
 ├── inference.py
 └── README.md

