import torch
from pathlib import Path
from PIL import Image
from models.blip import blip_decoder
import utils
import gradio as gr

# ---------------------------
# Device Setup
# ---------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------
# Model Initialization
# ---------------------------
def load_model():
    # Ensure checkpoints folder exists
    if not Path("checkpoints").is_dir():
        print("Creating 'checkpoints' directory...")
        utils.create_dir("checkpoints")

    # Download checkpoint if missing
    checkpoint_path = Path("checkpoints/model_large_caption.pth")
    if not checkpoint_path.is_file():
        print("Downloading BLIP checkpoint...")
        utils.download_checkpoint()

    # Load BLIP decoder
    print("Loading BLIP model...")
    model = blip_decoder(pretrained=str(checkpoint_path), image_size=384, vit="large")
    model.eval()
    model = model.to(device)
    print(f"Model loaded on {device}")
    return model

model = load_model()

# ---------------------------
# Caption Generation Function
# ---------------------------
def generate_caption(image: Image.Image):
    """
    Generate caption for a single PIL image.
    """
    transformed_image = utils.prep_images([image], device)  # returns a list
    with torch.no_grad():
        caption = model.generate(
            transformed_image[0],
            sample=False,
            num_beams=3,
            max_length=20,
            min_length=5
        )
    return caption[0]

# ---------------------------
# Gradio Interface
# ---------------------------
def caption_images(images):
    """
    Handles single or multiple images.
    Returns captions as a list.
    """
    if not isinstance(images, list):
        images = [images]
    captions = [generate_caption(img) for img in images]
    return captions if len(captions) > 1 else captions[0]

iface = gr.Interface(
    fn=caption_images,
    inputs=gr.Image(type="pil", label="Upload Image(s)"),  # just type and label
    outputs=gr.Textbox(label="Generated Caption(s)"),
    title="AutoCaption - Image Captioning",
    description="Upload an image (or multiple images) to generate descriptive captions using the BLIP model.",
    allow_flagging="never"
)



# Launch Gradio app
if __name__ == "__main__":
    iface.launch(share=True)
