import torch
from transformers import pipeline
from PIL import Image

# 1. Target Apple Silicon (MPS)
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Loading Depth Anything V2 Large on {device}...")

# Hugging Face handles the download automatically into your local cache
pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Large-hf",
    device=device
)

# 2. Load your Sentinel-2 RGB image
image_path = "Darjeeling_RGB.jpg"
image = Image.open(image_path).convert("RGB")

print(f"Running depth estimation on {image_path}...")
result = pipe(image)

# 3. Save the resulting depth map
depth_image = result["depth"]
depth_image.save("Darjeeling_Predicted_Depth.png")
print("✅ Saved Darjeeling_Predicted_Depth.png successfully!")
