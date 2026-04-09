import time
import torch
from PIL import Image
import open_clip
from transformers import AutoProcessor, AutoModel

# =========================
# Config
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_PATH = "test.jpg"   # <-- change this
MODEL_TYPE = "qwen"     # options: clip_b, clip_l, qwen
NUM_RUNS = 50

# =========================
# Load Image
# =========================
def load_image(path):
    return Image.open(path).convert("RGB")

# =========================
# CLIP Loader
# =========================
def load_clip(model_name):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained="openai"
    )
    model = model.to(DEVICE).eval()
    return model, preprocess

# =========================
# Qwen2-VL Loader (vision encoder only)
# =========================
def load_qwen():
    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name, dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)
    model.eval()
    return model, processor

# =========================
# Benchmark CLIP
# =========================
def run_clip(model_name):
    print(f"\n=== Running {model_name} ===")

    model, preprocess = load_clip(model_name)
    image = preprocess(load_image(IMAGE_PATH)).unsqueeze(0).to(DEVICE)

    # warmup
    with torch.inference_mode():
        for _ in range(5):
            _ = model.encode_image(image)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    with torch.inference_mode():
        for _ in range(NUM_RUNS):
            feat = model.encode_image(image)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    end = time.time()
    latency = (end - start) / NUM_RUNS * 1000

    print(f"Feature shape: {feat.shape}")
    print(f"Avg latency: {latency:.2f} ms")


# =========================
# Benchmark Qwen2-VL
# =========================
def run_qwen():
    print("\n=== Running Qwen2-VL Encoder ===")

    model, processor = load_qwen()
    image = load_image(IMAGE_PATH)

    # Qwen2-VL's combined processor expects text (it looks for an image token).
    # For vision-encoder-only benchmarking, use the image processor directly.
    if hasattr(processor, "image_processor") and processor.image_processor is not None:
        inputs = processor.image_processor(images=image, return_tensors="pt").to(DEVICE)
    else:
        inputs = processor(images=image, text=[""], return_tensors="pt").to(DEVICE)

    # warmup
    with torch.inference_mode():
        for _ in range(5):
            _ = model.get_image_features(**inputs)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    with torch.inference_mode():
        for _ in range(NUM_RUNS):
            feat = model.get_image_features(**inputs)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    end = time.time()
    latency = (end - start) / NUM_RUNS * 1000

    print(f"Feature shape: {feat.last_hidden_state.shape}")
    print(f"Avg latency: {latency:.2f} ms")


# =========================
# Main
# =========================
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Image: {IMAGE_PATH}")

    if MODEL_TYPE == "clip_b":
        run_clip("ViT-B-32")

    elif MODEL_TYPE == "clip_l":
        run_clip("ViT-L-14")

    elif MODEL_TYPE == "qwen":
        run_qwen()

    else:
        print("Invalid MODEL_TYPE")