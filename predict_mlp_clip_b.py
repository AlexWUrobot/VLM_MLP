import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Predict action class number (1-8) using CLIP-B + trained MLP")
    parser.add_argument("--image", required=True, help="Path to a JPG/PNG image")
    parser.add_argument("--ckpt", default="mlp_clip_b.pt", help="Path to trained checkpoint")
    parser.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.ckpt)
    payload = torch.load(ckpt_path, map_location="cpu")

    labels = payload["labels_in_order"]
    feature_dim = int(payload["feature_dim"])
    hidden = int(payload["hidden"])

    model = MLP(in_dim=feature_dim, hidden=hidden, num_classes=len(labels)).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()

    import open_clip

    clip_model_name = payload.get("clip_model", "ViT-B-32")
    clip_pretrained = payload.get("clip_pretrained", "openai")

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model_name, pretrained=clip_pretrained
    )
    clip_model = clip_model.to(device).eval()

    image_path = Path(args.image)
    if not image_path.exists() and image_path.parent == Path("."):
        candidate = Path("Data-proccessed") / image_path.name
        if candidate.exists():
            image_path = candidate

    if not image_path.exists():
        raise SystemExit(
            "Image file not found.\n"
            f"- Provided: {args.image}\n"
            f"- Resolved: {image_path}\n"
            f"- CWD: {Path.cwd()}\n"
            "Tip: pass the full path, e.g. Data-proccessed/come_0_533ms.jpg"
        )

    image = Image.open(image_path).convert("RGB")
    image_t = preprocess(image).unsqueeze(0).to(device)

    feat = clip_model.encode_image(image_t).float()
    feat = F.normalize(feat, dim=-1)

    logits = model(feat)
    pred0 = int(logits.argmax(dim=1).item())
    number = pred0 + 1
    label = labels[pred0]

    print(number)
    print(label)


if __name__ == "__main__":
    main()
