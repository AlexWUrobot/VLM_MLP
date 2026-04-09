import argparse
import time
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Webcam live action prediction (YOLO crop + CLIP-B + MLP)")
    p.add_argument("--ckpt", default="mlp_clip_b.pt", help="Path to trained MLP checkpoint")
    p.add_argument("--camera", type=int, default=0, help="Webcam index (default 0)")
    p.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    p.add_argument("--yolo", default="yolov8n.pt", help="Ultralytics YOLO model for person detection")
    p.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO confidence threshold")
    p.add_argument(
        "--every",
        type=int,
        default=1,
        help="Run prediction every N frames (increase to speed up).",
    )
    p.add_argument("--window", default="Live Prediction", help="OpenCV window name")
    return p.parse_args()


def crop_closest_person(frame_bgr, yolo_model, yolo_conf: float):
    """Return (crop_bgr, bbox_xyxy) or (None, None)."""
    results = yolo_model.predict(frame_bgr, conf=yolo_conf, classes=[0], verbose=False)
    r = results[0]
    if getattr(r, "boxes", None) is None or len(r.boxes) == 0:
        return None, None

    best = None
    best_area = -1
    h, w = frame_bgr.shape[:2]
    for box in r.boxes:
        if box.xyxy is None or len(box.xyxy) == 0:
            continue
        x1f, y1f, x2f, y2f = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1f), int(y1f), int(x2f), int(y2f)
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area > best_area:
            best_area = area
            best = (x1, y1, x2, y2)

    if best is None or best_area <= 0:
        return None, None

    x1, y1, x2, y2 = best
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None
    return crop, best


@torch.inference_mode()
def main() -> None:
    args = parse_args()

    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise SystemExit("Failed to import cv2. Install opencv-python.") from exc

    try:
        import open_clip  # type: ignore
    except Exception as exc:
        raise SystemExit("Failed to import open_clip. Install open_clip_torch.") from exc

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        raise SystemExit("Failed to import ultralytics. Install ultralytics.") from exc

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(Path(args.ckpt), map_location="cpu")
    labels = payload["labels_in_order"]
    feature_dim = int(payload["feature_dim"])
    hidden = int(payload["hidden"])

    mlp = MLP(in_dim=feature_dim, hidden=hidden, num_classes=len(labels)).to(device)
    mlp.load_state_dict(payload["state_dict"], strict=True)
    mlp.eval()

    clip_model_name = payload.get("clip_model", "ViT-B-32")
    clip_pretrained = payload.get("clip_pretrained", "openai")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model_name, pretrained=clip_pretrained
    )
    clip_model = clip_model.to(device).eval()

    yolo = YOLO(args.yolo)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    last_pred_text = ""
    last_bbox = None

    frame_i = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_i += 1

        if frame_i % max(1, int(args.every)) == 0:
            crop, bbox = crop_closest_person(frame, yolo, args.yolo_conf)
            last_bbox = bbox

            if crop is not None:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(crop_rgb)
                image_t = preprocess(pil).unsqueeze(0).to(device)

                feat = clip_model.encode_image(image_t).float()
                feat = F.normalize(feat, dim=-1)
                logits = mlp(feat)
                pred0 = int(logits.argmax(dim=1).item())
                number = pred0 + 1
                label = labels[pred0]
                last_pred_text = f"{number} {label}"
            else:
                last_pred_text = "no_person"

        # draw bbox
        if last_bbox is not None:
            x1, y1, x2, y2 = last_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # fps
        dt = time.time() - t0
        fps = frame_i / dt if dt > 0 else 0.0

        cv2.putText(
            frame,
            f"pred: {last_pred_text}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"fps: {fps:.1f}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(args.window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
