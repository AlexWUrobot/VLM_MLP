"""
Zero-shot action recognition from a webcam using CLIP ViT-B-32.
Recognises 7 motions:
  1. come        – beckoning / waving someone closer
  2. wave        – waving hand as greeting
  3. stop        – palm facing forward, halt gesture
  4. play phone  – looking down at phone in hand
  5. phone call  – holding phone to ear
  6. take a picture – holding up phone / camera to take a photo
  7. idle        – standing / doing nothing

Uses YOLO to crop the closest person, then CLIP text–image similarity
to pick the best matching action.  Press 'q' or ESC to quit.
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ── 7 action labels and their text prompts ──────────────────────────
LABELS = [
    "come",
    "wave",
    "stop",
    "play phone",
    "phone call",
    "take a picture",
    "idle",
]

# Multiple prompts per class to improve zero-shot accuracy
TEXT_PROMPTS: dict[str, list[str]] = {
    "come": [
        "a person beckoning with their hand",
        "a person waving someone to come closer",
        "a person gesturing come here",
    ],
    "wave": [
        "a person waving their hand as a greeting",
        "a person waving hello",
        "a person raising hand and waving",
    ],
    "stop": [
        "a person holding up their palm to signal stop",
        "a person making a stop gesture with hand",
        "a person showing a halt hand sign",
    ],
    "play phone": [
        "a person looking down at their phone",
        "a person using a smartphone",
        "a person scrolling on a mobile phone",
    ],
    "phone call": [
        "a person holding a phone to their ear",
        "a person talking on the phone",
        "a person making a phone call",
    ],
    "take a picture": [
        "a person holding up a phone to take a photo",
        "a person taking a picture with a camera",
        "a person photographing with a smartphone",
    ],
    "idle": [
        "a person standing still doing nothing",
        "a person standing with arms at their sides",
        "a person standing idle",
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Zero-shot 7-action recognition from webcam (CLIP ViT-B-32)"
    )
    p.add_argument("--camera", type=int, default=0, help="Webcam index (default 0)")
    p.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    p.add_argument("--yolo", default="yolo11m.pt", help="YOLO model for person detection")
    p.add_argument("--yolo-conf", type=float, default=0.30, help="YOLO confidence threshold")
    p.add_argument("--clip-model", default="ViT-B-32", help="OpenCLIP model name")
    p.add_argument("--clip-pretrained", default="openai", help="OpenCLIP pretrained tag")
    p.add_argument(
        "--max-people", type=int, default=5,
        help="Max people to classify per frame (largest bbox first)",
    )
    p.add_argument(
        "--every", type=int, default=1,
        help="Run prediction every N frames (skip in between to speed up)",
    )
    p.add_argument(
        "--inertia", type=int, default=3,
        help="Change label only after N consecutive identical predictions (1 = disable)",
    )
    p.add_argument("--window", default="CLIP 7-Action Recognition", help="OpenCV window name")
    return p.parse_args()


# ── helpers ──────────────────────────────────────────────────────────

def _resize_list(lst: list, n: int, fill):
    """Resize list to length n, padding with fill or truncating."""
    if len(lst) >= n:
        return lst[:n]
    return lst + [fill] * (n - len(lst))


def _apply_inertia(
    raw_classes: list[int | None],
    stable_classes: list[int | None],
    pending_classes: list[int | None],
    pending_counts: list[int],
    inertia: int,
) -> tuple[list[int | None], list[int | None], list[int]]:
    k = len(raw_classes)
    stable_classes = _resize_list(stable_classes, k, None)
    pending_classes = _resize_list(pending_classes, k, None)
    pending_counts = _resize_list(pending_counts, k, 0)
    inertia = max(1, int(inertia))

    for i in range(k):
        raw = raw_classes[i]
        if raw is None:
            pending_classes[i] = None
            pending_counts[i] = 0
            continue
        if stable_classes[i] is None:
            stable_classes[i] = raw
            pending_classes[i] = None
            pending_counts[i] = 0
            continue
        if raw == stable_classes[i]:
            pending_classes[i] = None
            pending_counts[i] = 0
            continue
        if pending_classes[i] == raw:
            pending_counts[i] += 1
        else:
            pending_classes[i] = raw
            pending_counts[i] = 1
        if pending_counts[i] >= inertia:
            stable_classes[i] = raw
            pending_classes[i] = None
            pending_counts[i] = 0

    return stable_classes, pending_classes, pending_counts


def get_person_boxes(frame_bgr, yolo_model, yolo_conf: float) -> list[tuple[int, int, int, int]]:
    """Return person bboxes (x1,y1,x2,y2) sorted by area descending."""
    results = yolo_model.predict(frame_bgr, conf=yolo_conf, classes=[0], verbose=False)
    r = results[0]
    if getattr(r, "boxes", None) is None or len(r.boxes) == 0:
        return []
    h, w = frame_bgr.shape[:2]
    boxes, areas = [], []
    for box in r.boxes:
        if box.xyxy is None or len(box.xyxy) == 0:
            continue
        x1f, y1f, x2f, y2f = box.xyxy[0].tolist()
        x1, y1 = max(0, int(x1f)), max(0, int(y1f))
        x2, y2 = min(w, int(x2f)), min(h, int(y2f))
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area <= 0:
            continue
        boxes.append((x1, y1, x2, y2))
        areas.append(area)
    order = sorted(range(len(boxes)), key=lambda i: areas[i], reverse=True)
    return [boxes[i] for i in order]


LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "come":           (0, 255, 0),    # green
    "wave":           (0, 255, 0),    # green
    "stop":           (0, 0, 255),    # red
    "play phone":     (0, 165, 255),  # orange
    "phone call":     (0, 165, 255),  # orange
    "take a picture": (255, 200, 0),  # cyan-ish
    "idle":           (180, 180, 180),# grey
}

MIN_DISPLAY_CONF = 0.15  # show label only above this similarity


@torch.inference_mode()
def build_text_features(
    clip_model, tokenizer, device: str
) -> tuple[torch.Tensor, list[int]]:
    """
    Encode all text prompts and average per class.
    Returns (text_feats [7, dim], class_indices mapping prompt→class).
    """
    all_texts: list[str] = []
    class_ids: list[int] = []
    for cls_i, label in enumerate(LABELS):
        prompts = TEXT_PROMPTS[label]
        for t in prompts:
            all_texts.append(t)
            class_ids.append(cls_i)

    tokens = tokenizer(all_texts).to(device)
    raw_feats = clip_model.encode_text(tokens).float()
    raw_feats = F.normalize(raw_feats, dim=-1)

    # average per class
    n_classes = len(LABELS)
    dim = raw_feats.shape[1]
    class_feats = torch.zeros(n_classes, dim, device=device)
    for idx, cls_i in enumerate(class_ids):
        class_feats[cls_i] += raw_feats[idx]
    class_feats = F.normalize(class_feats, dim=-1)

    return class_feats


def _can_write_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def _resolve_cache_dir() -> str:
    """Find a writable HF hub cache directory."""
    for env_name in ("HUGGINGFACE_HUB_CACHE", "HF_HOME"):
        raw = os.environ.get(env_name)
        if not raw:
            continue
        candidate = Path(raw).expanduser()
        if env_name == "HF_HOME":
            candidate = candidate / "hub"
        if _can_write_dir(candidate):
            return str(candidate)

    for candidate in (
        Path(__file__).resolve().parent / ".cache" / "huggingface" / "hub",
        Path.home() / ".cache" / "huggingface" / "hub",
    ):
        if _can_write_dir(candidate):
            return str(candidate)

    raise SystemExit(
        "No writable HF cache directory found. "
        "Set HUGGINGFACE_HUB_CACHE to a writable path."
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ── load CLIP ────────────────────────────────────────────────────
    import open_clip

    cache_dir = _resolve_cache_dir()
    print(f"[INFO] HF cache: {cache_dir}")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained, cache_dir=cache_dir,
    )
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    text_feats = build_text_features(clip_model, tokenizer, device)
    print(f"[INFO] CLIP {args.clip_model} loaded on {device}, text features: {text_feats.shape}")

    # ── load YOLO ────────────────────────────────────────────────────
    from ultralytics import YOLO

    yolo = YOLO(args.yolo)
    print(f"[INFO] YOLO loaded: {args.yolo}")

    # ── open camera ──────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")
    print(f"[INFO] Camera {args.camera} opened. Press 'q' or ESC to quit.")

    # ── per-person tracking state ────────────────────────────────────
    stable_classes: list[int | None] = []
    stable_confs: list[float] = []
    pending_classes: list[int | None] = []
    pending_counts: list[int] = []

    frame_i = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame_i += 1

        # ── YOLO person detection ───────────────────────────────────
        boxes = get_person_boxes(frame, yolo, args.yolo_conf)
        boxes = boxes[: max(1, int(args.max_people))]

        stable_classes = _resize_list(stable_classes, len(boxes), None)
        stable_confs = _resize_list(stable_confs, len(boxes), 0.0)
        pending_classes = _resize_list(pending_classes, len(boxes), None)
        pending_counts = _resize_list(pending_counts, len(boxes), 0)

        do_predict = (frame_i % max(1, int(args.every))) == 0
        if do_predict:
            if boxes:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                crops: list[Image.Image] = []
                crop_to_box: list[int] = []

                for bi, (x1, y1, x2, y2) in enumerate(boxes):
                    crop = frame_rgb[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    crops.append(Image.fromarray(crop))
                    crop_to_box.append(bi)

                raw_classes: list[int | None] = [None] * len(boxes)
                raw_confs: list[float] = [0.0] * len(boxes)

                if crops:
                    imgs_t = torch.stack([preprocess(im) for im in crops]).to(device)
                    img_feats = clip_model.encode_image(imgs_t).float()
                    img_feats = F.normalize(img_feats, dim=-1)

                    # cosine similarity → softmax → pick best
                    sims = img_feats @ text_feats.T            # [N, 7]
                    probs = torch.softmax(sims * 100.0, dim=1) # temperature=100 (CLIP default logit_scale)
                    preds = probs.argmax(dim=1).tolist()
                    confs = probs.max(dim=1).values.tolist()

                    for pred, conf, bi in zip(preds, confs, crop_to_box, strict=False):
                        raw_classes[bi] = int(pred)
                        raw_confs[bi] = float(conf)

                stable_classes, pending_classes, pending_counts = _apply_inertia(
                    raw_classes, stable_classes, pending_classes, pending_counts, args.inertia,
                )
                for i in range(len(boxes)):
                    if stable_classes[i] is not None and stable_classes[i] == raw_classes[i]:
                        stable_confs[i] = raw_confs[i]
            else:
                stable_classes, stable_confs = [], []
                pending_classes, pending_counts = [], []

        # ── draw ─────────────────────────────────────────────────────
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            cls = stable_classes[i] if i < len(stable_classes) else None
            conf = stable_confs[i] if i < len(stable_confs) else 0.0

            if cls is None:
                text = ""
                color = (0, 255, 0)
            elif conf < MIN_DISPLAY_CONF:
                color = (160, 160, 160)
                text = f"{conf * 100:.0f}%"
            else:
                label_name = LABELS[int(cls)]
                color = LABEL_COLORS.get(label_name, (0, 255, 0))
                text = f"{int(cls) + 1} {label_name} {conf * 100:.0f}%"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if text:
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                ty = max(th + 4, y1 - 8)
                cv2.putText(frame, text, (x1, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        # HUD
        dt = time.time() - t0
        fps = frame_i / dt if dt > 0 else 0.0
        cv2.putText(frame, f"people: {len(boxes)}  fps: {fps:.1f}",
                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(args.window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[INFO] Done. {frame_i} frames, avg {fps:.1f} fps.")


if __name__ == "__main__":
    main()
