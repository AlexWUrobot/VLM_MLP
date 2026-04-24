"""
Zero-shot action recognition from a webcam using CLIP ViT-B-32 + RTMW pose.
Recognises 8 motions:
  1. come        – beckoning / waving someone closer
  2. wave        – waving hand as greeting
  3. stop        – palm facing forward, halt gesture
  4. play phone  – looking down at phone in hand
  5. phone call  – holding phone to ear
  6. take a picture – holding up phone / camera to take a photo
  7. idle        – standing / doing nothing
  8. love        – finger heart or arm heart gesture

CLIP picks the broad action class.  When CLIP predicts stop/wave/come,
RTMW wrist-motion tracking refines the prediction by analysing hand
trajectory (still → stop, oscillating → wave, other movement → come).

Press 'q' or ESC to quit.
"""

import argparse
import os
import time
from collections import deque
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
    "love",
    "high_wave",
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
    "love": [
        "a person making a heart shape with their fingers",
        "a person forming a love heart sign with both hands above their head",
        "a person making a finger heart gesture",
        "a person crossing arms above head to form a big heart shape",
    ],
    "high_wave": [
        "a person waving their hand high above their head",
        "a person raising hand above head and waving",
        "a person waving with arm fully extended overhead",
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
    p.add_argument("--width", type=int, default=640, help="Camera capture width (default 1280)")
    p.add_argument("--height", type=int, default=480, help="Camera capture height (default 720)")
    p.add_argument("--fps", type=int, default=30, help="Camera capture FPS (default 30)")
    p.add_argument("--no-trt", action="store_true", help="Disable TensorRT for RTMW, use CUDA EP")
    p.add_argument("--rtmw-model", default=None, help="Path to RTMW ONNX model (auto-download if omitted)")
    p.add_argument("--bbox-pad", type=float, default=1.25, help="Bbox padding ratio for RTMW crop")
    p.add_argument("--window", default="CLIP 8-Action Recognition", help="OpenCV window name")
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
    "love":           (255, 0, 255),  # magenta
    "high_wave":      (0, 255, 255),  # yellow
}

MIN_DISPLAY_CONF = 0.15  # show label only above this similarity


# ── RTMW hand-gesture refinement ─────────────────────────────────────
# When CLIP predicts stop / wave / come, use wrist motion to disambiguate.

COME_IDX = LABELS.index("come")      # 0
WAVE_IDX = LABELS.index("wave")      # 1
STOP_IDX = LABELS.index("stop")      # 2
IDLE_IDX = LABELS.index("idle")      # 6
HIGH_WAVE_IDX = LABELS.index("high_wave")  # 8
HAND_GESTURE_SET = {COME_IDX, WAVE_IDX, STOP_IDX, HIGH_WAVE_IDX}

# RTMW body keypoint for head reference (nose)
KP_NOSE = 0

# RTMW body keypoint indices
KP_LWRIST, KP_RWRIST = 9, 10
# RTMW hand keypoint offsets (21 per hand)
# Left hand: 91..111, Right hand: 112..132
# Wrist=0, fingertips = 4(thumb), 8(index), 12(middle), 16(ring), 20(pinky)
LHAND_BASE = 91
RHAND_BASE = 112
FINGERTIP_OFFSETS = [4, 8, 12, 16, 20]  # relative to hand base
# MCP (base) of index=5, middle=9, pinky=17, thumb_tip=4 relative to hand base
INDEX_MCP_OFF = 5
PINKY_MCP_OFF = 17
MIDDLE_MCP_OFF = 9
THUMB_TIP_OFF = 4


def _hand_scale(kpts: np.ndarray, scores: np.ndarray, hand_base: int,
                min_score: float = 0.25) -> float | None:
    """Measure avg distance from hand wrist to fingertips (proxy for depth)."""
    wrist = kpts[hand_base]
    ws = scores[hand_base]
    if ws < min_score:
        return None
    dists = []
    for off in FINGERTIP_OFFSETS:
        idx = hand_base + off
        if scores[idx] >= min_score:
            dists.append(np.linalg.norm(kpts[idx] - wrist))
    if len(dists) < 3:
        return None
    return float(np.mean(dists))


def _palm_facing_camera(
    kpts: np.ndarray, scores: np.ndarray, hand_base: int,
    min_score: float = 0.25,
) -> bool | None:
    """
    Determine if the palm faces the camera using 2D hand keypoints.

    Uses the cross product of (wrist→middle_MCP) × (index_MCP→pinky_MCP).
    For a right hand, palm facing camera: cross > 0  (thumb on viewer's left).
    For a left hand,  palm facing camera: cross < 0  (mirror).
    Returns True if palm faces camera, False if back-of-hand faces camera,
    None if keypoints are insufficient.
    """
    wrist_idx = hand_base
    mid_idx = hand_base + MIDDLE_MCP_OFF
    idx_idx = hand_base + INDEX_MCP_OFF
    pin_idx = hand_base + PINKY_MCP_OFF

    needed = [wrist_idx, mid_idx, idx_idx, pin_idx]
    if any(scores[i] < min_score for i in needed):
        return None

    # Vector A: wrist → middle finger MCP  (roughly up the hand)
    A = kpts[mid_idx] - kpts[wrist_idx]
    # Vector B: index MCP → pinky MCP  (across the palm)
    B = kpts[pin_idx] - kpts[idx_idx]

    cross_z = float(A[0] * B[1] - A[1] * B[0])  # z-component of 2D cross

    # For right hand (base=112): palm-facing-camera → cross < 0
    # For left hand  (base=91):  palm-facing-camera → cross > 0
    if hand_base == RHAND_BASE:
        return cross_z < 0
    else:
        return cross_z > 0


class WristTracker:
    """Per-person-slot wrist position + hand scale + palm orientation history."""

    def __init__(self, max_history: int = 15):
        self.max_history = max_history
        self._pos_hist: list[deque] = []     # (x, y) wrist positions
        self._scale_hist: list[deque] = []   # hand scale (finger spread)
        self._palm_hist: list[deque] = []    # bool: palm facing camera?

    def resize(self, n: int):
        while len(self._pos_hist) < n:
            self._pos_hist.append(deque(maxlen=self.max_history))
            self._scale_hist.append(deque(maxlen=self.max_history))
            self._palm_hist.append(deque(maxlen=self.max_history))
        self._pos_hist = self._pos_hist[:n]
        self._scale_hist = self._scale_hist[:n]
        self._palm_hist = self._palm_hist[:n]

    def update(self, idx: int, xy: tuple[float, float] | None,
              scale: float | None, palm_facing: bool | None):
        if 0 <= idx < len(self._pos_hist):
            if xy is not None:
                self._pos_hist[idx].append(xy)
            if scale is not None:
                self._scale_hist[idx].append(scale)
            if palm_facing is not None:
                self._palm_hist[idx].append(palm_facing)

    def get_pos(self, idx: int) -> list[tuple[float, float]]:
        if 0 <= idx < len(self._pos_hist):
            return list(self._pos_hist[idx])
        return []

    def get_scale(self, idx: int) -> list[float]:
        if 0 <= idx < len(self._scale_hist):
            return list(self._scale_hist[idx])
        return []

    def get_palm(self, idx: int) -> list[bool]:
        if 0 <= idx < len(self._palm_hist):
            return list(self._palm_hist[idx])
        return []

    def clear_all(self):
        self._pos_hist.clear()
        self._scale_hist.clear()
        self._palm_hist.clear()


def _dominant_wrist(
    kpts: np.ndarray, scores: np.ndarray, min_score: float = 0.3,
) -> tuple[tuple[float, float] | None, int | None]:
    """Return ((x, y), hand_base) of the raised (higher) wrist, or (None, None)."""
    lw_ok = scores[KP_LWRIST] >= min_score
    rw_ok = scores[KP_RWRIST] >= min_score
    if lw_ok and rw_ok:
        if kpts[KP_LWRIST, 1] < kpts[KP_RWRIST, 1]:  # lower y = higher
            idx, hbase = KP_LWRIST, LHAND_BASE
        else:
            idx, hbase = KP_RWRIST, RHAND_BASE
    elif lw_ok:
        idx, hbase = KP_LWRIST, LHAND_BASE
    elif rw_ok:
        idx, hbase = KP_RWRIST, RHAND_BASE
    else:
        return None, None
    return (float(kpts[idx, 0]), float(kpts[idx, 1])), hbase


def classify_hand_gesture(
    wrist_history: list[tuple[float, float]],
    scale_history: list[float],
    palm_history: list[bool],
    bbox_h: float,
) -> int | None:
    """
    Classify stop / wave / come from wrist trajectory + palm orientation.

    come requires: palm facing away AND hand scale shrinking (moving away
    from camera).  Pure palm-away with no motion → stop (hand just resting).
    """
    MIN_FRAMES = 5
    if len(wrist_history) < MIN_FRAMES:
        return None

    pts = np.array(wrist_history[-15:])
    norm = max(bbox_h, 1.0)

    diffs = np.diff(pts, axis=0)
    speeds = np.linalg.norm(diffs, axis=1) / norm
    avg_speed = float(np.mean(speeds))

    # x-direction reversals → oscillation count
    dx = diffs[:, 0]
    signs = np.sign(dx)
    nonzero = signs[signs != 0]
    x_dir_changes = int(np.sum(np.abs(np.diff(nonzero)) > 0)) if len(nonzero) > 1 else 0
    x_range = float(np.ptp(pts[:, 0])) / norm

    is_oscillating = x_dir_changes >= 2 and x_range > 0.03

    # ── Palm orientation vote (majority of recent frames) ──
    palm_facing = None  # True = palm toward camera, False = away
    if len(palm_history) >= 3:
        recent = palm_history[-10:]
        n_facing = sum(recent)
        palm_facing = n_facing > len(recent) * 0.5

    # ── Scale trend (depth proxy) ──
    scale_shrinking = False
    if len(scale_history) >= MIN_FRAMES:
        scales = np.array(scale_history[-15:])
        if len(scales) >= 3:
            t = np.arange(len(scales), dtype=np.float32)
            slope = float(np.polyfit(t, scales, 1)[0])
            mean_scale = float(np.mean(scales))
            rel_slope = slope / max(mean_scale, 1.0)
            scale_shrinking = rel_slope < -0.002  # hand getting smaller → away

    # ── Decision tree ──
    if palm_facing is not None:
        if not palm_facing:
            # Back of hand visible → COME if scale shrinking (even slowly)
            if scale_shrinking:
                return COME_IDX
            # Palm away but hand still → idle (not stop)
            return IDLE_IDX
        else:
            # Palm facing camera → WAVE or STOP
            if is_oscillating:
                return WAVE_IDX
            return STOP_IDX

    # ── Fallback when palm orientation unavailable: use scale + motion ──
    if is_oscillating:
        return WAVE_IDX

    if len(scale_history) >= MIN_FRAMES:
        if avg_speed < 0.012:
            return STOP_IDX
        if scale_shrinking:
            return COME_IDX
        return STOP_IDX

    if avg_speed < 0.015:
        return STOP_IDX
    return COME_IDX


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

    # ── load RTMW for hand gesture refinement ────────────────────────
    import RTMW_detector_acc as rtmw

    rtmw_path = args.rtmw_model or rtmw.ensure_rtmw_model()
    rtmw_sess = rtmw.create_session(rtmw_path, use_trt=not args.no_trt)
    rtmw_in = rtmw_sess.get_inputs()[0].name
    rtmw_outs = [o.name for o in rtmw_sess.get_outputs()]
    wrist_tracker = WristTracker(max_history=15)
    come_hold_until: list[float] = []  # per-person timestamp until which "come" is held
    COME_HOLD_SEC = 0.55
    wave_hold_until: list[float] = []   # per-person timestamp until which "wave" is held
    WAVE_HOLD_SEC = 0.55
    wave_first_seen: list[float] = []   # per-person: when wave was first continuously detected
    WAVE_CONFIRM_SEC = 0.25              # must see wave for this long before displaying
    high_wave_hold_until: list[float] = []  # per-person timestamp until which "high_wave" is held
    HIGH_WAVE_HOLD_SEC = 0.3
    high_wave_first_seen: list[float] = []  # per-person: when high_wave was first continuously detected
    HIGH_WAVE_CONFIRM_SEC = 0.25
    head_y_per_person: list[float | None] = []  # per-person nose y-coordinate (updated each frame)
    print("[INFO] RTMW loaded for hand-gesture refinement")

    # ── open camera ──────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Camera {args.camera}: {actual_w}x{actual_h} @ {actual_fps:.0f}fps. Press 'q' or ESC to quit.")

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
        come_hold_until = _resize_list(come_hold_until, len(boxes), 0.0)
        wave_hold_until = _resize_list(wave_hold_until, len(boxes), 0.0)
        wave_first_seen = _resize_list(wave_first_seen, len(boxes), 0.0)
        high_wave_hold_until = _resize_list(high_wave_hold_until, len(boxes), 0.0)
        high_wave_first_seen = _resize_list(high_wave_first_seen, len(boxes), 0.0)
        head_y_per_person = _resize_list(head_y_per_person, len(boxes), None)

        # ── RTMW wrist tracking (every frame for smooth motion) ─────
        wrist_tracker.resize(len(boxes))
        for bi, (x1, y1, x2, y2) in enumerate(boxes):
            try:
                inp, warp = rtmw.preprocess(frame, (x1, y1, x2, y2), args.bbox_pad)
                outs = rtmw_sess.run(rtmw_outs, {rtmw_in: inp})
                kpts, kscores = rtmw.postprocess(outs, warp)
                wrist_xy, hbase = _dominant_wrist(kpts, kscores)
                scale = _hand_scale(kpts, kscores, hbase) if hbase is not None else None
                palm = _palm_facing_camera(kpts, kscores, hbase) if hbase is not None else None
                wrist_tracker.update(bi, wrist_xy, scale, palm)
                # Track head (nose) y-coordinate for high_wave detection
                if kscores[KP_NOSE] >= 0.3:
                    head_y_per_person[bi] = float(kpts[KP_NOSE, 1])
                else:
                    head_y_per_person[bi] = None
            except Exception:
                pass

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

                    # ── low-confidence gate: fall back to idle ──
                    LOW_CONF_THRESHOLD = 0.60
                    LOW_CONF_ACTIONS = {STOP_IDX, COME_IDX,
                                        LABELS.index("play phone"),
                                        LABELS.index("take a picture"),
                                        LABELS.index("love"),
                                        LABELS.index("phone call")}
                    for bi_idx in range(len(raw_classes)):
                        cls = raw_classes[bi_idx]
                        if cls is not None and cls in LOW_CONF_ACTIONS:
                            if raw_confs[bi_idx] < LOW_CONF_THRESHOLD:
                                raw_classes[bi_idx] = IDLE_IDX

                    # ── refine stop / wave / come via wrist motion ──
                    now = time.time()
                    for bi_idx in range(len(raw_classes)):
                        cls = raw_classes[bi_idx]
                        if cls is not None and cls in HAND_GESTURE_SET:
                            bh = boxes[bi_idx][3] - boxes[bi_idx][1]
                            gesture = classify_hand_gesture(
                                wrist_tracker.get_pos(bi_idx),
                                wrist_tracker.get_scale(bi_idx),
                                wrist_tracker.get_palm(bi_idx),
                                bh,
                            )
                            if gesture is not None:
                                # Check if wrist is at or above head level
                                wrist_above_head = False
                                nose_y = head_y_per_person[bi_idx] if bi_idx < len(head_y_per_person) else None
                                wrist_positions = wrist_tracker.get_pos(bi_idx)
                                if nose_y is not None and wrist_positions:
                                    wrist_y = wrist_positions[-1][1]
                                    # "around head" margin: 15% of bbox height
                                    margin = bh * 0.15
                                    wrist_above_head = wrist_y <= nose_y + margin

                                if gesture == COME_IDX:
                                    come_hold_until[bi_idx] = now + COME_HOLD_SEC
                                    raw_classes[bi_idx] = COME_IDX
                                    wave_first_seen[bi_idx] = 0.0
                                    high_wave_first_seen[bi_idx] = 0.0
                                elif now < come_hold_until[bi_idx]:
                                    # Hold come for duration after last trigger
                                    raw_classes[bi_idx] = COME_IDX
                                elif now < high_wave_hold_until[bi_idx]:
                                    # Hold high_wave for duration after last trigger
                                    raw_classes[bi_idx] = HIGH_WAVE_IDX
                                elif now < wave_hold_until[bi_idx]:
                                    # Hold wave for duration after last trigger
                                    raw_classes[bi_idx] = WAVE_IDX
                                elif gesture == WAVE_IDX and wrist_above_head:
                                    # High wave: oscillating + hand at/above head
                                    if high_wave_first_seen[bi_idx] == 0.0:
                                        high_wave_first_seen[bi_idx] = now
                                    if now - high_wave_first_seen[bi_idx] >= HIGH_WAVE_CONFIRM_SEC:
                                        high_wave_hold_until[bi_idx] = now + HIGH_WAVE_HOLD_SEC
                                        raw_classes[bi_idx] = HIGH_WAVE_IDX
                                        wave_first_seen[bi_idx] = 0.0
                                    else:
                                        raw_classes[bi_idx] = IDLE_IDX
                                elif gesture == WAVE_IDX:
                                    # Normal wave: oscillating, hand NOT above head
                                    high_wave_first_seen[bi_idx] = 0.0
                                    if wave_first_seen[bi_idx] == 0.0:
                                        wave_first_seen[bi_idx] = now
                                    if now - wave_first_seen[bi_idx] >= WAVE_CONFIRM_SEC:
                                        wave_hold_until[bi_idx] = now + WAVE_HOLD_SEC
                                        raw_classes[bi_idx] = WAVE_IDX
                                    else:
                                        raw_classes[bi_idx] = IDLE_IDX
                                elif gesture == STOP_IDX and wrist_above_head:
                                    # Hand still at head level – check if actually oscillating
                                    # (stop with hand above head stays stop)
                                    raw_classes[bi_idx] = STOP_IDX
                                    wave_first_seen[bi_idx] = 0.0
                                    high_wave_first_seen[bi_idx] = 0.0
                                else:
                                    wave_first_seen[bi_idx] = 0.0
                                    high_wave_first_seen[bi_idx] = 0.0
                                    raw_classes[bi_idx] = gesture

                stable_classes, pending_classes, pending_counts = _apply_inertia(
                    raw_classes, stable_classes, pending_classes, pending_counts, args.inertia,
                )
                for i in range(len(boxes)):
                    if stable_classes[i] is not None and stable_classes[i] == raw_classes[i]:
                        stable_confs[i] = raw_confs[i]
            else:
                stable_classes, stable_confs = [], []
                pending_classes, pending_counts = [], []
                wrist_tracker.clear_all()
                come_hold_until = []
                wave_hold_until = []
                wave_first_seen = []
                high_wave_hold_until = []
                high_wave_first_seen = []
                head_y_per_person = []

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
