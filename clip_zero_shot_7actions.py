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
    "small_love",
    "middle_finger",
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
    "small_love": [
        "a person making a small finger heart gesture with thumb and index finger",
        "a person crossing thumb tip and index fingertip to form a tiny heart",
        "a person holding up a Korean finger heart sign",
    ],
    "middle_finger": [
        "a person showing their middle finger",
        "a person giving the finger gesture",
        "a person flipping the bird with hand raised",
    ],
}

# Display names override: internal label → on-screen text
DISPLAY_NAMES: dict[str, str] = {
    "middle_finger": "Jerin, I know you XD",
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
    "small_love":     (255, 105, 180),# pink
    "middle_finger":     (0, 0, 255),  # red
}

MIN_DISPLAY_CONF = 0.15  # show label only above this similarity

LOVE_IDX = LABELS.index("love")

# ── Heart animation helpers ──────────────────────────────────────────
ANIM_DURATION = 2.0  # seconds

import math as _math


def _heart_contour(cx: int, cy: int, size: float) -> np.ndarray:
    """Return a heart-shaped contour centred at (cx, cy) with given size."""
    t = np.linspace(0, 2 * _math.pi, 80)
    x = 16 * np.sin(t) ** 3
    y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    scale = size / 32.0  # raw heart is ~32 units wide
    pts = np.column_stack([x * scale + cx, y * scale + cy]).astype(np.int32)
    return pts.reshape((-1, 1, 2))


def _draw_heart(frame: np.ndarray, cx: int, cy: int, size: float,
                color: tuple[int, int, int], alpha: float = 1.0) -> None:
    """Draw a filled heart with optional transparency on *frame*."""
    if alpha <= 0 or size < 2:
        return
    contour = _heart_contour(cx, cy, size)
    if alpha >= 0.95:
        cv2.fillPoly(frame, [contour], color, cv2.LINE_AA)
    else:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [contour], color, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def _draw_love_animation(frame: np.ndarray, cx: int, cy: int,
                         bbox_h: int, progress: float) -> None:
    """Single large heart: scale up then fade out. progress ∈ [0, 1]."""
    # Phase 1 (0→0.4): grow from 0 to full size, full opacity
    # Phase 2 (0.4→1): stay full size, fade out
    max_size = bbox_h * 0.6
    if progress < 0.4:
        t = progress / 0.4
        size = max_size * t
        alpha = 1.0
    else:
        t = (progress - 0.4) / 0.6
        size = max_size
        alpha = 1.0 - t
    _draw_heart(frame, cx, cy, size, (180, 0, 255), alpha)


def _draw_small_love_animation(frame: np.ndarray, cx: int, bot_y: int,
                               bbox_h: int, progress: float) -> None:
    """Multiple small hearts floating upward. progress ∈ [0, 1]."""
    # Use fixed seed offsets for deterministic but varied positions
    _offsets = [(-0.25, 0.0), (0.15, 0.12), (-0.10, 0.25), (0.25, 0.38),
                (-0.20, 0.50), (0.10, 0.62), (0.0, 0.75)]
    heart_size = bbox_h * 0.08
    travel = bbox_h * 1.2  # how far hearts travel upward

    for dx_frac, delay in _offsets:
        t = progress - delay * 0.5  # staggered start
        if t < 0 or t > 1:
            continue
        # Float upward
        y = int(bot_y - t * travel)
        x = int(cx + dx_frac * bbox_h * 0.5)
        # Fade out near end
        alpha = 1.0 - max(0.0, (t - 0.6) / 0.4)
        # Slight size pulse
        s = heart_size * (0.8 + 0.4 * _math.sin(t * _math.pi))
        _draw_heart(frame, x, y, s, (180, 105, 255), alpha)


# ── RTMW hand-gesture refinement ─────────────────────────────────────
# When CLIP predicts stop / wave / come, use wrist motion to disambiguate.

COME_IDX = LABELS.index("come")      # 0
WAVE_IDX = LABELS.index("wave")      # 1
STOP_IDX = LABELS.index("stop")      # 2
IDLE_IDX = LABELS.index("idle")      # 6
HIGH_WAVE_IDX = LABELS.index("high_wave")  # 8
SMALL_LOVE_IDX = LABELS.index("small_love")  # 9
MIDFINGER_IDX = LABELS.index("middle_finger")  # 10
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
RING_MCP_OFF = 13
THUMB_TIP_OFF = 4

# Additional hand keypoint offsets for finger-heart (small_love) detection
# Per-hand 21 keypoints: 0=wrist, 1–4=thumb, 5–8=index, 9–12=middle, 13–16=ring, 17–20=pinky
# Tip offsets: thumb=4, index=8, middle=12, ring=16, pinky=20
# PIP (proximal interphalangeal) offsets: index=6, middle=10, ring=14, pinky=18
INDEX_TIP_OFF = 8
MIDDLE_TIP_OFF = 12
RING_TIP_OFF = 16
PINKY_TIP_OFF = 20
INDEX_PIP_OFF = 6
MIDDLE_PIP_OFF = 10
RING_PIP_OFF = 14
PINKY_PIP_OFF = 18
THUMB_IP_OFF = 2  # thumb interphalangeal joint (PIP equivalent)


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


def _detect_finger_heart(
    kpts: np.ndarray, scores: np.ndarray, hand_base: int,
    min_score: float = 0.25,
    min_hand_scale_px: float = 60.0,
) -> bool:
    """
    Detect the Korean finger-heart (small love) gesture using 2D hand keypoints.

    Criteria:
      0. Hand must be close to camera (hand_scale >= min_hand_scale_px pixels).
      1. Thumb tip and index fingertip are very close (touching / crossing).
      2. Index tip y ≤ thumb tip y (index crosses over thumb, forming a V).
      3. Middle, ring, and pinky fingers are curled (tip closer to wrist than PIP).
    """
    thumb_tip = hand_base + THUMB_TIP_OFF
    idx_tip = hand_base + INDEX_TIP_OFF
    mid_tip = hand_base + MIDDLE_TIP_OFF
    ring_tip = hand_base + RING_TIP_OFF
    pink_tip = hand_base + PINKY_TIP_OFF
    mid_pip = hand_base + MIDDLE_PIP_OFF
    ring_pip = hand_base + RING_PIP_OFF
    pink_pip = hand_base + PINKY_PIP_OFF
    wrist = hand_base  # offset 0

    # Need thumb tip, index tip, and wrist with decent confidence
    essential = [thumb_tip, idx_tip, wrist]
    if any(scores[i] < min_score for i in essential):
        return False

    # 1. Thumb tip ↔ index tip proximity (normalised by hand size)
    hand_sc = _hand_scale(kpts, scores, hand_base, min_score)
    if hand_sc is None or hand_sc < 1.0:
        return False
    # 0. Hand must be large enough in frame (close to camera)
    if hand_sc < min_hand_scale_px:
        return False
    tip_dist = float(np.linalg.norm(kpts[thumb_tip] - kpts[idx_tip]))
    if tip_dist / hand_sc > 0.38:  # tips must be very close relative to hand size
        return False

    # 2. Index tip should be at or above thumb tip (lower y = higher in image)
    #    Allow a small tolerance downward (10% of hand scale)
    if kpts[idx_tip, 1] > kpts[thumb_tip, 1] + hand_sc * 0.10:
        return False

    # 2b. Thumb and index tips must both be above (lower y) the wrist —
    #     a finger-heart is held up, not down.
    wrist_y = float(kpts[wrist, 1])
    if float(kpts[thumb_tip, 1]) > wrist_y or float(kpts[idx_tip, 1]) > wrist_y:
        return False

    # 3. Other fingers curled: tip closer to wrist than PIP is
    curled_count = 0
    for ftip, fpip in [(mid_tip, mid_pip), (ring_tip, ring_pip), (pink_tip, pink_pip)]:
        if scores[ftip] < min_score or scores[fpip] < min_score:
            continue
        tip_to_wrist = float(np.linalg.norm(kpts[ftip] - kpts[wrist]))
        pip_to_wrist = float(np.linalg.norm(kpts[fpip] - kpts[wrist]))
        if tip_to_wrist < pip_to_wrist * 1.05:
            curled_count += 1
    if curled_count < 3:  # all 3 remaining fingers must be curled
        return False

    return True


def _hand_bbox(
    kpts: np.ndarray, scores: np.ndarray, hand_base: int,
    img_h: int, img_w: int, min_score: float = 0.25, pad: float = 0.4,
) -> tuple[int, int, int, int] | None:
    """Return (x1, y1, x2, y2) bounding box around visible hand keypoints, or None."""
    xs, ys = [], []
    for off in range(21):
        idx = hand_base + off
        if scores[idx] >= min_score:
            xs.append(float(kpts[idx, 0]))
            ys.append(float(kpts[idx, 1]))
    if len(xs) < 4:
        return None
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    half_w = (max(xs) - min(xs)) / 2 * (1 + pad)
    half_h = (max(ys) - min(ys)) / 2 * (1 + pad)
    side = max(half_w, half_h)  # square crop
    x1 = max(0, int(cx - side))
    y1 = max(0, int(cy - side))
    x2 = min(img_w, int(cx + side))
    y2 = min(img_h, int(cy + side))
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    return (x1, y1, x2, y2)


def _detect_middle_finger(
    kpts: np.ndarray, scores: np.ndarray, hand_base: int,
    min_score: float = 0.25,
) -> bool:
    """
    Detect the middle-finger gesture using 2D hand keypoints.

    Uses spatial position verification: the extended fingertip must be
    physically located at the centre of the hand (between index MCP and
    pinky MCP), not at the edges.  MCP joints on the palm are reliably
    detected and don't get swapped, so this catches cases where RTMW
    mislabels which finger is extended.
    """
    wrist = hand_base
    mid_tip = hand_base + MIDDLE_TIP_OFF
    mid_pip = hand_base + MIDDLE_PIP_OFF
    mid_mcp = hand_base + MIDDLE_MCP_OFF
    idx_tip = hand_base + INDEX_TIP_OFF
    idx_pip = hand_base + INDEX_PIP_OFF
    idx_mcp = hand_base + INDEX_MCP_OFF
    ring_tip = hand_base + RING_TIP_OFF
    ring_pip = hand_base + RING_PIP_OFF
    ring_mcp = hand_base + RING_MCP_OFF
    pink_tip = hand_base + PINKY_TIP_OFF
    pink_pip = hand_base + PINKY_PIP_OFF
    pink_mcp = hand_base + PINKY_MCP_OFF
    thumb_tip = hand_base + THUMB_TIP_OFF
    thumb_ip = hand_base + THUMB_IP_OFF

    # Need middle finger tip, MCP, wrist, and palm MCPs for spatial check
    essential = [wrist, mid_tip, mid_mcp, idx_mcp, pink_mcp]
    if any(scores[i] < min_score for i in essential):
        return False

    hand_sc = _hand_scale(kpts, scores, hand_base, min_score)
    if hand_sc is None or hand_sc < 1.0:
        return False

    # 1. Middle finger must be extended: tip is far from wrist and above MCP
    mid_tip_to_wrist = float(np.linalg.norm(kpts[mid_tip] - kpts[wrist]))
    mid_mcp_to_wrist = float(np.linalg.norm(kpts[mid_mcp] - kpts[wrist]))
    if mid_tip_to_wrist < mid_mcp_to_wrist * 1.4:
        return False  # middle finger not extended enough
    # Middle tip should be above (lower y) its MCP
    if kpts[mid_tip, 1] > kpts[mid_mcp, 1]:
        return False

    # ── SPATIAL POSITION CHECK ──────────────────────────────────────
    # Project the extended fingertip onto the palm axis (index_MCP → pinky_MCP)
    # to verify it's physically at the middle-finger position, not at the
    # index/thumb/pinky edge.  This is robust to RTMW keypoint swaps.
    palm_vec = kpts[pink_mcp] - kpts[idx_mcp]  # vector across the palm
    palm_len_sq = float(np.dot(palm_vec, palm_vec))
    if palm_len_sq < 1.0:
        return False  # degenerate palm detection
    # Where does the middle fingertip project onto this axis?  (0 = index side, 1 = pinky side)
    tip_frac = float(np.dot(kpts[mid_tip] - kpts[idx_mcp], palm_vec)) / palm_len_sq
    # Where does the true middle MCP project?  (should be ~0.3-0.5)
    mcp_frac = float(np.dot(kpts[mid_mcp] - kpts[idx_mcp], palm_vec)) / palm_len_sq
    # The tip must be near the middle MCP position (within ±0.25)
    if abs(tip_frac - mcp_frac) > 0.25:
        return False  # tip is physically at wrong finger position
    # Reject if tip projects outside the plausible middle-finger zone [0.15, 0.65]
    if tip_frac < 0.15 or tip_frac > 0.65:
        return False  # too far to index side or pinky side

    # 1b. Index finger must NOT be extended (rejects thumb+index poses
    #     where RTMW swaps index/middle keypoints).
    if scores[idx_tip] >= min_score and scores[idx_mcp] >= min_score:
        if kpts[idx_tip, 1] < kpts[idx_mcp, 1]:  # index tip above MCP → extended
            return False

    # 1c. Thumb must NOT be extended.
    if scores[thumb_tip] >= min_score:
        thumb_dist = float(np.linalg.norm(kpts[thumb_tip] - kpts[wrist]))
        if thumb_dist > hand_sc * 0.65:
            return False  # thumb looks extended
        # Also reject if thumb tip is above its IP joint (pointing up)
        if scores[thumb_ip] >= min_score and kpts[thumb_tip, 1] < kpts[thumb_ip, 1] - 3:
            return False

    # 1d. Middle fingertip must be the HIGHEST point (lowest y) among all
    #     visible fingertips.
    mid_y = float(kpts[mid_tip, 1])
    for otip in [idx_tip, ring_tip, pink_tip, thumb_tip]:
        if scores[otip] < min_score:
            continue
        if float(kpts[otip, 1]) < mid_y - 3.0:  # 3px tolerance
            return False  # another fingertip is higher → not middle finger

    # 2. Adjacent fingers (index & ring) MUST be visible and clearly shorter.
    for adj_tip in [idx_tip, ring_tip]:
        if scores[adj_tip] < min_score:
            return False  # can't verify adjacent finger is curled → reject
        adj_dist = float(np.linalg.norm(kpts[adj_tip] - kpts[wrist]))
        if adj_dist >= mid_tip_to_wrist * 0.70:
            return False  # adjacent finger too extended

    # 3. ALL other fingertips: verify at least 3 are visibly shorter.
    other_tips = [idx_tip, ring_tip, pink_tip, thumb_tip]
    verified_shorter = 0
    for otip in other_tips:
        if scores[otip] < min_score:
            continue
        other_dist = float(np.linalg.norm(kpts[otip] - kpts[wrist]))
        if other_dist >= mid_tip_to_wrist * 0.80:
            return False  # another finger is similarly or more extended
        verified_shorter += 1
    if verified_shorter < 3:
        return False  # can't confirm enough fingers are shorter

    # 4. Other fingers must be curled (tip closer to wrist than their PIP)
    curled = 0
    checks = [
        (idx_tip, idx_pip),
        (ring_tip, ring_pip),
        (pink_tip, pink_pip),
        (thumb_tip, thumb_ip),
    ]
    for ftip, fpip in checks:
        if scores[ftip] < min_score or scores[fpip] < min_score:
            continue
        tip_dist = float(np.linalg.norm(kpts[ftip] - kpts[wrist]))
        pip_dist = float(np.linalg.norm(kpts[fpip] - kpts[wrist]))
        if tip_dist < pip_dist * 1.2:
            curled += 1
    if curled < 3:  # at least 3 of 4 non-middle fingers must be curled
        return False

    return True


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
    small_love_per_person: list[bool] = []  # per-person: finger-heart detected this frame
    small_love_history: list[deque] = []     # per-person: rolling window of last 10 detections
    SMALL_LOVE_CONFIRM_FRAMES = 8            # need 8/10 recent frames positive
    small_love_hold_until: list[float] = []  # per-person timestamp until which "small_love" is held
    SMALL_LOVE_HOLD_SEC = 0.4
    midfinger_per_person: list[bool] = []   # per-person: middle finger detected this frame
    midfinger_hold_until: list[float] = []  # per-person timestamp until which "middle_finger" is held
    MIDFINGER_HOLD_SEC = 0.5
    # Per-person heart animation state
    anim_type: list[str | None] = []    # None / "love" / "small_love"
    anim_start: list[float] = []        # timestamp when animation started
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
        small_love_per_person = _resize_list(small_love_per_person, len(boxes), False)
        while len(small_love_history) < len(boxes):
            small_love_history.append(deque(maxlen=10))
        small_love_history = small_love_history[:len(boxes)]
        small_love_hold_until = _resize_list(small_love_hold_until, len(boxes), 0.0)
        midfinger_per_person = _resize_list(midfinger_per_person, len(boxes), False)
        midfinger_hold_until = _resize_list(midfinger_hold_until, len(boxes), 0.0)
        anim_type = _resize_list(anim_type, len(boxes), None)
        anim_start = _resize_list(anim_start, len(boxes), 0.0)

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
                # Check both hands for finger-heart (small_love)
                fh_left = _detect_finger_heart(kpts, kscores, LHAND_BASE)
                fh_right = _detect_finger_heart(kpts, kscores, RHAND_BASE)
                fh_detected = fh_left or fh_right
                # Only accept if hand is nearly still (low wrist speed)
                if fh_detected:
                    bh = y2 - y1
                    pos_hist = wrist_tracker.get_pos(bi)
                    if len(pos_hist) >= 3:
                        pts = np.array(pos_hist[-10:])
                        diffs = np.diff(pts, axis=0)
                        speeds = np.linalg.norm(diffs, axis=1) / max(bh, 1.0)
                        avg_speed = float(np.mean(speeds))
                        if avg_speed > 0.012:  # hand moving too fast → reject
                            fh_detected = False
                    else:
                        fh_detected = False  # not enough history yet
                small_love_per_person[bi] = fh_detected
                # Update rolling history and only confirm if 8/10 recent frames
                small_love_history[bi].append(fh_detected)
                if fh_detected and sum(small_love_history[bi]) < SMALL_LOVE_CONFIRM_FRAMES:
                    small_love_per_person[bi] = False  # not enough consecutive evidence
                # Check both hands for middle finger (Jerin)
                # Step 1: RTMW pose check
                mf_hand_base = None
                if _detect_middle_finger(kpts, kscores, LHAND_BASE):
                    mf_hand_base = LHAND_BASE
                elif _detect_middle_finger(kpts, kscores, RHAND_BASE):
                    mf_hand_base = RHAND_BASE
                # Step 2: CLIP visual double-check on hand crop
                mf_confirmed = False
                if mf_hand_base is not None:
                    h_fr, w_fr = frame.shape[:2]
                    hbox = _hand_bbox(kpts, kscores, mf_hand_base, h_fr, w_fr)
                    if hbox is not None:
                        hx1, hy1, hx2, hy2 = hbox
                        hand_crop = cv2.cvtColor(frame[hy1:hy2, hx1:hx2], cv2.COLOR_BGR2RGB)
                        if hand_crop.size > 0:
                            hand_pil = Image.fromarray(hand_crop)
                            hand_t = preprocess(hand_pil).unsqueeze(0).to(device)
                            hand_feat = clip_model.encode_image(hand_t).float()
                            hand_feat = F.normalize(hand_feat, dim=-1)
                            hand_sims = (hand_feat @ text_feats.T).squeeze(0)
                            hand_probs = torch.softmax(hand_sims * 100.0, dim=0)
                            midfinger_conf = float(hand_probs[MIDFINGER_IDX])
                            if midfinger_conf > 0.60:
                                mf_confirmed = True
                midfinger_per_person[bi] = mf_confirmed
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
                        # ── 1. Hold check (unconditional – overrides CLIP) ──
                        if now < midfinger_hold_until[bi_idx]:
                            raw_classes[bi_idx] = MIDFINGER_IDX
                            continue
                        if now < small_love_hold_until[bi_idx]:
                            raw_classes[bi_idx] = SMALL_LOVE_IDX
                            continue
                        if now < come_hold_until[bi_idx]:
                            raw_classes[bi_idx] = COME_IDX
                            continue
                        if now < high_wave_hold_until[bi_idx]:
                            raw_classes[bi_idx] = HIGH_WAVE_IDX
                            continue
                        if now < wave_hold_until[bi_idx]:
                            raw_classes[bi_idx] = WAVE_IDX
                            continue

                        # ── 1b. Finger-heart override (pose-based, any CLIP class) ──
                        if midfinger_per_person[bi_idx]:
                            midfinger_hold_until[bi_idx] = now + MIDFINGER_HOLD_SEC
                            raw_classes[bi_idx] = MIDFINGER_IDX
                            continue
                        if small_love_per_person[bi_idx]:
                            small_love_hold_until[bi_idx] = now + SMALL_LOVE_HOLD_SEC
                            raw_classes[bi_idx] = SMALL_LOVE_IDX
                            continue

                        # ── 2. Gesture refinement (only for hand-gesture classes) ──
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
                                    # Hand still at head level (not oscillating → stop)
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
                small_love_per_person = []
                small_love_history = []
                small_love_hold_until = []
                midfinger_per_person = []
                midfinger_hold_until = []
                anim_type = []
                anim_start = []

        # ── draw ─────────────────────────────────────────────────────
        now_draw = time.time()
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
                display_name = DISPLAY_NAMES.get(label_name, label_name)
                color = LABEL_COLORS.get(label_name, (0, 255, 0))
                text = f"{int(cls) + 1} {display_name} {conf * 100:.0f}%"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if text:
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                ty = max(th + 4, y1 - 8)
                cv2.putText(frame, text, (x1, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

            # ── trigger / update heart animations ──
            if cls == LOVE_IDX:
                if anim_type[i] != "love":
                    anim_type[i] = "love"
                    anim_start[i] = now_draw
            elif cls == SMALL_LOVE_IDX:
                if anim_type[i] != "small_love":
                    anim_type[i] = "small_love"
                    anim_start[i] = now_draw
            else:
                # Don't clear mid-animation — let it finish naturally
                pass

            # ── render heart animation ──
            if anim_type[i] is not None and anim_start[i] > 0:
                elapsed = now_draw - anim_start[i]
                if elapsed < ANIM_DURATION:
                    progress = elapsed / ANIM_DURATION
                    bh = y2 - y1
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    if anim_type[i] == "love":
                        _draw_love_animation(frame, cx, cy, bh, progress)
                    elif anim_type[i] == "small_love":
                        _draw_small_love_animation(frame, cx, y2, bh, progress)
                else:
                    anim_type[i] = None
                    anim_start[i] = 0.0

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
