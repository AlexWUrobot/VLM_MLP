import argparse
import os
import threading
import time
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class FusionTemporalMLP(nn.Module):
    def __init__(
        self,
        *,
        clip_dim: int,
        pose_dim: int,
        hand_dim: int,
        hand_mask_dim: int,
        phone_probe_dim: int = 0,
        proj_dim: int,
        hidden: int,
        num_classes: int,
        num_frames: int,
        feat_agg: str,
    ):
        super().__init__()

        self.clip_dim = int(clip_dim)
        self.pose_dim = int(pose_dim)
        self.hand_dim = int(hand_dim)
        self.hand_mask_dim = int(hand_mask_dim)
        self.phone_probe_dim = int(phone_probe_dim)
        self.per_frame_dim = self.clip_dim + self.pose_dim + self.hand_dim + self.hand_mask_dim + self.phone_probe_dim

        self.num_frames = max(1, int(num_frames))
        self.feat_agg = str(feat_agg).lower().strip()
        if self.feat_agg not in {"concat", "mean"}:
            raise ValueError("feat_agg must be 'concat' or 'mean'")

        self.proj_dim = int(proj_dim)

        def _proj(in_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(int(in_dim), self.proj_dim),
                nn.LayerNorm(self.proj_dim),
                nn.ReLU(),
            )

        self.clip_proj = _proj(self.clip_dim)
        self.pose_proj = _proj(self.pose_dim)
        self.hand_proj = _proj(self.hand_dim)
        self.hand_mask_proj = _proj(self.hand_mask_dim)
        if self.phone_probe_dim > 0:
            self.phone_probe_proj = _proj(self.phone_probe_dim)

        self.frame_embed_dim = self.proj_dim * (5 if self.phone_probe_dim > 0 else 4)

        if self.feat_agg == "concat":
            head_in = self.frame_embed_dim * self.num_frames
        else:
            head_in = self.frame_embed_dim

        self.fc1 = nn.Linear(head_in, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.feat_agg == "concat":
            if x.shape[1] != self.num_frames * self.per_frame_dim:
                raise RuntimeError(
                    f"input dim mismatch: got {x.shape[1]} expected {self.num_frames * self.per_frame_dim}"
                )
            xf = x.view(x.shape[0], self.num_frames, self.per_frame_dim)
        else:
            if x.shape[1] != self.per_frame_dim:
                raise RuntimeError(f"input dim mismatch: got {x.shape[1]} expected {self.per_frame_dim}")
            xf = x.view(x.shape[0], 1, self.per_frame_dim)

        o1 = self.clip_dim
        o2 = o1 + self.pose_dim
        o3 = o2 + self.hand_dim
        o4 = o3 + self.hand_mask_dim
        clip = xf[:, :, :o1]
        pose = xf[:, :, o1:o2]
        hands = xf[:, :, o2:o3]
        hmask = xf[:, :, o3:o4]

        b, t, _ = xf.shape
        clip_e = self.clip_proj(clip.reshape(b * t, self.clip_dim))
        pose_e = self.pose_proj(pose.reshape(b * t, self.pose_dim))
        hand_e = self.hand_proj(hands.reshape(b * t, self.hand_dim))
        mask_e = self.hand_mask_proj(hmask.reshape(b * t, self.hand_mask_dim))
        parts = [clip_e, pose_e, hand_e, mask_e]
        if self.phone_probe_dim > 0:
            phone = xf[:, :, o4:]
            phone_e = self.phone_probe_proj(phone.reshape(b * t, self.phone_probe_dim))
            parts.append(phone_e)

        frame_e = torch.cat(parts, dim=1).view(b, t, self.frame_embed_dim)

        if self.feat_agg == "concat":
            h = frame_e.reshape(b, t * self.frame_embed_dim)
        else:
            h = frame_e.mean(dim=1)

        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


class FusionTemporalGRU(nn.Module):
    """GRU variant — processes frame sequences with temporal awareness."""

    def __init__(
        self,
        *,
        clip_dim: int,
        pose_dim: int,
        hand_dim: int,
        hand_mask_dim: int,
        phone_probe_dim: int = 0,
        proj_dim: int,
        hidden: int,
        num_classes: int,
        num_frames: int,
        rnn_hidden: int = 192,
        rnn_layers: int = 1,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.clip_dim = int(clip_dim)
        self.pose_dim = int(pose_dim)
        self.hand_dim = int(hand_dim)
        self.hand_mask_dim = int(hand_mask_dim)
        self.phone_probe_dim = int(phone_probe_dim)
        self.per_frame_dim = self.clip_dim + self.pose_dim + self.hand_dim + self.hand_mask_dim + self.phone_probe_dim
        self.num_frames = max(1, int(num_frames))

        self.proj_dim = int(proj_dim)
        self.rnn_hidden = int(rnn_hidden)
        self.rnn_layers = max(1, int(rnn_layers))
        self.bidirectional = bool(bidirectional)

        def _proj(in_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(int(in_dim), self.proj_dim),
                nn.LayerNorm(self.proj_dim),
                nn.ReLU(),
            )

        self.clip_proj = _proj(self.clip_dim)
        self.pose_proj = _proj(self.pose_dim)
        self.hand_proj = _proj(self.hand_dim)
        self.hand_mask_proj = _proj(self.hand_mask_dim)
        if self.phone_probe_dim > 0:
            self.phone_probe_proj = _proj(self.phone_probe_dim)
        self.frame_embed_dim = self.proj_dim * (5 if self.phone_probe_dim > 0 else 4)

        self.gru = nn.GRU(
            input_size=self.frame_embed_dim,
            hidden_size=self.rnn_hidden,
            num_layers=self.rnn_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=0.2 if self.rnn_layers > 1 else 0.0,
        )

        head_in = self.rnn_hidden * (2 if self.bidirectional else 1)
        self.fc1 = nn.Linear(head_in, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.num_frames * self.per_frame_dim:
            raise RuntimeError(
                f"input dim mismatch: got {x.shape[1]} expected {self.num_frames * self.per_frame_dim}"
            )
        xf = x.view(x.shape[0], self.num_frames, self.per_frame_dim)

        o1 = self.clip_dim
        o2 = o1 + self.pose_dim
        o3 = o2 + self.hand_dim
        o4 = o3 + self.hand_mask_dim
        clip = xf[:, :, :o1]
        pose = xf[:, :, o1:o2]
        hands = xf[:, :, o2:o3]
        hmask = xf[:, :, o3:o4]

        b, t, _ = xf.shape
        clip_e = self.clip_proj(clip.reshape(b * t, self.clip_dim))
        pose_e = self.pose_proj(pose.reshape(b * t, self.pose_dim))
        hand_e = self.hand_proj(hands.reshape(b * t, self.hand_dim))
        mask_e = self.hand_mask_proj(hmask.reshape(b * t, self.hand_mask_dim))
        parts = [clip_e, pose_e, hand_e, mask_e]
        if self.phone_probe_dim > 0:
            phone = xf[:, :, o4:]
            phone_e = self.phone_probe_proj(phone.reshape(b * t, self.phone_probe_dim))
            parts.append(phone_e)
        frame_e = torch.cat(parts, dim=1).view(b, t, self.frame_embed_dim)

        out, hidden = self.gru(frame_e)
        if self.bidirectional:
            last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            last_hidden = hidden[-1]

        h = F.relu(self.fc1(last_hidden))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


POSE_MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}
POSE_MODEL_PATHS = {
    k: os.path.join(os.path.dirname(__file__), "models", f"pose_landmarker_{k}.task")
    for k in POSE_MODEL_URLS
}

HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
HAND_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")

POSE_DIM = 165
HAND_DIM = 126
HAND_MASK_DIM = HAND_DIM

PHONE_PROBE_PROMPTS = [
    "a person holding a cell phone",
    "a person talking on a phone",
    "a person looking at a phone screen",
    "a person with empty hands",
    "a person waving their hand",
]
PHONE_PROBE_DIM = len(PHONE_PROBE_PROMPTS)

COCO_PHONE_CLASS = 67  # cell phone in COCO

DEFAULT_CLIP_ONNX = "models/clip_vit_b32_visual.onnx"


def _export_clip_visual_onnx(
    clip_model_name: str,
    clip_pretrained: str,
    onnx_path: str,
    device: str = "cpu",
) -> None:
    """Export the CLIP visual encoder to ONNX (dynamic batch)."""
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(clip_model_name, pretrained=clip_pretrained)
    model = model.to(device).eval()
    visual = model.visual

    # Wrap so output is just the feature vector (before L2 norm)
    class _VisualWrapper(torch.nn.Module):
        def __init__(self, vis):
            super().__init__()
            self.vis = vis

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.vis(x)

    wrapper = _VisualWrapper(visual).eval()
    dummy = torch.randn(1, 3, 224, 224, device=device)
    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy,),
        onnx_path,
        input_names=["pixel_values"],
        output_names=["image_features"],
        dynamic_axes={"pixel_values": {0: "batch"}, "image_features": {0: "batch"}},
        opset_version=17,
    )
    print(f"[clip onnx] exported to {onnx_path}")


def _create_ort_session(onnx_path: str):
    """Create an ONNX Runtime session: TensorRT FP16 > CUDA > CPU."""
    import onnxruntime as ort

    providers_tried: list[str] = []

    # Try TensorRT EP with FP16
    if "TensorrtExecutionProvider" in ort.get_available_providers():
        trt_opts = {
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": os.path.join(os.path.dirname(onnx_path), "trt_cache"),
        }
        try:
            sess = ort.InferenceSession(
                onnx_path,
                providers=[("TensorrtExecutionProvider", trt_opts), "CUDAExecutionProvider"],
            )
            active = sess.get_providers()
            print(f"[clip ort] providers={active}")
            return sess
        except Exception as e:
            providers_tried.append(f"TRT failed: {e}")

    # Try CUDA EP
    if "CUDAExecutionProvider" in ort.get_available_providers():
        try:
            sess = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider"])
            active = sess.get_providers()
            print(f"[clip ort] providers={active}")
            return sess
        except Exception as e:
            providers_tried.append(f"CUDA failed: {e}")

    # CPU fallback
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    active = sess.get_providers()
    print(f"[clip ort] providers={active} (tried: {providers_tried})")
    return sess


def ensure_model(*, model_url: str, model_path: str, label: str) -> None:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if os.path.exists(model_path):
        return
    print(f"Downloading {label} model to {model_path} ...")
    try:
        urllib.request.urlretrieve(model_url, model_path)
    except Exception as exc:
        raise SystemExit(
            f"Failed to download the MediaPipe {label} model. "
            f"Download it manually from {model_url} and save it to {model_path}."
        ) from exc


def _lm_xyz(lm) -> Tuple[float, float, float]:
    return float(lm.x), float(lm.y), float(lm.z)


def pose_vector_165(pose_landmarks: Sequence) -> list[float]:
    nan = float("nan")
    vec: list[float] = []

    lm_list = list(pose_landmarks)
    if len(lm_list) < 33:
        lm_list = lm_list + [None] * (33 - len(lm_list))
    elif len(lm_list) > 33:
        lm_list = lm_list[:33]

    for lm in lm_list:
        if lm is None:
            vec.extend([nan, nan, nan, nan, nan])
            continue
        x, y, z = _lm_xyz(lm)
        vis = float(getattr(lm, "visibility", nan))
        pres = float(getattr(lm, "presence", nan))
        vec.extend([x, y, z, vis, pres])

    if len(vec) != POSE_DIM:
        raise RuntimeError(f"pose vector length mismatch: {len(vec)}")
    return vec


def hand_vector_two_hands(
    hand_landmarks_list: Sequence[Sequence],
    handedness_list: Optional[Sequence] = None,
) -> list[float]:
    nan = float("nan")
    one_hand_len = 21 * 3
    out = [nan] * (2 * one_hand_len)

    def _hand_side(i: int) -> Optional[str]:
        if not handedness_list or i >= len(handedness_list) or not handedness_list[i]:
            return None
        cat0 = handedness_list[i][0]
        for attr in ("category_name", "display_name", "name"):
            v = getattr(cat0, attr, None)
            if isinstance(v, str) and v:
                return v
        return None

    filled = {"Left": False, "Right": False}

    for i, hand_landmarks in enumerate(hand_landmarks_list):
        side = _hand_side(i)
        side_norm: Optional[str] = None
        if isinstance(side, str):
            s = side.strip().lower()
            if s == "left":
                side_norm = "Left"
            elif s == "right":
                side_norm = "Right"

        if side_norm is None:
            side_norm = "Left" if not filled["Left"] else ("Right" if not filled["Right"] else None)

        if side_norm is None:
            continue

        vec: list[float] = []
        for lm in hand_landmarks:
            x, y, z = _lm_xyz(lm)
            vec.extend([x, y, z])

        if len(vec) != one_hand_len:
            continue

        offset = 0 if side_norm == "Left" else one_hand_len
        out[offset : offset + one_hand_len] = vec
        filled[side_norm] = True

    if len(out) != HAND_DIM:
        raise RuntimeError(f"hand vector length mismatch: {len(out)}")
    return out


def combine_frame_features(
    clip_feat: torch.Tensor,
    pose_vec: list[float],
    hand_vec: list[float],
    phone_probe: torch.Tensor | None = None,
) -> torch.Tensor:
    # pose: impute NaN -> 0 (no mask)
    pose_t = torch.tensor(pose_vec, dtype=torch.float32)
    pose_t = torch.nan_to_num(pose_t, nan=0.0)

    # hands: impute NaN -> 0 + mask only for hands
    hands_t = torch.tensor(hand_vec, dtype=torch.float32)
    hand_mask_t = torch.isnan(hands_t).to(torch.float32)
    hands_t = torch.nan_to_num(hands_t, nan=0.0)

    clip_t = clip_feat.to(torch.float32)

    # [clip(512), pose(165), hands(126), hand_mask(126), phone_probe(P)]
    parts = [clip_t, pose_t, hands_t, hand_mask_t]
    if phone_probe is not None:
        parts.append(phone_probe.to(torch.float32))
    return torch.cat(parts, dim=0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Webcam live action prediction (YOLO crop + CLIP-B + pose/hands + temporal MLP)"
    )
    p.add_argument("--ckpt", default="mlp_clip_b_8frames_marker.pt", help="Path to trained MLP checkpoint")
    p.add_argument("--camera", type=int, default=0, help="Webcam index (default 0)")
    p.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    p.add_argument("--yolo", default="yolo11n.engine", help="Ultralytics YOLO model for person detection")
    p.add_argument("--yolo-conf", type=float, default=0.5, help="YOLO confidence threshold")
    p.add_argument(
        "--yolo-iou",
        type=float,
        default=0.01,
        help="YOLO NMS IoU threshold (lower = fewer duplicate boxes)",
    )
    p.add_argument(
        "--max-people",
        type=int,
        default=5,
        help="Max number of people to classify per frame (sorted by bbox area).",
    )
    p.add_argument(
        "--every",
        type=int,
        default=1,
        help="Run prediction every N frames (increase to speed up).",
    )
    p.add_argument(
        "--drop",
        type=int,
        default=2,
        help="Drop N queued camera frames each loop to reduce latency (default: 2).",
    )
    p.add_argument(
        "--inertia",
        type=int,
        default=3,
        help="Change a label only after N consecutive predictions (default: 3). Use 1 to disable.",
    )
    p.add_argument(
        "--ema-alpha",
        type=float,
        default=0.4,
        help="EMA smoothing factor for softmax probabilities (0=full history, 1=no smoothing). Default: 0.4.",
    )
    p.add_argument(
        "--phone-gate",
        type=float,
        default=0,
        help="When YOLO detects NO phone near a person, multiply phone/play_phone probs by this factor "
             "(0.0=fully suppress, 1.0=disable gate). Default: 0.3.",
    )
    p.add_argument(
        "--phone-sticky",
        type=float,
        default=0.9,
        help="Keep phone-detected flag True for this many seconds after last detection (default: 0.5).",
    )
    p.add_argument(
        "--pose-model",
        default="lite",
        choices=["lite", "full", "heavy"],
        help="MediaPipe pose landmarker weight: lite (fastest), full, or heavy (most accurate). Default: lite.",
    )
    p.add_argument(
        "--clip-onnx",
        nargs="?",
        const=DEFAULT_CLIP_ONNX,
        default=None,
        help="Path to CLIP visual ONNX model. If flag given without path, uses default. "
             "Enables ONNX Runtime with TensorRT FP16 > CUDA > CPU for CLIP encoding.",
    )
    p.add_argument("--window", default="Live Prediction", help="OpenCV window name")
    return p.parse_args()


def _resize_list(lst: list, n: int, fill):
    if len(lst) >= n:
        return lst[:n]
    return lst + [fill] * (n - len(lst))


def _resize_histories(histories: list[deque], n: int, maxlen: int) -> list[deque]:
    if len(histories) > n:
        return histories[:n]
    while len(histories) < n:
        histories.append(deque(maxlen=maxlen))
    return histories


def _make_input_from_history(hist: deque[torch.Tensor], num_frames: int, feat_agg: str) -> torch.Tensor | None:
    if len(hist) < num_frames:
        return None
    last = list(hist)[-num_frames:]
    if feat_agg == "mean":
        return torch.stack(last, dim=0).mean(dim=0)
    return torch.cat(last, dim=0)


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


def _color_for_label_name(label_name: str) -> tuple[int, int, int]:
    # OpenCV uses BGR
    alert = {"stop", "phone", "play_phone"}
    return (0, 0, 255) if label_name.strip().lower() in alert else (0, 255, 0)


MIN_DISPLAY_CONF = 0.5

TARGET_W = 640
TARGET_H = 480


class LatestFrameReader:
    def __init__(self, cap, target_w: int, target_h: int, cv2_module):
        self._cap = cap
        self._target_w = int(target_w)
        self._target_h = int(target_h)
        self._cv2 = cv2_module
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._frame = None
        self._frame_id = 0
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._cam_frame_count = 0
        self._cam_t0 = time.perf_counter()
        self._cam_fps = 0.0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        self._thread.join(timeout=1.0)

    @property
    def cam_fps(self) -> float:
        with self._lock:
            return self._cam_fps

    def wait_for_new(self, last_id: int, timeout: float = 0.1):
        with self._cond:
            while not self._stop:
                if self._frame is not None and self._frame_id != last_id:
                    return self._frame, self._frame_id
                self._cond.wait(timeout=timeout)
            return None, last_id

    def _run(self) -> None:
        while not self._stop:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue

            h0, w0 = frame.shape[:2]
            if (w0, h0) != (self._target_w, self._target_h):
                frame = self._cv2.resize(
                    frame,
                    (self._target_w, self._target_h),
                    interpolation=self._cv2.INTER_AREA,
                )

            now = time.perf_counter()
            with self._cond:
                self._frame = frame
                self._frame_id += 1
                self._cam_frame_count += 1
                dt = now - self._cam_t0
                if dt >= 1.0:
                    self._cam_fps = self._cam_frame_count / dt
                    self._cam_frame_count = 0
                    self._cam_t0 = now
                self._cond.notify_all()


def get_person_boxes(
    frame_bgr, yolo_model, yolo_conf: float, yolo_iou: float
) -> list[tuple[int, int, int, int]]:
    """Return a list of person bboxes (x1,y1,x2,y2) sorted by area desc."""
    results = yolo_model.predict(
        frame_bgr,
        conf=yolo_conf,
        iou=float(yolo_iou),
        classes=[0],
        verbose=False,
    )
    r = results[0]
    if getattr(r, "boxes", None) is None or len(r.boxes) == 0:
        return []

    h, w = frame_bgr.shape[:2]
    person_boxes: list[tuple[int, int, int, int]] = []
    person_areas: list[int] = []

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
        if area <= 0:
            continue
        person_boxes.append((x1, y1, x2, y2))
        person_areas.append(area)

    order = sorted(range(len(person_boxes)), key=lambda i: person_areas[i], reverse=True)
    return [person_boxes[i] for i in order]


def _hand_bboxes_in_frame(
    hand_landmarks_list: Sequence[Sequence],
    person_box: tuple[int, int, int, int],
    frame_h: int,
    frame_w: int,
    padding: float = 1.8,
) -> list[tuple[int, int, int, int]]:
    """Padded bounding boxes around each detected hand in full-frame coordinates.

    hand_landmarks_list: MediaPipe hand landmarks (normalised 0-1 to *person crop*).
    person_box: (x1, y1, x2, y2) of the person in the full frame.
    Returns a list of (x1, y1, x2, y2) boxes in full-frame pixel coords.
    """
    x1p, y1p, x2p, y2p = person_box
    crop_w = x2p - x1p
    crop_h = y2p - y1p
    bboxes: list[tuple[int, int, int, int]] = []
    for hand_lms in hand_landmarks_list:
        xs = [float(lm.x) * crop_w + x1p for lm in hand_lms]
        ys = [float(lm.y) * crop_h + y1p for lm in hand_lms]
        hx1, hx2 = min(xs), max(xs)
        hy1, hy2 = min(ys), max(ys)
        cx, cy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
        side = max(hx2 - hx1, hy2 - hy1) * padding
        side = max(side, 40.0)  # minimum crop size in pixels
        bx1 = int(max(0, cx - side / 2))
        by1 = int(max(0, cy - side / 2))
        bx2 = int(min(frame_w, cx + side / 2))
        by2 = int(min(frame_h, cy + side / 2))
        if (bx2 - bx1) > 10 and (by2 - by1) > 10:
            bboxes.append((bx1, by1, bx2, by2))
    return bboxes


def detect_phones_in_hand_crops(
    frame_bgr,
    hand_bboxes_per_person: list[list[tuple[int, int, int, int]]],
    yolo_model,
    conf: float = 0.25,
) -> list[bool]:
    """Run YOLO phone detection on hand-region crops.

    Returns a per-person bool: True if a phone was detected in any of that
    person's hand crops.
    """
    n = len(hand_bboxes_per_person)
    flags = [False] * n
    for pi, hbboxes in enumerate(hand_bboxes_per_person):
        for hx1, hy1, hx2, hy2 in hbboxes:
            crop = frame_bgr[hy1:hy2, hx1:hx2]
            if crop.size == 0:
                continue
            results = yolo_model.predict(
                crop, conf=conf, classes=[COCO_PHONE_CLASS], verbose=False,
            )
            r = results[0]
            if getattr(r, "boxes", None) is not None and len(r.boxes) > 0:
                flags[pi] = True
                break  # found phone for this person, skip remaining hands
    return flags


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

    try:
        import mediapipe as mp  # type: ignore
        from mediapipe.tasks.python import vision  # type: ignore
        from mediapipe.tasks.python.core import base_options  # type: ignore
    except Exception as exc:
        raise SystemExit("Failed to import mediapipe. Install mediapipe.") from exc

    pose_weight = str(args.pose_model).lower().strip()
    if pose_weight not in POSE_MODEL_URLS:
        pose_weight = "lite"
    pose_model_url = POSE_MODEL_URLS[pose_weight]
    pose_model_path = POSE_MODEL_PATHS[pose_weight]
    print(f"[pose] using pose_landmarker_{pose_weight}")
    ensure_model(model_url=pose_model_url, model_path=pose_model_path, label=f"Pose Landmarker ({pose_weight})")
    ensure_model(model_url=HAND_MODEL_URL, model_path=HAND_MODEL_PATH, label="Hand Landmarker")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(Path(args.ckpt), map_location="cpu")
    labels = payload["labels_in_order"]
    feature_dim = int(payload["feature_dim"])
    hidden = int(payload["hidden"])
    proj_dim = int(payload.get("proj_dim", 128))

    num_frames = int(payload.get("num_frames", 1))
    feat_agg = str(payload.get("feat_agg", "concat")).lower().strip()
    if feat_agg not in {"concat", "mean"}:
        feat_agg = "concat"

    temporal_model_type = str(payload.get("temporal_model", payload.get("model_type", "mlp"))).lower().strip()
    use_velocity = bool(payload.get("use_velocity", False))
    print(f"[ckpt] temporal_model={temporal_model_type} num_frames={num_frames} feat_agg={feat_agg} use_velocity={use_velocity} classes={len(labels)}")

    _clip_dim = int(payload.get("clip_feature_dim", 512))
    _pose_dim = int(payload.get("pose_dim", POSE_DIM))
    _hand_dim = int(payload.get("hand_dim", HAND_DIM))
    _hand_mask_dim = int(payload.get("hand_mask_dim", HAND_MASK_DIM))
    _phone_probe_dim = int(payload.get("phone_probe_dim", 0))

    if temporal_model_type == "gru":
        model = FusionTemporalGRU(
            clip_dim=_clip_dim,
            pose_dim=_pose_dim,
            hand_dim=_hand_dim,
            hand_mask_dim=_hand_mask_dim,
            phone_probe_dim=_phone_probe_dim,
            proj_dim=proj_dim,
            hidden=hidden,
            num_classes=len(labels),
            num_frames=num_frames,
            rnn_hidden=int(payload.get("rnn_hidden", 192)),
            rnn_layers=int(payload.get("rnn_layers", 1)),
            bidirectional=bool(payload.get("bidirectional", False)),
        ).to(device)
    else:
        model = FusionTemporalMLP(
            clip_dim=_clip_dim,
            pose_dim=_pose_dim,
            hand_dim=_hand_dim,
            hand_mask_dim=_hand_mask_dim,
            phone_probe_dim=_phone_probe_dim,
            proj_dim=proj_dim,
            hidden=hidden,
            num_classes=len(labels),
            num_frames=num_frames,
            feat_agg=feat_agg,
        ).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()

    clip_model_name = payload.get("clip_model", "ViT-B-32")
    clip_pretrained = payload.get("clip_pretrained", "openai")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model_name, pretrained=clip_pretrained
    )
    clip_model = clip_model.to(device).eval()

    # Encode phone probe text prompts (zero-shot phone detection)
    if _phone_probe_dim > 0:
        tokenizer = open_clip.get_tokenizer(clip_model_name)
        text_tokens = tokenizer(PHONE_PROBE_PROMPTS).to(device)
        phone_text_feats = clip_model.encode_text(text_tokens).float()
        phone_text_feats = F.normalize(phone_text_feats, dim=-1)
        print(f"[phone probe] {len(PHONE_PROBE_PROMPTS)} prompts encoded, dim={_phone_probe_dim}")
    else:
        phone_text_feats = None

    # ── CLIP ONNX Runtime acceleration ──
    clip_ort_session = None
    if args.clip_onnx is not None:
        onnx_path = str(args.clip_onnx)
        if not os.path.exists(onnx_path):
            print(f"[clip onnx] {onnx_path} not found, exporting ...")
            _export_clip_visual_onnx(clip_model_name, clip_pretrained, onnx_path, device="cpu")
        try:
            clip_ort_session = _create_ort_session(onnx_path)
            # Free PyTorch CLIP visual weights from GPU
            clip_model.visual = clip_model.visual.cpu()
            print(f"[clip onnx] using ORT for CLIP image encoding")
        except Exception as e:
            print(f"[clip onnx] ORT init failed ({e}), falling back to PyTorch")
            clip_ort_session = None

    expected_per_frame_dim = int(payload.get("base_feature_dim", 0))
    expected_total_dim = int(payload.get("feature_dim", feature_dim))

    if _clip_dim != 512:
        print(f"[warn] clip_feature_dim in ckpt is {_clip_dim} (expected 512)")

    if expected_per_frame_dim:
        implied = _clip_dim + POSE_DIM + HAND_DIM + HAND_MASK_DIM + _phone_probe_dim
        if implied != expected_per_frame_dim:
            raise SystemExit(
                f"Checkpoint base_feature_dim mismatch: ckpt={expected_per_frame_dim} implied={implied}"
            )

        if feat_agg == "concat" and expected_per_frame_dim * max(1, int(num_frames)) != expected_total_dim:
            raise SystemExit(
                f"Checkpoint feature_dim mismatch for concat: expected {expected_per_frame_dim}*{num_frames}="
                f"{expected_per_frame_dim * max(1, int(num_frames))} but ckpt feature_dim={expected_total_dim}"
            )
        if feat_agg == "mean" and expected_per_frame_dim != expected_total_dim:
            raise SystemExit(
                f"Checkpoint feature_dim mismatch for mean: expected {expected_per_frame_dim} but ckpt feature_dim={expected_total_dim}"
            )

    pose_options = vision.PoseLandmarkerOptions(
        base_options=base_options.BaseOptions(model_asset_path=pose_model_path),
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
    )

    hand_options = vision.HandLandmarkerOptions(
        base_options=base_options.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
    hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

    yolo = YOLO(args.yolo)

    pose_pool = ThreadPoolExecutor(max_workers=2)
    hand_pool = ThreadPoolExecutor(max_workers=2)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    # Try to reduce internal buffering (may be ignored depending on backend/driver)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Ask camera/driver for 640x480 (may be ignored by some webcams)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)

    reported_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    reported_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[camera] requested={TARGET_W}x{TARGET_H} reported={reported_w}x{reported_h}")

    printed_frame_shape = False

    # Start a camera thread that always keeps only the latest frame.
    reader = LatestFrameReader(cap, TARGET_W, TARGET_H, cv2)
    reader.start()
    last_seen_frame_id = -1

    stable_classes: list[int | None] = []
    stable_confs: list[float] = []
    pending_classes: list[int | None] = []
    pending_counts: list[int] = []

    feat_histories: list[deque[torch.Tensor]] = []
    # EMA-smoothed softmax probabilities per tracked person
    ema_probs: list[torch.Tensor | None] = []
    ema_alpha = max(0.0, min(1.0, float(args.ema_alpha)))
    print(f"[smoothing] ema_alpha={ema_alpha:.2f} (0=full history, 1=no smoothing)")

    phone_gate = max(0.0, min(1.0, float(args.phone_gate)))
    phone_sticky_sec = max(0.0, float(args.phone_sticky))
    # Per-person timestamp of last phone detection (for sticky flag)
    phone_last_seen: list[float] = []
    # Resolve label indices for phone gating
    phone_label_idx = labels.index("phone") if "phone" in labels else -1
    play_phone_label_idx = labels.index("play_phone") if "play_phone" in labels else -1
    print(f"[phone gate] factor={phone_gate:.2f} sticky={phone_sticky_sec:.2f}s phone_idx={phone_label_idx} play_phone_idx={play_phone_label_idx}")

    frame_i = 0
    t0 = time.time()
    prev_loop_end = time.perf_counter()
    display_fps = 0.0
    DISPLAY_EMA_ALPHA = 0.6

    # Per-stage timing (ms), updated each prediction frame
    timing_ms: dict[str, float] = {}

    try:
        while True:
            t_grab = time.perf_counter()
            frame, frame_id = reader.wait_for_new(last_seen_frame_id)
            if frame is None:
                continue
            timing_ms["grab"] = (time.perf_counter() - t_grab) * 1000.0
            last_seen_frame_id = frame_id

            if not printed_frame_shape:
                h, w = frame.shape[:2]
                print(f"[camera] first_frame_shape={w}x{h}")
                printed_frame_shape = True

            frame_i += 1

            _t = time.perf_counter()
            boxes = get_person_boxes(frame, yolo, args.yolo_conf, args.yolo_iou)
            timing_ms["yolo"] = (time.perf_counter() - _t) * 1000.0
            boxes = boxes[: max(1, int(args.max_people))]
            # Will be updated after hand landmarks when phone_gate < 1.0
            has_phone_flags = [True] * len(boxes)

            stable_classes = _resize_list(stable_classes, len(boxes), None)
            stable_confs = _resize_list(stable_confs, len(boxes), 0.0)
            pending_classes = _resize_list(pending_classes, len(boxes), None)
            pending_counts = _resize_list(pending_counts, len(boxes), 0)
            feat_histories = _resize_histories(feat_histories, len(boxes), max(1, num_frames))
            ema_probs = _resize_list(ema_probs, len(boxes), None)
            phone_last_seen = _resize_list(phone_last_seen, len(boxes), 0.0)

            do_predict = frame_i % max(1, int(args.every)) == 0
            if do_predict:
                if boxes:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    crops: list[Image.Image] = []
                    crop_rgbs: list = []
                    crop_to_box_i: list[int] = []
                    for box_i, (x1, y1, x2, y2) in enumerate(boxes):
                        crop_rgb = frame_rgb[y1:y2, x1:x2]
                        if crop_rgb.size == 0:
                            continue
                        crops.append(Image.fromarray(crop_rgb))
                        crop_rgbs.append(crop_rgb)
                        crop_to_box_i.append(box_i)

                    raw_labels: list[str | None] = [None] * len(boxes)
                    if crops:
                        image_t = torch.stack([preprocess(im) for im in crops], dim=0)
                        _t = time.perf_counter()
                        if clip_ort_session is not None:
                            # Run one image at a time — CLIP ViT internal reshapes
                            # hardcode batch=1 in the ONNX graph.
                            feats = []
                            for _i in range(image_t.shape[0]):
                                single = image_t[_i : _i + 1].numpy()
                                feats.append(clip_ort_session.run(None, {"pixel_values": single})[0])
                            frame_clip = torch.from_numpy(np.concatenate(feats, axis=0)).float().to(device)
                        else:
                            frame_clip = clip_model.encode_image(image_t.to(device)).float()
                        frame_clip = F.normalize(frame_clip, dim=-1)
                        timing_ms["clip_enc"] = (time.perf_counter() - _t) * 1000.0

                        # Phone probe: cosine similarity with text prompts
                        if phone_text_feats is not None:
                            _t = time.perf_counter()
                            clip_phone_scores = frame_clip @ phone_text_feats.T
                            timing_ms["phone_probe"] = (time.perf_counter() - _t) * 1000.0
                        else:
                            clip_phone_scores = None

                        # Update per-person feature histories using current detections.
                        nan_pose = [float("nan")] * POSE_DIM

                        # Prepare all MediaPipe images upfront
                        mp_images = [
                            mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop_rgbs[j]))
                            for j in range(len(crop_to_box_i))
                        ]

                        # Run pose + hand in parallel per person
                        def _detect_pose(idx):
                            return pose_landmarker.detect(mp_images[idx])

                        def _detect_hand(idx):
                            return hand_landmarker.detect(mp_images[idx])

                        _t_pose = time.perf_counter()
                        pose_futures = [pose_pool.submit(_detect_pose, j) for j in range(len(crop_to_box_i))]
                        pose_results = [f.result() for f in pose_futures]
                        timing_ms["pose_lm"] = (time.perf_counter() - _t_pose) * 1000.0

                        _t_hand = time.perf_counter()
                        hand_futures = [hand_pool.submit(_detect_hand, j) for j in range(len(crop_to_box_i))]
                        hand_results = [f.result() for f in hand_futures]
                        timing_ms["hand_lm"] = (time.perf_counter() - _t_hand) * 1000.0

                        # ── Phone detection on hand crops (before combine so it feeds into features) ──
                        if phone_gate < 1.0:
                            fh, fw = frame.shape[:2]
                            hand_bboxes_per_person: list[list[tuple[int, int, int, int]]] = [
                                [] for _ in range(len(boxes))
                            ]
                            for j, box_i in enumerate(crop_to_box_i):
                                hr = hand_results[j]
                                if hr.hand_landmarks:
                                    hbbs = _hand_bboxes_in_frame(
                                        hr.hand_landmarks, boxes[box_i], fh, fw,
                                    )
                                    hand_bboxes_per_person[box_i] = hbbs
                            _t_phone = time.perf_counter()
                            has_phone_flags = detect_phones_in_hand_crops(
                                frame, hand_bboxes_per_person, yolo,
                            )
                            timing_ms["phone_det"] = (time.perf_counter() - _t_phone) * 1000.0

                            # Sticky phone flag: keep True for phone_sticky_sec after last detection
                            now_ts = time.perf_counter()
                            for _pi in range(len(has_phone_flags)):
                                if has_phone_flags[_pi]:
                                    phone_last_seen[_pi] = now_ts
                                elif (now_ts - phone_last_seen[_pi]) < phone_sticky_sec:
                                    has_phone_flags[_pi] = True  # still within sticky window

                        # Build full phone_scores: append YOLO detection flag if model expects it
                        _needs_yolo_dim = _phone_probe_dim > len(PHONE_PROBE_PROMPTS)
                        if clip_phone_scores is not None and _needs_yolo_dim:
                            phone_det_col = torch.zeros(clip_phone_scores.shape[0], 1, device=clip_phone_scores.device)
                            if phone_gate < 1.0:
                                for j_idx, bx_i in enumerate(crop_to_box_i):
                                    if bx_i < len(has_phone_flags) and has_phone_flags[bx_i]:
                                        phone_det_col[j_idx, 0] = 1.0
                            phone_scores = torch.cat([clip_phone_scores, phone_det_col], dim=1)
                        else:
                            phone_scores = clip_phone_scores

                        for j, box_i in enumerate(crop_to_box_i):
                            pose_result = pose_results[j]
                            hand_result = hand_results[j]

                            if pose_result.pose_landmarks:
                                pose_vec = pose_vector_165(pose_result.pose_landmarks[0])
                            else:
                                pose_vec = nan_pose

                            if hand_result.hand_landmarks:
                                hand_vec = hand_vector_two_hands(
                                    hand_result.hand_landmarks,
                                    hand_result.handedness,
                                )
                            else:
                                hand_vec = hand_vector_two_hands([])

                            combined = combine_frame_features(
                                frame_clip[j].detach().cpu(),
                                pose_vec,
                                hand_vec,
                                phone_probe=phone_scores[j].detach().cpu() if phone_scores is not None else None,
                            )
                            if expected_per_frame_dim and combined.numel() != expected_per_frame_dim:
                                raise SystemExit(
                                    f"per-frame feature dim mismatch: got {combined.numel()} expected {expected_per_frame_dim}"
                                )

                            feat_histories[int(box_i)].append(combined)

                        raw_classes: list[int | None] = [None] * len(boxes)
                        raw_confs: list[float] = [0.0] * len(boxes)

                        # Build a batch for boxes that have enough history.
                        xs: list[torch.Tensor] = []
                        xs_box_i: list[int] = []
                        for box_i in range(len(boxes)):
                            x = _make_input_from_history(feat_histories[box_i], max(1, num_frames), feat_agg)
                            if x is None:
                                continue
                            xs.append(x)
                            xs_box_i.append(box_i)

                        if xs:
                            xb = torch.stack(xs, dim=0).to(device)
                            _t = time.perf_counter()
                            logits = model(xb)
                            timing_ms["model"] = (time.perf_counter() - _t) * 1000.0
                            probs = torch.softmax(logits, dim=1)

                            # EMA temporal smoothing per person
                            for idx_in_batch, box_i in enumerate(xs_box_i):
                                raw_p = probs[idx_in_batch].detach().cpu()
                                prev = ema_probs[box_i]
                                if prev is None:
                                    ema_probs[box_i] = raw_p
                                else:
                                    ema_probs[box_i] = ema_alpha * raw_p + (1.0 - ema_alpha) * prev

                            # Derive predictions from smoothed probabilities
                            for box_i in xs_box_i:
                                sp = ema_probs[box_i]
                                if sp is not None:
                                    # Phone gate: suppress phone/play_phone when no phone detected
                                    if phone_gate < 1.0 and box_i < len(has_phone_flags) and not has_phone_flags[box_i]:
                                        sp = sp.clone()
                                        if phone_label_idx >= 0:
                                            sp[phone_label_idx] *= phone_gate
                                        if play_phone_label_idx >= 0:
                                            sp[play_phone_label_idx] *= phone_gate
                                        denom = sp.sum()
                                        if denom > 0:
                                            sp = sp / denom
                                    raw_classes[box_i] = int(sp.argmax().item())
                                    raw_confs[box_i] = float(sp.max().item())

                        stable_classes, pending_classes, pending_counts = _apply_inertia(
                            raw_classes,
                            stable_classes,
                            pending_classes,
                            pending_counts,
                            args.inertia,
                        )

                        for i in range(len(boxes)):
                            if stable_classes[i] is not None and stable_classes[i] == raw_classes[i]:
                                stable_confs[i] = raw_confs[i]
                    else:
                        raw_classes = [None] * len(boxes)
                        stable_classes, pending_classes, pending_counts = _apply_inertia(
                            raw_classes,
                            stable_classes,
                            pending_classes,
                            pending_counts,
                            args.inertia,
                        )
                else:
                    stable_classes = []
                    stable_confs = []
                    pending_classes = []
                    pending_counts = []
                    feat_histories = []
                    ema_probs = []

            # draw bboxes + labels
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                cls = stable_classes[i] if i < len(stable_classes) else None
                conf = stable_confs[i] if i < len(stable_confs) else 0.0

                if cls is None:
                    # Not enough history yet (e.g., need 8 frames) or no stable prediction.
                    have = len(feat_histories[i]) if i < len(feat_histories) else 0
                    need = max(1, int(num_frames))
                    text = f"warming up {have}/{need}"
                    color = (0, 255, 0)
                else:
                    label_name = str(labels[int(cls)])
                    if conf < MIN_DISPLAY_CONF:
                        color = (160, 160, 160)
                    else:
                        color = _color_for_label_name(label_name)
                    text = f"{int(cls) + 1} {label_name} {conf * 100.0:.0f}%"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                if text:
                    cv2.putText(
                        frame,
                        text,
                        (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

            # fps
            dt = time.time() - t0
            avg_fps = frame_i / dt if dt > 0 else 0.0
            cam_fps = reader.cam_fps

            # HUD (left column): small yellow text
            hud_color = (0, 255, 255)  # BGR yellow
            hud_scale = 0.6
            hud_thick = 2
            x0 = 10
            y0 = 25
            dy = 20

            hud_lines = [
                f"people: {len(boxes)}",
                f"phones: {sum(1 for f in has_phone_flags if f)} det" if phone_gate < 1.0 else "",
                f"every: {max(1, int(args.every))}",
                f"inertia: {max(1, int(args.inertia))}",
                f"display: {display_fps:.1f} fps",
                f"camera: {cam_fps:.1f} fps",
                f"avg: {avg_fps:.1f} fps",
            ]
            hud_lines = [s for s in hud_lines if s]

            # Append per-stage latency lines
            if timing_ms:
                hud_lines.append("")  # blank separator
                stage_order = ["grab", "yolo", "clip_enc", "pose_lm", "hand_lm", "phone_det", "phone_probe", "model"]
                total_ms = 0.0
                for stage in stage_order:
                    if stage in timing_ms:
                        ms = timing_ms[stage]
                        total_ms += ms
                        hud_lines.append(f"{stage}: {ms:.1f}ms")
                hud_lines.append(f"total: {total_ms:.1f}ms")

            for li, s in enumerate(hud_lines):
                if not s:
                    continue
                cv2.putText(
                    frame,
                    s,
                    (x0, y0 + li * dy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    hud_scale,
                    hud_color,
                    hud_thick,
                    cv2.LINE_AA,
                )

            t_show = time.perf_counter()
            cv2.imshow(args.window, frame)
            key = cv2.pollKey() & 0xFF
            timing_ms["show"] = (time.perf_counter() - t_show) * 1000.0
            if key == ord("q") or key == 27:
                break

            now = time.perf_counter()
            loop_dt = now - prev_loop_end
            prev_loop_end = now
            inst_fps = 1.0 / loop_dt if loop_dt > 0 else 0.0
            display_fps = (
                DISPLAY_EMA_ALPHA * display_fps + (1.0 - DISPLAY_EMA_ALPHA) * inst_fps
                if display_fps > 0 else inst_fps
            )

            if frame_i % 30 == 0:
                total_ms = sum(
                    timing_ms.get(stage, 0.0)
                    for stage in ["grab", "yolo", "clip_enc", "pose_lm", "hand_lm", "phone_det", "phone_probe", "model", "show"]
                )
                print(
                    f"[perf] grab={timing_ms.get('grab', 0.0):.0f}ms  yolo={timing_ms.get('yolo', 0.0):.0f}ms  "
                    f"clip={timing_ms.get('clip_enc', 0.0):.0f}ms  pose={timing_ms.get('pose_lm', 0.0):.0f}ms  "
                    f"hand={timing_ms.get('hand_lm', 0.0):.0f}ms  phone_det={timing_ms.get('phone_det', 0.0):.0f}ms  "
                    f"probe={timing_ms.get('phone_probe', 0.0):.0f}ms  "
                    f"model={timing_ms.get('model', 0.0):.0f}ms  show={timing_ms.get('show', 0.0):.0f}ms  "
                    f"total={total_ms:.0f}ms  cam={cam_fps:.1f}fps  display={display_fps:.1f}fps"
                )
    finally:
        reader.stop()
        pose_pool.shutdown(wait=True)
        hand_pool.shutdown(wait=True)
        try:
            pose_landmarker.close()
        except Exception:
            pass
        try:
            hand_landmarker.close()
        except Exception:
            pass

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
