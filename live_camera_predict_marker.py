import argparse
import math
import os
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MarkerTemporalMLP(nn.Module):
    def __init__(
        self,
        *,
        pose_dim: int,
        hand_dim: int,
        hand_mask_dim: int,
        proj_dim: int,
        hidden: int,
        num_classes: int,
        num_frames: int,
        feat_agg: str,
    ):
        super().__init__()

        self.pose_dim = int(pose_dim)
        self.hand_dim = int(hand_dim)
        self.hand_mask_dim = int(hand_mask_dim)
        self.per_frame_dim = self.pose_dim + self.hand_dim + self.hand_mask_dim

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

        self.pose_proj = _proj(self.pose_dim)
        self.hand_proj = _proj(self.hand_dim)
        self.hand_mask_proj = _proj(self.hand_mask_dim)

        self.frame_embed_dim = self.proj_dim * 3

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

        pose = xf[:, :, : self.pose_dim]
        hands = xf[:, :, self.pose_dim : self.pose_dim + self.hand_dim]
        hmask = xf[:, :, -self.hand_mask_dim :]

        b, t, _ = xf.shape
        pose_e = self.pose_proj(pose.reshape(b * t, self.pose_dim))
        hand_e = self.hand_proj(hands.reshape(b * t, self.hand_dim))
        mask_e = self.hand_mask_proj(hmask.reshape(b * t, self.hand_mask_dim))

        frame_e = torch.cat([pose_e, hand_e, mask_e], dim=1).view(b, t, self.frame_embed_dim)

        if self.feat_agg == "concat":
            h = frame_e.reshape(b, t * self.frame_embed_dim)
        else:
            h = frame_e.mean(dim=1)

        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


class MarkerTemporalRNN(nn.Module):
    def __init__(
        self,
        *,
        pose_dim: int,
        hand_dim: int,
        hand_mask_dim: int,
        proj_dim: int,
        hidden: int,
        num_classes: int,
        num_frames: int,
        rnn_type: str,
        rnn_hidden: int,
        rnn_layers: int,
        bidirectional: bool,
    ):
        super().__init__()

        self.pose_dim = int(pose_dim)
        self.hand_dim = int(hand_dim)
        self.hand_mask_dim = int(hand_mask_dim)
        self.per_frame_dim = self.pose_dim + self.hand_dim + self.hand_mask_dim
        self.num_frames = max(1, int(num_frames))
        self.proj_dim = int(proj_dim)

        self.rnn_type = str(rnn_type).lower().strip()
        if self.rnn_type not in {"gru", "lstm"}:
            raise ValueError("rnn_type must be 'gru' or 'lstm'")

        self.rnn_hidden = int(rnn_hidden)
        self.rnn_layers = max(1, int(rnn_layers))
        self.bidirectional = bool(bidirectional)

        def _proj(in_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(int(in_dim), self.proj_dim),
                nn.LayerNorm(self.proj_dim),
                nn.ReLU(),
            )

        self.pose_proj = _proj(self.pose_dim)
        self.hand_proj = _proj(self.hand_dim)
        self.hand_mask_proj = _proj(self.hand_mask_dim)
        self.frame_embed_dim = self.proj_dim * 3

        rnn_cls = nn.GRU if self.rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=self.frame_embed_dim,
            hidden_size=self.rnn_hidden,
            num_layers=self.rnn_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=0.0,
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
        pose = xf[:, :, : self.pose_dim]
        hands = xf[:, :, self.pose_dim : self.pose_dim + self.hand_dim]
        hmask = xf[:, :, -self.hand_mask_dim :]

        b, t, _ = xf.shape
        pose_e = self.pose_proj(pose.reshape(b * t, self.pose_dim))
        hand_e = self.hand_proj(hands.reshape(b * t, self.hand_dim))
        mask_e = self.hand_mask_proj(hmask.reshape(b * t, self.hand_mask_dim))
        frame_e = torch.cat([pose_e, hand_e, mask_e], dim=1).view(b, t, self.frame_embed_dim)

        _out, hidden = self.rnn(frame_e)
        if self.rnn_type == "lstm":
            hidden = hidden[0]

        if self.bidirectional:
            last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            last_hidden = hidden[-1]

        h = F.relu(self.fc1(last_hidden))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
POSE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "pose_landmarker_heavy.task")

HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
HAND_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")

POSE_DIM = 165
HAND_DIM = 126
HAND_MASK_DIM = HAND_DIM


HAND_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]


POSE_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (3, 7),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 23),
    (12, 24),
    (23, 24),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),
    (27, 28),
]


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


def _probe_camera_indices(cv2_module, max_index: int = 10) -> list[int]:
    opened: list[int] = []
    for i in range(int(max_index)):
        cap = cv2_module.VideoCapture(i, cv2_module.CAP_V4L2)
        ok = cap.isOpened()
        cap.release()
        if ok:
            opened.append(i)
    return opened


def combine_frame_features(
    pose_vec: list[float],
    hand_vec: list[float],
) -> torch.Tensor:
    # pose: impute NaN -> 0 (no mask)
    pose_t = torch.tensor(pose_vec, dtype=torch.float32)
    pose_t = torch.nan_to_num(pose_t, nan=0.0)

    # hands: impute NaN -> 0 + mask only for hands
    hands_t = torch.tensor(hand_vec, dtype=torch.float32)
    hand_mask_t = torch.isnan(hands_t).to(torch.float32)
    hands_t = torch.nan_to_num(hands_t, nan=0.0)

    # [pose(165), hands(126), hand_mask(126)]
    return torch.cat([pose_t, hands_t, hand_mask_t], dim=0)


def _build_temporal_model(payload: dict, num_classes: int) -> tuple[nn.Module, str, int, int]:
    pose_dim = int(payload.get("pose_dim", POSE_DIM))
    hand_dim = int(payload.get("hand_dim", HAND_DIM))
    hand_mask_dim = int(payload.get("hand_mask_dim", HAND_MASK_DIM))
    proj_dim = int(payload.get("proj_dim", 128))
    hidden = int(payload["hidden"])
    num_frames = int(payload.get("num_frames", 1))
    feat_agg = str(payload.get("feat_agg", "concat")).lower().strip()
    temporal_model = str(payload.get("temporal_model", "mlp")).lower().strip()
    if temporal_model not in {"mlp", "gru", "lstm"}:
        temporal_model = "mlp"

    if temporal_model == "mlp":
        model = MarkerTemporalMLP(
            pose_dim=pose_dim,
            hand_dim=hand_dim,
            hand_mask_dim=hand_mask_dim,
            proj_dim=proj_dim,
            hidden=hidden,
            num_classes=num_classes,
            num_frames=num_frames,
            feat_agg=feat_agg,
        )
        effective_feat_agg = feat_agg if feat_agg in {"concat", "mean"} else "concat"
    else:
        model = MarkerTemporalRNN(
            pose_dim=pose_dim,
            hand_dim=hand_dim,
            hand_mask_dim=hand_mask_dim,
            proj_dim=proj_dim,
            hidden=hidden,
            num_classes=num_classes,
            num_frames=num_frames,
            rnn_type=temporal_model,
            rnn_hidden=int(payload.get("rnn_hidden", 192)),
            rnn_layers=int(payload.get("rnn_layers", 1)),
            bidirectional=bool(payload.get("bidirectional_rnn", False)),
        )
        effective_feat_agg = "concat"

    return model, effective_feat_agg, num_frames, pose_dim + hand_dim + hand_mask_dim


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Webcam live action prediction (YOLO crop + pose/hands markers + temporal MLP)"
    )
    p.add_argument("--ckpt", default="mlp_3_frames_marker.pt", help="Path to trained MLP checkpoint")
    p.add_argument("--camera", type=int, default=0, help="Webcam index (default 0)")
    p.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    p.add_argument("--yolo", default="yolov8n.pt", help="Ultralytics YOLO model for person detection")
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
        "--width",
        type=int,
        default=1280,
        help="Requested capture width (default: 1280)",
    )
    p.add_argument(
        "--height",
        type=int,
        default=720,
        help="Requested capture height (default: 720)",
    )
    p.add_argument(
        "--crop-pad",
        type=float,
        default=0.2,
        help="Pad each YOLO person bbox by this fraction (default: 0.2) to help MediaPipe detect skeleton.",
    )
    p.add_argument(
        "--landmark-pad",
        type=float,
        default=None,
        help="Optional extra pad for the ROI passed to MediaPipe landmarks. Defaults to max(crop-pad, 0.35).",
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
        "--max-pose-nan-frac",
        type=float,
        default=0.5,
        help="Skip updating history for crops where pose NaN fraction is above this (default: 0.5).",
    )
    p.add_argument(
        "--max-hands-nan-frac",
        type=float,
        default=0.9,
        help="Skip updating history for crops where hands NaN fraction is above this (default: 0.9).",
    )
    p.add_argument(
        "--pose-det-conf",
        type=float,
        default=0.3,
        help="MediaPipe pose min_pose_detection_confidence (default: 0.3).",
    )
    p.add_argument(
        "--pose-pres-conf",
        type=float,
        default=0.3,
        help="MediaPipe pose min_pose_presence_confidence (default: 0.3).",
    )
    p.add_argument(
        "--hand-det-conf",
        type=float,
        default=0.3,
        help="MediaPipe hand min_hand_detection_confidence (default: 0.3).",
    )
    p.add_argument(
        "--hand-track-conf",
        type=float,
        default=0.3,
        help="MediaPipe hand min_tracking_confidence (default: 0.3).",
    )
    p.add_argument(
        "--no-draw-skeleton",
        action="store_true",
        help="Disable drawing pose/hand skeleton overlay on the OpenCV window.",
    )
    p.add_argument("--window", default="Live Prediction", help="OpenCV window name")
    return p.parse_args()


def _nan_frac(vals: list[float]) -> float:
    if not vals:
        return 1.0
    n_nan = 0
    for v in vals:
        if isinstance(v, float) and math.isnan(v):
            n_nan += 1
    return float(n_nan) / float(len(vals))


def _pad_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    w: int,
    h: int,
    pad_frac: float,
) -> tuple[int, int, int, int]:
    pad_frac = max(0.0, float(pad_frac))
    bw = max(1, int(x2 - x1))
    bh = max(1, int(y2 - y1))
    px = int(round(bw * pad_frac))
    py = int(round(bh * pad_frac))
    nx1 = max(0, min(int(x1 - px), w - 1))
    ny1 = max(0, min(int(y1 - py), h - 1))
    nx2 = max(1, min(int(x2 + px), w))
    ny2 = max(1, min(int(y2 + py), h))
    if nx2 <= nx1 + 1 or ny2 <= ny1 + 1:
        return x1, y1, x2, y2
    return nx1, ny1, nx2, ny2


def _hand_side_from_handedness(handedness_list: Optional[Sequence], i: int) -> Optional[str]:
    if not handedness_list or i >= len(handedness_list) or not handedness_list[i]:
        return None
    cat0 = handedness_list[i][0]
    for attr in ("category_name", "display_name", "name"):
        value = getattr(cat0, attr, None)
        if isinstance(value, str) and value:
            side = value.strip().lower()
            if side == "left":
                return "Left"
            if side == "right":
                return "Right"
    return None


def _draw_connections(frame_bgr, pts: list[tuple[int, int] | None], connections: Sequence[tuple[int, int]], color) -> None:
    import cv2  # type: ignore

    for a, b in connections:
        if a < len(pts) and b < len(pts) and pts[a] and pts[b]:
            cv2.line(frame_bgr, pts[a], pts[b], color, 2, lineType=cv2.LINE_AA)


def _draw_landmarks_on_frame(
    frame_bgr,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    pose_lms,
    hand_lms_list,
    handedness_list,
) -> None:
    import cv2  # type: ignore

    bw = max(1, int(x2 - x1))
    bh = max(1, int(y2 - y1))

    def _pt(lm) -> tuple[int, int] | None:
        if lm is None:
            return None
        x = getattr(lm, "x", None)
        y = getattr(lm, "y", None)
        if x is None or y is None:
            return None
        try:
            xf = float(x)
            yf = float(y)
        except Exception:
            return None
        # Clamp to frame bounds instead of rejecting — limbs often extend beyond crop
        fh, fw = frame_bgr.shape[:2]
        px = max(0, min(x1 + int(round(xf * bw)), fw - 1))
        py = max(0, min(y1 + int(round(yf * bh)), fh - 1))
        return px, py

    if pose_lms is not None:
        pts = [_pt(lm) for lm in pose_lms]
        _draw_connections(frame_bgr, pts, POSE_CONNECTIONS, (0, 255, 0))
        for p in pts:
            if p:
                cv2.circle(frame_bgr, p, 2, (0, 255, 0), -1)

    if hand_lms_list:
        for i, hand_lms in enumerate(hand_lms_list):
            pts = [_pt(lm) for lm in hand_lms]
            side = _hand_side_from_handedness(handedness_list, i)
            if side == "Right":
                color = (255, 0, 0)
            elif side == "Left":
                color = (0, 0, 255)
            else:
                color = (0, 255, 255)

            _draw_connections(frame_bgr, pts, HAND_CONNECTIONS, color)
            for p in pts:
                if p:
                    cv2.circle(frame_bgr, p, 2, color, -1)


def _prune_missing_keys(state: dict, active_keys: set[int]) -> None:
    for key in list(state.keys()):
        if key not in active_keys:
            del state[key]


def _make_input_from_history(hist: deque[torch.Tensor], num_frames: int, feat_agg: str) -> torch.Tensor | None:
    if len(hist) < num_frames:
        return None
    last = list(hist)[-num_frames:]
    if feat_agg == "mean":
        return torch.stack(last, dim=0).mean(dim=0)
    return torch.cat(last, dim=0)


def _apply_inertia_value(
    raw_class: int | None,
    stable_class: int | None,
    pending_class: int | None,
    pending_count: int,
    inertia: int,
) -> tuple[int | None, int | None, int]:
    inertia = max(1, int(inertia))

    if raw_class is None:
        return stable_class, None, 0

    if stable_class is None:
        return raw_class, None, 0

    if raw_class == stable_class:
        return stable_class, None, 0

    if pending_class == raw_class:
        pending_count += 1
    else:
        pending_class = raw_class
        pending_count = 1

    if pending_count >= inertia:
        return raw_class, None, 0

    return stable_class, pending_class, pending_count


def _color_for_label_name(label_name: str) -> tuple[int, int, int]:
    # OpenCV uses BGR
    alert = {"stop", "phone", "play_phone"}
    return (0, 0, 255) if label_name.strip().lower() in alert else (0, 255, 0)


MIN_DISPLAY_CONF = 0.5


class LatestFrameReader:
    def __init__(self, cap):
        self._cap = cap
        self._lock = threading.Lock()
        self._frame = None
        self._frame_id = 0
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        self._thread.join(timeout=1.0)

    def read_latest(self):
        with self._lock:
            if self._frame is None:
                return None, self._frame_id
            return self._frame.copy(), self._frame_id

    def _run(self) -> None:
        while not self._stop:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue

            with self._lock:
                self._frame = frame
                self._frame_id += 1


def get_person_boxes(
    frame_bgr, yolo_model, yolo_conf: float, yolo_iou: float
) -> list[tuple[int, int, tuple[int, int, int, int], float]]:
    """Return tracked people as (track_id, box_index, (x1,y1,x2,y2), score) sorted by area desc."""
    results = yolo_model.track(
        frame_bgr,
        persist=True,
        conf=yolo_conf,
        iou=float(yolo_iou),
        classes=[0],
        verbose=False,
    )
    if not results:
        return []

    r = results[0]
    if getattr(r, "boxes", None) is None or len(r.boxes) == 0:
        return []

    h, w = frame_bgr.shape[:2]
    boxes = r.boxes
    xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else np.zeros((0, 4), dtype=np.float32)
    confs = boxes.conf.cpu().numpy() if getattr(boxes, "conf", None) is not None else np.zeros((len(xyxy),), dtype=np.float32)
    ids = (
        boxes.id.cpu().numpy().astype(int)
        if getattr(boxes, "id", None) is not None and boxes.id is not None
        else -np.ones((len(xyxy),), dtype=int)
    )

    tracked_people: list[tuple[int, int, tuple[int, int, int, int], float]] = []
    areas: list[int] = []

    for box_i, coords in enumerate(xyxy.tolist()):
        x1f, y1f, x2f, y2f = coords
        x1, y1, x2, y2 = int(x1f), int(y1f), int(x2f), int(y2f)
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area <= 0:
            continue
        tracked_people.append((int(ids[box_i]), box_i, (x1, y1, x2, y2), float(confs[box_i])))
        areas.append(area)

    order = sorted(range(len(tracked_people)), key=lambda i: areas[i], reverse=True)
    return [tracked_people[i] for i in order]


@torch.inference_mode()
def main() -> None:
    args = parse_args()

    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise SystemExit("Failed to import cv2. Install opencv-python.") from exc

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

    ensure_model(model_url=POSE_MODEL_URL, model_path=POSE_MODEL_PATH, label="Pose Landmarker (heavy)")
    ensure_model(model_url=HAND_MODEL_URL, model_path=HAND_MODEL_PATH, label="Hand Landmarker")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(Path(args.ckpt), map_location="cpu")
    labels = payload["labels_in_order"]
    model, feat_agg, num_frames, implied_per_frame_dim = _build_temporal_model(payload, len(labels))
    model = model.to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()

    expected_per_frame_dim = int(payload.get("base_feature_dim", 0))
    expected_total_dim = int(payload.get("feature_dim", 0))
    if expected_per_frame_dim and not expected_total_dim:
        expected_total_dim = (
            expected_per_frame_dim * max(1, int(num_frames))
            if feat_agg == "concat"
            else expected_per_frame_dim
        )

    if expected_per_frame_dim:
        implied = implied_per_frame_dim
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
        base_options=base_options.BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=float(args.pose_det_conf),
        min_pose_presence_confidence=float(args.pose_pres_conf),
    )

    hand_options = vision.HandLandmarkerOptions(
        base_options=base_options.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=float(args.hand_det_conf),
        min_tracking_confidence=float(args.hand_track_conf),
    )

    pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
    hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

    yolo = YOLO(args.yolo)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        opened = _probe_camera_indices(cv2, 10)
        hint = f". OpenCV can open indices: {opened}" if opened else ". No openable indices found (0..9)."
        raise SystemExit(f"Could not open camera index {args.camera}{hint}")

    # Try to reduce internal buffering (may be ignored depending on backend/driver)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if int(args.width) > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.width))
    if int(args.height) > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.height))

    reported_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    reported_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[camera] requested={int(args.width)}x{int(args.height)} reported={reported_w}x{reported_h}")

    printed_frame_shape = False

    # Start a camera thread that always keeps only the latest frame.
    reader = LatestFrameReader(cap)
    reader.start()
    last_seen_frame_id = -1

    stable_classes: dict[int, int | None] = {}
    stable_confs: dict[int, float] = {}
    pending_classes: dict[int, int | None] = {}
    pending_counts: dict[int, int] = {}

    feat_histories: dict[int, deque[torch.Tensor]] = {}

    frame_i = 0
    t0 = time.time()
    start_time_s = time.monotonic()

    last_pose_ok = 0
    last_hands_ok = 0
    last_pose_det = 0
    last_hand_det = 0
    last_crops = 0

    try:
        while True:
            frame, frame_id = reader.read_latest()
            if frame is None or frame_id == last_seen_frame_id:
                time.sleep(0.001)
                continue
            last_seen_frame_id = frame_id

            if not printed_frame_shape:
                h, w = frame.shape[:2]
                print(f"[camera] first_frame_shape={w}x{h}")
                printed_frame_shape = True

            frame_i += 1

            tracked_people = get_person_boxes(frame, yolo, args.yolo_conf, args.yolo_iou)
            tracked_people = tracked_people[: max(1, int(args.max_people))]
            person_keys: list[int] = []
            boxes: list[tuple[int, int, int, int]] = []
            track_ids: list[int] = []

            for order_i, (track_id, _box_i, box, _score) in enumerate(tracked_people):
                person_key = int(track_id) if int(track_id) >= 0 else -(order_i + 1)
                person_keys.append(person_key)
                track_ids.append(int(track_id))
                boxes.append(box)
                if person_key not in feat_histories:
                    feat_histories[person_key] = deque(maxlen=max(1, num_frames))
                stable_classes.setdefault(person_key, None)
                stable_confs.setdefault(person_key, 0.0)
                pending_classes.setdefault(person_key, None)
                pending_counts.setdefault(person_key, 0)

            active_keys = set(person_keys)
            _prune_missing_keys(stable_classes, active_keys)
            _prune_missing_keys(stable_confs, active_keys)
            _prune_missing_keys(pending_classes, active_keys)
            _prune_missing_keys(pending_counts, active_keys)
            _prune_missing_keys(feat_histories, active_keys)

            do_predict = frame_i % max(1, int(args.every)) == 0
            if do_predict:
                if boxes:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    crop_rgbs: list = []
                    crop_to_person_key: list[int] = []
                    crop_boxes: list[tuple[int, int, int, int]] = []
                    hF, wF = frame_rgb.shape[:2]
                    landmark_pad = float(args.landmark_pad) if args.landmark_pad is not None else max(float(args.crop_pad), 0.5)
                    for box_i, (x1, y1, x2, y2) in enumerate(boxes):
                        x1, y1, x2, y2 = _pad_box(x1, y1, x2, y2, wF, hF, landmark_pad)
                        crop_rgb = np.ascontiguousarray(frame_rgb[y1:y2, x1:x2])
                        if crop_rgb.size == 0:
                            continue
                        crop_rgbs.append(crop_rgb)
                        crop_to_person_key.append(person_keys[box_i])
                        crop_boxes.append((x1, y1, x2, y2))

                    last_pose_ok = 0
                    last_hands_ok = 0
                    last_pose_det = 0
                    last_hand_det = 0
                    last_crops = len(crop_rgbs)

                    if crop_rgbs:
                        # Update per-person feature histories using current detections.
                        nan_pose = [float("nan")] * POSE_DIM
                        timestamp_ms = int((time.monotonic() - start_time_s) * 1000.0)
                        for j, person_key in enumerate(crop_to_person_key):
                            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgbs[j])
                            ts = int(timestamp_ms + j)

                            pose_result = pose_landmarker.detect_for_video(mp_image, ts)
                            hand_result = hand_landmarker.detect_for_video(mp_image, ts)

                            if pose_result.pose_landmarks:
                                pose_vec = pose_vector_165(pose_result.pose_landmarks[0])
                                last_pose_det += 1
                            else:
                                pose_vec = nan_pose

                            if hand_result.hand_landmarks:
                                last_hand_det += 1
                                hand_vec = hand_vector_two_hands(
                                    hand_result.hand_landmarks,
                                    hand_result.handedness,
                                )
                            else:
                                hand_vec = hand_vector_two_hands([])

                            # Draw skeleton overlay for debugging.
                            if not args.no_draw_skeleton:
                                x1, y1, x2, y2 = crop_boxes[j]
                                pose_lms = None
                                if pose_result.pose_landmarks:
                                    pose_lms = pose_result.pose_landmarks[0]
                                hand_lms_list = []
                                if hand_result.hand_landmarks:
                                    hand_lms_list = list(hand_result.hand_landmarks)
                                _draw_landmarks_on_frame(
                                    frame,
                                    x1,
                                    y1,
                                    x2,
                                    y2,
                                    pose_lms=pose_lms,
                                    hand_lms_list=hand_lms_list,
                                    handedness_list=hand_result.handedness,
                                )

                            pose_nan_frac = _nan_frac(pose_vec)
                            hands_nan_frac = _nan_frac(hand_vec)
                            pose_ok = pose_nan_frac <= float(args.max_pose_nan_frac)
                            hands_ok = hands_nan_frac <= float(args.max_hands_nan_frac)

                            if pose_ok:
                                last_pose_ok += 1
                            if hands_ok:
                                last_hands_ok += 1

                            # If BOTH pose and hands are too incomplete, skip this update.
                            # Do NOT require both to be present: hands are often missing and the hand_mask
                            # is explicitly designed to handle that.
                            if not (pose_ok or hands_ok):
                                continue

                            combined = combine_frame_features(
                                pose_vec,
                                hand_vec,
                            )
                            if expected_per_frame_dim and combined.numel() != expected_per_frame_dim:
                                raise SystemExit(
                                    f"per-frame feature dim mismatch: got {combined.numel()} expected {expected_per_frame_dim}"
                                )

                            feat_histories[person_key].append(combined)

                        raw_classes: dict[int, int | None] = {person_key: None for person_key in person_keys}
                        raw_confs: dict[int, float] = {person_key: 0.0 for person_key in person_keys}

                        # Build a batch for boxes that have enough history.
                        xs: list[torch.Tensor] = []
                        xs_person_keys: list[int] = []
                        for person_key in person_keys:
                            x = _make_input_from_history(feat_histories[person_key], max(1, num_frames), feat_agg)
                            if x is None:
                                continue
                            xs.append(x)
                            xs_person_keys.append(person_key)

                        if xs:
                            xb = torch.stack(xs, dim=0).to(device)
                            logits = model(xb)
                            probs = torch.softmax(logits, dim=1)
                            pred0s = probs.argmax(dim=1).tolist()
                            top1_confs = probs.max(dim=1).values.tolist()

                            for pred0, conf, person_key in zip(pred0s, top1_confs, xs_person_keys, strict=False):
                                raw_classes[person_key] = int(pred0)
                                raw_confs[person_key] = float(conf)

                        for person_key in person_keys:
                            stable_class, pending_class, pending_count = _apply_inertia_value(
                                raw_classes[person_key],
                                stable_classes.get(person_key),
                                pending_classes.get(person_key),
                                pending_counts.get(person_key, 0),
                                args.inertia,
                            )
                            stable_classes[person_key] = stable_class
                            pending_classes[person_key] = pending_class
                            pending_counts[person_key] = pending_count

                            if stable_class is not None and stable_class == raw_classes[person_key]:
                                stable_confs[person_key] = raw_confs[person_key]
                    else:
                        for person_key in person_keys:
                            stable_class, pending_class, pending_count = _apply_inertia_value(
                                None,
                                stable_classes.get(person_key),
                                pending_classes.get(person_key),
                                pending_counts.get(person_key, 0),
                                args.inertia,
                            )
                            stable_classes[person_key] = stable_class
                            pending_classes[person_key] = pending_class
                            pending_counts[person_key] = pending_count
                else:
                    stable_classes = {}
                    stable_confs = {}
                    pending_classes = {}
                    pending_counts = {}
                    feat_histories = {}

            # draw bboxes + labels
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                person_key = person_keys[i]
                track_id = track_ids[i]
                cls = stable_classes.get(person_key)
                conf = stable_confs.get(person_key, 0.0)

                if cls is None:
                    # Not enough history yet (need num_frames frames) or no stable prediction.
                    have = len(feat_histories.get(person_key, ()))
                    need = max(1, int(num_frames))
                    prefix = f"ID {track_id} " if track_id >= 0 else ""
                    text = f"{prefix}warming up {have}/{need}"
                    color = (0, 255, 0)
                else:
                    label_name = str(labels[int(cls)])
                    if conf < MIN_DISPLAY_CONF:
                        color = (160, 160, 160)
                    else:
                        color = _color_for_label_name(label_name)
                    prefix = f"ID {track_id} " if track_id >= 0 else ""
                    text = f"{prefix}{int(cls) + 1} {label_name} {conf * 100.0:.0f}%"

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
            fps = frame_i / dt if dt > 0 else 0.0

            # HUD (left column): small yellow text
            hud_color = (0, 255, 255)  # BGR yellow
            hud_scale = 0.6
            hud_thick = 2
            x0 = 10
            y0 = 25
            dy = 20

            hud_lines = [
                f"people: {len(boxes)}",
                f"pose_det: {last_pose_det}/{last_crops}",
                f"hand_det: {last_hand_det}/{last_crops}",
                f"pose_ok: {last_pose_ok}/{last_crops}",
                f"hands_ok: {last_hands_ok}/{last_crops}",
                f"every: {max(1, int(args.every))}",
                f"inertia: {max(1, int(args.inertia))}",
                f"fps: {fps:.1f}",
            ]

            for li, s in enumerate(hud_lines):
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

            cv2.imshow(args.window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        reader.stop()
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
