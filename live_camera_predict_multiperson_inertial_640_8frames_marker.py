import argparse
import os
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import Optional, Sequence, Tuple

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
        self.per_frame_dim = self.clip_dim + self.pose_dim + self.hand_dim + self.hand_mask_dim

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

        self.frame_embed_dim = self.proj_dim * 4

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

        clip = xf[:, :, : self.clip_dim]
        pose = xf[:, :, self.clip_dim : self.clip_dim + self.pose_dim]
        hands = xf[:, :, self.clip_dim + self.pose_dim : self.clip_dim + self.pose_dim + self.hand_dim]
        hmask = xf[:, :, -self.hand_mask_dim :]

        b, t, _ = xf.shape
        clip_e = self.clip_proj(clip.reshape(b * t, self.clip_dim))
        pose_e = self.pose_proj(pose.reshape(b * t, self.pose_dim))
        hand_e = self.hand_proj(hands.reshape(b * t, self.hand_dim))
        mask_e = self.hand_mask_proj(hmask.reshape(b * t, self.hand_mask_dim))

        frame_e = torch.cat([clip_e, pose_e, hand_e, mask_e], dim=1).view(b, t, self.frame_embed_dim)

        if self.feat_agg == "concat":
            h = frame_e.reshape(b, t * self.frame_embed_dim)
        else:
            h = frame_e.mean(dim=1)

        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
POSE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "pose_landmarker_heavy.task")

HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
HAND_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")

POSE_DIM = 165
HAND_DIM = 126
HAND_MASK_DIM = HAND_DIM


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
) -> torch.Tensor:
    # pose: impute NaN -> 0 (no mask)
    pose_t = torch.tensor(pose_vec, dtype=torch.float32)
    pose_t = torch.nan_to_num(pose_t, nan=0.0)

    # hands: impute NaN -> 0 + mask only for hands
    hands_t = torch.tensor(hand_vec, dtype=torch.float32)
    hand_mask_t = torch.isnan(hands_t).to(torch.float32)
    hands_t = torch.nan_to_num(hands_t, nan=0.0)

    clip_t = clip_feat.to(torch.float32)

    # [clip(512), pose(165), hands(126), hand_mask(126)]
    return torch.cat([clip_t, pose_t, hands_t, hand_mask_t], dim=0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Webcam live action prediction (YOLO crop + CLIP-B + pose/hands + temporal MLP)"
    )
    p.add_argument("--ckpt", default="mlp_clip_b_8frames_marker.pt", help="Path to trained MLP checkpoint")
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

            h0, w0 = frame.shape[:2]
            if (w0, h0) != (self._target_w, self._target_h):
                frame = self._cv2.resize(
                    frame,
                    (self._target_w, self._target_h),
                    interpolation=self._cv2.INTER_AREA,
                )

            with self._lock:
                self._frame = frame
                self._frame_id += 1


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
    boxes: list[tuple[int, int, int, int]] = []
    areas: list[int] = []

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
        boxes.append((x1, y1, x2, y2))
        areas.append(area)

    order = sorted(range(len(boxes)), key=lambda i: areas[i], reverse=True)
    return [boxes[i] for i in order]


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

    ensure_model(model_url=POSE_MODEL_URL, model_path=POSE_MODEL_PATH, label="Pose Landmarker (heavy)")
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

    mlp = FusionTemporalMLP(
        clip_dim=int(payload.get("clip_feature_dim", 512)),
        pose_dim=int(payload.get("pose_dim", POSE_DIM)),
        hand_dim=int(payload.get("hand_dim", HAND_DIM)),
        hand_mask_dim=int(payload.get("hand_mask_dim", HAND_MASK_DIM)),
        proj_dim=proj_dim,
        hidden=hidden,
        num_classes=len(labels),
        num_frames=int(payload.get("num_frames", 8)),
        feat_agg=str(payload.get("feat_agg", "concat")),
    ).to(device)
    mlp.load_state_dict(payload["state_dict"], strict=True)
    mlp.eval()

    clip_model_name = payload.get("clip_model", "ViT-B-32")
    clip_pretrained = payload.get("clip_pretrained", "openai")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model_name, pretrained=clip_pretrained
    )
    clip_model = clip_model.to(device).eval()

    expected_per_frame_dim = int(payload.get("base_feature_dim", 0))
    expected_total_dim = int(payload.get("feature_dim", feature_dim))
    clip_dim_expected = int(payload.get("clip_feature_dim", 512))


    if clip_dim_expected != 512:
        print(f"[warn] clip_feature_dim in ckpt is {clip_dim_expected} (expected 512)")

    if expected_per_frame_dim:
        implied = clip_dim_expected + POSE_DIM + HAND_DIM + HAND_MASK_DIM
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

    frame_i = 0
    t0 = time.time()

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

            boxes = get_person_boxes(frame, yolo, args.yolo_conf, args.yolo_iou)
            boxes = boxes[: max(1, int(args.max_people))]

            stable_classes = _resize_list(stable_classes, len(boxes), None)
            stable_confs = _resize_list(stable_confs, len(boxes), 0.0)
            pending_classes = _resize_list(pending_classes, len(boxes), None)
            pending_counts = _resize_list(pending_counts, len(boxes), 0)
            feat_histories = _resize_histories(feat_histories, len(boxes), max(1, num_frames))

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
                        image_t = torch.stack([preprocess(im) for im in crops], dim=0).to(device)
                        frame_clip = clip_model.encode_image(image_t).float()
                        frame_clip = F.normalize(frame_clip, dim=-1)

                        # Update per-person feature histories using current detections.
                        nan_pose = [float("nan")] * POSE_DIM
                        for j, box_i in enumerate(crop_to_box_i):
                            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgbs[j])

                            pose_result = pose_landmarker.detect(mp_image)
                            hand_result = hand_landmarker.detect(mp_image)

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
                            logits = mlp(xb)
                            probs = torch.softmax(logits, dim=1)
                            pred0s = probs.argmax(dim=1).tolist()
                            top1_confs = probs.max(dim=1).values.tolist()

                            for pred0, conf, box_i in zip(pred0s, top1_confs, xs_box_i, strict=False):
                                raw_classes[int(box_i)] = int(pred0)
                                raw_confs[int(box_i)] = float(conf)

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
            fps = frame_i / dt if dt > 0 else 0.0

            cv2.putText(
                frame,
                f"people: {len(boxes)}  every: {max(1, int(args.every))}  inertia: {max(1, int(args.inertia))}",
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
