import argparse
import os
import time
import urllib.request
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options
from ultralytics import YOLO


POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
POSE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "pose_landmarker_heavy.task")

HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
HAND_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")


POSE_VECTOR_SIZE = 33 * 5  # 33 landmarks × (x,y,z,visibility,presence)
HAND_VECTOR_SIZE = 2 * 21 * 3  # 2 hands × 21 landmarks/hand × (x,y,z)


# MediaPipe Tasks package (as installed here) doesn't expose `mp.solutions.*`.
# For drawing skeletons, define the standard landmark connections explicitly.
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    # Index
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    # Middle
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    # Ring
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    # Pinky
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    # Palm
    (0, 17),
]


POSE_CONNECTIONS: List[Tuple[int, int]] = [
    # Face (lightweight outline)
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (3, 7),
    (6, 8),
    (9, 10),
    # Torso
    (11, 12),
    (11, 23),
    (12, 24),
    (23, 24),
    # Left arm
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    # Right arm
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    # Left leg
    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),
    # Right leg
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),
    # Feet / lower-body bridge
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
        raise RuntimeError(
            f"Failed to download the MediaPipe {label} model. "
            f"Download it manually from {model_url} and save it to {model_path}."
        ) from exc


def _lm_xyz(lm) -> Tuple[float, float, float]:
    return float(lm.x), float(lm.y), float(lm.z)


def pose_vector_165(pose_landmarks: Sequence) -> List[float]:
    """Return a fixed-length 165 pose vector.

    33 pose landmarks × (x,y,z,visibility,presence) = 165.
    Missing attributes are filled with NaN.
    """
    nan = float("nan")
    vec: List[float] = []

    # Expected 33 landmarks; if different, pad/truncate.
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

    if len(vec) != POSE_VECTOR_SIZE:
        raise RuntimeError(f"pose vector length mismatch: {len(vec)}")
    return vec


def hand_vector_two_hands(
    hand_landmarks_list: Sequence[Sequence],
    handedness_list: Optional[Sequence] = None,
) -> List[float]:
    """Return a fixed-length 126 hand vector.

    Always 2 hands × 21 landmarks/hand × (x,y,z) = 126.
    If a hand is missing, its slot is filled with NaN.
    Output order: [Left(63), Right(63)].
    """
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

        vec: List[float] = []
        for lm in hand_landmarks:
            x, y, z = _lm_xyz(lm)
            vec.extend([x, y, z])

        if len(vec) != one_hand_len:
            continue

        offset = 0 if side_norm == "Left" else one_hand_len
        out[offset : offset + one_hand_len] = vec
        filled[side_norm] = True

    if len(out) != HAND_VECTOR_SIZE:
        raise RuntimeError(f"hand vector length mismatch: {len(out)}")
    return out


def _draw_normalized_landmarks_points(
    frame_bgr,
    landmarks: Sequence,
    color: Tuple[int, int, int],
    radius: int = 2,
) -> None:
    h, w = frame_bgr.shape[:2]
    for lm in landmarks:
        x = int(float(getattr(lm, "x", -1.0)) * w)
        y = int(float(getattr(lm, "y", -1.0)) * h)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(frame_bgr, (x, y), int(radius), color, -1)


def _draw_normalized_landmarks_skeleton(
    frame_bgr,
    landmarks: Sequence,
    connections: Sequence[Tuple[int, int]],
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    h, w = frame_bgr.shape[:2]
    # Precompute pixel coords; invalid coords become None.
    pts: List[Optional[Tuple[int, int]]] = []
    for lm in landmarks:
        x = int(float(getattr(lm, "x", -1.0)) * w)
        y = int(float(getattr(lm, "y", -1.0)) * h)
        if 0 <= x < w and 0 <= y < h:
            pts.append((x, y))
        else:
            pts.append(None)

    for a, b in connections:
        if a < 0 or b < 0 or a >= len(pts) or b >= len(pts):
            continue
        pa = pts[a]
        pb = pts[b]
        if pa is None or pb is None:
            continue
        cv2.line(frame_bgr, pa, pb, color, int(thickness), lineType=cv2.LINE_AA)


def _draw_landmarks_points_in_roi(
    frame_bgr,
    landmarks: Sequence,
    roi_xyxy: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    radius: int = 2,
) -> None:
    x1, y1, x2, y2 = roi_xyxy
    roi_w = max(1, x2 - x1)
    roi_h = max(1, y2 - y1)
    for lm in landmarks:
        x = x1 + int(float(getattr(lm, "x", -1.0)) * roi_w)
        y = y1 + int(float(getattr(lm, "y", -1.0)) * roi_h)
        if x1 <= x < x2 and y1 <= y < y2:
            cv2.circle(frame_bgr, (x, y), int(radius), color, -1)


def _draw_landmarks_skeleton_in_roi(
    frame_bgr,
    landmarks: Sequence,
    connections: Sequence[Tuple[int, int]],
    roi_xyxy: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    x1, y1, x2, y2 = roi_xyxy
    roi_w = max(1, x2 - x1)
    roi_h = max(1, y2 - y1)

    pts: List[Optional[Tuple[int, int]]] = []
    for lm in landmarks:
        x = x1 + int(float(getattr(lm, "x", -1.0)) * roi_w)
        y = y1 + int(float(getattr(lm, "y", -1.0)) * roi_h)
        if x1 <= x < x2 and y1 <= y < y2:
            pts.append((x, y))
        else:
            pts.append(None)

    for a, b in connections:
        if a < 0 or b < 0 or a >= len(pts) or b >= len(pts):
            continue
        pa = pts[a]
        pb = pts[b]
        if pa is None or pb is None:
            continue
        cv2.line(frame_bgr, pa, pb, color, int(thickness), lineType=cv2.LINE_AA)


def _expand_and_clip_xyxy(
    xyxy: Tuple[float, float, float, float],
    pad: float,
    w: int,
    h: int,
) -> Tuple[int, int, int, int]:
    x1f, y1f, x2f, y2f = xyxy
    bw = max(1.0, x2f - x1f)
    bh = max(1.0, y2f - y1f)
    x1 = int(np.floor(x1f - pad * bw))
    y1 = int(np.floor(y1f - pad * bh))
    x2 = int(np.ceil(x2f + pad * bw))
    y2 = int(np.ceil(y2f + pad * bh))
    x1 = int(np.clip(x1, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    x2 = int(np.clip(x2, x1 + 1, w))
    y2 = int(np.clip(y2, y1 + 1, h))
    return x1, y1, x2, y2


def _probe_camera_indices(max_index: int = 10) -> List[int]:
    opened: List[int] = []
    for i in range(int(max_index)):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        ok = cap.isOpened()
        cap.release()
        if ok:
            opened.append(i)
    return opened


def _hand_side_from_handedness(handedness_list: Optional[Sequence], i: int) -> Optional[str]:
    if not handedness_list or i >= len(handedness_list) or not handedness_list[i]:
        return None
    cat0 = handedness_list[i][0]
    for attr in ("category_name", "display_name", "name"):
        v = getattr(cat0, attr, None)
        if isinstance(v, str) and v:
            s = v.strip().lower()
            if s == "left":
                return "Left"
            if s == "right":
                return "Right"
    return None


def _fmt_float(x: float) -> str:
    if np.isnan(x):
        return "nan"
    return f"{x:.6f}"


def write_vectors_txt(txt_path: Path, pose_vec: List[float], hand_vec: List[float]) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    pose_arr = np.asarray(pose_vec, dtype=np.float32)
    hand_arr = np.asarray(hand_vec, dtype=np.float32)

    if pose_arr.shape != (POSE_VECTOR_SIZE,):
        raise RuntimeError(f"pose vector shape mismatch: {pose_arr.shape}")
    if hand_arr.shape != (HAND_VECTOR_SIZE,):
        raise RuntimeError(f"hand vector shape mismatch: {hand_arr.shape}")

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("Pose vector size: 165\n")
        f.write("33 pose landmarks × (x,y,z,visibility,presence) = 165\n")
        f.write("pose: " + " ".join(_fmt_float(float(x)) for x in pose_arr.tolist()) + "\n")
        f.write("\n")
        f.write("Hand vector size: 126\n")
        f.write("always 2 hands × 21 landmarks/hand × (x,y,z) = 2 × 21 × 3 = 126\n")
        f.write("if a hand is missing, its slot is filled with NaN (still length 126).\n")
        f.write("hands: " + " ".join(_fmt_float(float(x)) for x in hand_arr.tolist()) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live pose + hand landmarking (MediaPipe Tasks)")
    p.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index (default: 0)",
    )
    p.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Requested capture width",
    )
    p.add_argument(
        "--height",
        type=int,
        default=720,
        help="Requested capture height",
    )
    p.add_argument(
        "--flip",
        action="store_true",
        help="Flip horizontally for selfie view",
    )
    p.add_argument(
        "--pose-conf",
        type=float,
        default=0.6,
        help="Minimum pose detection confidence",
    )
    p.add_argument(
        "--hand-conf",
        type=float,
        default=0.6,
        help="Minimum hand detection confidence",
    )
    p.add_argument(
        "--yolo-model",
        default="yolov8n.pt",
        help="YOLO model path/name for person detection (e.g. yolov8n.pt)",
    )
    p.add_argument(
        "--yolo-conf",
        type=float,
        default=0.35,
        help="YOLO confidence threshold for person detection",
    )
    p.add_argument(
        "--yolo-iou",
        type=float,
        default=0.5,
        help="YOLO IoU threshold for tracking",
    )
    p.add_argument(
        "--max-people",
        type=int,
        default=6,
        help="Max number of tracked people to run landmarking on per frame",
    )
    p.add_argument(
        "--crop-pad",
        type=float,
        default=0.2,
        help="Padding ratio around YOLO box before landmarking (e.g. 0.2 = 20%%)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    ensure_model(model_url=POSE_MODEL_URL, model_path=POSE_MODEL_PATH, label="Pose Landmarker (heavy)")
    ensure_model(model_url=HAND_MODEL_URL, model_path=HAND_MODEL_PATH, label="Hand Landmarker")

    pose_options = vision.PoseLandmarkerOptions(
        base_options=base_options.BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=float(args.pose_conf),
        min_pose_presence_confidence=float(args.pose_conf),
    )

    hand_options = vision.HandLandmarkerOptions(
        base_options=base_options.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=float(args.hand_conf),
    )

    # YOLO person detector + tracker
    yolo = YOLO(str(args.yolo_model))

    # On Linux, prefer V4L2 backend to avoid other camera backends misinterpreting indices.
    cap = cv2.VideoCapture(int(args.camera_index), cv2.CAP_V4L2)
    if not cap.isOpened():
        opened = _probe_camera_indices(10)
        hint = f". OpenCV can open indices: {opened}" if opened else ". No openable indices found (0..9)."
        raise RuntimeError(f"Failed to open camera index {args.camera_index}{hint}")

    if int(args.width) > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.width))
    if int(args.height) > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.height))

    body_green = (0, 255, 0)
    right_blue = (255, 0, 0)
    left_red = (0, 0, 255)

    with (
        vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker,
        vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker,
    ):
        t0 = cv2.getTickCount()
        last_fps = 0.0
        fps_frames = 0
        start_time_s = time.monotonic()
        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break

            if args.flip:
                frame_bgr = cv2.flip(frame_bgr, 1)

            # MediaPipe Tasks VIDEO mode requires a timestamp in ms.
            timestamp_ms = int((time.monotonic() - start_time_s) * 1000.0)
            annotated = frame_bgr.copy()

            # YOLO tracking (class 0 = person). We use tracking IDs to keep people stable.
            yolo_results = yolo.track(
                source=frame_bgr,
                persist=True,
                classes=[0],
                conf=float(args.yolo_conf),
                iou=float(args.yolo_iou),
                verbose=False,
            )

            if yolo_results and len(yolo_results) > 0 and getattr(yolo_results[0], "boxes", None) is not None:
                boxes = yolo_results[0].boxes

                # Sort by box area (largest first) so we spend compute on prominent people.
                xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else np.zeros((0, 4), dtype=np.float32)
                confs = boxes.conf.cpu().numpy() if getattr(boxes, "conf", None) is not None else np.zeros((len(xyxy),), dtype=np.float32)
                ids = (
                    boxes.id.cpu().numpy().astype(int)
                    if getattr(boxes, "id", None) is not None and boxes.id is not None
                    else -np.ones((len(xyxy),), dtype=int)
                )

                areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1]) if len(xyxy) else np.zeros((0,))
                order = np.argsort(-areas)
                if int(args.max_people) > 0:
                    order = order[: int(args.max_people)]

                H, W = frame_bgr.shape[:2]
                for j, idx in enumerate(order.tolist()):
                    x1f, y1f, x2f, y2f = [float(v) for v in xyxy[idx].tolist()]
                    roi = _expand_and_clip_xyxy((x1f, y1f, x2f, y2f), float(args.crop_pad), W, H)
                    x1, y1, x2, y2 = roi
                    if x2 - x1 < 5 or y2 - y1 < 5:
                        continue

                    # Draw bbox + ID
                    track_id = int(ids[idx])
                    score = float(confs[idx])
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    label = f"ID {track_id} {score:.2f}" if track_id >= 0 else f"person {score:.2f}"
                    cv2.putText(
                        annotated,
                        label,
                        (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        lineType=cv2.LINE_AA,
                    )

                    crop_bgr = frame_bgr[y1:y2, x1:x2]
                    if crop_bgr.size == 0:
                        continue

                    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                    crop_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)

                    # Keep timestamps monotonic even when calling multiple times per frame.
                    ts = int(timestamp_ms + j)
                    pose_result = pose_landmarker.detect_for_video(crop_mp, ts)
                    hand_result = hand_landmarker.detect_for_video(crop_mp, ts)

                    # Pose skeleton (green) in ROI
                    if pose_result.pose_landmarks:
                        for pose_lms in pose_result.pose_landmarks:
                            _draw_landmarks_skeleton_in_roi(
                                annotated,
                                pose_lms,
                                POSE_CONNECTIONS,
                                roi,
                                body_green,
                                thickness=2,
                            )
                            _draw_landmarks_points_in_roi(annotated, pose_lms, roi, body_green, radius=2)

                    # Hands skeletons (Right=blue, Left=red) in ROI
                    if hand_result.hand_landmarks:
                        for i, hand_lms in enumerate(hand_result.hand_landmarks):
                            side = _hand_side_from_handedness(hand_result.handedness, i)
                            if side == "Right":
                                color = right_blue
                            elif side == "Left":
                                color = left_red
                            else:
                                color = (0, 255, 255)

                            _draw_landmarks_skeleton_in_roi(
                                annotated,
                                hand_lms,
                                HAND_CONNECTIONS,
                                roi,
                                color,
                                thickness=2,
                            )
                            _draw_landmarks_points_in_roi(annotated, hand_lms, roi, color, radius=2)

            # FPS overlay
            t1 = cv2.getTickCount()
            dt = (t1 - t0) / cv2.getTickFrequency()
            if dt > 0.25:
                last_fps = fps_frames / dt
                t0 = t1
                fps_frames = 0
            fps_frames += 1

            cv2.putText(
                annotated,
                f"FPS: {last_fps:.1f}  (q/ESC to quit)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                lineType=cv2.LINE_AA,
            )

            cv2.imshow("Live Hand + Body Detector", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
