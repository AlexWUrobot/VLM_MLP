import argparse
import os
import urllib.request
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options


POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
POSE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "pose_landmarker_heavy.task")

HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
HAND_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")


POSE_VECTOR_SIZE = 33 * 5  # 33 landmarks × (x,y,z,visibility,presence)
HAND_VECTOR_SIZE = 2 * 21 * 3  # 2 hands × 21 landmarks/hand × (x,y,z)


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
    p = argparse.ArgumentParser(description="Batch pose + hand landmarking for images (MediaPipe Tasks)")
    p.add_argument(
        "--in-dir",
        default="Data-proccessed",
        help="Input folder containing images (e.g. Data-proccessed)",
    )
    p.add_argument(
        "--out-dir",
        default="Data-processed-landmarker",
        help="Output folder for annotated images + .txt vectors",
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
    return p.parse_args()


def main() -> int:
    args = parse_args()

    ensure_model(model_url=POSE_MODEL_URL, model_path=POSE_MODEL_PATH, label="Pose Landmarker (heavy)")
    ensure_model(model_url=HAND_MODEL_URL, model_path=HAND_MODEL_PATH, label="Hand Landmarker")

    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        alt = Path("Data-proccessed") if str(in_dir) == "Data-processed" else Path("Data-processed")
        if alt.exists():
            in_dir = alt
        else:
            raise RuntimeError(f"Input dir not found: {args.in_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png"}
    img_paths = [p for p in sorted(in_dir.rglob("*")) if p.is_file() and p.suffix.lower() in exts]
    if not img_paths:
        raise RuntimeError(f"No images found under: {in_dir}")


    pose_options = vision.PoseLandmarkerOptions(
        base_options=base_options.BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=float(args.pose_conf),
        min_pose_presence_confidence=float(args.pose_conf),
    )

    hand_options = vision.HandLandmarkerOptions(
        base_options=base_options.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=float(args.hand_conf),
    )

    with (
        vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker,
        vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker,
    ):
        for p in img_paths:
            rel = p.relative_to(in_dir)
            out_img_path = out_dir / rel
            out_txt_path = out_img_path.with_suffix(".txt")

            frame_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if frame_bgr is None:
                continue

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            pose_result = pose_landmarker.detect(mp_image)
            hand_result = hand_landmarker.detect(mp_image)

            # vectors
            if pose_result.pose_landmarks:
                pose_vec = pose_vector_165(pose_result.pose_landmarks[0])
            else:
                pose_vec = [float("nan")] * POSE_VECTOR_SIZE

            if hand_result.hand_landmarks:
                hand_vec = hand_vector_two_hands(hand_result.hand_landmarks, hand_result.handedness)
            else:
                hand_vec = hand_vector_two_hands([])

            write_vectors_txt(out_txt_path, pose_vec, hand_vec)

            # draw landmarks and save annotated image
            annotated = frame_bgr.copy()

            if pose_result.pose_landmarks:
                for pose_lms in pose_result.pose_landmarks:
                    _draw_normalized_landmarks_points(annotated, pose_lms, (0, 255, 0), radius=2)

            if hand_result.hand_landmarks:
                for i, hand_lms in enumerate(hand_result.hand_landmarks):
                    side = _hand_side_from_handedness(hand_result.handedness, i)
                    if side == "Left":
                        color = (255, 0, 0)
                    elif side == "Right":
                        color = (0, 0, 255)
                    else:
                        color = (0, 255, 255)
                    _draw_normalized_landmarks_points(annotated, hand_lms, color, radius=2)

            out_img_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_img_path), annotated)

    print(f"Processed {len(img_paths)} images from {in_dir}")
    print(f"Saved annotated images + txt vectors to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
