#!/usr/bin/env python3
"""
RTMW-l: Real-Time Whole-Body 133-Keypoint Pose Detector (TensorRT FP16)

Detects 133 keypoints per person:
    body (17) + feet (6) + face (68) + left hand (21) + right hand (21)

Requirements:
    pip install onnxruntime-gpu ultralytics opencv-python numpy

Usage:
    python3 RTMW_detector.py --camera 0
    python3 RTMW_detector.py --camera 4 --yolo yolo11m.pt
    python3 RTMW_detector.py --camera 4 --yolo yolo11m.pt --no-trt
"""

import argparse
import os
import threading
import time
import urllib.request
import zipfile

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Model paths & constants
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
RTMW_L_ONNX = os.path.join(MODEL_DIR, "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx")
RTMW_L_ZIP_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmw/"
    "onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip"
)

# NCHW = (1, 3, 384, 288)  — H=384, W=288
MODEL_INPUT_H = 384
MODEL_INPUT_W = 288
SIMCC_SPLIT_RATIO = 2.0

# ImageNet normalisation
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

NUM_KEYPOINTS = 133

# ---------------------------------------------------------------------------
# Skeleton definitions (global 0-132 keypoint indices)
# ---------------------------------------------------------------------------
BODY_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

FEET_SKELETON = [
    (15, 17), (15, 18), (15, 19), (17, 18),
    (16, 20), (16, 21), (16, 22), (20, 21),
]

# Face: 68 points at indices 23-90
_face_chain = lambda s, n: [(s + i, s + i + 1) for i in range(n - 1)]
_face_loop = lambda s, n: [(s + i, s + (i + 1) % n) for i in range(n)]

FACE_SKELETON: list[tuple[int, int]] = (
    _face_chain(23, 17)     # jaw contour
    + _face_chain(40, 5)    # right eyebrow
    + _face_chain(45, 5)    # left eyebrow
    + _face_chain(50, 4)    # nose bridge
    + _face_chain(54, 5)    # nose bottom
    + _face_loop(59, 6)     # right eye
    + _face_loop(65, 6)     # left eye
    + _face_loop(71, 12)    # outer lip
    + _face_loop(83, 8)     # inner lip
)

_HAND_EDGES_REL = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16), # ring
    (13, 17), (17, 18), (18, 19), (19, 20),# pinky
    (0, 17),                                # palm
]

LHAND_SKELETON = [(91 + a, 91 + b) for a, b in _HAND_EDGES_REL]
RHAND_SKELETON = [(112 + a, 112 + b) for a, b in _HAND_EDGES_REL]

# BGR colours
BODY_COLOR = (0, 255, 0)
FEET_COLOR = (255, 255, 0)
FACE_COLOR = (200, 200, 200)
LHAND_COLOR = (255, 128, 0)
RHAND_COLOR = (0, 128, 255)

SKELETON_GROUPS: list[tuple[list[tuple[int, int]], tuple[int, int, int]]] = [
    (BODY_SKELETON, BODY_COLOR),
    (FEET_SKELETON, FEET_COLOR),
    (FACE_SKELETON, FACE_COLOR),
    (LHAND_SKELETON, LHAND_COLOR),
    (RHAND_SKELETON, RHAND_COLOR),
]


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------
def ensure_rtmw_model() -> str:
    """Return local path to the RTMW-l ONNX model, downloading if needed."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.isfile(RTMW_L_ONNX):
        return RTMW_L_ONNX

    # Check for any rtmw .onnx already in MODEL_DIR
    for fname in sorted(os.listdir(MODEL_DIR)):
        if fname.lower().startswith("rtmw") and fname.endswith(".onnx"):
            return os.path.join(MODEL_DIR, fname)

    # Try downloading zip
    zip_path = RTMW_L_ONNX + ".zip"
    print(f"[RTMW] Downloading model from\n  {RTMW_L_ZIP_URL} ...")
    try:
        urllib.request.urlretrieve(RTMW_L_ZIP_URL, zip_path)
    except Exception as exc:
        if os.path.exists(zip_path):
            os.remove(zip_path)
        raise SystemExit(
            "[RTMW] Auto-download failed.\n\n"
            "  Get the model manually:\n"
            "  1) pip install rtmlib\n"
            '     python -c "from rtmlib import Wholebody; Wholebody(mode=\'performance\')"\n'
            f"     Then copy the rtmw*.onnx from ~/.cache/rtmlib/ to {MODEL_DIR}/\n"
            "  2) Or download from https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose\n"
            f"     and place the .onnx inside {MODEL_DIR}/\n"
        ) from exc

    # Extract .onnx from the zip
    print("[RTMW] Extracting .onnx from zip ...")
    onnx_path = None
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith(".onnx"):
                dest = os.path.join(MODEL_DIR, os.path.basename(name))
                with zf.open(name) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
                onnx_path = dest
                break
    os.remove(zip_path)

    if onnx_path is None:
        raise SystemExit(
            "[RTMW] Downloaded zip contains no .onnx file.  Please download manually."
        )

    print(f"[RTMW] Model ready: {onnx_path}")
    return onnx_path


# ---------------------------------------------------------------------------
# ONNX Runtime session (TensorRT FP16 > CUDA > CPU)
# ---------------------------------------------------------------------------
def create_session(model_path: str, *, use_trt: bool = True, force_cpu: bool = False):
    import onnxruntime as ort

    available = ort.get_available_providers()
    providers: list = []

    if not force_cpu and use_trt and "TensorrtExecutionProvider" in available:
        cache_dir = os.path.join(MODEL_DIR, "trt_cache")
        os.makedirs(cache_dir, exist_ok=True)
        providers.append((
            "TensorrtExecutionProvider",
            {
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": cache_dir,
                "trt_max_workspace_size": 2 * 1024 * 1024 * 1024,  # 2 GB
            },
        ))

    if not force_cpu and "CUDAExecutionProvider" in available:
        providers.append(("CUDAExecutionProvider", {"device_id": 0}))

    providers.append(("CPUExecutionProvider", {}))

    sess = ort.InferenceSession(model_path, providers=providers)
    active = sess.get_providers()
    print(f"[RTMW] Active providers: {active}")
    if "TensorrtExecutionProvider" in active:
        print("[RTMW] TensorRT FP16 active — first run may take a few minutes to build the engine.")
    return sess


def is_cuda_toolchain_error(exc: Exception) -> bool:
    msg = str(exc)
    return (
        "cudaErrorUnsupportedPtxVersion" in msg
        or "unsupported toolchain" in msg
        or "CUDA error" in msg
    )


# ---------------------------------------------------------------------------
# Preprocessing: bbox -> affine warp -> normalised tensor
# ---------------------------------------------------------------------------
def _bbox_center_scale(
    x1: float, y1: float, x2: float, y2: float, padding: float = 1.25,
) -> tuple[np.ndarray, np.ndarray]:
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = max(1.0, x2 - x1) * padding
    h = max(1.0, y2 - y1) * padding
    aspect = MODEL_INPUT_W / MODEL_INPUT_H  # 288/384 = 0.75
    if w / h > aspect:
        h = w / aspect
    else:
        w = h * aspect
    return np.array([cx, cy], dtype=np.float32), np.array([w, h], dtype=np.float32)


def _get_warp(center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    src = np.float32([
        [center[0] - scale[0] * 0.5, center[1] - scale[1] * 0.5],
        [center[0] + scale[0] * 0.5, center[1] - scale[1] * 0.5],
        [center[0] - scale[0] * 0.5, center[1] + scale[1] * 0.5],
    ])
    dst = np.float32([
        [0, 0],
        [MODEL_INPUT_W, 0],
        [0, MODEL_INPUT_H],
    ])
    return cv2.getAffineTransform(src, dst)


def preprocess(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    padding: float = 1.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (input_tensor NCHW float32, warp_mat 2x3)."""
    x1, y1, x2, y2 = bbox
    center, scale = _bbox_center_scale(x1, y1, x2, y2, padding)
    warp_mat = _get_warp(center, scale)

    crop = cv2.warpAffine(
        frame_bgr, warp_mat, (MODEL_INPUT_W, MODEL_INPUT_H),
        flags=cv2.INTER_LINEAR,
    )

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = (rgb - MEAN) / STD
    nchw = np.transpose(rgb, (2, 0, 1))[np.newaxis]   # (1, 3, H, W)
    return nchw, warp_mat


# ---------------------------------------------------------------------------
# Postprocessing: SimCC decode -> original-image keypoints
# ---------------------------------------------------------------------------
def postprocess(
    outputs: list[np.ndarray],
    warp_mat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (keypoints (133,2), scores (133,))."""
    if len(outputs) >= 2:
        # Two outputs: determine x vs y by last-dim size (x < y)
        o0, o1 = outputs[0], outputs[1]
        if o0.shape[-1] <= o1.shape[-1]:
            simcc_x, simcc_y = o0, o1
        else:
            simcc_x, simcc_y = o1, o0
    else:
        # Single concatenated output — split at W*split
        x_bins = int(MODEL_INPUT_W * SIMCC_SPLIT_RATIO)
        simcc_x = outputs[0][..., :x_bins]
        simcc_y = outputs[0][..., x_bins:]

    # Remove batch dim
    if simcc_x.ndim == 3:
        simcc_x = simcc_x[0]
        simcc_y = simcc_y[0]

    x_locs = np.argmax(simcc_x, axis=-1).astype(np.float32) / SIMCC_SPLIT_RATIO
    y_locs = np.argmax(simcc_y, axis=-1).astype(np.float32) / SIMCC_SPLIT_RATIO

    x_vals = np.max(simcc_x, axis=-1)
    y_vals = np.max(simcc_y, axis=-1)
    scores = np.minimum(x_vals, y_vals)

    # Map back to original image via inverse affine
    kpts_model = np.stack([x_locs, y_locs], axis=-1)  # (133, 2)
    inv_warp = cv2.invertAffineTransform(warp_mat)
    ones = np.ones((NUM_KEYPOINTS, 1), dtype=np.float32)
    kpts_orig = np.hstack([kpts_model, ones]) @ inv_warp.T  # (133, 2)

    return kpts_orig, scores


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_wholebody(
    frame_bgr: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    *,
    kpt_thr: float = 0.3,
    draw_face: bool = True,
) -> None:
    """Draw 133-keypoint skeleton on *frame_bgr* in-place."""
    h, w = frame_bgr.shape[:2]

    for connections, color in SKELETON_GROUPS:
        if not draw_face and color is FACE_COLOR:
            continue
        for a, b in connections:
            if scores[a] < kpt_thr or scores[b] < kpt_thr:
                continue
            pa = (int(keypoints[a, 0]), int(keypoints[a, 1]))
            pb = (int(keypoints[b, 0]), int(keypoints[b, 1]))
            cv2.line(frame_bgr, pa, pb, color, 2, cv2.LINE_AA)

    for i in range(NUM_KEYPOINTS):
        if scores[i] < kpt_thr:
            continue
        if not draw_face and 23 <= i <= 90:
            continue
        px = int(keypoints[i, 0])
        py = int(keypoints[i, 1])
        # colour by body part
        if i <= 16:
            c = BODY_COLOR
        elif i <= 22:
            c = FEET_COLOR
        elif i <= 90:
            c = FACE_COLOR
        elif i <= 111:
            c = LHAND_COLOR
        else:
            c = RHAND_COLOR
        radius = 1 if 23 <= i <= 90 else 3
        cv2.circle(frame_bgr, (px, py), radius, c, -1)


# ---------------------------------------------------------------------------
# Camera reader (low-latency: always keeps only the latest frame)
# ---------------------------------------------------------------------------
class LatestFrameReader:
    def __init__(self, cap):
        self._cap = cap
        self._lock = threading.Lock()
        self._frame = None
        self._frame_id = 0
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop = True
        self._thread.join(timeout=1.0)

    def read_latest(self):
        with self._lock:
            if self._frame is None:
                return None, self._frame_id
            return self._frame.copy(), self._frame_id

    def _run(self):
        while not self._stop:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame = frame
                self._frame_id += 1


# ---------------------------------------------------------------------------
# YOLO person detection (reuses user's ultralytics setup)
# ---------------------------------------------------------------------------
def get_person_boxes(frame_bgr, yolo_model, yolo_conf, yolo_iou, max_people):
    results = yolo_model.track(
        frame_bgr, persist=True, conf=yolo_conf, iou=yolo_iou,
        classes=[0], verbose=False,
    )
    if not results:
        return []
    r = results[0]
    if getattr(r, "boxes", None) is None or len(r.boxes) == 0:
        return []

    h, w = frame_bgr.shape[:2]
    boxes = r.boxes
    xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else np.zeros((0, 4))
    confs = (
        boxes.conf.cpu().numpy()
        if getattr(boxes, "conf", None) is not None
        else np.zeros(len(xyxy))
    )
    ids = (
        boxes.id.cpu().numpy().astype(int)
        if getattr(boxes, "id", None) is not None and boxes.id is not None
        else -np.ones(len(xyxy), dtype=int)
    )

    people: list[tuple[int, tuple[int, int, int, int], float]] = []
    for i, (x1f, y1f, x2f, y2f) in enumerate(xyxy.tolist()):
        x1, y1 = max(0, int(x1f)), max(0, int(y1f))
        x2, y2 = min(w, int(x2f)), min(h, int(y2f))
        if (x2 - x1) * (y2 - y1) <= 0:
            continue
        people.append((int(ids[i]), (x1, y1, x2, y2), float(confs[i])))

    people.sort(key=lambda p: (p[1][2] - p[1][0]) * (p[1][3] - p[1][1]), reverse=True)
    return people[:max_people]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="RTMW-l 133-keypoint whole-body pose detector (TensorRT FP16)"
    )
    p.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    p.add_argument("--yolo", default="yolo11m.pt", help="YOLO model for person detection")
    p.add_argument("--yolo-conf", type=float, default=0.5, help="YOLO confidence threshold")
    p.add_argument("--yolo-iou", type=float, default=0.4, help="YOLO NMS IoU threshold")
    p.add_argument("--max-people", type=int, default=5, help="Max people per frame")
    p.add_argument("--width", type=int, default=1280, help="Capture width")
    p.add_argument("--height", type=int, default=720, help="Capture height")
    p.add_argument("--kpt-thr", type=float, default=0.3, help="Keypoint confidence for drawing")
    p.add_argument("--bbox-pad", type=float, default=1.25, help="Bbox padding ratio for RTMW crop")
    p.add_argument("--no-trt", action="store_true", help="Disable TensorRT, fall back to CUDA EP")
    p.add_argument("--cpu", action="store_true", help="Force CPU inference and skip CUDA/TensorRT")
    p.add_argument("--no-face", action="store_true", help="Skip drawing face keypoints")
    p.add_argument("--model", default=None, help="Path to RTMW ONNX model (auto-download if omitted)")
    p.add_argument("--window", default="RTMW Detector", help="OpenCV window name")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    from ultralytics import YOLO  # type: ignore

    # ---- Load RTMW-l ONNX model ----
    model_path = args.model if args.model else ensure_rtmw_model()
    sess = create_session(model_path, use_trt=not args.no_trt, force_cpu=args.cpu)
    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]
    input_shape = sess.get_inputs()[0].shape
    print(f"[RTMW] input: {input_name}  shape={input_shape}")
    print(f"[RTMW] outputs: {output_names}")
    gpu_fallback_used = args.cpu

    # ---- Load YOLO ----
    yolo = YOLO(args.yolo)

    # ---- Open camera ----
    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    rw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[camera] {rw}x{rh}")

    reader = LatestFrameReader(cap)
    reader.start()

    last_id = -1
    frame_count = 0
    t0 = time.time()

    try:
        while True:
            frame, fid = reader.read_latest()
            if frame is None or fid == last_id:
                time.sleep(0.001)
                continue
            last_id = fid
            frame_count += 1

            # ---- Detect people ----
            people = get_person_boxes(
                frame, yolo, args.yolo_conf, args.yolo_iou, args.max_people
            )

            # ---- Per-person RTMW inference ----
            for track_id, bbox, det_conf in people:
                inp, warp_mat = preprocess(frame, bbox, args.bbox_pad)
                try:
                    outs = sess.run(output_names, {input_name: inp})
                except Exception as exc:
                    if gpu_fallback_used or not is_cuda_toolchain_error(exc):
                        raise
                    print(
                        "[RTMW] CUDA/TensorRT inference failed; switching to CPU. "
                        "Install a stable ONNX Runtime build matched to your NVIDIA driver "
                        "to restore GPU inference."
                    )
                    print(f"[RTMW] original error: {exc}")
                    sess = create_session(model_path, use_trt=False, force_cpu=True)
                    input_name = sess.get_inputs()[0].name
                    output_names = [o.name for o in sess.get_outputs()]
                    gpu_fallback_used = True
                    outs = sess.run(output_names, {input_name: inp})
                kpts, scores = postprocess(outs, warp_mat)

                draw_wholebody(
                    frame, kpts, scores,
                    kpt_thr=args.kpt_thr,
                    draw_face=not args.no_face,
                )

                x1, y1, x2, y2 = bbox
                prefix = f"ID {track_id} " if track_id >= 0 else ""
                label = f"{prefix}{det_conf:.0%}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
                )

            # ---- HUD ----
            dt = time.time() - t0
            fps = frame_count / dt if dt > 0 else 0
            cv2.putText(
                frame,
                f"fps: {fps:.1f}  people: {len(people)}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
            )

            cv2.imshow(args.window, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        reader.stop()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
