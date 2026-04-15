import argparse
import threading
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


MIN_DISPLAY_CONF = 0.85

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

            do_predict = frame_i % max(1, int(args.every)) == 0
            if do_predict:
                if boxes:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    crops: list[Image.Image] = []
                    crop_to_box_i: list[int] = []
                    for box_i, (x1, y1, x2, y2) in enumerate(boxes):
                        crop_rgb = frame_rgb[y1:y2, x1:x2]
                        if crop_rgb.size == 0:
                            continue
                        crops.append(Image.fromarray(crop_rgb))
                        crop_to_box_i.append(box_i)

                    raw_labels: list[str | None] = [None] * len(boxes)
                    if crops:
                        image_t = torch.stack([preprocess(im) for im in crops], dim=0).to(device)
                        feats = clip_model.encode_image(image_t).float()
                        feats = F.normalize(feats, dim=-1)
                        logits = mlp(feats)
                        probs = torch.softmax(logits, dim=1)
                        pred0s = probs.argmax(dim=1).tolist()
                        top1_confs = probs.max(dim=1).values.tolist()

                        raw_classes: list[int | None] = [None] * len(boxes)
                        raw_confs: list[float] = [0.0] * len(boxes)
                        for pred0, conf, box_i in zip(pred0s, top1_confs, crop_to_box_i, strict=False):
                            p = int(pred0)
                            raw_classes[box_i] = p
                            raw_confs[box_i] = float(conf)

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

            # draw bboxes + labels
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                cls = stable_classes[i] if i < len(stable_classes) else None
                conf = stable_confs[i] if i < len(stable_confs) else 0.0
                if cls is None:
                    text = ""
                    color = (0, 255, 0)
                else:
                    if conf < MIN_DISPLAY_CONF:
                        color = (160, 160, 160)
                        text = f"{conf * 100.0:.0f}%"
                    else:
                        label_name = str(labels[int(cls)])
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

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
