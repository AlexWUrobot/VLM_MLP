import argparse
import csv
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample frames from a video and save only the cropped closest person"
    )
    parser.add_argument("--video", required=True, help="Path to input .mp4 video")
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Ultralytics YOLO model path/name (e.g. yolov8n.pt)",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument(
        "--device",
        default=None,
        help="Device for inference (e.g. 'cpu', '0' for CUDA:0). Default: Ultralytics chooses.",
    )
    parser.add_argument(
        "--random-save",
        type=int,
        default=0,
        help="Randomly save N cropped person images from the video (0 disables).",
    )
    parser.add_argument(
        "--random-outdir",
        default="random_frames",
        help="Output directory for saved crops.",
    )
    parser.add_argument(
        "--random-prefix",
        default="come",
        help="Filename prefix. Crops are saved as <prefix>_<timestamp>ms.jpg",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible frame sampling.",
    )
    parser.add_argument(
        "--max-tries",
        type=int,
        default=0,
        help="Maximum attempts to find N frames with a person (0 = auto).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Failed to import cv2. Install OpenCV, e.g. `pip install opencv-python`.\n"
            "On Linux you may also need system GUI libs (libgl/libgtk)."
        ) from exc

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Failed to import ultralytics. Install it via `pip install ultralytics`."
        ) from exc

    video_path = Path(args.video)
    if not video_path.exists() and video_path.parent == Path("."):
        train_candidate = Path("Train") / video_path.name
        if train_candidate.exists():
            video_path = train_candidate

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(
            "Could not open video.\n"
            f"- Provided: {args.video}\n"
            f"- Resolved: {video_path}\n"
            f"- CWD: {Path.cwd()}\n"
            "Tip: pass the full/relative path (e.g. Train/<name>.mp4)."
        )

    random_save_n = max(0, int(args.random_save))
    outdir = Path(args.random_outdir)
    if random_save_n <= 0:
        raise SystemExit("Nothing to do: set --random-save N (N > 0).")

    outdir.mkdir(parents=True, exist_ok=True)
    if args.seed is not None:
        random.seed(int(args.seed))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        raise SystemExit(
            "This video reader did not report frame count; random sampling requires a known frame count."
        )

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    max_tries = int(args.max_tries) if int(args.max_tries) > 0 else min(frame_count, random_save_n * 50)

    saved_rows: list[tuple[str, int, int, int, int, int]] = []

    model = YOLO(args.model)

    predict_kwargs = {
        "conf": args.conf,
        "classes": [0],  # COCO class 0 == person
        "verbose": False,
    }
    if args.device is not None:
        predict_kwargs["device"] = args.device

    tried_indices: set[int] = set()
    saves = 0
    tries = 0
    while saves < random_save_n and tries < max_tries and len(tried_indices) < frame_count:
        tries += 1
        frame_index = random.randrange(frame_count)
        if frame_index in tried_indices:
            continue
        tried_indices.add(frame_index)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        timestamp_ms = int(round(cap.get(cv2.CAP_PROP_POS_MSEC) or 0))
        if timestamp_ms <= 0 and fps > 0:
            timestamp_ms = int(round((frame_index / fps) * 1000.0))

        results = model.predict(frame, **predict_kwargs)
        result = results[0]
        if getattr(result, "boxes", None) is None or len(result.boxes) == 0:
            continue

        best = None
        best_area = -1
        for box in result.boxes:
            if box.xyxy is None or len(box.xyxy) == 0:
                continue
            x1f, y1f, x2f, y2f = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1f), int(y1f), int(x2f), int(y2f)
            x1 = max(0, min(x1, frame.shape[1] - 1))
            x2 = max(0, min(x2, frame.shape[1]))
            y1 = max(0, min(y1, frame.shape[0] - 1))
            y2 = max(0, min(y2, frame.shape[0]))
            area = max(0, x2 - x1) * max(0, y2 - y1)
            if area > best_area:
                best_area = area
                best = (x1, y1, x2, y2)

        if best is None or best_area <= 0:
            continue

        x1, y1, x2, y2 = best
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        filename = f"{args.random_prefix}_{timestamp_ms}ms.jpg"
        outpath = outdir / filename
        cv2.imwrite(str(outpath), crop)
        saved_rows.append((filename, timestamp_ms, frame_index, x1, y1, x2, y2))
        saves += 1

    if saved_rows:
        manifest_path = outdir / f"{args.random_prefix}_timestamps.csv"
        with manifest_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "timestamp_ms", "frame_index", "x1", "y1", "x2", "y2"])
            for row in sorted(saved_rows, key=lambda r: r[1]):
                writer.writerow(list(row))
    else:
        raise SystemExit(
            "No crops were saved. Try increasing --max-tries or lowering --conf."
        )

    cap.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


if __name__ == "__main__":
    main()
