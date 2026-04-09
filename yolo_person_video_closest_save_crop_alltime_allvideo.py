import argparse
import csv
import glob
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save cropped closest-person images from one or more videos at a fixed time interval"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", help="Path to a single input video")
    group.add_argument(
        "--videos-glob",
        help="Glob for multiple videos (e.g. 'Train/come_*.mp4' or 'come_*.mp4')",
    )
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
        "--interval-ms",
        type=int,
        default=250,
        help="Save one crop every N milliseconds.",
    )
    parser.add_argument(
        "--outdir",
        default="random_frames",
        help="Output directory for saved crops.",
    )
    parser.add_argument(
        "--prefix",
        default="come",
        help="(single-video only) Filename prefix. Crops are saved as <prefix>_<timestamp>ms.jpg",
    )
    parser.add_argument(
        "--start-ms",
        type=int,
        default=0,
        help="Start time in ms (default 0).",
    )
    parser.add_argument(
        "--end-ms",
        type=int,
        default=0,
        help="End time in ms (0 = until end of video).",
    )
    return parser.parse_args()


def _resolve_video_path(video_arg: str) -> Path:
    video_path = Path(video_arg)
    if not video_path.exists() and video_path.parent == Path("."):
        train_candidate = Path("Train") / video_path.name
        if train_candidate.exists():
            video_path = train_candidate
    return video_path


def _iter_video_paths(args: argparse.Namespace) -> list[Path]:
    if args.video:
        return [_resolve_video_path(args.video)]

    assert args.videos_glob
    matches = sorted(glob.glob(args.videos_glob))
    paths = [_resolve_video_path(m) for m in matches]
    return [p for p in paths if p.exists()]


def _crop_closest_person(frame, result) -> tuple[object | None, tuple[int, int, int, int] | None]:
    if getattr(result, "boxes", None) is None or len(result.boxes) == 0:
        return None, None

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
        return None, None

    x1, y1, x2, y2 = best
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None

    return crop, best


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

    video_paths = _iter_video_paths(args)
    if not video_paths:
        raise SystemExit(
            "No videos found.\n"
            f"- --video: {args.video}\n"
            f"- --videos-glob: {args.videos_glob}\n"
            f"- CWD: {Path.cwd()}"
        )

    interval_ms = int(args.interval_ms)
    if interval_ms <= 0:
        raise SystemExit("--interval-ms must be > 0")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    predict_kwargs = {
        "conf": args.conf,
        "classes": [0],  # COCO class 0 == person
        "verbose": False,
    }
    if args.device is not None:
        predict_kwargs["device"] = args.device

    all_rows: list[tuple[str, str, int, int, int, int, int, int, int]] = []

    start_ms = max(0, int(args.start_ms))
    end_ms_arg = int(args.end_ms)

    for video_path in video_paths:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        duration_ms = 0
        if frame_count > 0 and fps > 0:
            duration_ms = int(round((frame_count / fps) * 1000.0))

        end_ms = end_ms_arg
        if end_ms <= 0:
            end_ms = duration_ms
        if end_ms and end_ms < start_ms:
            cap.release()
            raise SystemExit("--end-ms must be >= --start-ms")

        video_prefix = args.prefix if args.video else video_path.stem

        target_ms = start_ms
        while True:
            if end_ms and target_ms > end_ms:
                break

            cap.set(cv2.CAP_PROP_POS_MSEC, float(target_ms))
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            actual_ms = int(round(cap.get(cv2.CAP_PROP_POS_MSEC) or target_ms))
            frame_index = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES) or -1))

            results = model.predict(frame, **predict_kwargs)
            result = results[0]
            crop, best = _crop_closest_person(frame, result)
            if crop is None or best is None:
                target_ms += interval_ms
                continue

            x1, y1, x2, y2 = best

            filename = f"{video_prefix}_{actual_ms}ms.jpg"
            outpath = outdir / filename
            if outpath.exists():
                stem = outpath.stem
                suffix = outpath.suffix
                k = 2
                while outpath.exists():
                    outpath = outdir / f"{stem}_{k}{suffix}"
                    k += 1
                filename = outpath.name

            cv2.imwrite(str(outpath), crop)
            all_rows.append((str(video_path), filename, target_ms, actual_ms, frame_index, x1, y1, x2, y2))

            target_ms += interval_ms

        cap.release()

    if not all_rows:
        raise SystemExit("No crops were saved. Try lowering --conf or checking the video content.")

    manifest_name = f"{args.prefix}_timestamps.csv" if args.video else "all_timestamps.csv"
    manifest_path = outdir / manifest_name
    with manifest_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "video",
                "filename",
                "target_timestamp_ms",
                "actual_timestamp_ms",
                "frame_index",
                "x1",
                "y1",
                "x2",
                "y2",
            ]
        )
        for row in sorted(all_rows, key=lambda r: (r[0], r[3])):
            writer.writerow(list(row))

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


if __name__ == "__main__":
    main()
