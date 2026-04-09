import argparse
import csv
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO person detection on an MP4 with imshow")
    parser.add_argument("--video", required=True, help="Path to input .mp4 video")
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Ultralytics YOLO model path/name (e.g. yolov8n.pt)",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="Ultralytics tracker config (e.g. bytetrack.yaml, botsort.yaml)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for inference (e.g. 'cpu', '0' for CUDA:0). Default: Ultralytics chooses.",
    )
    parser.add_argument(
        "--random-save",
        type=int,
        default=0,
        help="Randomly save N frames from the video (0 disables).",
    )
    parser.add_argument(
        "--random-outdir",
        default="random_frames",
        help="Output directory for randomly saved frames.",
    )
    parser.add_argument(
        "--random-prefix",
        default="come",
        help="Filename prefix. Images are saved as <prefix>_<timestamp>ms.jpg",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible frame sampling.",
    )
    parser.add_argument("--window", default="YOLO Person Detection", help="OpenCV window title")
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

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay_ms = int(round(1000.0 / fps)) if fps and fps > 0 else 1

    random_save_n = max(0, int(args.random_save))
    outdir = Path(args.random_outdir)
    if random_save_n > 0:
        outdir.mkdir(parents=True, exist_ok=True)
        if args.seed is not None:
            random.seed(int(args.seed))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    chosen_indices: set[int] = set()
    use_reservoir = False
    if random_save_n > 0:
        if frame_count > 0:
            k = min(random_save_n, frame_count)
            chosen_indices = set(random.sample(range(frame_count), k=k))
        else:
            use_reservoir = True

    saved_rows: list[tuple[str, int]] = []
    reservoir: list[tuple[object, int]] = []

    model = YOLO(args.model)

    closest_id: int | None = None
    middle_id: int | None = None

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        raw_frame = frame.copy() if random_save_n > 0 else frame
        timestamp_ms = int(round(cap.get(cv2.CAP_PROP_POS_MSEC) or 0))

        if random_save_n > 0:
            if use_reservoir:
                if len(reservoir) < random_save_n:
                    reservoir.append((raw_frame, timestamp_ms))
                else:
                    j = random.randint(0, frame_index)
                    if j < random_save_n:
                        reservoir[j] = (raw_frame, timestamp_ms)
            else:
                if frame_index in chosen_indices:
                    filename = f"{args.random_prefix}_{timestamp_ms}ms.jpg"
                    outpath = outdir / filename
                    cv2.imwrite(str(outpath), raw_frame)
                    saved_rows.append((filename, timestamp_ms))
                    chosen_indices.remove(frame_index)

        predict_kwargs = {
            "conf": args.conf,
            "classes": [0],  # COCO class 0 == person
            "verbose": False,
        }
        if args.device is not None:
            predict_kwargs["device"] = args.device

        results = model.track(frame, persist=True, tracker=args.tracker, **predict_kwargs)
        result = results[0]

        frame_h, frame_w = frame.shape[:2]
        frame_cx = frame_w / 2.0

        candidates: list[dict] = []
        if getattr(result, "boxes", None) is not None and len(result.boxes) > 0:
            for box in result.boxes:
                if box.xyxy is None or len(box.xyxy) == 0:
                    continue
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, xyxy)
                conf = float(box.conf[0]) if box.conf is not None else 0.0

                track_id = None
                if getattr(box, "id", None) is not None:
                    try:
                        track_id = int(box.id[0])
                    except Exception:
                        track_id = None

                area = max(0, x2 - x1) * max(0, y2 - y1)
                box_cx = (x1 + x2) / 2.0
                candidates.append(
                    {
                        "id": track_id,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "conf": conf,
                        "area": area,
                        "dist_to_center": abs(box_cx - frame_cx),
                    }
                )

        valid = [c for c in candidates if c["id"] is not None]
        ids_present = {c["id"] for c in valid}

        if valid:
            if closest_id is None or closest_id not in ids_present:
                closest_id = max(valid, key=lambda c: c["area"])["id"]
            if middle_id is None or middle_id not in ids_present:
                middle_id = min(valid, key=lambda c: c["dist_to_center"])["id"]

        for c in valid:
            x1, y1, x2, y2 = c["x1"], c["y1"], c["x2"], c["y2"]
            conf = c["conf"]
            track_id = c["id"]

            color = (0, 255, 0)
            thickness = 2
            label = f"id {track_id} person {conf:.2f}"
            if track_id == closest_id:
                color = (0, 0, 255)
                thickness = 3
                label = f"CLOSEST id {track_id} {conf:.2f}"
            elif track_id == middle_id:
                color = (255, 0, 0)
                thickness = 3
                label = f"MIDDLE id {track_id} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                frame,
                label,
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        if closest_id is not None or middle_id is not None:
            cv2.putText(
                frame,
                f"closest_id={closest_id}  middle_id={middle_id}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(args.window, frame)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord("q"):
            break

        frame_index += 1

    if random_save_n > 0 and use_reservoir:
        reservoir_sorted = sorted(reservoir, key=lambda t: t[1])
        for raw_frame, timestamp_ms in reservoir_sorted:
            filename = f"{args.random_prefix}_{timestamp_ms}ms.jpg"
            outpath = outdir / filename
            cv2.imwrite(str(outpath), raw_frame)
            saved_rows.append((filename, timestamp_ms))

    if random_save_n > 0 and saved_rows:
        manifest_path = outdir / f"{args.random_prefix}_timestamps.csv"
        with manifest_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "timestamp_ms"])
            for filename, ts_ms in sorted(saved_rows, key=lambda r: r[1]):
                writer.writerow([filename, ts_ms])

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
