import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record camera video to mixed_action_videos and save on Ctrl-C.",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--width", type=int, default=640, help="Capture width.")
    parser.add_argument("--height", type=int, default=480, help="Capture height.")
    parser.add_argument(
        "--output-dir",
        default="mixed_action_videos",
        help="Directory where MP4 files are saved.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="Output video FPS. Use 0 to auto-detect and fall back to 30.",
    )
    return parser.parse_args()


def build_output_path(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"record_{timestamp}.mp4"


def resolve_video_fps(cap: cv2.VideoCapture, requested_fps: float) -> float:
    if requested_fps > 0:
        return requested_fps

    detected_fps = cap.get(cv2.CAP_PROP_FPS)
    if detected_fps and 1.0 <= detected_fps <= 120.0:
        return detected_fps
    return 30.0


def main() -> None:
    args = parse_args()
    output_path = build_output_path(Path(args.output_dir))

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera index {args.camera}")

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise SystemExit("Cannot read first frame from camera")

    frame_height, frame_width = frame.shape[:2]
    video_fps = resolve_video_fps(cap, args.fps)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_fps,
        (frame_width, frame_height),
    )

    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"Cannot open video writer for {output_path}")

    prev_time = time.time()
    fps = 0.0
    alpha = 0.9
    frame_count = 0

    print(f"Recording to: {output_path}")
    print("Press Ctrl-C in the terminal to stop and save. Press q in the window to stop too.")

    try:
        while True:
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                instant_fps = 1.0 / dt
                fps = alpha * fps + (1 - alpha) * instant_fps if fps > 0 else instant_fps

            writer.write(frame)
            frame_count += 1

            display = frame.copy()
            cv2.putText(
                display,
                f"FPS: {fps:.1f}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Camera Recorder", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Stopping on 'q' key and saving video...")
                break

            ret, frame = cap.read()
            if not ret:
                print("Camera frame read failed. Saving video collected so far...")
                break
    except KeyboardInterrupt:
        print("\nCtrl-C received. Saving video...")
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"Saved video: {output_path}")
    print(f"Frames written: {frame_count}")
    print(f"Preview FPS: {fps:.1f}")


if __name__ == "__main__":
    main()