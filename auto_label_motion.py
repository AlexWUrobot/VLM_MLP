"""Auto human motion label agent.

Samples frames from a video, passes them to a generative VLM (Ollama/Qwen2.5-VL),
and lets the model freely describe and label the human motion — no predefined labels required.

Usage:
    python3 auto_label_motion.py --video short_videos/raw/come/come_0.mp4
    python3 auto_label_motion.py --video short_videos/raw/come/come_0.mp4 --num-frames 6
    python3 auto_label_motion.py --videos-glob 'short_videos/raw/*/*.mp4' --output-dir labels/
"""

import argparse
import base64
import glob
import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-label human motion in video clips using a generative VLM (Ollama).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", help="Path to a single input video file.")
    group.add_argument("--videos-glob", help="Glob pattern to process multiple videos.")

    parser.add_argument(
        "--num-frames",
        type=int,
        default=6,
        help="Number of frames to sample from the video and send to the VLM.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Output JSON path (single video mode). Defaults to <video_stem>_label.json.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory for batch mode (--videos-glob). One JSON per video.",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        help="Ollama API base URL.",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5vl",
        help="Ollama model name for vision-language inference.",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Custom prompt override. If empty, uses the built-in motion labeling prompt.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Resize frames to this max dimension before sending to the VLM.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP request timeout in seconds.",
    )
    return parser.parse_args()


def require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("Install OpenCV: pip install opencv-python") from exc
    return cv2


def sample_video_frames(video_path: Path, num_frames: int, max_size: int, cv2_mod) -> list[Any]:
    cap = cv2_mod.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2_mod.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        frame_count = 9999

    wanted = max(1, min(num_frames, frame_count))
    positions = sorted({
        min(frame_count - 1, int(round(i * (frame_count - 1) / max(1, wanted - 1))))
        for i in range(wanted)
    })

    frames = []
    for pos in positions:
        cap.set(cv2_mod.CAP_PROP_POS_FRAMES, float(pos))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        h, w = frame.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            frame = cv2_mod.resize(frame, (int(w * scale), int(h * scale)))
        frames.append(frame)

    cap.release()
    if not frames:
        raise SystemExit(f"No readable frames in: {video_path}")
    return frames


def frames_to_base64(frames: list[Any], cv2_mod) -> list[str]:
    result = []
    for frame in frames:
        _, buf = cv2_mod.imencode(".jpg", frame, [cv2_mod.IMWRITE_JPEG_QUALITY, 85])
        result.append(base64.b64encode(buf.tobytes()).decode("ascii"))
    return result


DEFAULT_PROMPT = """You are a human motion/gesture recognition expert. These images are sequential frames from a single video clip, ordered from first to last in time.

Your task: Identify what BODY MOTION or GESTURE the person is performing across the full sequence. Focus on how the body and hands MOVE between frames, not static appearance.

Common human motions include (but are NOT limited to): waving, beckoning/come here, stop gesture, walking, running, pointing, thumbs up, clapping, talking on phone, using phone, nodding, shaking head, bowing, stretching, dancing, pushing, pulling, reaching, etc.

Respond ONLY with a JSON object:
{
  "label": "<short 1-3 word motion label in lowercase>",
  "description": "<one sentence describing the motion trajectory across frames>",
  "body_parts_involved": "<which body parts are moving: hands, arms, legs, head, full body>",
  "confidence": <0.0 to 1.0>
}

Rules:
- Focus on MOVEMENT PATTERNS across the frame sequence, not static poses or objects
- The label must describe the ACTION/MOTION, not what the person is holding or wearing
- Be specific about gesture direction (e.g. "beckoning" = hand curling inward repeatedly, "waving" = hand swinging side to side)
- If the person is walking toward the camera, that counts as motion
- If no clear human motion is visible, use label "idle"
"""


def call_ollama(
    base64_images: list[str],
    prompt: str,
    model: str,
    ollama_url: str,
    timeout: int,
) -> dict[str, Any]:
    messages = [
        {
            "role": "user",
            "content": prompt,
            "images": base64_images,
        }
    ]
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
    }).encode()

    req = urllib.request.Request(
        f"{ollama_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"Cannot reach Ollama at {ollama_url}. Is it running? Error: {exc}"
        ) from exc

    content = data.get("message", {}).get("content", "")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"label": "parse_error", "description": content, "confidence": 0.0}

    return {
        "model": model,
        "provider": "ollama",
        "raw_response": content,
        "parsed": parsed,
    }


def label_single_video(
    video_path: Path,
    args: argparse.Namespace,
    cv2_mod,
) -> dict[str, Any]:
    frames = sample_video_frames(video_path, args.num_frames, args.image_size, cv2_mod)
    base64_images = frames_to_base64(frames, cv2_mod)
    prompt = args.prompt if args.prompt.strip() else DEFAULT_PROMPT

    result = call_ollama(base64_images, prompt, args.model, args.ollama_url, args.timeout)
    result["video"] = str(video_path)
    result["num_frames_sent"] = len(base64_images)
    return result


def main() -> None:
    args = parse_args()
    cv2_mod = require_cv2()

    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            raise SystemExit(f"Video not found: {video_path}")

        result = label_single_video(video_path, args, cv2_mod)

        output_path = Path(args.output_json) if args.output_json else video_path.with_suffix(".label.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        parsed = result.get("parsed", {})
        print(f"video: {video_path}")
        print(f"label: {parsed.get('label', '?')}")
        print(f"description: {parsed.get('description', '?')}")
        print(f"confidence: {parsed.get('confidence', '?')}")
        print(f"saved: {output_path}")

    else:
        pattern = args.videos_glob
        videos = sorted(glob.glob(pattern, recursive=True))
        if not videos:
            raise SystemExit(f"No videos matched: {pattern}")

        output_dir = Path(args.output_dir) if args.output_dir else Path("labels")
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing {len(videos)} videos...")
        results = []
        for i, vpath in enumerate(videos, 1):
            video_path = Path(vpath)
            try:
                result = label_single_video(video_path, args, cv2_mod)
                parsed = result.get("parsed", {})
                label = parsed.get("label", "?")
                print(f"[{i}/{len(videos)}] {video_path.name} -> {label}")
            except Exception as exc:
                result = {"video": str(video_path), "error": str(exc)}
                label = "ERROR"
                print(f"[{i}/{len(videos)}] {video_path.name} -> ERROR: {exc}")

            out_file = output_dir / f"{video_path.stem}.label.json"
            out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
            results.append(result)

        summary_path = output_dir / "_summary.json"
        summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nDone. Summary: {summary_path}")


if __name__ == "__main__":
    main()
