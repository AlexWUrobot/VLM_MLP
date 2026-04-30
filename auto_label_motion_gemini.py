"""Auto human motion label agent — Gemini 2.5 Pro.

Uploads the WHOLE video to Gemini 2.5 Pro and lets the model freely describe
and label the human motion — no predefined labels, no frame sampling needed.

Usage:
    python3 auto_label_motion_gemini.py --video short_videos/raw/come/come_0.mp4
    python3 auto_label_motion_gemini.py --videos-glob 'short_videos/raw/*/*.mp4' --output-dir labels_gemini/

Requires:
    pip install google-genai python-dotenv
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path


def load_api_key() -> str:
    """Load GEMINI_API_KEY from .env file or environment."""
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent / ".env")
    except ImportError:
        pass

    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "GEMINI_API_KEY not found. Set it in .env or as an environment variable."
        )
    return key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-label human motion in video clips using Gemini 2.5 Pro (whole video upload).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", help="Path to a single input video file.")
    group.add_argument("--videos-glob", help="Glob pattern to process multiple videos.")

    parser.add_argument(
        "--output-json",
        default="",
        help="Output JSON path (single video mode). Defaults to <video>.gemini_label.json.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory for batch mode. One JSON per video.",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-pro",
        help="Gemini model name.",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Custom prompt override. If empty, uses the built-in motion labeling prompt.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Request timeout in seconds (video upload + inference).",
    )
    return parser.parse_args()


DEFAULT_PROMPT = """\
You are a human motion/gesture recognition expert. Watch this video clip carefully.

Your task: Identify what BODY MOTION or GESTURE the person is performing throughout the video.
Focus on how the body, arms, and hands MOVE over time — not static appearance or objects.

Respond ONLY with a JSON object (no markdown, no code fences):
{
  "label": "<short 1-3 word motion label in lowercase>",
  "description": "<one sentence describing the motion in detail>",
  "body_parts_involved": "<which body parts are moving: hands, arms, legs, head, full body>",
  "confidence": <0.0 to 1.0>
}

Rules:
- Focus on MOVEMENT PATTERNS, not static poses or held objects
- The label must describe the ACTION/MOTION (e.g. "beckoning", "waving", "scratching head")
- Be specific about gesture details (e.g. "beckoning" = hand curling inward, "waving goodbye" = hand swinging side to side)
- If multiple people are visible, focus on the most prominent person
- If no clear human motion is visible, use label "idle"
"""


def create_client(api_key: str):
    """Create and return a Gemini client."""
    try:
        from google import genai
    except ImportError:
        raise SystemExit("Install the Gemini SDK: pip install google-genai")
    return genai.Client(api_key=api_key)


def upload_and_label(
    client,
    video_path: Path,
    model: str,
    prompt: str,
    timeout: int,
) -> dict:
    """Upload a video to Gemini and get the motion label."""
    from google import genai

    # Upload the video file
    mime_map = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm",
    }
    suffix = video_path.suffix.lower()
    mime = mime_map.get(suffix, "video/mp4")

    print(f"  Uploading {video_path.name} ({video_path.stat().st_size / 1024:.0f} KB)...")
    video_file = client.files.upload(
        file=video_path,
        config={"mime_type": mime},
    )

    # Wait for processing
    max_wait = timeout
    waited = 0
    while video_file.state.name == "PROCESSING" and waited < max_wait:
        time.sleep(2)
        waited += 2
        video_file = client.files.get(name=video_file.name)

    if video_file.state.name != "ACTIVE":
        raise RuntimeError(
            f"Video processing failed: state={video_file.state.name} after {waited}s"
        )

    # Generate content with the video
    response = client.models.generate_content(
        model=model,
        contents=[video_file, prompt],
    )

    raw = response.text.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        # Remove first line (```json) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"label": "parse_error", "description": raw, "confidence": 0.0}

    # Clean up the uploaded file
    try:
        client.files.delete(name=video_file.name)
    except Exception:
        pass

    return {
        "video": str(video_path),
        "model": model,
        "provider": "gemini",
        "raw_response": raw,
        "parsed": parsed,
    }


def main() -> None:
    args = parse_args()
    api_key = load_api_key()
    client = create_client(api_key)
    prompt = args.prompt.strip() if args.prompt.strip() else DEFAULT_PROMPT

    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            raise SystemExit(f"Video not found: {video_path}")

        result = upload_and_label(client, video_path, args.model, prompt, args.timeout)

        output_path = (
            Path(args.output_json)
            if args.output_json
            else video_path.with_suffix(".gemini_label.json")
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        parsed = result.get("parsed", {})
        print(f"video:       {video_path}")
        print(f"label:       {parsed.get('label', '?')}")
        print(f"description: {parsed.get('description', '?')}")
        print(f"body_parts:  {parsed.get('body_parts_involved', '?')}")
        print(f"confidence:  {parsed.get('confidence', '?')}")
        print(f"saved:       {output_path}")

    else:
        pattern = args.videos_glob
        videos = sorted(glob.glob(pattern, recursive=True))
        if not videos:
            raise SystemExit(f"No videos matched: {pattern}")

        output_dir = Path(args.output_dir) if args.output_dir else Path("labels_gemini")
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing {len(videos)} videos with Gemini 2.5 Pro...\n")
        results = []
        for i, vpath in enumerate(videos, 1):
            video_path = Path(vpath)
            try:
                result = upload_and_label(
                    client, video_path, args.model, prompt, args.timeout
                )
                parsed = result.get("parsed", {})
                label = parsed.get("label", "?")
                conf = parsed.get("confidence", "?")
                print(f"[{i}/{len(videos)}] {video_path.name:30s} -> {label} ({conf})")
            except Exception as exc:
                result = {"video": str(video_path), "error": str(exc)}
                print(f"[{i}/{len(videos)}] {video_path.name:30s} -> ERROR: {exc}")

            out_file = output_dir / f"{video_path.stem}.gemini_label.json"
            out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
            results.append(result)

        summary_path = output_dir / "_summary.json"
        summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nDone. Summary: {summary_path}")


if __name__ == "__main__":
    main()
