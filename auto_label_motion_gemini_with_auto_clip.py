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

import cv2

LABEL_DEFS_FILE = Path(__file__).resolve().parent / "label_definitions_gemini.json"


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
        description="Auto-segment human motion in video clips using Gemini with timestamped motions.",
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
    parser.add_argument(
        "--label-defs",
        default=str(LABEL_DEFS_FILE),
        help="Path to label_definitions_gemini.json (label memory file).",
    )
    parser.add_argument(
        "--clips-dir",
        default="",
        help="Directory where clipped MP4 segments are saved. Defaults to the source video's folder.",
    )
    return parser.parse_args()


DEFAULT_PROMPT_TEMPLATE = """\
You are a human motion/gesture recognition expert. Watch this video clip carefully.

Your task: segment the full video timeline into motion clips.
Focus on how the body, arms, and hands MOVE over time — not static appearance or objects.

{known_labels_section}

Respond ONLY with a JSON object (no markdown, no code fences):
{{
    "video_summary": "<short summary of the whole video>",
    "distinct_motion_count": <number of different motions in the video>,
    "known_motion_labels_present": ["<existing label>", "..."],
    "new_motion_labels_present": ["<new label>", "..."],
    "segments": [
        {{
            "start_sec": <segment start time in seconds>,
            "end_sec": <segment end time in seconds>,
            "label": "<short motion label in lowercase>",
            "description": "<one sentence describing this segment>",
            "body_parts_involved": "<hands, arms, legs, head, full body>",
            "is_new_motion": <true if this motion does NOT match any known label above, false if it matches>,
            "confidence": <0.0 to 1.0>
        }}
    ]
}}

Rules:
- FIRST check if each motion matches one of the KNOWN LABELS above. If it does, you MUST use that exact label.
- Only create a new label if the motion is clearly different from all known labels.
- Merge adjacent time spans if the motion label is the same.
- Cover the whole meaningful motion timeline with timestamps in seconds.
- Use non-overlapping segments ordered by time.
- Focus on MOVEMENT PATTERNS, not static poses or held objects.
- The label must describe the ACTION/MOTION (e.g. "beckoning", "waving", "scratching head").
- Be specific about gesture details.
- If multiple people are visible, focus on the most prominent person.
- If no clear human motion is visible, use label "idle".
"""


def load_label_defs(path: Path) -> dict:
    """Load known label definitions from JSON file."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_label_defs(path: Path, defs: dict) -> None:
    """Save label definitions back to JSON file."""
    path.write_text(json.dumps(defs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_prompt(label_defs: dict, custom_prompt: str) -> str:
    """Build the prompt with known labels injected."""
    if custom_prompt:
        return custom_prompt

    if label_defs:
        lines = ["KNOWN LABELS (use these if the motion matches):"]
        for label, desc in label_defs.items():
            lines.append(f'  - "{label}": {desc}')
        known_section = "\n".join(lines)
    else:
        known_section = "No known labels yet. You will create the first one."

    return DEFAULT_PROMPT_TEMPLATE.format(known_labels_section=known_section)


def _as_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_segments(parsed: dict) -> list[dict]:
    """Normalize Gemini's segment list into a stable clip manifest."""
    raw_segments = parsed.get("segments")
    if not isinstance(raw_segments, list):
        return []

    normalized = []
    for index, segment in enumerate(raw_segments, 1):
        if not isinstance(segment, dict):
            continue

        label = str(segment.get("label", "")).strip().lower()
        if not label:
            continue

        start_sec = max(0.0, _as_float(segment.get("start_sec"), 0.0))
        end_sec = max(start_sec, _as_float(segment.get("end_sec"), start_sec))
        normalized.append(
            {
                "segment_index": index,
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "label": label,
                "description": str(segment.get("description", "")).strip(),
                "body_parts_involved": str(segment.get("body_parts_involved", "")).strip(),
                "is_new_motion": bool(segment.get("is_new_motion", False)),
                "confidence": _as_float(segment.get("confidence"), 0.0),
            }
        )

    normalized.sort(key=lambda item: (item["start_sec"], item["end_sec"], item["segment_index"]))
    return normalized


def build_clip_manifest(video_path: Path, segments: list[dict]) -> list[dict]:
    """Build downstream-friendly clip entries with filenames and durations."""
    stem = video_path.stem
    manifest = []
    for segment in segments:
        start_sec = segment["start_sec"]
        end_sec = segment["end_sec"]
        label = segment["label"]
        manifest.append(
            {
                **segment,
                "duration_sec": round(max(0.0, end_sec - start_sec), 3),
                "start_ms": int(round(start_sec * 1000)),
                "end_ms": int(round(end_sec * 1000)),
                "suggested_clip_name": (
                    f"{stem}_from_{int(round(start_sec * 1000)):04d}_to_{int(round(end_sec * 1000)):04d}_label_{sanitize_label(label)}.mp4"
                ),
            }
        )
    return manifest


def sanitize_label(label: str) -> str:
    """Convert labels into safe filename fragments."""
    cleaned = []
    for char in label.lower().strip():
        if char.isalnum():
            cleaned.append(char)
        elif char in {" ", "-", "_"}:
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "unknown"


def resolve_clips_dir(args: argparse.Namespace, video_path: Path) -> Path:
    """Pick the directory where clip MP4 files are written."""
    if args.clips_dir:
        return Path(args.clips_dir)
    return video_path.parent


def write_video_clips(video_path: Path, clip_manifest: list[dict], clips_dir: Path) -> list[str]:
    """Cut clip MP4 files from the source video according to the manifest."""
    if not clip_manifest:
        return []

    clips_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source video for clipping: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Cannot read source video dimensions: {video_path}")

    written_files = []
    try:
        for clip in clip_manifest:
            start_frame = max(0, int(round(clip["start_sec"] * fps)))
            end_frame = int(round(clip["end_sec"] * fps))
            if frame_count > 0:
                start_frame = min(start_frame, max(0, frame_count - 1))
                end_frame = min(end_frame, frame_count)
            if end_frame <= start_frame:
                end_frame = start_frame + 1

            output_path = clips_dir / clip["suggested_clip_name"]
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Cannot open clip writer for {output_path}")

            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
                current_frame = start_frame
                while current_frame < end_frame:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                    writer.write(frame)
                    current_frame += 1
            finally:
                writer.release()

            clip["output_path"] = str(output_path)
            written_files.append(str(output_path))
    finally:
        cap.release()

    return written_files


def update_label_defs(label_defs: dict, parsed: dict, defs_path: Path) -> int:
    """Add newly discovered motions from parsed segments. Returns number of new labels."""
    new_labels = 0
    for segment in normalize_segments(parsed):
        label = segment["label"]
        if not label or label == "parse_error" or label in label_defs:
            continue
        if not segment.get("is_new_motion", False):
            continue

        description = segment.get("description", "") or "new motion discovered by Gemini"
        label_defs[label] = description
        new_labels += 1

    if new_labels:
        save_label_defs(defs_path, label_defs)
    return new_labels


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

    # Generate content with the video (with retry for 503/429 overload)
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[video_file, prompt],
            )
            break
        except Exception as exc:
            err = str(exc)
            if "503" in err or "UNAVAILABLE" in err or "overloaded" in err.lower():
                wait = 10 * (attempt + 1)
                print(f"  Server busy, retrying in {wait}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
                if attempt == max_retries - 1:
                    raise
            elif "429" in err or "RESOURCE_EXHAUSTED" in err:
                # Extract retry delay if available
                import re
                m = re.search(r'retry in ([\d.]+)s', err, re.IGNORECASE)
                wait = int(float(m.group(1))) + 5 if m else 60
                print(f"  Rate limited, waiting {wait}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
                if attempt == max_retries - 1:
                    raise
            else:
                raise

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


def finalize_result(result: dict, video_path: Path) -> dict:
    """Attach normalized segments and clip manifest for downstream clipping."""
    parsed = result.get("parsed", {})
    segments = normalize_segments(parsed)
    clip_manifest = build_clip_manifest(video_path, segments)
    known_labels = sorted({segment["label"] for segment in segments if not segment.get("is_new_motion", False)})
    new_labels = sorted({segment["label"] for segment in segments if segment.get("is_new_motion", False)})
    parsed["segments"] = segments
    parsed["distinct_motion_count"] = len({segment["label"] for segment in segments})
    parsed["known_motion_labels_present"] = known_labels
    parsed["new_motion_labels_present"] = new_labels
    parsed["known_motion_count"] = len(known_labels)
    parsed["new_motion_count"] = len(new_labels)
    result["parsed"] = parsed
    result["clip_manifest"] = clip_manifest
    return result


def main() -> None:
    args = parse_args()
    api_key = load_api_key()
    client = create_client(api_key)
    defs_path = Path(args.label_defs)
    label_defs = load_label_defs(defs_path)
    custom_prompt = args.prompt.strip()

    print(f"Label memory: {defs_path} ({len(label_defs)} known labels)")

    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            raise SystemExit(f"Video not found: {video_path}")

        prompt = build_prompt(label_defs, custom_prompt)
        result = upload_and_label(client, video_path, args.model, prompt, args.timeout)
        result = finalize_result(result, video_path)
        clips_dir = resolve_clips_dir(args, video_path)
        written_files = write_video_clips(video_path, result.get("clip_manifest", []), clips_dir)
        result["clips_dir"] = str(clips_dir)
        result["written_clips"] = written_files

        parsed = result.get("parsed", {})
        new_labels = update_label_defs(label_defs, parsed, defs_path)

        output_path = (
            Path(args.output_json)
            if args.output_json
            else video_path.with_suffix(".gemini_label.json")
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        print(f"video:       {video_path}")
        print(f"motions:     {parsed.get('distinct_motion_count', 0)}")
        print(f"known:       {parsed.get('known_motion_count', 0)} {parsed.get('known_motion_labels_present', [])}")
        print(f"new:         {parsed.get('new_motion_count', 0)} {parsed.get('new_motion_labels_present', [])}")
        for clip in result.get("clip_manifest", []):
            print(
                "clip:        "
                f"{clip['start_sec']:.3f}-{clip['end_sec']:.3f}s | {clip['label']} | {clip.get('output_path', clip['suggested_clip_name'])}"
            )
        if new_labels:
            print(f"NEW LABELS:  {new_labels} added to {defs_path} -> now {len(label_defs)} labels")
        print(f"saved:       {output_path}")

    else:
        pattern = args.videos_glob
        videos = sorted(glob.glob(pattern, recursive=True))
        if not videos:
            raise SystemExit(f"No videos matched: {pattern}")

        output_dir = Path(args.output_dir) if args.output_dir else Path("labels_gemini")
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing {len(videos)} videos...\n")
        results = []
        new_count = 0
        for i, vpath in enumerate(videos, 1):
            video_path = Path(vpath)
            try:
                # Rebuild prompt each iteration so newly discovered labels are included
                prompt = build_prompt(label_defs, custom_prompt)
                result = upload_and_label(
                    client, video_path, args.model, prompt, args.timeout
                )
                result = finalize_result(result, video_path)
                clips_dir = resolve_clips_dir(args, video_path)
                written_files = write_video_clips(video_path, result.get("clip_manifest", []), clips_dir)
                result["clips_dir"] = str(clips_dir)
                result["written_clips"] = written_files
                parsed = result.get("parsed", {})
                new_labels = update_label_defs(label_defs, parsed, defs_path)
                new_count += new_labels
                motion_count = parsed.get("distinct_motion_count", 0)
                labels = ", ".join(sorted({segment["label"] for segment in parsed.get("segments", [])}))
                tag = f" [NEW:{new_labels}]" if new_labels else ""
                print(f"[{i}/{len(videos)}] {video_path.name:30s} -> {motion_count} motions | {labels} | {len(written_files)} clips{tag}")
            except Exception as exc:
                result = {"video": str(video_path), "error": str(exc)}
                print(f"[{i}/{len(videos)}] {video_path.name:30s} -> ERROR: {exc}")

            out_file = output_dir / f"{video_path.stem}.gemini_label.json"
            out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
            results.append(result)

        summary_path = output_dir / "_summary.json"
        summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nDone. {new_count} new labels discovered. Total labels: {len(label_defs)}")
        print(f"Label memory: {defs_path}")
        print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
