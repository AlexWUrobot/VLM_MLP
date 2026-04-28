import argparse
import base64
import collections
import json
import math
import mimetypes
import os
import statistics
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class FrameQuality:
    index: int
    timestamp_ms: int
    blur_score: float
    brightness_mean: float
    contrast_std: float
    width: int
    height: int
    path: str | None = None


@dataclass
class QualityReport:
    source_type: str
    source_path: str
    sampled_frames: int
    min_width: int
    min_height: int
    avg_blur_score: float
    min_blur_score: float
    avg_brightness_mean: float
    avg_contrast_std: float
    duration_ms: int | None
    fps: float | None
    passed: bool
    failure_reasons: list[str]
    frames: list[FrameQuality]


@dataclass
class WindowLabelResult:
    window_index: int
    start_ms: int
    end_ms: int
    sample_paths: list[str]
    labeling: dict[str, Any] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run quality checks on a video or image directory, then optionally ask a VLM to label the motion."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", help="Path to an input video file")
    group.add_argument("--images-dir", help="Path to a directory of images representing one clip")
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=8,
        help="How many frames/images to sample for quality check and AI labeling.",
    )
    parser.add_argument(
        "--output-json",
        default="quality_report.json",
        help="Where to save the quality and labeling report.",
    )
    parser.add_argument(
        "--save-samples-dir",
        default="",
        help="Optional directory to save sampled frames from a video.",
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=320,
        help="Reject if any sampled frame is narrower than this.",
    )
    parser.add_argument(
        "--min-height",
        type=int,
        default=240,
        help="Reject if any sampled frame is shorter than this.",
    )
    parser.add_argument(
        "--min-brightness",
        type=float,
        default=40.0,
        help="Reject if average brightness is below this.",
    )
    parser.add_argument(
        "--max-brightness",
        type=float,
        default=215.0,
        help="Reject if average brightness is above this.",
    )
    parser.add_argument(
        "--min-contrast",
        type=float,
        default=18.0,
        help="Reject if average grayscale contrast is below this.",
    )
    parser.add_argument(
        "--min-laplacian-variance",
        type=float,
        default=45.0,
        help="Reject if the minimum blur score is below this. Higher is sharper.",
    )
    parser.add_argument(
        "--provider",
        choices=("none", "gemini", "openai", "ollama"),
        default="none",
        help="LLM provider to use after quality passes.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Override the provider model name.",
    )
    parser.add_argument(
        "--labels",
        default="come,wave,stop,idle,talk_phone,play_phone",
        help="Comma-separated suggested motion labels. The model may still propose a new label if none fit.",
    )
    parser.add_argument(
        "--closed-set-labels",
        action="store_true",
        help="Force the model to pick only from --labels instead of inventing a new label.",
    )
    parser.add_argument(
        "--extra-prompt",
        default="",
        help="Optional extra instructions appended to the labeling prompt.",
    )
    parser.add_argument(
        "--ollama-host",
        default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
        help="Ollama server base URL. Used when --provider ollama.",
    )
    parser.add_argument(
        "--video-mode",
        choices=("windowed", "global"),
        default="windowed",
        help="For videos, label one global sample set or multiple temporal windows.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=4.0,
        help="Window size in seconds when --video-mode windowed is used.",
    )
    parser.add_argument(
        "--window-overlap",
        type=float,
        default=0.5,
        help="Fractional overlap between adjacent windows in [0, 0.95).",
    )
    return parser.parse_args()


def require_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Failed to import cv2. Install OpenCV with `pip install opencv-python`."
        ) from exc
    return cv2


def _laplacian_variance(gray, cv2_module) -> float:
    return float(cv2_module.Laplacian(gray, cv2_module.CV_64F).var())


def _frame_quality_from_bgr(frame, frame_index: int, timestamp_ms: int, path: str | None, cv2_module) -> FrameQuality:
    gray = cv2_module.cvtColor(frame, cv2_module.COLOR_BGR2GRAY)
    blur_score = _laplacian_variance(gray, cv2_module)
    brightness_mean = float(gray.mean())
    contrast_std = float(gray.std())
    height, width = gray.shape[:2]
    return FrameQuality(
        index=frame_index,
        timestamp_ms=timestamp_ms,
        blur_score=blur_score,
        brightness_mean=brightness_mean,
        contrast_std=contrast_std,
        width=width,
        height=height,
        path=path,
    )


def _resolve_output_path(path_str: str) -> Path:
    out_path = Path(path_str)
    if out_path.parent == Path(""):
        return Path.cwd() / out_path
    return out_path


def _compute_sample_positions(total_count: int, sample_count: int, start_index: int = 0, end_index: int | None = None) -> list[int]:
    if total_count <= 0:
        return list(range(max(1, sample_count)))
    bounded_end = total_count - 1 if end_index is None else max(start_index, min(total_count - 1, end_index))
    bounded_start = max(0, min(start_index, bounded_end))
    span = bounded_end - bounded_start + 1
    wanted = max(1, min(sample_count, span))
    positions = {
        bounded_start + min(span - 1, int(round(i * (span - 1) / max(1, wanted - 1))))
        for i in range(wanted)
    }
    return sorted(positions)


def sample_video_frames(video_path: Path, sample_frames: int, save_samples_dir: str, cv2_module) -> tuple[list[FrameQuality], list[Path], dict[str, Any]]:
    cap = cv2_module.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2_module.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2_module.CAP_PROP_FPS) or 0.0)
    duration_ms = None
    if frame_count > 0 and fps > 0:
        duration_ms = int(round((frame_count / fps) * 1000.0))

    sample_count = max(1, int(sample_frames))
    positions = _compute_sample_positions(frame_count, sample_count)

    saved_paths: list[Path] = []
    frame_reports: list[FrameQuality] = []
    output_dir = Path(save_samples_dir) if save_samples_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for sample_idx, frame_pos in enumerate(positions):
        cap.set(cv2_module.CAP_PROP_POS_FRAMES, float(frame_pos))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        timestamp_ms = int(round(cap.get(cv2_module.CAP_PROP_POS_MSEC) or 0.0))
        sample_path: Path | None = None
        if output_dir is not None:
            sample_path = output_dir / f"{video_path.stem}_sample_{sample_idx:02d}_{timestamp_ms}ms.jpg"
            cv2_module.imwrite(str(sample_path), frame)
            saved_paths.append(sample_path)
        frame_reports.append(
            _frame_quality_from_bgr(
                frame,
                frame_index=int(frame_pos),
                timestamp_ms=timestamp_ms,
                path=str(sample_path) if sample_path is not None else None,
                cv2_module=cv2_module,
            )
        )

    cap.release()
    meta = {"duration_ms": duration_ms, "fps": fps if fps > 0 else None}
    return frame_reports, saved_paths, meta


def sample_video_window_frames(
    video_path: Path,
    sample_frames: int,
    save_samples_dir: str,
    cv2_module,
    window_index: int,
    start_ms: int,
    end_ms: int,
) -> tuple[list[FrameQuality], list[Path]]:
    cap = cv2_module.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2_module.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2_module.CAP_PROP_FPS) or 0.0)
    if frame_count <= 0 or fps <= 0:
        cap.release()
        return [], []

    start_index = max(0, min(frame_count - 1, int(math.floor((start_ms / 1000.0) * fps))))
    end_index = max(start_index, min(frame_count - 1, int(math.ceil((end_ms / 1000.0) * fps)) - 1))
    positions = _compute_sample_positions(frame_count, max(1, int(sample_frames)), start_index, end_index)

    output_dir = Path(save_samples_dir) if save_samples_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    frame_reports: list[FrameQuality] = []
    saved_paths: list[Path] = []
    for sample_idx, frame_pos in enumerate(positions):
        cap.set(cv2_module.CAP_PROP_POS_FRAMES, float(frame_pos))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        timestamp_ms = int(round(cap.get(cv2_module.CAP_PROP_POS_MSEC) or 0.0))
        sample_path: Path | None = None
        if output_dir is not None:
            sample_path = output_dir / (
                f"{video_path.stem}_window_{window_index:03d}_sample_{sample_idx:02d}_{timestamp_ms}ms.jpg"
            )
            cv2_module.imwrite(str(sample_path), frame)
            saved_paths.append(sample_path)
        frame_reports.append(
            _frame_quality_from_bgr(
                frame,
                frame_index=int(frame_pos),
                timestamp_ms=timestamp_ms,
                path=str(sample_path) if sample_path is not None else None,
                cv2_module=cv2_module,
            )
        )

    cap.release()
    return frame_reports, saved_paths


def sample_image_directory(images_dir: Path, sample_frames: int, cv2_module) -> tuple[list[FrameQuality], list[Path], dict[str, Any]]:
    candidates = sorted(
        path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTS
    )
    if not candidates:
        raise SystemExit(f"No supported images found in: {images_dir}")

    sample_count = min(len(candidates), max(1, int(sample_frames)))
    positions = sorted({min(len(candidates) - 1, int(round(i * (len(candidates) - 1) / max(1, sample_count - 1)))) for i in range(sample_count)})

    frame_reports: list[FrameQuality] = []
    sampled_paths: list[Path] = []
    for idx in positions:
        path = candidates[idx]
        frame = cv2_module.imread(str(path))
        if frame is None:
            continue
        frame_reports.append(
            _frame_quality_from_bgr(
                frame,
                frame_index=idx,
                timestamp_ms=idx,
                path=str(path),
                cv2_module=cv2_module,
            )
        )
        sampled_paths.append(path)

    return frame_reports, sampled_paths, {"duration_ms": None, "fps": None}


def build_quality_report(
    source_type: str,
    source_path: str,
    frames: list[FrameQuality],
    meta: dict[str, Any],
    args: argparse.Namespace,
) -> QualityReport:
    if not frames:
        raise SystemExit("No readable frames were sampled.")

    avg_blur = float(statistics.fmean(frame.blur_score for frame in frames))
    min_blur = float(min(frame.blur_score for frame in frames))
    avg_brightness = float(statistics.fmean(frame.brightness_mean for frame in frames))
    avg_contrast = float(statistics.fmean(frame.contrast_std for frame in frames))
    min_width = int(min(frame.width for frame in frames))
    min_height = int(min(frame.height for frame in frames))

    failure_reasons: list[str] = []
    if min_width < args.min_width:
        failure_reasons.append(f"minimum sampled width {min_width} < required {args.min_width}")
    if min_height < args.min_height:
        failure_reasons.append(f"minimum sampled height {min_height} < required {args.min_height}")
    if avg_brightness < args.min_brightness:
        failure_reasons.append(
            f"average brightness {avg_brightness:.1f} < required {args.min_brightness:.1f}"
        )
    if avg_brightness > args.max_brightness:
        failure_reasons.append(
            f"average brightness {avg_brightness:.1f} > allowed {args.max_brightness:.1f}"
        )
    if avg_contrast < args.min_contrast:
        failure_reasons.append(f"average contrast {avg_contrast:.1f} < required {args.min_contrast:.1f}")
    if min_blur < args.min_laplacian_variance:
        failure_reasons.append(
            f"minimum sharpness {min_blur:.1f} < required {args.min_laplacian_variance:.1f}"
        )

    return QualityReport(
        source_type=source_type,
        source_path=source_path,
        sampled_frames=len(frames),
        min_width=min_width,
        min_height=min_height,
        avg_blur_score=avg_blur,
        min_blur_score=min_blur,
        avg_brightness_mean=avg_brightness,
        avg_contrast_std=avg_contrast,
        duration_ms=meta.get("duration_ms"),
        fps=meta.get("fps"),
        passed=not failure_reasons,
        failure_reasons=failure_reasons,
        frames=frames,
    )


def _mime_type_for_path(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    return mime_type or "image/jpeg"


def _to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _prompt_for_motion_labels(labels: list[str], extra_prompt: str, closed_set_labels: bool) -> str:
    label_text = ", ".join(label.strip() for label in labels if label.strip())
    prompt = "You are labeling a short clip of one person. Look across the provided frames and infer the main human motion/action. "
    if label_text:
        prompt += f"Known labels to prefer when they fit well: {label_text}. "
    if closed_set_labels:
        prompt += (
            "You must choose a label from the known labels only. "
            "Return strict JSON with keys: label, label_source, matched_candidate, confidence, summary, evidence. "
            "Set label_source to 'known'. matched_candidate must equal the chosen known label. "
        )
    else:
        prompt += (
            "If one known label fits well, use it exactly. "
            "If none fit well, invent a short new snake_case label that better describes the motion. "
            "Return strict JSON with keys: label, label_source, matched_candidate, confidence, summary, evidence. "
            "Set label_source to 'known' when label comes from the known labels, otherwise set it to 'new'. "
            "Set matched_candidate to the chosen known label when one fits, otherwise null. "
        )
    prompt += (
        "The label should be a short snake_case string. confidence should be a number from 0 to 1. "
        "summary should be one sentence. evidence should be an array of short reasons."
    )
    if extra_prompt.strip():
        prompt = f"{prompt} Additional instructions: {extra_prompt.strip()}"
    return prompt


def _try_parse_json_text(raw_text: str) -> dict[str, Any] | None:
    cleaned = raw_text.strip()
    if not cleaned:
        return None
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            cleaned = "\n".join(lines[1:-1]).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_label_and_confidence(labeling: dict[str, Any] | None) -> tuple[str | None, float | None]:
    if not labeling:
        return None, None
    parsed = labeling.get("parsed_json")
    if not isinstance(parsed, dict):
        return None, None
    label = parsed.get("label")
    confidence = parsed.get("confidence")
    normalized_label = str(label).strip() if label is not None else ""
    if not normalized_label:
        return None, None
    normalized_confidence: float | None = None
    if isinstance(confidence, (int, float)):
        normalized_confidence = float(confidence)
    return normalized_label, normalized_confidence


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"API request failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"API request failed: {exc}") from exc


def _ollama_model_not_found_message(model_name: str) -> str:
    return (
        f"Ollama model '{model_name}' is not installed locally. "
        f"Run `ollama pull {model_name}` first, then retry."
    )


def call_gemini(
    sample_paths: list[Path],
    labels: list[str],
    extra_prompt: str,
    model: str,
    closed_set_labels: bool,
) -> dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY or GOOGLE_API_KEY before using --provider gemini.")

    model_name = model or "gemini-2.5-flash"
    parts: list[dict[str, Any]] = [{"text": _prompt_for_motion_labels(labels, extra_prompt, closed_set_labels)}]
    for path in sample_paths:
        parts.append(
            {
                "inline_data": {
                    "mime_type": _mime_type_for_path(path),
                    "data": _to_base64(path),
                }
            }
        )

    payload = {"contents": [{"role": "user", "parts": parts}]}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    response = _post_json(url, payload, {"Content-Type": "application/json"})
    text_parts: list[str] = []
    for candidate in response.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            if isinstance(part, dict) and "text" in part:
                text_parts.append(str(part["text"]))
    joined = "\n".join(text_parts).strip()
    return {
        "provider": "gemini",
        "model": model_name,
        "raw_text": joined,
        "parsed_json": _try_parse_json_text(joined),
        "response": response,
    }


def call_openai(
    sample_paths: list[Path],
    labels: list[str],
    extra_prompt: str,
    model: str,
    closed_set_labels: bool,
) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY before using --provider openai.")

    model_name = model or "gpt-4.1-mini"
    input_parts: list[dict[str, Any]] = [{"type": "input_text", "text": _prompt_for_motion_labels(labels, extra_prompt, closed_set_labels)}]
    for path in sample_paths:
        input_parts.append(
            {
                "type": "input_image",
                "image_url": f"data:{_mime_type_for_path(path)};base64,{_to_base64(path)}",
            }
        )

    payload = {
        "model": model_name,
        "input": [{"role": "user", "content": input_parts}],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = _post_json("https://api.openai.com/v1/responses", payload, headers)

    text_chunks: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and "text" in content:
                text_chunks.append(str(content["text"]))
    joined = "\n".join(text_chunks).strip()
    return {
        "provider": "openai",
        "model": model_name,
        "raw_text": joined,
        "parsed_json": _try_parse_json_text(joined),
        "response": response,
    }


def call_ollama(
    sample_paths: list[Path],
    labels: list[str],
    extra_prompt: str,
    model: str,
    ollama_host: str,
    closed_set_labels: bool,
) -> dict[str, Any]:
    model_name = model or "qwen2.5vl"
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": _prompt_for_motion_labels(labels, extra_prompt, closed_set_labels),
                "images": [_to_base64(path) for path in sample_paths],
            }
        ],
        "stream": False,
        "format": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "label_source": {"type": "string", "enum": ["known", "new"]},
                "matched_candidate": {"type": ["string", "null"]},
                "confidence": {"type": "number"},
                "summary": {"type": "string"},
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["label", "label_source", "matched_candidate", "confidence", "summary", "evidence"],
        },
        "options": {
            "temperature": 0,
        },
    }
    url = ollama_host.rstrip("/") + "/api/chat"
    try:
        response = _post_json(url, payload, {"Content-Type": "application/json"})
    except SystemExit as exc:
        message = str(exc)
        if "model '" in message and "not found" in message:
            raise SystemExit(_ollama_model_not_found_message(model_name)) from exc
        raise
    message = response.get("message", {}) if isinstance(response, dict) else {}
    joined = str(message.get("content") or "").strip()
    return {
        "provider": "ollama",
        "model": model_name,
        "host": ollama_host,
        "raw_text": joined,
        "parsed_json": _try_parse_json_text(joined),
        "response": response,
    }


def maybe_label_clip(
    provider: str,
    sample_paths: list[Path],
    labels: list[str],
    extra_prompt: str,
    model: str,
    ollama_host: str,
    closed_set_labels: bool,
) -> dict[str, Any] | None:
    if provider == "none":
        return None
    if not sample_paths:
        raise SystemExit("No sampled image paths available for AI labeling.")
    if provider == "gemini":
        return call_gemini(sample_paths, labels, extra_prompt, model, closed_set_labels)
    if provider == "openai":
        return call_openai(sample_paths, labels, extra_prompt, model, closed_set_labels)
    if provider == "ollama":
        return call_ollama(sample_paths, labels, extra_prompt, model, ollama_host, closed_set_labels)
    raise SystemExit(f"Unsupported provider: {provider}")


def label_video_in_windows(
    video_path: Path,
    meta: dict[str, Any],
    args: argparse.Namespace,
    labels: list[str],
    cv2_module,
    save_samples_dir: str,
) -> dict[str, Any]:
    duration_ms = meta.get("duration_ms")
    if not isinstance(duration_ms, int) or duration_ms <= 0:
        raise SystemExit("Windowed video labeling requires a readable video duration.")
    if not 0.0 <= args.window_overlap < 0.95:
        raise SystemExit("--window-overlap must be in the range [0, 0.95).")
    if args.window_seconds <= 0:
        raise SystemExit("--window-seconds must be > 0.")

    window_ms = max(1, int(round(args.window_seconds * 1000.0)))
    stride_ms = max(1, int(round(window_ms * (1.0 - args.window_overlap))))

    windows: list[WindowLabelResult] = []
    start_ms = 0
    window_index = 0
    while start_ms < duration_ms:
        end_ms = min(duration_ms, start_ms + window_ms)
        _, sample_paths = sample_video_window_frames(
            video_path,
            args.sample_frames,
            save_samples_dir,
            cv2_module,
            window_index,
            start_ms,
            end_ms,
        )
        labeling = maybe_label_clip(
            args.provider,
            sample_paths,
            labels,
            args.extra_prompt,
            args.model,
            args.ollama_host,
            args.closed_set_labels,
        )
        windows.append(
            WindowLabelResult(
                window_index=window_index,
                start_ms=start_ms,
                end_ms=end_ms,
                sample_paths=[str(path) for path in sample_paths],
                labeling=labeling,
            )
        )
        if end_ms >= duration_ms:
            break
        start_ms += stride_ms
        window_index += 1

    label_counter: collections.Counter[str] = collections.Counter()
    label_confidences: dict[str, list[float]] = {}
    for window in windows:
        label, confidence = _extract_label_and_confidence(window.labeling)
        if label is None:
            continue
        label_counter[label] += 1
        if confidence is not None:
            label_confidences.setdefault(label, []).append(confidence)

    winning_label: str | None = None
    winning_votes = 0
    winning_avg_confidence: float | None = None
    if label_counter:
        ranked = sorted(
            label_counter.items(),
            key=lambda item: (
                item[1],
                statistics.fmean(label_confidences.get(item[0], [0.0])),
                item[0],
            ),
            reverse=True,
        )
        winning_label, winning_votes = ranked[0]
        if label_confidences.get(winning_label):
            winning_avg_confidence = float(statistics.fmean(label_confidences[winning_label]))

    return {
        "provider": args.provider,
        "model": args.model or ("qwen2.5vl" if args.provider == "ollama" else None),
        "mode": "windowed",
        "window_seconds": args.window_seconds,
        "window_overlap": args.window_overlap,
        "windows": [asdict(window) for window in windows],
        "summary": {
            "label": winning_label,
            "votes": winning_votes,
            "total_windows": len(windows),
            "avg_confidence": winning_avg_confidence,
            "vote_counts": dict(label_counter),
        },
    }


def print_quality_summary(report: QualityReport) -> None:
    print(f"source: {report.source_path}")
    print(f"sampled_frames: {report.sampled_frames}")
    if report.duration_ms is not None:
        print(f"duration_ms: {report.duration_ms}")
    if report.fps is not None:
        print(f"fps: {report.fps:.2f}")
    print(f"min_resolution: {report.min_width}x{report.min_height}")
    print(f"avg_brightness: {report.avg_brightness_mean:.2f}")
    print(f"avg_contrast: {report.avg_contrast_std:.2f}")
    print(f"avg_sharpness: {report.avg_blur_score:.2f}")
    print(f"min_sharpness: {report.min_blur_score:.2f}")
    print(f"quality_passed: {report.passed}")
    if report.failure_reasons:
        print("failure_reasons:")
        for reason in report.failure_reasons:
            print(f"- {reason}")


def main() -> None:
    args = parse_args()
    cv2_module = require_cv2()

    output_path = _resolve_output_path(args.output_json)
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    if not labels:
        raise SystemExit("--labels must contain at least one label")

    if args.video:
        source_path = Path(args.video)
        if not source_path.exists():
            raise SystemExit(f"Video not found: {source_path}")
        save_samples_dir = args.save_samples_dir
        if args.provider != "none" and not save_samples_dir:
            save_samples_dir = tempfile.mkdtemp(prefix="label_samples_")
        frames, saved_paths, meta = sample_video_frames(source_path, args.sample_frames, save_samples_dir, cv2_module)
        source_type = "video"
        ai_paths = saved_paths
    else:
        source_path = Path(args.images_dir)
        if not source_path.exists() or not source_path.is_dir():
            raise SystemExit(f"Image directory not found: {source_path}")
        frames, sampled_paths, meta = sample_image_directory(source_path, args.sample_frames, cv2_module)
        source_type = "images_dir"
        ai_paths = sampled_paths

    report = build_quality_report(source_type, str(source_path), frames, meta, args)
    print_quality_summary(report)

    result: dict[str, Any] = {"quality": asdict(report), "labeling": None}
    if report.passed:
        if args.video and args.provider != "none" and args.video_mode == "windowed":
            labeling = label_video_in_windows(
                source_path,
                meta,
                args,
                labels,
                cv2_module,
                save_samples_dir,
            )
        else:
            labeling = maybe_label_clip(
                args.provider,
                ai_paths,
                labels,
                args.extra_prompt,
                args.model,
                args.ollama_host,
                args.closed_set_labels,
            )
        result["labeling"] = labeling
        if labeling is not None:
            print("labeling_provider:", labeling.get("provider"))
            if args.video and args.provider != "none" and args.video_mode == "windowed":
                summary = labeling.get("summary", {}) if isinstance(labeling, dict) else {}
                print("windowed_label:", summary.get("label"))
                print("window_votes:", summary.get("votes"))
                print("window_total:", summary.get("total_windows"))
            else:
                print("labeling_model:", labeling.get("model"))
                parsed = labeling.get("parsed_json", {}) if isinstance(labeling, dict) else {}
                if isinstance(parsed, dict):
                    print("label_source:", parsed.get("label_source"))
                    print("matched_candidate:", parsed.get("matched_candidate"))
                raw_text = str(labeling.get("raw_text") or "").strip()
                if raw_text:
                    print("labeling_raw_text:")
                    print(raw_text)
    else:
        print("Skipping AI labeling because quality check failed.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main()
