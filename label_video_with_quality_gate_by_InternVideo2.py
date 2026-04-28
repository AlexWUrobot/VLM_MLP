import argparse
import importlib
import importlib.util
import json
import os
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BERT_LARGE_CONFIG = "configs/config_bert_large.json"


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


class ConfigNode(dict):
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    @classmethod
    def from_mapping(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return cls({k: cls.from_mapping(v) for k, v in value.items()})
        if isinstance(value, list):
            return [cls.from_mapping(item) for item in value]
        return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run quality checks, then label a whole clip offline with InternVideo2.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", help="Path to an input video file")
    group.add_argument("--images-dir", help="Path to a directory of images representing one clip")
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=8,
        help="How many frames/images to sample for the quality gate.",
    )
    parser.add_argument(
        "--save-samples-dir",
        default="",
        help="Optional directory to save sampled frames from the quality check.",
    )
    parser.add_argument(
        "--output-json",
        default="quality_report_internvideo2.json",
        help="Where to save the quality and labeling report.",
    )
    parser.add_argument("--min-width", type=int, default=320)
    parser.add_argument("--min-height", type=int, default=240)
    parser.add_argument("--min-brightness", type=float, default=40.0)
    parser.add_argument("--max-brightness", type=float, default=215.0)
    parser.add_argument("--min-contrast", type=float, default=18.0)
    parser.add_argument("--min-laplacian-variance", type=float, default=45.0)
    parser.add_argument(
        "--labels",
        default="come,wave,stop,idle,talk_phone,play_phone",
        help="Comma-separated candidate motion labels for InternVideo2 zero-shot classification.",
    )
    parser.add_argument(
        "--closed-set-labels",
        action="store_true",
        help="Always choose one of --labels even if the top InternVideo2 score is low.",
    )
    parser.add_argument(
        "--label-definitions-json",
        default="",
        help="Optional JSON file mapping labels to richer natural-language definitions.",
    )
    parser.add_argument(
        "--label-prompt-template",
        default="a video of a person {}",
        help="Prompt template used for each label or label definition. Use one {} placeholder.",
    )
    parser.add_argument(
        "--unknown-threshold",
        type=float,
        default=0.35,
        help="If the best score is below this and --closed-set-labels is off, emit unknown_motion.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many top candidate labels to include in the report.",
    )
    parser.add_argument(
        "--extra-prompt",
        default="",
        help="Optional extra note appended to the label prompt text.",
    )
    parser.add_argument(
        "--internvideo2-root",
        default=os.getenv("INTERNVIDEO2_ROOT", ""),
        help="Path to a local clone of the InternVideo repository or its InternVideo2/multi_modality folder.",
    )
    parser.add_argument(
        "--internvideo2-vision-pretrained",
        default=os.getenv("INTERNVIDEO2_VISION_PRETRAINED", ""),
        help="Path to the InternVideo2 vision checkpoint used by the demo config.",
    )
    parser.add_argument(
        "--internvideo2-pretrained-path",
        default=os.getenv("INTERNVIDEO2_PRETRAINED_PATH", ""),
        help="Optional full InternVideo2 stage-2 checkpoint to load after model construction.",
    )
    parser.add_argument(
        "--internvideo2-num-frames",
        type=int,
        default=4,
        help="Number of frames passed into InternVideo2 for whole-clip labeling.",
    )
    parser.add_argument(
        "--internvideo2-image-size",
        type=int,
        default=224,
        help="Video frame resolution expected by the InternVideo2 demo model.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, cuda, or cuda:0 style values.",
    )
    return parser.parse_args()


def require_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit("Failed to import cv2. Install OpenCV with `pip install opencv-python`.") from exc
    return cv2


def require_numpy():
    try:
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit("Failed to import numpy. Install it with `pip install numpy`.") from exc
    return np


def require_torch():
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit("Failed to import torch. Install PyTorch before using InternVideo2.") from exc
    return torch


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
    output_path = Path(path_str)
    if output_path.parent == Path(""):
        return Path.cwd() / output_path
    return output_path


def _compute_sample_positions(total_count: int, sample_count: int) -> list[int]:
    if total_count <= 0:
        return list(range(max(1, sample_count)))
    wanted = max(1, min(sample_count, total_count))
    positions = {
        min(total_count - 1, int(round(index * (total_count - 1) / max(1, wanted - 1))))
        for index in range(wanted)
    }
    return sorted(positions)


def sample_video_frames(
    video_path: Path,
    sample_frames: int,
    save_samples_dir: str,
    cv2_module,
) -> tuple[list[Any], list[FrameQuality], dict[str, Any]]:
    cap = cv2_module.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2_module.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2_module.CAP_PROP_FPS) or 0.0)
    duration_ms = None
    if frame_count > 0 and fps > 0:
        duration_ms = int(round((frame_count / fps) * 1000.0))

    positions = _compute_sample_positions(frame_count, max(1, int(sample_frames)))
    output_dir = Path(save_samples_dir) if save_samples_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    sampled_frames_bgr: list[Any] = []
    frame_reports: list[FrameQuality] = []
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
        sampled_frames_bgr.append(frame)
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
    return sampled_frames_bgr, frame_reports, {"duration_ms": duration_ms, "fps": fps if fps > 0 else None}


def sample_image_directory(
    images_dir: Path,
    sample_frames: int,
    cv2_module,
) -> tuple[list[Any], list[FrameQuality], dict[str, Any]]:
    candidates = sorted(path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTS)
    if not candidates:
        raise SystemExit(f"No supported images found in: {images_dir}")

    positions = _compute_sample_positions(len(candidates), max(1, int(sample_frames)))
    sampled_frames_bgr: list[Any] = []
    frame_reports: list[FrameQuality] = []
    for index in positions:
        path = candidates[index]
        frame = cv2_module.imread(str(path))
        if frame is None:
            continue
        sampled_frames_bgr.append(frame)
        frame_reports.append(
            _frame_quality_from_bgr(
                frame,
                frame_index=index,
                timestamp_ms=index,
                path=str(path),
                cv2_module=cv2_module,
            )
        )

    return sampled_frames_bgr, frame_reports, {"duration_ms": None, "fps": None}


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
        failure_reasons.append(f"average brightness {avg_brightness:.1f} < required {args.min_brightness:.1f}")
    if avg_brightness > args.max_brightness:
        failure_reasons.append(f"average brightness {avg_brightness:.1f} > allowed {args.max_brightness:.1f}")
    if avg_contrast < args.min_contrast:
        failure_reasons.append(f"average contrast {avg_contrast:.1f} < required {args.min_contrast:.1f}")
    if min_blur < args.min_laplacian_variance:
        failure_reasons.append(f"minimum sharpness {min_blur:.1f} < required {args.min_laplacian_variance:.1f}")

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


def _resolve_internvideo2_root(root: str) -> Path:
    if not root.strip():
        raise SystemExit(
            "Set --internvideo2-root to a local clone of OpenGVLab/InternVideo or export INTERNVIDEO2_ROOT."
        )
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise SystemExit(
            f"InternVideo2 root does not exist: {root_path}. Clone OpenGVLab/InternVideo first or point "
            "--internvideo2-root at the real repo location."
        )
    if (root_path / "InternVideo2" / "multi_modality").exists():
        return root_path / "InternVideo2" / "multi_modality"
    if (root_path / "demo" / "utils.py").exists() and (root_path / "models").exists():
        return root_path
    raise SystemExit(
        f"Could not find InternVideo2/multi_modality under: {root_path}. "
        "Point --internvideo2-root at the repo root or at InternVideo2/multi_modality."
    )


def _import_internvideo2_utils(multi_modality_root: Path):
    if multi_modality_root.name != "multi_modality" or multi_modality_root.parent.name != "InternVideo2":
        raise SystemExit(
            f"Unsupported InternVideo2 package layout at: {multi_modality_root}. Expected .../InternVideo2/multi_modality."
        )
    repo_root = multi_modality_root.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return importlib.import_module("InternVideo2.multi_modality.demo.utils")


def _resolve_device(device_arg: str, torch_module) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def _ensure_writable_transformers_cache() -> None:
    cache_root = Path.cwd() / ".cache" / "huggingface"
    cache_root.mkdir(parents=True, exist_ok=True)

    current_transformers_cache = os.getenv("TRANSFORMERS_CACHE", "").strip()
    if current_transformers_cache:
        configured = Path(current_transformers_cache).expanduser()
        if configured.exists() and os.access(configured, os.W_OK):
            return

    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_root / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "transformers")


def _build_internvideo2_config(args: argparse.Namespace, multi_modality_root: Path, device_name: str) -> ConfigNode:
    if not args.internvideo2_vision_pretrained.strip():
        raise SystemExit(
            "Set --internvideo2-vision-pretrained to the InternVideo2 vision checkpoint path or export INTERNVIDEO2_VISION_PRETRAINED."
        )
    vision_pretrained_path = Path(args.internvideo2_vision_pretrained).expanduser()
    if not vision_pretrained_path.exists():
        raise SystemExit(
            f"InternVideo2 vision checkpoint not found: {vision_pretrained_path}. Set --internvideo2-vision-pretrained "
            "to a real checkpoint file."
        )
    if args.internvideo2_pretrained_path.strip():
        stage2_path = Path(args.internvideo2_pretrained_path).expanduser()
        if not stage2_path.exists():
            raise SystemExit(
                f"InternVideo2 stage-2 checkpoint not found: {stage2_path}. Set --internvideo2-pretrained-path "
                "to a real checkpoint file or omit it."
            )
    text_encoder = {
        "name": "bert_large",
        "pretrained": "bert-large-uncased",
        "config": str(multi_modality_root / BERT_LARGE_CONFIG),
        "d_model": 1024,
        "fusion_layer": 19,
    }
    config = {
        "num_frames": int(args.internvideo2_num_frames),
        "origin_num_frames": int(args.internvideo2_num_frames),
        "max_txt_l": 40,
        "device": device_name,
        "use_half_precision": False,
        "use_bf16": False,
        "compile_model": False,
        "gradient_checkpointing": True,
        "pretrained_path": args.internvideo2_pretrained_path,
        "model": {
            "model_cls": "InternVideo2_Stage2",
            "vision_encoder": {
                "name": "pretrain_internvideo2_1b_patch14_224",
                "img_size": int(args.internvideo2_image_size),
                "num_frames": int(args.internvideo2_num_frames),
                "tubelet_size": 1,
                "patch_size": 14,
                "d_model": 1408,
                "clip_embed_dim": 768,
                "clip_teacher_embed_dim": 3200,
                "clip_teacher_final_dim": 768,
                "clip_norm_type": "l2",
                "clip_return_layer": 6,
                "clip_student_return_interval": 1,
                "pretrained": str(vision_pretrained_path),
                "use_checkpoint": True,
                "checkpoint_num": 40,
                "use_flash_attn": False,
                "use_fused_rmsnorm": False,
                "use_fused_mlp": False,
                "clip_teacher": None,
                "clip_input_resolution": int(args.internvideo2_image_size),
                "clip_teacher_return_interval": 1,
                "video_mask_type": "random",
                "video_mask_ratio": 0.8,
                "image_mask_type": "random",
                "image_mask_ratio": 0.5,
                "sep_image_video_pos_embed": True,
                "keep_temporal": False,
                "only_mask": True,
            },
            "text_encoder": text_encoder,
            "multimodal": {"enable": True},
            "embed_dim": 512,
            "temp": 0.07,
            "find_unused_parameters": False,
        },
    }
    return ConfigNode.from_mapping(config)


def _frames_to_internvideo2_tensor(frames_bgr: list[Any], num_frames: int, image_size: int, torch_module, np_module, device_name: str):
    if len(frames_bgr) < num_frames:
        raise SystemExit(
            f"InternVideo2 requires at least {num_frames} frames, but only {len(frames_bgr)} sampled frames were available."
        )
    step = max(1, len(frames_bgr) // num_frames)
    picked = frames_bgr[::step][:num_frames]
    mean = np_module.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np_module.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    resized = []
    cv2_module = require_cv2()
    for frame in picked:
        rgb = cv2_module.resize(frame[:, :, ::-1], (image_size, image_size))
        normalized = (rgb / 255.0 - mean) / std
        resized.append(normalized)
    video = np_module.stack(resized, axis=0)
    video = np_module.transpose(video, (0, 3, 1, 2))
    video = np_module.expand_dims(video, axis=0)
    tensor = torch_module.from_numpy(video).to(device_name, non_blocking=True).float()
    return tensor


def _load_label_definitions(definitions_path: str) -> dict[str, str]:
    if not definitions_path.strip():
        return {}
    path = Path(definitions_path)
    if not path.exists():
        raise SystemExit(f"Label definitions file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("--label-definitions-json must point to a JSON object mapping label -> definition.")
    return {str(key): str(value) for key, value in data.items()}


def _label_phrase(label: str, definitions: dict[str, str]) -> str:
    if label in definitions:
        return definitions[label]
    return label.replace("_", " ").replace("-", " ").strip()


def _build_candidate_prompts(
    labels: list[str],
    definitions: dict[str, str],
    template: str,
    extra_prompt: str,
) -> list[tuple[str, str]]:
    if "{}" not in template:
        raise SystemExit("--label-prompt-template must contain one {} placeholder.")
    prompts: list[tuple[str, str]] = []
    suffix = f". {extra_prompt.strip()}" if extra_prompt.strip() else ""
    for label in labels:
        phrase = _label_phrase(label, definitions)
        prompt = template.format(phrase).strip()
        prompts.append((label, f"{prompt}{suffix}".strip()))
    return prompts


def _candidate_scores_to_list(
    ranked_indices: list[int],
    ranked_scores: list[float],
    candidates: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, score in zip(ranked_indices, ranked_scores):
        label, prompt = candidates[index]
        results.append({
            "label": label,
            "prompt": prompt,
            "score": float(score),
        })
    return results


def label_with_internvideo2(frames_bgr: list[Any], args: argparse.Namespace, labels: list[str]) -> dict[str, Any]:
    torch_module = require_torch()
    np_module = require_numpy()
    device_name = _resolve_device(args.device, torch_module)
    _ensure_writable_transformers_cache()
    multi_modality_root = _resolve_internvideo2_root(args.internvideo2_root)
    utils_module = _import_internvideo2_utils(multi_modality_root)
    config = _build_internvideo2_config(args, multi_modality_root, device_name)
    try:
        model, _ = utils_module.setup_internvideo2(config)
    except Exception as exc:
        raise SystemExit(
            "Failed to initialize InternVideo2. Verify that the InternVideo repository dependencies are installed, "
            "that bert-large-uncased is available locally or downloadable, and that the checkpoint paths are correct. "
            f"Original error: {exc}"
        ) from exc

    frames_tensor = _frames_to_internvideo2_tensor(
        frames_bgr,
        int(args.internvideo2_num_frames),
        int(args.internvideo2_image_size),
        torch_module,
        np_module,
        device_name,
    )
    definitions = _load_label_definitions(args.label_definitions_json)
    candidate_prompts = _build_candidate_prompts(labels, definitions, args.label_prompt_template, args.extra_prompt)

    with torch_module.no_grad():
        vid_feat = model.get_vid_feat(frames_tensor)
        text_feats = [model.get_txt_feat(prompt) for _, prompt in candidate_prompts]
        text_feat_tensor = torch_module.cat(text_feats, dim=0)
        top_k = min(max(1, int(args.top_k)), len(candidate_prompts))
        top_probs, top_labels = model.predict_label(vid_feat, text_feat_tensor, top=top_k)

    ranked_indices = [int(item) for item in top_labels[0].tolist()]
    ranked_scores = [float(item) for item in top_probs[0].tolist()]
    ranked_candidates = _candidate_scores_to_list(ranked_indices, ranked_scores, candidate_prompts)

    best_candidate = ranked_candidates[0]
    known_label = best_candidate["label"]
    best_score = float(best_candidate["score"])
    if not args.closed_set_labels and best_score < float(args.unknown_threshold):
        label = "unknown_motion"
        label_source = "new"
        matched_candidate = None
        summary = (
            f"InternVideo2 did not strongly match any provided label. Best candidate was {known_label} "
            f"with score {best_score:.3f}."
        )
    else:
        label = known_label
        label_source = "known"
        matched_candidate = known_label
        summary = f"InternVideo2 ranked {known_label} as the best whole-video match."

    evidence = [
        f"{candidate['label']}: {candidate['score']:.3f}" for candidate in ranked_candidates
    ]
    parsed = {
        "label": label,
        "label_source": label_source,
        "matched_candidate": matched_candidate,
        "confidence": best_score,
        "summary": summary,
        "evidence": evidence,
    }
    return {
        "provider": "internvideo2",
        "model": "InternVideo2_Stage2_1B",
        "device": device_name,
        "candidate_prompts": [
            {"label": label_name, "prompt": prompt_text} for label_name, prompt_text in candidate_prompts
        ],
        "top_candidates": ranked_candidates,
        "parsed_json": parsed,
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
        frames_bgr, quality_frames, meta = sample_video_frames(
            source_path,
            args.sample_frames,
            args.save_samples_dir,
            cv2_module,
        )
        source_type = "video"
    else:
        source_path = Path(args.images_dir)
        if not source_path.exists() or not source_path.is_dir():
            raise SystemExit(f"Image directory not found: {source_path}")
        frames_bgr, quality_frames, meta = sample_image_directory(source_path, args.sample_frames, cv2_module)
        source_type = "images_dir"

    report = build_quality_report(source_type, str(source_path), quality_frames, meta, args)
    print_quality_summary(report)

    result: dict[str, Any] = {"quality": asdict(report), "labeling": None}
    if report.passed:
        labeling = label_with_internvideo2(frames_bgr, args, labels)
        result["labeling"] = labeling
        parsed = labeling.get("parsed_json", {}) if isinstance(labeling, dict) else {}
        print("labeling_provider:", labeling.get("provider"))
        print("labeling_model:", labeling.get("model"))
        print("label:", parsed.get("label"))
        print("confidence:", parsed.get("confidence"))
        print("matched_candidate:", parsed.get("matched_candidate"))
    else:
        print("Skipping InternVideo2 labeling because quality check failed.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main()
