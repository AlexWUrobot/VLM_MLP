import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SOURCE_LABELS_IN_ORDER = [
    "come",
    "idle_back",
    "idle_front",
    "none",
    "phone",
    "play_phone",
    "stop",
    "wave",
]

# Train/predict with 6 categories.
LABELS_IN_ORDER = [
    "come",
    "idle",
    "phone",
    "play_phone",
    "stop",
    "wave",
]

SOURCE_TO_TARGET_LABEL = {
    "come": "come",
    "idle_back": "idle",
    "idle_front": "idle",
    "none": "idle",
    "phone": "phone",
    "play_phone": "play_phone",
    "stop": "stop",
    "wave": "wave",
}


def label_to_index_0_based(label: str) -> int:
    return LABELS_IN_ORDER.index(label)


def infer_source_label_from_filename(name: str) -> str | None:
    # Prefer longest labels first to avoid matching 'phone' before 'play_phone'
    for label in sorted(SOURCE_LABELS_IN_ORDER, key=len, reverse=True):
        if name == label or name.startswith(label + "_"):
            return label
    return None


def infer_target_label_from_filename(name: str) -> str | None:
    src = infer_source_label_from_filename(name)
    if src is None:
        return None
    return SOURCE_TO_TARGET_LABEL.get(src)


def parse_clip_id_and_time_ms_from_path(path_str: str) -> tuple[str, int, int] | None:
    """Parse filenames like '<label>_<clipId>_<time>ms.txt' (or .jpg).

    Returns (source_label, clip_id, time_ms) or None if the pattern doesn't match.
    """
    stem = Path(path_str).stem
    source_label = infer_source_label_from_filename(stem)
    if source_label is None:
        return None

    rest = stem[len(source_label) :]
    if not rest.startswith("_"):
        return None

    parts = rest[1:].split("_")
    if len(parts) < 2:
        return None

    clip_id_s = parts[0]
    t_s = parts[1]
    if t_s.endswith("ms"):
        t_s = t_s[:-2]

    try:
        clip_id = int(clip_id_s)
        t_ms = int(t_s)
    except ValueError:
        return None

    return source_label, clip_id, t_ms


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str


class MarkerTemporalMLP(nn.Module):
    def __init__(
        self,
        *,
        pose_dim: int,
        hand_dim: int,
        hand_mask_dim: int,
        proj_dim: int,
        hidden: int,
        num_classes: int,
        num_frames: int,
        feat_agg: str,
    ):
        super().__init__()

        self.pose_dim = int(pose_dim)
        self.hand_dim = int(hand_dim)
        self.hand_mask_dim = int(hand_mask_dim)
        self.per_frame_dim = self.pose_dim + self.hand_dim + self.hand_mask_dim

        self.num_frames = max(1, int(num_frames))
        self.feat_agg = str(feat_agg).lower().strip()
        if self.feat_agg not in {"concat", "mean"}:
            raise ValueError("feat_agg must be 'concat' or 'mean'")

        self.proj_dim = int(proj_dim)

        def _proj(in_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(int(in_dim), self.proj_dim),
                nn.LayerNorm(self.proj_dim),
                nn.ReLU(),
            )

        self.pose_proj = _proj(self.pose_dim)
        self.hand_proj = _proj(self.hand_dim)
        self.hand_mask_proj = _proj(self.hand_mask_dim)

        self.frame_embed_dim = self.proj_dim * 3

        if self.feat_agg == "concat":
            head_in = self.frame_embed_dim * self.num_frames
        else:
            head_in = self.frame_embed_dim

        self.fc1 = nn.Linear(head_in, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is either:
        # - concat: (B, num_frames * per_frame_dim)
        # - mean:   (B, per_frame_dim)
        if self.feat_agg == "concat":
            if x.shape[1] != self.num_frames * self.per_frame_dim:
                raise RuntimeError(
                    f"input dim mismatch: got {x.shape[1]} expected {self.num_frames * self.per_frame_dim}"
                )
            xf = x.view(x.shape[0], self.num_frames, self.per_frame_dim)
        else:
            if x.shape[1] != self.per_frame_dim:
                raise RuntimeError(f"input dim mismatch: got {x.shape[1]} expected {self.per_frame_dim}")
            xf = x.view(x.shape[0], 1, self.per_frame_dim)

        pose = xf[:, :, : self.pose_dim]
        hands = xf[:, :, self.pose_dim : self.pose_dim + self.hand_dim]
        hmask = xf[:, :, -self.hand_mask_dim :]

        # project each block per frame
        b, t, _ = xf.shape
        pose_e = self.pose_proj(pose.reshape(b * t, self.pose_dim))
        hand_e = self.hand_proj(hands.reshape(b * t, self.hand_dim))
        mask_e = self.hand_mask_proj(hmask.reshape(b * t, self.hand_mask_dim))

        frame_e = torch.cat([pose_e, hand_e, mask_e], dim=1).view(b, t, self.frame_embed_dim)

        if self.feat_agg == "concat":
            h = frame_e.reshape(b, t * self.frame_embed_dim)
        else:
            h = frame_e.mean(dim=1)

        h = F.relu(self.fc1(h))
        h = F.dropout(h, p=0.2, training=self.training)
        h = F.relu(self.fc2(h))
        h = F.dropout(h, p=0.2, training=self.training)
        return self.fc3(h)


class MarkerTemporalRNN(nn.Module):
    def __init__(
        self,
        *,
        pose_dim: int,
        hand_dim: int,
        hand_mask_dim: int,
        proj_dim: int,
        hidden: int,
        num_classes: int,
        num_frames: int,
        rnn_type: str,
        rnn_hidden: int,
        rnn_layers: int,
        bidirectional: bool,
    ):
        super().__init__()

        self.pose_dim = int(pose_dim)
        self.hand_dim = int(hand_dim)
        self.hand_mask_dim = int(hand_mask_dim)
        self.per_frame_dim = self.pose_dim + self.hand_dim + self.hand_mask_dim
        self.num_frames = max(1, int(num_frames))

        self.proj_dim = int(proj_dim)
        self.rnn_type = str(rnn_type).lower().strip()
        if self.rnn_type not in {"gru", "lstm"}:
            raise ValueError("rnn_type must be 'gru' or 'lstm'")

        self.rnn_hidden = int(rnn_hidden)
        self.rnn_layers = max(1, int(rnn_layers))
        self.bidirectional = bool(bidirectional)

        def _proj(in_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(int(in_dim), self.proj_dim),
                nn.LayerNorm(self.proj_dim),
                nn.ReLU(),
            )

        self.pose_proj = _proj(self.pose_dim)
        self.hand_proj = _proj(self.hand_dim)
        self.hand_mask_proj = _proj(self.hand_mask_dim)
        self.frame_embed_dim = self.proj_dim * 3

        rnn_cls = nn.GRU if self.rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=self.frame_embed_dim,
            hidden_size=self.rnn_hidden,
            num_layers=self.rnn_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=0.2 if self.rnn_layers > 1 else 0.0,
        )

        head_in = self.rnn_hidden * (2 if self.bidirectional else 1)
        self.fc1 = nn.Linear(head_in, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.num_frames * self.per_frame_dim:
            raise RuntimeError(
                f"input dim mismatch: got {x.shape[1]} expected {self.num_frames * self.per_frame_dim}"
            )

        xf = x.view(x.shape[0], self.num_frames, self.per_frame_dim)
        pose = xf[:, :, : self.pose_dim]
        hands = xf[:, :, self.pose_dim : self.pose_dim + self.hand_dim]
        hmask = xf[:, :, -self.hand_mask_dim :]

        b, t, _ = xf.shape
        pose_e = self.pose_proj(pose.reshape(b * t, self.pose_dim))
        hand_e = self.hand_proj(hands.reshape(b * t, self.hand_dim))
        mask_e = self.hand_mask_proj(hmask.reshape(b * t, self.hand_mask_dim))
        frame_e = torch.cat([pose_e, hand_e, mask_e], dim=1).view(b, t, self.frame_embed_dim)

        out, hidden = self.rnn(frame_e)
        if self.rnn_type == "lstm":
            hidden = hidden[0]

        if self.bidirectional:
            last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            last_hidden = hidden[-1]

        h = F.relu(self.fc1(last_hidden))
        h = F.dropout(h, p=0.2, training=self.training)
        h = F.relu(self.fc2(h))
        h = F.dropout(h, p=0.2, training=self.training)
        return self.fc3(h)

POSE_DIM = 165
HAND_DIM = 126
HAND_MASK_DIM = HAND_DIM


def _parse_floats(s: str) -> list[float]:
    out: list[float] = []
    for tok in s.strip().split():
        t = tok.strip().lower()
        if t == "nan":
            out.append(float("nan"))
        else:
            out.append(float(t))
    return out


def load_pose_hand_from_txt(txt_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load pose(165) + hands(126) from a .txt produced by hand_body_detector.py."""
    if not txt_path.exists():
        pose = np.full((POSE_DIM,), np.nan, dtype=np.float32)
        hands = np.full((HAND_DIM,), np.nan, dtype=np.float32)
        return pose, hands

    pose_line: str | None = None
    hands_line: str | None = None

    for line in txt_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("pose:"):
            pose_line = line
        elif line.startswith("hands:"):
            hands_line = line

    if pose_line is None or hands_line is None:
        pose = np.full((POSE_DIM,), np.nan, dtype=np.float32)
        hands = np.full((HAND_DIM,), np.nan, dtype=np.float32)
        return pose, hands

    pose_vals = _parse_floats(pose_line[len("pose:") :])
    hand_vals = _parse_floats(hands_line[len("hands:") :])

    if len(pose_vals) != POSE_DIM or len(hand_vals) != HAND_DIM:
        pose = np.full((POSE_DIM,), np.nan, dtype=np.float32)
        hands = np.full((HAND_DIM,), np.nan, dtype=np.float32)
        return pose, hands

    return np.asarray(pose_vals, dtype=np.float32), np.asarray(hand_vals, dtype=np.float32)


def combine_pose_hands_handmask(
    pose: np.ndarray,
    hands: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (combined_feats, hand_missing_mask) for a single frame.

    Per-frame feature is:
            pose(165) + hands(126) + hand_mask(126)

    Missing values (NaN) are imputed with 0.
    Mask is ONLY for hands (126 dims): 1 where hand value was missing, else 0.

    No manual weighting here; the model learns per-block mixing.
    """
    pose_imp = np.nan_to_num(pose.astype(np.float32, copy=False), nan=0.0)

    hand_mask = np.isnan(hands).astype(np.float32)
    hands_imp = np.nan_to_num(hands.astype(np.float32, copy=False), nan=0.0)

    combined = np.concatenate([pose_imp, hands_imp, hand_mask], axis=0).astype(np.float32, copy=False)
    return combined, hand_mask


def build_marker_samples(marker_dir: Path, exts: tuple[str, ...] = (".txt",)) -> list[Sample]:
    paths: list[Path] = []
    for ext in exts:
        paths.extend(marker_dir.rglob(f"*{ext}"))
        paths.extend(marker_dir.rglob(f"*{ext.upper()}"))

    samples: list[Sample] = []
    for p in sorted(set(paths)):
        target = infer_target_label_from_filename(p.stem)
        if target is None:
            continue
        samples.append(Sample(path=p, label=target))

    if not samples:
        raise SystemExit(
            f"No labeled marker .txt found in {marker_dir}. Expected filename prefixes: {SOURCE_LABELS_IN_ORDER}"
        )
    return samples


def build_temporal_windows(
    feats: np.ndarray,
    ys: np.ndarray,
    paths: list[str],
    num_frames: int,
    stride: int,
    feat_agg: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build an 8-frame (or N-frame) feature vector per training example.

    We group frames by (label, clip_id) inferred from the filename, sort by timestamp,
    then take sliding windows of length num_frames.

    Returns (X, y, keys) where keys are per-window clip keys for leakage-free splitting.
    """
    num_frames = max(1, int(num_frames))
    stride = max(1, int(stride))
    feat_agg = str(feat_agg).lower().strip()
    if feat_agg not in {"concat", "mean"}:
        raise SystemExit("--feat-agg must be 'concat' or 'mean'")

    # key -> list[(time_ms, idx)]
    groups: dict[str, list[tuple[int, int]]] = {}
    for i, p in enumerate(paths):
        stem = Path(p).stem
        parsed = parse_clip_id_and_time_ms_from_path(p)
        if parsed is None:
            continue
        source_label, clip_id, t_ms = parsed
        target_label = SOURCE_TO_TARGET_LABEL.get(source_label)
        if target_label is None:
            continue
        # Keep source_label in the key to avoid mixing sequences that share clip_id across sources.
        key = f"{target_label}__{source_label}__{clip_id}"
        groups.setdefault(key, []).append((t_ms, i))

    xs: list[np.ndarray] = []
    ys_out: list[int] = []
    keys_out: list[str] = []

    for key, items in groups.items():
        items.sort(key=lambda it: it[0])
        frame_idxs = [idx for _, idx in items]
        if len(frame_idxs) < num_frames:
            continue

        # All frames in the same clip should share the same label.
        y0 = int(ys[frame_idxs[0]])

        for j in range(0, len(frame_idxs) - num_frames + 1, stride):
            win = frame_idxs[j : j + num_frames]
            f = feats[win]  # (num_frames, base_dim)
            if feat_agg == "mean":
                x = f.mean(axis=0)
            else:
                x = f.reshape(-1)
            xs.append(x.astype(np.float32, copy=False))
            ys_out.append(y0)
            keys_out.append(key)

    if not xs:
        raise SystemExit(
            "No temporal windows could be built. Expected filenames like '<label>_<clipId>_<time>ms.txt'."
        )

    X = np.stack(xs, axis=0).astype(np.float32, copy=False)
    y = np.asarray(ys_out, dtype=np.int64)
    return X, y, keys_out


def split_by_key(
    X: np.ndarray,
    y: np.ndarray,
    keys: list[str],
    seed: int,
    val_ratio: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    uniq = np.asarray(sorted(set(keys)))
    rng.shuffle(uniq)

    n_val = int(round(len(uniq) * float(val_ratio)))
    n_val = max(1, min(len(uniq) - 1, n_val)) if len(uniq) >= 2 else 0

    val_keys = set(uniq[:n_val].tolist())

    train_idx: list[int] = []
    val_idx: list[int] = []
    for i, k in enumerate(keys):
        (val_idx if k in val_keys else train_idx).append(i)

    x_train = X[train_idx]
    y_train = y[train_idx]
    x_val = X[val_idx]
    y_val = y[val_idx]

    return x_train, y_train, x_val, y_val


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    device: str,
    *,
    pose_dim: int,
    hand_dim: int,
    hand_mask_dim: int,
    proj_dim: int,
    num_frames: int,
    feat_agg: str,
    temporal_model: str,
    hidden: int,
    rnn_hidden: int,
    rnn_layers: int,
    bidirectional_rnn: bool,
    lr: float,
    epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[nn.Module, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train)
    x_val_t = torch.from_numpy(x_val)
    y_val_t = torch.from_numpy(y_val)

    temporal_model = str(temporal_model).lower().strip()
    if temporal_model == "mlp":
        model = MarkerTemporalMLP(
            pose_dim=int(pose_dim),
            hand_dim=int(hand_dim),
            hand_mask_dim=int(hand_mask_dim),
            proj_dim=int(proj_dim),
            hidden=hidden,
            num_classes=len(LABELS_IN_ORDER),
            num_frames=int(num_frames),
            feat_agg=str(feat_agg),
        ).to(device)
    elif temporal_model in {"gru", "lstm"}:
        if str(feat_agg).lower().strip() != "concat":
            raise SystemExit("Recurrent temporal models require --feat-agg=concat to preserve frame order.")
        model = MarkerTemporalRNN(
            pose_dim=int(pose_dim),
            hand_dim=int(hand_dim),
            hand_mask_dim=int(hand_mask_dim),
            proj_dim=int(proj_dim),
            hidden=hidden,
            num_classes=len(LABELS_IN_ORDER),
            num_frames=int(num_frames),
            rnn_type=temporal_model,
            rnn_hidden=int(rnn_hidden),
            rnn_layers=int(rnn_layers),
            bidirectional=bool(bidirectional_rnn),
        ).to(device)
    else:
        raise SystemExit("--temporal-model must be one of: mlp, gru, lstm")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
        pred = logits.argmax(dim=1)
        return float((pred == y).float().mean().item())

    metrics = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

    for epoch in range(1, int(epochs) + 1):
        model.train()
        perm = torch.randperm(x_train_t.shape[0])
        x_train_e = x_train_t[perm]
        y_train_e = y_train_t[perm]

        losses: list[float] = []
        accs: list[float] = []

        for i in range(0, x_train_e.shape[0], int(batch_size)):
            xb = x_train_e[i : i + int(batch_size)].to(device)
            yb = y_train_e[i : i + int(batch_size)].to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()

            losses.append(float(loss.item()))
            accs.append(accuracy(logits.detach(), yb.detach()))

        model.eval()
        with torch.inference_mode():
            if x_val_t.numel():
                val_logits = model(x_val_t.to(device))
                val_loss = float(F.cross_entropy(val_logits, y_val_t.to(device)).item())
                val_acc = accuracy(val_logits, y_val_t.to(device))
            else:
                val_loss = 0.0
                val_acc = 0.0

        train_loss = float(np.mean(losses)) if losses else 0.0
        train_acc = float(np.mean(accs)) if accs else 0.0

        metrics["train_loss"].append(train_loss)
        metrics["train_acc"].append(train_acc)
        metrics["val_loss"].append(val_loss)
        metrics["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:03d}/{epochs} | train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}"
        )

    return model, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an MLP on temporal windows of pose+hands landmark features (marker-only, e.g., 3 frames)"
    )
    parser.add_argument(
        "--marker-dir",
        default="Data-processed-landmarker",
        help="Folder containing per-image .txt with pose(165) + hands(126) vectors",
    )

    parser.add_argument("--num-frames", type=int, default=3, help="Number of frames per window")
    parser.add_argument("--stride", type=int, default=1, help="Sliding window stride (in frames)")
    parser.add_argument(
        "--feat-agg",
        default="concat",
        choices=["concat", "mean"],
        help="How to aggregate N frames into one vector: concat or mean",
    )
    parser.add_argument(
        "--temporal-model",
        default="mlp",
        choices=["mlp", "gru", "lstm"],
        help="Temporal head to train: the original MLP, a GRU, or an LSTM.",
    )

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--proj-dim", type=int, default=128, help="Per-block projection dim")
    parser.add_argument("--rnn-hidden", type=int, default=192, help="GRU/LSTM hidden size")
    parser.add_argument("--rnn-layers", type=int, default=1, help="Number of GRU/LSTM layers")
    parser.add_argument(
        "--bidirectional-rnn",
        action="store_true",
        help="Use a bidirectional GRU/LSTM temporal head.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="cpu or cuda (default: auto)")

    parser.add_argument(
        "--max-pose-nan-frac",
        type=float,
        default=0.5,
        help="Drop frames where pose NaN fraction is above this (default: 0.5).",
    )
    parser.add_argument(
        "--max-hands-nan-frac",
        type=float,
        default=0.9,
        help="Drop frames where hands NaN fraction is above this (default: 0.9).",
    )

    parser.add_argument("--out", default="mlp_3_frames_marker.pt", help="Output model checkpoint")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    marker_dir = Path(args.marker_dir)
    if not marker_dir.exists():
        raise SystemExit(f"Marker dir not found: {marker_dir}")

    samples = build_marker_samples(marker_dir)
    print(f"Found {len(samples)} labeled marker .txt in {marker_dir}")

    # Per-frame features from marker files: pose + hands + hand_mask
    feats_list: list[np.ndarray] = []
    ys_list: list[int] = []
    paths: list[str] = []

    missing_hands_any = 0
    dropped_incomplete = 0
    pose_nan_fracs: list[float] = []
    hands_nan_fracs: list[float] = []

    frame_label_counts: dict[str, int] = {k: 0 for k in LABELS_IN_ORDER}
    for s in samples:
        pose, hands = load_pose_hand_from_txt(s.path)

        pose_nan_frac = float(np.isnan(pose).mean())
        hands_nan_frac = float(np.isnan(hands).mean())
        pose_nan_fracs.append(pose_nan_frac)
        hands_nan_fracs.append(hands_nan_frac)

        # Drop incomplete skeleton frames (common cause of model collapsing to a dominant class).
        if pose_nan_frac > float(args.max_pose_nan_frac) or hands_nan_frac > float(args.max_hands_nan_frac):
            dropped_incomplete += 1
            continue

        if np.isnan(hands).any():
            missing_hands_any += 1
        combined, _hand_mask = combine_pose_hands_handmask(pose, hands)
        feats_list.append(combined)
        ys_list.append(label_to_index_0_based(s.label))
        paths.append(str(s.path))

        frame_label_counts[str(s.label)] = frame_label_counts.get(str(s.label), 0) + 1

    kept = len(paths)
    print(
        f"Filtered incomplete frames: kept={kept} dropped={dropped_incomplete} "
        f"(max_pose_nan_frac={float(args.max_pose_nan_frac):.2f} max_hands_nan_frac={float(args.max_hands_nan_frac):.2f})"
    )
    if kept == 0:
        raise SystemExit(
            "All frames were filtered out as incomplete. Relax --max-pose-nan-frac/--max-hands-nan-frac or check marker files."
        )

    if pose_nan_fracs:
        print(
            f"NaN stats (frames before filter): pose mean={float(np.mean(pose_nan_fracs)):.3f} "
            f"p95={float(np.percentile(pose_nan_fracs, 95)):.3f} | hands mean={float(np.mean(hands_nan_fracs)):.3f} "
            f"p95={float(np.percentile(hands_nan_fracs, 95)):.3f}"
        )

    print("Frame label counts (after filter):")
    for lab in LABELS_IN_ORDER:
        print(f"  {lab}: {int(frame_label_counts.get(lab, 0))}")

    feats_combined = np.stack(feats_list, axis=0).astype(np.float32, copy=False)
    ys = np.asarray(ys_list, dtype=np.int64)

    per_frame_dim = int(feats_combined.shape[1])

    print(
        f"Per-frame dims: pose={POSE_DIM} hands={HAND_DIM} hand_mask={HAND_MASK_DIM} => {per_frame_dim} | "
        f"frames_with_any_missing_hands={missing_hands_any}/{len(paths)}"
    )

    X, y, keys = build_temporal_windows(
        feats=feats_combined,
        ys=ys,
        paths=paths,
        num_frames=args.num_frames,
        stride=args.stride,
        feat_agg=args.feat_agg,
    )

    win_label_counts: dict[str, int] = {k: 0 for k in LABELS_IN_ORDER}
    for yi in y.tolist():
        win_label_counts[str(LABELS_IN_ORDER[int(yi)])] += 1
    print("Window label counts:")
    for lab in LABELS_IN_ORDER:
        print(f"  {lab}: {int(win_label_counts.get(lab, 0))}")

    x_train, y_train, x_val, y_val = split_by_key(X, y, keys, seed=args.seed, val_ratio=0.1)
    print(
        f"Temporal windows: X={X.shape} (per_frame_dim={per_frame_dim}) | train={x_train.shape[0]} val={x_val.shape[0]} | "
        f"num_frames={int(args.num_frames)} feat_agg={args.feat_agg} stride={int(args.stride)}"
    )

    model, metrics = train_model(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        device=device,
        pose_dim=POSE_DIM,
        hand_dim=HAND_DIM,
        hand_mask_dim=HAND_MASK_DIM,
        proj_dim=args.proj_dim,
        num_frames=args.num_frames,
        feat_agg=args.feat_agg,
        temporal_model=args.temporal_model,
        hidden=args.hidden,
        rnn_hidden=args.rnn_hidden,
        rnn_layers=args.rnn_layers,
        bidirectional_rnn=args.bidirectional_rnn,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    out_path = Path(args.out)
    payload = {
        "state_dict": model.state_dict(),
        "labels_in_order": LABELS_IN_ORDER,
        "label_to_number": {label: i + 1 for i, label in enumerate(LABELS_IN_ORDER)},
        "pose_dim": int(POSE_DIM),
        "hand_dim": int(HAND_DIM),
        "hand_mask_dim": int(HAND_MASK_DIM),
        "model_type": "marker_temporal_rnn" if str(args.temporal_model) in {"gru", "lstm"} else "marker_temporal_mlp",
        "temporal_model": str(args.temporal_model),
        "base_feature_dim": int(per_frame_dim),
        "feature_dim": int(X.shape[1]),
        "hidden": int(args.hidden),
        "proj_dim": int(args.proj_dim),
        "num_frames": int(args.num_frames),
        "feat_agg": str(args.feat_agg),
        "rnn_hidden": int(args.rnn_hidden),
        "rnn_layers": int(args.rnn_layers),
        "bidirectional_rnn": bool(args.bidirectional_rnn),
        "stride": int(args.stride),
        "metrics": metrics,
    }
    torch.save(payload, out_path)

    meta_path = out_path.with_suffix(out_path.suffix + ".json")
    meta_path.write_text(json.dumps({k: payload[k] for k in payload if k != "state_dict"}, indent=2))

    print(f"Saved model: {out_path}")
    print(f"Saved metadata: {meta_path}")
    print("Class numbers (output = index+1):")
    for i, label in enumerate(LABELS_IN_ORDER, start=1):
        print(f"  {i}. {label}")


if __name__ == "__main__":
    main()
