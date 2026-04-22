import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ── Source labels (filenames on disk) ──
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

# ── Merged 6-class labels (what the model predicts) ──
LABELS_IN_ORDER = [
    "come",
    "idle",
    "phone",
    "play_phone",
    "stop",
    "wave",
]

_MERGE_TO_IDLE = {"idle_back", "idle_front", "none"}

_ALL_FILENAME_LABELS = sorted(
    list(LABELS_IN_ORDER) + list(_MERGE_TO_IDLE),
    key=len, reverse=True,
)


def label_to_index_0_based(label: str) -> int:
    return LABELS_IN_ORDER.index(label)


def infer_label_from_filename(name: str) -> str | None:
    """Recognise old 8-class AND new 6-class filenames, merging idle variants."""
    for label in _ALL_FILENAME_LABELS:
        if name == label or name.startswith(label + "_"):
            if label in _MERGE_TO_IDLE:
                return "idle"
            return label
    return None


def parse_clip_id_and_time_ms_from_path(path_str: str) -> tuple[int, int] | None:
    """Parse filenames like '<label>_<clipId>_<time>ms.jpg'.

    Returns (clip_id, time_ms) or None if the pattern doesn't match.
    """
    stem = Path(path_str).stem
    label = infer_label_from_filename(stem)
    if label is None:
        return None

    rest = stem[len(label) :]
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

    return clip_id, t_ms


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str


class ImageDataset(Dataset):
    def __init__(self, samples: list[Sample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample.path).convert("RGB")
        y = label_to_index_0_based(sample.label)
        return image, y, str(sample.path)


class FusionTemporalMLP(nn.Module):
    def __init__(
        self,
        *,
        clip_dim: int,
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

        self.clip_dim = int(clip_dim)
        self.pose_dim = int(pose_dim)
        self.hand_dim = int(hand_dim)
        self.hand_mask_dim = int(hand_mask_dim)
        self.per_frame_dim = self.clip_dim + self.pose_dim + self.hand_dim + self.hand_mask_dim

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

        self.clip_proj = _proj(self.clip_dim)
        self.pose_proj = _proj(self.pose_dim)
        self.hand_proj = _proj(self.hand_dim)
        self.hand_mask_proj = _proj(self.hand_mask_dim)

        self.frame_embed_dim = self.proj_dim * 4

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

        clip = xf[:, :, : self.clip_dim]
        pose = xf[:, :, self.clip_dim : self.clip_dim + self.pose_dim]
        hands = xf[:, :, self.clip_dim + self.pose_dim : self.clip_dim + self.pose_dim + self.hand_dim]
        hmask = xf[:, :, -self.hand_mask_dim :]

        # project each block per frame
        b, t, _ = xf.shape
        clip_e = self.clip_proj(clip.reshape(b * t, self.clip_dim))
        pose_e = self.pose_proj(pose.reshape(b * t, self.pose_dim))
        hand_e = self.hand_proj(hands.reshape(b * t, self.hand_dim))
        mask_e = self.hand_mask_proj(hmask.reshape(b * t, self.hand_mask_dim))

        frame_e = torch.cat([clip_e, pose_e, hand_e, mask_e], dim=1).view(b, t, self.frame_embed_dim)

        if self.feat_agg == "concat":
            h = frame_e.reshape(b, t * self.frame_embed_dim)
        else:
            h = frame_e.mean(dim=1)

        h = F.relu(self.fc1(h))
        h = F.dropout(h, p=0.2, training=self.training)
        h = F.relu(self.fc2(h))
        h = F.dropout(h, p=0.2, training=self.training)
        return self.fc3(h)


class FusionTemporalGRU(nn.Module):
    """GRU variant of the fusion model — processes frame sequences with temporal awareness.

    Each frame is projected per-modality (CLIP / pose / hands / hand_mask) the same
    way as FusionTemporalMLP, then the sequence of frame embeddings is fed into a
    GRU so the model can learn temporal ordering and motion patterns.
    """

    def __init__(
        self,
        *,
        clip_dim: int,
        pose_dim: int,
        hand_dim: int,
        hand_mask_dim: int,
        proj_dim: int,
        hidden: int,
        num_classes: int,
        num_frames: int,
        rnn_hidden: int = 192,
        rnn_layers: int = 1,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.clip_dim = int(clip_dim)
        self.pose_dim = int(pose_dim)
        self.hand_dim = int(hand_dim)
        self.hand_mask_dim = int(hand_mask_dim)
        self.per_frame_dim = self.clip_dim + self.pose_dim + self.hand_dim + self.hand_mask_dim
        self.num_frames = max(1, int(num_frames))

        self.proj_dim = int(proj_dim)
        self.rnn_hidden = int(rnn_hidden)
        self.rnn_layers = max(1, int(rnn_layers))
        self.bidirectional = bool(bidirectional)

        def _proj(in_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(int(in_dim), self.proj_dim),
                nn.LayerNorm(self.proj_dim),
                nn.ReLU(),
            )

        self.clip_proj = _proj(self.clip_dim)
        self.pose_proj = _proj(self.pose_dim)
        self.hand_proj = _proj(self.hand_dim)
        self.hand_mask_proj = _proj(self.hand_mask_dim)
        self.frame_embed_dim = self.proj_dim * 4

        self.gru = nn.GRU(
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

        clip = xf[:, :, : self.clip_dim]
        pose = xf[:, :, self.clip_dim : self.clip_dim + self.pose_dim]
        hands = xf[:, :, self.clip_dim + self.pose_dim : self.clip_dim + self.pose_dim + self.hand_dim]
        hmask = xf[:, :, -self.hand_mask_dim :]

        b, t, _ = xf.shape
        clip_e = self.clip_proj(clip.reshape(b * t, self.clip_dim))
        pose_e = self.pose_proj(pose.reshape(b * t, self.pose_dim))
        hand_e = self.hand_proj(hands.reshape(b * t, self.hand_dim))
        mask_e = self.hand_mask_proj(hmask.reshape(b * t, self.hand_mask_dim))
        frame_e = torch.cat([clip_e, pose_e, hand_e, mask_e], dim=1).view(b, t, self.frame_embed_dim)

        out, hidden = self.gru(frame_e)

        if self.bidirectional:
            last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            last_hidden = hidden[-1]

        h = F.relu(self.fc1(last_hidden))
        h = F.dropout(h, p=0.2, training=self.training)
        h = F.relu(self.fc2(h))
        h = F.dropout(h, p=0.2, training=self.training)
        return self.fc3(h)


def build_samples(data_dir: Path, exts: tuple[str, ...] = (".jpg", ".jpeg", ".png")) -> list[Sample]:
    paths: list[Path] = []
    for ext in exts:
        paths.extend(data_dir.rglob(f"*{ext}"))
        paths.extend(data_dir.rglob(f"*{ext.upper()}"))

    samples: list[Sample] = []
    for p in sorted(set(paths)):
        label = infer_label_from_filename(p.stem)
        if label is None:
            continue
        samples.append(Sample(path=p, label=label))

    if not samples:
        raise SystemExit(
            f"No labeled images found in {data_dir}. Expected filename prefixes: {LABELS_IN_ORDER}"
        )
    return samples


@torch.inference_mode()
def extract_clip_features(
    samples: list[Sample],
    batch_size: int,
    device: str,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()

    dataset = ImageDataset(samples)

    def collate(batch):
        images, ys, paths = zip(*batch)
        images_t = torch.stack([preprocess(img) for img in images], dim=0)
        ys_t = torch.tensor(ys, dtype=torch.long)
        return images_t, ys_t, list(paths)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    feats_list: list[torch.Tensor] = []
    ys_list: list[torch.Tensor] = []
    paths_out: list[str] = []

    for images_t, ys_t, paths in tqdm(loader, desc="Extracting CLIP features"):
        images_t = images_t.to(device)
        feats = model.encode_image(images_t)
        feats = feats.float()
        feats = F.normalize(feats, dim=-1)
        feats_list.append(feats.cpu())
        ys_list.append(ys_t)
        paths_out.extend(paths)

    feats_all = torch.cat(feats_list, dim=0).numpy().astype(np.float32)
    ys_all = torch.cat(ys_list, dim=0).numpy().astype(np.int64)
    return feats_all, ys_all, paths_out


def save_cache(cache_path: Path, feats: np.ndarray, ys: np.ndarray, paths: list[str]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, feats=feats, ys=ys, paths=np.array(paths, dtype=object))


def load_cache(cache_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    data = np.load(cache_path, allow_pickle=True)
    feats = data["feats"]
    ys = data["ys"]
    paths = data["paths"].tolist()
    return feats, ys, paths


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


def combine_clip_pose_hands_handmask(
    clip_feats: np.ndarray,
    pose: np.ndarray,
    hands: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (combined_feats, hand_missing_mask) for a single frame.

    Per-frame feature is:
      clip(512) + pose(165) + hands(126) + hand_mask(126)

    Missing values (NaN) are imputed with 0.
    Mask is ONLY for hands (126 dims): 1 where hand value was missing, else 0.

    No manual weighting here; the model learns per-block mixing.
    """
    clip = clip_feats.astype(np.float32, copy=False)
    pose_imp = np.nan_to_num(pose.astype(np.float32, copy=False), nan=0.0)

    hand_mask = np.isnan(hands).astype(np.float32)
    hands_imp = np.nan_to_num(hands.astype(np.float32, copy=False), nan=0.0)

    combined = np.concatenate([clip, pose_imp, hands_imp, hand_mask], axis=0).astype(np.float32, copy=False)
    return combined, hand_mask


def build_temporal_windows(
    feats: np.ndarray,
    ys: np.ndarray,
    paths: list[str],
    num_frames: int,
    stride: int,
    feat_agg: str,
    use_velocity: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build an N-frame feature vector per training example.

    When use_velocity=True AND feat_agg='concat', appends (N-1) frame-difference
    vectors (landmark velocity) so the model gets explicit motion signals.
    This is critical for distinguishing come/wave (dynamic) from stop/idle (static).

    Output dim per window:
      feat_agg='concat', use_velocity=False:  N * D
      feat_agg='concat', use_velocity=True:   N * D + (N-1) * D = (2N-1) * D
      feat_agg='mean':                        D  (velocity ignored)
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
        label = infer_label_from_filename(stem)
        if label is None:
            continue
        parsed = parse_clip_id_and_time_ms_from_path(p)
        if parsed is None:
            continue
        clip_id, t_ms = parsed
        key = f"{label}__{clip_id}"
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
                parts = [f.reshape(-1)]
                if use_velocity and num_frames > 1:
                    diffs = f[1:] - f[:-1]  # (N-1, D) — landmark velocity
                    parts.append(diffs.reshape(-1))
                x = np.concatenate(parts)
            xs.append(x.astype(np.float32, copy=False))
            ys_out.append(y0)
            keys_out.append(key)

    if not xs:
        raise SystemExit(
            "No temporal windows could be built. Expected filenames like '<label>_<clipId>_<time>ms.jpg'."
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
    clip_dim: int,
    pose_dim: int,
    hand_dim: int,
    hand_mask_dim: int,
    proj_dim: int,
    num_frames: int,
    feat_agg: str,
    hidden: int,
    lr: float,
    epochs: int,
    batch_size: int,
    seed: int,
    temporal_model: str = "mlp",
    rnn_hidden: int = 192,
    rnn_layers: int = 1,
    bidirectional: bool = False,
) -> tuple[nn.Module, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train)
    x_val_t = torch.from_numpy(x_val)
    y_val_t = torch.from_numpy(y_val)

    temporal_model = str(temporal_model).lower().strip()
    if temporal_model == "gru":
        if str(feat_agg).lower().strip() != "concat":
            raise SystemExit("GRU requires --feat-agg=concat to preserve frame order.")
        model = FusionTemporalGRU(
            clip_dim=int(clip_dim),
            pose_dim=int(pose_dim),
            hand_dim=int(hand_dim),
            hand_mask_dim=int(hand_mask_dim),
            proj_dim=int(proj_dim),
            hidden=hidden,
            num_classes=len(LABELS_IN_ORDER),
            num_frames=int(num_frames),
            rnn_hidden=int(rnn_hidden),
            rnn_layers=int(rnn_layers),
            bidirectional=bool(bidirectional),
        ).to(device)
    else:
        model = FusionTemporalMLP(
            clip_dim=int(clip_dim),
            pose_dim=int(pose_dim),
            hand_dim=int(hand_dim),
            hand_mask_dim=int(hand_mask_dim),
            proj_dim=int(proj_dim),
            hidden=hidden,
            num_classes=len(LABELS_IN_ORDER),
            num_frames=int(num_frames),
            feat_agg=str(feat_agg),
        ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    # Sqrt-inverse-frequency class weights: boosts rare classes without
    # crushing well-represented ones.
    class_counts = np.bincount(y_train, minlength=len(LABELS_IN_ORDER)).astype(np.float64)
    class_counts = np.maximum(class_counts, 1.0)
    inv_freq = 1.0 / np.sqrt(class_counts)
    class_weights = torch.tensor(inv_freq / inv_freq.sum() * len(LABELS_IN_ORDER), dtype=torch.float32).to(device)
    print(f"[class weights] {dict(zip(LABELS_IN_ORDER, [f'{w:.3f}' for w in class_weights.tolist()]))}")
    print(f"[class counts ] {dict(zip(LABELS_IN_ORDER, [int(c) for c in class_counts.tolist()]))}")

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
            loss = F.cross_entropy(logits, yb, weight=class_weights)
            loss.backward()
            opt.step()

            losses.append(float(loss.item()))
            accs.append(accuracy(logits.detach(), yb.detach()))

        model.eval()
        with torch.inference_mode():
            if x_val_t.numel():
                val_logits = model(x_val_t.to(device))
                val_loss = float(F.cross_entropy(val_logits, y_val_t.to(device), weight=class_weights).item())
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
        description="Train an MLP on temporal windows of CLIP-B + pose/hands landmark features (e.g., 8 frames)"
    )
    parser.add_argument("--data-dir", default="Data-proccessed", help="Folder containing labeled JPGs")
    parser.add_argument(
        "--marker-dir",
        default="Data-processed-landmarker",
        help="Folder containing per-image .txt with pose(165) + hands(126) vectors",
    )
    parser.add_argument("--cache", default="clip_b_features.npz", help="Per-frame CLIP feature cache filename")
    parser.add_argument("--no-cache", action="store_true", help="Disable feature caching")

    parser.add_argument("--num-frames", type=int, default=8, help="Number of frames per window")
    parser.add_argument("--stride", type=int, default=1, help="Sliding window stride (in frames)")
    parser.add_argument(
        "--feat-agg",
        default="concat",
        choices=["concat", "mean"],
        help="How to aggregate N frames into one vector: concat or mean",
    )

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--proj-dim", type=int, default=128, help="Per-block projection dim")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="cpu or cuda (default: auto)")

    parser.add_argument(
        "--temporal-model",
        default="mlp",
        choices=["mlp", "gru"],
        help="Temporal head: MLP (flat concat) or GRU (sequence-aware). Default: mlp.",
    )
    parser.add_argument("--rnn-hidden", type=int, default=192, help="GRU hidden size")
    parser.add_argument("--rnn-layers", type=int, default=1, help="Number of GRU layers")
    parser.add_argument("--bidirectional", action="store_true", help="Use a bidirectional GRU.")
    parser.add_argument(
        "--use-velocity",
        action="store_true",
        help="Append inter-frame difference (velocity) features. Critical for come vs stop.",
    )

    parser.add_argument("--out", default="mlp_clip_b_8frames_marker.pt", help="Output model checkpoint")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Data dir not found: {data_dir}")

    samples = build_samples(data_dir)
    print(f"Found {len(samples)} labeled images in {data_dir}")

    cache_path = Path(args.cache)
    if not cache_path.is_absolute():
        cache_path = data_dir / cache_path

    if (not args.no_cache) and cache_path.exists():
        feats, ys, paths = load_cache(cache_path)
        print(f"Loaded cached per-frame features: {cache_path} | feats={feats.shape}")
        # Re-derive labels from filenames to handle 8->6 class merge
        ys_new = []
        for p in paths:
            label = infer_label_from_filename(Path(p).stem)
            ys_new.append(label_to_index_0_based(label) if label else 0)
        ys = np.asarray(ys_new, dtype=np.int64)
    else:
        feats, ys, paths = extract_clip_features(samples, batch_size=args.batch_size, device=device)
        print(f"Extracted per-frame features: feats={feats.shape}")
        if not args.no_cache:
            save_cache(cache_path, feats, ys, paths)
            print(f"Saved per-frame feature cache: {cache_path}")

    clip_dim = int(feats.shape[1])

    marker_dir = Path(args.marker_dir)
    if not marker_dir.exists():
        raise SystemExit(f"Marker dir not found: {marker_dir}")

    # Build per-frame combined features: clip + pose + hands + hand_mask
    combined_list: list[np.ndarray] = []
    missing_hands_any = 0


    for i, p in enumerate(paths):
        img_path = Path(p)
        try:
            rel = img_path.relative_to(data_dir)
        except Exception:
            rel = img_path.name  # best-effort
            rel = Path(rel)

        txt_path = (marker_dir / rel).with_suffix(".txt")
        pose, hands = load_pose_hand_from_txt(txt_path)

        if np.isnan(hands).any():
            missing_hands_any += 1

        combined, _hand_mask = combine_clip_pose_hands_handmask(feats[i], pose, hands)
        combined_list.append(combined)

    feats_combined = np.stack(combined_list, axis=0).astype(np.float32, copy=False)
    per_frame_dim = int(feats_combined.shape[1])

    print(
        f"Per-frame dims: clip={clip_dim} pose={POSE_DIM} hands={HAND_DIM} hand_mask={HAND_MASK_DIM} => {per_frame_dim} | "
        f"frames_with_any_missing_hands={missing_hands_any}/{len(paths)}"
    )

    # GRU processes raw frame sequences; velocity only used for MLP
    use_velocity = bool(args.use_velocity) and (args.temporal_model != "gru")
    if args.temporal_model == "gru" and args.feat_agg != "concat":
        raise SystemExit("GRU requires --feat-agg=concat to preserve frame order.")

    X, y, keys = build_temporal_windows(
        feats=feats_combined,
        ys=ys,
        paths=paths,
        num_frames=args.num_frames,
        stride=args.stride,
        feat_agg=args.feat_agg,
        use_velocity=use_velocity,
    )

    x_train, y_train, x_val, y_val = split_by_key(X, y, keys, seed=args.seed, val_ratio=0.1)
    print(
        f"Temporal windows: X={X.shape} (per_frame_dim={per_frame_dim}, velocity={use_velocity}) | "
        f"train={x_train.shape[0]} val={x_val.shape[0]} | "
        f"num_frames={int(args.num_frames)} feat_agg={args.feat_agg} stride={int(args.stride)} "
        f"temporal_model={args.temporal_model}"
    )

    model, metrics = train_model(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        device=device,
        clip_dim=clip_dim,
        pose_dim=POSE_DIM,
        hand_dim=HAND_DIM,
        hand_mask_dim=HAND_MASK_DIM,
        proj_dim=args.proj_dim,
        num_frames=args.num_frames,
        feat_agg=args.feat_agg,
        hidden=args.hidden,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        temporal_model=args.temporal_model,
        rnn_hidden=args.rnn_hidden,
        rnn_layers=args.rnn_layers,
        bidirectional=args.bidirectional,
    )

    out_path = Path(args.out)
    payload = {
        "state_dict": model.state_dict(),
        "labels_in_order": LABELS_IN_ORDER,
        "label_to_number": {label: i + 1 for i, label in enumerate(LABELS_IN_ORDER)},
        "clip_model": "ViT-B-32",
        "clip_pretrained": "openai",
        "clip_feature_dim": clip_dim,
        "pose_dim": int(POSE_DIM),
        "hand_dim": int(HAND_DIM),
        "hand_mask_dim": int(HAND_MASK_DIM),
        "model_type": str(args.temporal_model),
        "base_feature_dim": int(per_frame_dim),
        "feature_dim": int(X.shape[1]),
        "hidden": int(args.hidden),
        "proj_dim": int(args.proj_dim),
        "num_frames": int(args.num_frames),
        "feat_agg": str(args.feat_agg),
        "use_velocity": use_velocity,
        "stride": int(args.stride),
        "temporal_model": str(args.temporal_model),
        "rnn_hidden": int(args.rnn_hidden),
        "rnn_layers": int(args.rnn_layers),
        "bidirectional": bool(args.bidirectional),
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
