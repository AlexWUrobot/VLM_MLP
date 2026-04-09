import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


LABELS_IN_ORDER = [
    "come",
    "idle_back",
    "idle_front",
    "none",
    "phone",
    "play_phone",
    "stop",
    "wave",
]


def label_to_index_0_based(label: str) -> int:
    return LABELS_IN_ORDER.index(label)


def index_to_number_1_based(index0: int) -> int:
    return index0 + 1


def infer_label_from_filename(name: str) -> str | None:
    # Prefer longest labels first to avoid matching 'phone' before 'play_phone'
    for label in sorted(LABELS_IN_ORDER, key=len, reverse=True):
        if name == label or name.startswith(label + "_"):
            return label
    return None


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


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        return self.fc3(x)


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


def train_mlp(
    feats: np.ndarray,
    ys: np.ndarray,
    device: str,
    hidden: int,
    lr: float,
    epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[MLP, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = feats.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)

    split = int(round(n * 0.9))
    train_idx = idx[:split]
    val_idx = idx[split:]

    x_train = torch.from_numpy(feats[train_idx])
    y_train = torch.from_numpy(ys[train_idx])
    x_val = torch.from_numpy(feats[val_idx])
    y_val = torch.from_numpy(ys[val_idx])

    model = MLP(in_dim=feats.shape[1], hidden=hidden, num_classes=len(LABELS_IN_ORDER)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
        pred = logits.argmax(dim=1)
        return float((pred == y).float().mean().item())

    metrics = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(x_train.shape[0])
        x_train_e = x_train[perm]
        y_train_e = y_train[perm]

        losses = []
        accs = []
        for i in range(0, x_train_e.shape[0], batch_size):
            xb = x_train_e[i : i + batch_size].to(device)
            yb = y_train_e[i : i + batch_size].to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()

            losses.append(float(loss.item()))
            accs.append(accuracy(logits.detach(), yb.detach()))

        model.eval()
        with torch.inference_mode():
            val_logits = model(x_val.to(device)) if x_val.numel() else torch.empty((0, len(LABELS_IN_ORDER)))
            val_loss = (
                float(F.cross_entropy(val_logits, y_val.to(device)).item()) if x_val.numel() else 0.0
            )
            val_acc = accuracy(val_logits, y_val.to(device)) if x_val.numel() else 0.0

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
    parser = argparse.ArgumentParser(description="Train an MLP on CLIP-B features from Data-proccessed images")
    parser.add_argument("--data-dir", default="Data-proccessed", help="Folder containing labeled JPGs")
    parser.add_argument("--cache", default="clip_b_features.npz", help="Feature cache filename")
    parser.add_argument("--no-cache", action="store_true", help="Disable feature caching")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    parser.add_argument("--out", default="mlp_clip_b.pt", help="Output model checkpoint")
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
        print(f"Loaded cached features: {cache_path} | feats={feats.shape}")
    else:
        feats, ys, paths = extract_clip_features(samples, batch_size=args.batch_size, device=device)
        print(f"Extracted features: feats={feats.shape}")
        if not args.no_cache:
            save_cache(cache_path, feats, ys, paths)
            print(f"Saved feature cache: {cache_path}")

    model, metrics = train_mlp(
        feats=feats,
        ys=ys,
        device=device,
        hidden=args.hidden,
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
        "clip_model": "ViT-B-32",
        "clip_pretrained": "openai",
        "feature_dim": int(feats.shape[1]),
        "hidden": int(args.hidden),
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
