import argparse
import gc
from collections import Counter

import torch


def bytes_to_gib(num_bytes: int) -> float:
    return num_bytes / (1024**3)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def param_storage_bytes(model: torch.nn.Module) -> int:
    # Actual parameter storage in current dtype(s)
    return sum(p.numel() * p.element_size() for p in model.parameters())


def dtype_breakdown(model: torch.nn.Module) -> str:
    dtypes = Counter(p.dtype for p in model.parameters())
    # Sort by bytes-per-element then name for stable output
    items = sorted(dtypes.items(), key=lambda kv: (torch.tensor([], dtype=kv[0]).element_size(), str(kv[0])))
    return ", ".join(f"{dt} x{cnt}" for dt, cnt in items)


def cuda_mem() -> tuple[int, int]:
    # (allocated, reserved)
    return torch.cuda.memory_allocated(), torch.cuda.memory_reserved()


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def load_clip(model_name: str, device: str, fp16: bool):
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained="openai")
    model.eval()
    model.to(device)
    if fp16 and device.startswith("cuda"):
        model.half()
    return model


def load_qwen(model_name: str, device: str):
    from transformers import AutoModel

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModel.from_pretrained(model_name, dtype=dtype)
    model.eval()
    model.to(device)
    return model


def report_one(name: str, loader_fn):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cleanup_cuda()
    if device == "cuda":
        before_alloc, before_res = cuda_mem()
    else:
        before_alloc, before_res = 0, 0

    model = loader_fn()

    if device == "cuda":
        torch.cuda.synchronize()
        after_alloc, after_res = cuda_mem()
        peak_alloc = torch.cuda.max_memory_allocated()
    else:
        after_alloc, after_res, peak_alloc = 0, 0, 0

    n_params = count_params(model)
    weight_bytes = param_storage_bytes(model)

    print(f"\n=== {name} ===")
    print(f"device: {device}")
    print(f"#params: {n_params:,}")
    print(f"param dtypes: {dtype_breakdown(model)}")
    print(f"weights (params only): {bytes_to_gib(weight_bytes):.3f} GiB")

    if device == "cuda":
        print(f"cuda allocated Δ (load): {bytes_to_gib(after_alloc - before_alloc):.3f} GiB")
        print(f"cuda reserved  Δ (load): {bytes_to_gib(after_res - before_res):.3f} GiB")
        print(f"cuda peak allocated (since reset): {bytes_to_gib(peak_alloc):.3f} GiB")

    # Free ASAP before next model
    del model
    cleanup_cuda()


def main():
    parser = argparse.ArgumentParser(description="Print param counts/dtypes and rough GPU memory usage for CLIP and Qwen2-VL models.")
    parser.add_argument("--fp16-clip", action="store_true", help="Cast CLIP models to fp16 on CUDA before measuring.")
    parser.add_argument("--qwen-model", default="Qwen/Qwen2-VL-2B-Instruct", help="HF model id for Qwen2-VL.")
    args = parser.parse_args()

    report_one(
        "CLIP ViT-B-32 (openai)",
        lambda: load_clip("ViT-B-32", device="cuda" if torch.cuda.is_available() else "cpu", fp16=args.fp16_clip),
    )
    report_one(
        "CLIP ViT-L-14 (openai)",
        lambda: load_clip("ViT-L-14", device="cuda" if torch.cuda.is_available() else "cpu", fp16=args.fp16_clip),
    )
    report_one(
        f"Qwen2-VL ({args.qwen_model})",
        lambda: load_qwen(args.qwen_model, device="cuda" if torch.cuda.is_available() else "cpu"),
    )


if __name__ == "__main__":
    main()
