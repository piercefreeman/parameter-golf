#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import os
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn

import train_gpt as tg


MIB = 1024.0 * 1024.0
MB = 1_000_000.0


@dataclass(frozen=True)
class Variant:
    name: str
    overrides: dict[str, int]


@dataclass
class VariantResult:
    name: str
    status: str
    persistent_bytes: int
    trainable_bytes: int
    buffer_bytes: int
    grad_bytes: int
    optimizer_state_bytes: int
    peak_allocated: int | None
    peak_reserved: int | None
    analytic_peak_tensor_bytes: int


VARIANTS = (
    Variant("baseline", {"model_dim": 512, "num_heads": 8, "num_kv_heads": 4}),
    Variant("model_dim=3072_num_heads=48_num_kv_heads=24", {"model_dim": 3072, "num_heads": 48, "num_kv_heads": 24}),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single train_gpt forward/backward step and print tensor-size and CUDA memory diagnostics."
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--top", type=int, default=12, help="Number of top tensors/records to print per section.")
    parser.add_argument(
        "--variants",
        default="baseline,model_dim=3072_num_heads=48_num_kv_heads=24",
        help="Comma-separated subset of built-in variants to run.",
    )
    parser.add_argument("--compile", action="store_true", help="Wrap the model in torch.compile before running.")
    parser.add_argument(
        "--optimizer-step",
        action="store_true",
        help="Also run one optimizer step and report optimizer-state memory after the step.",
    )
    return parser.parse_args()


def fmt_bytes(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "n/a"
    return f"{num_bytes} bytes ({num_bytes / MIB:.2f} MiB, {num_bytes / MB:.3f} MB)"


def shape_text(shape: Iterable[int]) -> str:
    return "[" + ", ".join(str(dim) for dim in shape) + "]"


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def storage_key(t: Tensor) -> tuple[str, int | None, int, int]:
    storage = t.untyped_storage()
    return (t.device.type, t.device.index, int(storage.data_ptr()), int(storage.nbytes()))


def iter_tensors(obj: object) -> Iterable[Tensor]:
    if torch.is_tensor(obj):
        yield obj
        return
    if isinstance(obj, dict):
        for value in obj.values():
            yield from iter_tensors(value)
        return
    if isinstance(obj, (list, tuple)):
        for value in obj:
            yield from iter_tensors(value)


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cuda_stats(device: torch.device) -> dict[str, int] | None:
    if device.type != "cuda":
        return None
    cuda_sync(device)
    return {
        "allocated": int(torch.cuda.memory_allocated(device)),
        "reserved": int(torch.cuda.memory_reserved(device)),
        "max_allocated": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved": int(torch.cuda.max_memory_reserved(device)),
    }


def print_cuda_stats(label: str, device: torch.device) -> None:
    stats = cuda_stats(device)
    if stats is None:
        print(f"cuda:{label}: unavailable")
        return
    print(
        f"cuda:{label}: allocated={fmt_bytes(stats['allocated'])} reserved={fmt_bytes(stats['reserved'])} "
        f"peak_allocated={fmt_bytes(stats['max_allocated'])} peak_reserved={fmt_bytes(stats['max_reserved'])}"
    )


def make_args(overrides: dict[str, int]) -> tg.Hyperparameters:
    args = tg.Hyperparameters()
    for key, value in overrides.items():
        setattr(args, key, value)
    args.wandb_enabled = False
    args.warmup_steps = 0
    args.iterations = 1
    args.val_loss_every = 0
    args.train_log_every = 1
    args.max_wallclock_seconds = 0.0
    return args


def pick_device(device_flag: str) -> torch.device:
    if device_flag == "cpu":
        return torch.device("cpu")
    if device_flag == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device=cuda, but CUDA is unavailable.")
        return torch.device("cuda", 0)
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    return torch.device("cpu")


def build_model_and_optimizers(args: tg.Hyperparameters, device: torch.device, compile_model: bool) -> tuple[nn.Module, nn.Module, list[torch.optim.Optimizer]]:
    if args.linear_impl not in tg.LINEAR_IMPL_CHOICES:
        raise ValueError(f"LINEAR_IMPL must be one of {tg.LINEAR_IMPL_CHOICES}, got {args.linear_impl}")
    base_model = tg.GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        linear_impl=args.linear_impl,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_fixed_seed=args.lora_fixed_seed,
        model_seed=args.seed,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        device=device,
    )
    model: nn.Module = torch.compile(base_model, dynamic=False, fullgraph=True) if compile_model else base_model

    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        param
        for name, param in block_named_params
        if param.ndim == 2 and not any(pattern in name for pattern in tg.CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        param
        for name, param in block_named_params
        if param.ndim < 2 or any(pattern in name for pattern in tg.CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    adam_kwargs = dict(betas=(args.beta1, args.beta2), eps=args.adam_eps)
    if device.type == "cuda":
        adam_kwargs["fused"] = True
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        **adam_kwargs,
    )
    optimizer_muon = tg.Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        **adam_kwargs,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            **adam_kwargs,
        )
        optimizers.insert(1, optimizer_head)
    return base_model, model, optimizers


def persistent_tensor_rows(module: nn.Module) -> list[tuple[str, Tensor, str]]:
    rows: list[tuple[str, Tensor, str]] = []
    for name, tensor in module.named_parameters():
        rows.append((name, tensor, "param"))
    for name, tensor in module.named_buffers():
        rows.append((name, tensor, "buffer"))
    rows.sort(key=lambda row: tensor_nbytes(row[1]), reverse=True)
    return rows


def print_top_named_tensors(title: str, rows: list[tuple[str, Tensor, str]], top: int) -> None:
    print(title)
    if not rows:
        print("  <none>")
        return
    for name, tensor, kind in rows[:top]:
        print(
            f"  {kind:6s} {name:48s} dtype={str(tensor.dtype):14s} shape={shape_text(tensor.shape):20s} "
            f"size={fmt_bytes(tensor_nbytes(tensor))}"
        )


def print_analytic_tensor_sizes(args: tg.Hyperparameters, world_size: int) -> int:
    grad_accum_steps = 8 // world_size
    microbatch_tokens = args.train_batch_tokens // (world_size * grad_accum_steps)
    batch_size = microbatch_tokens // args.train_seq_len
    head_dim = args.model_dim // args.num_heads
    kv_dim = args.num_kv_heads * head_dim
    hidden = args.mlp_mult * args.model_dim
    bf16_bytes = torch.tensor([], dtype=torch.bfloat16).element_size()
    int64_bytes = torch.tensor([], dtype=torch.int64).element_size()
    rows = [
        ("input_ids", (batch_size, args.train_seq_len), int64_bytes),
        ("target_ids", (batch_size, args.train_seq_len), int64_bytes),
        ("residual_stream", (batch_size, args.train_seq_len, args.model_dim), bf16_bytes),
        ("skip_cache_total", (args.num_layers // 2, batch_size, args.train_seq_len, args.model_dim), bf16_bytes),
        ("attn_q_proj", (batch_size, args.train_seq_len, args.model_dim), bf16_bytes),
        ("attn_k_proj", (batch_size, args.train_seq_len, kv_dim), bf16_bytes),
        ("attn_v_proj", (batch_size, args.train_seq_len, kv_dim), bf16_bytes),
        ("attn_out", (batch_size, args.train_seq_len, args.model_dim), bf16_bytes),
        ("mlp_hidden", (batch_size, args.train_seq_len, hidden), bf16_bytes),
        ("logits", (batch_size * args.train_seq_len, args.vocab_size), bf16_bytes),
    ]
    print("analytic microstep tensor sizes")
    largest = 0
    for name, shape, element_size in rows:
        numel = 1
        for dim in shape:
            numel *= dim
        num_bytes = int(numel * element_size)
        largest = max(largest, num_bytes)
        print(f"  {name:18s} shape={shape_text(shape):28s} approx_size={fmt_bytes(num_bytes)}")
    print(
        f"  derived: world_size={world_size} grad_accum_steps={grad_accum_steps} "
        f"microbatch_tokens={microbatch_tokens} batch_size={batch_size} head_dim={head_dim} kv_dim={kv_dim}"
    )
    return largest


def optimizer_state_bytes(optimizers: list[torch.optim.Optimizer]) -> int:
    total = 0
    seen: set[tuple[str, int | None, int, int]] = set()
    for optimizer in optimizers:
        for state in optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    key = storage_key(value)
                    if key not in seen:
                        total += int(value.untyped_storage().nbytes())
                        seen.add(key)
    return total


def gradient_rows(module: nn.Module) -> list[tuple[str, Tensor, str]]:
    rows = [(name, param.grad, "grad") for name, param in module.named_parameters() if param.grad is not None]
    rows.sort(key=lambda row: tensor_nbytes(row[1]), reverse=True)
    return rows


def summarize_records(records: list[dict[str, object]], top: int, title: str) -> None:
    print(title)
    if not records:
        print("  <none>")
        return
    grouped: dict[tuple[object, ...], dict[str, object]] = {}
    for record in records:
        key = (record["name"], record["dtype"], tuple(record["shape"]))
        row = grouped.setdefault(
            key,
            {
                "name": record["name"],
                "module_type": record["module_type"],
                "dtype": record["dtype"],
                "shape": tuple(record["shape"]),
                "count": 0,
                "total_bytes": 0,
                "max_bytes": 0,
            },
        )
        row["count"] = int(row["count"]) + 1
        row["total_bytes"] = int(row["total_bytes"]) + int(record["bytes"])
        row["max_bytes"] = max(int(row["max_bytes"]), int(record["bytes"]))
    rows = sorted(grouped.values(), key=lambda row: (int(row["max_bytes"]), int(row["total_bytes"])), reverse=True)
    for row in rows[:top]:
        print(
            f"  {str(row['name']):48s} type={str(row['module_type']):20s} dtype={str(row['dtype']):14s} "
            f"shape={shape_text(row['shape']):24s} count={int(row['count']):2d} "
            f"max_tensor={fmt_bytes(int(row['max_bytes']))} total_seen={fmt_bytes(int(row['total_bytes']))}"
        )


def summarize_saved_tensors(records: list[dict[str, object]], top: int) -> None:
    print("saved tensors for backward")
    if not records:
        print("  <none>")
        return
    grouped: dict[tuple[object, ...], dict[str, object]] = {}
    unique_storage_bytes = 0
    unique_storage_keys: set[tuple[str, int | None, int, int]] = set()
    for record in records:
        key = (record["dtype"], tuple(record["shape"]))
        row = grouped.setdefault(
            key,
            {"dtype": record["dtype"], "shape": tuple(record["shape"]), "count": 0, "ref_bytes": 0, "storage_bytes": 0},
        )
        row["count"] = int(row["count"]) + 1
        row["ref_bytes"] = int(row["ref_bytes"]) + int(record["bytes"])
        row["storage_bytes"] = int(row["storage_bytes"]) + int(record["storage_bytes"])
        if record["storage_key"] not in unique_storage_keys:
            unique_storage_keys.add(record["storage_key"])
            unique_storage_bytes += int(record["storage_bytes"])
    rows = sorted(grouped.values(), key=lambda row: (int(row["ref_bytes"]), int(row["storage_bytes"])), reverse=True)
    print(f"  unique_saved_storage={fmt_bytes(unique_storage_bytes)} across {len(unique_storage_keys)} storages")
    for row in rows[:top]:
        print(
            f"  dtype={str(row['dtype']):14s} shape={shape_text(row['shape']):24s} count={int(row['count']):2d} "
            f"saved_refs={fmt_bytes(int(row['ref_bytes']))} storage_sum={fmt_bytes(int(row['storage_bytes']))}"
        )


def run_variant(variant: Variant, device: torch.device, top: int, compile_model: bool, optimizer_step: bool) -> VariantResult:
    args = make_args(variant.overrides)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 0 or 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE must be a positive divisor of 8, got {world_size}")
    grad_accum_steps = 8 // world_size
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    gc.collect()

    print(f"\n=== {variant.name} ===")
    print(
        f"config: layers={args.num_layers} model_dim={args.model_dim} heads={args.num_heads} kv_heads={args.num_kv_heads} "
        f"mlp_mult={args.mlp_mult} linear_impl={args.linear_impl}"
    )
    analytic_peak_tensor_bytes = print_analytic_tensor_sizes(args, world_size)
    base_model, model, optimizers = build_model_and_optimizers(args, device, compile_model)
    base_model.train()
    model.train()

    persistent_rows = persistent_tensor_rows(base_model)
    persistent_bytes = sum(tensor_nbytes(tensor) for _, tensor, _ in persistent_rows)
    trainable_bytes = sum(tensor_nbytes(param) for param in base_model.parameters() if param.requires_grad)
    buffer_bytes = sum(tensor_nbytes(buffer) for buffer in base_model.buffers())
    print(
        f"persistent tensors: total={fmt_bytes(persistent_bytes)} trainable={fmt_bytes(trainable_bytes)} "
        f"buffers={fmt_bytes(buffer_bytes)}"
    )
    print_top_named_tensors("top persistent tensors", persistent_rows, top)
    print_cuda_stats("after_model_init", device)

    loader = tg.DistributedTokenLoader(args.train_files, rank=0, world_size=world_size, device=device)
    x, y = loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
    print(
        f"batch: x shape={shape_text(x.shape)} dtype={x.dtype} size={fmt_bytes(tensor_nbytes(x))} "
        f"y shape={shape_text(y.shape)} dtype={y.dtype} size={fmt_bytes(tensor_nbytes(y))}"
    )
    print_cuda_stats("after_batch_load", device)

    activation_records: list[dict[str, object]] = []
    activation_handles = []

    def forward_hook(name: str):
        def hook(module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
            for tensor in iter_tensors(output):
                activation_records.append(
                    {
                        "name": name,
                        "module_type": type(module).__name__,
                        "dtype": str(tensor.dtype),
                        "shape": tuple(int(dim) for dim in tensor.shape),
                        "bytes": tensor_nbytes(tensor),
                    }
                )
        return hook

    hook_types = (nn.Embedding, tg.RMSNorm, tg.FixedLoRALinear, tg.CastedLinear, tg.CausalSelfAttention, tg.MLP, tg.Block)
    for name, module in base_model.named_modules():
        if name and isinstance(module, hook_types):
            activation_handles.append(module.register_forward_hook(forward_hook(name)))

    saved_tensor_records: list[dict[str, object]] = []

    def pack_hook(tensor: Tensor) -> Tensor:
        saved_tensor_records.append(
            {
                "dtype": str(tensor.dtype),
                "shape": tuple(int(dim) for dim in tensor.shape),
                "bytes": tensor_nbytes(tensor),
                "storage_bytes": int(tensor.untyped_storage().nbytes()),
                "storage_key": storage_key(tensor),
            }
        )
        return tensor

    def unpack_hook(tensor: Tensor) -> Tensor:
        return tensor

    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
        if device.type == "cuda"
        else nullcontext()
    )

    status = "ok"
    loss_value = None
    try:
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            with autocast_context:
                loss = model(x, y)
            loss_value = float(loss.item())
        print(f"forward: loss={loss_value:.6f}")
        print_cuda_stats("after_forward", device)
        loss.backward()
        print_cuda_stats("after_backward", device)
        if optimizer_step:
            for optimizer in optimizers:
                optimizer.step()
            print_cuda_stats("after_optimizer_step", device)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        status = f"oom: {exc}"
        print(f"runtime_status: {status}")
        print_cuda_stats("at_oom", device)
    finally:
        for handle in activation_handles:
            handle.remove()

    summarize_records(activation_records, top, "top module outputs seen during forward")
    summarize_saved_tensors(saved_tensor_records, top)

    grad_rows = gradient_rows(base_model)
    grad_bytes = sum(tensor_nbytes(tensor) for _, tensor, _ in grad_rows)
    print(f"gradient tensors: total={fmt_bytes(grad_bytes)}")
    print_top_named_tensors("top gradients", grad_rows, top)

    opt_state_bytes = optimizer_state_bytes(optimizers)
    if optimizer_step:
        print(f"optimizer state bytes after step: {fmt_bytes(opt_state_bytes)}")
    else:
        print(f"optimizer state bytes before any step: {fmt_bytes(opt_state_bytes)}")

    stats = cuda_stats(device)
    peak_allocated = None if stats is None else stats["max_allocated"]
    peak_reserved = None if stats is None else stats["max_reserved"]

    del x, y, base_model, model, optimizers
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return VariantResult(
        name=variant.name,
        status=status,
        persistent_bytes=persistent_bytes,
        trainable_bytes=trainable_bytes,
        buffer_bytes=buffer_bytes,
        grad_bytes=grad_bytes,
        optimizer_state_bytes=opt_state_bytes,
        peak_allocated=peak_allocated,
        peak_reserved=peak_reserved,
        analytic_peak_tensor_bytes=analytic_peak_tensor_bytes,
    )


def print_compare(results: list[VariantResult]) -> None:
    if len(results) < 2:
        return
    print("\n=== comparison ===")
    baseline = results[0]
    for result in results[1:]:
        print(f"{baseline.name} -> {result.name}")
        for label, left, right in (
            ("persistent_bytes", baseline.persistent_bytes, result.persistent_bytes),
            ("trainable_bytes", baseline.trainable_bytes, result.trainable_bytes),
            ("buffer_bytes", baseline.buffer_bytes, result.buffer_bytes),
            ("grad_bytes", baseline.grad_bytes, result.grad_bytes),
            ("optimizer_state_bytes", baseline.optimizer_state_bytes, result.optimizer_state_bytes),
            ("analytic_peak_tensor_bytes", baseline.analytic_peak_tensor_bytes, result.analytic_peak_tensor_bytes),
        ):
            ratio = float("inf") if left == 0 and right > 0 else (right / left if left else 1.0)
            print(f"  {label:26s} {fmt_bytes(left)} -> {fmt_bytes(right)} ratio={ratio:.2f}x")
        if baseline.peak_allocated is not None and result.peak_allocated is not None:
            ratio = float("inf") if baseline.peak_allocated == 0 and result.peak_allocated > 0 else (
                result.peak_allocated / baseline.peak_allocated if baseline.peak_allocated else 1.0
            )
            print(
                f"  peak_allocated             {fmt_bytes(baseline.peak_allocated)} -> "
                f"{fmt_bytes(result.peak_allocated)} ratio={ratio:.2f}x"
            )
        if baseline.peak_reserved is not None and result.peak_reserved is not None:
            ratio = float("inf") if baseline.peak_reserved == 0 and result.peak_reserved > 0 else (
                result.peak_reserved / baseline.peak_reserved if baseline.peak_reserved else 1.0
            )
            print(
                f"  peak_reserved              {fmt_bytes(baseline.peak_reserved)} -> "
                f"{fmt_bytes(result.peak_reserved)} ratio={ratio:.2f}x"
            )
        print(f"  status                     {baseline.status} -> {result.status}")


def main() -> None:
    parsed = parse_args()
    device = pick_device(parsed.device)
    selected = {name.strip() for name in parsed.variants.split(",") if name.strip()}
    variants = [variant for variant in VARIANTS if variant.name in selected]
    if not variants:
        valid = ", ".join(variant.name for variant in VARIANTS)
        raise ValueError(f"No valid variants selected. Choose from: {valid}")

    print(f"python={os.sys.version.split()[0]} torch={torch.__version__} device={device}")
    print(f"train_files={tg.Hyperparameters.train_files}")
    print(f"train_seq_len={tg.Hyperparameters.train_seq_len} train_batch_tokens={tg.Hyperparameters.train_batch_tokens}")
    if device.type != "cuda":
        print("CUDA is unavailable, so this run will still print tensor sizes but not real GPU allocation stats.")

    results = [run_variant(variant, device, parsed.top, parsed.compile, parsed.optimizer_step) for variant in variants]
    print_compare(results)


if __name__ == "__main__":
    main()
