from __future__ import annotations

import gc
import os
from typing import List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.pytorch_utils import Conv1D

from sampledkd.utils import _bnb_triton_available


def _cleanup_cuda(device_idx: int) -> None:
    """Aggressively free caches on the target CUDA device before loading."""
    if not torch.cuda.is_available():
        return
    try:
        prev = torch.cuda.current_device()
    except Exception:
        prev = None
    try:
        torch.cuda.set_device(device_idx)
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()
    except Exception:
        pass
    finally:
        if prev is not None:
            try:
                torch.cuda.set_device(prev)
            except Exception:
                pass
    gc.collect()


def load_teacher_8bit_strict(
    model_name: str,
    prefer_gpus: List[int],
    student_gpu: Optional[int],
) -> Tuple[torch.nn.Module, AutoTokenizer, torch.device]:
    """
    Strict 8-bit load on a single GPU, no CPU offload allowed.
    Falls back to an internal FP16→8bit conversion path if the CUDA allocator
    triggers the known bitsandbytes expandable-segment assertion.
    """
    if not _bnb_triton_available():
        raise RuntimeError("bitsandbytes/triton not available; cannot load teacher in 8-bit.")
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.6",
    )

    # Environment nudges for bnb + CUDA 11.8 on these nodes
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if "/usr/local/cuda/lib64" not in ld:
        os.environ["LD_LIBRARY_PATH"] = ld + (":" if ld else "") + "/usr/local/cuda/lib64"
    # Helps bnb pick the right kernels on older drivers
    os.environ.setdefault("BNB_CUDA_VERSION", "118")

    teacher_gpus = [g for g in prefer_gpus if g != student_gpu]
    if not teacher_gpus:
        raise RuntimeError("No GPUs available for teacher after excluding the student GPU.")
    gpu = teacher_gpus[0]

    tok = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        trust_remote_code=True,
        local_files_only=False,
    )
    if tok.pad_token_id is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    q8 = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    _cleanup_cuda(gpu)
    print(
        f"[teacher] Loading STRICT 8-bit on cuda:{gpu} "
        f"(cpu_offload={q8.llm_int8_enable_fp32_cpu_offload})",
        flush=True,
    )
    try:
        teacher = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=q8,
            device_map={"": gpu},          # pin every module to this GPU
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        load_mode = "direct-8bit"
    except Exception as exc:
        message = str(exc).lower()
        if "expandable_segment" in message or "cudaalloc" in message or "out of memory" in message:
            print(
                "[teacher] Detected allocator failure during direct 8-bit load; "
                "falling back to FP16→8-bit streaming conversion.",
                flush=True,
            )
            try:
                teacher = _load_teacher_fp16_then_quantize(
                    model_name=model_name,
                    device=torch.device(f"cuda:{gpu}"),
                    quant_config=q8,
                )
                load_mode = "manual-8bit"
            except Exception as manual_exc:
                raise RuntimeError(
                    f"Manual 8-bit conversion failed on cuda:{gpu}. Model={model_name}."
                ) from manual_exc
        else:
            raise RuntimeError(
                f"Strict 8-bit teacher load failed on cuda:{gpu}. Model={model_name}."
            ) from exc

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Sanity: ensure no layer got sharded/offloaded
    devmap = getattr(teacher, "hf_device_map", None)
    if isinstance(devmap, dict):
        bad = [k for k, v in devmap.items() if isinstance(v, str) and ("cpu" in v or "disk" in v)]
        if bad:
            raise RuntimeError(f"Unexpected CPU/offloaded modules in device_map: {bad}")

    print(f"[teacher] Teacher load completed via {load_mode}.", flush=True)
    return teacher, tok, torch.device(f"cuda:{gpu}")


def _convert_linear_module_to_int8(
    module: torch.nn.Module,
    quant_config: BitsAndBytesConfig,
    quant_device: torch.device,
) -> torch.nn.Module:
    import bitsandbytes as bnb

    has_fp16_weights = quant_config.llm_int8_has_fp16_weight
    threshold = quant_config.llm_int8_threshold

    if isinstance(module, torch.nn.Linear):
        in_features = module.in_features
        out_features = module.out_features
        weight = module.weight.detach()
    elif isinstance(module, Conv1D):
        # Conv1D stores weights as (in_features, out_features)
        in_features = module.weight.shape[0]
        out_features = module.weight.shape[1]
        weight = module.weight.detach().t()
    else:
        raise TypeError(f"Unsupported module type for int8 conversion: {type(module)}")

    new_layer = bnb.nn.Linear8bitLt(
        in_features,
        out_features,
        module.bias is not None,
        has_fp16_weights=has_fp16_weights,
        threshold=threshold,
        device=quant_device,
    )

    quant_weight = weight.to(device=quant_device, dtype=torch.float32, non_blocking=True)
    new_layer.weight = bnb.nn.Int8Params(
        quant_weight,
        requires_grad=False,
        has_fp16_weights=has_fp16_weights,
    )
    if module.bias is not None:
        bias = module.bias.detach().to(device=quant_device, dtype=torch.float32, non_blocking=True)
        new_layer.bias = torch.nn.Parameter(bias, requires_grad=False)
    new_layer.requires_grad_(False)

    # Explicit cleanup to avoid holding onto the original tensors.
    del weight
    del quant_weight
    if module.bias is not None:
        del bias
    return new_layer


def _convert_modules_to_int8_inplace(
    module: torch.nn.Module,
    quant_config: BitsAndBytesConfig,
    quant_device: torch.device,
    staging_device: torch.device,
    prefix: str = "",
    skip_modules: Optional[Sequence[str]] = None,
) -> None:
    skip_modules = skip_modules or ()
    for name, child in list(module.named_children()):
        child_prefix = f"{prefix}.{name}" if prefix else name
        should_skip = any(
            child_prefix == skip_name or child_prefix.startswith(f"{skip_name}.") for skip_name in skip_modules
        )
        if should_skip:
            continue

        if isinstance(child, (torch.nn.Linear, Conv1D)):
            converted = _convert_linear_module_to_int8(child, quant_config, quant_device)
            if staging_device != quant_device:
                try:
                    converted = converted.to(staging_device)
                except Exception as move_exc:
                    print(
                        f"[teacher] Warning: failed to move quantized module {child_prefix} to "
                        f"{staging_device}; keeping on {quant_device}. Error: {move_exc}",
                        flush=True,
                    )
            module._modules[name] = converted
            del child
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            _convert_modules_to_int8_inplace(
                child,
                quant_config=quant_config,
                quant_device=quant_device,
                staging_device=staging_device,
                prefix=child_prefix,
                skip_modules=skip_modules,
            )


def _load_teacher_fp16_then_quantize(
    model_name: str,
    device: torch.device,
    quant_config: BitsAndBytesConfig,
) -> torch.nn.Module:
    _cleanup_cuda(device.index if device.type == "cuda" else 0)
    print("[teacher] Loading FP16 weights on CPU before manual 8-bit conversion...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    skip = {"lm_head"}
    if quant_config.llm_int8_skip_modules is not None:
        skip.update(quant_config.llm_int8_skip_modules)

    quant_device = device if device.type == "cuda" else torch.device("cpu")
    staging_device = quant_device

    _convert_modules_to_int8_inplace(
        teacher,
        quant_config=quant_config,
        quant_device=quant_device,
        staging_device=staging_device,
        skip_modules=tuple(skip),
    )

    print(f"[teacher] Moving remaining non-int8 parameters to {device}...", flush=True)
    _move_non_int8_to_device(teacher, device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return teacher


def _move_non_int8_to_device(module: torch.nn.Module, device: torch.device) -> None:
    import bitsandbytes as bnb

    for child in module.children():
        if isinstance(child, bnb.nn.Linear8bitLt):
            continue
        _move_non_int8_to_device(child, device)

    for name, param in list(module._parameters.items()):
        if param is None:
            continue
        if isinstance(param, bnb.nn.Int8Params):
            continue
        if param.device == device:
            continue
        new_param = torch.nn.Parameter(
            param.detach().to(device=device, non_blocking=True),
            requires_grad=param.requires_grad,
        )
        module._parameters[name] = new_param

    for name, buf in list(module._buffers.items()):
        if buf is None:
            continue
        if buf.device == device:
            continue
        module._buffers[name] = buf.detach().to(device=device, non_blocking=True)
def load_teacher_with_fallback(
    model_name: str,
    prefer_gpus: List[int],          # Local GPU indices to use for the teacher, in order of preference
    student_gpu: Optional[int],      # Local GPU index reserved for the student (exclude from teacher)
) -> Tuple[torch.nn.Module, AutoTokenizer, torch.device]:
    """
    Load teacher strictly in 8-bit on a single GPU (with CPU offload when possible).
    Returns: (teacher_model, tokenizer, teacher_device_for_inputs)
    """
    try:
        return load_teacher_8bit_strict(
            model_name=model_name,
            prefer_gpus=prefer_gpus,
            student_gpu=student_gpu,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Teacher 8-bit load failed on GPUs {prefer_gpus} (student GPU={student_gpu})."
        ) from exc


def load_fineweb_subset(
    tokenizer,
    max_tokens: int,
    seed: int = 1337,
    max_seq_len: int = 512,
    packing_enabled: bool = True,
):
    """
    Load FineWeb-Edu subset with automatic caching.
    The first run streams data, tokenizes, and caches to disk; later runs reuse the cache.

    Returns a list of {prompt, answer} examples.
    """
    from sampledkd.data.cache import load_or_create_fineweb_cache

    return load_or_create_fineweb_cache(
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        max_seq_len=max_seq_len,
        seed=seed,
        batch_size=512,
        packing_enabled=packing_enabled,
    )
