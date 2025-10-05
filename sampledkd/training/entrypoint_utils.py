from __future__ import annotations

import gc
import os
from typing import List, Optional, Tuple

import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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

 
# def load_teacher_8bit_strict(
#     model_name: str,
#     prefer_gpus: List[int],
#     student_gpu: Optional[int],
# ) -> Tuple[torch.nn.Module, AutoTokenizer, torch.device]:
#     """
#     Strict 8-bit load on a single GPU, no CPU offload allowed.
#     If it fails, raise – do not fall back to FP16 or CPU.
#     """
#     # Environment nudges for bnb + CUDA 11.8 on these nodes
#     os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
#     ld = os.environ.get("LD_LIBRARY_PATH", "")
#     if "/usr/local/cuda/lib64" not in ld:
#         os.environ["LD_LIBRARY_PATH"] = ld + (":" if ld else "") + "/usr/local/cuda/lib64"
#     # Helps bnb pick the right kernels on older drivers
#     os.environ.setdefault("BNB_CUDA_VERSION", "118")

#     teacher_gpus = [g for g in prefer_gpus if g != student_gpu]
#     if not teacher_gpus:
#         raise RuntimeError("No GPUs available for teacher after excluding the student GPU.")
#     gpu = teacher_gpus[0]

#     tok = AutoTokenizer.from_pretrained(
#         model_name,
#         use_fast=False,
#         trust_remote_code=True,
#         local_files_only=False,
#     )
#     if tok.pad_token_id is None and tok.eos_token is not None:
#         tok.pad_token = tok.eos_token

#     q8 = BitsAndBytesConfig(
#         load_in_8bit=True,
#         llm_int8_threshold=6.0,
#         llm_int8_enable_fp32_cpu_offload=False,  # critical: keep everything on GPU
#     )

#     print(f"[teacher] Loading STRICT 8-bit on cuda:{gpu} (no CPU offload)", flush=True)
#     teacher = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         quantization_config=q8,
#         device_map={"": gpu},          # pin every module to this GPU
#         low_cpu_mem_usage=True,
#         trust_remote_code=True,
#     )
#     teacher.eval()
#     for p in teacher.parameters():
#         p.requires_grad_(False)

#     # Sanity: ensure no layer got sharded/offloaded
#     devmap = getattr(teacher, "hf_device_map", None)
#     if isinstance(devmap, dict):
#         bad = [k for k, v in devmap.items() if isinstance(v, str) and ("cpu" in v or "disk" in v)]
#         if bad:
#             raise RuntimeError(f"Unexpected CPU/offloaded modules in device_map: {bad}")

#     print("[teacher] 8-bit loaded on single 2080 Ti with no CPU offload.", flush=True)
#     return teacher, tok, torch.device(f"cuda:{gpu}")


# def load_teacher_8bit_strict_2gpus(
#     model_name: str,
#     prefer_gpus: List[int],
#     student_gpu: Optional[int],
# ) -> Tuple[torch.nn.Module, AutoTokenizer, torch.device]:
#     """
#     Strict 8-bit load sharded across two GPUs, no CPU offload.
#     """
#     # Env nudges
#     os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
#     ld = os.environ.get("LD_LIBRARY_PATH", "")
#     if "/usr/local/cuda/lib64" not in ld:
#         os.environ["LD_LIBRARY_PATH"] = ld + (":" if ld else "") + "/usr/local/cuda/lib64"
#     os.environ.setdefault("BNB_CUDA_VERSION", "118")

#     teacher_gpus = [g for g in prefer_gpus if g != student_gpu]
#     if len(teacher_gpus) < 2:
#         raise RuntimeError("Need at least two GPUs for strict 2-GPU 8-bit sharding.")
#     g0, g1 = teacher_gpus[:2]

#     tok = AutoTokenizer.from_pretrained(
#         model_name,
#         use_fast=False,
#         trust_remote_code=True,
#         local_files_only=False,
#     )
#     if tok.pad_token_id is None and tok.eos_token is not None:
#         tok.pad_token = tok.eos_token

#     q8 = BitsAndBytesConfig(
#         load_in_8bit=True,
#         llm_int8_threshold=6.0,
#         llm_int8_enable_fp32_cpu_offload=False,
#     )

#     # Important: only give max_memory for the two GPUs; do not mention CPU
#     max_memory = {
#         f"cuda:{g0}": "10GiB",
#         f"cuda:{g1}": "10GiB",
#     }

#     print(f"[teacher] Loading STRICT 8-bit across cuda:{g0},{g1} (no CPU offload)", flush=True)
#     teacher = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         quantization_config=q8,
#         device_map="balanced_low_0",       # spread layers across listed GPUs
#         max_memory=max_memory,             # constrain to these GPUs only
#         low_cpu_mem_usage=True,
#         trust_remote_code=True,
#     )
#     teacher.eval()
#     for p in teacher.parameters():
#         p.requires_grad_(False)

#     devmap = getattr(teacher, "hf_device_map", None)
#     if isinstance(devmap, dict):
#         bad = [k for k, v in devmap.items() if isinstance(v, str) and ("cpu" in v or "disk" in v)]
#         if bad:
#             raise RuntimeError(f"Unexpected CPU/offloaded modules in device_map: {bad}")

#     print(f"[teacher] 8-bit sharded. Device map: {devmap}", flush=True)
#     # Use the first teacher GPU for feeding inputs
#     return teacher, tok, torch.device(f"cuda:{g0}")


def load_teacher_with_fallback(
    model_name: str,
    prefer_gpus: List[int],          # Local GPU indices to use for the teacher, in order of preference
    student_gpu: Optional[int],      # Local GPU index reserved for the student (exclude from teacher)
) -> Tuple[torch.nn.Module, AutoTokenizer, torch.device]:
    """
    1) Try single-GPU FP16 on prefer_gpus[0]
    2) If OOM, try 2-GPU sharding across prefer_gpus[0:2] (excluding student_gpu)
    3) If still OOM, try 8-bit on single GPU
    4) If still OOM, try 4-bit on single GPU
    Returns: (teacher_model, tokenizer, teacher_device_for_inputs)
    """
    # Helper: make a local offload/cache dir (per job if SLURM), prefer node-local TMPDIR
    tmpdir = os.environ.get("TMPDIR", "/home/joberant/NLP_2425b/almogt/ekd/tmp")
    offload_dir = os.path.join(tmpdir, "hf_offload_teacher")
    os.makedirs(offload_dir, exist_ok=True)

    # Tokenizer first (slow & local to avoid cluster hiccups)
    tok = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        trust_remote_code=True,
        local_files_only=False,   # set True if your cache is complete
    )
    if tok.pad_token_id is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    # Filter teacher GPUs (exclude student's GPU if present)
    teacher_gpus = [g for g in prefer_gpus if g != student_gpu]
    if not teacher_gpus:
        raise RuntimeError("No GPUs available for teacher after excluding the student GPU.")

    # ===== 1) 8-bit quantization on single GPU =====
    try:
        if not _bnb_triton_available():
            raise RuntimeError("bitsandbytes/triton not available; skipping 8-bit fallback")
        from transformers import BitsAndBytesConfig
        q8 = BitsAndBytesConfig(load_in_8bit=True)
        one = teacher_gpus[0]
        print(f"[teacher] Trying 8-bit quantization on cuda:{one}", flush=True)
        teacher = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": one},
            quantization_config=q8,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print("[teacher] Loaded 8-bit.", flush=True)
        return teacher, tok, torch.device(f"cuda:{one}")
    except Exception as e:
        print(f"[teacher] 8-bit failed: {e}", flush=True)

    # # ===== 1.5) Force 8-bit on a single GPU (no CPU offload) =====
    # try:
    #     teacher, tok, teacher_inputs_device = load_teacher_8bit_strict(
    #         model_name="Qwen/Qwen3-8B",
    #         prefer_gpus=[0,1],     # local indices on the 2080 Ti box
    #         student_gpu=1,         # keep GPU 1 free for the student
    #     )
    # except Exception as e:
    #     print(f"[teacher] Second variant 8-bit failed: {e}", flush=True)
    # # If that raises OOM, switch to the 2-GPU 8-bit loader:
    # try:
    #     teacher, tok, teacher_inputs_device = load_teacher_8bit_strict_2gpus(
    #         model_name="Qwen/Qwen3-8B",
    #         prefer_gpus=[0,1],
    #         student_gpu=None,     # teacher uses both; schedule student elsewhere
    #     )
    # except Exception as e:
    #     print(f"[teacher] 2-GPU 8-bit failed: {e}", flush=True)

    # ===== 2) Single-GPU FP16 (best performance if it fits) =====
    try:
        one = teacher_gpus[0]
        print(f"[teacher] Clearing CUDA caches on cuda:{one} before FP16 load", flush=True)
        _cleanup_cuda(one)
        print(f"[teacher] Trying single-GPU FP16 on cuda:{one}", flush=True)
        teacher = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": one},
            dtype=torch.float16,             # use 'dtype' (not torch_dtype)
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print("[teacher] Loaded on single GPU.", flush=True)
        return teacher, tok, torch.device(f"cuda:{one}")
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            print(f"[teacher] Single-GPU load failed (non-OOM): {e}", flush=True)
            # continue anyway and try multi-GPU
        else:
            print("[teacher] Single-GPU OOM, trying 2-GPU sharding...", flush=True)

    # ===== 3) Multi-GPU sharding with Accelerate =====
    if len(teacher_gpus) >= 2:
        try:
            print(f"[teacher] Trying multi-GPU sharding on {len(teacher_gpus)} GPUs: {teacher_gpus}", flush=True)

            cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            with init_empty_weights():
                empty_model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)

            # Use all teacher_gpus with per-GPU caps
            max_memory = {g: "11GiB" for g in teacher_gpus}
            max_memory["cpu"] = "40GiB"

            teacher = load_checkpoint_and_dispatch(
                empty_model,
                checkpoint=model_name,
                device_map="balanced_low_0",
                max_memory=max_memory,
                offload_folder=offload_dir,
                dtype=torch.float16,
            )
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)

            print(f"[teacher] Multi-GPU sharding done. Device map: {getattr(teacher, 'hf_device_map', None)}", flush=True)
            # For model-parallel HF models, feeding inputs on the first teacher GPU is fine:
            return teacher, tok, torch.device(f"cuda:{teacher_gpus[0]}")
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                print(f"[teacher] Multi-GPU load failed (non-OOM): {e}", flush=True)
            else:
                print("[teacher] Multi-GPU OOM, falling back to quantization (you should always start with no quantization or FP16!)...", flush=True)
        except Exception as e:
            print(f"[teacher] Multi-GPU dispatch error: {e}", flush=True)

    # Optional: print one-shot diagnostics so logs show why quantization was/wasn't attempted
    try:
        import importlib
        _torch_ver = getattr(torch, "__version__", "?")
        try:
            _triton = importlib.import_module("triton")
            _triton_ver = getattr(_triton, "__version__", "installed")
        except Exception:
            _triton_ver = "not-installed"
        try:
            _bnb = importlib.import_module("bitsandbytes")
            _bnb_ver = getattr(_bnb, "__version__", "installed")
        except Exception:
            _bnb_ver = "not-installed"
        print(f"[teacher] Quant backends → available={_bnb_triton_available()} torch={_torch_ver} triton={_triton_ver} bitsandbytes={_bnb_ver}", flush=True)
    except Exception:
        pass

    # ===== 4) 4-bit quantization on single GPU (last resort) =====
    try:
        if not _bnb_triton_available():
            raise RuntimeError("bitsandbytes/triton not available; skipping 4-bit fallback")
        from transformers import BitsAndBytesConfig
        q4 = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        one = teacher_gpus[0]
        print(f"[teacher] Trying 4-bit quantization (last resort) on cuda:{one}", flush=True)
        teacher = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": one},
            quantization_config=q4,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print("[teacher] Loaded with 4-bit quantization (last resort).", flush=True)
        return teacher, tok, torch.device(f"cuda:{one}")
    except Exception as e:
        raise RuntimeError(f"[teacher] All GPU strategies failed. Last error: {e}")


def load_fineweb_subset(tokenizer, max_tokens: int, seed: int = 1337, max_seq_len: int = 512):
    """
    Load FineWeb-Edu subset with automatic caching.
    First run: streams, filters, and caches to disk.
    Subsequent runs: loads from cache instantly.
    
    Returns a list of {prompt, answer} examples.
    """
    from sampledkd.data.cache import load_or_create_fineweb_cache
    
    return load_or_create_fineweb_cache(
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        max_seq_len=max_seq_len,
        seed=seed,
        batch_size=512,
    )
