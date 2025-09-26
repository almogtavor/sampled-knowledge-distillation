import argparse
from pathlib import Path
from datetime import datetime
import os, shutil
from typing import List, Tuple, Optional

import torch
from transformers.utils import is_torch_cuda_available
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from ekd.data.dataset import AIMEJsonl, DistillCollator
from ekd.models.loader import load_model
from ekd.models.ollama_loader import OllamaModel
from ekd.training.distiller import Distiller
from ekd.config import TrainingConfig
from datasets import load_dataset

# Import logging utils with fallback
try:
    from ekd.logging.wandb_utils import create_training_combined_logger
except ImportError:
    create_training_combined_logger = lambda *args, **kwargs: None


def load_teacher_with_fallback(
    model_name: str,
    prefer_gpus: List[int],          # GPUs to use for the teacher, in order of preference
    student_gpu: Optional[int],      # GPU reserved for the student (exclude from teacher)
) -> Tuple[torch.nn.Module, AutoTokenizer, torch.device]:
    """
    1) Try single-GPU FP16 on prefer_gpus[0]
    2) If OOM, try 2-GPU sharding across prefer_gpus[0:2] (excluding student_gpu)
    3) If still OOM, try 8-bit on single GPU
    4) If still OOM, try 4-bit on single GPU
    Returns: (teacher_model, tokenizer, teacher_device_for_inputs)
    """
    # Helper: make a local offload/cache dir (per job if SLURM)
    job_id = os.environ.get("SLURM_JOB_ID", "nojid")
    # Use TMPDIR which is already set up in the SLURM script
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

    # ===== 1) Single-GPU FP16 (best performance if it fits) =====
    try:
        one = teacher_gpus[0]
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

    # ===== 2) Multi-GPU sharding with Accelerate =====
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

    # ===== 3) 8-bit quantization on single GPU =====
    try:
        from transformers import BitsAndBytesConfig
        q8 = BitsAndBytesConfig(load_in_8bit=True)
        one = teacher_gpus[0]
        print(f"[teacher] Trying 8-bit quantization fallback on cuda:{one}", flush=True)
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

    # ===== 4) 4-bit quantization on single GPU (last resort) =====
    try:
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


def parse_args_to_config() -> TrainingConfig:
    """Parse command line arguments and create TrainingConfig."""
    parser = argparse.ArgumentParser(description="Entropy-guided KD for LLMs")
    parser.add_argument("--teacher_model", required=True)
    parser.add_argument("--student_model", required=True)
    parser.add_argument("--student_quant_bits", type=int, choices=[4, 8], default=None,
                        help="Optionally quantize student for memory (not typical during training)")
    parser.add_argument("--distill_type", choices=["vanilla", "top-k-tok"], default="vanilla")
    parser.add_argument("--top_k_percent", type=int, default=20, help="for top-k-tok only")
    parser.add_argument("--enable_ce", action="store_true", default=True, 
                        help="Enable cross-entropy loss in addition to KD loss")
    parser.add_argument("--no_ce", dest="enable_ce", action="store_false",
                        help="Disable cross-entropy loss (use only KD loss)")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--prompt_col", type=str, default=None,
                        help="name of text prompt column for HF datasets")
    parser.add_argument("--answer_col", type=str, default=None,
                        help="name of answer column for HF datasets")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, 
                        help="Number of steps to accumulate gradients before updating")
    parser.add_argument("--max_seq_len", type=int, default=512)  # to save memory
    parser.add_argument("--lr", type=float, default=1e-5)  # Reduced from 5e-5
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tensorboard_dir", type=str, default="tb", 
                        help="Directory for TensorBoard logs")
    parser.add_argument("--checkpoint_steps", type=int, default=500,
                        help="Save checkpoint every N steps (0 to disable)")
    parser.add_argument("--keep_checkpoints", type=int, default=3,
                        help="Number of recent checkpoints to keep")
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="(Optional) HF dataset config, e.g. for gsm8k use '--dataset_config main' or 'socratic'"
    )
    args = parser.parse_args()
    
    # Convert argparse Namespace to TrainingConfig
    return TrainingConfig(**vars(args))


def main():
    """Main training function using Pydantic configuration."""
    config = parse_args_to_config()

    # Set CUDA memory management settings for better memory efficiency
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear any cached memory
        
    # Speed optimizations (safe)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # ----------------- teacher / student device planning -----------------
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if device_count == 0:
        raise RuntimeError("No CUDA devices available.")

    # Calculate total VRAM across all GPUs
    total_vram_gb = 0
    for i in range(device_count):
        vram_bytes = torch.cuda.get_device_properties(i).total_memory
        total_vram_gb += vram_bytes / (1024**3)  # Convert bytes to GB
    print("Success: Detected " + str(device_count) + " GPUs with " + str(round(total_vram_gb, 1)) + " GB total VRAM available.")

    # Dynamic GPU allocation based on CUDA_VISIBLE_DEVICES
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    avail = [int(x) for x in visible.split(",")] if visible else list(range(device_count))
    student_gpu = avail[1] if len(avail) >= 2 else avail[0]
    student_device = torch.device(f"cuda:{student_gpu}")
    prefer_for_teacher = [g for g in avail if g != student_gpu]
    
    print(f"CUDA_VISIBLE_DEVICES: {visible}")
    print(f"Available GPUs: {avail}")
    print(f"Student GPU: {student_gpu}")
    print(f"Teacher GPUs: {prefer_for_teacher}")

    print("Loading teacher with GPU-first fallback...", flush=True)
    teacher, tok, teacher_inputs_device = load_teacher_with_fallback(
        model_name=config.teacher_model,
        prefer_gpus=prefer_for_teacher,
        student_gpu=student_gpu,
    )

    print("Loading student on its own GPU...", flush=True)
    student, _ = load_model(  # your existing helper is fine for student
        config.student_model,
        device_map=student_gpu,     # {'': 1}
        quant_bits=config.student_quant_bits,
    )

    # Freeze teacher, no grads (redundant but safe)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    # Reduce memory footprint during forward
    if hasattr(teacher, "config"):
        teacher.config.use_cache = False

    # Ensure student is in training mode - keep parameters in FP32 for gradient computation
    student.train()
    # Enable gradient checkpointing to reduce activation memory
    if hasattr(student, "gradient_checkpointing_enable"):
        try:
            student.gradient_checkpointing_enable()
        except Exception:
            pass
    if hasattr(student, "config"):
        student.config.use_cache = False
        
    print(f"Teacher device: {teacher_inputs_device}")
    print(f"Student device: {student_device}")
    
    if student_device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(student_device.index)} (device {student_device.index})")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(student_device) / 1024**3:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved(student_device) / 1024**3:.2f} GB")
    else:
        print(f"Using device: {student_device}")

    if all(p.endswith(".jsonl") for p in config.datasets):
        # Use local JSONL if paths are given TODO: make this generic
        dataset = AIMEJsonl([Path(p) for p in config.datasets])
    else:
        # Load from Hugging Face dataset if user passes HF dataset name
        print(f"Loading Hugging Face dataset: {config.datasets[0]}")
        hf_dataset = load_dataset(config.datasets[0], config.dataset_config)["train"] if config.dataset_config \
            else load_dataset(config.datasets[0])["train"]
        print(f"Using columns - prompt: '{config.prompt_col}', answer: '{config.answer_col}'")
        examples = []
        for ex in hf_dataset:
            examples.append({
                "prompt": ex[config.prompt_col],
                "answer": ex[config.answer_col],
            })
        dataset = examples

    collate = DistillCollator(tok, config.max_seq_len)
    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=min(8, os.cpu_count() or 1),
        pin_memory=True,
        persistent_workers=True
    )

    # Initialize logging with experiment name
    current_date = datetime.now().strftime("%Y%m%d_%H%M")
    job_id = os.getenv("SLURM_JOB_ID", "local")
    experiment_name = (
        f"distill-{config.distill_type}-{current_date}_{job_id}"
        + (f"_k={config.top_k_percent}" if config.distill_type != "vanilla" else "")
    )
    
    # Initialize combined logger (W&B + TensorBoard)
    combined_logger = create_training_combined_logger(
        config, experiment_name, tensorboard_dir=config.tensorboard_dir
    )

    distiller = Distiller(
        teacher_model=teacher,
        student_model=student,
        tokenizer=tok,
        dataloader=dl,
        config=config,  # Pass the entire config instead of individual args
        teacher_device=teacher_inputs_device,
        student_device=student_device,
        logger=combined_logger,  # Use new combined logger
    )

    distiller.train(epochs=config.epochs)

    # Close logging  
    if combined_logger:
        try:
            combined_logger.log_artifact(config.output_dir, f"student_model_{experiment_name}", "model")
            combined_logger.finish()
        except Exception:
            pass

    print("Saving student to", config.output_dir)
    student.save_pretrained(config.output_dir)
    tok.save_pretrained(config.output_dir)


if __name__ == "__main__":
    main()
