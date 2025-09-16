import argparse
from pathlib import Path
from datetime import datetime

import torch
from transformers.utils import is_torch_cuda_available
from torch.utils.tensorboard import SummaryWriter

from ekd.data.dataset import AIMEJsonl, DistillCollator
from ekd.models.loader import load_model
from ekd.models.ollama_loader import OllamaModel
from ekd.training.distiller import Distiller
from ekd.config import TrainingConfig
from datasets import load_dataset

# ===== DDP bootstrap (single-node multi-GPU ready) =====
import os
def _ddp_env():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank       = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return local_rank, rank, world_size

LOCAL_RANK, RANK, WORLD_SIZE = _ddp_env()
USE_DDP = WORLD_SIZE > 1

if USE_DDP:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(LOCAL_RANK)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def is_main_process(): return RANK == 0
torch.backends.cudnn.benchmark = True

def parse_args_to_config() -> TrainingConfig:
    """Parse command line arguments and create TrainingConfig."""
    parser = argparse.ArgumentParser(description="Entropy-guided KD for LLMs")
    parser.add_argument("--teacher_model", required=True)
    parser.add_argument("--student_model", required=True)
    parser.add_argument("--teacher_quant_bits", type=int, choices=[4, 8], default=None,
                        help="Load teacher in 4-bit or 8-bit to reduce VRAM usage")
    parser.add_argument("--student_quant_bits", type=int, choices=[4, 8], default=None,
                        help="Optionally quantize student for memory (not typical during training)")
    parser.add_argument("--distill_type", choices=["vanilla", "ekd"], default="vanilla")
    parser.add_argument("--top_k_percent", type=int, default=20, help="for EKD only")
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
        # Set memory fraction to prevent OOM
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of GPU memory max

    # device placement
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count >= 2:
            total_vram_gb = sum(
                torch.cuda.get_device_properties(i).total_memory for i in range(device_count)
            ) / (1024**3)
            print(f"Success: Detected {device_count} GPUs with {round(total_vram_gb, 1)} GB total VRAM available.")
            teacher_device = torch.device("cuda:0")
            student_device = torch.device("cuda:1")
        else:
            teacher_device = student_device = torch.device("cuda:0")
    else:
        print("Warning: CUDA not available. Using CPU for training.")
        teacher_device = student_device = torch.device("cpu")

    # Load models first (use quantization if specified)
    print("Loading teacher...")
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        print("Loading teacher with device_map=auto (may shard across GPUs)...")
        teacher, tok = load_model(
            config.teacher_model,
            device_map="auto",
            quant_bits=config.teacher_quant_bits or 8,  # force 8-bit if not specified
        )
        # Important: do NOT move teacher to a specific GPU
        teacher_device = None  # marker: teacher is sharded
    else:
        teacher, tok = load_model(
            config.teacher_model,
            device_map=0 if teacher_device.type == "cuda" else "cpu",
            quant_bits=config.teacher_quant_bits,
        )

    print("Loading student...")
    student, _ = load_model(
        config.student_model,
        device_map=1 if (student_device.type == "cuda" and torch.cuda.device_count() >= 2) else (
            0 if student_device.type == "cuda" else "cpu"
        ),
        quant_bits=config.student_quant_bits,
    )

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    if hasattr(teacher, "config"):
        teacher.config.use_cache = False

    student.train()
    if hasattr(student, "gradient_checkpointing_enable"):
        try: student.gradient_checkpointing_enable()
        except Exception: pass
    if hasattr(student, "config"):
        student.config.use_cache = False

    print(f"Teacher device: {'sharded' if teacher_device is None else teacher_device}")
    print(f"Student device: {student_device}")

    # Load dataset
    if all(p.endswith(".jsonl") for p in config.datasets):
        # Use local JSONL if paths are given TODO: make this generic
        dataset = AIMEJsonl([Path(p) for p in config.datasets])
    else:
        print(f"Loading Hugging Face dataset: {config.datasets[0]}")
        hf_dataset = load_dataset(config.datasets[0], config.dataset_config)["train"] \
            if config.dataset_config else load_dataset(config.datasets[0])["train"]
        print(f"Using columns - prompt: '{config.prompt_col}', answer: '{config.answer_col}'")
        examples = [{"prompt": ex[config.prompt_col], "answer": ex[config.answer_col]} for ex in hf_dataset]
        dataset = examples

    collate = DistillCollator(tok, config.max_seq_len)
    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate
    )

    # Initialize TensorBoard writer with experiment name
    current_date = datetime.now().strftime("%Y%m%d_%H%M")
    experiment_name = f"distill-{config.distill_type}-{current_date}"
    tensorboard_path = Path(config.tensorboard_dir) / experiment_name
    print(f"Setting up TensorBoard logging in {tensorboard_path}")
    writer = SummaryWriter(tensorboard_path)

    # Distiller
    distiller = Distiller(
        teacher_model=teacher,
        student_model=student,
        tokenizer=tok,
        dataloader=dl,
        config=config,  # Pass the entire config instead of individual args
        teacher_device=teacher_device,   # None means sharded
        student_device=student_device,
        writer=writer,  # Pass TensorBoard writer to Distiller
    )

    distiller.train(epochs=config.epochs)

    # Close TensorBoard writer
    writer.close()

    print("Saving student to", config.output_dir)
    student.save_pretrained(config.output_dir)
    tok.save_pretrained(config.output_dir)


if __name__ == "__main__":
    main()
