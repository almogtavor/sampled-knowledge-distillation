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
            # Calculate total VRAM across all GPUs
            total_vram_gb = 0
            for i in range(device_count):
                vram_bytes = torch.cuda.get_device_properties(i).total_memory
                total_vram_gb += vram_bytes / (1024**3)  # Convert bytes to GB
            print("Success: Detected " + str(device_count) + " GPUs with " + str(round(total_vram_gb, 1)) + " GB total VRAM available.")
            teacher_device = torch.device("cuda:0")
            student_device = torch.device("cuda:1")
        else:
            teacher_device = student_device = torch.device("cuda:0")
    else:
        print("Warning: CUDA not available. Using CPU for training.")
        teacher_device = student_device = torch.device("cpu")

    # Load models first (use quantization if specified)
    print("Loading teacher...")
    # Use multi-GPU sharding but try without low_cpu_mem_usage to avoid hanging
    if teacher_device.type == "cuda" and torch.cuda.device_count() >= 2:
        print(f"Using device_map=auto for {torch.cuda.device_count()} GPUs with low_cpu_mem_usage=False")
        # Conservative memory allocation to prevent OOM
        max_memory = {i: "9500MiB" for i in range(torch.cuda.device_count())}
        max_memory[1] = "6000MiB"  # Reserve space on GPU 1 for student
        
        teacher, tok = load_model(
            config.teacher_model,
            device_map="auto",
            quant_bits=config.teacher_quant_bits,
            max_memory=max_memory,
        )
    else:
        teacher_device_map = 0 if teacher_device.type == "cuda" else "cpu"
        teacher, tok = load_model(
            config.teacher_model,
            device_map=teacher_device_map,
            quant_bits=config.teacher_quant_bits,
        )

    print("Loading student...")
    # Keep student on a single GPU (prefer GPU 1 if available)
    student_device_map = 1 if (student_device.type == "cuda" and torch.cuda.device_count() >= 2) else (0 if student_device.type == "cuda" else "cpu")
    student, _ = load_model(
        config.student_model,
        device_map=student_device_map,
        quant_bits=config.student_quant_bits,
    )

    # teacher: eval, no grads
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
        
    print(f"Teacher device: {teacher_device}")
    print(f"Student device: {student_device}")
    
    if student_device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(student_device)} (device {torch.cuda.current_device()})")
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
        collate_fn=collate
    )

    # Initialize TensorBoard writer with experiment name
    current_date = datetime.now().strftime("%Y%m%d_%H%M")
    experiment_name = f"distill-{config.distill_type}-{current_date}"
    tensorboard_path = Path(config.tensorboard_dir) / experiment_name
    print(f"Setting up TensorBoard logging in {tensorboard_path}")
    writer = SummaryWriter(tensorboard_path)

    distiller = Distiller(
        teacher_model=teacher,
        student_model=student,
        tokenizer=tok,
        dataloader=dl,
        config=config,  # Pass the entire config instead of individual args
        teacher_device=teacher_device,
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
