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

    if not is_torch_cuda_available():
        print("Warning: CUDA not available. Using CPU for training.")
        teacher_device = torch.device("cpu")
        student_device = torch.device("cpu")
    else:
        teacher_device = torch.device("cuda")
        student_device = torch.device("cuda")

    print("Loading teacher...")
    teacher, tok = load_model(config.teacher_model, device_map=teacher_device)

    print("Loading student...")
    student, _ = load_model(config.student_model, device_map=student_device)
    
    # Ensure student is in training mode - keep parameters in FP32 for gradient computation
    student.train()
    # Don't convert model to half - keep parameters in FP32 for gradient stability
    teacher.resize_token_embeddings(len(tok))  # optional if needed for teacher
    student.resize_token_embeddings(len(tok))  # TODO: verify its safe to do!!
    
    if student_device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(student_device)} (device {torch.cuda.current_device()})")
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
