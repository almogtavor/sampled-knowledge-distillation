import argparse
from pathlib import Path

import torch
from transformers.utils import is_torch_cuda_available
from torch.utils.tensorboard import SummaryWriter

from ekd.data.dataset import AIMEJsonl, DistillCollator
from ekd.models.loader import load_model
from ekd.models.ollama_loader import OllamaModel
from ekd.training.distiller import Distiller
from datasets import load_dataset


def main():
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
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="(Optional) HF dataset config, e.g. for gsm8k use '--dataset_config main' or 'socratic'"
    )
    args = parser.parse_args()

    if not is_torch_cuda_available():
        print("Warning: CUDA not available. Using CPU for training.")
        teacher_device = torch.device("cpu")
        student_device = torch.device("cpu")
    else:
        teacher_device = torch.device("cpu")  # Teacher on CPU to save GPU memory
        student_device = torch.device("cuda")  # Student on GPU

    print("Loading teacher...")
    # teacher = OllamaModel(args.teacher_model)
    # for non ollama models: # TODO: decide if we want to support non ollama models
    teacher, tok = load_model(args.teacher_model, device_map="cpu", quant_bits=8)  # Use 8-bit for teacher to save memory

    print("Loading student...")
    # Load student without quantization but with smaller model to save memory
    if student_device.type == "cuda":
        student, _ = load_model(args.student_model, device_map="auto", quant_bits=None)  # No quantization for training
    else:
        student, _ = load_model(args.student_model, device_map="cpu", quant_bits=None)  # CPU training
    
    # Ensure student is in training mode - keep parameters in FP32 for gradient computation
    student.train()
    # Don't convert model to half - keep parameters in FP32 for gradient stability
    # student.resize_token_embeddings(len(tok))  # align vocab if needed
    teacher.resize_token_embeddings(len(tok))  # optional if needed for teacher
    student.resize_token_embeddings(len(tok))  # TODO: verify its safe to do!!
    
    # Ensure models are on correct devices after resizing
    if teacher_device.type == "cpu":
        teacher = teacher.to("cpu")  # Keep teacher on CPU
    else:
        teacher = teacher.to(teacher_device)
    student = student.to(student_device)

    if student_device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(student_device)} (device {torch.cuda.current_device()})")
    else:
        print(f"Using device: {student_device}")

    # build DataLoader
    if all(p.endswith(".jsonl") for p in args.datasets):
        # Use local JSONL if paths are given TODO: make this generic
        dataset = AIMEJsonl([Path(p) for p in args.datasets])
    else:
        # Load from Hugging Face dataset if user passes HF dataset name
        print(f"Loading Hugging Face dataset: {args.datasets[0]}")
        hf_dataset = load_dataset(args.datasets[0], args.dataset_config)["train"] if args.dataset_config \
            else load_dataset(args.datasets[0])["train"]
        print(f"Using columns - prompt: '{args.prompt_col}', answer: '{args.answer_col}'")
        # Reformat to match AIMEJsonl format: list of dicts with prompt, answer
        examples = []
        for ex in hf_dataset:
            examples.append({
                "prompt": ex[args.prompt_col],
                "answer": ex[args.answer_col],
            })
        dataset = examples

    collate = DistillCollator(tok, args.max_seq_len)
    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate
    )

    # Initialize TensorBoard writer
    print(f"Setting up TensorBoard logging in {args.tensorboard_dir}")
    writer = SummaryWriter(args.tensorboard_dir)

    distiller = Distiller(
        teacher_model=teacher,
        student_model=student,
        tokenizer=tok,
        dataloader=dl,
        distill_type=args.distill_type,
        top_k_percent=args.top_k_percent,
        lr=args.lr,
        teacher_device=teacher_device,
        student_device=student_device,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        writer=writer,  # Pass TensorBoard writer to Distiller
    )

    distiller.train(epochs=args.epochs)

    # Close TensorBoard writer
    writer.close()

    print("Saving student to", args.output_dir)
    student.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
