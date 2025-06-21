import argparse
from pathlib import Path

import torch
from transformers.utils import is_torch_cuda_available

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
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    if not is_torch_cuda_available():
        raise RuntimeError("CUDA-enabled PyTorch required for training.")

    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(device)} (device {torch.cuda.current_device()})")

    print("Loading teacher (Ollama)...")
    # teacher = OllamaModel(args.teacher_model)
    # for non ollama models: # TODO: decide if we want to support non ollama models
    teacher, tok = load_model(args.teacher_model, device_map="auto", quant_bits=4)
    tok = teacher.tokenizer  # Use teacher's tokenizer for both models

    print("Loading student...")
    student, _ = load_model(args.student_model, device_map="auto", quant_bits=8)
    # student.resize_token_embeddings(len(tok))  # align vocab if needed
    student.resize_token_embeddings(len(teacher.tokenizer)) #TODO: verify its safe to do!!

    # build DataLoader
    if all(p.endswith(".jsonl") for p in args.datasets):
        # Use local JSONL if paths are given
        dataset = AIMEJsonl([Path(p) for p in args.datasets])
    else:
        # Load from Hugging Face dataset if user passes HF dataset name
        print(f"Loading Hugging Face dataset: {args.datasets[0]}")
        hf_dataset = load_dataset(args.datasets[0])["train"]
        # Reformat to match AIMEJsonl format: list of dicts with id, question, answer
        examples = [
            {"id": x["ID"], "prompt": x["Problem"], "answer": x["Answer"]}
            for x in hf_dataset
        ]
        dataset = examples
    collate = DistillCollator(tok, args.max_seq_len)
    dl = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    distiller = Distiller(
        teacher_model=teacher,
        student_model=student,
        tokenizer=tok,
        dataloader=dl,
        distill_type=args.distill_type,
        top_k_percent=args.top_k_percent,
        lr=args.lr,
        device=device,
    )

    distiller.train(epochs=args.epochs)

    print("Saving student to", args.output_dir)
    student.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main() 