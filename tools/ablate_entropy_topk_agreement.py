#!/usr/bin/env python3
"""
Ablation: agreement between exact entropy and truncated Top-k+Tail entropy.

Required args:
    --model            HF model id or a local model directory
    --dataset          HF dataset name (always uses the train split)
    --prompt_col       Column name for the input prompt/question
    --answer_col       Column name for the target answer (used only for batching length; not evaluated here)

Optional args:
    --dataset_config   Dataset configuration (e.g., 'main' for gsm8k)
    --batch_size       Mini-batch size (default: 4). Reduce if OOM.
    --max_seq_len      Max sequence length (default: 512). Reduce if OOM.
    --k_percent        Percent of valid tokens to take as top-k (default: 20)
    --m                Top-m for truncated entropy (default: 20)
    --device           Device to run on (default: cuda:0 if available else cpu)
    --dtype            Model dtype: float16|bfloat16|float32 (default: float16)
"""
import argparse
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import time
import logging

# Ensure local 'ekd' package is importable when running this script directly
_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_here, os.pardir))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

try:
    from ekd.data.dataset import DistillCollator
    from ekd.training.entropy_utils import truncated_entropy_topk_tail, token_entropy
except ModuleNotFoundError as e:
    raise SystemExit(
        "Failed to import 'ekd'. Run from the repo root, or install the package (pip install -e .).\n"
        f"sys.path[0]={sys.path[0]} repo_root={_repo_root} original error: {e}"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local HF model directory (preferred). For a raw checkpoint, use --base_model and --checkpoint.")
    ap.add_argument("--base_model", default=None, help="Optional: base HF model used to initialize when providing a raw checkpoint.")
    ap.add_argument("--checkpoint", default=None, help="Optional: path to a .pt checkpoint (used with --base_model) to materialize a temporary HF model for ablation.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dataset_config", default=None)
    ap.add_argument("--prompt_col", required=True)
    ap.add_argument("--answer_col", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--k_percent", type=float, default=20.0)
    ap.add_argument("--m", type=int, default=20, help="top-m for truncated entropy")
    ap.add_argument("--device", default=None, help="cuda:0, cpu, etc. Defaults to cuda:0 if available else cpu")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"], help="Model dtype")
    args = ap.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("ablate.topk_agreement")

    log.info("Starting ablation run")
    log.info(f"Python {sys.version.split()[0]} | torch {torch.__version__}")
    log.info(f"CUDA available={torch.cuda.is_available()} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    log.info(f"Args: model={args.model} base={args.base_model} ckpt={args.checkpoint} dataset={args.dataset} cfg={args.dataset_config} bs={args.batch_size} max_len={args.max_seq_len} k%={args.k_percent} m={args.m} dtype={args.dtype}")

    # Resolve device
    device_str = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    log.info(f"Using device: {device}")
    # Resolve dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    model_dtype = dtype_map.get(args.dtype, torch.float16)
    log.info(f"Using dtype: {model_dtype}")

    t0 = time.time()
    # If a raw checkpoint is provided, export a temporary HF model dir
    export_tmp = None
    model_path_for_load = args.model
    if args.base_model and args.checkpoint:
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from ekd.ekd.evaluations.eval import export_hf_model  # reuse exporter
        export_tmp = TemporaryDirectory()
        tmp_dir = Path(export_tmp.name) / "ablate_export"
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise SystemExit(f"Checkpoint not found: {ckpt_path}")
        log.info(f"Exporting temporary HF model from base={args.base_model} ckpt={ckpt_path} -> {tmp_dir}")
        export_hf_model(args.base_model, ckpt_path, tmp_dir)
        model_path_for_load = str(tmp_dir)

    log.info("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(model_path_for_load, use_fast=False, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    log.info("Tokenizer ready")

    log.info("Loading dataset train split...")
    ds = load_dataset(args.dataset, args.dataset_config)["train"]
    log.info(f"Dataset loaded: {len(ds)} rows")
    examples = [{"prompt": ex[args.prompt_col], "answer": ex[args.answer_col]} for ex in ds]
    collate = DistillCollator(tok, args.max_seq_len)
    dl = torch.utils.data.DataLoader(examples, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    log.info(f"DataLoader ready: batches={len(dl)} bs={args.batch_size}")

    log.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_path_for_load, dtype=model_dtype)
    model.to(device)
    model.eval()
    log.info("Model loaded and moved to device")

    import math
    tot, agree_sum = 0, 0
    last_log = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(dl):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits = model(ids, attention_mask=mask).logits
            pred = logits[:, :-1, :]
            valid = mask[:, 1:].bool()

            B, Lm1, _ = pred.shape
            ent_exact = []
            ent_trunc = []
            for i in range(B):
                ent_exact.append(token_entropy(pred[i]).cpu())                     # [L-1]
                ent_trunc.append(truncated_entropy_topk_tail(pred[i], k=args.m).cpu())
            ent_exact = torch.stack(ent_exact)   # [B, L-1]
            ent_trunc = torch.stack(ent_trunc)   # [B, L-1]

            for i in range(B):
                v = valid[i].cpu()
                n_valid = int(v.sum())
                if n_valid < 3: 
                    continue
                k = max(1, math.ceil(n_valid * args.k_percent / 100.0))
                idx = torch.nonzero(v, as_tuple=False).squeeze(-1)

                e1 = ent_exact[i, idx]
                e2 = ent_trunc[i, idx]

                top1 = idx[torch.topk(e1, k=k, largest=True, sorted=False).indices]
                top2 = idx[torch.topk(e2, k=k, largest=True, sorted=False).indices]

                set1 = set(top1.tolist())
                set2 = set(top2.tolist())
                inter = len(set1 & set2)
                agree_sum += inter / float(k)
                tot += 1

            # Periodic progress logging
            now = time.time()
            if now - last_log > 2.0:
                last_log = now
                avg = (agree_sum / tot) if tot else 0.0
                log.info(f"Progress: batch {bi+1}/{len(dl)} | partial avg overlap={avg:.4f} over {tot} sequences")

    if tot == 0:
        log.warning("No valid sequences.")
    else:
        avg = agree_sum / tot
        dt = time.time() - t0
        log.info(f"Done in {dt:.1f}s")
        print(f"Average top-k overlap ratio (exact vs truncated, m={args.m}): {avg:.4f} over {tot} sequences.")

    # Cleanup temp export if used
    if export_tmp is not None:
        try:
            export_tmp.cleanup()
        except Exception:
            pass

if __name__ == "__main__":
    main()