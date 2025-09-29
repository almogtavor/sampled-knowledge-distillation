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
    from ekd.training.entropy_utils import (
        token_entropy,
        truncated_entropy_topk_tail,
        truncated_entropy_topk_tail_midpoint,
        truncated_entropy_topk_tail_uniform,
        entropy_topm_plus_tail_is,
        entropy_topm_plus_cv_is,
    )
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
    
    # Define entropy variants to test
    entropy_variants = {
        "topk_tail_lower": lambda x, m: truncated_entropy_topk_tail(x, k=m),
        "topk_tail_midpoint": lambda x, m: truncated_entropy_topk_tail_midpoint(x, k=m),
        "topk_tail_uniform": lambda x, m: truncated_entropy_topk_tail_uniform(x, k=m),
        "topm_plus_tail_is": lambda x, m: entropy_topm_plus_tail_is(x, m=m, s=5),
        "topm_plus_cv_is": lambda x, m: entropy_topm_plus_cv_is(x, m=m, s=5),
    }
    
    # Results storage: variant_name -> list of overlap ratios
    results = {name: [] for name in entropy_variants.keys()}
    correlations = {name: [] for name in entropy_variants.keys()}
    
    # Track detailed statistics
    total_sequences = 0
    total_tokens = 0
    total_valid_tokens = 0
    sequence_lengths = []
    valid_token_ratios = []
    
    tot = 0
    last_log = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(dl):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits = model(ids, attention_mask=mask).logits
            pred = logits[:, :-1, :]
            valid = mask[:, 1:].bool()

            B, Lm1, _ = pred.shape
            
            # Compute exact entropy once
            ent_exact = []
            for i in range(B):
                pred32 = pred[i].float()
                ent_exact.append(token_entropy(pred32).cpu())
            ent_exact = torch.stack(ent_exact)  # [B, L-1]
            
            # Test each variant
            for variant_name, variant_fn in entropy_variants.items():
                ent_variant = []
                for i in range(B):
                    pred32 = pred[i].float()
                    ent_variant.append(variant_fn(pred32, args.m).cpu())
                ent_variant = torch.stack(ent_variant)  # [B, L-1]
                
                # Compute overlap and correlation for this batch
                for i in range(B):
                    v = valid[i].cpu()
                    n_valid = int(v.sum())
                    seq_len = len(v)
                    
                    # Track detailed statistics (only once per sequence, not per variant)
                    if variant_name == list(entropy_variants.keys())[0]:  # First variant only
                        total_sequences += 1
                        total_tokens += seq_len
                        total_valid_tokens += n_valid
                        sequence_lengths.append(seq_len)
                        valid_token_ratios.append(n_valid / seq_len if seq_len > 0 else 0.0)
                    
                    if n_valid < 3: 
                        continue
                    k = max(1, math.ceil(n_valid * args.k_percent / 100.0))
                    idx = torch.nonzero(v, as_tuple=False).squeeze(-1)

                    e1 = ent_exact[i, idx]
                    e2 = ent_variant[i, idx]

                    # Top-k overlap
                    top1 = idx[torch.topk(e1, k=k, largest=True, sorted=False).indices]
                    top2 = idx[torch.topk(e2, k=k, largest=True, sorted=False).indices]
                    set1 = set(top1.tolist())
                    set2 = set(top2.tolist())
                    inter = len(set1 & set2)
                    overlap_ratio = inter / float(k)
                    results[variant_name].append(overlap_ratio)
                    
                    # Correlation
                    if len(e1) > 1 and len(e2) > 1:
                        corr = torch.corrcoef(torch.stack([e1.flatten(), e2.flatten()]))[0,1].item()
                        if not math.isnan(corr):
                            correlations[variant_name].append(corr)
                    
            tot += B

            # Periodic progress logging
            now = time.time()
            if now - last_log > 3.0:
                last_log = now
                # Show progress for first variant
                first_variant = list(entropy_variants.keys())[0]
                n_samples = len(results[first_variant])
                if n_samples > 0:
                    avg = sum(results[first_variant]) / n_samples
                    log.info(f"Progress: batch {bi+1}/{len(dl)} | {total_sequences} sequences, {total_valid_tokens}/{total_tokens} valid tokens | sample avg overlap={avg:.4f}")

    if total_sequences == 0:
        log.warning("No valid sequences.")
        return
    
    dt = time.time() - t0
    log.info(f"Done in {dt:.1f}s")
    
    # Compute detailed statistics
    avg_seq_len = sum(sequence_lengths) / len(sequence_lengths) if sequence_lengths else 0
    avg_valid_ratio = sum(valid_token_ratios) / len(valid_token_ratios) if valid_token_ratios else 0
    
    # Print results table
    print("\n" + "="*90)
    print("ENTROPY APPROXIMATION COMPARISON RESULTS")
    print("="*90)
    print(f"Dataset: {args.dataset} | Model: {args.model}")
    print(f"Parameters: m={args.m}, k_percent={args.k_percent}%, batch_size={args.batch_size}")
    print(f"Total sequences: {total_sequences} | Total tokens: {total_tokens:,} | Valid tokens: {total_valid_tokens:,} ({total_valid_tokens/total_tokens*100:.1f}%)")
    print(f"Avg sequence length: {avg_seq_len:.1f} tokens | Avg valid ratio: {avg_valid_ratio:.1f}%")
    print()
    
    # Table header
    print(f"{'Variant':<25} {'N_Samples':<10} {'Avg_Overlap':<12} {'Std_Overlap':<12} {'Avg_Corr':<10} {'Std_Corr':<10}")
    print("-" * 90)
    
    # Table rows
    for variant_name in entropy_variants.keys():
        overlaps = results[variant_name]
        corrs = correlations[variant_name]
        
        if len(overlaps) > 0:
            avg_overlap = sum(overlaps) / len(overlaps)
            if len(overlaps) > 1:
                var_overlap = sum((x - avg_overlap)**2 for x in overlaps) / (len(overlaps) - 1)
                std_overlap = math.sqrt(var_overlap)
            else:
                std_overlap = 0.0
        else:
            avg_overlap = std_overlap = 0.0
            
        if len(corrs) > 0:
            avg_corr = sum(corrs) / len(corrs)
            if len(corrs) > 1:
                var_corr = sum((x - avg_corr)**2 for x in corrs) / (len(corrs) - 1)
                std_corr = math.sqrt(var_corr)
            else:
                std_corr = 0.0
        else:
            avg_corr = std_corr = 0.0
            
        print(f"{variant_name:<25} {len(overlaps):<10} {avg_overlap:<12.4f} {std_overlap:<12.4f} {avg_corr:<10.4f} {std_corr:<10.4f}")
    
    print("="*90)
    print("Legend:")
    print("  Avg_Overlap: Average ratio of top-k positions that match between exact and approximate entropy")
    print("  Avg_Corr:    Average Pearson correlation between exact and approximate entropy values")
    print("  Higher values indicate better approximation quality")
    print("  Valid tokens: Tokens that are not padding and contribute to next-token prediction")
    print()

    # Cleanup temp export if used
    if export_tmp is not None:
        try:
            export_tmp.cleanup()
        except Exception:
            pass

if __name__ == "__main__":
    main()