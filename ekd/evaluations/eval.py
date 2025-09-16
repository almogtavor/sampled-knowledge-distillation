#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()
# Optional logging imports
try:
    import wandb
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# ---------- Utility ----------

def run(cmd: List[str], env: Optional[Dict[str, str]] = None, cwd: Optional[str] = None, timeout: Optional[int] = None) -> Tuple[int, str]:
    print(f"\n$ {' '.join(cmd)}")
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, env=env, cwd=cwd, text=True, timeout=timeout)
        print(out)
        return 0, out
    except subprocess.CalledProcessError as e:
        print(e.output)
        return e.returncode, e.output
    except subprocess.TimeoutExpired as e:
        print(f"Command timed out after {timeout} seconds: {e}")
        return 124, f"Timeout after {timeout} seconds"

def find_latest_checkpoint(dir_path: Path) -> Optional[Path]:
    if not dir_path.exists():
        return None
    candidates = sorted(dir_path.glob("checkpoint_epoch*_step*.pt"))
    if not candidates:
        return None
    # Prefer the highest (epoch, step); fallback to newest mtime
    def key(p: Path):
        m = re.search(r"epoch(\d+)_step(\d+)", p.name)
        if m:
            return (int(m.group(1)), int(m.group(2)))
        return (-1, -1)
    candidates.sort(key=key)
    return candidates[-1]

def export_hf_model(base_model_dir: str, ckpt_path: Path, export_dir: Path) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting model from base '{base_model_dir}' with state_dict '{ckpt_path.name}' -> '{export_dir}'")

    # Use slow tokenizer to avoid potential parallelism/NFS hangs on clusters
    tok = AutoTokenizer.from_pretrained(base_model_dir, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_model_dir, dtype=torch.float16, device_map=None, trust_remote_code=True)
    chk = torch.load(ckpt_path, map_location="cpu")
    state = chk.get("model_state_dict")
    if state is None:
        raise ValueError(f"No 'model_state_dict' in {ckpt_path}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")

    model.save_pretrained(export_dir)
    tok.save_pretrained(export_dir)
    # Metadata for traceability
    meta = {
        "source_checkpoint": str(ckpt_path),
        "epoch": chk.get("epoch"),
        "step": chk.get("step"),
        "global_step": chk.get("global_step"),
        "distill_type": chk.get("distill_type"),
        "top_k_percent": chk.get("top_k_percent"),
        "export_time_utc": datetime.utcnow().isoformat() + "Z",
    }
    with open(export_dir / "kd_export_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

def ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)

def which(bin_name: str) -> bool:
    return shutil.which(bin_name) is not None

# ---------- LM-Eval (reasoning/QA/IF-Eval) ----------

LMEVAL_TASKS = [
    # # grade-school + math
    # "gsm8k", "svamp", "asdiv", "hendrycks_math",
    # # advanced reasoning
    # "gpqa",       # if your install expects gpqa_diamond or gpqa_main, the script retries
    # "bbh",        # group of BBH tasks
    # "agieval",    # group of AGIEval tasks
    # # QA
    # "squadv2", "nq_open", "hotpotqa",
    # # Instruction following (IFEval lives in an optional extra)
    # "ifeval",

    # Common sense reasoning (fast)
    "hellaswag", "winogrande", 
    # Knowledge and reasoning (medium)
    "arc_easy", "arc_challenge",
    # Math reasoning (slow)
    "gsm8k",
    # Reading comprehension (very slow)
    "mmlu",
]

# Fast subset for initial evaluation
LMEVAL_FAST_TASKS = [
    "hellaswag", "winogrande", "arc_easy"
]

def run_lmeval(model_dir: Path, tag: str, results_dir: Path, device: str, fast_mode: bool = True) -> Optional[Path]:
    out_dir = results_dir / f"lmeval_{tag}"
    ensure_dir(out_dir)

    # Choose task set based on mode
    tasks_to_try = list(LMEVAL_FAST_TASKS if fast_mode else LMEVAL_TASKS)
    print(f"Running {'fast' if fast_mode else 'full'} evaluation with tasks: {tasks_to_try}")
    
    # Use smaller, fixed batch size for stability
    base_args = [
        "lm-eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_dir},trust_remote_code=True",
        "--device", device,
        "--batch_size", "4",  # Fixed small batch size instead of auto
        "--output_path", str(out_dir),
        "--log_samples",
    ]

    # Run tasks individually to avoid hanging on problematic tasks
    good_any = False
    task_timeouts = {
        "hellaswag": 600,      # 10 minutes
        "winogrande": 300,     # 5 minutes
        "arc_easy": 600,       # 10 minutes
        "arc_challenge": 900,  # 15 minutes
        "gsm8k": 1800,         # 30 minutes
        "mmlu": 3600,          # 60 minutes
    }
    
    for task in tasks_to_try:
        timeout = task_timeouts.get(task, 1200)  # Default 20 minutes
        print(f"Running task: {task} (timeout: {timeout}s)")
        try:
            code_t, output = run(base_args + ["--tasks", task], timeout=timeout)
            if code_t == 0:
                good_any = True
                print(f"✅ Task {task} completed successfully")
            elif code_t == 124:  # Timeout
                print(f"⏰ Task {task} timed out after {timeout} seconds")
            else:
                print(f"❌ Task {task} failed with code {code_t}")
        except Exception as e:
            print(f"❌ Task {task} failed with exception: {e}")
            continue
    
    return out_dir if good_any else None

# ---------- Lighteval (summarization) ----------

LIGHTEVAL_TASKS = ["helm|summarization:cnn-dm", "helm|summarization:xsum"]

def run_lighteval(model_dir: Path, tag: str, results_dir: Path) -> Optional[Path]:
    out_path = results_dir / f"lighteval_{tag}.json"
    cmd = [
        "lighteval", "run",
        "--model", "hf", str(model_dir),
        "--tasks", ",".join(LIGHTEVAL_TASKS),
        "--results", str(out_path),
    ]
    code, _ = run(cmd)
    return out_path if code == 0 and out_path.exists() else None

# ---------- EvalPlus (code: HumanEval/MBPP) ----------

def run_evalplus(model_dir: Path, tag: str, datasets: List[str]) -> Dict[str, Optional[Path]]:
    out = {}
    for ds_name in datasets:
        cmd = [
            "evalplus.evaluate",
            "--model", str(model_dir),
            "--dataset", ds_name,
            "--backend", "hf",
            "--greedy",
        ]
        code, _ = run(cmd)
        # EvalPlus writes inside ./evalplus_results/<ds_name>/...
        results_root = Path("evalplus_results") / ds_name
        out[ds_name] = results_root if results_root.exists() else None
    return out

# ---------- AlpacaEval 2 (LC win-rates) ----------

def run_alpacaeval(model_dir: Path, tag: str, results_dir: Path) -> Optional[Path]:
    if os.getenv("OPENAI_API_KEY") is None:
        print("Skipping AlpacaEval 2: OPENAI_API_KEY not set.")
        return None
    out_dir = results_dir / f"alpacaeval_{tag}"
    ensure_dir(out_dir)
    # Use the built-in command that runs prompts against a local HF model and annotates with GPT-4 turbo.
    # See repo docs for 'evaluate_from_model'.
    cmd = [
        "alpaca_eval",
        "evaluate_from_model",
        "--model", str(model_dir),
        "--output_path", str(out_dir),
    ]
    code, _ = run(cmd)
    return out_dir if code == 0 else None

# ---------- Safety (JailbreakBench + HarmBench) ----------

def run_jailbreakbench(model_dir: Path, tag: str, results_dir: Path) -> Optional[Path]:
    # Most JBB recipes assume a vLLM/OpenAI-compatible endpoint. If you already serve the model,
    # set JBB_BASE_URL and JBB_MODEL envs; otherwise we skip.
    base_url = os.getenv("JBB_BASE_URL")
    model_name = os.getenv("JBB_MODEL")
    if not which("python") or base_url is None or model_name is None:
        print("Skipping JailbreakBench: set JBB_BASE_URL and JBB_MODEL to use an OpenAI-style endpoint.")
        return None
    out_dir = results_dir / f"jbb_{tag}"
    ensure_dir(out_dir)
    # Minimal example using their Python API via a small shim
    shim = f"""
import json, os
import jailbreakbench as jbb

base_url = os.environ.get("JBB_BASE_URL")
model = os.environ.get("JBB_MODEL")
prompts = jbb.load_default_prompts()  # defaults; customize as needed
evaluation = jbb.evaluate_prompts(prompts, llm_provider="litellm", base_url=base_url, model=model)
with open("{(out_dir / 'jbb_results.json').as_posix()}", "w") as f:
    json.dump(evaluation, f)
print("Wrote JBB results")
"""
    code, _ = run([sys.executable, "-c", shim])
    return out_dir if code == 0 else None

def run_harmbench(model_dir: Path, tag: str, results_dir: Path) -> Optional[Path]:
    # HarmBench expects a config; we only run if a config path is provided.
    config = os.getenv("HARMBENCH_CONFIG")
    if config is None:
        print("Skipping HarmBench: set HARMBENCH_CONFIG to a YAML that points to your HF model.")
        return None
    out_dir = results_dir / f"harmbench_{tag}"
    ensure_dir(out_dir)
    code, _ = run([sys.executable, "-m", "harmbench", "--config", config])
    # Users typically write their own output path in the config. We just mark the run directory.
    return out_dir if code == 0 else None

# ---------- Results aggregation -> LaTeX ----------

def collect_lmeval_metrics(lmeval_dir: Path) -> Dict[str, Dict[str, float]]:
    results = {}
    if not lmeval_dir or not lmeval_dir.exists():
        return results
    for p in lmeval_dir.rglob("results.json"):
        try:
            blob = json.load(open(p))
            # Typical LM-Eval structure: {"results": {"task": {"metric": value, ...}}, ...}
            r = blob.get("results", {})
            for task, metrics in r.items():
                results[task] = {}
                for mk, mv in metrics.items():
                    if isinstance(mv, (int, float)):
                        results[task][mk] = float(mv)
        except Exception as e:
            print(f"Failed to parse {p}: {e}")
    return results

def collect_lighteval_metrics(out_file: Path) -> Dict[str, Dict[str, float]]:
    results = {}
    if not out_file or not out_file.exists():
        return results
    try:
        blob = json.load(open(out_file))
        # Lighteval dumps a dict of tasks -> metrics
        for task, task_res in blob.items():
            results[task] = {}
            for mk, mv in task_res.items():
                if isinstance(mv, (int, float)):
                    results[task][mk] = float(mv)
    except Exception as e:
        print(f"Failed to parse {out_file}: {e}")
    return results

def collect_evalplus_metrics(root: Optional[Path], ds_name: str) -> Dict[str, Dict[str, float]]:
    results = {}
    if not root or not root.exists():
        return results
    # Try common summary files
    candidates = list(root.rglob("summary.json")) + list(root.rglob("report.json")) + list(root.rglob("scores.json"))
    if not candidates:
        return results
    # Pick the newest
    p = max(candidates, key=lambda x: x.stat().st_mtime)
    try:
        blob = json.load(open(p))
        # Heuristic: look for pass@k variants
        task = f"{ds_name}"
        results[task] = {}
        for k in ["pass@1", "pass@5", "pass@10", "pass@100", "mbpp_score", "humaneval_score"]:
            if k in blob and isinstance(blob[k], (int, float)):
                results[task][k] = float(blob[k])
        # Fallback: search numeric fields
        for k, v in blob.items():
            if k not in results[task] and isinstance(v, (int, float)) and ("pass" in k or "score" in k or "acc" in k):
                results[task][k] = float(v)
    except Exception as e:
        print(f"Failed to parse EvalPlus summary at {p}: {e}")
    return results

def collect_alpacaeval_metrics(out_dir: Optional[Path]) -> Dict[str, Dict[str, float]]:
    results = {}
    if not out_dir or not out_dir.exists():
        return results
    # Look for leaderboard.json or metrics.json
    candidates = list(out_dir.rglob("*.json"))
    for p in candidates:
        try:
            blob = json.load(open(p))
            # Search for LC metrics
            if isinstance(blob, dict):
                task = "AlpacaEval2"
                vals = {}
                for k in ["length_controlled_win_rate", "win_rate", "pairwise_win_rate", "length_controlled"]:
                    if k in blob and isinstance(blob[k], (int, float)):
                        vals[k] = float(blob[k])
                if vals:
                    results[task] = vals
                    break
        except Exception:
            continue
    return results

def collect_simple_json(root: Optional[Path], task_name: str, filename: str) -> Dict[str, Dict[str, float]]:
    results = {}
    if not root:
        return results
    p = root / filename
    if p.exists():
        try:
            blob = json.load(open(p))
            if isinstance(blob, dict):
                results[task_name] = {}
                for k, v in blob.items():
                    if isinstance(v, (int, float)):
                        results[task_name][k] = float(v)
        except Exception as e:
            print(f"Failed to parse {p}: {e}")
    return results

def merge_model_results(per_source: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    merged = {}
    for src in per_source:
        for task, metrics in src.items():
            merged.setdefault(task, {})
            for mk, mv in metrics.items():
                merged[task][mk] = mv
    return merged

def log_to_wandb(tag: str, merged_metrics: Dict[str, Dict[str, float]], project: str = "ekd-evaluation") -> None:
    """Log evaluation metrics to Weights & Biases."""
    if not WANDB_AVAILABLE:
        print("W&B not available, skipping wandb logging")
        return
    
    try:
        wandb.init(project=project, name=f"eval-{tag}", reinit=True)
        
        # Flatten metrics for W&B
        wandb_metrics = {}
        for task, metrics in merged_metrics.items():
            for metric, val in metrics.items():
                wandb_metrics[f"{task}/{metric}"] = val
        
        wandb.log(wandb_metrics)
        wandb.finish()
        print(f"✓ Logged {len(wandb_metrics)} metrics to W&B project '{project}'")
    except Exception as e:
        print(f"Failed to log to W&B: {e}")

def log_to_tensorboard(tag: str, merged_metrics: Dict[str, Dict[str, float]], log_dir: str = "tb_logs") -> None:
    """Log evaluation metrics to TensorBoard."""
    if not TENSORBOARD_AVAILABLE:
        print("TensorBoard not available, skipping tensorboard logging")
        return
    
    try:
        tb_path = Path(log_dir) / tag
        tb_path.mkdir(parents=True, exist_ok=True)
        
        writer = SummaryWriter(log_dir=str(tb_path))
        
        # Log each metric as a scalar
        metric_count = 0
        for task, metrics in merged_metrics.items():
            for metric, val in metrics.items():
                writer.add_scalar(f"{task}/{metric}", val)
                metric_count += 1
        
        writer.close()
        print(f"✓ Logged {metric_count} metrics to TensorBoard at {tb_path}")
    except Exception as e:
        print(f"Failed to log to TensorBoard: {e}")

def print_latex_table(all_models_metrics: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    # Collect union of metric names
    all_metrics = sorted({m for model in all_models_metrics.values() for task in model.values() for m in task.keys()})
    tasks = sorted({t for model in all_models_metrics.values() for t in model.keys()})
    print("\n% ---- LaTeX table (booktabs) ----")
    print(r"\begin{table}")
    print(r"\centering")
    print(r"\caption{Evaluation across public benchmarks (same decoding params across systems).}")
    print(r"\begin{tabular}{l" + "c" * (len(all_metrics) * len(all_models_metrics)) + r"}")
    print(r"\toprule")
    # Header has metrics per model
    header = ["Task"]
    model_names = list(all_models_metrics.keys())
    for mname in model_names:
        for met in all_metrics:
            header.append(f"{mname}:{met}")
    print(" & ".join(header) + r" \\")
    print(r"\midrule")
    for t in tasks:
        row = [t]
        for mname in model_names:
            mm = all_models_metrics[mname].get(t, {})
            for met in all_metrics:
                val = mm.get(met, None)
                if isinstance(val, float):
                    row.append(f"{val:.2f}")
                elif isinstance(val, int):
                    row.append(str(val))
                else:
                    row.append("-")
        print(" & ".join(row) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

# ---------- Main pipeline ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_dir", type=str, required=True, help="HF dir used to initialize the student (architecture + tokenizer).")
    parser.add_argument("--vanilla_ckpt_dir", type=str, default="kd_vanilla_run_out_model")
    parser.add_argument("--ekd_ckpt_dir", type=str, default="kd_ekd_run_out_model")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--work_dir", type=str, default="eval_runs")
    # Optional: evaluate a single tag+checkpoint instead of both latest
    parser.add_argument("--tag", type=str, choices=["vanilla", "ekd"], help="Evaluate only this run type (vanilla or ekd).")
    parser.add_argument("--checkpoint_path", type=str, help="Path to a specific checkpoint .pt to evaluate (used with --tag).")
    # Logging configuration
    parser.add_argument("--wandb_project", type=str, default="ekd-evaluation", help="W&B project name for logging.")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable W&B logging.")
    parser.add_argument("--disable_tensorboard", action="store_true", help="Disable TensorBoard logging.")
    parser.add_argument("--fast_eval", action="store_true", help="Use fast evaluation (subset of tasks) to avoid hanging.")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    exports_dir = work_dir / "exports"
    results_dir = work_dir / "results"
    ensure_dir(exports_dir)
    ensure_dir(results_dir)

    # Prepare models
    model_specs = []
    if args.tag and args.checkpoint_path:
        tag = args.tag
        ckpt = Path(args.checkpoint_path)
        if not ckpt.exists():
            print(f"Error: checkpoint not found: {ckpt}")
            sys.exit(2)
        # Include checkpoint stem in export dir to avoid collisions
        export_dir = exports_dir / f"{tag}_export_{ckpt.stem}"
        export_hf_model(args.base_model_dir, ckpt, export_dir)
        model_specs.append((tag, export_dir))
    else:
        for tag, ckdir in [("vanilla", args.vanilla_ckpt_dir), ("ekd", args.ekd_ckpt_dir)]:
            ckpt_dir = Path(ckdir)
            ckpt = find_latest_checkpoint(ckpt_dir)
            if ckpt is None:
                print(f"Warning: no checkpoints in {ckpt_dir}, skipping {tag}")
                continue
            export_dir = exports_dir / f"{tag}_export"
            export_hf_model(args.base_model_dir, ckpt, export_dir)
            model_specs.append((tag, export_dir))

    if not model_specs:
        print("No models exported. Exiting.")
        sys.exit(1)

    all_models_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}

    for tag, model_dir in model_specs:
        print(f"\n=== Running benchmarks for {tag} ===")

        # LM-Eval
        lmeval_root = run_lmeval(model_dir, tag, results_dir, args.device, fast_mode=args.fast_eval)
        lmeval_metrics = collect_lmeval_metrics(lmeval_root) if lmeval_root else {}

        # Lighteval summarization
        lighteval_file = run_lighteval(model_dir, tag, results_dir)
        lighteval_metrics = collect_lighteval_metrics(lighteval_file)

        # EvalPlus (HumanEval/MBPP)
        evalplus_roots = run_evalplus(model_dir, tag, ["humaneval", "mbpp"])
        he_metrics = collect_evalplus_metrics(evalplus_roots.get("humaneval"), "HumanEval+")
        mbpp_metrics = collect_evalplus_metrics(evalplus_roots.get("mbpp"), "MBPP+")

        # AlpacaEval 2
        alpaca_dir = run_alpacaeval(model_dir, tag, results_dir)
        alpaca_metrics = collect_alpacaeval_metrics(alpaca_dir)

        # Safety (optional)
        jbb_dir = run_jailbreakbench(model_dir, tag, results_dir)  # may be None
        jbb_metrics = collect_simple_json(jbb_dir, "JailbreakBench", "jbb_results.json")
        hb_dir = run_harmbench(model_dir, tag, results_dir)        # may be None
        hb_metrics = collect_simple_json(hb_dir, "HarmBench", "harmbench_results.json")

        merged = merge_model_results([
            lmeval_metrics,
            lighteval_metrics,
            he_metrics,
            mbpp_metrics,
            alpaca_metrics,
            jbb_metrics,
            hb_metrics
        ])
        all_models_metrics[tag] = merged

        # Log to W&B and TensorBoard
        print(f"\n=== Logging metrics for {tag} ===")
        if not args.disable_wandb:
            log_to_wandb(tag, merged, project=args.wandb_project)
        if not args.disable_tensorboard:
            log_to_tensorboard(tag, merged, log_dir=str(work_dir / "tb_logs"))

    # Print LaTeX
    print_latex_table(all_models_metrics)

if __name__ == "__main__":
    main()
