#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCH_INFERENCE_MODE", "1")
if "HF_DATASETS_CACHE" not in os.environ:
    _tmp = os.environ.get("TMPDIR", "/tmp")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(_tmp, "hf_datasets")

# ---------- Logging imports ----------
try:
    from ekd.logging.wandb_utils import log_evaluation_to_wandb, log_evaluation_to_tensorboard
except ImportError:
    log_evaluation_to_wandb = log_evaluation_to_tensorboard = lambda *args, **kwargs: None

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

def run_async(cmd: List[str], env: Optional[Dict[str, str]] = None) -> subprocess.Popen:
    print(f"\n$ (async) {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)

def wait_with_timeout(proc: subprocess.Popen, timeout: int) -> Tuple[int, str]:
    try:
        out, _ = proc.communicate(timeout=timeout)
        rc = proc.returncode
        if rc is None:
            rc = 0
        print(out or "")
        return rc, out or ""
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
        print(out or "")
        return 124, out or ""

def find_latest_checkpoint(dir_path: Path) -> Optional[Path]:
    if not dir_path.exists():
        return None
    candidates = sorted(dir_path.glob("checkpoint_epoch*_step*.pt"))
    if not candidates:
        return None
    def key(p: Path):
        m = re.search(r"epoch(\d+)_step(\d+)", p.name)
        if m:
            return (int(m.group(1)), int(m.group(2)))
        return (-1, -1)
    candidates.sort(key=key)
    return candidates[-1]

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def export_hf_model(base_model_dir: str, ckpt_path: Path, export_dir: Path) -> None:
    """Export a HF-ready directory from base weights + checkpoint.
       Skips work if kd_export_meta.json matches the checkpoint hash.
    """
    export_dir.mkdir(parents=True, exist_ok=True)
    meta_path = export_dir / "kd_export_meta.json"
    ckpt_hash = sha256_file(ckpt_path)
    if meta_path.exists():
        try:
            old = json.load(open(meta_path))
            if old.get("ckpt_sha256") == ckpt_hash:
                print(f"[export] Cache hit for {export_dir} (ckpt unchanged). Skipping export.")
                return
        except Exception:
            pass
    print(f"Exporting model from base '{base_model_dir}' with state_dict '{ckpt_path.name}' -> '{export_dir}'")
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
    meta = {
        "source_checkpoint": str(ckpt_path),
        "epoch": chk.get("epoch"),
        "step": chk.get("step"),
        "global_step": chk.get("global_step"),
        "distill_type": chk.get("distill_type"),
        "top_k_percent": chk.get("top_k_percent"),
        "export_time_utc": datetime.utcnow().isoformat() + "Z",
        "ckpt_sha256": ckpt_hash,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[export] Wrote {meta_path}")

def ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)

def which(bin_name: str) -> bool:
    return shutil.which(bin_name) is not None

# ---------- GPU helpers ----------
def visible_gpu_ids() -> List[int]:
    """Return the list of visible GPU ids from CUDA_VISIBLE_DEVICES or torch."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        try:
            return [int(x) for x in cvd.split(",") if x != ""]
        except Exception:
            pass
    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []

def pick_gpu_pool(max_workers: Optional[int] = None) -> List[int]:
    ids = visible_gpu_ids()
    if max_workers is not None:
        ids = ids[:max_workers]
    if not ids:
        print("No CUDA devices visible, defaulting to CPU (will be slow).")
    return ids

# ==========================================================
#                      SUITE DEFINITIONS
# ==========================================================

# NOTE: Task names follow lm-eval harness conventions. Some tasks may be unavailable
# in older harness versions; failures are tolerated and reported.
# LIGHT suite (coffee-break): strict caps via --limit (first-N selection by harness)
LIGHT_LMEVAL_TASKS: List[Tuple[str, Optional[int]]] = [
    ("gsm8k", 100),            # accuracy
    ("svamp", 100),            # accuracy
    ("arc_challenge", 300),    # acc_norm
    ("hellaswag", 500),        # acc_norm
    ("gpqa_diamond", None),    # accuracy (tiny)
    ("aime24", None),          # exact-match (tiny, if available)
]
# Optional tiny adds (off by default): BoolQ 200, HumanEval full
LIGHT_ENABLE_OPTIONALS = os.environ.get("LIGHT_EXTRAS", "0") == "1"
LIGHT_OPTIONALS: List[Tuple[str, Optional[int]]] = [
    ("boolq", 200),
]

# HEAVY suite (paper): broad coverage; full sets (no --limit) unless noted
HEAVY_LMEVAL_TASKS: List[Tuple[str, Optional[int]]] = [
    # Reasoning & math
    ("gsm8k", None),
    ("svamp", None),
    ("asdiv", None),               # arithmetic subset (ASDiv-A handled inside task)
    ("hendrycks_math", None),      # MATH
    ("aime24", None),
    ("aime25", None),
    ("olympiadbench", None),
    ("gpqa_diamond", None),
    # General reasoning
    ("bbh", None),                  # BIG-Bench Hard group
    ("agieval", None),              # AGIEval group
    # QA / multi-hop
    ("squadv2", None),
    ("hotpotqa", None),
    ("nq_open", None),
    # Commonsense
    ("hellaswag", None),
    ("arc_challenge", None),
]

# Per-task timeouts (seconds). Be generous on heavy tasks.
TASK_TIMEOUTS = {
    # light-ish baselines
    "boolq": 600,
    "arc_challenge": 1500,
    "hellaswag": 1200,
    "svamp": 900,
    "gsm8k": 1200,
    "gpqa_diamond": 900,
    "aime24": 900,
    "aime25": 900,
    # heavy add-ons
    "asdiv": 1200,
    "hendrycks_math": 3600,
    "olympiadbench": 5400,
    "bbh": 5400,
    "agieval": 5400,
    "squadv2": 2400,
    "hotpotqa": 3600,
    "nq_open": 3600,
}

# Summarization (Lighteval)
LIGHTEVAL_TASKS = ["helm|summarization:cnn_dailymail", "helm|summarization:xsum"]

# ==========================================================
#                  LM-Eval runner with suites
# ==========================================================
def run_lmeval_suite(
    model_dir: Path,
    tag: str,
    results_dir: Path,
    gpu_ids: List[int],
    suite: str,
) -> Optional[Path]:
    """Run the requested LM-Eval suite (light/heavy). Executes one task per GPU in waves."""
    out_dir = results_dir / f"lmeval_{suite}_{tag}"
    ensure_dir(out_dir)

    tasks_with_limits: List[Tuple[str, Optional[int]]]
    if suite == "light":
        tasks_with_limits = list(LIGHT_LMEVAL_TASKS)
        if LIGHT_ENABLE_OPTIONALS:
            tasks_with_limits += list(LIGHT_OPTIONALS)
    else:
        tasks_with_limits = list(HEAVY_LMEVAL_TASKS)

    if not tasks_with_limits:
        print("No LM-Eval tasks selected.")
        return None

    base_args = [
        "lm-eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_dir},trust_remote_code=True",
        "--batch_size", "4",
        "--output_path", str(out_dir),
    ]

    # CPU fallback (sequential)
    if not gpu_ids:
        print("[lm-eval] No GPUs detected; running sequentially on CPU.")
        good_any = False
        for (task, limit) in tasks_with_limits:
            args = base_args + ["--tasks", task, "--device", "cpu"]
            if isinstance(limit, (int, float)):
                args += ["--limit", str(limit)]
            timeout = TASK_TIMEOUTS.get(task, 3600)
            code, _ = run(args, timeout=timeout)
            good_any |= (code == 0)
        return out_dir if good_any else None

    # Parallel across physical GPUs (mask each process to one GPU)
    good_any = False
    i = 0
    while i < len(tasks_with_limits):
        procs = []
        wave = tasks_with_limits[i : i + len(gpu_ids)]
        wave_names = [t for (t, _) in wave]
        print(f"[lm-eval] Launching wave: {wave_names} on GPUs {gpu_ids}")
        for (task, limit), gid in zip(wave, gpu_ids):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gid)  # inside, cuda:0 maps to this physical GPU
            cmd = base_args + ["--tasks", task, "--device", "cuda:0"]
            if isinstance(limit, (int, float)):
                cmd += ["--limit", str(limit)]
            p = run_async(cmd, env=env)
            timeout = TASK_TIMEOUTS.get(task, 3600)
            procs.append((task, p, timeout))
        for task, p, to in procs:
            rc, out = wait_with_timeout(p, timeout=to)
            if rc == 0:
                print(f"✅ Task {task} completed successfully")
                good_any = True
            elif rc == 124:
                print(f"⏰ Task {task} timed out after {to} seconds")
            else:
                print(f"❌ Task {task} failed with code {rc}")
        i += len(gpu_ids)
    return out_dir if good_any else None

# ---------- Lighteval (summarization) ----------
def run_lighteval(model_dir: Path, tag: str, results_dir: Path, suite: str) -> Optional[Path]:
    """Run summarization only for HEAVY suite (CNN/DM, XSum)."""
    if suite != "heavy":
        print("[lighteval] Skipping summarization for light suite.")
        return None
    out_path = results_dir / f"lighteval_{tag}.json"
    tasks = [t.replace("-", "_") for t in LIGHTEVAL_TASKS]  # normalize hyphen → underscore
    cmd = [
        "lighteval",
        "--model", "hf", str(model_dir),
        "--tasks", ",".join(tasks),
        "--results", str(out_path),
    ]
    code, _ = run(cmd)
    if code != 0:
        # Fallback: module invocation
        cmd = [
            sys.executable, "-m", "lighteval",
            "--model", "hf", str(model_dir),
            "--tasks", ",".join(tasks),
            "--results", str(out_path),
        ]
        code, _ = run(cmd)
    return out_path if code == 0 and out_path.exists() else None

# ---------- EvalPlus (code: HumanEval/MBPP) ----------
def run_evalplus(model_dir: Path, tag: str, datasets: List[str], suite: str) -> Dict[str, Optional[Path]]:
    """Run code-gen only for HEAVY suite by default."""
    if suite != "heavy":
        print("[evalplus] Skipping code-gen for light suite.")
        return {}
    out = {}
    evalplus_home = Path(os.environ.get("TMPDIR", "/tmp")) / "evalplus_home"
    (evalplus_home / ".cache").mkdir(parents=True, exist_ok=True)
    for ds_name in datasets:
        cmd = [
            "evalplus.evaluate",
            "--model", str(model_dir),
            "--dataset", ds_name,
            "--backend", "hf",
            "--greedy",
        ]
        env = os.environ.copy()
        env["HOME"] = str(evalplus_home)             # avoid $HOME
        env["XDG_CACHE_HOME"] = str(evalplus_home / ".cache")
        env.setdefault("EVALPLUS_TRUST_REMOTE_CODE", "1")
        code, _ = run(cmd, env=env)
        results_root = Path("evalplus_results") / ds_name
        out[ds_name] = results_root if results_root.exists() and code == 0 else None
    return out

# ---------- AlpacaEval 2 (LC win-rates) ----------
def run_alpacaeval(model_dir: Path, tag: str, results_dir: Path, suite: str) -> Optional[Path]:
    if suite != "heavy":
        print("[alpacaeval] Skipping for light suite.")
        return None
    if os.getenv("OPENAI_API_KEY") is None:
        print("Skipping AlpacaEval 2: OPENAI_API_KEY not set.")
        return None
    out_dir = results_dir / f"alpacaeval_{tag}"
    ensure_dir(out_dir)
    cmd = [
        "alpaca_eval",
        "evaluate_from_model",
        "--model", str(model_dir),
        "--output_path", str(out_dir),
    ]
    code, _ = run(cmd)
    return out_dir if code == 0 else None

# ---------- IF-Eval (instruction-following compliance) ----------
def run_ifeval(model_dir: Path, tag: str, results_dir: Path, suite: str) -> Optional[Path]:
    """Run IF-Eval if available and only for HEAVY suite."""
    if suite != "heavy":
        print("[ifeval] Skipping for light suite.")
        return None
    if not which("ifeval"):
        print("Skipping IF-Eval: 'ifeval' CLI not found (pip install may be required).")
        return None
    out_dir = results_dir / f"ifeval_{tag}"
    ensure_dir(out_dir)
    # Minimal CLI; adjust if your local ifeval expects different flags.
    cmd = ["ifeval", "--model", str(model_dir), "--output", str(out_dir)]
    code, _ = run(cmd)
    return out_dir if code == 0 else None

# ---------- Safety (JailbreakBench + HarmBench) ----------
def run_jailbreakbench(model_dir: Path, tag: str, results_dir: Path, suite: str) -> Optional[Path]:
    if suite != "heavy":
        print("[jbb] Skipping for light suite.")
        return None
    base_url = os.getenv("JBB_BASE_URL")
    model_name = os.getenv("JBB_MODEL")
    if not which("python") or base_url is None or model_name is None:
        print("Skipping JailbreakBench: set JBB_BASE_URL and JBB_MODEL to use an OpenAI-style endpoint.")
        return None
    out_dir = results_dir / f"jbb_{tag}"
    ensure_dir(out_dir)
    shim = f"""
import json, os
import jailbreakbench as jbb
base_url = os.environ.get("JBB_BASE_URL")
model = os.environ.get("JBB_MODEL")
prompts = jbb.load_default_prompts()
evaluation = jbb.evaluate_prompts(prompts, llm_provider="litellm", base_url=base_url, model=model)
with open("{(out_dir / 'jbb_results.json').as_posix()}", "w") as f:
    json.dump(evaluation, f)
print("Wrote JBB results")
"""
    code, _ = run([sys.executable, "-c", shim])
    return out_dir if code == 0 else None

def run_harmbench(model_dir: Path, tag: str, results_dir: Path, suite: str) -> Optional[Path]:
    if suite != "heavy":
        print("[harmbench] Skipping for light suite.")
        return None
    if importlib.util.find_spec("harmbench") is None:
        print("Skipping HarmBench: package 'harmbench' not installed.")
        return None
    config = os.getenv("HARMBENCH_CONFIG")
    if config is None:
        print("Skipping HarmBench: set HARMBENCH_CONFIG to a YAML that points to your HF model.")
        return None
    out_dir = results_dir / f"harmbench_{tag}"
    ensure_dir(out_dir)
    code, _ = run([sys.executable, "-m", "harmbench", "--config", config])
    return out_dir if code == 0 else None


# ---------- Results aggregation ----------
def collect_lmeval_metrics(lmeval_dir: Path) -> Dict[str, Dict[str, float]]:
    results = {}
    if not lmeval_dir or not lmeval_dir.exists():
        return results
    for p in lmeval_dir.rglob("results.json"):
        try:
            blob = json.load(open(p))
            r = blob.get("results", {})
            for task, metrics in r.items():
                results.setdefault(task, {})
                for mk, mv in metrics.items():
                    if isinstance(mv, dict) and "value" in mv:
                        if isinstance(mv["value"], (int, float)):
                            results[task][mk] = float(mv["value"])
                        if "stderr" in mv and isinstance(mv["stderr"], (int, float)):
                            results[task][f"{mk}_stderr"] = float(mv["stderr"])
                    elif isinstance(mv, (int, float)):
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
    candidates = list(root.rglob("summary.json")) + list(root.rglob("report.json")) + list(root.rglob("scores.json"))
    if not candidates:
        return results
    p = max(candidates, key=lambda x: x.stat().st_mtime)
    try:
        blob = json.load(open(p))
        task = f"{ds_name}"
        results[task] = {}
        for k in ["pass@1", "pass@5", "pass@10", "pass@100", "mbpp_score", "humaneval_score"]:
            if k in blob and isinstance(blob[k], (int, float)):
                results[task][k] = float(blob[k])
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
    candidates = list(out_dir.rglob("*.json"))
    for p in candidates:
        try:
            blob = json.load(open(p))
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

def print_latex_table(all_models_metrics: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    all_metrics = sorted({m for model in all_models_metrics.values() for task in model.values() for m in task.keys()})
    tasks = sorted({t for model in all_models_metrics.values() for t in model.keys()})
    print("\n% ---- LaTeX table (booktabs) ----")
    print(r"\begin{table}")
    print(r"\centering")
    print(r"\caption{Evaluation across public benchmarks (same decoding params across systems).}")
    print(r"\begin{tabular}{l" + "c" * (len(all_metrics) * len(all_models_metrics)) + r"}")
    print(r"\toprule")
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
    # Parallel/GPU controls
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="Comma-separated physical GPU ids for parallel lm-eval (e.g., '0,1,2'). Defaults to visible GPUs.")
    parser.add_argument("--max_parallel", type=int, default=None,
                        help="Cap the number of parallel lm-eval workers (<= number of provided GPUs).")
    parser.add_argument("--work_dir", type=str, default="eval_runs")
    # Logging configuration (default project updated per request)
    parser.add_argument("--wandb_project", type=str, default="selective-entropy-knowledge-distillation",
                        help="W&B project slug (e.g., 'selective-entropy-knowledge-distillation').")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable W&B logging.")
    parser.add_argument("--disable_tensorboard", action="store_true", help="Disable TensorBoard logging.")
    # Suite selection
    parser.add_argument("--suite", type=str, choices=["light", "heavy"], default="light",
                        help="Evaluation suite to run: 'light' (quick) or 'heavy' (paper).")
    # Optional: evaluate a single tag+checkpoint instead of both latest
    parser.add_argument("--tag", type=str, choices=["vanilla", "ekd"], help="Evaluate only this run type (vanilla or ekd).")
    parser.add_argument("--checkpoint_path", type=str, help="Path to a specific checkpoint .pt to evaluate (used with --tag).")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    exports_dir = work_dir / "exports"
    results_dir = work_dir / "results"
    ensure_dir(exports_dir)
    ensure_dir(results_dir)

    # Build GPU pool
    if args.gpu_ids:
        gpu_pool = [int(x) for x in args.gpu_ids.split(",") if x.strip() != ""]
    else:
        gpu_pool = visible_gpu_ids()
    if args.max_parallel is not None and args.max_parallel > 0:
        gpu_pool = gpu_pool[: args.max_parallel]
    print(f"[gpu] Using GPUs: {gpu_pool}" if gpu_pool else "[gpu] No GPUs detected/selected.")

    # Prepare models (export with caching)
    model_specs: List[Tuple[str, Path]] = []
    if args.tag and args.checkpoint_path:
        tag = args.tag
        ckpt = Path(args.checkpoint_path)
        if not ckpt.exists():
            print(f"Error: checkpoint not found: {ckpt}")
            sys.exit(2)
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
        print(f"\n=== Running benchmarks for {tag} (suite={args.suite}) ===")

        # LM-Eval (parallel across GPUs according to the chosen suite)
        lmeval_root = run_lmeval_suite(
            model_dir=model_dir,
            tag=tag,
            results_dir=results_dir,
            gpu_ids=gpu_pool,
            suite=args.suite,
        )
        lmeval_metrics = collect_lmeval_metrics(lmeval_root) if lmeval_root else {}

        # Summarization (HEAVY only)
        lighteval_file = run_lighteval(model_dir, tag, results_dir, suite=args.suite)
        lighteval_metrics = collect_lighteval_metrics(lighteval_file)

        # Code-gen (HEAVY only)
        evalplus_roots = run_evalplus(model_dir, tag, ["humaneval", "mbpp"], suite=args.suite)
        he_metrics = collect_evalplus_metrics(evalplus_roots.get("humaneval"), "HumanEval+")
        mbpp_metrics = collect_evalplus_metrics(evalplus_roots.get("mbpp"), "MBPP+")

        # Instruction following (HEAVY only, if available)
        ifeval_dir = run_ifeval(model_dir, tag, results_dir, suite=args.suite)
        ifeval_metrics = collect_simple_json(ifeval_dir, "IF-Eval", "results.json")

        # Instruction-following win-rate (HEAVY only, requires API key)
        alpaca_dir = run_alpacaeval(model_dir, tag, results_dir, suite=args.suite)
        alpaca_metrics = collect_alpacaeval_metrics(alpaca_dir)

        # Safety (HEAVY only)
        jbb_dir = run_jailbreakbench(model_dir, tag, results_dir, suite=args.suite)
        jbb_metrics = collect_simple_json(jbb_dir, "JailbreakBench", "jbb_results.json")
        hb_dir = run_harmbench(model_dir, tag, results_dir, suite=args.suite)
        hb_metrics = collect_simple_json(hb_dir, "HarmBench", "harmbench_results.json")

        merged = merge_model_results([
            lmeval_metrics,
            lighteval_metrics,
            he_metrics,
            mbpp_metrics,
            ifeval_metrics,
            alpaca_metrics,
            jbb_metrics,
            hb_metrics
        ])
        all_models_metrics[tag] = merged

        # Log results to W&B and TensorBoard
        try:
            if not args.disable_wandb:
                log_evaluation_to_wandb(tag, merged, args.wandb_project)
            if not args.disable_tensorboard:
                log_evaluation_to_tensorboard(tag, merged, str(work_dir / "tb_logs"))
        except Exception as e:
            print(f"Error logging {tag} metrics: {e}")

    # Print LaTeX summary
    print_latex_table(all_models_metrics)

if __name__ == "__main__":
    main()
