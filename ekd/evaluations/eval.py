#!/usr/bin/env python3
import argparse
import math
import json
import os
import re
import shutil
import subprocess
import sys
import hashlib
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from ekd.run_registry import compute_params_hash, upsert_eval_results
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import importlib

load_dotenv()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCH_INFERENCE_MODE", "1")
if "HF_DATASETS_CACHE" not in os.environ:
    _tmp = os.environ.get("TMPDIR", "/tmp")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(_tmp, "hf_datasets")

# ---------- Logging imports ----------
try:
    from ekd.logging.wandb_utils import (
        WandBLogger,
        log_evaluation_to_wandb,
        log_evaluation_to_tensorboard,
    )
except ImportError:
    WandBLogger = None  # type: ignore
    log_evaluation_to_wandb = log_evaluation_to_tensorboard = lambda *args, **kwargs: None

# ---------- Pydantic Models for Configuration ----------
class GenerationConfig(BaseModel):
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.0

class MetricConfig(BaseModel):
    metric: str
    aggregation: str = "mean"
    higher_is_better: bool = True

class TaskConfig(BaseModel):
    task: str  # lm-eval expects "task" field
    dataset_path: str
    dataset_name: Optional[str] = None
    test_split: Optional[str] = None
    validation_split: Optional[str] = None
    train_split: Optional[str] = None
    output_type: str = "generate_until"
    generation_kwargs: GenerationConfig = Field(default_factory=GenerationConfig)
    metric_list: List[MetricConfig] = Field(default_factory=list)
    process_docs: Optional[str] = None
    process_results: Optional[str] = None
    num_fewshot: int = 0

class BenchmarkConfig(BaseModel):
    task: List[TaskConfig] = Field(alias="manual_tasks")
    model_config = ConfigDict(populate_by_name=True)

class _BenchmarkSafeLoader(yaml.SafeLoader):
    """YAML loader that gracefully handles custom tags like !function."""

def _construct_unknown_tag(loader: _BenchmarkSafeLoader, tag_suffix: str, node: yaml.Node):
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    raise TypeError(f"Unsupported YAML node type for custom tag: {type(node)}")

_BenchmarkSafeLoader.add_multi_constructor("!", _construct_unknown_tag)

def load_benchmark_config(config_path: Path) -> BenchmarkConfig:
    """Load and validate benchmark configuration from YAML."""
    with open(config_path, "r") as f:
        data = yaml.load(f, Loader=_BenchmarkSafeLoader)
    return BenchmarkConfig(**data)


def _bool_to_yaml(value: bool) -> str:
    return "true" if value else "false"

def _normalize_func_ref(ref: str) -> str:
    # Accept "path/to/file.py:fn" or "pkg.mod:fn". Convert the latter to a file path.
    if ":" not in ref:
        return ref
    head, func = ref.split(":", 1)
    if head.endswith(".py") or "/" in head:
        return f"{head}:{func}"
    return f"{head.replace('.', '/')}.py:{func}"

def _render_manual_task(task: TaskConfig) -> str:
    lines = [f"task: {task.task}", f"dataset_path: {task.dataset_path}"]
    if task.dataset_name:
        lines.append(f"dataset_name: {task.dataset_name}")
    if task.test_split:
        lines.append(f"test_split: {task.test_split}")
    if task.validation_split:
        lines.append(f"validation_split: {task.validation_split}")
    if task.train_split:
        lines.append(f"train_split: {task.train_split}")
    g = task.generation_kwargs
    lines.append(f"output_type: {task.output_type}")
    lines.append('doc_to_text: "{{input}}"')
    lines.append('doc_to_target: "{{answer}}"')
    lines.append("generation_kwargs:")
    lines.append(f"  max_new_tokens: {g.max_new_tokens}")
    lines.append(f"  do_sample: {_bool_to_yaml(g.do_sample)}")
    lines.append(f"  temperature: {g.temperature}")
    lines.append("  until:")
    lines.append('    - "\\\\n\\\\n"')
    lines.append('    - "</s>"')
    lines.append('    - "<|im_end|>"')
    if task.process_docs:
        lines.append(f"process_docs: !function {_normalize_func_ref(task.process_docs)}")
    if task.metric_list:
        lines.append("metric_list:")
        for metric in task.metric_list:
            lines.extend([
                f"  - metric: {metric.metric}",
                f"    aggregation: {metric.aggregation}",
                f"    higher_is_better: {_bool_to_yaml(metric.higher_is_better)}",
            ])
    if task.process_results:
        lines.append(f"process_results: !function {_normalize_func_ref(task.process_results)}")
    lines.extend([f"num_fewshot: {task.num_fewshot}", ""])
    return "\n".join(lines)


def materialize_manual_tasks(config: BenchmarkConfig, cache_root: Path) -> Path:
    """Create per-task YAML files lm-eval can consume from the benchmark config."""
    manual_dir = cache_root / "_manual_tasks"
    ensure_dir(manual_dir)
    for existing in manual_dir.glob("*.yaml"):
        existing.unlink()
    for task in config.task:
        (manual_dir / f"{task.task}.yaml").write_text(_render_manual_task(task))
    src_utils = Path(__file__).with_name("utils.py")
    if src_utils.exists():
        # Copy as a flattened module name for lm-eval to find
        shutil.copy2(src_utils, manual_dir / "ekd.evaluations.utils.py")
    return manual_dir

# ---------- Utility ----------
def run(cmd: List[str], env: Optional[Dict[str, str]] = None, cwd: Optional[str] = None, timeout: Optional[int] = None) -> Tuple[int, str]:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{timestamp}] $ {' '.join(cmd)}")
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, env=env, cwd=cwd, text=True, timeout=timeout)
        end_timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{end_timestamp}] Command completed:")
        print(out)
        return 0, out
    except subprocess.CalledProcessError as e:
        end_timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{end_timestamp}] Command failed:")
        print(e.output)
        return e.returncode, e.output
    except subprocess.TimeoutExpired as e:
        end_timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{end_timestamp}] Command timed out after {timeout} seconds: {e}")
        return 124, f"Timeout after {timeout} seconds"

def run_async(cmd: List[str], env: Optional[Dict[str, str]] = None) -> subprocess.Popen:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{timestamp}] $ (async) {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)

def wait_with_timeout(proc: subprocess.Popen, timeout: int) -> Tuple[int, str]:
    try:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] Waiting for process (timeout={timeout}s)...")
        out, _ = proc.communicate(timeout=timeout)
        rc = proc.returncode
        if rc is None:
            rc = 0
        end_timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{end_timestamp}] Process completed with code {rc}")
        print(out or "")
        return rc, out or ""
    except subprocess.TimeoutExpired:
        end_timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{end_timestamp}] Process timed out after {timeout}s, killing...")
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

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def export_hf_model(base_model_dir: str, ckpt_path: Path, export_dir: Path) -> None:
    """Export a HF-ready directory from base weights + checkpoint.
       Skips work if kd_export_meta.json matches the checkpoint hash.
    """
    export_dir.mkdir(parents=True, exist_ok=True)
    meta_path = export_dir / "kd_export_meta.json"
    ckpt_hash = sha256_file(ckpt_path)

    # Skip if unchanged
    if meta_path.exists():
        try:
            old = json.load(open(meta_path))
            if old.get("ckpt_sha256") == ckpt_hash:
                print(f"[export] Cache hit for {export_dir} (ckpt unchanged). Skipping export.")
                return
        except Exception:
            pass

    print(f"Exporting model from base '{base_model_dir}' with state_dict '{ckpt_path.name}' -> '{export_dir}'")

    # Always load tokenizer
    tok = AutoTokenizer.from_pretrained(base_model_dir, use_fast=False, trust_remote_code=True)

    # ---- Try 8-bit first, then fallback ----
    model = None
    try:
        print("[export] Trying 8-bit quantized load...", flush=True)
        q8 = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_dir,
            quantization_config=q8,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        for p in model.parameters():
            p.requires_grad_(False)
        print("[export] Loaded model in 8-bit.", flush=True)
    except Exception as e:
        print(f"[export] 8-bit load failed ({e}); falling back to float16.", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_dir,
            torch_dtype=torch.float16,
            device_map=None,
            trust_remote_code=True,
        )

    # ---- Load checkpoint weights ----
    chk = torch.load(ckpt_path, map_location="cpu")
    state = chk.get("model_state_dict")
    if state is None:
        raise ValueError(f"No 'model_state_dict' in {ckpt_path}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")

    # ---- Save final HF export ----
    model.save_pretrained(export_dir, safe_serialization=True)
    tok.save_pretrained(export_dir)

    # Copy run_params.json if present
    try:
        candidate = ckpt_path.parent.parent / "run_params.json"
        if candidate.exists():
            shutil.copy2(candidate, export_dir / "run_params.json")
    except Exception:
        pass

    meta = {
        "source_checkpoint": str(ckpt_path),
        "epoch": chk.get("epoch"),
        "step": chk.get("step"),
        "global_step": chk.get("global_step"),
        "distill_type": chk.get("distill_type"),
        "k_percent": chk.get("k_percent"),
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
# LIGHT suite (coffee-break): each tuple is (task, sample_count). We attempt to
# draw a SEEDED RANDOM subset of that many samples per task (when the lm-eval CLI
# supports random limiting). If the installed lm-eval version lacks such a flag,
# we fall back to the harness default (typically first-N) and print a warning.
LIGHT_LMEVAL_TASKS: List[Tuple[str, Optional[int]]] = [
    # accuracy (percentage of correct answers)
    ("gsm8k", None),
    ("svamp", None),
    ("lambada_openai", None), 
    # normalized accuracy - multiple-choice datasets.raw accuracy can mislead so normalization accounts for imbalanced choices
    ("arc_challenge", None),
    ("arc_easy", None),
    # ("hellaswag", 500),
    # ("piqa", 500),
    # exact-match
    ("aime25", None),
    # ("ifeval", None)
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
    ("asdiv", None), # arithmetic subset (ASDiv-A handled inside task)
    ("hendrycks_math", None),  # MATH
    ("aime25", None),
    ("olympiadbench", None),
    ("piqa", None),
    ("lambada_openai", None),
     # General reasoning
    ("agieval", None),
    ("bbh", None), # BIG-Bench Hard
    ("agieval", None),
    # QA / multi-hop
    ("squadv2", None),
    ("hotpotqa", None),
    ("nq_open", None),
     # Commonsense
    ("hellaswag", None),
    ("arc_easy", None),
    ("arc_challenge", None),
]

# Per-task timeouts (seconds).
TASK_TIMEOUTS = {
    # light-ish baselines
    "arc_challenge": 6000,
    "arc_easy": 6000,
    "gsm8k": 6000,
    "svamp": 6000,
    "aime25": 6000,
    "boolq": 600,
    "hellaswag": 1200,
    # heavy add-ons
    "asdiv": 1200,
    "hendrycks_math": 3600,
    "olympiadbench": 5400,
    "bbh": 5400,
    "agieval": 5400,
    "squadv2": 1200,
    "hotpotqa": 3600,
    "nq_open": 3600,
    # "ifeval": 10
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
) -> Tuple[Optional[Path], Dict[str, str]]:
    """Run the requested LM-Eval suite (light/heavy). Executes one task per GPU in waves."""
    out_dir = results_dir / f"lmeval_{suite}_{tag}"
    ensure_dir(out_dir)

    tasks_with_limits: List[Tuple[str, Optional[int]]] = list(
        LIGHT_LMEVAL_TASKS if suite == "light" else HEAVY_LMEVAL_TASKS
    )
    if suite == "light" and LIGHT_ENABLE_OPTIONALS:
        tasks_with_limits += list(LIGHT_OPTIONALS)

    if not tasks_with_limits:
        print("No LM-Eval tasks selected.")
        return None

    # Validate benchmark config if it exists
    include_root = Path(__file__).parent
    manual_include_dir: Optional[Path] = None
    benchmark_config_path = include_root / "benchmark_tasks.yaml"
    if benchmark_config_path.exists():
        try:
            config = load_benchmark_config(benchmark_config_path)
            manual_task_names = [task.task for task in config.task]
            manual_include_dir = materialize_manual_tasks(config, results_dir)
            print(
                f"✅ Validated benchmark config with {len(manual_task_names)} custom tasks: {manual_task_names}\n"
                f"➡️  Materialized manual tasks at {manual_include_dir}"
            )
        except Exception as e:
            print(f"⚠️ Warning: Invalid benchmark config: {e}")

    include_path = (manual_include_dir or include_root).as_posix()
    repo_root = Path(__file__).resolve().parents[2].as_posix()

    if not gpu_ids:
        print("[lm-eval] No GPUs detected/selected; skipping suite.")
        return None

    # preflight: load task registry to filter unknown tasks (but allow external include_path)
    try:
        from lm_eval import tasks as _tasks_mod  # type: ignore
        _available = set(getattr(_tasks_mod, "TASK_REGISTRY", {}).keys())
    except Exception:
        _available = set()

    # Include local task YAMLs if present
    # Seed for downstream tools; default to env or EVAL_SEED set by main
    seed = int(os.environ.get("EVAL_SEED", "42"))
    # Probe lm-eval CLI support for seed and random limiting (version-dependent)
    _help_code, _help_out = run(["lm-eval", "--help"])
    _has_seed_flag = _help_code == 0 and ("--seed" in _help_out)
    _has_fewshot_seed_flag = _help_code == 0 and ("--fewshot-seed" in _help_out)
    _lm_eval_supports_seed = _has_seed_flag or _has_fewshot_seed_flag
    # Random limiting flags have changed across versions; detect both variants
    _has_limit_type = _help_code == 0 and ("--limit-type" in _help_out)
    _has_limit_mode = _help_code == 0 and ("--limit_mode" in _help_out)
    # Per-sample logging (needed for ECE)
    _has_log_samples = _help_code == 0 and ("--log_samples" in _help_out)

    base_args = [
        "lm-eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_dir},trust_remote_code=True,dtype=float16",
        "--output_path", str(out_dir),
        "--include_path", include_path,
    ]
    # Enable per-sample logging if supported (helps compute ECE later)
    if _has_log_samples:
        base_args += ["--log_samples"]
    else:
        print("[lm-eval] This lm-eval version lacks --log_samples; will skip ECE metrics.")
    if _has_seed_flag:
        base_args += ["--seed", str(seed)]
    elif _has_fewshot_seed_flag:
        base_args += ["--fewshot-seed", str(seed)]

    _warned_no_random_limit = False

    def _task_cmd(task: str, limit: Optional[int], device: str) -> List[str]:
        nonlocal _warned_no_random_limit
        cmd = base_args + ["--tasks", task, "--device", device]
        # Some generate_until tasks stall when probing batch_size=auto; force bs=1
        GENERATE_BS1 = {"gsm8k", "svamp", "aime25"}
        if task in GENERATE_BS1:
            cmd += ["--batch_size", "1"]
        else:
            cmd += ["--batch_size", "auto"]
        if isinstance(limit, (int, float)):
            cmd += ["--limit", str(limit)]
            # Prefer randomized subset selection when supported by this lm-eval version
            if _has_limit_type:
                cmd += ["--limit-type", "random"]
            elif _has_limit_mode:
                cmd += ["--limit_mode", "random"]
            else:
                if not _warned_no_random_limit:
                    print("[lm-eval] Warning: CLI lacks a random subset flag; using first-N for --limit.")
                    _warned_no_random_limit = True
        return cmd

    # Parallel across physical GPUs (mask each process to one GPU)
    task_status: Dict[str, str] = {}
    good_any = False
    i = 0
    while i < len(tasks_with_limits):
        procs = []
        wave = tasks_with_limits[i : i + len(gpu_ids)]
        wave_names = [t for (t, _) in wave]
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [lm-eval] Launching wave: {wave_names} on GPUs {gpu_ids}")
        for (task, limit), gid in zip(wave, gpu_ids):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gid)  # inside, cuda:0 maps to this physical GPU
            py_path_parts = [include_path, repo_root]
            if env.get("PYTHONPATH"):
                py_path_parts.append(env["PYTHONPATH"])
            env["PYTHONPATH"] = ":".join(py_path_parts)
            # Propagate seed-related env for deterministic child runs
            env.setdefault("PYTHONHASHSEED", str(seed))
            env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
            env.setdefault("EVAL_SEED", str(seed))
            cmd = _task_cmd(task, limit, "cuda:0")
            p = run_async(cmd, env=env)
            timeout = TASK_TIMEOUTS.get(task, 3600)
            procs.append((task, p, timeout))
        for task, p, to in procs:
            rc, out = wait_with_timeout(p, timeout=to)
            timestamp = datetime.now().strftime("%H:%M:%S")
            if rc == 0:
                print(f"[{timestamp}] ✅ Task {task} completed successfully")
                task_status[task] = "ok"
                good_any = True
            elif rc == 124:
                print(f"[{timestamp}] ⏰ Task {task} timed out after {to} seconds")
                task_status[task] = "timeout"
            else:
                print(f"[{timestamp}] ❌ Task {task} failed with code {rc}")
                task_status[task] = f"failed:{rc}"
        i += len(gpu_ids)
    return out_dir if good_any else None, task_status

# ---------- ECE (Expected Calibration Error) from LM-Eval samples ----------
def _softmax(xs: List[float]) -> List[float]:
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

def compute_ece(confidences: List[float], correctness: List[int], n_bins: int = 15) -> float:
    """Compute Expected Calibration Error for top-1 predictions.

    Args:
        confidences: list of predicted top-1 probabilities in [0,1]
        correctness: list of 0/1 indicating if prediction was correct
        n_bins: number of equal-width bins over [0,1]
    Returns:
        Scalar ECE value in [0,1].
    """
    assert len(confidences) == len(correctness)
    n = len(confidences)
    if n == 0:
        return float("nan")
    # Initialize bins
    bins: List[List[int]] = [[] for _ in range(n_bins)]
    bin_confs: List[List[float]] = [[] for _ in range(n_bins)]
    for c, y in zip(confidences, correctness):
        # Clamp to [0,1]
        c = 0.0 if c < 0 else (1.0 if c > 1 else c)
        # Rightmost-inclusive for c==1.0
        idx = min(int(c * n_bins), n_bins - 1)
        bins[idx].append(int(y))
        bin_confs[idx].append(float(c))
    ece = 0.0
    for i in range(n_bins):
        m = len(bins[i])
        if m == 0:
            continue
        acc_i = sum(bins[i]) / m
        conf_i = sum(bin_confs[i]) / m
        ece += (m / n) * abs(acc_i - conf_i)
    return ece

def _parse_sample_line_for_mc(line: Dict[str, Any]) -> Optional[Tuple[List[float], int]]:
    """Attempt to extract (choice_scores, gold_index) from an lm-eval sample JSONL line.

    Returns list of scores (higher is better, e.g., log-likelihoods) and gold index.
    Returns None if the line does not look like a multiple-choice sample with usable data.
    """
    # Try common fields for label index
    gold_idx = None
    for k in ("label", "gold_idx", "gold_index"):
        if isinstance(line.get(k), int):
            gold_idx = int(line[k])
            break
    if gold_idx is None:
        # Some tasks log gold as letter (A/B/C/D)
        gold = line.get("gold") or line.get("answer")
        if isinstance(gold, str) and len(gold) == 1 and gold in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            gold_idx = ord(gold) - ord('A')
    # Collect choice scores
    scores: Optional[List[float]] = None
    # Direct list of numbers
    if isinstance(line.get("choice_scores"), list) and all(isinstance(x, (int, float)) for x in line["choice_scores"]):
        scores = [float(x) for x in line["choice_scores"]]
    # Choices as list of dicts with score-like fields
    if scores is None and isinstance(line.get("choices"), list) and line["choices"]:
        cand_keys = ("loglikelihood", "logprob", "score")
        first = line["choices"][0]
        if isinstance(first, dict):
            for k in cand_keys:
                if k in first and all(isinstance(c.get(k), (int, float)) for c in line["choices"]):
                    scores = [float(c[k]) for c in line["choices"]]
                    break
    if scores is None or gold_idx is None:
        return None
    if not (0 <= gold_idx < len(scores)):
        return None
    return scores, gold_idx

def collect_ece_from_lmeval_samples(lmeval_dir: Optional[Path], bins: int = 15) -> Dict[str, Dict[str, float]]:
    """Parse lm-eval sample logs (if present) and compute ECE per multiple-choice task.

    Looks for samples/*.jsonl under the lm-eval output directory, extracts per-example
    choice scores and gold labels, converts scores to probabilities with softmax,
    and computes top-1 ECE.
    """
    if not lmeval_dir or not lmeval_dir.exists():
        return {}
    # Gather all sample files
    sample_files = list(lmeval_dir.rglob("*.jsonl"))
    # Filter to the typical samples/ subdir first if present
    preferred = [p for p in sample_files if "samples" in p.as_posix().split("/")]
    files = preferred or sample_files
    per_task: Dict[str, Dict[str, float]] = {}

    def _normalize_task_from_filename(stem: str) -> str:
        # Drop common suffixes like "_samples" or ".samples"
        if stem.endswith("_samples"):
            stem = stem[: -len("_samples")]
        if stem.endswith(".samples"):
            stem = stem[: -len(".samples")]
        # If there is a normalization variant like ",none", trim to base task
        if "," in stem:
            stem = stem.split(",", 1)[0]
        return stem
    for p in files:
        try:
            # Task name from filename (strip extension and normalize)
            task_name = _normalize_task_from_filename(p.stem)
            confidences: List[float] = []
            correctness: List[int] = []
            with open(p, "r", encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        obj = json.loads(ln)
                    except Exception:
                        continue
                    parsed = _parse_sample_line_for_mc(obj)
                    if parsed is None:
                        continue
                    scores, gold_idx = parsed
                    probs = _softmax(scores)
                    if not probs:
                        continue
                    pred_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
                    conf = float(probs[pred_idx])
                    confidences.append(conf)
                    correctness.append(1 if pred_idx == gold_idx else 0)
            if len(confidences) >= 10:
                ece = compute_ece(confidences, correctness, n_bins=bins)
                per_task.setdefault(task_name, {})["ece"] = float(ece)
                per_task[task_name]["ece_n"] = float(len(confidences))
        except Exception as e:
            print(f"Failed to compute ECE from {p}: {e}")
    return per_task

def _map_ece_to_existing_tasks(ece_metrics: Dict[str, Dict[str, float]],
                               lmeval_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Map ECE task keys to match the keys present in lm-eval metrics when possible.

    lm-eval often uses task keys like "arc_challenge,none" while sample files may yield
    base names like "arc_challenge". This function attempts to align ECE entries to the
    exact keys used in lmeval_metrics so they appear under the same task in the JSON.
    """
    if not ece_metrics or not lmeval_metrics:
        return ece_metrics

    # Build index from base -> full task keys found in lmeval
    base_to_full: Dict[str, List[str]] = {}
    for full_key in lmeval_metrics.keys():
        base_key = full_key.split(",", 1)[0]
        base_to_full.setdefault(base_key, []).append(full_key)

    remapped: Dict[str, Dict[str, float]] = {}
    for ece_key, metrics in ece_metrics.items():
        base_key = ece_key.split(",", 1)[0]
        candidates = base_to_full.get(base_key, [])
        if len(candidates) == 1:
            remapped[candidates[0]] = metrics
        else:
            # If ambiguous or no match, keep original key
            remapped[ece_key] = metrics
    return remapped

# ---------- Lighteval (summarization) ----------
def run_lighteval(model_dir: Path, tag: str, results_dir: Path, suite: str) -> Optional[Path]:
    """Run summarization only for HEAVY suite (CNN/DM, XSum)."""
    if suite != "heavy":
        print("[lighteval] Skipping summarization for light suite.")
        return None
    out_path = results_dir / f"lighteval_{tag}.json"
    tasks = [t.replace("-", "_") for t in LIGHTEVAL_TASKS]  # normalize hyphen → underscore
    code = 1
    for cmd in (
        [
            "lighteval", "--model", "hf", str(model_dir),
            "--tasks", ",".join(tasks), "--results", str(out_path),
        ],
        [
            sys.executable, "-m", "lighteval",
            "--model", "hf", str(model_dir),
            "--tasks", ",".join(tasks), "--results", str(out_path),
        ],
    ):
        code, _ = run(cmd)
        if code == 0:
            break
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
    env = os.environ.copy()
    env["HOME"] = str(evalplus_home)
    env["XDG_CACHE_HOME"] = str(evalplus_home / ".cache")
    env.setdefault("EVALPLUS_TRUST_REMOTE_CODE", "1")
    for ds_name in datasets:
        cmd = [
            "evalplus.evaluate",
            "--model", str(model_dir),
            "--dataset", ds_name,
            "--backend", "hf",
            "--greedy",
        ]
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
    if not lmeval_dir or not lmeval_dir.exists():
        return {}
    results: Dict[str, Dict[str, float]] = {}
    # Look for both per-task and aggregated outputs
    patterns = [
        "results.json",
        "aggregated_results.json",
        "*results*.json",
        "*aggregated*.json",
        "*results*.yaml",
    ]
    targets: List[Path] = []
    for pat in patterns:
        targets.extend(lmeval_dir.rglob(pat))
    if not targets:
        return {}
    if not targets:
        return {}
    for p in targets:
        try:
            if p.suffix == ".yaml":
                blob = yaml.safe_load(open(p))
            else:
                blob = json.load(open(p))
            r = blob.get("results", blob if isinstance(blob, dict) else {})
            if not isinstance(r, dict):
                continue
            for task, metrics in r.items():
                if not isinstance(metrics, dict):
                    continue
                results.setdefault(task, {})
                for mk, mv in metrics.items():
                    if isinstance(mv, dict) and "value" in mv:
                        v = mv.get("value")
                        if isinstance(v, (int, float)):
                            results[task][mk] = float(v)
                        se = mv.get("stderr")
                        if isinstance(se, (int, float)):
                            results[task][f"{mk}_stderr"] = float(se)
                    elif isinstance(mv, (int, float)):
                        results[task][mk] = float(mv)
        except Exception as e:
            print(f"Failed to parse {p}: {e}")
    return results

def collect_lighteval_metrics(out_file: Path) -> Dict[str, Dict[str, float]]:
    if not out_file or not out_file.exists():
        return {}
    try:
        blob = json.load(open(out_file))
        return {
            task: {mk: float(mv) for mk, mv in task_res.items() if isinstance(mv, (int, float))}
            for task, task_res in blob.items()
        }
    except Exception as e:
        print(f"Failed to parse {out_file}: {e}")
    return {}

def collect_evalplus_metrics(root: Optional[Path], ds_name: str) -> Dict[str, Dict[str, float]]:
    if not root or not root.exists():
        return {}
    candidates = [p for name in ("summary.json", "report.json", "scores.json") for p in root.rglob(name)]
    if not candidates:
        return {}
    p = max(candidates, key=lambda x: x.stat().st_mtime)
    try:
        blob = json.load(open(p))
        metrics = {
            k: float(blob[k])
            for k in ["pass@1", "pass@5", "pass@10", "pass@100", "mbpp_score", "humaneval_score"]
            if isinstance(blob.get(k), (int, float))
        }
        metrics.update(
            {
                k: float(v)
                for k, v in blob.items()
                if k not in metrics and isinstance(v, (int, float)) and ("pass" in k or "score" in k or "acc" in k)
            }
        )
        return {ds_name: metrics}
    except Exception as e:
        print(f"Failed to parse EvalPlus summary at {p}: {e}")
    return {}

def collect_alpacaeval_metrics(out_dir: Optional[Path]) -> Dict[str, Dict[str, float]]:
    if not out_dir or not out_dir.exists():
        return {}
    for p in out_dir.rglob("*.json"):
        try:
            blob = json.load(open(p))
            if isinstance(blob, dict):
                vals = {
                    k: float(blob[k])
                    for k in ["length_controlled_win_rate", "win_rate", "pairwise_win_rate", "length_controlled"]
                    if isinstance(blob.get(k), (int, float))
                }
                if vals:
                    return {"AlpacaEval2": vals}
        except Exception:
            continue
    return {}

def collect_simple_json(root: Optional[Path], task_name: str, filename: str) -> Dict[str, Dict[str, float]]:
    if not root:
        return {}
    p = root / filename
    if not p.exists():
        return {}
    try:
        blob = json.load(open(p))
        if isinstance(blob, dict):
            return {
                task_name: {k: float(v) for k, v in blob.items() if isinstance(v, (int, float))}
            }
    except Exception as e:
        print(f"Failed to parse {p}: {e}")
    return {}

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
    if not all_metrics or not tasks:
        print("\n% No metrics found — check timeouts/include_path. Skipping table.\n")
        return
    def _disp_metric(m: str) -> str:
        # Clean up filter suffixes like ",none" in lm-eval keys for nicer LaTeX headers
        if "," in m:
            head, tail = m.split(",", 1)
            return head if tail == "none" else m
        return m
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
            header.append(f"{mname}:{_disp_metric(met)}")
    print(" & ".join(header) + r" \\")
    print(r"\midrule")
    for t in tasks:
        row = [t]
        for mname in model_names:
            mm = all_models_metrics[mname].get(t, {})
            for met in all_metrics:
                val = mm.get(met, None)
                if isinstance(val, float):
                    # Avoid printing NaN/inf in LaTeX
                    row.append(f"{val:.2f}" if math.isfinite(val) else "-")
                elif isinstance(val, int):
                    row.append(str(val))
                else:
                    row.append("-")
        print(" & ".join(row) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

def save_json(obj: dict, path: Path) -> None:
    """Persist a Python dict to pretty-printed UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"[json] Wrote {path}")

def compute_averages(merged: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Compute macro-averages across tasks for each numeric metric.

    - Excludes keys ending with '_stderr'.
    - Excludes obvious count-like fields such as 'ece_n'.
    - Returns per-metric averages and 'avg_all' over the per-task combined means.
    """
    metric_values: Dict[str, List[float]] = {}
    # Accumulate values per metric key across tasks
    for task_metrics in merged.values():
        for mk, mv in task_metrics.items():
            if not isinstance(mv, (int, float)):
                continue
            if mk.endswith("_stderr"):
                continue
            if mk in {"ece_n"}:
                continue
            if not math.isfinite(float(mv)):
                continue
            metric_values.setdefault(mk, []).append(float(mv))
    # Compute simple means
    averages: Dict[str, float] = {}
    for mk, vals in metric_values.items():
        if vals:
            averages[f"avg_{mk}"] = sum(vals) / len(vals)
    # Overall average across metric means (if any)
    per_metric_means = [v for k, v in averages.items()]
    if per_metric_means:
        averages["avg_all"] = sum(per_metric_means) / len(per_metric_means)
    return averages

# ---------- Main pipeline ----------
def main():
    parser = argparse.ArgumentParser()
    # Single input: trained student model path (either HF dir or .pt checkpoint). We'll auto-detect.
    parser.add_argument("model", type=str, help="Path to trained student model: HF directory (preferred) or a .pt checkpoint. If --from_hf is set, this may be a HF hub ID (e.g., 'Qwen/Qwen3-8B').")
    # Parallel/GPU controls
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="Comma-separated physical GPU ids for parallel lm-eval (e.g., '0,1,2'). Defaults to visible GPUs.")
    parser.add_argument("--max_parallel", type=int, default=None,
                        help="Cap the number of parallel lm-eval workers (<= number of provided GPUs).")
    parser.add_argument("--work_dir", type=str, default="eval_runs")
    parser.add_argument("--output_dir", type=str, default="evaluation_json_results")
    # Unified runs registry integration
    parser.add_argument("--runs_registry", type=str, default="results/runs.json",
                        help="Path to the unified runs JSON registry (shared with training).")
    parser.add_argument("--params_json", type=str, default=None,
                        help="Optional path to a JSON file containing the original training parameters. If not provided, attempts to infer from model dir (config.json).")
    # Logging configuration (default project updated per request)
    parser.add_argument("--wandb_project", type=str, default="selective-entropy-knowledge-distillation",
                        help="W&B project slug (e.g., 'selective-entropy-knowledge-distillation').")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable W&B logging.")
    parser.add_argument("--disable_tensorboard", action="store_true", help="Disable TensorBoard logging.")
    # Suite selection
    parser.add_argument("--suite", type=str, choices=["light", "heavy"], default="light",
                        help="Evaluation suite to run: 'light' (quick) or 'heavy' (paper).")
    # Source override: treat positional model arg as HF hub ID
    parser.add_argument("--from_hf", action="store_true", help="Interpret 'model' as a HF hub ID even if it is not a local path.")
    # No vanilla/ekd distinction; one model at a time
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    exports_dir = work_dir / "exports"
    results_dir = work_dir / "results"
    json_results_dir = Path(args.output_dir)
    ensure_dir(exports_dir)
    ensure_dir(results_dir)
    ensure_dir(json_results_dir)

    # Build GPU pool
    if args.gpu_ids:
        gpu_pool = [int(x) for x in args.gpu_ids.split(",") if x.strip() != ""]
    else:
        gpu_pool = visible_gpu_ids()
    if args.max_parallel is not None and args.max_parallel > 0:
        gpu_pool = gpu_pool[: args.max_parallel]
    print(f"[gpu] Using GPUs: {gpu_pool}" if gpu_pool else "[gpu] No GPUs detected/selected.")

    # Resolve the single model dir to evaluate
    model_specs: List[Tuple[str, Path]] = []
    # If explicitly flagged as HF hub, skip local checks and create a temp export alias
    if args.from_hf:
        hub_id = args.model
        # Sanitize tag for filenames/logs
        tag = re.sub(r"[^A-Za-z0-9_.-]", "_", hub_id)
        # Use a pseudo-path that downstream tools accept (lm-eval accepts hub ids directly via --model_args)
        # For consistency with the rest of the pipeline that expects a Path, use the hub id as-is
        model_specs.append((tag, Path(hub_id)))
    else:
        in_path = Path(args.model)
        tag = in_path.stem if in_path.is_file() else in_path.name
        if in_path.is_dir():
            print(f"[models] Using HF model directory: {in_path}")
            model_specs.append((tag, in_path))
        elif in_path.is_file() and in_path.suffix == ".pt":
            # Export HF dir based on base_model_dir recorded in checkpoint
            try:
                chk = torch.load(in_path, map_location="cpu")
            except Exception as e:
                print(f"Error loading checkpoint {in_path}: {e}")
                sys.exit(2)
            base_model_dir = chk.get("base_model_dir") or chk.get("base_model") or chk.get("student_model")
            if not base_model_dir:
                print("Error: checkpoint does not record base_model_dir. Re-train with newer code or evaluate a ready HF model directory.")
                sys.exit(2)
            export_dir = exports_dir / f"export_{in_path.stem}"
            print(f"[export] Exporting HF model from base='{base_model_dir}' and ckpt='{in_path.name}' -> '{export_dir}'")
            export_hf_model(base_model_dir, in_path, export_dir)
            model_specs.append((tag, export_dir))
        else:
            print(f"Error: provided model path is neither an HF directory nor a .pt file: {in_path}")
            sys.exit(2)

    if not model_specs:
        print("No models exported. Exiting.")
        sys.exit(1)

    all_models_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}

    for tag, model_dir in model_specs:
        print(f"\n=== Running benchmarks for {tag} (suite={args.suite}) ===")

        # LM-Eval (parallel across GPUs according to the chosen suite)
        lmeval_root, task_status = run_lmeval_suite(
            model_dir=model_dir,
            tag=tag,
            results_dir=results_dir,
            gpu_ids=gpu_pool,
            suite=args.suite,
        )
        lmeval_metrics = collect_lmeval_metrics(lmeval_root) if lmeval_root else {}
        # Derive ECE from sample logs when available and align its task keys
        raw_ece_metrics = collect_ece_from_lmeval_samples(lmeval_root) if lmeval_root else {}
        ece_metrics = _map_ece_to_existing_tasks(raw_ece_metrics, lmeval_metrics)

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
            ece_metrics,
            lighteval_metrics,
            he_metrics,
            mbpp_metrics,
            ifeval_metrics,
            alpaca_metrics,
            jbb_metrics,
            hb_metrics
        ])
        all_models_metrics[tag] = merged

        # --- Save results to JSON for this model ---
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_result_file = json_results_dir / f"eval_{args.suite}_{tag}_{ts}.json"
        averages = compute_averages(merged)
        payload = {"tag": tag, "suite": args.suite, "results": merged, "averages": averages, "task_status": task_status}
        save_json(payload, json_result_file)

        # Upsert into unified registry keyed by params hash
        try:
            params_blob: Optional[Dict[str, Any]] = None
            if args.params_json and Path(args.params_json).exists():
                params_blob = json.load(open(args.params_json))
            else:
                # Preferred: run_params.json saved by training
                rp = model_dir / "run_params.json"
                if rp.exists():
                    try:
                        rp_blob = json.load(open(rp))
                        if isinstance(rp_blob, dict):
                            params_blob = rp_blob.get("params") or rp_blob
                            # If file already contains id, prefer to trust it
                            if isinstance(rp_blob.get("id"), str):
                                params_hash = rp_blob["id"]
                                upsert_eval_results(Path(args.runs_registry), params_hash, args.suite, merged, averages, task_status)
                                raise StopIteration  # already upserted; skip remainder
                    except Exception:
                        pass
                # Try to derive from the HF model directory (if present)
                cfg_path = model_dir / "config.json"
                if cfg_path.exists():
                    try:
                        cfg_blob = json.load(open(cfg_path))
                        # We only need a stable mapping of training-related fields; many HF fields are irrelevant.
                        # Try to pass through fields that might have been stored by training code.
                        probable = {}
                        for k in [
                            "teacher_model", "student_model", "distill_type", "k_percent",
                            "enable_ce", "alpha_ce", "kd_temperature", "entropy_approx_temperature",
                            "anneal_kd_temperature", "kd_temperature_start", "kd_temperature_end", "kd_hold_frac",
                            "rs_alpha", "rs_floor", "bucket_lower_percent", "bucket_upper_percent",
                            "score_token_selection", "score_normalize", "score_entropy_weight", "score_ce_weight", "score_kl_weight",
                            "bandit_alpha", "bandit_lambda", "bandit_threshold", "bandit_min_tokens", "bandit_max_tokens", "bandit_device", "bandit_reward_clip",
                            "datasets", "prompt_col", "answer_col", "dataset_config", "fineweb_tokens",
                            "epochs", "batch_size", "gradient_accumulation_steps", "max_seq_len", "lr",
                            "seed", "deterministic",
                            "offline_cache", "entropy_approx_m", "rs_vocab_samples", "rs_vocab_beta", "H_hat_u8",
                            "eliminate_softmax", "sampled_softmax_negatives",
                        ]:
                            if k in cfg_blob:
                                probable[k] = cfg_blob[k]
                        if probable:
                            params_blob = probable
                    except Exception:
                        pass
            if params_blob is None:
                # As a last resort, attach an empty dict so the call still produces a stable id from empty params
                params_blob = {}
            params_hash = compute_params_hash(params_blob)
            upsert_eval_results(Path(args.runs_registry), params_hash, args.suite, merged, averages, task_status)
        except StopIteration:
            pass
        except Exception as e:
            print(f"[registry] Failed to upsert eval results: {e}")

        # Log results to W&B and TensorBoard
        try:
            if not args.disable_wandb:
                # Prefer robust WandBLogger if available
                if WandBLogger is not None:
                    eval_run = WandBLogger(
                        project=args.wandb_project,
                        entity=os.getenv("WANDB_ENTITY"),
                        name=f"eval-{tag}",
                        config={
                            "suite": args.suite,
                            "model": str(model_dir),
                            "task_status": task_status,
                        },
                        tags=["evaluation", args.suite],
                        group=os.getenv("WANDB_GROUP"),
                        job_type="eval",
                        notes=os.getenv("WANDB_NOTES"),
                        resume=os.getenv("WANDB_RESUME", "allow"),
                        run_id=os.getenv("WANDB_RUN_ID"),
                    )
                    if getattr(eval_run, "enabled", False):
                        flat = {}
                        for task, metrics in merged.items():
                            if isinstance(metrics, dict):
                                for k, v in metrics.items():
                                    if isinstance(v, (int, float)):
                                        flat[f"{task}/{k}"] = float(v)
                        # Add averages as top-level summary
                        for k, v in averages.items():
                            if isinstance(v, (int, float)):
                                flat[f"avg/{k}"] = float(v)
                        # Add counts for visibility
                        flat["meta/num_tasks"] = float(len(merged))
                        eval_run.log(flat)
                        eval_run.finish()
                    else:
                        # Fallback to legacy helper (respects env in hardened version)
                        log_evaluation_to_wandb(tag, merged, args.wandb_project)
                else:
                    log_evaluation_to_wandb(tag, merged, args.wandb_project)
            if not args.disable_tensorboard:
                log_evaluation_to_tensorboard(tag, merged, str(work_dir / "tb_logs"))
        except Exception as e:
            print(f"Error logging {tag} metrics: {e}")

    # Print LaTeX summary
    print_latex_table(all_models_metrics)

if __name__ == "__main__":
    main()
