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
from typing import Dict, List, Optional, Tuple
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
    from ekd.logging.wandb_utils import log_evaluation_to_wandb, log_evaluation_to_tensorboard
except ImportError:
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
# LIGHT suite (coffee-break): strict caps via --limit (first-N selection by harness)
LIGHT_LMEVAL_TASKS: List[Tuple[str, Optional[int]]] = [
    # accuracy (percentage of correct answers)
    ("gsm8k", 50),
    ("svamp", 100),
    ("lambada_openai", None), 
    # normalized accuracy - multiple-choice datasets.raw accuracy can mislead so normalization accounts for imbalanced choices
    ("arc_challenge", 300),
    ("arc_easy", 300),
    ("hellaswag", 500),
    ("piqa", 500),
    # exact-match
    ("aime25", None),
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
    ("aime24", None),
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
    "boolq": 600,
    "arc_challenge": 1500,
    "hellaswag": 1200,
    "svamp": 900,
    "gsm8k": 1200,
    "aime24": 900,
    "aime25": 2000,
    # heavy add-ons
    "asdiv": 1200,
    "hendrycks_math": 3600,
    "olympiadbench": 5400,
    "bbh": 5400,
    "agieval": 5400,
    "squadv2": 1200,
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
    # Seed for downstream tools; default to env or EKD_EVAL_SEED set by main
    seed = int(os.environ.get("EKD_EVAL_SEED", "42"))
    # Check if lm-eval CLI supports --seed (older versions may not)
    _help_code, _help_out = run(["lm-eval", "--help"])
    _lm_eval_supports_seed = _help_code == 0 and ("--seed" in _help_out or "--fewshot-seed" in _help_out)

    base_args = [
        "lm-eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_dir},trust_remote_code=True,dtype=float16",
        "--output_path", str(out_dir),
        "--include_path", include_path,
    ]
    if _lm_eval_supports_seed:
        base_args += ["--seed", str(seed)]

    def _task_cmd(task: str, limit: Optional[int], device: str) -> List[str]:
        cmd = base_args + ["--tasks", task, "--device", device]
        # Some generate_until tasks stall when probing batch_size=auto; force bs=1
        GENERATE_BS1 = {"gsm8k", "svamp", "aime25"}
        if task in GENERATE_BS1:
            cmd += ["--batch_size", "1"]
        else:
            cmd += ["--batch_size", "auto"]
        if isinstance(limit, (int, float)):
            cmd += ["--limit", str(limit)]
        return cmd

    # Parallel across physical GPUs (mask each process to one GPU)
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
            env.setdefault("EKD_EVAL_SEED", str(seed))
            cmd = _task_cmd(task, limit, "cuda:0")
            p = run_async(cmd, env=env)
            timeout = TASK_TIMEOUTS.get(task, 3600)
            procs.append((task, p, timeout))
        for task, p, to in procs:
            rc, out = wait_with_timeout(p, timeout=to)
            timestamp = datetime.now().strftime("%H:%M:%S")
            if rc == 0:
                print(f"[{timestamp}] ✅ Task {task} completed successfully")
                good_any = True
            elif rc == 124:
                print(f"[{timestamp}] ⏰ Task {task} timed out after {to} seconds")
            else:
                print(f"[{timestamp}] ❌ Task {task} failed with code {rc}")
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

# ---------- Main pipeline ----------
def main():
    parser = argparse.ArgumentParser()
    # Single input: trained student model path (either HF dir or .pt checkpoint). We'll auto-detect.
    parser.add_argument("model", type=str, help="Path to trained student model: HF directory (preferred) or a .pt checkpoint.")
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
    # No vanilla/ekd distinction; one model at a time
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    exports_dir = work_dir / "exports"
    results_dir = work_dir / "results"
    json_results_dir = Path("evaluation_json_results")
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
    in_path = Path(args.model)
    tag = in_path.stem if in_path.is_file() else in_path.name
    model_specs: List[Tuple[str, Path]] = []
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

        # --- Save results to JSON for this model ---
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = json_results_dir / f"eval_{args.suite}_{tag}_{ts}.json"
        payload = {"tag": tag, "suite": args.suite, "results": merged}
        save_json(payload, out_file)

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
