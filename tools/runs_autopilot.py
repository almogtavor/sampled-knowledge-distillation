#!/usr/bin/env python3
"""Simple SLURM orchestration loop driven by results/runs.json.

This utility keeps a small number of distillation runs active by:
- Inspecting the registry for runs that still need training or evaluation.
- Submitting SBATCH jobs (train.slurm / evals.slurm) with basic concurrency limits.
- Tracking a tiny bit of local state to avoid duplicate submissions and requeue failed jobs.

It intentionally sticks to the existing train/eval shell entrypoints so that
run_distillation.py continues to manage the registry updates (job ids,
completion flags, eval metrics, etc.).

Assumptions/limitations:
- The registry already contains the parameter blobs for every run we should execute.
  (Typically they land there once a job starts; unfinished runs will have status
  "started" with completed_train=False.)
- New evaluations are triggered in train.slurm by passing the desired suite as
  the third positional argument (default: "light"). For older runs that already
  finished training without an eval, this script can submit eval jobs directly.
- We only apply small environment overrides that existing shell scripts already
  understand (offline cache toggles, GLS/score flags, dataset overrides, etc.).
  Any exotic parameter that lacks a CLI/env hook will still fall back to the
  defaults baked into train.slurm / run_distillation.py.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

DEFAULT_REGISTRY = Path("results/runs.json")
DEFAULT_STATE = Path("results/automation_state.json")
DEFAULT_TRAIN_SLURM = Path("train.slurm")
DEFAULT_EVAL_SLURM = Path("evals.slurm")

TRAIN_JOB_PREFIX = "ekdT-"
EVAL_JOB_PREFIX = "ekdE-"

# SLURM truncates job names around 8 characters, so the default names from
# train.slurm / evals.slurm often show up as "ekd-trai" and "ekd-eval".
TRAIN_NAME_FALLBACKS = ("ekd-train", "ekd-trai")
EVAL_NAME_FALLBACKS = ("ekd-eval",)

# Optional manual submission sequence. Each entry is consumed in order and can
# override environment variables per training launch. The structure mirrors the
# sbatch lines used in kd_sweep.sh so you can copy/paste the combos you care
# about. Remove or comment out entries you don't need.
CUSTOM_TRAIN_SEQUENCE = [
    {
        "distill_type": "top-k-tok",
        "k_percent": 20,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "NO_OFFLINE": "1",
        },
    },
    {
        "distill_type": "top-k-tok",
        "k_percent": 30,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "NO_OFFLINE": "1",
        },
    },
    {
        "distill_type": "top-k-tok",
        "k_percent": 20,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
        },
    },
    {
        "distill_type": "top-k-tok",
        "k_percent": 30,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
        },
    },
    {
        "distill_type": "top-k-tok",
        "k_percent": 25,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "NO_OFFLINE": "1",
            "SCORE_TOKEN_SELECTION": 1,
            "SCORE_NORMALIZE": "z",
            "SCORE_ENTROPY_WEIGHT": 1.0,
            "SCORE_CE_WEIGHT": 1.0,
            "SCORE_KL_WEIGHT": 1.0
        },
    },
    # {
    #     "distill_type": "top-k-tok",
    #     "k_percent": 25,
    #     "env": {
    #         "NO_ELIMINATE_SOFTMAX": "1",
    #         "NO_OFFLINE": "1",
    #         "DATASETS": "gsm8k",
    #         "DATASET_CONFIG": "main",
    #         "PROMPT_COL": "question",
    #         "ANSWER_COL": "answer",
    #     },
    # },
    {
        "distill_type": "top-k-tok",
        "k_percent": 25,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "NO_OFFLINE": "1",
            "FINEWEB_TOKENS": "10000000",
        },
    },
    {
        "distill_type": "pos-rs-kd",
        "k_percent": 25,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "NO_OFFLINE": "1",
            "FINEWEB_TOKENS": "10000000",
        },
    },
    {
        "distill_type": "top-k-tok",
        "k_percent": 25,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "NO_OFFLINE": "1",
            "GLS_ENABLED": "1",
            "FINEWEB_TOKENS": "10000000",
        },
    },
    {
        "distill_type": "top-k-tok",
        "k_percent": 25,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "FINEWEB_TOKENS": "10000000",
        },
    },
    {
        "distill_type": "pos-rs-kd",
        "k_percent": 25,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "FINEWEB_TOKENS": "10000000",
        },
    },
    {
        "distill_type": "top-k-tok",
        "k_percent": 25,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "GLS_ENABLED": "1",
            "FINEWEB_TOKENS": "10000000",
        },
    },
]


def _now() -> datetime:
    return datetime.utcnow()


class TeeStream:
    def __init__(self, *streams):
        self.streams = tuple(streams)

    def write(self, data: str) -> int:
        for stream in self.streams:
            try:
                stream.write(data)
            except Exception:
                pass
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass


def load_registry(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Failed to parse registry at {path}: {exc}") from exc
    if not isinstance(data, list):
        raise RuntimeError(f"Registry at {path} is not a JSON list")
    return data


def load_state(path: Path) -> dict:
    if not path.exists():
        return {
            "train_jobs": {},
            "eval_jobs": {},
            "printed_evals": [],
            "train_sequence_idx": 0,
        }
    with path.open("r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError:
            return {
                "train_jobs": {},
                "eval_jobs": {},
                "printed_evals": [],
                "train_sequence_idx": 0,
            }
    data.setdefault("train_jobs", {})
    data.setdefault("eval_jobs", {})
    data.setdefault("printed_evals", [])
    data.setdefault("train_sequence_idx", 0)
    return data


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, ensure_ascii=False, sort_keys=True)
    tmp_path.replace(path)


@dataclass
class JobInfo:
    job_id: str
    name: str
    state: str
    reason: str

    @property
    def is_active(self) -> bool:
        return self.state in {"R", "PD", "CF", "CG", "S"}


def fetch_squeue(user: str) -> Dict[str, JobInfo]:
    cmd = [
        "squeue",
        "-u",
        user,
        "-h",
        "-o",
        "%i|%j|%T|%R",
    ]
    try:
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"[warn] Failed to query squeue ({exc}); assuming no active jobs.", file=sys.stderr)
        return {}
    jobs: Dict[str, JobInfo] = {}
    for line in res.stdout.strip().splitlines():
        parts = line.split("|", 3)
        if len(parts) != 4:
            continue
        job_id, name, state, reason = parts
        jobs[job_id] = JobInfo(job_id=job_id.strip(), name=name.strip(), state=state.strip(), reason=reason.strip())
    return jobs


def count_jobs_by_prefix(jobs: Iterable[JobInfo], prefixes: Sequence[str]) -> int:
    prefix_tuple = tuple(prefixes)
    return sum(1 for job in jobs if job.name.startswith(prefix_tuple))


def infer_sweep_tag(output_dir: Optional[str]) -> Optional[str]:
    if not output_dir:
        return None
    try:
        parts = Path(output_dir).parts
        for part in parts:
            if part.startswith("kd_") and part.endswith("_out"):
                return part[3:-4]
    except Exception:
        return None
    return None


def ensure_list(x: Optional[Iterable[str]]) -> List[str]:
    if not x:
        return []
    return list(x)


def run_sbatch(
    script: Path,
    args: Sequence[str],
    *,
    env: Optional[dict] = None,
    job_name: Optional[str] = None,
) -> Optional[str]:
    cmd = ["sbatch"]
    if job_name:
        cmd.append(f"--job-name={job_name}")
    cmd.append(str(script))
    cmd.extend(args)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    try:
        res = subprocess.run(cmd, check=True, capture_output=True, text=True, env=merged_env)
    except subprocess.CalledProcessError as exc:
        print(f"[error] sbatch failed for {' '.join(shlex.quote(a) for a in cmd)}\n{exc.stderr}", file=sys.stderr)
        return None
    line = res.stdout.strip()
    if "Submitted batch job" in line:
        return line.split()[-1]
    return None


def join_datasets(datasets: Sequence[str]) -> str:
    return " ".join(datasets)


@dataclass
class SchedulerContext:
    registry: List[dict]
    state: dict
    squeue: Dict[str, JobInfo]
    train_script: Path
    eval_script: Path
    eval_suite: str
    max_train: int
    max_eval: int
    retry_delay: timedelta
    tag_default: str
    dry_run: bool
    train_sequence: List[dict]

    def active_train_jobs(self) -> Dict[str, JobInfo]:
        return {
            job_id: job
            for job_id, job in self.squeue.items()
            if job.name.startswith((TRAIN_JOB_PREFIX,) + TRAIN_NAME_FALLBACKS)
        }

    def active_eval_jobs(self) -> Dict[str, JobInfo]:
        return {
            job_id: job
            for job_id, job in self.squeue.items()
            if job.name.startswith((EVAL_JOB_PREFIX,) + EVAL_NAME_FALLBACKS)
        }

    def cleanup_state(self) -> None:
        # Drop stale train jobs
        active_train_ids = {job.job_id for job in self.active_train_jobs().values()}
        to_delete = []
        for run_id, meta in self.state.get("train_jobs", {}).items():
            job_id = str(meta.get("job_id", ""))
            if job_id and job_id in active_train_ids:
                continue
            # Remove once training completed OR job disappeared and delay passed
            entry = next((r for r in self.registry if r.get("id") == run_id), None)
            if entry and entry.get("completed_train"):
                to_delete.append(run_id)
                continue
            submitted_at = meta.get("submitted_at")
            if submitted_at:
                try:
                    submitted_time = datetime.fromisoformat(submitted_at)
                except ValueError:
                    submitted_time = None
            else:
                submitted_time = None
            if submitted_time and _now() - submitted_time < self.retry_delay:
                continue
            # otherwise drop so it can be retried immediately
            to_delete.append(run_id)
        for run_id in to_delete:
            self.state["train_jobs"].pop(run_id, None)

        # Drop stale eval jobs
        active_eval_ids = {job.job_id for job in self.active_eval_jobs().values()}
        to_delete = []
        for run_id, meta in self.state.get("eval_jobs", {}).items():
            job_id = str(meta.get("job_id", ""))
            if job_id and job_id in active_eval_ids:
                continue
            entry = next((r for r in self.registry if r.get("id") == run_id), None)
            if entry and entry.get("completed_eval"):
                to_delete.append(run_id)
                continue
            submitted_at = meta.get("submitted_at")
            if submitted_at:
                try:
                    submitted_time = datetime.fromisoformat(submitted_at)
                except ValueError:
                    submitted_time = None
            else:
                submitted_time = None
            if submitted_time and _now() - submitted_time < self.retry_delay:
                continue
            to_delete.append(run_id)
        for run_id in to_delete:
            self.state["eval_jobs"].pop(run_id, None)

    def consume_sequence_env(self, distill_type: str, k_percent: int) -> Dict[str, str]:
        cursor = int(self.state.get("train_sequence_idx", 0))
        for idx in range(cursor, len(self.train_sequence)):
            item = self.train_sequence[idx]
            if item.get("distill_type") not in (None, distill_type):
                continue
            if item.get("k_percent") not in (None, k_percent):
                continue
            self.state["train_sequence_idx"] = idx + 1
            env = item.get("env") or {}
            if env:
                print(f"[sequence] Applying template #{idx+1}: {distill_type} k={k_percent} env={env}")
            return dict(env)
        return {}

    def submit_training_if_needed(self) -> None:
        active_jobs = self.active_train_jobs()
        in_queue = len(active_jobs)
        available_slots = max(self.max_train - in_queue, 0)
        if available_slots <= 0:
            return

        for entry in self.registry:
            run_id = entry.get("id")
            if not run_id:
                continue
            if entry.get("completed_train"):
                continue
            if run_id in self.state["train_jobs"]:
                continue
            train_meta = entry.get("runs", {}).get("train", {})
            existing_job = train_meta.get("job_id")
            if existing_job and existing_job in self.squeue:
                # Already running
                continue

            params = entry.get("params", {})
            distill_type = params.get("distill_type", "top-k-tok")
            k_percent = params.get("k_percent", 0)
            datasets = ensure_list(params.get("datasets"))
            if not datasets:
                print(f"[warn] Run {run_id} missing datasets; skipping.")
                continue

            sweep_tag = infer_sweep_tag(train_meta.get("output_dir"))
            if not sweep_tag:
                sweep_tag = self.tag_default

            env = self.build_training_env(params)
            template_env = self.consume_sequence_env(distill_type, k_percent)
            if template_env:
                env.update(template_env)
            job_name = f"{TRAIN_JOB_PREFIX}{run_id[:8]}"
            args = [
                distill_type,
                str(k_percent),
                self.eval_suite,
                sweep_tag,
            ]
            if self.dry_run:
                print(f"[dry-run] Would submit train job for {run_id} -> {job_name} env={env}")
            else:
                job_id = run_sbatch(self.train_script, args, env=env, job_name=job_name)
                if job_id:
                    print(f"[submit] Train run {run_id[:8]} job={job_id} ({distill_type} k={k_percent})")
                    self.state["train_jobs"][run_id] = {
                        "job_id": job_id,
                        "submitted_at": _now().isoformat(),
                        "distill_type": distill_type,
                        "k_percent": k_percent,
                    }
                    available_slots -= 1
                    if available_slots <= 0:
                        break

    def build_training_env(self, params: dict) -> dict:
        env: Dict[str, str] = {}
        env["SEED"] = str(params.get("seed", 1337))
        env["EPOCHS"] = str(params.get("epochs", 1))
        env["FINEWEB_TOKENS"] = str(params.get("fineweb_tokens", 4000000))
        datasets = ensure_list(params.get("datasets"))
        if datasets:
            env["DATASETS"] = join_datasets(datasets)
        if params.get("dataset_config"):
            env["DATASET_CONFIG"] = str(params["dataset_config"])
        if params.get("prompt_col"):
            env["PROMPT_COL"] = str(params["prompt_col"])
        if params.get("answer_col"):
            env["ANSWER_COL"] = str(params["answer_col"])
        if not params.get("offline_cache", True):
            env["NO_OFFLINE"] = "1"
        if not params.get("eliminate_softmax", True):
            env["NO_ELIMINATE_SOFTMAX"] = "1"
        if params.get("gls_enabled"):
            env["GLS_ENABLED"] = "1"
        if params.get("deterministic"):
            env["DETERMINISTIC"] = "1"
        if params.get("anneal_kd_temperature"):
            env["ANNEAL_FLAG"] = "anneal"
        if params.get("score_token_selection"):
            env["SCORE_TOKEN_SELECTION"] = "1"
            env["SCORE_NORMALIZE"] = str(params.get("score_normalize", "z"))
            env["SCORE_ENTROPY_WEIGHT"] = str(params.get("score_entropy_weight", 1.0))
            env["SCORE_CE_WEIGHT"] = str(params.get("score_ce_weight", 1.0))
            env["SCORE_KL_WEIGHT"] = str(params.get("score_kl_weight", 1.0))
        else:
            # ensure we don't leak previous settings
            env.pop("SCORE_TOKEN_SELECTION", None)
        if params.get("distill_type") == "bucket":
            env["BUCKET_LOWER_PERCENT"] = str(params.get("bucket_lower_percent", 70))
            env["BUCKET_UPPER_PERCENT"] = str(params.get("bucket_upper_percent", 80))
        # Mirror RS-KD overrides when provided
        if "rs_alpha" in params:
            env["RS_ALPHA"] = str(params["rs_alpha"])
        if "rs_epsilon" in params:
            env["RS_EPSILON"] = str(params["rs_epsilon"])
        if "rs_floor" in params:
            env["RS_FLOOR"] = str(params["rs_floor"])
        if "rs_vocab_samples" in params:
            env["RS_VOCAB_SAMPLES"] = str(params["rs_vocab_samples"])
        if "rs_vocab_beta" in params:
            env["RS_VOCAB_BETA"] = str(params["rs_vocab_beta"])
        if "sampled_softmax_negatives" in params:
            env["SAMPLED_SOFTMAX_NEGATIVES"] = str(params["sampled_softmax_negatives"])
        return env

    def submit_eval_if_needed(self) -> None:
        active_jobs = self.active_eval_jobs()
        in_queue = len(active_jobs)
        available_slots = max(self.max_eval - in_queue, 0)
        if available_slots <= 0:
            return

        for entry in self.registry:
            run_id = entry.get("id")
            if not run_id or entry.get("completed_eval"):
                continue
            if entry.get("completed_train") is not True:
                continue
            if run_id in self.state["eval_jobs"]:
                continue
            if run_id in self.state["train_jobs"]:
                # let training-driven eval dependency handle it
                continue
            train_meta = entry.get("runs", {}).get("train", {})
            out_dir = train_meta.get("output_dir")
            if not out_dir:
                continue
            args = [out_dir, self.eval_suite, "from_path"]
            env = {}
            job_name = f"{EVAL_JOB_PREFIX}{run_id[:8]}"
            if self.dry_run:
                print(f"[dry-run] Would submit eval job for {run_id} -> {job_name}")
            else:
                job_id = run_sbatch(self.eval_script, args, env=env, job_name=job_name)
                if job_id:
                    print(f"[submit] Eval run {run_id[:8]} job={job_id} ({self.eval_suite})")
                    self.state["eval_jobs"][run_id] = {
                        "job_id": job_id,
                        "submitted_at": _now().isoformat(),
                    }
                    available_slots -= 1
                    if available_slots <= 0:
                        break

    def emit_eval_summaries(self) -> None:
        printed: List[str] = self.state.get("printed_evals", [])
        for entry in self.registry:
            run_id = entry.get("id")
            if not run_id or run_id in printed:
                continue
            evals = entry.get("evals", {})
            suite_data = evals.get(self.eval_suite)
            if not suite_data:
                continue
            averages = suite_data.get("averages", {})
            avg_all = averages.get("avg_all")
            model_path = suite_data.get("model_evaluated") or entry.get("runs", {}).get("train", {}).get("output_dir")
            print(f"[eval] Run {run_id[:8]} avg_all={avg_all} model={model_path}")
            printed.append(run_id)
        self.state["printed_evals"] = printed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automation loop for KD runs registry")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY, help="Path to runs.json registry")
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE, help="Path to persist small scheduling state")
    parser.add_argument("--train-script", type=Path, default=DEFAULT_TRAIN_SLURM, help="Path to train.slurm (or wrapper)")
    parser.add_argument("--eval-script", type=Path, default=DEFAULT_EVAL_SLURM, help="Path to evals.slurm script")
    parser.add_argument("--eval-suite", default="light", help="Evaluation suite to request (passed to train/eval scripts)")
    parser.add_argument("--max-train", type=int, default=4, help="Maximum concurrent train jobs (states R/PD) to allow")
    parser.add_argument("--max-eval", type=int, default=3, help="Maximum concurrent eval jobs")
    parser.add_argument("--interval", type=int, default=900, help="Polling interval in seconds (default 15 minutes)")
    parser.add_argument("--retry-minutes", type=int, default=30, help="Delay before requeueing a failed job")
    parser.add_argument("--tag", default=None, help="Fallback KD_SWEEP_TAG when none can be inferred")
    parser.add_argument("--user", default=os.environ.get("USER"), help="SLURM account/user to monitor (default: $USER)")
    parser.add_argument("--log-file", type=Path, default=None, help="Optional path to append all console output")
    parser.add_argument("--dry-run", action="store_true", help="Print intended actions without calling sbatch")
    parser.add_argument("--once", action="store_true", help="Run single iteration instead of looping")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    user = args.user or os.environ.get("USER")
    if not user:
        print("[error] Unable to determine SLURM user; set --user", file=sys.stderr)
        return 2

    tag_default = args.tag or f"auto_{datetime.utcnow().strftime('%Y%m%d_%H%M')}"
    registry_path = args.registry.resolve()
    train_script = args.train_script.resolve()
    eval_script = args.eval_script.resolve()
    state_path = args.state_file.resolve()

    state = load_state(state_path)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_handle = None
    if args.log_file:
        log_path = args.log_file.expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8", buffering=1)
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        log_handle.write(f"\n[{timestamp}] runs_autopilot start -- tag={tag_default}\n")
        log_handle.flush()
        sys.stdout = TeeStream(original_stdout, log_handle)
        sys.stderr = TeeStream(original_stderr, log_handle)

        def _restore_streams() -> None:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            if log_handle:
                try:
                    log_handle.flush()
                except Exception:
                    pass
                try:
                    log_handle.close()
                except Exception:
                    pass

        atexit.register(_restore_streams)
        print(f"[log] teeing output to {log_path}")

    while True:
        try:
            registry = load_registry(registry_path)
        except RuntimeError as exc:
            print(f"[error] {exc}", file=sys.stderr)
            time.sleep(args.interval)
            continue

        squeue = fetch_squeue(user)
        ctx = SchedulerContext(
            registry=registry,
            state=state,
            squeue=squeue,
            train_script=train_script,
            eval_script=eval_script,
            eval_suite=args.eval_suite,
            max_train=args.max_train,
            max_eval=args.max_eval,
            retry_delay=timedelta(minutes=args.retry_minutes),
            tag_default=tag_default,
            dry_run=args.dry_run,
            train_sequence=CUSTOM_TRAIN_SEQUENCE,
        )
        ctx.cleanup_state()
        ctx.emit_eval_summaries()
        ctx.submit_training_if_needed()
        ctx.submit_eval_if_needed()

        if not args.dry_run:
            save_state(state_path, state)

        if args.once:
            break
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
