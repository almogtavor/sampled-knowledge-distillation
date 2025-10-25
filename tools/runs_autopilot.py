#!/usr/bin/env python3
"""Simple SLURM orchestration loop driven by results/runs_test.json.

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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Ensure repo root is on sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pydantic import ValidationError

from sampledkd.config import TrainingConfig
from sampledkd.run_registry import compute_params_hash

DEFAULT_REGISTRY = Path("results/runs_test.json")
DEFAULT_STATE = Path("results/automation_state.json")
DEFAULT_TRAIN_SLURM = Path("train.slurm")
DEFAULT_EVAL_SLURM = Path("evals.slurm")

TRAIN_JOB_PREFIX = "ekdT-"
EVAL_JOB_PREFIX = "ekdE-"

# SLURM truncates job names around 8 characters, so the default names from
# train.slurm / evals.slurm often show up as "ekd-trai" and "ekd-eval".
TRAIN_NAME_FALLBACKS = ("ekd-train", "ekd-trai")
EVAL_NAME_FALLBACKS = ("ekd-eval",)

COUNTER_FILE = REPO_ROOT / "results/.autopilot_serial"

_DUMMY_OUTPUT_DIR = str(REPO_ROOT / "_autopilot_dummy_output")
_DUMMY_TENSORBOARD_DIR = "tb/_autopilot_dummy"


def _load_next_run_serial(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    last = 0
    try:
        raw = path.read_text(encoding="utf-8").strip()
        if raw:
            last = int(raw)
    except Exception:
        last = 0
    serial = last + 1
    path.write_text(str(serial), encoding="utf-8")
    return serial

# Optional manual submission sequence. Each entry is consumed in order and can
# override environment variables per training launch. The structure mirrors the
# sbatch lines used in kd_sweep.sh so you can copy/paste the combos you care
# about. Remove or comment out entries you don't need.
CUSTOM_TRAIN_SEQUENCE = [
    # RS-KD (distill all tokens) - to create the cache
    {
        "distill_type": "top-k-tok",
        "k_percent": 100,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "FINEWEB_TOKENS": "5000000",
            "ALPHA_CE": "0.3",
        },
        "display_name": "RS-KD (distill all tokens)",
    },
    {
        "distill_type": "top-k-tok",
        "k_percent": 100,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "FINEWEB_TOKENS": "5000000",
            "ALPHA_CE": "0.3",
            "NO_OFFLINE": "1",
        },
        "display_name": "RS-KD (distill all tokens, offline disabled)",
    },
    # RS-KD with higher CE weight
    # {
    #     "distill_type": "top-k-tok",
    #     "k_percent": 20,
    #     "env": {
    #         "NO_ELIMINATE_SOFTMAX": "1",
    #         "FINEWEB_TOKENS": "5000000",
    #         "ALPHA_CE": "0.3",
    #     },
    # },
    # RS-KD no CE
    {
        "distill_type": "top-k-tok",
        "k_percent": 20,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "FINEWEB_TOKENS": "5000000",
            "ALPHA_CE": "0.0",
        },
        "display_name": "RS-KD (top-20%, CE=0.0)",
    },
    {
        "distill_type": "top-k-tok",
        "k_percent": 20,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "FINEWEB_TOKENS": "5000000",
            "ALPHA_CE": "1.0",
        },
        "display_name": "RS-KD (top-20%, CE=1.0)",
    },
    # RS-KD only CE
    {
        "distill_type": "top-k-tok",
        "k_percent": 100,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "FINEWEB_TOKENS": "5000000",
            "ALPHA_CE": "0.0",
            "NO_OFFLINE": "1",
        },
        "display_name": "RS-KD (all tokens, CE=0.0, offline disabled)",
    },
    {
        "distill_type": "top-k-tok",
        "k_percent": 100,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "FINEWEB_TOKENS": "5000000",
            "ALPHA_CE": "0.0001",
            "NO_OFFLINE": "1",
        },
        "display_name": "RS-KD (all tokens, CE=0.0001, offline disabled)",
    },
    {
        "distill_type": "top-k-tok",
        "k_percent": 100,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "FINEWEB_TOKENS": "5000000",
            "ALPHA_CE": "0.1",
            "NO_OFFLINE": "1",
        },
        "display_name": "Full KD (all tokens, full softmax, CE=0.1)",
    },
    {
        "distill_type": "top-k-tok",
        "k_percent": 100,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "FINEWEB_TOKENS": "5000000",
            "ALPHA_CE": "1.0",
        },
        "display_name": "RS-KD (all tokens, CE=1.0)",
    },
    # Without offline cache (TSKD):
    # RS-KD (distill all tokens)
    {
        "distill_type": "top-k-tok",
        "k_percent": 100,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "NO_OFFLINE": "1",
            "FINEWEB_TOKENS": "5000000",
            "ALPHA_CE": "0.3",
        },
        "display_name": "Token-selective KD (distill all tokens, CE=0.3)",
    },

    # TSKD (entropy top-15%)
    {
        "distill_type": "top-k-tok",
        "k_percent": 15,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "FINEWEB_TOKENS": "5000000",
            "NO_OFFLINE": "1",
            "ALPHA_CE": "0.3",
        },
        "display_name": "Token-selective KD (entropy top-15%)",
    },

    # TSKD (entropy top-20%)
    {
        "distill_type": "top-k-tok",
        "k_percent": 20,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "FINEWEB_TOKENS": "5000000",
            "NO_OFFLINE": "1",
            "ALPHA_CE": "0.3",
        },
        "display_name": "Token-selective KD (entropy top-20%, CE=0.3)",
    },

    # TSKD (entropy top-25%)
    {
        "distill_type": "top-k-tok",
        "k_percent": 25,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "FINEWEB_TOKENS": "5000000",
            "NO_OFFLINE": "1",
            "ALPHA_CE": "0.3",
        },
        "display_name": "Token-selective KD (entropy top-25%)",
    },

    # TSKD (entropy top-30%)
    {
        "distill_type": "top-k-tok",
        "k_percent": 30,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "FINEWEB_TOKENS": "5000000",
            "NO_OFFLINE": "1",
            "ALPHA_CE": "0.3",
        },
        "display_name": "Token-selective KD (entropy top-30%)",
    },

    # TSKD (entropy top-75%)
    {
        "distill_type": "top-k-tok",
        "k_percent": 75,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "FINEWEB_TOKENS": "5000000",
            "NO_OFFLINE": "1",
            "ALPHA_CE": "0.3",
        },
        "display_name": "Token-selective KD (entropy top-75%)",
    },

    # TSKD (bucket of 5%-20%)
    {
        "distill_type": "bucket",
        "k_percent": 0,  # ignored by bucket; env below defines the band
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "FINEWEB_TOKENS": "5000000",
            "BUCKET_LOWER_PERCENT": "5",
            "BUCKET_UPPER_PERCENT": "20",
            "NO_OFFLINE": "1",
            "ALPHA_CE": "0.3",
        },
        "display_name": "Token-selective KD (bucket of 5%-20%)",
    },

    # TSKD (random 20%)
    {
        "distill_type": "random",
        "k_percent": 20,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "FINEWEB_TOKENS": "5000000",
            "RANDOM_TOKEN_SELECTION": "1",  # your training script should read this flag
            "NO_OFFLINE": "1",
            "ALPHA_CE": "0.3",
        },
        "display_name": "Token-selective KD (random 20%)",
    },

    # TSKD (pos-rs-kd top-20%)
    {
        "distill_type": "pos-rs-kd",
        "k_percent": 20,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "FINEWEB_TOKENS": "5000000",
            "NO_OFFLINE": "1",
            "ALPHA_CE": "0.3",
        },
        "display_name": "Token-selective KD (pos-rs-kd 20%, CE=0.3)",
    },

    # TSKD (top-20%, GLS queue 50k, offline cache disabled)
    {
        "distill_type": "top-k-tok",
        "k_percent": 20,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "FINEWEB_TOKENS": "5000000",
            "NO_OFFLINE": "1",
            "GLS_ENABLED": "1",
            "GLS_QUEUE_SIZE": "50000",
            "ALPHA_CE": "0.3",
        },
        "display_name": "Token-selective KD (entropy top-20%, GLS 50K, offline disabled)",
    },

    # TSKD (top-k-tok top-20%, GLS queue 50k, offline cache enabled)
    {
        "distill_type": "top-k-tok",
        "k_percent": 20,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "FINEWEB_TOKENS": "5000000",
            "GLS_ENABLED": "1",
            "GLS_QUEUE_SIZE": "50000",
            "ALPHA_CE": "0.3",
        },
        "display_name": "Token-selective KD (entropy top-20%, GLS 50K)",
    },

    # TSKD (entropy top-20%, GLS)
    {
        "distill_type": "top-k-tok",
        "k_percent": 20,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "GLS_ENABLED": "1",
            "FINEWEB_TOKENS": "5000000",
            "NO_OFFLINE": "1",
            "ALPHA_CE": "0.3",
        },
        "display_name": "Token-selective KD (entropy top-20%, GLS)",
    },

    # TSKD (score top-20%)
    # Combined score with z-normalization (entropy + CE + KL).
    {
        "distill_type": "top-k-tok",
        "k_percent": 20,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "FINEWEB_TOKENS": "5000000",
            "SCORE_TOKEN_SELECTION": "1",
            "SCORE_NORMALIZE": "z",
            "SCORE_ENTROPY_WEIGHT": "1.0",
            "SCORE_CE_WEIGHT": "1.0",
            "SCORE_KL_WEIGHT": "1.0",
            "NO_OFFLINE": "1",
        },
        "display_name": "Token-selective KD (score (equal) top-20%)",
    },

    # TSKD (LinUCB)
    {
        "distill_type": "linucb",
        "k_percent": 20,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "FINEWEB_TOKENS": "5000000",
            "NO_OFFLINE": "1",
        },
        "display_name": "Token-selective KD (LinUCB)",
    },

    
    
    # # RS-KD (distill all tokens)
    # {
    #     "distill_type": "top-k-tok",
    #     "k_percent": 100,
    #     "env": {
    #         "NO_ELIMINATE_SOFTMAX": "1",
    #         "FINEWEB_TOKENS": "5000000",
    #     },
    # },

    # Sampled KD (entropy top-15%)
    {
        "distill_type": "top-k-tok",
        "k_percent": 15,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "FINEWEB_TOKENS": "5000000",
            "ALPHA_CE": "0.3",
        },
        "display_name": "Sampled KD (entropy top-15%)",
    },

    # Sampled KD (entropy top-20%)
    {
        "distill_type": "top-k-tok",
        "k_percent": 20,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "FINEWEB_TOKENS": "5000000",
            "ALPHA_CE": "0.3",
        },
        "display_name": "Sampled KD (entropy top-20%)",
    },

    # Sampled KD (entropy top-25%)
    {
        "distill_type": "top-k-tok",
        "k_percent": 25,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "FINEWEB_TOKENS": "5000000",
            "ALPHA_CE": "0.3",
        },
        "display_name": "Sampled KD (entropy top-25%)",
    },

    # Sampled KD (entropy top-30%)
    {
        "distill_type": "top-k-tok",
        "k_percent": 30,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "FINEWEB_TOKENS": "5000000",
            "ALPHA_CE": "0.3",
        },
        "display_name": "Sampled KD (entropy top-30%)",
    },

    # Sampled KD (entropy top-75%)
    {
        "distill_type": "top-k-tok",
        "k_percent": 75,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "FINEWEB_TOKENS": "5000000",
            "ALPHA_CE": "0.3",
        },
        "display_name": "Sampled KD (entropy top-75%)",
    },

    # Sampled KD (bucket of 5%-20%)
    {
        "distill_type": "bucket",
        "k_percent": 0,  # ignored by bucket; env below defines the band
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "FINEWEB_TOKENS": "5000000",
            "BUCKET_LOWER_PERCENT": "5",
            "BUCKET_UPPER_PERCENT": "20",
            "ALPHA_CE": "0.3",
        },
        "display_name": "Sampled KD (bucket of 5%-20%)",
    },

    # Sampled KD (random 20%)
    {
        "distill_type": "random",
        "k_percent": 20,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "FINEWEB_TOKENS": "5000000",
            "RANDOM_TOKEN_SELECTION": "1",
            "ALPHA_CE": "0.3",
        },
        "display_name": "Sampled KD (random 20%)",
    },

    # Sampled KD (pos-rs-kd top-20%)
    {
        "distill_type": "pos-rs-kd",
        "k_percent": 20,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "FINEWEB_TOKENS": "5000000",
            "ALPHA_CE": "0.3",
        },
        "display_name": "Sampled KD (pos-rs-kd 20%)",
    },

    # Sampled KD (entropy top-20%, GLS)
    {
        "distill_type": "top-k-tok",
        "k_percent": 20,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "GLS_ENABLED": "1",
            "FINEWEB_TOKENS": "5000000",
            "ALPHA_CE": "0.3",
        },
        "display_name": "Sampled KD (entropy top-20%, GLS)",
    },

    # Sampled KD (score top-20%)
    # Combined score with z-normalization (entropy + CE + KL).
    {
        "distill_type": "top-k-tok",
        "k_percent": 20,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "FINEWEB_TOKENS": "5000000",
            "SCORE_TOKEN_SELECTION": "1",
            "SCORE_NORMALIZE": "z",
            "SCORE_ENTROPY_WEIGHT": "1.0",
            "SCORE_CE_WEIGHT": "1.0",
            "SCORE_KL_WEIGHT": "1.0",
        },
        "display_name": "Sampled KD (score top-20%)",
    },

    # Sampled KD (LinUCB)
    {
        "distill_type": "linucb",
        "k_percent": 20,
        "env": {
            "NO_ELIMINATE_SOFTMAX": "1",
            "FINEWEB_TOKENS": "5000000",
        },
        "display_name": "Sampled KD (LinUCB)",
    },
    # {
    #     "distill_type": "top-k-tok",
    #     "k_percent": 100,
    #     "env": {
    #         "NO_ELIMINATE_SOFTMAX": "1",
    #         "FINEWEB_TOKENS": "4000000",
    #     },
    # },
    # {
    #     "distill_type": "top-k-tok",
    #     "k_percent": 25,
    #     "env": {
    #         "NO_ELIMINATE_SOFTMAX": "1",
    #         "GLS_ENABLED": "1",
    #         "FINEWEB_TOKENS": "10000000",
    #     },
    # },
    # {
    #     "distill_type": "top-k-tok",
    #     "k_percent": 30,
    #     "env": {
    #         "NO_ELIMINATE_SOFTMAX": "1",
    #         "NO_OFFLINE": "1",
    #     },
    # },
    # {
    #     "distill_type": "top-k-tok",
    #     "k_percent": 20,
    #     "env": {
    #         "NO_ELIMINATE_SOFTMAX": "1",
    #     },
    # },
    # {
    #     "distill_type": "top-k-tok",
    #     "k_percent": 30,
    #     "env": {
    #         "NO_ELIMINATE_SOFTMAX": "1",
    #     },
    # },
    # {
    #     "distill_type": "top-k-tok",
    #     "k_percent": 25,
    #     "env": {
    #         "NO_ELIMINATE_SOFTMAX": "1",
    #         "NO_OFFLINE": "1",
    #         "STUDENT_MODEL": "Qwen/Qwen3-1.7B",
    #     },
    # },
    # {
    #     "distill_type": "top-k-tok",
    #     "k_percent": 25,
    #     "env": {
    #         "NO_ELIMINATE_SOFTMAX": "1",
    #         "STUDENT_MODEL": "Qwen/Qwen3-1.7B",
    #     },
    # },
    # {
    #     "distill_type": "top-k-tok",
    #     "k_percent": 25,
    #     "env": {
    #         "NO_ELIMINATE_SOFTMAX": "1",
    #         "NO_OFFLINE": "1",
    #         "SCORE_TOKEN_SELECTION": 1,
    #         "SCORE_NORMALIZE": "z",
    #         "SCORE_ENTROPY_WEIGHT": 1.0,
    #         "SCORE_CE_WEIGHT": 1.0,
    #         "SCORE_KL_WEIGHT": 1.0
    #     },
    # },
    # # {
    # #     "distill_type": "top-k-tok",
    # #     "k_percent": 25,
    # #     "env": {
    # #         "NO_ELIMINATE_SOFTMAX": "1",
    # #         "NO_OFFLINE": "1",
    # #         "DATASETS": "gsm8k",
    # #         "DATASET_CONFIG": "main",
    # #         "PROMPT_COL": "question",
    # #         "ANSWER_COL": "answer",
    # #     },
    # # },
    # {
    #     "distill_type": "top-k-tok",
    #     "k_percent": 25,
    #     "env": {
    #         "NO_ELIMINATE_SOFTMAX": "1",
    #         "NO_OFFLINE": "1",
    #         "FINEWEB_TOKENS": "10000000",
    #     },
    # },
    # {
    #     "distill_type": "pos-rs-kd",
    #     "k_percent": 25,
    #     "env": {
    #         "NO_ELIMINATE_SOFTMAX": "1",
    #         "NO_OFFLINE": "1",
    #         "FINEWEB_TOKENS": "10000000",
    #     },
    # },
    # {
    #     "distill_type": "top-k-tok",
    #     "k_percent": 25,
    #     "env": {
    #         "NO_ELIMINATE_SOFTMAX": "1",
    #         "NO_OFFLINE": "1",
    #         "GLS_ENABLED": "1",
    #         "FINEWEB_TOKENS": "10000000",
    #     },
    # },
    # {
    #     "distill_type": "top-k-tok",
    #     "k_percent": 25,
    #     "env": {
    #         "NO_ELIMINATE_SOFTMAX": "1",
    #         "FINEWEB_TOKENS": "10000000",
    #     },
    # },
    # {
    #     "distill_type": "pos-rs-kd",
    #     "k_percent": 25,
    #     "env": {
    #         "NO_ELIMINATE_SOFTMAX": "1",
    #         "FINEWEB_TOKENS": "10000000",
    #     },
    # },
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
        for key, value in env.items():
            if value is None:
                continue
            merged_env[key] = str(value)
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
    sequence_only: bool
    run_serial: int

    def __post_init__(self) -> None:
        self.completed_hashes = {
            entry.get("id")
            for entry in self.registry
            if entry.get("id") and entry.get("completed_train") and entry.get("completed_eval")
        }
        self._dummy_output_dir = _DUMMY_OUTPUT_DIR
        self._dummy_tensorboard_dir = _DUMMY_TENSORBOARD_DIR
        self.run_label = f"run{self.run_serial:04d}"
        if self.tag_default:
            self.tag_with_run = f"{self.tag_default}-{self.run_label}"
        else:
            self.tag_with_run = self.run_label

    @staticmethod
    def _coerce_int(value: Optional[object]) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_float(value: Optional[object]) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_bool(value: Optional[object]) -> Optional[bool]:
        if value is None:
            return None
        val = str(value).strip().lower()
        if val in {"0", "false", "off", "no", "none", ""}:
            return False
        if val in {"1", "true", "on", "yes"}:
            return True
        return None

    @staticmethod
    def _parse_iso_timestamp(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            ts = value.strip()
            if ts.endswith("Z"):
                ts = ts[:-1] + "+00:00"
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is not None:
                return dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt
        except Exception:
            return None

    def _apply_env_overrides(self, params: dict, env: Dict[str, str]) -> None:
        if not env:
            return
        if "NO_OFFLINE" in env:
            params["offline_cache"] = False
        if "OFFLINE_CACHE" in env:
            value = self._coerce_bool(env.get("OFFLINE_CACHE"))
            if value is not None:
                params["offline_cache"] = value
        if "NO_DDP_OFFLINE" in env:
            params["ddp_offline"] = False
        if "DDP_OFFLINE" in env:
            value = self._coerce_bool(env.get("DDP_OFFLINE"))
            if value is not None:
                params["ddp_offline"] = value
        if "NO_ELIMINATE_SOFTMAX" in env:
            params["eliminate_softmax"] = False
        if "GLS_ENABLED" in env:
            value = self._coerce_bool(env.get("GLS_ENABLED"))
            if value is not None:
                params["gls_enabled"] = value
        if "SCORE_TOKEN_SELECTION" in env:
            value = self._coerce_bool(env.get("SCORE_TOKEN_SELECTION"))
            if value is not None:
                params["score_token_selection"] = value
        if "SCORE_NORMALIZE" in env:
            params["score_normalize"] = str(env.get("SCORE_NORMALIZE"))
        for key_env, key_param in (
            ("SCORE_ENTROPY_WEIGHT", "score_entropy_weight"),
            ("SCORE_CE_WEIGHT", "score_ce_weight"),
            ("SCORE_KL_WEIGHT", "score_kl_weight"),
            ("ALPHA_CE", "alpha_ce"),
            ("RS_ALPHA", "rs_alpha"),
            ("RS_EPSILON", "rs_epsilon"),
            ("RS_FLOOR", "rs_floor"),
            ("RS_VOCAB_BETA", "rs_vocab_beta"),
        ):
            value = self._coerce_float(env.get(key_env))
            if value is not None:
                params[key_param] = value
        for key_env, key_param in (
            ("FINEWEB_TOKENS", "fineweb_tokens"),
            ("RS_VOCAB_SAMPLES", "rs_vocab_samples"),
            ("SAMPLED_SOFTMAX_NEGATIVES", "sampled_softmax_negatives"),
            ("BUCKET_LOWER_PERCENT", "bucket_lower_percent"),
            ("BUCKET_UPPER_PERCENT", "bucket_upper_percent"),
            ("GLS_QUEUE_SIZE", "gls_queue_size"),
            ("SEED", "seed"),
            ("EPOCHS", "epochs"),
        ):
            value = self._coerce_int(env.get(key_env))
            if value is not None:
                params[key_param] = value
        if env.get("DATASETS"):
            params["datasets"] = str(env.get("DATASETS")).split()
        if env.get("DATASET_CONFIG"):
            params["dataset_config"] = str(env.get("DATASET_CONFIG"))
        if env.get("PROMPT_COL"):
            params["prompt_col"] = str(env.get("PROMPT_COL"))
        if env.get("ANSWER_COL"):
            params["answer_col"] = str(env.get("ANSWER_COL"))

    def _build_candidate_params(
        self,
        base_params: Optional[dict],
        env: Dict[str, str],
        distill_type: Optional[str],
        k_percent: Optional[int],
    ) -> dict:
        params = dict(base_params or {})
        if distill_type is not None:
            params["distill_type"] = distill_type
        if k_percent is not None:
            params["k_percent"] = k_percent
        self._apply_env_overrides(params, env)
        return params

    def _canonicalize_params(self, params: dict) -> dict:
        if not params:
            return {}
        candidate = dict(params)
        candidate.setdefault("output_dir", self._dummy_output_dir)
        candidate.setdefault("tensorboard_dir", self._dummy_tensorboard_dir)
        try:
            cfg = TrainingConfig(**candidate)
        except ValidationError:
            return dict(params)
        canonical = cfg.model_dump()
        canonical.pop("output_dir", None)
        canonical.pop("tensorboard_dir", None)
        return canonical

    def _params_already_completed(self, params: dict) -> bool:
        if not params:
            return False
        hashes = set()

        def _add_hash(variant: dict) -> None:
            canonical = self._canonicalize_params(variant)
            target = canonical if canonical else dict(variant)
            try:
                hashes.add(compute_params_hash(target))
            except Exception:
                pass

        base = dict(params)
        _add_hash(base)

        if "ddp_offline" in base:
            alt = dict(base)
            alt["ddp_offline"] = not bool(base.get("ddp_offline"))
            _add_hash(alt)
            alt2 = dict(base)
            alt2.pop("ddp_offline", None)
            _add_hash(alt2)
        else:
            alt_true = dict(base)
            alt_true["ddp_offline"] = True
            _add_hash(alt_true)
            alt_false = dict(base)
            alt_false["ddp_offline"] = False
            _add_hash(alt_false)

        return any(h in self.completed_hashes for h in hashes if h)

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
            if meta.get("pseudo"):
                to_delete.append(run_id)
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

    def consume_sequence_env(
        self,
        distill_type: Optional[str],
        k_percent: Optional[object],
    ) -> Tuple[Dict[str, str], Optional[str], Optional[int], bool]:
        total_templates = len(self.train_sequence)
        if total_templates == 0:
            return {}, None, None, False

        cursor = int(self.state.get("train_sequence_idx", 0))
        distill_hint = distill_type
        k_hint = self._coerce_int(k_percent)
        for offset in range(total_templates):
            idx = (cursor + offset) % total_templates
            item = self.train_sequence[idx]
            item_distill = item.get("distill_type")
            item_k = self._coerce_int(item.get("k_percent"))
            if item_distill is not None and distill_hint is not None and item_distill != distill_hint:
                continue
            if item_k is not None and k_hint is not None and item_k != k_hint:
                continue
            self.state["train_sequence_idx"] = cursor + offset + 1
            env_raw = dict(item.get("env") or {})
            env = {key: str(value) for key, value in env_raw.items() if value is not None}
            display_name = item.get("display_name")
            if isinstance(display_name, str) and display_name.strip():
                env.setdefault("RUN_DISPLAY_NAME", display_name.strip())
            resolved_distill = item_distill or distill_hint
            resolved_k = item_k if item_k is not None else k_hint
            log_distill = resolved_distill or "default"
            log_k = resolved_k if resolved_k is not None else "default"
            if env:
                print(f"[sequence] Applying template #{idx+1}: {log_distill} k={log_k} env={env}")
            else:
                print(f"[sequence] Applying template #{idx+1}: {log_distill} k={log_k}")
            return env, resolved_distill, resolved_k, True
        return {}, None, None, False

    def submit_training_if_needed(self) -> None:
        active_jobs = self.active_train_jobs()
        in_queue = len(active_jobs)
        available_slots = max(self.max_train - in_queue, 0)
        if available_slots <= 0:
            return

        base_params: dict = {}
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
            if not base_params:
                base_params = params
            distill_type_param = params.get("distill_type")
            distill_type = distill_type_param or "top-k-tok"
            k_param = params.get("k_percent")
            k_percent = self._coerce_int(k_param)

            sweep_tag = infer_sweep_tag(train_meta.get("output_dir"))
            if not sweep_tag:
                sweep_tag = self.tag_with_run

            env = self.build_training_env(params)
            entry_display_name = entry.get("display_name")
            if isinstance(entry_display_name, str) and entry_display_name.strip():
                env.setdefault("RUN_DISPLAY_NAME", entry_display_name.strip())
            template_env, template_distill, template_k, matched_template = self.consume_sequence_env(
                distill_type_param,
                k_param,
            )
            if matched_template:
                if template_env:
                    env.update(template_env)
                if template_distill:
                    distill_type = template_distill
                if template_k is not None:
                    k_percent = template_k
            elif self.sequence_only:
                prefix = "[dry-run skip]" if self.dry_run else "[skip]"
                print(
                    f"{prefix} Run {run_id[:8]} distill_type={distill_type} k={k_percent} not in custom sequence."
                )
                continue
            job_name = f"{TRAIN_JOB_PREFIX}{self.run_label}-{run_id[:8]}"
            final_k_percent = k_percent if k_percent is not None else 0
            candidate_params = self._build_candidate_params(entry.get("params", {}), env, distill_type, k_percent)
            if self._params_already_completed(candidate_params):
                prefix = "[dry-run skip]" if self.dry_run else "[skip]"
                print(f"{prefix} Run {run_id[:8]} already completed (registry hash match).")
                continue
            args = [
                distill_type,
                str(final_k_percent),
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
                        "k_percent": final_k_percent,
                    }
                    available_slots -= 1
                    if available_slots <= 0:
                        break

        if available_slots <= 0:
            return

        if not base_params and self.registry:
            base_params = self.registry[0].get("params", {})

        base_env = self.build_training_env(base_params)
        base_env.pop("RUN_DISPLAY_NAME", None)
        default_distill = base_params.get("distill_type", "top-k-tok") if base_params else "top-k-tok"
        default_k = self._coerce_int(base_params.get("k_percent")) if base_params else None

        while available_slots > 0:
            template_env, template_distill, template_k, matched_template = self.consume_sequence_env(
                None,
                None,
            )
            if not matched_template:
                break

            distill_type = template_distill or default_distill or "top-k-tok"
            k_percent = template_k if template_k is not None else (default_k if default_k is not None else 0)

            env = dict(base_env)
            env.update(template_env)

            job_seq_idx = int(self.state.get("train_sequence_idx", 0)) - 1
            job_name = f"{TRAIN_JOB_PREFIX}{self.run_label}-seq{job_seq_idx:05d}"
            final_k_percent = k_percent if k_percent is not None else 0
            candidate_params = self._build_candidate_params(base_params, env, distill_type, k_percent)
            if self._params_already_completed(candidate_params):
                prefix = "[dry-run skip]" if self.dry_run else "[skip]"
                print(
                    f"{prefix} Sequence template idx={job_seq_idx} already completed (registry hash match)."
                )
                continue
            args = [
                distill_type,
                str(final_k_percent),
                self.eval_suite,
                self.tag_with_run,
            ]
            if self.dry_run:
                print(f"[dry-run] Would submit sequence job idx={job_seq_idx} -> {job_name} env={env}")
            else:
                job_id = run_sbatch(self.train_script, args, env=env, job_name=job_name)
                if job_id:
                    print(f"[submit] Sequence job idx={job_seq_idx} job={job_id} ({distill_type} k={k_percent})")
                    key = f"sequence::{job_id}"
                    self.state["train_jobs"][key] = {
                        "job_id": job_id,
                        "submitted_at": _now().isoformat(),
                        "distill_type": distill_type,
                        "k_percent": final_k_percent,
                        "template_idx": job_seq_idx,
                        "pseudo": True,
                    }
                    available_slots -= 1
                else:
                    break

    def build_training_env(self, params: dict) -> dict:
        env: Dict[str, str] = {}
        env["AUTOPILOT_RUN_SERIAL"] = str(self.run_serial)
        env["AUTOPILOT_RUN_LABEL"] = self.run_label
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
        ddp_offline = params.get("ddp_offline")
        if ddp_offline is True:
            env["DDP_OFFLINE"] = "1"
            env.pop("NO_DDP_OFFLINE", None)
        elif ddp_offline is False:
            env["NO_DDP_OFFLINE"] = "1"
            env.pop("DDP_OFFLINE", None)
        if not params.get("eliminate_softmax", True):
            env["NO_ELIMINATE_SOFTMAX"] = "1"
        if params.get("gls_enabled"):
            env["GLS_ENABLED"] = "1"
        if "gls_queue_size" in params:
            env["GLS_QUEUE_SIZE"] = str(params["gls_queue_size"])
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
            completed_at_raw = train_meta.get("completed_at")
            completed_at = self._parse_iso_timestamp(completed_at_raw)
            if completed_at is None:
                continue
            age = _now() - completed_at
            if age < timedelta(hours=9):
                continue
            args = [out_dir, self.eval_suite, "from_path"]
            env = {
                "AUTOPILOT_RUN_SERIAL": str(self.run_serial),
                "AUTOPILOT_RUN_LABEL": self.run_label,
            }
            display_name = entry.get("display_name")
            if isinstance(display_name, str) and display_name.strip():
                env["RUN_DISPLAY_NAME"] = display_name.strip()
            job_name = f"{EVAL_JOB_PREFIX}{self.run_label}-{run_id[:8]}"
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
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY, help="Path to runs_test.json registry")
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE, help="Path to persist small scheduling state")
    parser.add_argument("--train-script", type=Path, default=DEFAULT_TRAIN_SLURM, help="Path to train.slurm (or wrapper)")
    parser.add_argument("--eval-script", type=Path, default=DEFAULT_EVAL_SLURM, help="Path to evals.slurm script")
    parser.add_argument("--eval-suite", default="light", help="Evaluation suite to request (passed to train/eval scripts)")
    parser.add_argument("--max-train", type=int, default=4, help="Maximum concurrent train jobs (states R/PD) to allow")
    parser.add_argument("--max-eval", type=int, default=3, help="Maximum concurrent eval jobs")
    parser.add_argument("--interval", type=int, default=900, help="Polling interval in seconds (default 15 minutes)")
    parser.add_argument(
        "--min-interval",
        type=int,
        default=60,
        help="Minimum interval to use when free slots are available (default 60 seconds)",
    )
    parser.add_argument("--retry-minutes", type=int, default=30, help="Delay before requeueing a failed job")
    parser.add_argument("--tag", default=None, help="Fallback KD_SWEEP_TAG when none can be inferred")
    parser.add_argument("--user", default=os.environ.get("USER"), help="SLURM account/user to monitor (default: $USER)")
    parser.add_argument("--log-file", type=Path, default="logs/autopilot.log", help="Optional path to append all console output")
    parser.add_argument(
        "--allow-registry-fallback",
        action="store_true",
        help="Allow submitting registry entries that do not match the custom train sequence",
    )
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
    sequence_only = not args.allow_registry_fallback

    run_serial = _load_next_run_serial(COUNTER_FILE)
    run_label = f"run{run_serial:04d}"

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_handle = None
    if args.log_file:
        log_path = args.log_file.expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8", buffering=1)
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        log_handle.write(f"\n[{timestamp}] runs_autopilot start -- tag={tag_default} {run_label}\n")
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

    print(f"[session] autopilot run serial={run_serial} ({run_label})")

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
            sequence_only=sequence_only,
            run_serial=run_serial,
        )
        ctx.cleanup_state()
        ctx.emit_eval_summaries()
        ctx.submit_training_if_needed()
        ctx.submit_eval_if_needed()

        if not args.dry_run:
            save_state(state_path, state)

        if args.once:
            break

        sleep_seconds = args.interval
        if (args.max_train - len(ctx.active_train_jobs()) > 0) or (args.max_eval - len(ctx.active_eval_jobs()) > 0):
            sleep_seconds = min(args.interval, max(args.min_interval, 1))

        time.sleep(sleep_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
