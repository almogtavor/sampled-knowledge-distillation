# Runs Registry Guide

## Overview

The `results/runs.json` file is a **unified registry** that tracks all training runs and their evaluations in one place. Each entry is keyed by a **deterministic hash** of the training hyperparameters, making it easy to:

- Avoid duplicate training runs
- Link evaluation results to their training configurations
- Compare experiments across time
- Track which models have been evaluated

## Registry Structure

Each entry in `runs.json` has this structure:

```json
{
  "id": "<sha256-hash-of-params>",
  "params": {
    "student_model": "Qwen/Qwen3-0.6B",
    "teacher_model": "Qwen/Qwen3-8B",
    "distill_type": "top-k-tok",
    "k_percent": 25,
    "batch_size": 1,
    "max_seq_len": 384,
    "epochs": 1,
    "fineweb_tokens": 4000000,
    ...
  },
  "status": "evaluated",
  "completed_train": true,
  "completed_eval": true,
  "runs": {
    "train": {
      "experiment": "distill-top-k-tok-20251002_1234",
      "job_id": "12345",
      "output_dir": "/path/to/trained/model"
    }
  },
  "evals": {
    "light": {
      "model_evaluated": "/path/to/evaluated/model",
      "results": {
        "gsm8k": {"exact_match,strict-match": 0.272, ...},
        "hellaswag": {"acc,none": 0.305, ...},
        ...
      },
      "averages": {
        "avg_exact_match,strict-match": 0.272,
        "avg_acc,none": 0.297,
        ...
      },
      "task_status": {
        "gsm8k": "ok",
        "hellaswag": "ok",
        ...
      }
    },
    "heavy": { ... }
  }
}
```

## Key Fields

### Top-level
- **`id`**: SHA256 hash of normalized params (deterministic identifier)
- **`params`**: Training configuration (filtered to exclude output paths, logging configs)
- **`status`**: Current state (`"started"`, `"trained"`, `"evaluated"`, `"skipped"`)
- **`completed_train`**: Boolean flag indicating training finished
- **`completed_eval`**: Boolean flag indicating evaluation finished

### `runs.train`
- **`experiment`**: Human-readable run name
- **`job_id`**: Slurm job ID (if applicable)
- **`output_dir`**: Path to trained model

### `evals.<suite>`
- **`model_evaluated`**: Path or HF hub ID of the evaluated model
- **`results`**: Per-task metrics (nested dict)
- **`averages`**: Aggregated metrics across tasks
- **`task_status`**: Per-task execution status (`"ok"`, `"failed"`, etc.)
- **`calibration`** *(optional)*: Summary of ECE/perplexity metrics, with per-task and average values

## How It Works

### During Training (`run_distillation.py`)
1. Computes hash from training config
2. Checks `runs.json` for existing entry with same hash
3. If `completed_train=True`, **skips training** (unless `--override`)
4. Otherwise, creates/updates entry with `status="started"`
5. After training, sets `status="trained"` and `completed_train=True`

### During Evaluation (`sampledkd/evaluations/eval.py`)
1. Loads model's training params (from `run_params.json` or `config.json`)
2. Computes hash from those params
3. Finds matching entry in `runs.json`
4. Stores evaluation results under `evals.<suite>`
5. Records which model was evaluated in `model_evaluated` field

## Common Queries

### Find all runs with a specific config
```bash
cat results/runs.json | jq '.[] | select(.params.distill_type == "top-k-tok" and .params.k_percent == 25)'
```

### Check which models have been evaluated
```bash
cat results/runs.json | jq '.[] | {id: .id, model: .evals.light.model_evaluated, avg: .evals.light.averages.avg_all}'
```

### Find runs that completed training but haven't been evaluated
```bash
cat results/runs.json | jq '.[] | select(.completed_train == true and .completed_eval == false)'
```

### Get best performing model by avg_all
```bash
cat results/runs.json | jq 'sort_by(.evals.light.averages.avg_all) | reverse | .[0]'
```

## Design Choices

### No Timestamps
The registry **intentionally excludes timestamps** to keep it clean and focused:
- Git tracks when the file changed
- Results are timeless facts (reproducibility)
- Cleaner diffs when comparing experiments

### Deterministic Hashing
Training configs are hashed **after filtering** out non-semantic fields like:
- Output directories (`output_dir`)
- Logging configs (`wandb_project`, `tensorboard_dir`)
- Checkpoint cadence (`checkpoint_steps`)

This ensures that two runs with identical training semantics get the same ID, even if they write to different paths.

### Multiple Evaluations Per Run
The same training run can be evaluated multiple times:
- Different suites (`light`, `heavy`)
- Different checkpoints from the same training job
- The `model_evaluated` field tracks which specific model was benchmarked

## Files

- **Registry**: `results/runs.json`
- **Registry logic**: `sampledkd/run_registry.py`
- **Training integration**: `run_distillation.py` (calls `upsert_run_start`, `mark_trained`)
- **Eval integration**: `sampledkd/evaluations/eval.py` (calls `upsert_eval_results`)

---

**TL;DR**: `runs.json` is a content-addressed database of training runs and their evaluations. Hash = training config. No duplicates, easy lookups, clean diffs.

## Automation helper

To keep a sweep moving without babysitting SLURM, you can use the polling helper in `tools/runs_autopilot.py`. It reads `results/runs.json`, keeps a small number of training jobs active, and will trigger evaluations once training finishes.

```bash
python tools/runs_autopilot.py \
  --max-train 3 \
  --max-eval 2 \
  --interval 900
```

Key assumptions:

- The registry already contains the parameter blobs for runs you want to execute (they appear automatically once a run starts).
- Training jobs are launched through `train.slurm`, which in turn submits `evals.slurm` when the `--eval-suite` argument is provided (default `light`).
- The helper keeps a tiny state file at `results/automation_state.json` so it can avoid resubmitting the same job if it is already queued.

Use `--dry-run` to see what _would_ be submitted without actually calling `sbatch`, and `--once` to run a single pass for scripting/debugging.
