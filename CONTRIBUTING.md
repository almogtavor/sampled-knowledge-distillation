## Quick Start

1. Install dependencies (you have to install [uv](https://docs.astral.sh/uv/) first):
```bash
uv venv
# Linux:
source .venv/bin/activate
# Windows
.venv/Scripts/activate
pip install uv
uv pip install -e .

# Installing CUDA dependencies (adjust to your CUDA version):
uv pip install .[cu118]
```

2. Install [Ollama](https://ollama.com/download) & Start with Qwen3-8B:
```bash
ollama run qwen3:8b
```

3. Run distillation:

We provide several distillation types:
* **vanilla** — KL between teacher & student on **all** tokens.
* **ekd** — KL computed **only** on *fork tokens*, i.e. those whose
  teacher-entropy is in the top-`--k_percent` percentile inside each
  example.
* **bucket** — KL computed on tokens with entropy in a specific percentile range,
  e.g., 70th-80th percentile (excludes both very low and very high entropy tokens).
* **linucb** — a contextual bandit (LinUCB) that observes a 6D feature vector per
  token (teacher entropy, teacher CE, student CE, KL, coarse POS bucket, normalized
  position) and learns online which tokens to distill. It enforces at least one
  token per sequence and logs overlap with the top-entropy quartile alongside reward
  improvements.

Add `--score_token_selection` to the **top-k-tok** or **bucket** modes to rank
tokens by a composite score that mixes teacher entropy, student
cross-entropy, and teacher-student KL. Tune the mix with
`--score_entropy_weight`, `--score_ce_weight`, `--score_kl_weight`, and pick a
normalization via `--score_normalize`.

We use Ollama's Qwen3-8B in 4-bit as teacher:
```bash
# Vanilla KD
python run_distillation.py \
    --teacher_model Qwen/Qwen3-8B \
    --student_model Qwen/Qwen3-0.6B \
    --distill_type vanilla \
    --datasets gsm8k \
    --dataset_config main \
    --prompt_col question \
    --answer_col answer \
    --output_dir ./kd_vanilla_run

# Entropy-guided KD (top-k token selection)
python run_distillation.py \
    --teacher_model Qwen/Qwen3-8B \
    --student_model Qwen/Qwen3-0.6B \
    --distill_type top-k-tok \
  --k_percent 20 \
    --datasets gsm8k \
    --dataset_config main \
    --prompt_col question \
    --answer_col answer \
    --output_dir ./kd_top_k_tok_run

# Bucket distillation (e.g., distill on 70th-80th percentile entropy tokens)
python run_distillation.py \
    --teacher_model Qwen/Qwen3-8B \
    --student_model Qwen/Qwen3-0.6B \
    --distill_type bucket \
    --bucket_lower_percent 70 \
    --bucket_upper_percent 80 \
    --datasets gsm8k \
    --dataset_config main \
    --prompt_col question \
    --answer_col answer \
    --output_dir ./kd_bucket_run

# Top-k selection with score-based ranking (entropy + CE + KL)
python run_distillation.py \
  --teacher_model Qwen/Qwen3-8B \
  --student_model Qwen/Qwen3-0.6B \
  --distill_type top-k-tok \
  --k_percent 20 \
  --score_token_selection \
  --score_entropy_weight 1.0 \
  --score_ce_weight 1.0 \
  --score_kl_weight 1.0 \
  --datasets gsm8k \
  --dataset_config main \
  --prompt_col question \
  --answer_col answer \
  --output_dir ./kd_score_run

# LinUCB contextual bandit distillation
python run_distillation.py \
  --teacher_model Qwen/Qwen3-8B \
  --student_model Qwen/Qwen3-0.6B \
  --distill_type linucb \
  --datasets gsm8k \
  --dataset_config main \
  --prompt_col question \
  --answer_col answer \
  --bandit_alpha 0.75 \
  --bandit_threshold 0.0 \
  --bandit_min_tokens 1 \
  --output_dir ./kd_linucb_run
```

Tune LinUCB-specific knobs with `--bandit_alpha`, `--bandit_lambda`, `--bandit_threshold`,
and `--bandit_max_tokens`. The trainer logs per-step selection statistics
(`bandit/selected_tokens`, `bandit/overlap_selected`) and the average reward
achieved by the bandit after each optimizer update (`bandit/avg_reward`,
`bandit/positive_reward_rate`).

## Requirements

- Python >=3.10
- CUDA-enabled GPU
- Ollama installed and running with Qwen3-8B
- Memory: Keep `batch_size * seq_len < 4000` to fit in low GPU memory (11GB VRAM).

## Benchmarks
Out-of-the-box the script supports the **AIME-2024** and **AIME-2025**
math benchmarks. We prepare each dataset as a JSONL file:

```json
{
  "id": "AIME24_q1",
  "question": "Find the sum of ...",
  "answer": "123"
}
```

Place them in a directory and pass `--datasets aime24.jsonl aime25.jsonl`.
The script prompts the teacher to generate a chain-of-thought (CoT)
followed by the boxed final answer, caches the result, and then trains the
student by teacher-forcing.


## Dataset Format

Prepare datasets as JSONL files:
```json
{
  "id": "example_1",
  "question": "Your question here...",
  "answer": "123"
}
```

# 🚀 Quick Training Guide

## 🔧 Initial Setup (One-time)

### 1. Connect with TensorBoard tunnel:
```bash
ssh -L 6006:localhost:6006 YOUR_USER@c-001.cs.tau.ac.il
```

### 2. Navigate to project directory:
```bash
cd /home/joberant/NLP_2425b/YOUR_USER/ekd/
```

### 3. Create virtual environment:
```bash
python3.10 -m venv --without-pip venv
source ./venv/bin/activate
```

For future sessions just run:

```bash
source ./venv/bin/activate
```

### 4. Install dependencies:
```bash
curl -sS https://bootstrap.pypa.io/get-pip.py | python
export TMPDIR="$PWD/tmp"; mkdir -p "$TMPDIR"
export XDG_CACHE_HOME="$TMPDIR/xdg-cache"
export PIP_CACHE_DIR="$TMPDIR/pip-cache"

pip install --extra-index-url https://download.pytorch.org/whl/cu118 \
    torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118

pip install --prefer-binary -r requirements.txt
```

## 📁 Sync Local Changes to Remote

### One-time sync:
```bash
rsync -avz --progress ./ YOUR_USER@c-001.cs.tau.ac.il:/home/joberant/NLP_2425b/YOUR_USER/ekd/
```

### Continuous sync (run in separate terminal):
```bash
watch -n 5 'rsync -avz ./ YOUR_USER@c-001.cs.tau.ac.il:/home/joberant/NLP_2425b/YOUR_USER/ekd/'
```

## 🏃 Cluster Workflows

### Automation (recommended)

Keep a small pool of jobs running without babysitting by using the orchestration loop in `tools/runs_autopilot.py`. It inspects `results/runs.json`, respects concurrency caps, and submits `train.slurm`/`evals.slurm` jobs for unfinished configs.

```bash
python tools/runs_autopilot.py --max-train 3 --max-eval 2 --interval 600 --tag april25
```

- Console output is tee'd to `logs/autopilot.log`, and a small state file is stored at `results/automation_state.json`.
- The default behaviour is to walk through `CUSTOM_TRAIN_SEQUENCE` inside the script. Edit that list for the combos you care about, or pass `--allow-registry-fallback` to let it pick runs straight from the registry backlog.
- Every job gets a `KD_SWEEP_TAG` derived from `--tag` plus the run serial, so checkpoints collect under `results/kd_<tag>-runXXXX_out/`.
- Add `--dry-run --once` if you want to inspect the plan without submitting.

### Manual sbatch training

```bash
sbatch train.slurm <distill_type> <k_percent> [light|heavy] [KD_SWEEP_TAG] [epochs] [anneal_flag]
```

- Example:
  ```bash
  sbatch train.slurm top-k-tok 20 light april25
  ```
- Omit the third argument (or pass an empty string) to skip auto-evaluation; passing `light` or `heavy` queues `evals.slurm` after a successful train.
- `KD_SWEEP_TAG` names the output bucket (e.g. `april25` → `results/kd_april25_out/`).
- Override knobs through environment variables: `sbatch --export=ALL,NO_OFFLINE=1,ALPHA_CE=0.3 train.slurm top-k-tok 20 light mytag`.
- `run_distillation.py` updates `results/runs.json`; if an identical config is already marked trained the job exits early (code 10) and no eval is re-submitted.
- Training logs live at `logs/train.<jobid>.log` and the full run artefacts are copied to `results/logs/<run_label>/`.

Monitor active jobs with:

```bash
tail -f logs/train.<jobid>.log
squeue -u $USER
```

Common one-liners:

```bash
sbatch train.slurm top-k-tok 20 light oct25
sbatch --export=ALL,NO_OFFLINE=1,ALPHA_CE=0.3 train.slurm top-k-tok 20 "" oct25
```

### Housekeeping

```bash
scancel -u $USER
tail -f $(ls -t logs/train.*.log | head -1)
nvidia-smi
```

## 📊 TensorBoard Monitoring

### Start TensorBoard server:
```bash
tensorboard --logdir tb --port 6006 --bind_all &
```

### View in browser:
Open: http://localhost:6006

## 📋 Quick Workflow

1. **Sync code**: `rsync -avz --progress ./ YOUR_USER@c-001.cs.tau.ac.il:/home/joberant/NLP_2425b/YOUR_USER/ekd/`
2. **Launch jobs**:
   - `python tools/runs_autopilot.py --max-train 3 --max-eval 2 --interval 600 --tag <tag>` (recommended), or
   - `sbatch train.slurm top-k-tok 20 light <tag>` for a single run.
3. **Monitor**: `squeue -u $USER` and `tail -f logs/train.<jobid>.log`
4. **Inspect metrics**: open TensorBoard on http://localhost:6006

## 🧪 Model Evaluation

`train.slurm` already submits `evals.slurm` when you pass `light` or `heavy`, but you can launch or repeat evaluations manually.

### Slurm (preferred)

```bash
sbatch evals.slurm <MODEL_PATH> <light|heavy> [from_hf|from_path] [OUTPUT_DIR] [extra args...]
```

- Typical calls:
  ```bash
  sbatch evals.slurm results/kd_april25_out/models/model_12345_20240422_2359_top_k_tok_k20 light
  sbatch evals.slurm results/kd_april25_out/models/model_12345_20240422_2359_top_k_tok_k20 heavy
  sbatch evals.slurm kd_top_k_tok_run_out_models/model_2617 light from_hf
  ```
- `MODEL_PATH` accepts HF repo IDs, local HF exports, or raw `.pt` checkpoints (auto-exported to a temporary HF directory).
- Include `from_hf` when the first argument is a Hub ID; omit it for local paths.
- Outputs default to `evaluation_json_results/`, and logs are streamed to `eval_runs/<model>.<jobid>.<suite>.log`.
- For a quick retry with automatic GPU fallbacks you can still call `./submit_eval.sh <MODEL_PATH> <light|heavy>`, but `evals.slurm` is the canonical entrypoint.

### Direct Python (rare)

```bash
python -m sampledkd.evaluations.eval <MODEL_PATH> --suite light --work_dir eval_runs --output_dir evaluation_json_results
```

Examples (only when you need to bypass Slurm for debugging):

```bash
python -m sampledkd.evaluations.eval results/kd_april25_out/models/model_12345_20240422_2359_top_k_tok_k20 --suite light
python -m sampledkd.evaluations.eval kd_top_k_tok_run_out_models/model_2617 --suite heavy --from_hf
```

### Monitoring

```bash
tail -f logs/eval.<jobid>.log
squeue -u $USER
```

Evaluation results are appended to `results/runs.json` so you can track coverage and avoid duplicate runs.

## 🌀 How to run (Slurm + W&B Sweeps)

We support both one-off Slurm runs and Weights & Biases Sweeps.

### Slurm (single run)

Run a training job with your desired method and K:

```bash
sbatch train.slurm top-k-tok 20 light 3
#            ^distill_type ^k_percent ^eval   ^epochs
```

Defaults used by the script:
- Dataset: FineWeb-Edu streaming (respecting `FINEWEB_TOKENS`, default 5,000,000)
- Models: teacher=`Qwen/Qwen3-8B`, student=`Qwen/Qwen3-0.6B`
- Checkpoints saved to `kd_<method>_run_out_models/model_<JOBID>`

Monitor logs:

```bash
tail -f logs/train.<jobid>.log
```

### W&B Sweeps (grid launchers)

We provide ready-to-use sweep YAMLs in `sweeps/`. First, create a sweep and then start an agent.

1) Create a sweep (pick one):

```bash
wandb sweep sweeps/sweep_compare_k.yaml
wandb sweep sweeps/sweep_compare_methods.yaml
wandb sweep sweeps/sweep_anneal_compare_methods.yaml
```

2) Copy the printed SWEEP_ID (e.g., `user/project/abcd1234`) and start an agent:

```bash
wandb agent user/project/abcd1234
```

Notes:
- The Slurm script exports all environment variables (`#SBATCH --export=ALL`) so the agent’s `WANDB_*` env flows into the job.
- You can run multiple agents (even on different nodes) to parallelize submissions; Slurm will queue them.
- For a quick pass, edit the YAML to set `epochs: [1]`.
- To customize the virtual env per run, add a sweep param and pass `--venv=${venv}` to the command in the YAML.

## 🔬 Entropy Ablation (Top‑k overlap)

Use `ablate.slurm` to run the entropy agreement ablation between exact entropy and truncated Top‑k+Tail.
sbatch ablate.slurm --model <HF_MODEL_DIR_OR_ID> --dataset <HF_DATASET> --prompt_col <PROMPT_COL> --answer_col <ANSWER_COL> [--dataset_config <NAME>] [--batch_size 4] [--max_seq_len 512] [--k_percent 20] [--m 20]
General form (sbatch forwards flags directly to the Python tool):
    --model kd_top_k_tok_run_out_models/model_2617 

```bash
sbatch ablate.slurm --model <MODEL_OR_DIR> --dataset <HF_DATASET> --prompt_col <PROMPT_COL> --answer_col <ANSWER_COL> [--dataset_config <NAME>] [--batch_size 4] [--max_seq_len 512] [--k_percent 20] [--m 20]
```

Examples:

- AIME (AI‑MO validation set)
- Columns: problem, answer | Dataset: AI-MO/aimo-validation-aime
```bash
sbatch ablate.slurm \
    --model eval_runs/exports/ekd_export_checkpoint_epoch2_step5055 \
    --dataset AI-MO/aimo-validation-aime \
    --prompt_col problem \
    --answer_col answer \
    --batch_size 4 \
    --max_seq_len 512 \
    --k_percent 20 \
    --m 20
```

- SVAMP
- Columns: Body, Answer | Dataset: ChilleD/SVAMP
```bash
sbatch ablate.slurm \
    --model Qwen/Qwen3-0.6B \
    --dataset ChilleD/SVAMP \
    --prompt_col Body \
    --answer_col Answer \
    --batch_size 4 \
    --max_seq_len 512 \
    --k_percent 20 \
    --m 20
```

- GSM8K (main config)
- Columns: question, answer | Dataset: gsm8k with config 'main'
```bash
sbatch ablate.slurm \
    --model Qwen/Qwen3-0.6B \
    --dataset gsm8k \
    --dataset_config main \
    --prompt_col question \
    --answer_col answer \
    --batch_size 4 \
    --max_seq_len 512 \
    --k_percent 20 \
    --m 20
```

Notes:
- The script picks the most free GPU and runs on cuda:0 inside that view.
- You can pin GPUs by setting CUDA_VISIBLE_DEVICES before sbatch.
- If you see OOM, reduce `--batch_size` or `--max_seq_len`.

## �📁 Output Locations

- **EKD model**: `/home/joberant/NLP_2425b/YOUR_USER/kd_ekd_run_out_model`
- **Vanilla model**: `/home/joberant/NLP_2425b/YOUR_USER/kd_vanilla_run_out_model`
- **TensorBoard logs**: `tb/ekd_experiment/` or `tb/vanilla_experiment/`
- **Training logs**: `logs/train.<jobid>.log`
- **Evaluation logs**: `logs/eval.<jobid>.log`
