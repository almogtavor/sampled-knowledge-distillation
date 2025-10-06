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
python3.10 -m venv --without-pip fastenv310
source ./fastenv310/bin/activate
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

## 🏃 Training Commands

### Submit Top K tokens training:
```bash
sbatch train.slurm top-k-tok
```

### Submit Vanilla training:
```bash
sbatch train.slurm vanilla
```

### Monitor training (replace `<jobid>` with actual job ID):
```bash
tail -f logs/train.<jobid>.log
```

### Check job status:
```bash
squeue -u YOUR_USER
```

## 📊 TensorBoard Monitoring

### Start TensorBoard server:
```bash
tensorboard --logdir tb --port 6006 --bind_all &
```

### View in browser:
Open: http://localhost:6006

## 🛠️ Useful Commands

### Kill all jobs:
```bash
scancel -u YOUR_USER
```

### View latest log automatically:
```bash
tail -f $(ls -t logs/train.*.log | head -1)
```

### Check GPU usage:
```bash
nvidia-smi
```

## 📋 Quick Workflow

1. **Sync code**: `rsync -avz --progress ./ YOUR_USER@c-001.cs.tau.ac.il:/home/joberant/NLP_2425b/YOUR_USER/ekd/`
2. **Submit job**: `sbatch train.slurm ekd`
3. **Monitor**: `tail -f $(ls -t logs/train.*.log | head -1)`
4. **View metrics**: http://localhost:6006

## 🧪 Model Evaluation

### Submit evaluation job:
```bash
./submit_eval.sh ekd <CHECKPOINT_NAME> light
```

**Examples:**
```bash
# Evaluate specific checkpoint
./submit_eval.sh ekd checkpoint_epoch1_step4527.pt light

# Evaluate final model (model.safetensors)
./submit_eval.sh ekd model.safetensors light
```

### Monitor evaluation:
```bash
# Check job status
squeue -u YOUR_USER

# View evaluation logs
tail -f logs/eval.<jobid>.log
```

### Available checkpoints:
- Check saved checkpoints: `ls -la kd_ekd_run_out_model/checkpoints/`
- Final trained model: `kd_ekd_run_out_model/model.safetensors`

### Evaluation details:
## 🧪 Model Evaluation

### Submit evaluation job (HF dir or checkpoint)

The evaluator now accepts a single model path:
- HF model directory (preferred), e.g. `kd_top_k_tok_run_out_models/model_2617`
- or a `.pt` checkpoint, which will be exported automatically to a temporary HF directory

```bash
# SLURM (recommended)
sbatch evals.slurm <MODEL_PATH> <SUITE: light|heavy>

# Direct (if you already have an env active)
python -m sampledkd.evaluations.eval <MODEL_PATH> --suite light
```

**Examples:**
```bash
# Evaluate a ready HF model directory (no export)
sbatch evals.slurm kd_top_k_tok_run_out_models/model_2617 light

# Evaluate a raw checkpoint (auto-export)
sbatch evals.slurm kd_top_k_tok_run_out_models/checkpoints/checkpoint_epoch2_step5055.pt heavy
```

No need to specify the original base model; the evaluator reads it from the checkpoint if needed.

### Monitor evaluation
- **Benchmarks**: LM-Eval, Lighteval, EvalPlus, AlpacaEval
- **GPU allocation**: Automatic fallback (3→2→1 GPUs as available)
- **Cache management**: Handled via SLURM environment variables
- **Results**: Logged to W&B and TensorBoard

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

## � Entropy Ablation (Top‑k overlap)

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
