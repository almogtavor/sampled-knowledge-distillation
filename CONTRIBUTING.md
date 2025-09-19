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
curl -sS https://bootstrap.pypa.io/get-pip.py | python
```

### 4. Install dependencies:
```bash
source ./fastenv310/bin/activate
python -m pip install --no-cache-dir --prefer-binary --only-binary=:all: --index-url https://download.pytorch.org/whl/cu118 -r requirements.txt
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

### Submit EKD training:
```bash
sbatch train.slurm ekd
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
- **Benchmarks**: LM-Eval, Lighteval, EvalPlus, AlpacaEval
- **GPU allocation**: Automatic fallback (3→2→1 GPUs as available)
- **Cache management**: Handled via SLURM environment variables
- **Results**: Logged to W&B and TensorBoard

## 📁 Output Locations

- **EKD model**: `/home/joberant/NLP_2425b/YOUR_USER/kd_ekd_run_out_model`
- **Vanilla model**: `/home/joberant/NLP_2425b/YOUR_USER/kd_vanilla_run_out_model`
- **TensorBoard logs**: `tb/ekd_experiment/` or `tb/vanilla_experiment/`
- **Training logs**: `logs/train.<jobid>.log`
- **Evaluation logs**: `logs/eval.<jobid>.log`
