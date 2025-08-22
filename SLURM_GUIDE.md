# SLURM Training Guide for TAU CS Cluster

## Overview
This guide helps you run your entropy knowledge distillation model on TAU's SLURM cluster.

## Files Created
1. `train.slurm` - Main training script (with containers)
2. `train_simple.slurm` - Simple training script (no containers)
3. `test.slurm` - Test script to verify setup
4. `run_train.sh` - Updated training shell script

## Step-by-Step Instructions

### 1. First Time Setup (on cluster)
```bash
# SSH to cluster client for setup
ssh c-001.cs.tau.ac.il

# Navigate to your project directory
cd /home/joberant/NLP_2425b/$USER/ekd

# ONE-TIME package installation in PROJECT DIRECTORY (not home!)
# This installs packages in /home/joberant/NLP_2425b/$USER/ekd/.venv/
sbatch install_packages.slurm

# Check installation progress
squeue --me

# When done, verify it worked
cat logs/install.<jobid>.out

# Alternative: Manual installation (if you prefer)
# chmod +x setup_local.sh
# ./setup_local.sh
```

### 2. Check Your Partitions
```bash
# Check what partitions you have access to
sacctmgr -P -i show user -s $USER

# Check available resources
sinfo -o "%20N  %10c  %10m  %25f  %10G"
```

### 3. Test Your Setup First
```bash
# Submit test job first to verify everything works
sbatch test.slurm

# Check job status
squeue --me

# Check output when done
cat logs/ekd-test.<jobid>.out
cat logs/ekd-test.<jobid>.err
```

### 4. Run Training

#### Main workflow (after one-time setup)
```bash
# After packages are installed once, just run training
sbatch train.slurm

# Training will automatically detect installed packages and skip reinstallation
```

#### If you forgot to install packages
```bash
# Training script will warn you and install packages anyway (slower)
# Or run the test first
sbatch test.slurm
```

### 5. Monitor Your Jobs
```bash
# Check job queue
squeue --me

# Check detailed job info
scontrol show job <jobid>

# Check job history
sacct -j <jobid>

# Cancel a job if needed
scancel <jobid>
```

## Partition Recommendations

Based on your access:

1. **For training**: `studentkillable` (1 day limit, available to you)
2. **For testing**: Use short time limits on `studentkillable`

Note: You only have access to `studentkillable` partition currently.

## Resource Guidelines

### GPU Requirements
- Your models (Qwen2.5-3B teacher, Qwen2.5-0.5B student) should fit on 1 GPU
- RTX 3090 (24GB) or A6000 (48GB) recommended
- Use `--constraint="geforce_rtx_3090|a6000"` if you want specific GPUs

### Memory and Time
- Start with 32GB RAM, 8 CPUs
- Begin with 12 hours, adjust based on actual training time
- Monitor first job to calibrate resource needs

## Troubleshooting

### Common Issues:
1. **Permission denied**: Check file permissions with `chmod +x run_train.sh`
2. **Import errors**: Verify Python environment setup
3. **GPU not found**: Check with `nvidia-smi` in job output
4. **Out of memory**: Reduce batch size or max_seq_len
5. **Time limit exceeded**: Increase `--time` parameter

### Debugging Tips:
1. Always run test job first
2. Check both .out and .err log files
3. Start with small experiments (1 epoch, small batch)
4. Use `studentrun` partition for quick tests

## File Locations on Cluster
- Project: `/home/joberant/NLP_2425b/$USER/ekd/`
- **Packages**: `/home/joberant/NLP_2425b/$USER/ekd/.venv/` ⭐ (NEW - in project dir!)
- Cache: `/home/joberant/NLP_2425b/$USER/hf_cache/`
- Output: `/home/joberant/NLP_2425b/$USER/kd_ekd_run_out_model/`
- Logs: `/home/joberant/NLP_2425b/$USER/ekd/logs/`

## Best Practices
1. Test with small datasets first
2. Use gradient accumulation to simulate larger batches
3. Save checkpoints regularly
4. Monitor GPU utilization
5. Don't hog resources - request only what you need
6. Plan ahead - cluster gets busy before deadlines

## Next Steps
1. Run `sbatch test.slurm` first
2. If test passes, run `sbatch train.slurm`
3. Monitor progress with `squeue --me`
4. Check logs in `logs/` directory
