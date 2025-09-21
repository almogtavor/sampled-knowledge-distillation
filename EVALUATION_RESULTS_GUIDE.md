# Evaluation Results Viewing Guide

## 📊 **Current Evaluation Configuration**

**Jobs Running:**
- Job 1005471: EKD model evaluation with math tasks
- Job 1005474: Vanilla model evaluation with math tasks

**Task Selection (Optimized for 15min max, now with math!):**
```python
LMEVAL_TASKS_SMALL = [
    "boolq",      # Boolean reasoning (2.5min)
    "piqa",       # Physical reasoning (1.8min) 
    "openbookqa", # Science QA (1.5min)
    "winogrande", # Commonsense reasoning (1.6min)
    "arc_easy",   # Grade school science (2.7min)
    "gsm8k",      # Grade school math word problems (2.0min)
    "svamp"       # Simple math variations (1.5min)
]
```
**Expected Total Time:** ~13-15 minutes per model (at 15min limit)

## 🔍 **How to View Results**

### **1. Weights & Biases (W&B)**
- **Project:** `selective-entropy-knowledge-distillation`
- **URL:** https://wandb.ai/selective-entropy-knowledge-distillation/selective-entropy-knowledge-distillation
- **Run Names:** 
  - `eval-vanilla` (vanilla knowledge distillation)
  - `eval-ekd` (entropy knowledge distillation)

**Metrics Format:**
- `winogrande/acc` - Winogrande accuracy
- `arc_easy/acc` - ARC Easy accuracy  
- `boolq/acc` - BoolQ accuracy
- `piqa/acc` - PIQA accuracy
- `openbookqa/acc` - OpenBookQA accuracy
- `gsm8k/acc` - GSM8K math accuracy
- `svamp/acc` - SVAMP math accuracy

### **2. TensorBoard**
**On Remote Server:**
```bash
# SSH with port forwarding
ssh -L 6006:localhost:6006 almogt@c-001.cs.tau.ac.il

# Navigate to project directory
cd /home/joberant/NLP_2425b/almogt/ekd

# Start TensorBoard
tensorboard --logdir=eval_runs/tb_logs --host=0.0.0.0 --port=6006
```

**Local Access:**
- Open browser to: http://localhost:6006
- View metrics in "Scalars" tab
- Logs located in: `eval_runs/tb_logs/vanilla/` and `eval_runs/tb_logs/ekd/`

### **3. Raw Results Files**
```bash
# SSH to server
ssh almogt@c-001.cs.tau.ac.il

# Navigate to results directory
cd /home/joberant/NLP_2425b/almogt/ekd/eval_runs/results

# View LM-Eval results
ls -la lmeval_*/
cat lmeval_*/results.json
```

## 📈 **Expected Metrics**

**Tasks and What They Measure:**
1. **BoolQ** - Yes/No reading comprehension
2. **PIQA** - Physical commonsense reasoning  
3. **OpenBookQA** - Elementary science knowledge
4. **Winogrande** - Pronoun resolution (commonsense)
5. **ARC Easy** - Grade-school level science questions
6. **GSM8K** - Grade school math word problems
7. **SVAMP** - Simple mathematical reasoning variations

**Key Metrics to Watch:**
- **Accuracy (`acc`)** - Primary metric for all tasks
- **Normalized scores** - Comparable across tasks
- **Standard errors** - Statistical confidence

## ⚡ **Performance Optimizations Applied**

**Multi-GPU Parallel Execution:**
- Each task runs on separate GPU for speed
- Timeout protection: 3-6min per task (task-specific)
- Automatic fallback if tasks fail

**Task Selection Rationale:**
- ❌ **Excluded `hellaswag`**: Always times out (>10min)
- ❌ **Excluded `hendrycks_math`**: Advanced math, too slow
- ✅ **Included math tasks**: GSM8K & SVAMP for mathematical reasoning
- ✅ **Included fast, reliable tasks**: 7 diverse reasoning types including math
- ✅ **Total runtime**: ~13-15min vs previous 20+ min attempts

## 🔄 **Alternative Configurations**

If you want different speed/coverage tradeoffs:

```python
# Ultra-fast (4min): Core tasks only
["winogrande", "arc_easy"]

# Fast (6min): Add physical reasoning  
["piqa", "winogrande", "arc_easy"]

# Balanced (8min): Add boolean QA + simple math
["boolq", "winogrande", "arc_easy", "svamp"] 

# Comprehensive (10min): All reliable tasks, no math
["boolq", "piqa", "openbookqa", "winogrande", "arc_easy"]

# Full (15min): All tasks including math
["boolq", "piqa", "openbookqa", "winogrande", "arc_easy", "gsm8k", "svamp"]
```

## 📝 **Current Status**
- ✅ W&B logging: Fully implemented and tested
- ✅ TensorBoard logging: Fully implemented and tested  
- ✅ 7-task evaluation with math: Running (jobs 1005471, 1005474)
- ✅ 15min time limit: Fully utilized for maximum coverage
- 🔄 Results: Available shortly in both W&B and TensorBoard

**Next Steps:**
1. Wait for job completion (~13-15min total)
2. Access results via W&B dashboard or TensorBoard
3. Compare EKD vs Vanilla performance across 7 benchmarks including math reasoning
