---
name: Bug report
about: Create a bug report to help us improve Alpamayo
title: "[BUG]"
labels: "? - Needs Triage, bug"
assignees: 'yesfandiari'

---

**Describe the bug**
A clear and concise description of what the bug is.

**Steps/Code to reproduce bug**
Follow this guide http://matthewrocklin.com/blog/work/2018/02/28/minimal-bug-reports to craft a minimal bug report. This helps us reproduce the issue and resolve it more quickly.

**Expected behavior**
A clear and concise description of what you expected to happen.

**Environment overview (please complete the following information)**
 - Recipe affected: [e.g. recipes/alpamayo1_sft, recipes/alpamayo1_5_sft, recipes/alpamayo1_x_rl, recipes/alpamayo1_5_quant]
 - Deployment: [local from source, Slurm, or Cloud (specify provider)]
 - Framework for that recipe: [HF Trainer + DeepSpeed / Cosmos-RL (GRPO) / Model Optimizer (FP8, NVFP4)]
 - Base model / checkpoint: [e.g. Alpamayo-1.5-10B]; HuggingFace gated access granted? (yes/no)
 - Dataset: Physical AI AV data prepared? (`scripts/download_pai.py` / `scripts/curate_pai_samples.py`)

**Environment details**
 - Hardware: GPU type(s), VRAM, number of GPUs / nodes (many recipes are multi-GPU)
 - Operating System
 - CUDA / NVIDIA driver version (from `nvidia-smi`)
 - Key package versions: PyTorch, plus DeepSpeed / Cosmos-RL / Model Optimizer as applicable

**Additional context**
Add any other context about the problem here.
