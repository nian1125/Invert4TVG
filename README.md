
<p align="center">
  <h1 align="center">🔄 Invert4TVG</h1>
  <p align="center">
    <strong>A Temporal Video Grounding Framework with Inversion Tasks Preserving Action Understanding Ability</strong>
  </p>
  <p align="center">
    <a href="https://huggingface.co/ekko126/Invert4TVG-3B">
      <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow?style=for-the-badge" alt="Hugging Face Model">
    </a>
    <a href="#">
      <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10">
    </a>
    <a href="#">
      <img src="https://img.shields.io/badge/PyTorch-2.6+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
    </a>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/status-updating-brightgreen" alt="Status">
    <img src="https://img.shields.io/badge/license-MIT-blue" alt="License">
  </p>
</p>



## 📋 Table of Contents

- [Updates](#-updates)
- [Environment Setup](#-environment-setup)
- [Installation](#-installation)
- [Training & Evaluation](#-training--evaluation)
- [Citation](#-citation)

---

## 🚀 Updates

> **More updates are coming soon!** Stay tuned for new features and improvements.

🔥 **Model Released**: [Invert4TVG-3B](https://huggingface.co/ekko126/Invert4TVG-3B) is now available on Hugging Face!

---

## 🛠️ Environment Setup

### System Requirements

| Component | Specification |
|:---------:|:-------------:|
| **OS** | Ubuntu 20.04+ / macOS 12+ |
| **Python** | 3.10 (Anaconda recommended) |
| **Framework** | PyTorch 2.0+ / TensorFlow 2.12+ |
| **GPU** | NVIDIA A100 |
| **CUDA** | 11.8 |

### Quick Check

```bash
# Verify CUDA availability
nvidia-smi

# Check Python version
python --version  # Should be 3.10.x
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Invert4TVG.git
cd Invert4TVG

# Create conda environment (recommended)
conda create -n invert4tvg python=3.10
conda activate invert4tvg

# Install dependencies
pip install -r requirements.txt
```

<details>
<summary>📦 <strong>View full requirements</strong></summary>

```txt
torch>=2.0.0
transformers>=4.30.0
accelerate>=0.20.0
# ... other dependencies
```

</details>

---

## 🎯 Training & Evaluation

### 🏋️ Training

```bash
# Start training with GRPO on 3B model
bash Invert4TVG/qwen-vl-finetune/grpo_3b.sh
```

> 💡 **Tip**: Make sure you have sufficient GPU memory (A100 recommended) before starting training.

### 📊 Evaluation

```bash
# Run inference and evaluation
python Invert4TVG/qwen-vl-finetune/qwenvl/inference/evaluate.py \
    --model_path path/to/checkpoint \
    --test_data path/to/test.json
```

### Available Scripts

| Script | Purpose | GPU Memory |
|--------|---------|------------|
| `grpo_3b.sh` | Train 3B model with GRPO | ~40GB |
| `evaluate.py` | Inference and evaluation | ~20GB |

---






```
