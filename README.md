## 🔧 1. Environment Dependencies  

| **Category** | **Version / Requirements**       |
| ------------ | -------------------------------- |
| OS           | Ubuntu 20.04+ or macOS 12+       |
| Python       | 3.10 (Anaconda recommended)      |
| DL Framework | PyTorch 2.0+ or TensorFlow 2.12+ |
| GPU & CUDA   | A100 GPU , CUDA 11.8             |
| Dependencies | See `requirements.txt`           |

---

## ⚙️ 2. Installation & Configuration  

```bash
cd Invert4TVG
pip install -r requirements.txt
```

## ⚙️ 3. train  & eva

```bash
#for train
bash Invert4TVG/qwen-vl-finetune/grpo_3b.sh
#for eva
python Invert4TVG/qwen-vl-finetune/qwenvl/inference/evaluate.py
```

