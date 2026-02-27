# <Invert4TVG> – AAAI 2026 Code Appendix  
---

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





### Data Preparation

1. **Public datasets**  
   
   ```bash
   see Invert4TVG\dataset\charades-sta
   ```





### set config

config 

```bash
torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12362" \
    Invert4TVG/qwen-vl-finetune/qwenvl/train/train_grpo.py \
    --deepspeed Invert4TVG/qwen-vl-finetune/scripts/zero3_offload.json \
    --output_dir testmodel \
    --model_name_or_path /home/share/svmd5vm0/home/scut_czy1/Qwen2.5-VL-7B-Instruct \
    --train_data_path Invert4TVG/dataset/charades-sta/train_verbswitch.json \
    --eval_data_path Charades/charades_annotation/val_verbswitch.json \
    --video_folder datasets/Charades_v1_480 \
    --dataset_name charades \
    --max_prompt_length 8192 \
    --max_completion_length 1024 \
    --num_generations 8 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --gradient_checkpointing True \
    --num_train_epochs 2 \
    --report_to tensorboard \
    --save_steps 100 \
    --save_total_limit 3 \
    --run_name qwen2vl-baseline \
```
