
# export CUDA_VISIBLE_DEVICES=2

# MODEL_BASE=Qwen/Qwen2.5-VL-3B-Instruct
MODEL_BASE=output/20250607_152539/checkpoint-2100

python qwen-vl-finetune/qwenvl/inference/evaluate.py \
     --model_base $MODEL_BASE \
     --dataset charades \
     --checkpoint_dir ckpt_charades
