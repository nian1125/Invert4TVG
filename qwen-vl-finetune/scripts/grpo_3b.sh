# export CUDA_VISIBLE_DEVICES=2

torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12362" \
    /home/share/svmd5vm0/home/scut_czy1/Invert4TVG/qwen-vl-finetune/qwenvl/train/train_grpo.py \
    --deepspeed /home/share/svmd5vm0/home/scut_czy1/Invert4TVG/qwen-vl-finetune/scripts/zero3_offload.json \
    --output_dir /home/svmd5vm0/whcs-share43/czy_output/testmodel \
    --model_name_or_path /home/share/svmd5vm0/home/scut_czy1/Qwen2.5-VL-3B-Instruct \
    --train_data_path /home/share/svmd5vm0/home/scut_czy1/Invert4TVG/dataset/charades-sta/train_verbswitch.json \
    --eval_data_path /home/share/svmd5vm0/home/scut_czy1/TimeZero/Charades/charades_annotation/val_verbswitch.json \
    --video_folder /home/share/svmd5vm0/home/scut_czy1/datasets/Charades_v1_480 \
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
    







