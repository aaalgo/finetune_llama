#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python3 finetune_peft.py \
    --model_path decapoda-research/llama-7b-hf \
    --dataset_path alpaca_data.json \
    --peft_mode lora \
    --lora_rank 8 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --max_steps 2500 \
    --learning_rate 2e-4 \
    --fp16 \
    --logging_steps 10 \
    --output_dir 7b
