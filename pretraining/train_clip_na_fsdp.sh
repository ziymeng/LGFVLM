#!/bin/bash

# run "accelerate config" first!
# 50 epoch / 10h

#SBATCH -o trainPubMed_bs4_size96_cropped.%j.out    # 脚本执行的输出保存文件，%j是任务ID
#SBATCH -e trainPubMed_bs4_size96_cropped.%j.err    # 脚本执行的错误日志保存文件，%j是任务ID
#SBATCH --partition=compute  # 脚本执行的分区，无需修改
#SBATCH -J CroppedBioBertPretrain    # 脚本执行的任务名称（testJob01），可自定义
#SBATCH --nodes=1            # 脚本执行的节点（4卡以下=1,4卡以上=2）
#SBATCH --ntasks-per-node=6  # 脚本执行的CPU数（每个任务6核，可修改1~32）
#SBATCH --gres=gpu:4         # 脚本占用gpu数（1-8）

accelerate launch --config_file default_config.yaml --main_process_port 20688 pretraining_CLIP_fine-grained.py \
    --language_model_name_or_path /research/d1/rshr/xgzhou/code/M3D/M3D/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
    --version v0 \
    --ddp_timeout 7200 \
    --local_loss False \
    --gather_loss True \
    --bf16 True \
    --output_dir ./LaMed/output/zxg_train_valid_fsdp\
    --num_train_epochs 100 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_accumulation_steps 1 \
    --eval_steps 0.04 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True\
    --dataloader_num_workers 24 \
    --report_to tensorboard   

