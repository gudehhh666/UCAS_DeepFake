#!/bin/bash

MODEL_DIR="/home/custom/FSFM-main/fsfm-3c/finuetune/Celeb-tuning"
OUT_DIR='/home/custom/data/val_processed/'
export PYTHONPATH=/home/custom/detection
# for pth_file in "$MODEL_DIR"/*.pth; do
#     echo "正在评估: $pth_file"
#     CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 main_test_DfD.py \
#         --eval \
#         --apply_simple_augment \
#         --model vit_base_patch16 \
#         --nb_classes 2 \
#         --batch_size 320 \
#         --resume "$pth_file" \
#         --output_dir '/home/custom/data/val_processed/'
# done
# 
# pth=/home/custom/FSFM-main/fsfm-3c/pretrain/checkpoint/pretrained_models/VF2_ViT-B/checkpoint-te-400.pth
# pth=/home/custom/FSFM-main/fsfm-3c/finuetune/Celeb-tuning-0.4p0-ebd/checkpoint-0.pth
pth=/home/custom/FSFM-main/fsfm-3c/finuetune/Celeb-tuning-ebd/checkpoint-max_auc.pth
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29509 main_test_DfD.py \
    --eval \
    --apply_simple_augment \
    --model vit_base_patch16 \
    --nb_classes 2 \
    --batch_size 320 \
    --resume "$pth" \
    --output_dir $OUT_DIR