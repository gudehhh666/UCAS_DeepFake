#!/bin/bash

MODEL_DIR="/home/custom/FSFM-main/fsfm-3c/finuetune/Celeb-tuning"
OUT_DIR='/home/custom/data/test1/processed'
export PYTHONPATH=/home/custom/detection

pth=/home/custom/FSFM-main/fsfm-3c/finuetune/Celeb-tuning-ebd/checkpoint-max_auc.pth


python inference.py \
    --resume $pth \
    --data_path $OUT_DIR \
    --model vit_base_patch16 \
    --nb_classes 2 \
    --batch_size 320 \
    --eval