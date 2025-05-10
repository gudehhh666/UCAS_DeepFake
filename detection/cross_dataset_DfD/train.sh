
outdir='/home/custom/FSFM-main/fsfm-3c/finuetune/Celeb-tuning-ebd-blk0'

rm -r $outdir
mkdir $outdir
export PYTHONPATH=/home/custom/detection


CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=1 python -m torch.distributed.launch --node_rank=0 --nproc_per_node=2 --master_port=29511 main_finetune_DfD.py \
    --accum_iter 1 \
    --apply_simple_augment \
    --batch_size 64 \
    --nb_classes 2 \
    --model vit_base_patch16 \
    --epochs 10 \
    --blr 1e-5 \
    --layer_decay 0.65 \
    --weight_decay 0.05 \
    --drop_path 0.1 \
    --reprob 0.25 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --dist_eval \
    \
    --finetune '/home/custom/FSFM-main/fsfm-3c/pretrain/checkpoint/pretrained_models/VF2_ViT-B/checkpoint-te-400.pth' \
    --finetune_data_path '/home/custom/FSFM-main/datasets/finetune_datasets/deepfakes_detection/Celeb-DF-v2/32_frames' \
    --output_dir $outdir  # default to ./checkpoint/$USR/experiments_finetune/$PID$ 