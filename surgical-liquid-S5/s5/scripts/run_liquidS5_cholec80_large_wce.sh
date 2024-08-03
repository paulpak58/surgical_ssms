#!/bin/bash
#SBATCH --job-name=LIQUIDS5_CHOLEC80
#SBATCH --output=/home/ppak/S5/s5/script_outputs/LIQUIDS5_TEST_7.txt
#SBATCH -c 40
#SBATCH -n 1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=6000
#SBATCH --gres=gpu:volta:2

#source /etc/profile

#export LD_LIBRARY_PATH=/home/anaconda3/envs/surgical_ncp/lib

home=/ppak/surgical_ncp/src/cholec_pipeline/surgical-liquid-S5
python ${home}/s5/train_liquidS5_cholec80.py \
    --C_init=lecun_normal \
    --batchnorm=True \
    --bidirectional=True \
    --blocks=16 \
    --bsz=1 \
    --clip_eigs=True \
    --d_model=1024 \
    --dataset=cholec80 \
    --epochs=30 \
    --jax_seed=16416 \
    --lr_factor=4 \
    --n_layers=6 \
    --opt_config=standard \
    --p_dropout=0.0 \
    --ssm_lr_base=0.0001 \
    --ssm_size_base=512 \
    --warmup_end=5 \
    --weight_decay=0.05 \
    --eval_file_name=liquidS5_eval_large_wce_0
