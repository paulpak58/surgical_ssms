#!/bin/bash
#SBATCH --job-name=LIQUIDS5_CHOLEC80
#SBATCH --output=/home/ppak/S5/s5/script_outputs/LIQUIDS5_TEST_3.txt
#SBATCH -c 40
#SBATCH -n 1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=6000
#SBATCH --gres=gpu:volta:2

#source /etc/profile

#export LD_LIBRARY_PATH=/home/anaconda3/envs/surgical_ncp/lib

home=/ppak/surgical_ncp/src/cholec_pipeline/surgical-liquid-S5
python ${home}/s5/weighted_train_tcn_s5.py \
    --C_init=lecun_normal \
    --batchnorm=True \
    --bidirectional=True \
    --blocks=16 \
    --bsz=1 \
    --clip_eigs=True \
    --d_model=32 \
    --dataset=cholec80 \
    --epochs=35 \
    --jax_seed=16416 \
    --lr_factor=4.0 \
    --n_layers=12 \
    --opt_config=standard \
    --p_dropout=0.0 \
    --ssm_lr_base=0.01 \
    --ssm_size_base=32 \
    --warmup_end=1 \
    --weight_decay=0.05 \
    --eval_file_name=tcn_liquidS5_eval_0 \
    --seq_len=2700
