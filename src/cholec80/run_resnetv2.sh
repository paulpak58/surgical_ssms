#!/bin/bash
#SBATCH --job-name=RUN_RESNETv2
#SBATCH --output=/home/ppak/surgical_ncp/RUN_RESNETv2_TEST5_SGD_4040_LR5e-4.txt
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=6000
#SBATCH --gres=gpu:volta:2

source /etc/profile
home=/home/ppak

export LD_LIBRARY_PATH=/home/anaconda3/envs/surgical_ncp_main/lib:${LD_LIBRARY_PATH}
python3 ${home}/surgical_ncp/train_resnetv2.py --model cnn --backbone resnet --lr 0.0005 --opt SGD --batch_size 100 --seq_len 1 --crop_mode 1 --wd 0 --epochs 60