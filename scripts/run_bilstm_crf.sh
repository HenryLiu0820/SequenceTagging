#!/bin/bash
#SBATCH --array=70-71
#SBATCH -p rise # partition (queue)
#SBATCH --nodelist=pavia
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=6 # number of cores per task
#SBATCH --gres=gpu:2
#SBATCH -t 1-24:00 # time requested (D-HH:MM)

pwd
hostname
date
echo starting job...
source ~/.bashrc
conda activate lzhenv
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

root=/scratch/zhliu/repos/SequenceTagging
cd ${root}

name=BiLSTM-CRF
load=True
print_tofile=True
datadir=/scratch/zhliu/data/seqtag
vocab_size=100000
seq_len=400
embed_dim=300
hidden_dim=128
epoch=10
batch_size=8
cuda=True
lr=0.001
weight_decay=5e-4
ckpt_path=/scratch/zhliu/checkpoints/${name}/epoch_${epoch}/lr_${lr}/embed_dim_${embed_dim}/hidden_dim_${hidden_dim}

mkdir -p ${ckpt_path}

cd src
pwd
CUDA_VISIBLE_DEVICES=0,1  python train.py \
    --name ${name} \
    --load ${load} \
    --print_tofile ${print_tofile} \
    --ckpt_path ${ckpt_path} \
    --datadir ${datadir} \
    --seq_len ${seq_len} \
    --vocab_size ${vocab_size} \
    --embed_dim ${embed_dim} \
    --hidden_dim ${hidden_dim} \
    --epoch ${epoch} \
    --batch_size ${batch_size} \
    --cuda ${cuda} \
    --lr ${lr} \
    --weight_decay ${weight_decay} \
