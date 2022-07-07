#!/bin/bash

#SBATCH --job-name=train_contig_%j
#SBATCH --output=train_contig_%j.out
#SBATCH --cpus-per-task=8 # -c
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=24:00:00 
#SBATCH --gpus=a40:1
#SBATCH --partition=gpupro

eval "$(conda shell.bash hook)"

conda activate /dhc/projects/ukbiobank/derived/projects/contrastive_genetics_imaging/cm_r50_burden_scores_interpret/env/contig_cuda116

which python

nvidia-smi 

python train_contig_burden_simple.py

