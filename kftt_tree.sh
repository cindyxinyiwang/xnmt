#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=35g
#SBATCH -t 0


module load singularity
singularity shell --nv /projects/tir1/singularity/ubuntu-16.04-lts_tensorflow-1.4.0_cudnn-8.0-v6.0.img

python xnmt/xnmt_run_experiments.py examples/kftt_wordlstm.yaml \
		--dynet-gpu \
		--dynet-mem 10000 \
		--dynet-autobatch 1
