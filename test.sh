#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=35g

module load singularity
singularity exec --nv /projects/tir1/singularity/ubuntu-16.04-lts_tensorflow-1.4.0_cudnn-8.0-v6.0.img \ 
		python xnmt/xnmt_run_experiments.py examples/oromo_seq.yaml \
		--dynet-gpu \
		--dynet-mem 10000 
