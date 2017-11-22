#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=35g
#SBATCH -t 0


module load singularity
singularity exec --nv /projects/tir1/singularity/ubuntu-16.04-lts_tensorflow-1.4.0_cudnn-8.0-v6.0.img /projects/tir1/users/xinyiw1/xnmt/dynet_runner.sh
