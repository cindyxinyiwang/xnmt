#!/bin/bash
#SBATCH -N 1
#SBATCH -n 5
#SBATCH --gres=gpu:1
#SBATCH --mem=35g
#SBATCH -t 0

set -x  # echo commands to stdout
set -e  # exit on error
module load singularity
singularity shell /projects/tir1/singularity/tensorflow-ubuntu-16.04.2-lts-nvidia-375.26.img


module load cuda-8.0 cudnn-8.0-5.1
#module load cudnn-8.0-5.1
export CUDA_HOME="/projects/tir1/cuda-8.0.27.1"
PATH=$HOME/bin:$PATH
PATH=$CUDA_HOME/bin:$PATH
export PATH  
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export CPATH=${CUDA_HOME}/include:$CPATH
export LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
#export CUDA_HOME="/projects/tir1/cuda-8.0.27.1"
export LD_LIBRARY_PATH=/home/xinyiw1/software/dynet-base/dynet/build/dynet/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/cudnn-8.0/lib64:$LD_LIBRARY_PATH
export CPATH=/opt/cudnn-8.0/include:$CPATH
export LIBRARY_PATH=/opt/cudnn-8.0/lib64:$LD_LIBRARY_PATH


echo $LD_LIBRARY_PATH
model_name=seq
python xnmt/xnmt_run_experiments.py examples/kftt_seq.yaml \
		--dynet-gpu \
		--dynet-mem 10000 
