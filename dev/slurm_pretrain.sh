#!/bin/bash -l
#SBATCH --partition=a100
#SBATCH --account=a100
#SBATCH --nodelist=euctemon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-01:00:00

hostname
echo $SLURM_JOB_ID
echo ${USER}
source /home/${USER}/.bashrc
pwd
source ~/miniconda3/etc/profile.d/conda.sh
conda activate visslcuda
which python
python tools/run_distributed_engines.py config=pretrain/mammo/simclr_resnet.yaml
