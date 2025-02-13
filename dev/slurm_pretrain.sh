#!/bin/bash -l
#SBATCH --partition=gpu_titanrtx
#SBATCH --gres=gpu:4
#SBATCH --time=5-00:00:00

hostname
echo $SLURM_JOB_ID
echo ${USER}
source /home/${USER}/.bashrc
pwd
conda activate visslmammo
which python

echo "klaar"
export VISSL_DATASET_CATALOG_PATH="configs/config/dataset_catalog.json"
python tools/run_distributed_engines.py config=pretrain/mammo/moco_resnet_400_imagenet_allviews.yaml
