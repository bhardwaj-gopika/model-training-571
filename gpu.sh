#!/bin/bash
#SBATCH --job-name=my_s3df_job   # Job name
#SBATCH --account=ad:ard-online  # Your account name
#SBATCH --partition=ampere       # Specify the partition (e.g., milano)
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --mem=80G                  # Amount of memory per node
#SBATCH --gpus=1
#SBATCH --time=10:00:00 

# Load environment details (conda env, modules, containers)
export CONDA_PREFIX=/sdf/group/ad/beamphysics/rroussel/miniforge3/
export PATH=${CONDA_PREFIX}/bin/:$PATH
source ${CONDA_PREFIX}/etc/profile.d/conda.sh
conda activate gpsr

python train2.py \
    --train-csv dataset-broad-train.csv \
    --val-csv dataset-broad-val.csv \
    --test-csv dataset-broad-test.csv \
    --cov-loss l1 --epochs 200 --patience 40 \
    --batch-size 256 --lr 1e-3 \
    --finetune-batch-sizes 32 8 \
    --finetune-epochs-per-stage 150 \
    --finetune-lr 1e-4 --finetune-lr-decay 0.5 \
    --finetune-plateau-patience 5 --finetune-min-lr 1e-6 \
    --dropout 0 \
    --output-dir model-output-571-broad
