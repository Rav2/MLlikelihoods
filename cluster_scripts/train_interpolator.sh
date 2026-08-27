#!/bin/bash -l
#SBATCH --job-name=train_interp
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err
#SBATCH --partition=g                # GPU partition
#SBATCH --gres=gpu:V100:1            # request 1 V100 GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2            # match your interactive session
#SBATCH --mem=8G                    # match your interactive session
#SBATCH --time=2-00:00:00
#SBATCH --qos=medium

CONTAINER=/users/rafal.maselek/tf2.15-gpu.sif
WORKDIR=/users/rafal.maselek/ML_LHClikelihoods
TABLEDIR=$WORKDIR/tables/1908.08215-500k-fluct20%
SCRIPT=training/train.py

# Run inside the container
apptainer exec --nv --cleanenv \
	  -B $WORKDIR:/workspace \
	    $CONTAINER \
	    env PYTHONPATH=/workspace \
	    python3 /workspace/training/train.py \
	    $TABLEDIR/1908.08215-fine-tuned.csv NNAsimov_fine-tuned_v37

