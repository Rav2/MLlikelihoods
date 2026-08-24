#!/bin/bash -l
#SBATCH --job-name=mcmc_array
#SBATCH --output=logs/mcmc_%A_%a.out
#SBATCH --error=logs/mcmc_%A_%a.err
#SBATCH --array=2761-2860
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=10G
#SBATCH --time=2-00:00:00
#SBATCH --qos=long

echo "Current working directory: $(pwd)"

# Activate any required modules
source /users/rafal.maselek/modules.sh
echo "Modules activated"

# Activate the Conda environment named 'likelihood'
eval "$(conda shell.bash hook)"
conda activate likelihood
echo "Conda environment 'likelihood' activated"

# Optionally check which Python is being used:
which python
python --version

# Change directory and run your Python script:
cd /users/rafal.maselek/ML_LHClikelihoods/sampling/
python sample.py /users/rafal.maselek/ML_LHClikelihoods/sampling/parameters.yaml --log_dir /users/rafal.maselek/slurm_scripts/logs/

echo "Python script finished!"
