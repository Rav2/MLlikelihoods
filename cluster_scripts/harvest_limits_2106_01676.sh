#!/bin/bash -l
#SBATCH --job-name=harvest_2106_01676
#SBATCH --output=logs/harvest_2106_01676_%j.out
#SBATCH --error=logs/harvest_2106_01676_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=10G
#SBATCH --time=1-00:00:00
#SBATCH --qos=long

# Compute the scan limits for 2106.01676 once and store them in the card, so
# later scans load them instead of re-running the find_min_S probe.
#
# The analysis has 4 patchsets over TWO background models (offshell x3 and
# onshell x1), so two probes cover all four.

echo "Current working directory: $(pwd)"

source /users/rafal.maselek/modules.sh
echo "Modules activated"

eval "$(conda shell.bash hook)"
conda activate likelihood
echo "Conda environment 'likelihood' activated"

which python
python --version

cd /users/rafal.maselek/ML_LHClikelihoods/sampling/

ANALYSIS=2106.01676
SIG=0.0                      # limits depend on the signal uncertainty
HARVEST=harvest_${ANALYSIS}

# 1. build the parameter file(s) needed to probe each distinct background model
python tools/harvest_limits.py prepare cards/${ANALYSIS}.yaml ${ANALYSIS} \
    --outdir ${HARVEST} --out-root ${HARVEST}/out --sig-rel-unc ${SIG} \
    --points 2 --low-lim-samples 50

# 2. run them (few points: only the limits matter here, not the statistics)
for params in ${HARVEST}/params-*.yaml; do
    echo "=== running ${params} ==="
    python sample.py "${params}" --log_dir ${HARVEST}/logs/
done

# 3. write the harvested limits into the card
python tools/harvest_limits.py merge cards/${ANALYSIS}.yaml \
    ${HARVEST}/out/*/*/metadata.json ${HARVEST}/out/*/*/results-*.json \
    --analysis ${ANALYSIS} --sig-rel-unc ${SIG}

echo "Harvest finished - cards/${ANALYSIS}.yaml now carries scan_limits."
