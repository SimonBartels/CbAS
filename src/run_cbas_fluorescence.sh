#!/bin/bash
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=5000M
# SBATCH --time=0-01:00:00
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
#export PYTHONPATH  # why is this necessary?
python run_cbas.py FLUORESCENCE $SLURM_ARRAY_TASK_ID