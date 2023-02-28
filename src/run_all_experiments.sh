#!/bin/bash
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=5000M
# SBATCH --time=0-01:00:00
sbatch run_cbas_fluorescence.sh --array=0-1
