#!/bin/bash
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=5000M
# SBATCH --time=0-01:00:00
sbatch  --array=0-4%1 run_cbas_fluorescence.sh
sbatch  --array=0-4%1 run_cbas_vae.sh
