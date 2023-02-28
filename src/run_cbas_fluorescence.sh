#!/bin/bash
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
#export PYTHONPATH  # why is this necessary?
python run_cbas.py FLUORESCENCE