#!/bin/bash
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
# TODO: replace fixed path environment by more generic env
# we may for example assume that the user created the environment in the default conda folder
BASEDIR=$(dirname "$0")
cd $BASEDIR
conda activate ../env
python poli_elbo_objective.py