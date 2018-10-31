#!/bin/bash
#
#	usage: ./meta_eval.sh
#

clear 

python3 metrics_setup.py build_ext --inplace

export OPENBLAS_NUM_THREADS=1

python3 metaseg_eval.py --NUM_CORES=40 --NUM_LASSO_LAMBDAS=50

