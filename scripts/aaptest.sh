#!/usr/bin/env bash
#
# Copyright (c) 2025 zhouwg(https://github.com/zhouwg)
#
# Accuracy And Performance test of various mulmat algotype on Hexagon-cDSP
#
set -e

PWD=`pwd`
PROJECT_HOME_PATH=`pwd`
PROJECT_ROOT_PATH=${PROJECT_HOME_PATH}

#algo_types="0 1 2 3 4 5 6 32 33"
algo_types="0 1 2 3 4 5 6 32"

for algo in $algo_types
do
    ${PROJECT_ROOT_PATH}/scripts/build-run-android.sh run_benchmark MUL_MAT 3 ${algo}
    ${PROJECT_ROOT_PATH}/scripts/build-run-android.sh run_testop    MUL_MAT 3 ${algo}
done
