#!/usr/bin/env bash
#
# Copyright (c) 2025 zhouwg(https://github.com/zhouwg)
#
# Accuracy And Performance test for ggml-hexagon's various mulmat algotype on Hexagon-cDSP
#
set -e

PWD=`pwd`
PROJECT_HOME_PATH=`pwd`
PROJECT_ROOT_PATH=${PROJECT_HOME_PATH}

#for my dev activity
#algo_types="0 1 2 3 4 5 6 32 33"
#algo_types="0 1 2 3 4 5 6 32"

#for third-party
#verified on Android phone equipped with Snapdragon 8Gen3 or Snapdragon 8Elite
algo_types="0 32"

${PROJECT_ROOT_PATH}/scripts/build-run-android.sh run_testops

for algo in $algo_types
do
    ${PROJECT_ROOT_PATH}/scripts/build-run-android.sh run_benchmark MUL_MAT 3 ${algo}
    ${PROJECT_ROOT_PATH}/scripts/build-run-android.sh run_testop    MUL_MAT 3 ${algo}
done

${PROJECT_ROOT_PATH}/scripts/build-run-android.sh run_llamacli  3
