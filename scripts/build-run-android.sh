#!/bin/bash
#
# Copyright (c) 2024-2025 The KanTV authors
#
# 1. build llama.cpp + ggml-hexagon backend on Linux for Android phone equipped with Qualcomm Snapdragon mobile SoC
#    this script will setup local dev envs automatically
#
# 2. verify prebuilt libggmldsp-skel.so on Android phone equipped with Qualcomm Snapdragon mobile SoC
#
# 3. compare performance of QNN-CPU,QNN-GPU,QNN-NPU,Hexagon-cDSP,ggml on Android phone equipped with Qualcomm Snapdragon mobile SoC
#
set -e

PWD=`pwd`
PROJECT_HOME_PATH=`pwd`
PROJECT_ROOT_PATH=${PROJECT_HOME_PATH}

#running path on Android phone
REMOTE_PATH=/data/local/tmp/
#LLM model file on Android phone
GGUF_MODEL_NAME=/sdcard/gemma-3-4b-it-Q8_0.gguf
#https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat-GGUF/blob/main/qwen1_5-1_8b-chat-q4_0.gguf
GGUF_MODEL_NAME=/sdcard/qwen1_5-1_8b-chat-q4_0.gguf

#Android NDK can be found at:
#https://developer.android.com/ndk/downloads
ANDROID_PLATFORM=android-34
ANDROID_NDK_VERSION=r28
ANDROID_NDK_NAME=android-ndk-${ANDROID_NDK_VERSION}
ANDROID_NDK_FULLNAME=${ANDROID_NDK_NAME}-linux.zip
ANDROID_NDK=${PROJECT_ROOT_PATH}/prebuilts/${ANDROID_NDK_NAME}

#QNN SDK can be found at:
#https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk
QNN_SDK_URL=https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk
QNN_SDK_VERSION=2.32.0.250228
QNN_SDK_VERSION=2.33.0.250327
QNN_SDK_VERSION=2.34.0.250424
QNN_SDK_PATH=${PROJECT_ROOT_PATH}/prebuilts/QNN_SDK/2.34.0.250424/

#Hexagon SDK can be found at:
#https://developer.qualcomm.com/software/hexagon-dsp-sdk/tools
HEXAGON_SDK_PATH=/opt/qcom/Hexagon_SDK/6.2.0.1
HEXAGON_SDK_PATH=${PROJECT_ROOT_PATH}/prebuilts/Hexagon_SDK/6.2.0.1

#available htp arch version:
#v68 --- Snapdragon 888
#v69 --- Snapdragon 8 Gen1
#v73 --- Snapdragon 8 Gen2
#v75 --- Snapdragon 8 Gen3
#v79 --- Snapdragon 8 Elite(aka Gen4)

#8Gen1
#HTP_ARCH_VERSION=v69
#HTP_ARCH_VERSION_a=V69

#8Gen2
#HTP_ARCH_VERSION=v73
#HTP_ARCH_VERSION_a=V73

#8Gen3
#HTP_ARCH_VERSION=v75
#HTP_ARCH_VERSION_a=V75

#8Elite
#HTP_ARCH_VERSION=v79
#HTP_ARCH_VERSION_a=V79

#default HTP_ARCH
#modify the following two lines to adapt to test phone
HTP_ARCH_VERSION=v79
HTP_ARCH_VERSION_a=V79

#available prebuilt libs can be found at prebuilts/ggml-dsp
#modify the following line to select the appropriate libggmldsp-skel.so
GGMLDSP_RELEASE_DATE=20250531

#running_params=" -mg 2 -ngl 99 -t 8 -fa 1 "
running_params=" -mg 2 -ngl 99 -t 8 "

function dump_vars()
{
    echo -e "ANDROID_NDK:          ${ANDROID_NDK}"
    echo -e "QNN_SDK_PATH:         ${QNN_SDK_PATH}"
    echo -e "HEXAGON_SDK_PATH:     ${HEXAGON_SDK_PATH}"
}


function show_pwd()
{
    echo -e "current working path:$(pwd)\n"
}


function check_hexagon_sdk()
{
    is_hexagon_llvm_exist=1
    if [ ! -f ${PROJECT_ROOT_PATH}/prebuilts/Hexagon_SDK/6.2.0.1/tools/HEXAGON_Tools/8.8.06/NOTICE.txt ]; then
        echo -e "${TEXT_RED}hexagon LLVM toolchain not exist, pls check...${TEXT_RESET}\n"
        is_hexagon_llvm_exist=0
    else
        printf "hexagon LLVM toolchain already exist\n\n"
    fi

    #download customized LLVM toolchain HEXAGON_TOOLs_8.8.06.tar.gz
    if [ ${is_hexagon_llvm_exist} -eq 0 ]; then
        echo -e "begin downloading hexagon LLVM toolchain \n"
        wget --no-config --quiet --show-progress -O ${PROJECT_ROOT_PATH}/prebuilts/Hexagon_SDK/6.2.0.1/tools/HEXAGON_Tools/HEXAGON_TOOLs_8.8.06.tar.gz https://github.com/kantv-ai/toolchain/raw/refs/heads/main/HEXAGON_TOOLs_8.8.06.tar.gz
        if [ $? -ne 0 ]; then
            printf "failed to download hexagon LLVM toolchain\n"
            exit 1
        else
            zcat ${PROJECT_ROOT_PATH}/prebuilts/Hexagon_SDK/6.2.0.1/tools/HEXAGON_Tools/HEXAGON_TOOLs_8.8.06.tar.gz | tar -C ${PROJECT_ROOT_PATH}/prebuilts/Hexagon_SDK/6.2.0.1/tools/HEXAGON_Tools -xvf -
            printf "install hexagon LLVM toolchain successfully\n\n"
        fi
    fi

    if [ ! -d ${HEXAGON_SDK_PATH} ]; then
        echo -e "HEXAGON_SDK_PATH ${HEXAGON_SDK_PATH} not exist, pls install it accordingly...\n"
        exit 0
    else
        printf "Qualcomm Hexagon SDK already exist:${HEXAGON_SDK_PATH} \n\n"
    fi
}


function check_and_download_qnn_sdk()
{
    is_qnn_sdk_exist=1

    if [ ! -d ${QNN_SDK_PATH} ]; then
        echo -e "QNN_SDK_PATH ${QNN_SDK_PATH} not exist, download it from ${QNN_SDK_URL}...\n"
        is_qnn_sdk_exist=0
    fi

    if [ ${is_qnn_sdk_exist} -eq 0 ]; then
        if [ ! -f ${PROJECT_ROOT_PATH}/prebuild/v${QNN_SDK_VERSION}.zip ]; then
            wget --no-config --quiet --show-progress -O ${PROJECT_ROOT_PATH}/prebuilts/v${QNN_SDK_VERSION}.zip https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/${QNN_SDK_VERSION}/v${QNN_SDK_VERSION}.zip
        fi
        if [ $? -ne 0 ]; then
            printf "failed to download Qualcomm QNN SDK to %s \n" "${QNN_SDK_PATH}"
            exit 1
        fi
        cd ${PROJECT_ROOT_PATH}/prebuilts/
        unzip v${QNN_SDK_VERSION}.zip
        printf "Qualcomm QNN SDK saved to ${QNN_SDK_PATH} \n\n"
        cd ${PROJECT_ROOT_PATH}
    else
        printf "Qualcomm QNN SDK already exist:${QNN_SDK_PATH} \n\n"
    fi
}


function check_and_download_ndk()
{
    is_android_ndk_exist=1

    if [ ! -d ${ANDROID_NDK} ]; then
        is_android_ndk_exist=0
    fi

    if [ ! -f ${ANDROID_NDK}/build/cmake/android.toolchain.cmake ]; then
        is_android_ndk_exist=0
    fi

    if [ ${is_android_ndk_exist} -eq 0 ]; then

        if [ ! -f ${PROJECT_ROOT_PATH}/prebuilts/${ANDROID_NDK_FULLNAME} ]; then
            wget --no-config --quiet --show-progress -O ${PROJECT_ROOT_PATH}/prebuilts/${ANDROID_NDK_FULLNAME} https://dl.google.com/android/repository/${ANDROID_NDK_FULLNAME}
        fi

        cd ${PROJECT_ROOT_PATH}/prebuilts/
        unzip ${ANDROID_NDK_FULLNAME}

        if [ $? -ne 0 ]; then
            printf "failed to download android ndk to %s \n" "${ANDROID_NDK}"
            exit 1
        fi
        cd ${PROJECT_ROOT_PATH}

        printf "android ndk saved to ${ANDROID_NDK} \n\n"
    else
        printf "android ndk already exist:${ANDROID_NDK} \n\n"
    fi
}


function build_arm64
{
    cmake -H. -B./out/android -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=latest -DCMAKE_C_FLAGS=-march=armv8.7-a -DGGML_HEXAGON=ON -DLLAMA_CURL=OFF -DQNN_SDK_PATH=${QNN_SDK_PATH} -DHEXAGON_SDK_PATH=${HEXAGON_SDK_PATH} -DHTP_ARCH_VERSION=${HTP_ARCH_VERSION}
    cd out/android
    make -j16
    show_pwd

    cd -
}

function build_arm64_debug
{
    cmake -H. -B./out/android -DCMAKE_BUILD_TYPE=Debug -DGGML_OPENMP=OFF -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=latest -DCMAKE_C_FLAGS=-march=armv8.7-a -DGGML_HEXAGON=ON -DLLAMA_CURL=OFF -DQNN_SDK_PATH=${QNN_SDK_PATH} -DHEXAGON_SDK_PATH=${HEXAGON_SDK_PATH} -DHTP_ARCH_VERSION=${HTP_ARCH_VERSION}
    cd out/android
    make -j16
    show_pwd

    cd -
}


function remove_temp_dir()
{
    if [ -d out/android ]; then
        echo "remove out/android directory in `pwd`"
        rm -rf out/android
    fi
}


function check_qnn_libs()
{
    #reuse the cached qnn libs on Android phone
    adb shell ls ${REMOTE_PATH}/libQnnCpu.so
    adb shell ls ${REMOTE_PATH}/libQnnGpu.so
    adb shell ls ${REMOTE_PATH}/libQnnHtp.so
    if [ $? -eq 0 ]; then
        printf "QNN libs already exist on Android phone\n"
    else
        update_qnn_libs
    fi
    update_qnn_cfg
}


function update_qnn_libs()
{
    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnSystem.so              ${REMOTE_PATH}/
    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnCpu.so                 ${REMOTE_PATH}/
    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnGpu.so                 ${REMOTE_PATH}/

    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnHtp.so                 ${REMOTE_PATH}/
    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnHtpNetRunExtensions.so ${REMOTE_PATH}/
    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnHtpPrepare.so          ${REMOTE_PATH}/
    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnHtp${HTP_ARCH_VERSION_a}Stub.so          ${REMOTE_PATH}/
    adb push ${QNN_SDK_PATH}/lib/hexagon-${HTP_ARCH_VERSION}/unsigned/libQnnHtp${HTP_ARCH_VERSION_a}Skel.so     ${REMOTE_PATH}/
}


function update_qnn_cfg()
{
    adb push ./scripts/ggml-hexagon.cfg ${REMOTE_PATH}/
}


function build_ggml_hexagon()
{
    show_pwd
    check_and_download_ndk
    check_and_download_qnn_sdk
    check_hexagon_sdk
    dump_vars
    remove_temp_dir
    build_arm64
}

function build_ggml_hexagon_debug()
{
    show_pwd
    check_and_download_ndk
    check_and_download_qnn_sdk
    check_hexagon_sdk
    dump_vars
    remove_temp_dir
    build_arm64_debug
}

#added on 05/31/2025, for purpose of non-tech factor
function prepare_ggmlhexagon()
{
    adb push ./scripts/ggml-hexagon-for-binary-lib.cfg ${REMOTE_PATH}/ggml-hexagon.cfg
    echo "adb push ${PROJECT_ROOT_PATH}/prebuilts/ggml-dsp/${GGMLDSP_RELEASE_DATE}/libggmlop-skel${HTP_ARCH_VERSION}.so ${REMOTE_PATH}/libggmlop-skel.so"
case "$HTP_ARCH_VERSION" in
    v69)
        adb push ${PROJECT_ROOT_PATH}/prebuilts/ggml-dsp/${GGMLDSP_RELEASE_DATE}/libggmlop-skel${HTP_ARCH_VERSION}.so ${REMOTE_PATH}/libggmlop-skel.so
    ;;
    v73)
        adb push ${PROJECT_ROOT_PATH}/prebuilts/ggml-dsp/${GGMLDSP_RELEASE_DATE}/libggmlop-skel${HTP_ARCH_VERSION}.so ${REMOTE_PATH}/libggmlop-skel.so
    ;;
    v75)
        adb push ${PROJECT_ROOT_PATH}/prebuilts/ggml-dsp/${GGMLDSP_RELEASE_DATE}/libggmlop-skel${HTP_ARCH_VERSION}.so ${REMOTE_PATH}/libggmlop-skel.so
    ;;
    v79)
        adb push ${PROJECT_ROOT_PATH}/prebuilts/ggml-dsp/${GGMLDSP_RELEASE_DATE}/libggmlop-skel${HTP_ARCH_VERSION}.so ${REMOTE_PATH}/libggmlop-skel.so
    ;;
    *)
        show_usage
        exit 1
    ;;
esac
}


function prepare_ggmldsp()
{
    adb push ./scripts/ggml-hexagon-for-binary-lib.cfg ${REMOTE_PATH}/ggml-hexagon.cfg
    echo "adb push ${PROJECT_ROOT_PATH}/prebuilts/ggml-dsp/${GGMLDSP_RELEASE_DATE}/libggmldsp-skel${HTP_ARCH_VERSION}.so ${REMOTE_PATH}/libggmldsp-skel.so"
case "$HTP_ARCH_VERSION" in
    v69)
        adb push ${PROJECT_ROOT_PATH}/prebuilts/ggml-dsp/${GGMLDSP_RELEASE_DATE}/libggmldsp-skel${HTP_ARCH_VERSION}.so ${REMOTE_PATH}/libggmldsp-skel.so
    ;;
    v73)
        adb push ${PROJECT_ROOT_PATH}/prebuilts/ggml-dsp/${GGMLDSP_RELEASE_DATE}/libggmldsp-skel${HTP_ARCH_VERSION}.so ${REMOTE_PATH}/libggmldsp-skel.so
    ;;
    v75)
        adb push ${PROJECT_ROOT_PATH}/prebuilts/ggml-dsp/${GGMLDSP_RELEASE_DATE}/libggmldsp-skel${HTP_ARCH_VERSION}.so ${REMOTE_PATH}/libggmldsp-skel.so
    ;;
    v79)
        adb push ${PROJECT_ROOT_PATH}/prebuilts/ggml-dsp/${GGMLDSP_RELEASE_DATE}/libggmldsp-skel${HTP_ARCH_VERSION}.so ${REMOTE_PATH}/libggmldsp-skel.so
    ;;
    *)
        show_usage
        exit 1
    ;;
esac
}


function prepare_run_on_phone()
{
    if [ $# != 1 ]; then
        print "invalid param"
        return
    fi
    program=$1

    check_qnn_libs

    if [ -f ./out/android/bin/libggml-cpu.so ]; then
        adb push ./out/android/bin/*.so ${REMOTE_PATH}/
    fi
    adb push ./out/android/bin/${program} ${REMOTE_PATH}/

    #for verify prebuilt binary library(built on 05/31/2025) on Hexagon cDSP
    #not used since 06/2025 and would be removed in the future
    #prepare_ggmlhexagon

    #for verify prebuilt binary library(after 06/2025) on Hexagon cDSP
    prepare_ggmldsp

    #for build library on Hexagon cDSP from the reference source codes in this project
    #adb push ./scripts/ggml-hexagon.cfg ${REMOTE_PATH}/ggml-hexagon.cfg

    adb shell chmod +x ${REMOTE_PATH}/${program}
}

function run_llamacli()
{
    prepare_run_on_phone llama-cli

    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/llama-cli ${running_params} -no-cnv -m ${GGUF_MODEL_NAME} -p \"introduce the movie Once Upon a Time in America briefly.\n\""

}


function run_llamabench()
{
    prepare_run_on_phone llama-bench

    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/llama-bench ${running_params} -m ${GGUF_MODEL_NAME}"

}


function run_test-ops()
{
    prepare_run_on_phone test-backend-ops

    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/test-backend-ops test"

}

function run_test-op()
{
    prepare_run_on_phone test-backend-ops

    echo "adb shell cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/test-backend-ops test -o $opname "

    echo "\n"
    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/test-backend-ops test -o $opname "

}


function run_benchmark()
{
    prepare_run_on_phone ggmlhexagon-benchmark

    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/ggmlhexagon-benchmark -t $opname -b $qnnbackend"

}


function print_oplist()
{
oplist="DUP
    ADD
    ADD1
    ACC
    SUB
    MUL
    DIV
    SQR
    SQRT
    LOG
    SIN
    COS
    SUM
    SUM_ROWS
    MEAN
    ARGMAX
    COUNT_EQUAL
    REPEAT
    REPEAT_BACK
    CONCAT
    SILU_BACK
    NORM
    RMS_NORM
    RMS_NORM_BACK
    GROUP_NORM

    MUL_MAT
    MUL_MAT_ID
    OUT_PROD

    SCALE
    SET
    CPY
    CONT
    RESHAPE
    VIEW
    PERMUTE
    TRANSPOSE
    GET_ROWS
    GET_ROWS_BACK
    DIAG
    DIAG_MASK_INF
    DIAG_MASK_ZERO
    SOFT_MAX
    SOFT_MAX_BACK
    ROPE
    ROPE_BACK
    CLAMP
    CONV_TRANSPOSE_1D
    IM2COL
    IM2COL_BACK
    CONV_TRANSPOSE_2D
    POOL_1D
    POOL_2D
    POOL_2D_BACK
    UPSCALE
    PAD
    PAD_REFLECT_1D
    ARANGE
    TIMESTEP_EMBEDDING
    ARGSORT
    LEAKY_RELU

    FLASH_ATTN_EXT
    FLASH_ATTN_BACK
    SSM_CONV
    SSM_SCAN
    WIN_PART
    WIN_UNPART
    GET_REL_POS
    ADD_REL_POS
    RWKV_WKV6
    GATED_LINEAR_ATTN"

echo "opname list: "
echo ${oplist}
}

function show_usage()
{
    echo "Usage:"
    echo "  $0 help"
    echo "  $0 print_oplist"
    echo "  $0 build"
    echo "  $0 build_debug (enable debug log for developers on ARM-AP side and cDSP side)"
    echo "  $0 updateqnnlib"
    echo "  $0 run_testops"
    echo "  $0 run_testop          [ADD/MUL_MAT]"
    echo "  $0 run_llamacli"
    echo "  $0 run_llamabench"
    echo "  $0 run_benchmark   ADD/MUL_MAT  0(QNN_CPU)/1(QNN_GPU)/2(QNN_NPU)/3(cdsp)/4(ggml)"

    echo -e "\n\n\n"
}


show_pwd

check_and_download_ndk
check_and_download_qnn_sdk
check_hexagon_sdk

if [ $# == 0 ]; then
    show_usage
    exit 1
elif [ $# == 1 ]; then
    if [ "$1" == "-h" ]; then
        show_usage
        exit 1
    elif [ "$1" == "help" ]; then
        show_usage
        exit 1
    elif [ "$1" == "print_oplist" ]; then
        print_oplist
        exit 1
    elif [ "$1" == "build" ]; then
        build_ggml_hexagon
        exit 0
    elif [ "$1" == "build_debug" ]; then
        build_ggml_hexagon_debug
        exit 0
    elif [ "$1" == "run_testops" ]; then
        run_test-ops
        exit 0
    elif [ "$1" == "run_llamacli" ]; then
        run_llamacli
        exit 0
    elif [ "$1" == "run_llamabench" ]; then
        run_llamabench
        exit 0
    elif [ "$1" == "updateqnnlib" ]; then
        update_qnn_libs
        exit 0
    else
        show_usage
        exit 1
    fi
elif [ $# == 2 ]; then
    opname=$2
#TODO: check opname in oplist
#opname can be found via print_oplist:

    run_test-op
    exit 0
elif [ $# == 3 ]; then
    opname=$2
    qnnbackend=$3
    run_benchmark
    exit 0
else
    show_usage
    exit 1
fi
