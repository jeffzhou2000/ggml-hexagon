#!/usr/bin/env bash
#
# Copyright (c) 2024 Qualcomm(https://www.qualcomm.com/developer/blog/2024/11/introducing-new-opn-cl-gpu-backend-llama-cpp-for-qualcomm-adreno-gpu, original-author)
# Copyright (c) 2025 rmatif(https://github.com/rmatif, core-author)
# Copyright (c) 2025 zhouwg(https://github.com/zhouwg, co-author)
#
# build llama.cpp + ggml-opencl backend on Linux for Android phone
# this script will setup local dev envs automatically
#
set -e

######## part-1: don't modify contents in this part ########

PWD=`pwd`
PROJECT_HOME_PATH=`pwd`
PROJECT_ROOT_PATH=${PROJECT_HOME_PATH}
HOST_CPU_COUNTS=`cat /proc/cpuinfo | grep "processor" | wc | awk '{print int($1)}'`

#running path on Android phone
REMOTE_PATH=/data/local/tmp

#Android NDK can be found at:
#https://developer.android.com/ndk/downloads
ANDROID_PLATFORM=android-34
ANDROID_NDK_VERSION=r28
ANDROID_NDK_NAME=android-ndk-${ANDROID_NDK_VERSION}
ANDROID_NDK_FULLNAME=${ANDROID_NDK_NAME}-linux.zip
ANDROID_NDK=${PROJECT_ROOT_PATH}/prebuilts/${ANDROID_NDK_NAME}

# --- Define NDK paths based on the absolute SDK path ---
NDK_TOOLCHAIN_SYSROOT_INCLUDE_PATH="${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include"
NDK_TOOLCHAIN_SYSROOT_ARM64_LIB_PATH="${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android"

#OpenCL Headers can be found at:
#https://https://github.com/KhronosGroup/OpenCL-Headers
OPENCL_SDK_URL=https://github.com/KhronosGroup/OpenCL-Headers
OPENCL_SDK_PATH=${PROJECT_ROOT_PATH}/prebuilts/OpenCL_SDK
OPENCL_HEADERS_PATH=${OPENCL_SDK_PATH}/OpenCL-Headers

#running_params=" -ngl 99 -t 4 -n 256 --no-warmup -fa 1 "
running_params=" -ngl 99 -t 4 -n 256 --no-warmup "

######## part-2: contents in this part can be modified ########
#supported htp arch version:
#v68 --- Snapdragon 888
#v69 --- Snapdragon 8 Gen1
#v73 --- Snapdragon 8 Gen2
#v75 --- Snapdragon 8 Gen3
#v79 --- Snapdragon 8 Elite
HTP_ARCH_VERSION=v79

PROMPT_STRING="introduce the movie Once Upon a Time in America briefly.\n"

#for llama-cli, 1.1 GiB, will be downloaded automatically via this script when running this script at the first time
TEST_MODEL_NAME=/sdcard/qwen1_5-1_8b-chat-q4_0.gguf
#for llama-cli, 4.5 GiB, will be downloadded automatically via this script when running this script at the first time
#TEST_MODEL_NAME=/sdcard/gemma-3n-E2B-it-Q8_0.gguf

#for llama-bench, 4.5 GiB, will be downloadded automatically via this script when running this script at the first time
GGUF_MODEL_NAME=/sdcard/gemma-3n-E2B-it-Q8_0.gguf
#for llama-bench, 1.12 GiB, will be downloadded automatically via this script when running this script at the first time
GGUF_MODEL_NAME=/sdcard/qwen1_5-1_8b-chat-q4_0.gguf
GGUF_MODEL_NAME=/sdcard/Llama-3.2-1B-Instruct-f16.gguf

######## part-3: utilities and functions ########

function dump_vars()
{
    echo -e "ANDROID_NDK:          ${ANDROID_NDK}"
    echo -e "OPENCL_SDK_PATH:      ${OPENCL_SDK_PATH}"
}


function show_pwd()
{
    echo -e "current working path:$(pwd)\n"
}


function check_command_in_host()
{
    set +e
    cmd=$1
    ls /usr/bin/${cmd}
    if [ $? -eq 0 ]; then
        #printf "${cmd} already exist on host machine\n"
        echo ""
    else
        printf "${cmd} not exist on host machine, pls install command line utility ${cmd} firstly and accordingly\n"
        exit 1
    fi
    set -e
}


function check_commands_in_host()
{
    check_command_in_host wget
    check_command_in_host git
    check_command_in_host ninja
}


function check_android_phone()
{
    adb shell ls /bin/ls
    if [ ! $? -eq 0 ]; then
        printf "pls check Android phone is connected properly\n"
        exit 1
    fi
}


function check_and_download_opencl_sdk()
{
    is_opencl_sdk_exist=1

    if [ ! -d ${OPENCL_SDK_PATH} ]; then
        echo -e "OPENCL_SDK_PATH ${OPENCL_SDK_PATH} not exist, download it from ${OPENCL_SDK_URL}...\n"
        is_opencl_sdk_exist=0
    fi
    if [ ! -f ${NDK_TOOLCHAIN_SYSROOT_ARM64_LIB_PATH}/libOpenCL.so ]; then
        echo -e "${NDK_TOOLCHAIN_SYSROOT_ARM64_LIB_PATH}/libOpenCL.so not exist...\n"
        is_opencl_sdk_exist=0
    fi

    if [ ${is_opencl_sdk_exist} -eq 0 ]; then
        mkdir -p ${OPENCL_SDK_PATH}
        cd ${OPENCL_SDK_PATH}

        if [ ! -d OpenCL-Headers ]; then
            echo "Cloning OpenCL-Headers..."
            git clone https://github.com/KhronosGroup/OpenCL-Headers
            if [ $? -ne 0 ]; then
                printf "failed to download OpenCL-Headers to %s \n" "${OPENCL_SDK_PATH}"
                exit 1
            fi
        fi
        cd ${PROJECT_ROOT_PATH}/prebuilts/OpenCL_SDK/OpenCL-Headers
        printf "Copying OpenCL Headers to Android NDK sysroot include: ${NDK_TOOLCHAIN_SYSROOT_INCLUDE_PATH}"
        mkdir -p ${NDK_TOOLCHAIN_SYSROOT_INCLUDE_PATH}
        /bin/cp -r -fv CL ${NDK_TOOLCHAIN_SYSROOT_INCLUDE_PATH}

        cd ${PROJECT_ROOT_PATH}/prebuilts/OpenCL_SDK
        if [ ! -d OpenCL-ICD-Loader ]; then
            echo "Cloning OpenCL-ICD-Loader..."
            git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader
            if [ $? -ne 0 ]; then
                printf "failed to download OpenCL-ICD-Loader to %s \n" "${OPENCL_SDK_PATH}"
                exit 1
            fi
        fi
        cd ${PROJECT_ROOT_PATH}/prebuilts/OpenCL_SDK/OpenCL-ICD-Loader
        mkdir -p build
        cd build
        cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=latest -DANDROID_STL=c++_shared -DOPENCL_ICD_LOADER_HEADERS_DIR=${NDK_TOOLCHAIN_SYSROOT_INCLUDE_PATH}
        echo "Building OpenCL-ICD-Loader with ninjia..."
        ninja
        if [ $? -ne 0 ]; then
            printf "failed to build OpenCL-ICD-Loader\n"
            exit 1
        fi
        mkdir -p ${NDK_TOOLCHAIN_SYSROOT_ARM64_LIB_PATH}
        /bin/cp -fv libOpenCL.so ${NDK_TOOLCHAIN_SYSROOT_ARM64_LIB_PATH}

        echo "OpenCL components setup complete"
        echo "OpenCL Headers are in: ${NDK_TOOLCHAIN_SYSROOT_INCLUDE_PATH}/CL"
        echo "libOpenCL.so is in:    ${NDK_TOOLCHAIN_SYSROOT_ARM64_LIB_PATH}/libOpenCL.so"

        cd ${PROJECT_ROOT_PATH}
    else
        printf "OpenCL SDK already exist:    ${OPENCL_SDK_PATH} \n\n"
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
            printf "failed to download Android NDK to %s \n" "${ANDROID_NDK}"
            exit 1
        fi
        cd ${PROJECT_ROOT_PATH}

        printf "Android NDK saved to ${ANDROID_NDK} \n\n"
    else
        printf "Android NDK already exist:   ${ANDROID_NDK} \n\n"
    fi
}


function build_arm64
{
    cmake -H. -B./out/ggmlopencl-android -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=latest -DGGML_OPENCL=ON -DLLAMA_CURL=OFF -DGGML_LLAMAFILE=OFF -DHTP_ARCH_VERSION=${HTP_ARCH_VERSION}
    cd out/ggmlopencl-android
    make -j${HOST_CPU_COUNTS}
    show_pwd

    cd -
}


function build_arm64_debug
{
    cmake -H. -B./out/ggmlopencl-android -DCMAKE_BUILD_TYPE=Debug -DGGML_OPENMP=OFF -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=latest -DGGML_OPENCL=ON -DLLAMA_CURL=OFF -DGGML_LLAMAFILE=OFF -DHTP_ARCH_VERSION=${HTP_ARCH_VERSION}
    cd out/ggmlopencl-android
    make -j${HOST_CPU_COUNTS}
    show_pwd

    cd -
}


function remove_temp_dir()
{
    if [ -d out/ggmlopencl-android ]; then
        echo "remove out/ggmlopencl-android directory in `pwd`"
        rm -rf out/ggmlopencl-android
    fi
}


function build_ggml_opencl()
{
    show_pwd
    check_and_download_ndk
    check_and_download_opencl_sdk
    dump_vars
    remove_temp_dir
    build_arm64
}


function build_ggml_opencl_debug()
{
    show_pwd
    check_and_download_ndk
    check_and_download_opencl_sdk
    dump_vars
    remove_temp_dir
    build_arm64_debug
}


function check_and_download_model()
{
    set +e

    model_name=$1
    model_url=$2

    adb shell ls /sdcard/${model_name}
    if [ $? -eq 0 ]; then
        printf "the prebuild LLM model ${model_name} already exist on Android phone\n"
    else
        printf "the prebuild LLM model ${model_name} not exist on Android phone\n"
        wget --no-config --quiet --show-progress -O ${PROJECT_ROOT_PATH}/models/${model_name} ${model_url}
        adb push ${PROJECT_ROOT_PATH}/models/${model_name} /sdcard/
    fi

    set -e
}


function check_prebuilt_models()
{
    set +e

    check_and_download_model qwen1_5-1_8b-chat-q4_0.gguf https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat-GGUF/resolve/main/qwen1_5-1_8b-chat-q4_0.gguf
    check_and_download_model gemma-3n-E2B-it-Q8_0.gguf https://huggingface.co/ggml-org/gemma-3n-E2B-it-GGUF/resolve/main/gemma-3n-E2B-it-Q8_0.gguf

    set -e
}


function prepare_run_on_phone()
{
    if [ $# != 1 ]; then
        print "invalid param"
        return
    fi
    program=$1

    check_prebuilt_models

    if [ -f ./out/ggmlopencl-android/bin/libggml-cpu.so ]; then
        adb push ./out/ggmlopencl-android/bin/*.so ${REMOTE_PATH}/
    fi
    adb push ./out/ggmlopencl-android/bin/${program} ${REMOTE_PATH}/

    adb shell ls -l ${REMOTE_PATH}/libggml-*.so

    adb shell chmod +x ${REMOTE_PATH}/${program}
}


function run_llamacli()
{
    prepare_run_on_phone llama-cli

    echo "${REMOTE_PATH}/llama-cli ${running_params} -no-cnv -m ${TEST_MODEL_NAME} -p \"${PROMPT_STRING}\""
    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/llama-cli ${running_params} -no-cnv -m ${TEST_MODEL_NAME} -p \"${PROMPT_STRING}\""

}


function run_llamabench()
{
    prepare_run_on_phone llama-bench

    echo "adb shell \"cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/llama-bench ${running_params} -m ${GGUF_MODEL_NAME}\""
    echo "${REMOTE_PATH}/llama-bench ${running_params} -m ${GGUF_MODEL_NAME}"

    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/llama-bench ${running_params} -m ${GGUF_MODEL_NAME}"

}


function run_threadsafety()
{
    prepare_run_on_phone test-thread-safety

    echo "${REMOTE_PATH}/test-thread-safety -np 2 -m ${GGUF_MODEL_NAME} -p \"hello,world\" -n 256 -ngl 99 "
    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/test-thread-safety -np 1 -m ${GGUF_MODEL_NAME} -p \"hello,world\" -n 256 -ngl 99 "

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
               && ${REMOTE_PATH}/test-backend-ops test -o ${opname}"

    echo "\n"
    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/test-backend-ops test -o ${opname}"

}


function run_perf-op()
{
    prepare_run_on_phone test-backend-ops

    echo "adb shell cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/test-backend-ops perf -o ${opname}"

    echo "\n"
    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/test-backend-ops perf -o ${opname}"

}


function show_usage()
{
    echo -e "\n\n\n"
    echo "Usage:"
    echo "  $0 help"
    echo "  $0 build"
    echo "  $0 build_debug"
    echo "  $0 run_testops"
    echo "  $0 run_testop     ADD/MUL_MAT                                                    (verify accuracy    of ADD/MUL_MAT)"
    echo "  $0 run_perfop     ADD/MUL_MAT                                                    (verify performance of ADD/MUL_MAT)"
    echo "  $0 run_llamacli"
    echo "  $0 run_llamabench"
    echo "  $0 run_threadsafety"
    echo -e "\n\n\n"
}


######## part-4: entry point  ########

show_pwd

check_commands_in_host
check_android_phone
check_and_download_ndk
check_and_download_opencl_sdk
check_prebuilt_models

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
    elif [ "$1" == "build" ]; then
        build_ggml_opencl
        exit 0
    elif [ "$1" == "build_debug" ]; then
        build_ggml_opencl_debug
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
    elif [ "$1" == "run_threadsafety" ]; then
        run_threadsafety
        exit 0
    else
        show_usage
        exit 1
    fi
elif [ $# == 2 ]; then
    if [ "$1" == "run_testop" ]; then
        opname=$2
        run_test-op
        exit 0
    elif [ "$1" == "run_perfop" ]; then
        opname=$2
        run_perf-op
        exit 1
    fi
else
    show_usage
    exit 1
fi
