#!/usr/bin/env bash
#
#  Copyright (c) 2024-2025 The ggml authors
#  Copyright (c) zhouwg(https://github.com/zhouwg, co-author)
#
# build llama.cpp + ggml-vulkan backend on Linux for Android phone
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

#Vulkan-Headers can be found at:
#https://github.com/KhronosGroup/Vulkan-Headers
VULKAN_SDK_URL=https://github.com/KhronosGroup/Vulkan-Headers
VULKAN_SDK_PATH=${PROJECT_ROOT_PATH}/prebuilts/Vulkan_SDK
VULKAN_HEADERS_PATH=${VULKAN_SDK_PATH}/Vulkan-Headers

#running_params=" -ngl 99 -t 4 -n 256 --no-warmup -fa 1 "
running_params=" -ngl 99 -t 4 -n 256 --no-warmup "

######## part-2: contents in this part can be modified ########
#for llama-cli, 1.1 GiB, will be downloaded automatically via this script when running this script at the first time
TEST_MODEL_NAME=/sdcard/qwen1_5-1_8b-chat-q4_0.gguf
#for llama-cli, 4.5 GiB, will be downloadded automatically via this script when running this script at the first time
TEST_MODEL_NAME=/sdcard/gemma-3n-E2B-it-Q8_0.gguf

#for llama-bench, 4.5 GiB, will be downloadded automatically via this script when running this script at the first time
GGUF_MODEL_NAME=/sdcard/gemma-3n-E2B-it-Q8_0.gguf
#for llama-bench, 1.12 GiB, will be downloadded automatically via this script when running this script at the first time
GGUF_MODEL_NAME=/sdcard/qwen1_5-1_8b-chat-q4_0.gguf

######## part-3: utilities and functions ########

function dump_vars()
{
    echo -e "ANDROID_NDK:          ${ANDROID_NDK}"
    echo -e "VULKAN_SDK_PATH:      ${VULKAN_SDK_PATH}"
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
}


function check_android_phone()
{
    adb shell ls /bin/ls
    if [ ! $? -eq 0 ]; then
        printf "pls check Android phone is connected properly\n"
        exit 1
    fi
}


function check_and_download_vulkan_sdk()
{
    is_vulkan_sdk_exist=1

    if [ ! -d ${VULKAN_SDK_PATH} ]; then
        echo -e "VULKAN_SDK_PATH ${VULKAN_SDK_PATH} not exist, download it from ${VULKAN_SDK_URL}...\n"
        is_vulkan_sdk_exist=0
    fi

    if [ ${is_vulkan_sdk_exist} -eq 0 ]; then
        mkdir -p ${VULKAN_SDK_PATH}
        cd ${VULKAN_SDK_PATH}

        if [ ! -d Vulkan-Headers ]; then
            echo "Cloning Vulkan-Headers..."
            git clone https://github.com/KhronosGroup/Vulkan-Headers
            if [ $? -ne 0 ]; then
                printf "failed to download Vulkan-Headers to %s \n" "${VULKAN_SDK_PATH}"
                exit 1
            fi
        fi

        echo "Vulkan components setup complete"
        echo "Vulkan Headers are in: ${VULKAN_HEADERS_PATH}"

        cd ${PROJECT_ROOT_PATH}
    else
        printf "Vulkan SDK already exist:    ${VULKAN_SDK_PATH} \n\n"
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
        /bin/cp -fv ${ANDROID_NDK}/shader-tools/linux-x86_64/glslc ${ANDROID_NDK}/shader-tools/linux-x86_64/glsls
        cd ${PROJECT_ROOT_PATH}

        printf "Android NDK saved to ${ANDROID_NDK} \n\n"
    else
        printf "Android NDK already exist:   ${ANDROID_NDK} \n\n"
    fi
}


function build_arm64
{
    cmake -H. -B./out/ggmlvulkan-android -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=latest -DCMAKE_C_FLAGS=-march=armv8.7-a -DGGML_VULKAN=ON -DLLAMA_CURL=OFF -DGGML_LLAMAFILE=OFF -DVulkan_GLSLC_EXECUTABLE=${ANDROID_NDK}/shader-tools/linux-x86_64/glsls -DVulkan_INCLUDE_DIR=${VULKAN_HEADERS_PATH}/include
    cd out/ggmlvulkan-android
    make -j${HOST_CPU_COUNTS}
    show_pwd

    cd -
}


function build_arm64_debug
{
    cmake -H. -B./out/ggmlvulkan-android -DCMAKE_BUILD_TYPE=Debug -DGGML_OPENMP=OFF -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=latest -DCMAKE_C_FLAGS=-march=armv8.7-a -DGGML_VULKAN=ON -DGGML_VULKAN_DEBUG=ON -DLLAMA_CURL=OFF -DGGML_LLAMAFILE=OFF -DVulkan_GLSLC_EXECUTABLE=${ANDROID_NDK}/shader-tools/linux-x86_64/glsls -DVulkan_INCLUDE_DIR=${VULKAN_HEADERS_PATH}/include
    cd out/ggmlvulkan-android
    make -j${HOST_CPU_COUNTS}
    show_pwd

    cd -
}


function remove_temp_dir()
{
    if [ -d out/ggmlvulkan-android ]; then
        echo "remove out/ggmlvulkan-android directory in `pwd`"
        rm -rf out/ggmlvulkan-android
    fi
}


function build_ggml_vulkan()
{
    show_pwd
    check_and_download_ndk
    check_and_download_vulkan_sdk
    dump_vars
    remove_temp_dir
    build_arm64
}


function build_ggml_vulkan_debug()
{
    show_pwd
    check_and_download_ndk
    check_and_download_vulkan_sdk
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

    if [ -f ./out/ggmlvulkan-android/bin/libggml-cpu.so ]; then
        adb push ./out/ggmlvulkan-android/bin/*.so ${REMOTE_PATH}/
    fi
    adb push ./out/ggmlvulkan-android/bin/${program} ${REMOTE_PATH}/
    adb push ./out/ggmlvulkan-android/bin/vulkan-shaders-gen ${REMOTE_PATH}/

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
check_and_download_vulkan_sdk
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
        build_ggml_vulkan
        exit 0
    elif [ "$1" == "build_debug" ]; then
        build_ggml_vulkan_debug
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
