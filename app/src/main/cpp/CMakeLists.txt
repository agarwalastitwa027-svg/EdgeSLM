cmake_minimum_required(VERSION 3.22.1)
project(edgeslm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Vendored OpenCL headers/loader targeting Qualcomm Adreno GPUs (see docs/backend/OPENCL.md in llama.cpp).
set(OpenCL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/opencl/include" CACHE PATH "" FORCE)
set(OpenCL_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/opencl/lib/${ANDROID_ABI}/libOpenCL.so" CACHE FILEPATH "" FORCE)

# llama.cpp build options - GPU (OpenCL/Adreno) inference only, no CLI tools/tests/server to keep the build lean.
set(GGML_OPENCL ON CACHE BOOL "" FORCE)
set(GGML_OPENCL_USE_ADRENO_KERNELS ON CACHE BOOL "" FORCE)
set(GGML_OPENCL_EMBED_KERNELS ON CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE ON CACHE BOOL "" FORCE)

# Build llama.cpp/ggml as static libs folded into our single edgeslm_jni.so.
# (common.h/sampling.h don't carry GGML_API-style export macros, so as separate
# .so's their symbols end up hidden across the library boundary; statically
# linking everything into one .so sidesteps that entirely and ships fewer files.)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)

add_subdirectory(llama.cpp)

add_library(edgeslm_jni SHARED
    llama_jni.cpp
)

target_include_directories(edgeslm_jni PRIVATE
    llama.cpp/include
    llama.cpp/ggml/include
    llama.cpp/common
)

find_library(log-lib log)
find_library(android-lib android)

target_link_libraries(edgeslm_jni PRIVATE
    llama
    ggml
    llama-common
    ${log-lib}
    ${android-lib}
)

# Aggressive optimization for on-device inference: NEON/dotprod/i8mm on arm64.
# (No IPO/LTO here: this file is a thin JNI shim, and CMake's Android IPO support
# forces the removed gold linker on NDK r27 - not worth fighting for glue code.)
target_compile_options(edgeslm_jni PRIVATE -O3 -fvisibility=hidden)
if(ANDROID_ABI STREQUAL "arm64-v8a")
    target_compile_options(edgeslm_jni PRIVATE -march=armv8.2-a+dotprod+fp16)
endif()
