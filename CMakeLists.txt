cmake_minimum_required(VERSION 3.16)

project(RabbitSAlign VERSION 0.0.1)

find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
#find_package(MPI REQUIRED)


option(ENABLE_AVX "Enable AVX2 support" OFF)
option(USE_RABBITFX "Parse input FASTA with RabbitFX" OFF)
option(CLOSE_NUMA_OPT "Close NUMA optimization" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
enable_language(CUDA)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type" FORCE)
endif()

set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release" "Debug")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

add_compile_options(-w)

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fopenmp")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -O0")

set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 -fopenmp")
set(CMAKE_CUDA_FLAGS_RELEASE "-g -O3 -fopenmp")

if(CLOSE_NUMA_OPT)
    add_compile_definitions(OPT_NUMA_CLOSE)
endif()

if(USE_RABBITFX)
    add_compile_definitions(RABBIT_FX)
    include_directories(RabbitFX/build/include)
    link_directories(RabbitFX/build/lib)
endif()

include(FetchContent)

set(ZStr_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/ext/zstrgitrepo-src")


FetchContent_Declare(ZStrGitRepo
    SOURCE_DIR ${ZStr_SOURCE_DIR}
)

FetchContent_MakeAvailable(ZStrGitRepo)

configure_file(
        "${PROJECT_SOURCE_DIR}/src/version.hpp.in"
        "${PROJECT_BINARY_DIR}/version.hpp"
        @ONLY
)
configure_file(
        "${PROJECT_SOURCE_DIR}/src/buildconfig.hpp.in"
        "${PROJECT_BINARY_DIR}/buildconfig.hpp"
)

set(SOURCES
        src/main.cpp
        src/refs.cpp
        src/fastq.cpp
        src/cmdline.cpp
        src/index.cpp
        src/indexparameters.cpp
        src/sam.cpp
        src/paf.cpp
        src/pc.cpp
        src/aln.cpp
        src/cigar.cpp
        src/aligner.cpp
        src/nam.cpp
        src/randstrobes.cpp
        src/readlen.cpp
        src/version.cpp
        src/io.cpp
        ext/xxhash.c
        ext/ssw/ssw_cpp.cpp
        ext/ssw/ssw.c
        src/gasal2_ssw.cpp
        src/gasal2_ssw.h
        src/my_struct.cu
        src/gpu_step.cu
)

#include_directories(src/ src/include ext/ ${PROJECT_BINARY_DIR} ${MPI_C_INCLUDE_DIRS})
include_directories(src/ src/include ext/ ${PROJECT_BINARY_DIR})

add_executable(rabbitsalign ${SOURCES})

target_link_directories(rabbitsalign PUBLIC
        /usr/local/cuda/targets/x86_64-linux/lib
        ${PROJECT_SOURCE_DIR}/GASAL2/lib
)

target_link_libraries(rabbitsalign PUBLIC
        gasal
        cudart
        Threads::Threads
        OpenMP::OpenMP_CXX
        ZLIB::ZLIB
        zstr::zstr
        #${MPI_C_LIBRARIES}
)

if(USE_RABBITFX)
    target_link_libraries(rabbitsalign PUBLIC rabbitfx)
endif()

if(ENABLE_AVX)
    target_compile_options(rabbitsalign PUBLIC "-mavx2")
endif()

set_target_properties(rabbitsalign PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_ARCHITECTURES "86"
)
