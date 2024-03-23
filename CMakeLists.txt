cmake_minimum_required(VERSION 3.16)

project(RabbitSAlign VERSION 0.0.1)
include(FetchContent)

option(ENABLE_AVX "Enable AVX2 support" OFF)
option(PYTHON_BINDINGS "Build Python bindings" OFF)
option(USE_RABBITFX "parse input fasta file with RabbitFX" OFF)
option(CLOSE_NUMA_OPT "close numa opt code" OFF)


find_package(ZLIB)
find_package(Threads)
find_package(OpenMP)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: RelWithDebInfo Debug Release" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "RelWithDebInfo" "Debug" "Release")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
add_compile_options(-w -Wall -Wextra -Werror=maybe-uninitialized)

FetchContent_Declare(ZStrGitRepo
#        GIT_REPOSITORY    "https://github.com/mateidavid/zstr"
        GIT_REPOSITORY    "git@github.com:mateidavid/zstr.git"
  GIT_TAG           "755da7890ea22478a702e3139092e6c964fab1f5"
)
FetchContent_MakeAvailable(ZStrGitRepo)

if(CLOSE_NUMA_OPT)
    set(CMAKE_CXX_FLAGS "-D OPT_NUMA_CLOSE ${CMAKE_CXX_FLAGS}")
endif()

if(USE_RABBITFX)
    set(CMAKE_CXX_FLAGS "-D RABBIT_FX ${CMAKE_CXX_FLAGS}")

    include_directories(RabbitFX/build/include)
    link_directories(RabbitFX/build/lib)
endif()



# Obtain version from Git or fall back to PROJECT_VERSION if not building
# from a Git repository
add_custom_target(version
  ${CMAKE_COMMAND}
  -D CONFIGIN="${PROJECT_SOURCE_DIR}/src/version.hpp.in"
  -D CONFIGOUT="${PROJECT_BINARY_DIR}/version.hpp"
  -D DEFAULT_VERSION="${PROJECT_VERSION}"
  -P ${CMAKE_SOURCE_DIR}/GitVersion.cmake
)

configure_file(
  "${PROJECT_SOURCE_DIR}/src/buildconfig.hpp.in"
  "${PROJECT_BINARY_DIR}/buildconfig.hpp"
)

add_library(salib STATIC ${SOURCES}
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
)

target_include_directories(salib PUBLIC src/ ext/ ${PROJECT_BINARY_DIR})

if(USE_RABBITFX)
    target_link_libraries(salib PUBLIC ZLIB::ZLIB Threads::Threads zstr::zstr rabbitfx)
else()
    target_link_libraries(salib PUBLIC ZLIB::ZLIB Threads::Threads zstr::zstr)
endif()

IF(ENABLE_AVX)
  target_compile_options(salib PUBLIC "-mavx2")
ENDIF()
add_dependencies(salib version)
set_target_properties(salib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(rabbitsalign src/main.cpp)
target_link_libraries(rabbitsalign PUBLIC salib)

set_target_properties(rabbitsalign PROPERTIES
        LINKER_LANGUAGE CUDA
        CUDA_STANDARD 14
)
target_link_libraries(rabbitsalign PUBLIC
        -L/usr/local/cuda/targets/x86_64-linux/lib
        -L${PROJECT_SOURCE_DIR}/GASAL2/lib
#        -Xcompiler
        -fopenmp -lcudart -lgasal
)


if(PYTHON_BINDINGS)
  add_subdirectory(src/python)
endif()
