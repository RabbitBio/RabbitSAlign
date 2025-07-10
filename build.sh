#!/bin/bash


# Check if the CUDA path and GPU compute capability are provided
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <path-to-cuda-installation-directory> <GPU-compute-capability>"
  exit 1
fi

CUDA_PATH=$1
GPU_ARCH=$2

# Check if the provided CUDA installation directory exists
if [ ! -d "$CUDA_PATH" ]; then
  echo "CUDA installation directory $CUDA_PATH does not exist."
  exit 1
fi

# Configure and compile GASAL2 if libgasal.a does not exist
if [ ! -f "GASAL2/lib/libgasal.a" ]; then
  echo "Configuring and compiling GASAL2..."
  cd GASAL2 && ./configure.sh $CUDA_PATH && make GPU_SM_ARCH=sm_$GPU_ARCH MAX_QUERY_LEN=600 N_CODE=0x4E
  cd ..
else
  echo "Skipping GASAL2 compilation as libgasal.a already exists."
fi

# Configure and compile RabbitFX if librabbitfx.a does not exist
if [ ! -f "RabbitFX/build/lib/librabbitfx.a" ]; then
  echo "Configuring and compiling RabbitFX..."
  cd RabbitFX
  mkdir -p build
  cd build
  cmake .. -DCMAKE_INSTALL_PREFIX=./
  make -j8
  make install
  cd ../..
else
  echo "Skipping RabbitFX compilation as librabbitfx.a already exists."
fi

# Update CMakeLists.txt with the correct CUDA path
sed -i "s|/usr/local/cuda|$CUDA_PATH|g" CMakeLists.txt

# Compile the main project
echo "Compiling the main project..."
mkdir -p build
cd build
cmake .. -DENABLE_AVX=ON -DUSE_RABBITFX=ON -DCLOSE_NUMA_OPT=OFF -DCMAKE_BUILD_TYPE=release
make -j8

echo "build successfully."

