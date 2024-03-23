# RabbitSAlign: GPU acceleration of short read alignment for heterogeneous multicore platforms

RabbitSAlign is a GPU-accelerated short-read aligner based on [strobealign](https://github.com/ksahlin/strobealign). It doubles the processing speed on real biological data by utilizing GPU to accelerate the extending phase and optimizing inefficient operations in the seeding process.

## Dependancy

- gcc 9.4.0 or newer
- nvcc 12.0 or newer

## Installation

```
git clone https://github.com/RabbitBio/RabbitSAlign
cd RabbitSAlign
bash build.sh <path-to-cuda-installation-directory> <GPU-compute-capability>
(eg: bash build.sh /usr/local/cuda 86)
```
The resulting binary is in `build/rabbitsalign`.

## Usage

Detailed usage can refer to the [strobealign](https://github.com/ksahlin/strobealign). RabbitSAlign can share the index file (.sti) with strobealign.

