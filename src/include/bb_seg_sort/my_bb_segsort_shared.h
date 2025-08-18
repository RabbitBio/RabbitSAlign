/*
* (c) 2015 Virginia Polytechnic Institute & State University (Virginia Tech)
*
*   This program is free software: you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*   the Free Software Foundation, version 2.1
*
*   This program is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License, version 2.1, for more details.
*
*   You should have received a copy of the GNU General Public License
*
*/

#ifndef _H_MY_BB_SEGSORT_SHARED
#define _H_MY_BB_SEGSORT_SHARED

#include <iostream>
#include <vector>
#include <algorithm>

#include "bb_bin_shared.h"
#include "bb_comput_l.h"

#include "my_regsort_kernels.h"
#include "my_shared_merge_kernels.h"
#include "my_shared_merge_kernels_full_warps.h"


#define CUDA_CHECK(_e, _s) if(_e != cudaSuccess) { \
        std::cout << "CUDA error (" << _s << "): " << cudaGetErrorString(_e) << std::endl; \
        return 0; }


int set_up_my_bb_segsort_shared(int *&d_bin_segs_id, int *&d_bin_counter, int length) {
    cudaError_t cuda_err;

    cuda_err = cudaMalloc((void **)&d_bin_counter, SEGBIN_NUM_SHARED * sizeof(int));
    CUDA_CHECK(cuda_err, "alloc d_bin_counter");
    cuda_err = cudaMalloc((void **)&d_bin_segs_id, length * sizeof(int));
    CUDA_CHECK(cuda_err, "alloc d_bin_segs_id");

    cuda_err = cudaMemset(d_bin_counter, 0, SEGBIN_NUM_SHARED * sizeof(int));
    CUDA_CHECK(cuda_err, "memset d_bin_counter");

    return 1;
}

template<class K, class T>
int my_bb_segsort_shared(K *keys_d, T *vals_d, K *keysB_d, T *valsB_d, int n,  int *d_segs, int *d_bin_segs_id, int *d_bin_counter, int length)
{
    int *h_bin_counter = new int[SEGBIN_NUM_SHARED];

    bb_bin_shared(d_bin_segs_id, d_bin_counter, d_segs, length, n, h_bin_counter);

    cudaStream_t streams[SEGBIN_NUM_SHARED-1];
    for(int i = 0; i < SEGBIN_NUM_SHARED-1; i++) cudaStreamCreate(&streams[i]);

    int subwarp_size, subwarp_num, factor;
    dim3 blocks(256, 1, 1);
    dim3 grids(1, 1, 1);

    // combines 0 - 2 due to coarse segments
    blocks.x = 1024;                                     //threads per block
    subwarp_size = 2;                                   //threads for one segment
    subwarp_num = h_bin_counter[1]-h_bin_counter[0];    //num of segments
    factor = blocks.x/subwarp_size;                     //segments that can be processed per block
    grids.x = (subwarp_num+factor-1)/factor;            //total blocks needed
    if(subwarp_num > 0)
    my_t2_ppt1_orig<<<grids, blocks, 0, streams[0]>>>(keys_d, vals_d, keysB_d, valsB_d,
        n, d_segs, d_bin_segs_id+h_bin_counter[0], subwarp_num, length);

    // combines 3 - 16 due to coarse segments
    blocks.x = 256;
    subwarp_size = 16;
    subwarp_num = h_bin_counter[2]-h_bin_counter[1];
    factor = blocks.x/subwarp_size;
    grids.x = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    my_t16_ppt1_orig<<<grids, blocks, 0, streams[1]>>>(keys_d, vals_d, keysB_d, valsB_d,
        n, d_segs, d_bin_segs_id+h_bin_counter[1], subwarp_num, length);

    blocks.x = 256;
    subwarp_size = 16;
    subwarp_num = h_bin_counter[3]-h_bin_counter[2];
    factor = blocks.x/subwarp_size;
    grids.x = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    my_t16_ppt2_orig<<<grids, blocks, 0, streams[2]>>>(keys_d, vals_d, keysB_d, valsB_d,
        n, d_segs, d_bin_segs_id+h_bin_counter[2], subwarp_num, length);

    blocks.x = 64;
    subwarp_size = 16;
    subwarp_num = h_bin_counter[4]-h_bin_counter[3];
    factor = blocks.x/subwarp_size;
    grids.x = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    my_t16_ppt4_strd<<<grids, blocks, 0, streams[3]>>>(keys_d, vals_d, keysB_d, valsB_d,
        n, d_segs, d_bin_segs_id+h_bin_counter[3], subwarp_num, length);

    blocks.x = 512;
    subwarp_size = 16;
    subwarp_num = h_bin_counter[5]-h_bin_counter[4];
    factor = blocks.x/subwarp_size;
    grids.x = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    my_t16_ppt8_strd<<<grids, blocks, 0, streams[4]>>>(keys_d, vals_d, keysB_d, valsB_d,
        n, d_segs, d_bin_segs_id+h_bin_counter[4], subwarp_num, length);

    blocks.x = 64;
    subwarp_size = 16;
    subwarp_num = h_bin_counter[6]-h_bin_counter[5];
    factor = blocks.x/subwarp_size;
    grids.x = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    my_t16_ppt16_strd<<<grids, blocks, 0, streams[5]>>>(keys_d, vals_d, keysB_d, valsB_d,
        n, d_segs, d_bin_segs_id+h_bin_counter[5], subwarp_num, length);

    blocks.x = 128;
    subwarp_num = h_bin_counter[7]-h_bin_counter[6];
    grids.x = subwarp_num;
    if(subwarp_num > 0)
    my_blk128_ppt4_shared<<<grids, blocks, 0, streams[6]>>>(keys_d, vals_d, keysB_d, valsB_d,
        d_segs, d_bin_segs_id+h_bin_counter[6], subwarp_num);

    blocks.x = 256;
    subwarp_num = h_bin_counter[8]-h_bin_counter[7];
    grids.x = subwarp_num;
    if(subwarp_num > 0)
    my_blk256_ppt4_shared_strd_only_full_warps<<<grids, blocks, 0, streams[7]>>>(keys_d, vals_d, keysB_d, valsB_d,
        d_segs, d_bin_segs_id+h_bin_counter[7], subwarp_num);

    blocks.x = 512;
    subwarp_num = h_bin_counter[9]-h_bin_counter[8];
    grids.x = subwarp_num;
    if(subwarp_num > 0)
    my_blk512_ppt4_shared_strd_only_full_warps<<<grids, blocks, 0, streams[8]>>>(keys_d, vals_d, keysB_d, valsB_d,
        d_segs, d_bin_segs_id+h_bin_counter[8], subwarp_num);

    blocks.x = 512;
    subwarp_num = h_bin_counter[10]-h_bin_counter[9];
    grids.x = subwarp_num;
    if(subwarp_num > 0)
    my_blk512_ppt8_shared_strd_only_full_warps<<<grids, blocks, 0, streams[9]>>>(keys_d, vals_d, keysB_d, valsB_d,
        d_segs, d_bin_segs_id+h_bin_counter[9], subwarp_num);


    cudaError_t cuda_err;
    int maxbytes = 8192 * (sizeof(K) + sizeof(int));
    void (*fncPtr)(K *key, T *val, K *keyB, T *valB, int *segs, int *bin, int bin_size) {&my_blk1024_ppt8_shared_strd_only_full_warps};
    cuda_err = cudaFuncSetAttribute(fncPtr, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    CUDA_CHECK(cuda_err, "set shared memory for sizes >48kB");

    blocks.x = 1024;
    subwarp_num = h_bin_counter[11]-h_bin_counter[10];
    grids.x = subwarp_num;
    if(subwarp_num > 0)
    my_blk1024_ppt8_shared_strd_only_full_warps<<<grids, blocks, maxbytes, streams[10]>>>(keys_d, vals_d, keysB_d, valsB_d,
        d_segs, d_bin_segs_id+h_bin_counter[10], subwarp_num);

    // sort long segments
    subwarp_num = length-h_bin_counter[11];
    if(subwarp_num > 0)
    gen_grid_kern_r2049(keys_d, vals_d, keysB_d, valsB_d,
        n, d_segs, d_bin_segs_id+h_bin_counter[11], subwarp_num, length);


    for (int i = 0; i < SEGBIN_NUM_SHARED - 1; i++) cudaStreamDestroy(streams[i]);
    delete[] h_bin_counter;

    return 1;
}

#endif
