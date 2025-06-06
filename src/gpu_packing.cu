#include <cuda_runtime.h>
#include <cstdio>
#include "gpu_packing.h"

__global__ void PackBatchesKernel1(
    char* query_data_ptr,
    char* target_data_ptr,
    const QueryTargetMeta* d_meta,
    int total_seqs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_seqs) return;

    const char* qsrc = d_meta[i].query_ptr;
    int qlen = d_meta[i].query_len;
    uint32_t qoff = d_meta[i].query_offset;
    char* qdst = query_data_ptr + qoff;
    //for (int j = 0; j < qlen; j++) qdst[j] = qsrc[j];
    memcpy(qdst, qsrc, qlen);
    int qpad = ((qlen + 7) & ~7) - qlen;
    //for (int j = 0; j < qpad; j++) qdst[qlen + j] = 0x4E;
    memset(qdst + qlen, 0x4E, qpad);

    const char* tsrc = d_meta[i].target_ptr;
    int tlen = d_meta[i].target_len;
    uint32_t toff = d_meta[i].target_offset;
    char* tdst = target_data_ptr + toff;
    //for (int j = 0; j < tlen; j++) tdst[j] = tsrc[j];
    memcpy(tdst, tsrc, tlen);
    int tpad = ((tlen + 7) & ~7) - tlen;
    //for (int j = 0; j < tpad; j++) tdst[tlen + j] = 0x4E;
    memset(tdst + tlen, 0x4E, tpad);
}

__global__ void PackBatchesKernel(
    char* query_data_ptr,
    char* target_data_ptr,
    const char** query_seqs,
    const int* query_lens,
    const uint32_t* query_offsets,
    const char** target_seqs,
    const int* target_lens,
    const uint32_t* target_offsets,
    int total_seqs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_seqs) return;

    const char* qsrc = query_seqs[i];
    int qlen = query_lens[i];
    uint32_t qoff = query_offsets[i];
    char* qdst = query_data_ptr + qoff;
    //for (int j = 0; j < qlen; j++) qdst[j] = qsrc[j];
    memcpy(qdst, qsrc, qlen);
    int qpad = ((qlen + 7) & ~7) - qlen;
    //for (int j = 0; j < qpad; j++) qdst[qlen + j] = 0x4E;
    memset(qdst + qlen, 0x4E, qpad);

    const char* tsrc = target_seqs[i];
    int tlen = target_lens[i];
    uint32_t toff = target_offsets[i];
    char* tdst = target_data_ptr + toff;
    //for (int j = 0; j < tlen; j++) tdst[j] = tsrc[j];
    memcpy(tdst, tsrc, tlen);
    int tpad = ((tlen + 7) & ~7) - tlen;
    //for (int j = 0; j < tpad; j++) tdst[tlen + j] = 0x4E;
    memset(tdst + tlen, 0x4E, tpad);
}

void LaunchPackBatchesKernel1(
    char* d_query_data,
    char* d_target_data,
    const QueryTargetMeta* d_meta,
    int total_seqs)
{
    int threads = 32;
    int blocks = (total_seqs + threads - 1) / threads;
    PackBatchesKernel1<<<blocks, threads>>>(
        d_query_data, d_target_data,
        d_meta,
        total_seqs);
    cudaDeviceSynchronize();
}

void LaunchPackBatchesKernel(
    char* d_query_data,
    char* d_target_data,
    const char** d_query_seqs,
    const int* d_query_lens,
    const uint32_t* d_query_offsets,
    const char** d_target_seqs,
    const int* d_target_lens,
    const uint32_t* d_target_offsets,
    int total_seqs)
{
    int threads = 32;
    int blocks = (total_seqs + threads - 1) / threads;
    PackBatchesKernel<<<blocks, threads>>>(
        d_query_data, d_target_data,
        d_query_seqs, d_query_lens, d_query_offsets,
        d_target_seqs, d_target_lens, d_target_offsets,
        total_seqs);
    cudaDeviceSynchronize();
}

