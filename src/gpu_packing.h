#pragma once
#include <cstdint>

struct QueryTargetMeta {
    const char* query_ptr;
    int query_len;
    uint32_t query_offset;
    const char* target_ptr;
    int target_len;
    uint32_t target_offset;
};


void LaunchPackBatchesKernel(
    char* d_query_data,
    char* d_target_data,
    const char** d_query_seqs,
    const int* d_query_lens,
    const uint32_t* d_query_offsets,
    const char** d_target_seqs,
    const int* d_target_lens,
    const uint32_t* d_target_offsets,
    int total_seqs);

void LaunchPackBatchesKernel1(
    char* d_query_data,
    char* d_target_data,
    const QueryTargetMeta* d_meta,
    int total_seqs);

