#pragma once
#include <cstdint>

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

