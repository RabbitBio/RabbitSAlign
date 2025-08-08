#ifndef RABBITSALIGN_GPU_ALIGNMENT_H
#define RABBITSALIGN_GPU_ALIGNMENT_H

#include "gpu_common.h"

// Alignment device functions
__device__ void my_hamming_align(const my_string &query, const my_string &ref, int match, int mismatch, int end_bonus, GPUAlignmentInfo& aln);
__device__ inline int gpu_hamming_distance(const my_string s, const my_string t);
__device__ bool gpu_extend_seed_part(GPUAlignTmpRes& align_tmp_res, const AlignmentParameters& aligner_parameters, const Nam& nam, const GPUReferences& references, const GPURead& read, bool consistent_nam);
__device__ bool gpu_rescue_mate_part(GPUAlignTmpRes& align_tmp_res, const AlignmentParameters& aligner_parameters, const Nam& nam, const GPUReferences& references, const GPURead& read, float mu, float sigma, int k);
__device__ bool gpu_reverse_nam_if_needed(Nam& nam, const GPURead& read, const GPUReferences& references, int k);
__device__ uint8_t gpu_get_mapq_seg(const my_vector<Nam>& nams, const Nam& n_max, const int* sorted_indices);
__device__ uint8_t gpu_get_mapq(const my_vector<Nam>& nams, const Nam& n_max);
__device__ float gpu_top_dropoff(my_vector<Nam>& nams);

// Functions to prepare data for alignment
__device__ void gpu_part2_extend_seed_get_str(GPUAlignTmpRes& align_tmp_res, int j, const GPURead& read1, const GPURead& read2, const GPUReferences& references, int* d_todo_cnt, char* d_query_ptr, char* d_ref_ptr, int* d_query_offset, int* d_ref_offset);
__device__ void gpu_part2_rescue_mate_get_str(GPUAlignTmpRes& align_tmp_res, int j, const GPURead& read1, const GPURead& read2, const GPUReferences& references, float mu, float sigma, int* d_todo_cnt, char* d_query_ptr, char* d_ref_ptr, int* d_query_offset, int* d_ref_offset);

#endif //RABBITSALIGN_GPU_ALIGNMENT_H