#ifndef RABBITSALIGN_GPU_SEEDING_H
#define RABBITSALIGN_GPU_SEEDING_H

#include "gpu_common.h"

// Seeding related device functions
__device__ inline randstrobe_hash_t gpu_get_hash(const RefRandstrobe *d_randstrobes, size_t d_randstrobes_size, my_bucket_index_t position);
__device__ inline bool gpu_is_filtered(const RefRandstrobe *d_randstrobes, size_t d_randstrobes_size, my_bucket_index_t position, unsigned int filter_cutoff);
__device__ int gpu_get_count(const RefRandstrobe *d_randstrobes, const my_bucket_index_t *d_randstrobe_start_indices, my_bucket_index_t position, int bits);
__device__ inline size_t gpu_find(const RefRandstrobe *d_randstrobes, const my_bucket_index_t *d_randstrobe_start_indices, const randstrobe_hash_t key, int bits);
__device__ void add_to_hits_per_ref(my_vector<my_pair<int, Hit>>& hits_per_ref, int query_start, int query_end, size_t position, const RefRandstrobe *d_randstrobes, size_t d_randstrobes_size, int k);

// CUDA Kernels for seeding
__global__ void gpu_get_randstrobes(int num_tasks, int read_num, int base_read_num, int *pre_sum, int *lens, char *all_seqs, IndexParameters *index_para, int *randstrobe_sizes, uint64_t *hashes, my_vector<QueryRandstrobe>* global_randstrobes);
__global__ void gpu_get_hits_pre(int bits, unsigned int filter_cutoff, int rescue_cutoff, const RefRandstrobe *d_randstrobes, size_t d_randstrobes_size, const my_bucket_index_t *d_randstrobe_start_indices, int num_tasks, IndexParameters *index_para, uint64_t *global_hits_num, my_vector<QueryRandstrobe>* global_randstrobes, my_vector<my_pair<int, Hit>>* hits_per_ref0s, my_vector<my_pair<int, Hit>>* hits_per_ref1s);
__global__ void gpu_get_hits_after(int bits, unsigned int filter_cutoff, int rescue_cutoff, const RefRandstrobe *d_randstrobes, size_t d_randstrobes_size, const my_bucket_index_t *d_randstrobe_start_indices, int num_tasks, IndexParameters *index_para, uint64_t *global_hits_num, my_vector<QueryRandstrobe>* global_randstrobes, my_vector<my_pair<int, Hit>>* hits_per_ref0s, my_vector<my_pair<int, Hit>>* hits_per_ref1s);
__global__ void gpu_rescue_get_hits(int bits, unsigned int filter_cutoff, int rescue_cutoff, const RefRandstrobe *d_randstrobes, size_t d_randstrobes_size, const my_bucket_index_t *d_randstrobe_start_indices, int num_tasks, IndexParameters *index_para, uint64_t *global_hits_num, my_vector<QueryRandstrobe>* global_randstrobes, my_vector<my_pair<int, Hit>>* hits_per_ref0s, my_vector<my_pair<int, Hit>>* hits_per_ref1s, int* global_todo_ids, int rescue_threshold);

#endif //RABBITSALIGN_GPU_SEEDING_H