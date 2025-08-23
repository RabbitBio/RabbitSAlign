#ifndef RABBITSALIGN_GPU_MERGING_H
#define RABBITSALIGN_GPU_MERGING_H

#include "gpu_common.h"

// Merging and sorting device functions
__device__ void sort_hits_single(my_vector<my_pair<int, Hit>>& hits_per_ref);
__device__ void merge_hits_seg(const my_vector<my_pair<int, Hit>>& original_hits, const int* sorted_indices, int task_start_offset, int task_end_offset, int k, bool is_revcomp, my_vector<Nam>& nams);
__device__ void merge_hits(my_vector<my_pair<int, Hit>>& hits_per_ref, int k, bool is_revcomp, my_vector<Nam>& nams);
__device__ void gpu_shuffle_top_nams(my_vector<Nam>& nams);
__device__ void sort_nams_by_score(my_vector<Nam>& nams, int mx_num);
__device__ void sort_nam_pairs_by_score(my_vector<gpu_NamPair>& joint_nam_scores, int mx_num);

// CUB-based sorting utilities
my_pair<int*, int*> sort_all_hits_with_cub(int todo_cnt, my_vector<my_pair<int, Hit>>* hits_per_refs, int* global_todo_ids, cudaStream_t stream, SegSortGpuResources& buffers,
                                           double *gpu_cost1 = nullptr, double *gpu_cost2 = nullptr, double *gpu_cost3 = nullptr, double *gpu_cost4 = nullptr);

my_pair<int*, int*> sort_all_hits_with_bb_segsort(int todo_cnt, my_vector<my_pair<int, Hit>>* hits_per_refs, int* global_todo_ids, cudaStream_t stream, SegSortGpuResources& buffers,
                                           double *gpu_cost1 = nullptr, double *gpu_cost2 = nullptr, double *gpu_cost3 = nullptr, double *gpu_cost4 = nullptr);

my_pair<int*, int*> sort_all_hits_with_cub_radix(int todo_cnt, my_vector<my_pair<int, Hit>>* hits_per_refs, int* global_todo_ids, cudaStream_t stream, SegSortGpuResources& buffers,
                                           double *gpu_cost1 = nullptr, double *gpu_cost2 = nullptr, double *gpu_cost3 = nullptr, double *gpu_cost4 = nullptr);

my_pair<int*, int*> sort_nams_by_score_with_cub(int todo_cnt, my_vector<Nam>* nams_per_task, int* global_todo_ids, cudaStream_t stream, SegSortGpuResources& buffers,
                                                double *gpu_cost1 = nullptr, double *gpu_cost2 = nullptr, double *gpu_cost3 = nullptr, double *gpu_cost4 = nullptr);

void sort_nams_by_score_in_place_with_cub(int todo_cnt, my_vector<Nam>* nams_per_task, int* global_todo_ids, cudaStream_t stream, SegSortGpuResources& buffers,
                                          double *gpu_cost1 = nullptr, double *gpu_cost2 = nullptr, double *gpu_cost3 = nullptr, double *gpu_cost4 = nullptr);

void sort_nams_by_score_in_place_with_cub_optimized(int todo_cnt, my_vector<Nam>* nams_per_task, int* global_todo_ids, cudaStream_t stream, SegSortGpuResources& buffers,
                                          double *gpu_cost1 = nullptr, double *gpu_cost2 = nullptr, double *gpu_cost3 = nullptr, double *gpu_cost4 = nullptr);

// Merging and sorting CUDA Kernels
__global__ void gpu_sort_hits(int num_tasks, my_vector<my_pair<int, Hit>>* hits_per_ref0s, my_vector<my_pair<int, Hit>>* hits_per_ref1s, int* global_todo_ids);
__global__ void gpu_rescue_sort_hits(int num_tasks, my_vector<my_pair<int, Hit>>* hits_per_ref0s, my_vector<my_pair<int, Hit>>* hits_per_ref1s, int* global_todo_ids);
__global__ void gpu_merge_hits_get_nams_seg(int num_tasks, IndexParameters *index_para, uint64_t *global_nams_info, my_vector<my_pair<int, Hit>>* hits_per_ref0s, my_vector<my_pair<int, Hit>>* hits_per_ref1s, const int* seg_offsets0, const int* sorted_indices0, const int* seg_offsets1, const int* sorted_indices1, my_vector<Nam> *global_nams, int* global_todo_ids);
__global__ void gpu_merge_hits_get_nams_1(int num_tasks, IndexParameters *index_para, uint64_t *global_nams_info, my_vector<my_pair<int, Hit>>* hits_per_ref0s, my_vector<my_pair<int, Hit>>* hits_per_ref1s, my_vector<int> *global_each_ref_size0, my_vector<int> *global_each_ref_size1, int* global_todo_ids);
__global__ void gpu_merge_hits_get_nams_2(int num_tasks, IndexParameters *index_para, bool is_revcomp, my_vector<my_pair<int, Hit>>* hits_per_ref0s,  my_vector<int> *global_each_ref_size0, int* each_ref_info0, my_vector<Nam> *global_nams_temp);
__global__ void gpu_merge_hits_get_nams_3(int num_tasks, IndexParameters *index_para, my_vector<my_pair<int, Hit>>* hits_per_ref0s, my_vector<my_pair<int, Hit>>* hits_per_ref1s, my_vector<int> *global_each_ref_size0, my_vector<int> *global_each_ref_size1, int* real_nams_range0, int* real_nams_range1, my_vector<Nam>* global_nams_temp0, my_vector<Nam>* global_nams_temp1, my_vector<Nam>* global_nams, int* global_todo_ids);
__global__ void gpu_merge_hits_get_nams(int num_tasks, IndexParameters *index_para, uint64_t *global_nams_info, my_vector<my_pair<int, Hit>>* hits_per_ref0s, my_vector<my_pair<int, Hit>>* hits_per_ref1s, my_vector<Nam> *global_nams, int* global_todo_ids);
__global__ void gpu_rescue_merge_hits_get_nams_seg(int num_tasks, IndexParameters *index_para, uint64_t *global_nams_info, my_vector<my_pair<int, Hit>>* hits_per_ref0s, my_vector<my_pair<int, Hit>>* hits_per_ref1s, const int* seg_offsets0, const int* sorted_indices0, const int* seg_offsets1, const int* sorted_indices1, my_vector<Nam> *global_nams, int* global_todo_ids);
__global__ void gpu_rescue_merge_hits_get_nams(int num_tasks, IndexParameters *index_para, uint64_t *global_nams_info, my_vector<my_pair<int, Hit>>* hits_per_ref0s, my_vector<my_pair<int, Hit>>* hits_per_ref1s, my_vector<Nam> *global_nams, int* global_todo_ids);
__global__ void gpu_sort_nams(int num_tasks, my_vector<Nam> *global_nams, MappingParameters *mapping_parameters, int is_se);

#endif //RABBITSALIGN_GPU_MERGING_H
