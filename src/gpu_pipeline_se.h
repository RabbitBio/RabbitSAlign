#ifndef RABBITSALIGN_GPU_PIPELINE_SE_H
#define RABBITSALIGN_GPU_PIPELINE_SE_H

#include "gpu_pipeline_common.h"

// --- Punto de Entrada Principal para el Pipeline de SE ---
void perform_task_async_se_fx_GPU(
        InputBuffer& input_buffer,
        OutputBuffer& output_buffer,
        AlignmentStatistics& statistics,
        int& done,
        const AlignmentParameters& aln_params,
        MappingParameters map_param,
        const IndexParameters& index_parameters,
        const References& references,
        const StrobemerIndex& index,
        const std::string& read_group_id,
        const int thread_id,
        rabbit::fq::FastqDataPool& fastqPool,
        rabbit::core::TDataQueue<rabbit::fq::FastqDataChunk> &dq,
        const bool use_good_numa,
        const int gpu_id,
        const int async_thread_id,
        const int batch_read_num,
        const int batch_total_read_len,
        const int chunk_num,
        const bool unordered_output
);

// --- Declaraciones de Kernels de SE ---
__global__ void gpu_pre_align_SE(
        int num_tasks,
        int s_len,
        IndexParameters *index_para,
        uint64_t *global_align_info,
        AlignmentParameters* aligner_parameters,
        int *pre_sum,
        int *lens,
        char *all_seqs,
        GPUReferences *global_references,
        MappingParameters *mapping_parameters,
        my_vector<Nam> *global_nams,
        int *global_todo_ids,
        GPUAlignTmpRes *global_align_res,
        int* d_todo_cnt,
        char* d_query_ptr, char* d_ref_ptr,
        int* d_query_offset, int* d_ref_offset
);

__global__ void gpu_align_SE(
        int num_tasks,
        int s_len,
        IndexParameters *index_para,
        uint64_t *global_align_info,
        AlignmentParameters* aligner_parameters,
        int *pre_sum,
        int *lens,
        char *all_seqs,
        GPUReferences *global_references,
        MappingParameters *mapping_parameters,
        my_vector<Nam> *global_nams,
        int *global_todo_ids,
        GPUAlignTmpRes *global_align_res,
        int* d_todo_cnt,
        char* d_query_ptr, char* d_ref_ptr,
        int* d_query_offset, int* d_ref_offset
);

#endif //RABBITSALIGN_GPU_PIPELINE_SE_H