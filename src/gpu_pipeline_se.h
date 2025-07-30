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

#endif //RABBITSALIGN_GPU_PIPELINE_SE_H