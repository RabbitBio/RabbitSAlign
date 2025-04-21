#ifndef RABBITSALIGN_GPU_STEP_H
#define RABBITSALIGN_GPU_STEP_H


#include "index.hpp"
#include "kseq++/kseq++.hpp"
#include "revcomp.hpp"
#include "robin_hood.h"
#include "sam.hpp"
#include "timer.hpp"
#include "aln.hpp"
#include "refs.hpp"
#include "fastq.hpp"


void perform_task_async_pe_fx_GPU(
    InputBuffer& input_buffer,
    OutputBuffer& output_buffer,
    AlignmentStatistics& statistics,
    int& done,
    const AlignmentParameters& aln_params,
    const MappingParameters& map_param,
    const IndexParameters& index_parameters,
    const References& references,
    const StrobemerIndex& index,
    const std::string& read_group_id,
    const int thread_id,
    rabbit::fq::FastqDataPool& fastqPool,
    rabbit::core::TDataQueue<rabbit::fq::FastqDataPairChunk> &dq,
    bool use_good_numa,
    int gpu_id
);
#endif  //RABBITSALIGN_GPU_STEP_H
