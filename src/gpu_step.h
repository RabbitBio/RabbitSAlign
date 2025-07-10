#ifndef RABBITSALIGN_GPU_STEP_H
#define RABBITSALIGN_GPU_STEP_H

#define _GNU_SOURCE
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <sys/time.h>
#include <thread>
#include <omp.h>
#include <unistd.h>

#include "revcomp.hpp"
#include "robin_hood.h"
#include "sam.hpp"
#include "aln.hpp"
#include "fastq.hpp"
#include "kseq++/kseq++.hpp"
#include "index.hpp"
#include "indexparameters.hpp"
#include "cmdline.hpp"
#include "exceptions.hpp"
#include "io.hpp"
#include "randstrobes.hpp"
#include "refs.hpp"
#include "logger.hpp"
#include "pc.hpp"
#include "readlen.hpp"
#include "my_struct.hpp"
#include "hash.hpp"
#include "timer.hpp"


struct GPURead {
    char* seq;
    char* rc;
    int length;
    __device__ int size() const { return length; }
};


struct GPUAlignTmpRes {
    int type;
    // type 0 : size1 == 0 size2 == 0, unmapped_pair
    // type 1 : size1 == 0, rescue read1
    // type 2 : size2 == 0, rescue read2
    // type 3 : good pair
    // type 4 : for loop
    int mapq1;
    int mapq2;
    int type4_loop_size;
    int type3_isize_val;
    my_vector<int> is_extend_seed;
    my_vector<int> consistent_nam;
    my_vector<int> is_read1;
    my_vector<Nam> type4_nams;
    my_vector<Nam> todo_nams;
    my_vector<int> done_align;
    // if done_align, align_res is the alignment results
    my_vector<GPUAlignment> align_res;
    my_vector<CigarData> cigar_info;
    my_vector<TODOInfos> todo_infos;
};

struct GPUReferences {
    my_vector<my_string> sequences;
    my_vector<int> lengths;
    int num_refs;
};

struct GPUAlignmentInfo {
    my_vector<uint32_t> cigar;
    unsigned int edit_distance{0};
    unsigned int ref_start{0};
    unsigned int ref_end{0};
    unsigned int query_start{0};
    unsigned int query_end{0};
    int sw_score{0};

    __device__ int ref_span() const { return ref_end - ref_start; }
};

void perform_task_async_pe_fx_GPU(
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
    rabbit::core::TDataQueue<rabbit::fq::FastqDataPairChunk> &dq,
    const bool use_good_numa,
    const int gpu_id,
    const int async_thread_id,
    const int batch_read_num,
    const int batch_total_read_len,
    const int chunk_num
);

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
        const int chunk_num
);

void init_shared_data(const References& references, const StrobemerIndex& index, const int gpu_id, int thread_id);

void init_mm_safe(uint64_t num_bytes, uint64_t seed, int gpu_id);

void init_global_big_data(int thread_id, int gpu_id, int max_tries, int batch_read_num);

#endif  //RABBITSALIGN_GPU_STEP_H
