//
// Created by ylf9811 on 2024/1/4.
//

#ifndef STROBEALIGN_GASAL2_SSW_H
#define STROBEALIGN_GASAL2_SSW_H
//#include "aligner.hpp"
#include "include/gasal_header.h"
#include <unistd.h>
#include <vector>
#include <cmath>
#include <sstream>
#include <cassert>

#define NB_STREAMS 1
#define THREAD_NUM_MAX 256

//#define STREAM_BATCH_SIZE (262144)
// this gives each stream HALF of the sequences.
//#define STREAM_BATCH_SIZE ceil((double)target_seqs.size() / (double)(2))

#define STREAM_BATCH_SIZE 512  //ceil((double)target_seqs.size() / (double)(2 * 2))

#define MAX_QUERY_LEN 500
#define MAX_TARGET_LEN 2000

#define DEBUG

#define MAX(a, b) (a > b ? a : b)

struct gasal_tmp_res{
    int score;
    int query_start;
    int query_end;
    int ref_start;
    int ref_end;
    std::string cigar_str;
};

struct gpu_batch {                     //a struct to hold data structures of a stream
    gasal_gpu_storage_t* gpu_storage;  //the struct that holds the GASAL2 data structures
    int n_seqs_batch;  //number of sequences in the batch (<= (target_seqs.size() / NB_STREAMS))
    int batch_start;   //starting index of batch
};

void solve_ssw_on_gpu(int thread_id, std::vector<gasal_tmp_res> &gasal_results, std::vector<std::string> &todo_querys, std::vector<std::string> &todo_refs,
                      int match_score = 2, int mismatch_score = 8, int gap_open_score = 12, int gap_extend_score = 1);
#endif  //STROBEALIGN_GASAL2_SSW_H
