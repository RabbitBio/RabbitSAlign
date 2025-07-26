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
#define GPU_NUM_MAX 16

//#define use_device_mem

extern uint64_t STREAM_BATCH_SIZE;
extern uint64_t STREAM_BATCH_SIZE_GPU;
extern uint64_t MAX_QUERY_LEN;
extern uint64_t MAX_TARGET_LEN;

#define MAX_CPU_SSW_SIZE (STREAM_BATCH_SIZE * MAX_QUERY_LEN * MAX_TARGET_LEN / 2 * 1.01)
#define MAX_GPU_SSW_SIZE (STREAM_BATCH_SIZE_GPU * MAX_QUERY_LEN * MAX_TARGET_LEN / 2 * 1.02)

#define GB_BYTE (1ll << 30)

#define MAX_RABBITFX_CHUNK_NUM 128
#define BIG_CHUNK_NUM 32
#define SMALL_CHUNK_NUM 4

#ifdef use_device_mem
#define DEVICE_TODO_SIZE_PER_CHUNK (8 << 20)
#else
#define DEVICE_TODO_SIZE_PER_CHUNK (0)
#endif

#define PRINT_GPU_TIMER
#define PRINT_CPU_TIMER



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

void solve_ssw_on_gpu_pre_copy(int thread_id, std::vector<gasal_tmp_res> &gasal_results, std::vector<int> &todo_querys, std::vector<int> &todo_refs,
                       int match_score, int mismatch_score, int gap_open_score, int gap_extend_score, char* device_query_ptr, char* device_target_ptr);

void solve_ssw_on_gpu2(int thread_id, std::vector<gasal_tmp_res> &gasal_results, std::vector<std::string_view> &todo_querys, std::vector<std::string_view> &todo_refs,
                      int match_score = 2, int mismatch_score = 8, int gap_open_score = 12, int gap_extend_score = 1);

void solve_ssw_on_gpu(int thread_id, std::vector<gasal_tmp_res> &gasal_results, std::vector<std::string_view> &todo_querys, std::vector<std::string_view> &todo_refs,
                       int match_score = 2, int mismatch_score = 8, int gap_open_score = 12, int gap_extend_score = 1);

void solve_ssw_on_gpu(int thread_id, std::vector<gasal_tmp_res> &gasal_results, std::vector<std::string> &todo_querys, std::vector<std::string> &todo_refs,
                      int match_score = 2, int mismatch_score = 8, int gap_open_score = 12, int gap_extend_score = 1);
#endif  //STROBEALIGN_GASAL2_SSW_H
