#ifndef RABBITSALIGN_GPU_COMMON_H
#define RABBITSALIGN_GPU_COMMON_H

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

#define my_bucket_index_t StrobemerIndex::bucket_index_t

#define MAX_TRIES_LIMIT (d_map_param->max_tries * 2 + 2)


// Forward Declarations
struct Nam;

// Common Structs
struct GPURead {
    char* seq;
    char* rc;
    int length;
    __device__ int size() const { return length; }
};

struct gpu_NamPair {
    int score;
    my_vector<Nam> *nams1;
    my_vector<Nam> *nams2;
    int i1, i2;
};

struct ref_ids_edge {
    int pre;
    int ref_id;
};

struct GPUReferences {
    my_vector<my_string> sequences;
    my_vector<int> lengths;
    int num_refs;
};

struct GPUScoredAlignmentPair {
    double score;
    std::pair<GPUAlignment, CigarData> alignment1;
    std::pair<GPUAlignment, CigarData> alignment2;
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

struct GPUAlignTmpRes {
    int type;
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
    my_vector<GPUAlignment> align_res;
    my_vector<CigarData> cigar_info;
    my_vector<TODOInfos> todo_infos;
};

struct GPUInsertSizeDistribution {
    float sample_size = 1;
    float mu = 300;
    float sigma = 100;
    float V = 10000;
    float SSE = 10000;

    __device__ __host__ void update(int dist);
};

// Common Device Functions & Utilities
inline double GetTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double) tv.tv_sec + (double) tv.tv_usec / 1000000;
}

__device__ static inline syncmer_hash_t gpu_syncmer_kmer_hash(uint64_t packed) {
    return xxh64(packed);
}

__device__ static inline syncmer_hash_t gpu_syncmer_smer_hash(uint64_t packed) {
    return xxh64(packed);
}

__device__  static inline randstrobe_hash_t gpu_randstrobe_hash(syncmer_hash_t hash1, syncmer_hash_t hash2) {
    return hash1 + hash2;
}

template <typename T>
struct DefaultCompare {
    __device__ __forceinline__
    bool operator()(const T& a, const T& b) const {
        return a < b;
    }
};

template <typename T, typename Compare = DefaultCompare<T>>
__device__ void quick_sort_iterative(
        T* data,
        int low,
        int high,
        Compare comp = DefaultCompare<T>()
) {
    if (low > high) return;
    int vec_size = high - low + 1;
    my_vector<int>stack_vec(vec_size * 2);
    int* stack = stack_vec.data;
    int top = -1;
    stack[++top] = low;
    stack[++top] = high;
    while (top >= 0) {
        high = stack[top--];
        low = stack[top--];
        // Partition
        T pivot = data[high];
        int i = low - 1;
        for (int j = low; j < high; ++j) {
            //if (data[j] < pivot) {
            if (comp(data[j], pivot)) {
                ++i;
                T temp = data[i];
                data[i] = data[j];
                data[j] = temp;
            }
        }
        T temp = data[i + 1];
        data[i + 1] = data[high];
        data[high] = temp;
        int pivot_index = i + 1;
        if (pivot_index - 1 > low) {
            stack[++top] = low;
            stack[++top] = pivot_index - 1;
        }
        if (pivot_index + 1 < high) {
            stack[++top] = pivot_index + 1;
            stack[++top] = high;
        }
    }
}


template <typename T>
__device__ void quick_sort(T* data, int size) {
    quick_sort_iterative(data, 0, size - 1, DefaultCompare<T>());
}

__device__ void print_nam(Nam nam);
__device__ void print_str(my_string str);

#endif //RABBITSALIGN_GPU_COMMON_H