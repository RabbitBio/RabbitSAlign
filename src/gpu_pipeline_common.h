#ifndef RABBITSALIGN_GPU_PIPELINE_COMMON_H
#define RABBITSALIGN_GPU_PIPELINE_COMMON_H

#include <mutex>
#include <thread>
#include <vector>
#include <string>
#include <random>
#include "gpu_common.h" // Assuming this header contains structs like neoRcRef, Nam, etc.

//#define use_seg_sort

//#define use_fast_merge1
//#define use_fast_merge2

#define RESCUE_THRESHOLD 1000

#define SMALL_CHUNK_FAC 1

#define THREADS_PER_BLOCK 1
#define THREADS_PER_BLOCK2 128

// --- Global External Variables Declarations ---
extern std::once_flag init_flag_ref[GPU_NUM_MAX];
extern std::once_flag init_flag_pool[GPU_NUM_MAX];
extern GPUReferences *global_references[GPU_NUM_MAX];
extern RefRandstrobe *d_randstrobes[GPU_NUM_MAX];
extern my_bucket_index_t *d_randstrobe_start_indices[GPU_NUM_MAX];

extern GPUAlignTmpRes *g_chunk0_global_align_res[THREAD_NUM_MAX];
extern GPUAlignTmpRes *g_chunk1_global_align_res[THREAD_NUM_MAX];
extern GPUAlignTmpRes *g_chunk2_global_align_res[THREAD_NUM_MAX];
extern char *g_chunk0_global_align_res_data[THREAD_NUM_MAX];
extern char *g_chunk1_global_align_res_data[THREAD_NUM_MAX];
extern char *g_chunk2_global_align_res_data[THREAD_NUM_MAX];

// --- GPU Timers (declared as extern thread_local) ---
extern thread_local double gpu_copy1;
extern thread_local double gpu_copy2;
extern thread_local double gpu_init1;
extern thread_local double gpu_init2;
extern thread_local double gpu_init3;
extern thread_local double gpu_init4;
extern thread_local double gpu_cost1;
extern thread_local double gpu_cost2;
extern thread_local double gpu_cost2_1;
extern thread_local double gpu_cost2_2;
extern thread_local double gpu_cost3;
extern thread_local double gpu_cost3_1;
extern thread_local double gpu_cost3_2;
extern thread_local double gpu_cost3_3;
extern thread_local double gpu_cost3_4;
extern thread_local double gpu_cost4;
extern thread_local double gpu_cost4_1;
extern thread_local double gpu_cost4_2;
extern thread_local double gpu_cost4_3;
extern thread_local double gpu_cost4_4;
extern thread_local double gpu_cost4_5;
extern thread_local double gpu_cost4_6;
extern thread_local double gpu_cost4_7;
extern thread_local double gpu_cost5;
extern thread_local double gpu_cost6;
extern thread_local double gpu_cost6_1;
extern thread_local double gpu_cost6_2;
extern thread_local double gpu_cost6_3;
extern thread_local double gpu_cost6_4;
extern thread_local double gpu_cost7;
extern thread_local double gpu_cost7_1;
extern thread_local double gpu_cost7_2;
extern thread_local double gpu_cost7_3;
extern thread_local double gpu_cost7_4;
extern thread_local double gpu_cost7_5;
extern thread_local double gpu_cost7_6;
extern thread_local double gpu_cost7_7;
extern thread_local double gpu_cost8;
extern thread_local double gpu_cost8_1;
extern thread_local double gpu_cost8_2;
extern thread_local double gpu_cost8_3;
extern thread_local double gpu_cost8_4;
extern thread_local double gpu_cost9;
extern thread_local double gpu_cost10;
extern thread_local double gpu_cost10_1;
extern thread_local double gpu_cost10_2;
extern thread_local double gpu_cost10_3;
extern thread_local double gpu_cost10_4;
extern thread_local double gpu_cost11;
extern thread_local double gpu_cost11_copy1;
extern thread_local double gpu_cost11_copy2;
extern thread_local double tot_cost;

// --- Thread Context ---
struct ThreadContext {
    int device_id;
    cudaStream_t stream;

    ThreadContext(int tid, int gpuid);
    ~ThreadContext();
};

// --- Utility Function Prototypes (Host) ---
klibpp::KSeq gpu_ConvertNeo2KSeq(neoReference ref);

void GPU_part2_extend_seed_store_res(
        GPUAlignTmpRes& align_tmp_res,
        int j,
        const neoRcRef &read1,
        const neoRcRef &read2,
        const References& references,
        const AlignmentInfo info
);

bool GPU_is_proper_pair(const std::pair<GPUAlignment, CigarData>& alignment1, const std::pair<GPUAlignment, CigarData>& alignment2, float mu, float sigma);

float GPU_normal_pdf(float x, float mu, float sigma);

void set_thread_affinity(int cpu_id);
void unset_thread_affinity();

// --- Initialization Functions ---
void init_shared_data(const References& references, const StrobemerIndex& index, const int gpu_id, int thread_id);
void init_mm_safe(uint64_t num_bytes, uint64_t seed, int gpu_id);
void init_global_big_data(int thread_id, int gpu_id, int max_tries, int batch_read_num);


// --- Device Function Prototypes ---
__device__ bool gpu_is_proper_pair(const GPUAlignment& alignment1, const GPUAlignment& alignment2, float mu, float sigma);

void init_seg_sort_resources(SegSortGpuResources& resources, size_t initial_capacity, size_t max_todo_cnt, size_t initial_scan_temp_bytes, size_t initial_sort_temp_bytes, cudaStream_t stream);

void free_seg_sort_resources(SegSortGpuResources& resources, cudaStream_t stream);


#endif //RABBITSALIGN_GPU_PIPELINE_COMMON_H
