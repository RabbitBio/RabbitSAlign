#ifndef RABBITSALIGN_GPU_PIPELINE_COMMON_H
#define RABBITSALIGN_GPU_PIPELINE_COMMON_H

#include <mutex>
#include <thread>
#include <vector>
#include <string>
#include <random>
#include "gpu_common.h" // Asumiendo que esta cabecera contiene structs como neoRcRef, Nam, etc.

#define use_seg_sort


#define RESCUE_THRESHOLD 1000

#define SMALL_CHUNK_FAC 1

#define THREADS_PER_BLOCK 1

// --- Declaraciones de Variables Globales Externas ---
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

// --- Temporizadores de GPU (declarados como extern thread_local) ---
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
extern thread_local double gpu_cost5;
extern thread_local double gpu_cost6;
extern thread_local double gpu_cost6_1;
extern thread_local double gpu_cost6_2;
extern thread_local double gpu_cost6_3;
extern thread_local double gpu_cost6_4;
extern thread_local double gpu_cost7;
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

// --- Contexto del Hilo ---
struct ThreadContext {
    int device_id;
    cudaStream_t stream;

    ThreadContext(int tid, int gpuid);
    ~ThreadContext();
};

// --- Prototipos de Funciones de Utilidad (Host) ---
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

// --- Funciones de Inicialización ---
void init_shared_data(const References& references, const StrobemerIndex& index, const int gpu_id, int thread_id);
void init_mm_safe(uint64_t num_bytes, uint64_t seed, int gpu_id);
void init_global_big_data(int thread_id, int gpu_id, int max_tries, int batch_read_num);


// --- Prototipos de Funciones de Dispositivo ---
__device__ bool gpu_is_proper_pair(const GPUAlignment& alignment1, const GPUAlignment& alignment2, float mu, float sigma);

#endif //RABBITSALIGN_GPU_PIPELINE_COMMON_H
