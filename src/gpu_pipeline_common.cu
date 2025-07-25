#include "gpu_pipeline_common.h"
#include <iostream>
#include <cuda_runtime.h>

// --- Definiciones de Variables Globales ---
std::once_flag init_flag_ref[GPU_NUM_MAX];
std::once_flag init_flag_pool[GPU_NUM_MAX];
GPUReferences *global_references[GPU_NUM_MAX];
RefRandstrobe *d_randstrobes[GPU_NUM_MAX];
my_bucket_index_t *d_randstrobe_start_indices[GPU_NUM_MAX];

GPUAlignTmpRes *g_chunk0_global_align_res[THREAD_NUM_MAX];
GPUAlignTmpRes *g_chunk1_global_align_res[THREAD_NUM_MAX];
GPUAlignTmpRes *g_chunk2_global_align_res[THREAD_NUM_MAX];
char *g_chunk0_global_align_res_data[THREAD_NUM_MAX];
char *g_chunk1_global_align_res_data[THREAD_NUM_MAX];
char *g_chunk2_global_align_res_data[THREAD_NUM_MAX];

// --- Temporizadores de GPU ---
thread_local double gpu_copy1 = 0;
thread_local double gpu_copy2 = 0;
thread_local double gpu_init1 = 0;
thread_local double gpu_init2 = 0;
thread_local double gpu_cost2_1 = 0;
thread_local double gpu_cost2_2 = 0;
thread_local double gpu_init3 = 0;
thread_local double gpu_init4 = 0;
thread_local double gpu_cost1 = 0;
thread_local double gpu_cost2 = 0;
thread_local double gpu_cost3 = 0;
thread_local double gpu_cost3_1 = 0;
thread_local double gpu_cost3_2 = 0;
thread_local double gpu_cost3_3 = 0;
thread_local double gpu_cost3_4 = 0;
thread_local double gpu_cost4 = 0;
thread_local double gpu_cost5 = 0;
thread_local double gpu_cost6 = 0;
thread_local double gpu_cost7 = 0;
thread_local double gpu_cost8 = 0;
thread_local double gpu_cost9 = 0;
thread_local double gpu_cost10 = 0;
thread_local double gpu_cost10_0 = 0;
thread_local double gpu_cost10_1 = 0;
thread_local double gpu_cost10_2 = 0;
thread_local double gpu_cost10_3 = 0;
thread_local double gpu_cost10_4 = 0;
thread_local double gpu_cost11 = 0;
thread_local double gpu_cost11_copy1 = 0;
thread_local double gpu_cost11_copy2 = 0;
thread_local double tot_cost = 0;


// --- Implementaciones de Funciones Comunes ---

ThreadContext::ThreadContext(int tid, int gpuid) {
    device_id = gpuid;
    cudaSetDevice(device_id);
    cudaStreamCreate(&stream);
}

ThreadContext::~ThreadContext() {
    cudaSetDevice(device_id);
    cudaStreamDestroy(stream);
}

klibpp::KSeq gpu_ConvertNeo2KSeq(neoReference ref) {
    klibpp::KSeq res;
    res.name = std::string((char *) ref.base + ref.pname, ref.lname);
    if (!res.name.empty()) {
        size_t space_pos = res.name.find(' ');
        int l_pos = 0;
        if (res.name[0] == '@') l_pos = 1;
        if (space_pos != std::string::npos) {
            res.name = res.name.substr(l_pos, space_pos - l_pos);
        } else {
            res.name = res.name.substr(l_pos);
        }
    }
    res.seq = std::string((char *) ref.base + ref.pseq, ref.lseq);
    res.comment = std::string((char *) ref.base + ref.pstrand, ref.lstrand);
    res.qual = std::string((char *) ref.base + ref.pqual, ref.lqual);
    return res;
}

__device__ bool gpu_is_proper_pair(const GPUAlignment& alignment1, const GPUAlignment& alignment2, float mu, float sigma) {
    const int dist = alignment2.ref_start - alignment1.ref_start;
    const bool same_reference = alignment1.ref_id == alignment2.ref_id;
    const bool both_aligned = same_reference && !alignment1.is_unaligned && !alignment2.is_unaligned;
    const bool r1_r2 = !alignment1.is_rc && alignment2.is_rc && dist >= 0; // r1 ---> <---- r2
    const bool r2_r1 = !alignment2.is_rc && alignment1.is_rc && dist <= 0; // r2 ---> <---- r1
    const bool rel_orientation_good = r1_r2 || r2_r1;
    const bool insert_good = std::abs(dist) <= mu + 6 * sigma;

    return both_aligned && insert_good && rel_orientation_good;
}

bool GPU_is_proper_pair(const std::pair<GPUAlignment, CigarData>& alignment1, const std::pair<GPUAlignment, CigarData>& alignment2, float mu, float sigma) {
    const int dist = alignment2.first.ref_start - alignment1.first.ref_start;
    const bool same_reference = alignment1.first.ref_id == alignment2.first.ref_id;
    const bool both_aligned = same_reference && !alignment1.first.is_unaligned && !alignment2.first.is_unaligned;
    const bool r1_r2 = !alignment1.first.is_rc && alignment2.first.is_rc && dist >= 0; // r1 ---> <---- r2
    const bool r2_r1 = !alignment2.first.is_rc && alignment1.first.is_rc && dist <= 0; // r2 ---> <---- r1
    const bool rel_orientation_good = r1_r2 || r2_r1;
    const bool insert_good = std::abs(dist) <= mu + 6 * sigma;

    return both_aligned && insert_good && rel_orientation_good;
}


void GPU_part2_extend_seed_store_res(
        GPUAlignTmpRes& align_tmp_res,
        int j,
        const neoRcRef &read1,
        const neoRcRef &read2,
        const References& references,
        const AlignmentInfo info
) {
    Nam nam = align_tmp_res.todo_nams[j];
    const neoRcRef &read = align_tmp_res.is_read1[j] ? read1 : read2;
    int result_ref_start;
    size_t query_size = read.read.lseq;
    const std::string& ref = references.sequences[nam.ref_id];

    const auto projected_ref_start = std::max(0, nam.ref_start - nam.query_start);
    const auto projected_ref_end = std::min(nam.ref_end + query_size - nam.query_end, ref.size());

    const int diff = std::abs(nam.ref_span() - nam.query_span());
    const int ext_left = std::min(50, projected_ref_start);
    const int ref_start = projected_ref_start - ext_left;
    const int ext_right = std::min(std::size_t(50), ref.size() - nam.ref_end);
    const auto ref_segm_size = query_size + diff + ext_left + ext_right;
    result_ref_start = ref_start + info.ref_start;
    int softclipped = info.query_start + (query_size - info.query_end);
    GPUAlignment& alignment = align_tmp_res.align_res[j];
    alignment.edit_distance = info.edit_distance;
    alignment.global_ed = info.edit_distance + softclipped;
    alignment.score = info.sw_score;
    alignment.ref_start = result_ref_start;
    alignment.length = info.ref_span();
    alignment.is_rc = nam.is_rc;
    alignment.is_unaligned = false;
    alignment.ref_id = nam.ref_id;
    alignment.gapped = true;

    align_tmp_res.cigar_info[j].has_realloc = 0;
    align_tmp_res.cigar_info[j].cigar = align_tmp_res.cigar_info[j].gpu_cigar;
    if (info.cigar.m_ops.size() + 1 > MAX_CIGAR_ITEM) {
        align_tmp_res.cigar_info[j].has_realloc = 1;
        uint32_t* tmp_cigar = (uint32_t*)malloc((info.cigar.m_ops.size() + 1) * sizeof(uint32_t));
        align_tmp_res.cigar_info[j].cigar = tmp_cigar;
    }
    align_tmp_res.cigar_info[j].cigar[0] = info.cigar.m_ops.size();
    for (int k = 0; k < info.cigar.m_ops.size(); k++) {
        align_tmp_res.cigar_info[j].cigar[k + 1] = info.cigar.m_ops[k];
    }
}

float GPU_normal_pdf(float x, float mu, float sigma) {
    static const float inv_sqrt_2pi = 0.3989422804014327;
    const float a = (x - mu) / sigma;

    return inv_sqrt_2pi / sigma * std::exp(-0.5f * a * a);
}

void unset_thread_affinity() {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    int num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    for (int i = 0; i < num_cpus; ++i) {
        CPU_SET(i, &cpuset);
    }
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

void set_thread_affinity(int cpu_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);

    pthread_t current_thread = pthread_self();
    if (pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) != 0) {
        std::cerr << "Error setting thread affinity to CPU " << cpu_id << std::endl;
    }
}


void init_shared_data(const References& references, const StrobemerIndex& index, const int gpu_id, int thread_id) {
    uint64_t ref_size = 0;
    uint64_t index_size = 0;
    cudaSetDevice(gpu_id);
    if (gpu_id == 0) printf("init_shared_data thread_id = %d, gpu_id = %d\n", thread_id, gpu_id);
    cudaMallocManaged(&global_references[gpu_id], sizeof(GPUReferences));
    global_references[gpu_id]->num_refs = references.size();
    cudaMalloc(&global_references[gpu_id]->sequences.data, references.size() * sizeof(my_string));
    ref_size += references.size() * sizeof(my_string);
    global_references[gpu_id]->sequences.length = references.size();
    global_references[gpu_id]->sequences.capacity = references.size();
    for (int i = 0; i < references.size(); i++) {
        my_string ref;
        ref.slen = references.lengths[i];
        cudaMalloc(&ref.data, references.lengths[i]);
        ref_size += references.lengths[i];
        cudaMemcpy(ref.data, references.sequences[i].data(), references.lengths[i], cudaMemcpyHostToDevice);
        cudaMemcpy(global_references[gpu_id]->sequences.data + i, &ref, sizeof(my_string), cudaMemcpyHostToDevice);
    }
    cudaMalloc(&global_references[gpu_id]->lengths.data, references.size() * sizeof(int));
    ref_size += references.size() * sizeof(int);
    cudaMemcpy(global_references[gpu_id]->lengths.data, references.lengths.data(), references.size() * sizeof(int), cudaMemcpyHostToDevice);
    global_references[gpu_id]->lengths.length = references.size();
    global_references[gpu_id]->lengths.capacity = references.size();

    cudaMalloc(&d_randstrobes[gpu_id], index.randstrobes.size() * sizeof(RefRandstrobe));
    index_size += index.randstrobes.size() * sizeof(RefRandstrobe);
    cudaMalloc(&d_randstrobe_start_indices[gpu_id], index.randstrobe_start_indices.size() * sizeof(my_bucket_index_t));
    index_size += index.randstrobe_start_indices.size() * sizeof(my_bucket_index_t);
    cudaMemset(d_randstrobes[gpu_id], 0, index.randstrobes.size() * sizeof(RefRandstrobe));
    cudaMemset(d_randstrobe_start_indices[gpu_id], 0, index.randstrobe_start_indices.size() * sizeof(my_bucket_index_t));
    cudaMemcpy(d_randstrobes[gpu_id], index.randstrobes.data(), index.randstrobes.size() * sizeof(RefRandstrobe), cudaMemcpyHostToDevice);
    cudaMemcpy(d_randstrobe_start_indices[gpu_id], index.randstrobe_start_indices.data(), index.randstrobe_start_indices.size() * sizeof(my_bucket_index_t), cudaMemcpyHostToDevice);

    if (gpu_id == 0) printf("--- ref GPU mem alloc %llu (%llu %llu)\n", ref_size + index_size, ref_size, index_size);
}

void init_mm_safe(uint64_t num_bytes, uint64_t seed, int gpu_id) {
    //cudaSetDevice(gpu_id);
    init_mm(num_bytes, seed);
    if (gpu_id == 0) printf("--- Gallatin GPU mem alloc %llu\n", num_bytes);
}

void init_global_big_data(int thread_id, int gpu_id, int max_tries, int batch_read_num) {
    cudaSetDevice(gpu_id);
    uint64_t pre_vec_size = 4 * sizeof(int) + 2 * sizeof(Nam) + sizeof(GPUAlignment) + sizeof(CigarData) + sizeof(TODOInfos);
    uint64_t global_align_res_data_size = batch_read_num * (max_tries * 2 + 2) * pre_vec_size;
    if (gpu_id == 0) printf("global_align_res_data_size -- %llu\n", global_align_res_data_size);

    cudaMallocManaged(&g_chunk0_global_align_res[thread_id], batch_read_num * 2 * sizeof(GPUAlignTmpRes));
    cudaMemset(g_chunk0_global_align_res[thread_id], 0, batch_read_num * 2 * sizeof(GPUAlignTmpRes));
    cudaMallocManaged(&g_chunk1_global_align_res[thread_id], batch_read_num * 2 * sizeof(GPUAlignTmpRes));
    cudaMemset(g_chunk1_global_align_res[thread_id], 0, batch_read_num * 2 * sizeof(GPUAlignTmpRes));
    cudaMallocManaged(&g_chunk2_global_align_res[thread_id], batch_read_num * 2 * sizeof(GPUAlignTmpRes));
    cudaMemset(g_chunk2_global_align_res[thread_id], 0, batch_read_num * 2 * sizeof(GPUAlignTmpRes));

    cudaMallocManaged(&g_chunk0_global_align_res_data[thread_id], global_align_res_data_size);
    cudaMemset(g_chunk0_global_align_res_data[thread_id], 0, global_align_res_data_size);
    cudaMallocManaged(&g_chunk1_global_align_res_data[thread_id], global_align_res_data_size);
    cudaMemset(g_chunk1_global_align_res_data[thread_id], 0, global_align_res_data_size);
    cudaMallocManaged(&g_chunk2_global_align_res_data[thread_id], global_align_res_data_size);
    cudaMemset(g_chunk2_global_align_res_data[thread_id], 0, global_align_res_data_size);

    if (gpu_id == 0) printf("--- align_res GPU mem alloc %llu\n", batch_read_num * 2 * sizeof(GPUAlignTmpRes) * 3
                                                                  + global_align_res_data_size * 3);
}