#include "gpu_pipeline_se.h"
#include "gpu_seeding.h"
#include "gpu_merging.h"
#include "gpu_alignment.h"
#include <cuda_runtime.h>

void GPU_align_SE_read_last(
        GPUAlignTmpRes& align_tmp_res,
        const neoRcRef &data,
        Sam& sam,
        std::string& outstring,
        const Aligner& aligner,
        const MappingParameters& map_param,
        const IndexParameters& index_parameters,
        const References& references,
        const StrobemerIndex& index,
        std::minstd_rand& random_engine,
        double &t1,
        double &t2,
        double &t3,
        double &t4
) {
    Timer extend_timer;
    auto dropoff_threshold = map_param.dropoff_threshold;
    auto max_tries = map_param.max_tries;
    auto max_secondary = map_param.max_secondary;
    auto k = index_parameters.syncmer.k;
    Details details;
    double t0 = GetTime();
//    auto record = gpu_ConvertNeo2KSeq(data.read);
    if (align_tmp_res.type == 0) {
        sam.add_unmapped(data.read);
        t1 += GetTime() - t0;
        return;
    }

//    Read read(record.seq);
    std::vector<std::pair<GPUAlignment, CigarData>> alignments;
    int tries = 0;
    Nam n_max = align_tmp_res.todo_nams[0];

    int best_edit_distance = std::numeric_limits<int>::max();
    int best_score = 0;
    int second_best_score = 0;
    int alignments_with_best_score = 0;
    size_t best_index = 0;

    std::pair<GPUAlignment, CigarData> best_alignment;
    best_alignment.first.is_unaligned = true;
    t1 += GetTime() - t0;


    t0 = GetTime();
    for (int i = 0; i < align_tmp_res.todo_nams.size(); i++) {
        Nam nam = align_tmp_res.todo_nams[i];
        float score_dropoff = (float) nam.n_hits / n_max.n_hits;
        if (tries >= max_tries || (tries > 1 && best_edit_distance == 0) ||
            score_dropoff < dropoff_threshold) {
            for(int j = i; j < align_tmp_res.todo_nams.size(); j++) {
                if(!align_tmp_res.done_align[j]) {
                    aligner.m_align_calls--;
                }
            }
            break;
        }
        bool consistent_nam = align_tmp_res.consistent_nam[i];
        details.nam_inconsistent += !consistent_nam;
        //auto alignment = extend_seed(aligner, nam, references, read, consistent_nam);
        auto alignment = std::make_pair(align_tmp_res.align_res[i], align_tmp_res.cigar_info[i]);
        details.tried_alignment++;
        details.gapped += alignment.first.gapped;

        if (max_secondary > 0) {
            alignments.emplace_back(alignment);
        }

        if (alignment.first.score >= best_score) {
            second_best_score = best_score;
            bool update_best = false;
            if (alignment.first.score > best_score) {
                alignments_with_best_score = 1;
                update_best = true;
            } else {
                assert(alignment.first.score == best_score);
                // Two or more alignments have the same best score - count them
                alignments_with_best_score++;

#ifndef GPU_ACC_TAG
                // Pick one randomly using reservoir sampling
                std::uniform_int_distribution<> distrib(1, alignments_with_best_score);
                if (distrib(random_engine) == 1) {
                    update_best = true;
                }
#endif
            }
            if (update_best) {
                best_score = alignment.first.score;
                best_alignment = std::move(alignment);
                best_index = tries;
                if (max_secondary == 0) {
                    best_edit_distance = best_alignment.first.global_ed;
                }
            }
        } else if (alignment.first.score > second_best_score) {
            second_best_score = alignment.first.score;
        }
        tries++;
    }
    t2 += GetTime() - t0;

    t0 = GetTime();
    uint8_t mapq = (60.0 * (best_score - second_best_score) + best_score - 1) / best_score;
    bool is_primary = true;
    sam.add(best_alignment, data.read, data.rc, mapq, is_primary, details);

    if (max_secondary == 0) {
        t3 += GetTime() - t0;
        return;
    }

    // Secondary alignments

    // Remove the alignment that was already output
    if (alignments.size() > 1) {
        std::swap(alignments[best_index], alignments[alignments.size() - 1]);
    }
    alignments.resize(alignments.size() - 1);

    // Sort remaining alignments by score, highest first
    std::sort(alignments.begin(), alignments.end(), [](const std::pair<GPUAlignment, CigarData>& a, const std::pair<GPUAlignment, CigarData>& b) -> bool {
        return a.first.score > b.first.score;
    });
    t3 += GetTime() - t0;

    // Output secondary alignments
    t0 = GetTime();
    size_t n = 0;
    for (const auto& alignment : alignments) {
        if (n >= max_secondary ||
            alignment.first.score - best_score > 2 * aligner.parameters.mismatch + aligner.parameters.gap_open) {
            break;
        }
        bool is_primary = false;
        sam.add(alignment, data.read, data.rc, mapq, is_primary, details);
        n++;
    }
    t4 += GetTime() - t0;

}


__device__ void align_SE_part(
        GPUAlignTmpRes& align_tmp_res,
        const AlignmentParameters& aligner_parameters,
        my_vector<Nam>& nams,
        char* seq, char* rc, int seq_len,
        int k,
        const GPUReferences& references,
        float dropoff_threshold,
        int max_tries,
        size_t max_secondary,
        int* d_todo_cnt,
        char* d_query_ptr, char* d_ref_ptr,
        int* d_query_offset, int* d_ref_offset
) {
    assert(!nams.empty());

    GPURead read{seq, rc, seq_len};
    int tries = 0;
    Nam n_max = nams[0];
    assert(nams.size() <= max_tries);

    for (int i = 0; i < nams.size(); i++) {
        Nam nam = nams[i];
        float score_dropoff = (float) nam.n_hits / n_max.n_hits;
        if (tries >= max_tries || score_dropoff < dropoff_threshold) {
            break;
        }
        bool consistent_nam = gpu_reverse_nam_if_needed(nam, read, references, k);
        align_tmp_res.consistent_nam.push_back(consistent_nam);
        align_tmp_res.is_read1.push_back(true);
        bool gapped = gpu_extend_seed_part(align_tmp_res, aligner_parameters, nam, references, read, consistent_nam);
        tries++;
    }
    size_t todo_size = align_tmp_res.todo_nams.size();
    for (size_t j = 0; j < todo_size; j++) {
        if (!align_tmp_res.done_align[j]) {
            if (align_tmp_res.is_extend_seed[j]) {
                gpu_part2_extend_seed_get_str(
                        align_tmp_res, j, read, read, references,
                        d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset
                );
            } else {
                assert(0);
            }
        }
    }

    return;
}


__global__ void gpu_align_SE(
        int num_tasks,
        int read_num,
        int base_read_num,
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
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num_tasks) {
        int read_id = (id + base_read_num);
        int type = global_align_res[id].type;
        assert(type <= 4);
        if (type == 0) {
            global_nams[id].release();
        } else if (type == 4) {
            size_t seq_len;
            seq_len = lens[read_id];
            char *seq, *rc;
            seq = all_seqs + pre_sum[read_id];
            rc = all_seqs + pre_sum[read_id + read_num];

            GPUAlignTmpRes* align_tmp_res = &global_align_res[id];
            align_SE_part(*align_tmp_res, *aligner_parameters, global_nams[id],
                          seq, rc, seq_len, index_para->syncmer.k, *global_references,
                          mapping_parameters->dropoff_threshold, mapping_parameters->max_tries, mapping_parameters->max_secondary,
                          d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset);
            global_nams[id].release();
        } else {
            assert(false);
        }
    }

}

__global__ void gpu_pre_align_SE(
        int num_tasks,
        IndexParameters *index_para,
        uint64_t *global_align_info,
        AlignmentParameters* aligner_parameters,
        GPUReferences *global_references,
        MappingParameters *mapping_parameters,
        my_vector<Nam> *global_nams,
        int *global_todo_ids,
        GPUAlignTmpRes *global_align_res,
        int* d_todo_cnt,
        char* d_query_ptr, char* d_ref_ptr,
        int* d_query_offset, int* d_ref_offset
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num_tasks) {
        if (global_nams[id].empty()) global_todo_ids[id] = 0;
        else {
            int tries = 0;
            Nam n_max = global_nams[id][0];
            assert(global_nams[id].size() <= mapping_parameters->max_tries);
            for (int i = 0; i < global_nams[id].size(); i++) {
                Nam nam = global_nams[id][i];
                float score_dropoff = (float) nam.n_hits / n_max.n_hits;
                if (tries >= mapping_parameters->max_tries || score_dropoff < mapping_parameters->dropoff_threshold) {
                    break;
                }
                tries++;
            }
            global_todo_ids[id] = tries;
        }
    }
}

void GPU_align_SE(std::vector<neoRcRef> &datas,
                  ThreadContext& ctx, std::vector<AlignTmpRes> &align_tmp_results,
                  uint64_t* global_hits_num, uint64_t* global_nams_info, uint64_t* global_align_info,
                  const StrobemerIndex& index, AlignmentParameters *d_aligner, MappingParameters* d_map_param, IndexParameters *d_index_para,
                  GPUReferences *global_references, RefRandstrobe *d_randstrobes, my_bucket_index_t *d_randstrobe_start_indices,
                  my_vector<QueryRandstrobe> *global_randstrobes, int *global_todo_ids, int *global_randstrobe_sizes, uint64_t * global_hashes_value,
                  my_vector<my_pair<int, Hit>> *global_hits_per_ref0s, my_vector<my_pair<int, Hit>> *global_hits_per_ref1s, my_vector<Nam> *global_nams,
                  GPUAlignTmpRes *global_align_res, char *global_align_res_data, uint64_t pre_vec_size,
                  char *d_seq, int *d_len, int *d_pre_sum, char *h_seq, int *h_len, int *h_pre_sum,
                  int* d_todo_cnt, char* d_query_ptr, char* d_ref_ptr, int* d_query_offset, int* d_ref_offset, const int batch_read_num, const int batch_total_read_len, int rescue_threshold) {

    assert(datas.size() <= batch_read_num);

    double t0, t1;
    t0 = GetTime();

    // pack read on host
    t1 = GetTime();
    uint64_t tot_len = 0;
    h_pre_sum[0] = 0;
    int total_data_size = datas.size();
    for (int i = 0; i < total_data_size * 2; i++) {
        int read_id = i % total_data_size;
        if (i < total_data_size) { // read seq
            h_len[i] = datas[read_id].read.lseq;
            h_pre_sum[i + 1] = h_pre_sum[i] + h_len[i];
            char* seq_ptr = (char*)datas[read_id].read.base + datas[read_id].read.pseq;
            memcpy(h_seq + h_pre_sum[i], seq_ptr, h_len[i]);
        } else { // read rc
            h_len[i] = datas[read_id].read.lseq;
            h_pre_sum[i + 1] = h_pre_sum[i] + h_len[i];
            char* seq_ptr = datas[read_id].rc;
            memcpy(h_seq + h_pre_sum[i], seq_ptr, h_len[i]);
        }
    }
    tot_len = h_pre_sum[total_data_size * 2];
    assert(tot_len <= batch_total_read_len * 2);
    gpu_copy1 += GetTime() - t1;

    t1 = GetTime();
    cudaMemcpy(d_seq, h_seq, tot_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_len, h_len, total_data_size * sizeof(int) * 2 + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pre_sum, h_pre_sum, total_data_size * sizeof(int) * 2 + 1, cudaMemcpyHostToDevice);
    gpu_copy2 += GetTime() - t1;

    const int small_batch_size = (batch_read_num / SMALL_CHUNK_FAC > 0) ? (batch_read_num / SMALL_CHUNK_FAC) : 1;
    uint64_t global_data_offset = 0;

    for (int l_id = 0; l_id < total_data_size; l_id += small_batch_size) {
        int r_id = l_id + small_batch_size;
        if (r_id > total_data_size) r_id = total_data_size;
        int s_len = r_id - l_id; // Current small batch size
        if (s_len <= 0) continue;

        // get randstrobes
        t1 = GetTime();
        int blocks_per_grid = (s_len + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        gpu_get_randstrobes<<<blocks_per_grid, THREADS_PER_BLOCK, 0, ctx.stream>>>(s_len, total_data_size, l_id, d_pre_sum, d_len, d_seq, d_index_para,
                                                                                   global_randstrobe_sizes, global_hashes_value, global_randstrobes);
        cudaDeviceSynchronize();
        gpu_cost1 += GetTime() - t1;

        // query database and get hits
        t1 = GetTime();
        blocks_per_grid = (s_len + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        gpu_get_hits_pre<<<blocks_per_grid, THREADS_PER_BLOCK, 0, ctx.stream>>>(index.bits, index.filter_cutoff, d_map_param->rescue_cutoff, d_randstrobes, index.randstrobes.size(), d_randstrobe_start_indices,
                                                                                s_len, d_index_para, global_hits_num, global_randstrobes,
                                                                                global_hits_per_ref0s, global_hits_per_ref1s);
        cudaDeviceSynchronize();

        blocks_per_grid = (s_len + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        gpu_get_hits_after<<<blocks_per_grid, THREADS_PER_BLOCK, 0, ctx.stream>>>(index.bits, index.filter_cutoff, d_map_param->rescue_cutoff, d_randstrobes, index.randstrobes.size(), d_randstrobe_start_indices,
                                                                                  s_len, d_index_para, global_hits_num, global_randstrobes,
                                                                                  global_hits_per_ref0s, global_hits_per_ref1s);
        cudaDeviceSynchronize();
        gpu_cost2 += GetTime() - t1;

        // reads which pass the filter, normal mode
        t1 = GetTime();
        int todo_cnt = 0;
        for (int i = 0; i < s_len; i++) {
            if (global_randstrobes[i].data == nullptr) {
                global_todo_ids[todo_cnt] = i;
                todo_cnt++;
            }
        }
        gpu_init1 += GetTime() - t1;

        t1 = GetTime();
        blocks_per_grid = (todo_cnt + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        gpu_sort_hits<<<blocks_per_grid, THREADS_PER_BLOCK, 0, ctx.stream>>>(todo_cnt, global_hits_per_ref0s, global_hits_per_ref1s, global_todo_ids);
        cudaDeviceSynchronize();
        gpu_cost3 += GetTime() - t1;

        t1 = GetTime();
        blocks_per_grid = (todo_cnt + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        gpu_merge_hits_get_nams<<<blocks_per_grid, THREADS_PER_BLOCK, 0, ctx.stream>>>(todo_cnt, d_index_para, global_nams_info,
                                                                                       global_hits_per_ref0s, global_hits_per_ref1s,
                                                                                       global_nams, global_todo_ids);
        cudaDeviceSynchronize();
        gpu_cost4 += GetTime() - t1;

        t1 = GetTime();
        todo_cnt = 0;
        for (int i = 0; i < s_len; i++) {
            if (global_randstrobes[i].data != nullptr) {
                global_todo_ids[todo_cnt] = i;
                todo_cnt++;
            }
        }
        gpu_init2 += GetTime() - t1;

        t1 = GetTime();
        blocks_per_grid = (todo_cnt + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        gpu_rescue_get_hits<<<blocks_per_grid, THREADS_PER_BLOCK, 0, ctx.stream>>>(index.bits, index.filter_cutoff, d_map_param->rescue_cutoff, d_randstrobes, index.randstrobes.size(), d_randstrobe_start_indices,
                                                                                   todo_cnt, d_index_para, global_hits_num, global_randstrobes,
                                                                                   global_hits_per_ref0s, global_hits_per_ref1s, global_todo_ids, rescue_threshold);
        cudaDeviceSynchronize();
        gpu_cost5 += GetTime() - t1;

        t1 = GetTime();
        blocks_per_grid = (todo_cnt + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        gpu_rescue_sort_hits<<<blocks_per_grid, THREADS_PER_BLOCK, 0, ctx.stream>>>(todo_cnt, global_hits_per_ref0s, global_hits_per_ref1s, global_todo_ids);
        cudaDeviceSynchronize();
        gpu_cost6 += GetTime() - t1;

        t1 = GetTime();
        blocks_per_grid = (todo_cnt + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        gpu_rescue_merge_hits_get_nams<<<blocks_per_grid, THREADS_PER_BLOCK, 0, ctx.stream>>>(todo_cnt, d_index_para, global_nams_info,
                                                                                              global_hits_per_ref0s, global_hits_per_ref1s, global_nams, global_todo_ids);
        cudaDeviceSynchronize();
        gpu_cost7 += GetTime() - t1;

        t1 = GetTime();
        blocks_per_grid = (s_len + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        gpu_sort_nams<<<blocks_per_grid, THREADS_PER_BLOCK, 0, ctx.stream>>>(s_len, global_nams, d_map_param, 1);
        cudaDeviceSynchronize();
        gpu_cost8 += GetTime() - t1;

//        printf("s_len %d, todo_cnt %d, -- %.2f\n", s_len, todo_cnt, 1.0 * todo_cnt / s_len);

        for (int i = 0; i < s_len; i++) global_todo_ids[i] = -1;

        GPUAlignTmpRes* local_align_res_offset = global_align_res + l_id;

        blocks_per_grid = (s_len + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        gpu_pre_align_SE<<<blocks_per_grid, THREADS_PER_BLOCK, 0, ctx.stream>>>(s_len, d_index_para, global_align_info, d_aligner,
                                                                                global_references, d_map_param, global_nams, global_todo_ids, local_align_res_offset,
                                                                                d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset);
        cudaDeviceSynchronize();
        gpu_cost9 += GetTime() - t1;

        t1 = GetTime();
        char* base_ptr = global_align_res_data + global_data_offset;
        for (int i = 0; i < s_len; i++) {
            int type = 4;
            if (global_nams[i].length == 0) type = 0;
            if (global_todo_ids[i] == -1 || global_todo_ids[i] > d_map_param->max_tries) {
                printf("global_todo_ids[%d] %d, type %d\n", i, global_todo_ids[i], type);
                assert(false);
            }
            int tries_num = global_todo_ids[i] * 2;
            if (tries_num > MAX_TRIES_LIMIT) tries_num = MAX_TRIES_LIMIT;
            if (type == 0) tries_num = 0;

            // Use the offset pointer to configure the result structure for the current read.
            GPUAlignTmpRes *tmp = local_align_res_offset + i;
            tmp->type = type;
            tmp->mapq1 = 0, tmp->mapq2 = 0, tmp->type4_loop_size = 0;

            // Sub-allocate from the scratchpad buffer.
            tmp->is_extend_seed.data = (int*)base_ptr; tmp->is_extend_seed.length = 0; tmp->is_extend_seed.capacity = tries_num; base_ptr += tries_num * sizeof(int);
            tmp->consistent_nam.data = (int*)base_ptr; tmp->consistent_nam.length = 0; tmp->consistent_nam.capacity = tries_num; base_ptr += tries_num * sizeof(int);
            tmp->is_read1.data = (int*)base_ptr; tmp->is_read1.length = 0; tmp->is_read1.capacity = tries_num; base_ptr += tries_num * sizeof(int);
            tmp->type4_nams.data = (Nam*)base_ptr; tmp->type4_nams.length = 0; tmp->type4_nams.capacity = tries_num; base_ptr += tries_num * sizeof(Nam);
            tmp->todo_nams.data = (Nam*)base_ptr; tmp->todo_nams.length = 0; tmp->todo_nams.capacity = tries_num; base_ptr += tries_num * sizeof(Nam);
            tmp->done_align.data = (int*)base_ptr; tmp->done_align.length = 0; tmp->done_align.capacity = tries_num; base_ptr += tries_num * sizeof(int);
            tmp->align_res.data = (GPUAlignment*)base_ptr; tmp->align_res.length = 0; tmp->align_res.capacity = tries_num; base_ptr += tries_num * sizeof(GPUAlignment);
            tmp->cigar_info.data = (CigarData*)base_ptr; tmp->cigar_info.length = 0; tmp->cigar_info.capacity = tries_num; base_ptr += tries_num * sizeof(CigarData);
            tmp->todo_infos.data = (TODOInfos*)base_ptr; tmp->todo_infos.length = 0; tmp->todo_infos.capacity = tries_num; base_ptr += tries_num * sizeof(TODOInfos);
        }
        global_data_offset = base_ptr - global_align_res_data;
        gpu_init3 += GetTime() - t1;

        blocks_per_grid = (s_len + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        gpu_align_SE<<<blocks_per_grid, THREADS_PER_BLOCK, 0, ctx.stream>>>(s_len, total_data_size, l_id, d_index_para, global_align_info, d_aligner, d_pre_sum, d_len, d_seq,
                                                                            global_references, d_map_param, global_nams, global_todo_ids, local_align_res_offset,
                                                                            d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset);
        cudaDeviceSynchronize();
        gpu_cost10 += GetTime() - t1;
    }
    tot_cost += GetTime() - t0;
}


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
) {
    if(use_good_numa) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(thread_id, &cpuset);
        pthread_t current_thread = pthread_self();
        if (pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) != 0) {
            std::cerr << "Error setting thread affinity" << std::endl;
        }
    }

    ThreadContext ctx(thread_id, gpu_id);

    bool eof = false;
    Aligner aligner{aln_params};
    std::minstd_rand random_engine;
    size_t chunk_index = 0;
    std::vector<AlignTmpRes> align_tmp_results;
    double time_tot = 0;        // total time
    double time0 = 0;           // pre-allocations and initializations
    double time0_1 = 0;         //
    double time0_2 = 0;         // pre-allocations for para
    double time0_3 = 0;         // pre-allocations for global data
    double time0_4 = 0;         // pre-allocations for seq data
    double time0_4_1 = 0;         // pre-allocations for seq data
    double time0_4_2 = 0;         // pre-allocations for seq data
    double time0_4_3 = 0;         // pre-allocations for seq data
    double time0_4_4 = 0;         // pre-allocations for seq data
    double time1_1 = 0;         // format data
    double time1_1_1 = 0;       // rabbitfx format data
    double time1_1_2 = 0;       // trim data
    double time1_2 = 0;         // seeding on GPU
    double time1_3 = 0;         //
    double time2_1 = 0;         // construct todo_info from align_res
    double time2_1_1 = 0;       //
    double time2_2 = 0;         // ssw on GPU
    double time2_3 = 0;         // post-ssw on CPU
    double time2_3_1 = 0;       // fail type
    double time2_3_2 = 0;       // success type
    double time2_4 = 0;         // update align_res using ssw result
    double time3_1 = 0;         // format align_res to SAM item
    double time3_1_1 = 0;       // format time for type1
    double time3_1_2 = 0;       // format time for type2
    double time3_1_3 = 0;       // format time for type3
    double time3_1_4 = 0;       // format time for type4
    double time3_2 = 0;         // output SAM data
    double time3_3 = 0;         // release rabbitfx chunk
    double time3_4 = 0;         // clear vectors
    double time3_5 = 0;         // swap data
    double time4 = 0;           // free tmp data

    double t_0, t_1, t_2;


    t_0 = GetTime();

    t_1 = GetTime();


    t_2 = GetTime();
    AlignmentParameters *d_aligner;
    cudaMallocManaged(&d_aligner, sizeof(AlignmentParameters));
    cudaMemcpy(d_aligner, &aln_params, sizeof(AlignmentParameters), cudaMemcpyHostToDevice);
    MappingParameters* d_map_param;
    cudaMallocManaged(&d_map_param, sizeof(MappingParameters));
    cudaMemcpy(d_map_param, &map_param, sizeof(MappingParameters), cudaMemcpyHostToDevice);
    IndexParameters *d_index_para;
    cudaMallocManaged(&d_index_para, sizeof(IndexParameters));
    cudaMemcpy(d_index_para, &index_parameters, sizeof(IndexParameters), cudaMemcpyHostToDevice);
    time0_2 += GetTime() - t_2;


    t_2 = GetTime();
    uint64_t meta_data_size = 0;
    my_vector<QueryRandstrobe> *global_randstrobes;
    cudaMallocManaged(&global_randstrobes, batch_read_num / SMALL_CHUNK_FAC * sizeof(my_vector<QueryRandstrobe>));
    meta_data_size + batch_read_num * sizeof(my_vector<QueryRandstrobe>);
    cudaMemset(global_randstrobes, 0, batch_read_num * sizeof(my_vector<QueryRandstrobe>));
    int *global_todo_ids;
    cudaMallocManaged(&global_todo_ids, batch_read_num * sizeof(int));
    meta_data_size += batch_read_num * sizeof(int);
    cudaMemset(global_todo_ids, 0, batch_read_num * sizeof(int));
    int *global_randstrobe_sizes;
    cudaMallocManaged(&global_randstrobe_sizes, batch_read_num * sizeof(int));
    meta_data_size += batch_read_num * sizeof(int);
    cudaMemset(global_randstrobe_sizes, 0, batch_read_num * sizeof(int));
    uint64_t * global_hashes_value;
    cudaMallocManaged(&global_hashes_value, batch_read_num * sizeof(uint64_t));
    meta_data_size += batch_read_num * sizeof(uint64_t);
    cudaMemset(global_hashes_value, 0, batch_read_num * sizeof(uint64_t));
    my_vector<my_pair<int, Hit>> *global_hits_per_ref0s;
    cudaMallocManaged(&global_hits_per_ref0s, batch_read_num / SMALL_CHUNK_FAC * sizeof(my_vector<my_pair<int, Hit>>));
    meta_data_size += batch_read_num * sizeof(my_vector<my_pair<int, Hit>>);
    cudaMemset(global_hits_per_ref0s, 0, batch_read_num * sizeof(my_vector<my_pair<int, Hit>>));
    my_vector<my_pair<int, Hit>> *global_hits_per_ref1s;
    cudaMallocManaged(&global_hits_per_ref1s, batch_read_num / SMALL_CHUNK_FAC * sizeof(my_vector<my_pair<int, Hit>>));
    meta_data_size += batch_read_num * sizeof(my_vector<my_pair<int, Hit>>);
    cudaMemset(global_hits_per_ref1s, 0, batch_read_num * sizeof(my_vector<my_pair<int, Hit>>));
    my_vector<Nam> *global_nams;
    cudaMallocManaged(&global_nams, batch_read_num / SMALL_CHUNK_FAC * sizeof(my_vector<Nam>));
    meta_data_size += batch_read_num * sizeof(my_vector<Nam>);
    cudaMemset(global_nams, 0, batch_read_num * sizeof(my_vector<Nam>));

    uint64_t * global_hits_num;
    cudaMallocManaged(&global_hits_num, batch_read_num * sizeof(uint64_t));
    meta_data_size += batch_read_num * sizeof(uint64_t);
    cudaMemset(global_hits_num, 0, batch_read_num * sizeof(uint64_t));

    uint64_t * global_nams_info;
    cudaMallocManaged(&global_nams_info, batch_read_num * sizeof(uint64_t));
    meta_data_size += batch_read_num * sizeof(uint64_t);
    cudaMemset(global_nams_info, 0, batch_read_num * sizeof(uint64_t));

    uint64_t * global_align_info;
    cudaMallocManaged(&global_align_info, batch_read_num * sizeof(uint64_t));
    meta_data_size += batch_read_num * sizeof(uint64_t);
    cudaMemset(global_align_info, 0, batch_read_num * sizeof(uint64_t));

    int *chunk0_real_chunk_nums = new int[MAX_RABBITFX_CHUNK_NUM];
    int *chunk1_real_chunk_nums = new int[MAX_RABBITFX_CHUNK_NUM];
    int *chunk2_real_chunk_nums = new int[MAX_RABBITFX_CHUNK_NUM];

    int *chunk0_real_chunk_ids = new int[MAX_RABBITFX_CHUNK_NUM];
    int *chunk1_real_chunk_ids = new int[MAX_RABBITFX_CHUNK_NUM];
    int *chunk2_real_chunk_ids = new int[MAX_RABBITFX_CHUNK_NUM];

    if (gpu_id == 0) printf("--- meta GPU mem alloc %llu\n", meta_data_size);

    uint64_t pre_vec_size = 4 * sizeof(int) + 2 * sizeof(Nam) + sizeof(GPUAlignment) + sizeof(CigarData) + sizeof(TODOInfos);
    GPUAlignTmpRes *chunk0_global_align_res = g_chunk0_global_align_res[thread_id];
    GPUAlignTmpRes *chunk1_global_align_res = g_chunk1_global_align_res[thread_id];
    GPUAlignTmpRes *chunk2_global_align_res = g_chunk2_global_align_res[thread_id];
    char *chunk0_global_align_res_data = g_chunk0_global_align_res_data[thread_id];
    char *chunk1_global_align_res_data = g_chunk1_global_align_res_data[thread_id];
    char *chunk2_global_align_res_data = g_chunk2_global_align_res_data[thread_id];
    time0_3 += GetTime() - t_2;

    t_2 = GetTime();
    const int seq_size_alloc = batch_total_read_len;
    char *d_seq;
    int *d_len;
    int *d_pre_sum;
    cudaMalloc(&d_seq, seq_size_alloc * 2);
    cudaMemset(d_seq, 0, seq_size_alloc * 2);
    cudaMalloc(&d_len, (batch_read_num + 1) * sizeof(int) * 2);
    cudaMemset(d_len, 0, (batch_read_num + 1) * sizeof(int) * 2);
    cudaMalloc(&d_pre_sum, (batch_read_num + 1) * sizeof(int) * 2);
    cudaMemset(d_pre_sum, 0, (batch_read_num + 1) * sizeof(int) * 2);

    if (gpu_id == 0) printf("--- seq GPU mem alloc %llu\n", seq_size_alloc * 2 + (batch_read_num + 1) * sizeof(int) * 2 * 2);

    int *h_len;
    int *h_pre_sum;
    char *h_seq;
    cudaHostAlloc(&h_seq, seq_size_alloc * 2, cudaHostAllocDefault);
    cudaMemset(h_seq, 0, seq_size_alloc * 2);
    cudaHostAlloc(&h_len, (batch_read_num + 1) * sizeof(int) * 2, cudaHostAllocDefault);
    cudaMemset(h_len, 0, (batch_read_num + 1) * sizeof(int) * 2);
    cudaHostAlloc(&h_pre_sum, (batch_read_num + 1) * sizeof(int) * 2, cudaHostAllocDefault);
    cudaMemset(h_pre_sum, 0, (batch_read_num + 1) * sizeof(int) * 2);

    const int mx_device_query_size = chunk_num * DEVICE_TODO_SIZE_PER_CHUNK;
    const int mx_device_ref_size = mx_device_query_size * 2;

    char* device_query_ptr;
    char* device_ref_ptr;
#ifdef use_device_mem
    cudaMalloc(&device_query_ptr, mx_device_query_size);
    cudaMalloc(&device_ref_ptr, mx_device_ref_size);
#endif

    if (gpu_id == 0) printf("--- todo GPU mem alloc %llu\n", mx_device_query_size + mx_device_ref_size);

    int* d_todo_cnt;
    cudaMalloc(&d_todo_cnt, sizeof(int));
    int *h_todo_cnt;
    cudaHostAlloc(&h_todo_cnt, sizeof(int), cudaHostAllocDefault);
    int* d_query_offset;
    cudaMalloc(&d_query_offset, sizeof(int));
    int* h_query_offset;
    cudaHostAlloc(&h_query_offset, sizeof(int), cudaHostAllocDefault);
    int* d_ref_offset;
    cudaMalloc(&d_ref_offset, sizeof(int));
    int* h_ref_offset;
    cudaHostAlloc(&h_ref_offset, sizeof(int), cudaHostAllocDefault);

    GPUInsertSizeDistribution* isize_est;
    cudaMallocManaged(&isize_est, sizeof(GPUInsertSizeDistribution));
    isize_est->sample_size = 1;
    isize_est->mu = 300;
    isize_est->sigma = 100;
    isize_est->V = 10000;
    isize_est->SSE = 10000;

    std::vector<gasal_tmp_res> gasal_results_tmp;
    std::vector<neoReference> neo_datas;
    std::vector<AlignmentInfo> info_results;
#ifdef use_device_mem
    std::vector<int> todo_querys;
    std::vector<int> todo_refs;
    todo_querys.reserve(batch_read_num * 2);
    todo_refs.reserve(batch_read_num * 2);
#endif

    char* chunk0_rc_data = new char[batch_total_read_len];
    rabbit::fq::FastqDataChunk *chunk0_fqdatachunks[MAX_RABBITFX_CHUNK_NUM];
    int chunk0_real_chunk_num;
    int chunk0_chunk_num;
    std::vector<neoRcRef> chunk0_datas;
    std::vector<std::string_view> chunk0_h_todo_querys;
    std::vector<std::string_view> chunk0_h_todo_refs;
    std::vector<gasal_tmp_res> chunk0_gasal_results;
    chunk0_h_todo_querys.reserve(batch_read_num * 2);
    chunk0_h_todo_refs.reserve(batch_read_num * 2);
    chunk0_gasal_results.reserve(batch_read_num * 2);


    char* chunk1_rc_data = new char[batch_total_read_len];
    rabbit::fq::FastqDataChunk *chunk1_fqdatachunks[MAX_RABBITFX_CHUNK_NUM];
    int chunk1_real_chunk_num;
    int chunk1_chunk_num;
    std::vector<neoRcRef> chunk1_datas;
    std::vector<std::string_view> chunk1_h_todo_querys;
    std::vector<std::string_view> chunk1_h_todo_refs;
    std::vector<gasal_tmp_res> chunk1_gasal_results;
    chunk1_h_todo_querys.reserve(batch_read_num * 2);
    chunk1_h_todo_refs.reserve(batch_read_num * 2);
    chunk1_gasal_results.reserve(batch_read_num * 2);


    char* chunk2_rc_data = new char[batch_total_read_len];
    rabbit::fq::FastqDataChunk *chunk2_fqdatachunks[MAX_RABBITFX_CHUNK_NUM];
    int chunk2_real_chunk_num;
    int chunk2_chunk_num;
    std::vector<neoRcRef> chunk2_datas;
    std::vector<std::string_view> chunk2_h_todo_querys;
    std::vector<std::string_view> chunk2_h_todo_refs;
    std::vector<gasal_tmp_res> chunk2_gasal_results;
    chunk2_h_todo_querys.reserve(batch_read_num * 2);
    chunk2_h_todo_refs.reserve(batch_read_num * 2);
    chunk2_gasal_results.reserve(batch_read_num * 2);

    rabbit::int64 id;
    int total_ssw = 0;
    int gpu_ssw = 0;

    time0_4 += GetTime() - t_2;

    time0 += GetTime() - t_1;

    const int small_chunk_num = chunk_num / 2;

    int read_len = 150;

    // step: f_0
    {
        bool res;
        // format data
        t_1 = GetTime();
        int rc_pos = 0;
        chunk0_chunk_num = rand() % small_chunk_num + small_chunk_num + 1;
        chunk0_real_chunk_num = 0;
        chunk0_datas.clear();
        for (int chunk_id = 0; chunk_id < chunk0_chunk_num; chunk_id++) {
            res = dq.Pop(id, chunk0_fqdatachunks[chunk_id]);
            if (res) {
                double t_3 = GetTime();
                neo_datas.clear();
                rabbit::fq::chunkFormat(chunk0_fqdatachunks[chunk_id], neo_datas);
                time1_1_1 += GetTime() - t_3;

                t_3 = GetTime();
                for (int i = 0; i < neo_datas.size(); i++) {
                    char *name = (char *) neo_datas[i].base + neo_datas[i].pname;
                    if (neo_datas[i].lname > 0 && name[0] == '@') {
                        neo_datas[i].pname++;
                        neo_datas[i].lname--;
                        name++;
                    }
                    for (int j = 0; j < neo_datas[i].lname; j++) {
                        if (name[j] == ' ') {
                            neo_datas[i].lname = j;
                            break;
                        }
                    }

                    read_len = std::max(read_len, (int)neo_datas[i].lseq);
                    char *seq = (char *) neo_datas[i].base + neo_datas[i].pseq;
                    chunk0_datas.push_back({neo_datas[i], chunk0_rc_data + rc_pos});
                    for (int j = 0; j < neo_datas[i].lseq; j++) {
                        chunk0_rc_data[rc_pos++] = rc_gpu_nt2nt[seq[neo_datas[i].lseq - 1 - j]];
                    }
                }
                time1_1_2 += GetTime() - t_3;
                chunk0_real_chunk_nums[chunk_id] = neo_datas.size();
                chunk0_real_chunk_ids[chunk_id] = id;
                chunk0_real_chunk_num++;
            } else break;
        }
        assert(rc_pos <= batch_total_read_len);
        time1_1 += GetTime() - t_1;
    }

    //int rescue_threshold = read_len;
    int rescue_threshold = RESCUE_THRESHOLD;
    printf("rescue_threshold %d\n", rescue_threshold);

    // step: f_1
    {
        bool res;
        // format data
        t_1 = GetTime();
        int rc_pos = 0;
        chunk1_chunk_num = rand() % small_chunk_num + small_chunk_num + 1;
        chunk1_real_chunk_num = 0;
        chunk1_datas.clear();
        for (int chunk_id = 0; chunk_id < chunk1_chunk_num; chunk_id++) {
            res = dq.Pop(id, chunk1_fqdatachunks[chunk_id]);
            if (res) {
                double t_3 = GetTime();
                neo_datas.clear();
                rabbit::fq::chunkFormat(chunk1_fqdatachunks[chunk_id], neo_datas);
                time1_1_1 += GetTime() - t_3;

                t_3 = GetTime();
                for (int i = 0; i < neo_datas.size(); i++) {
                    char *name = (char *) neo_datas[i].base + neo_datas[i].pname;
                    if (neo_datas[i].lname > 0 && name[0] == '@') {
                        neo_datas[i].pname++;
                        neo_datas[i].lname--;
                        name++;
                    }
                    for (int j = 0; j < neo_datas[i].lname; j++) {
                        if (name[j] == ' ') {
                            neo_datas[i].lname = j;
                            break;
                        }
                    }
                    char *seq = (char *) neo_datas[i].base + neo_datas[i].pseq;
                    chunk1_datas.push_back({neo_datas[i], chunk1_rc_data + rc_pos});
                    for (int j = 0; j < neo_datas[i].lseq; j++) {
                        chunk1_rc_data[rc_pos++] = rc_gpu_nt2nt[seq[neo_datas[i].lseq - 1 - j]];
                    }
                }
                time1_1_2 += GetTime() - t_3;
                chunk1_real_chunk_nums[chunk_id] = neo_datas.size();
                chunk1_real_chunk_ids[chunk_id] = id;
                chunk1_real_chunk_num++;
            } else break;
        }
        assert(rc_pos <= batch_total_read_len);
        time1_1 += GetTime() - t_1;
    }


    // step: s+e_0
    {
        // seeding on GPU
        t_1 = GetTime();
        chunk_index = id;
        random_engine.seed(chunk_index);
        cudaMemset(d_todo_cnt, 0, sizeof(int));
        cudaMemset(d_query_offset, 0, sizeof(int));
        cudaMemset(d_ref_offset, 0, sizeof(int));
        if (!chunk0_datas.empty()) GPU_align_SE(chunk0_datas,
                                                ctx,
                                                align_tmp_results,
                                                global_hits_num, global_nams_info, global_align_info,
                                                index, d_aligner, d_map_param, d_index_para,
                                                global_references[gpu_id], d_randstrobes[gpu_id], d_randstrobe_start_indices[gpu_id],
                                                global_randstrobes, global_todo_ids, global_randstrobe_sizes, global_hashes_value,
                                                global_hits_per_ref0s, global_hits_per_ref1s, global_nams,
                                                chunk0_global_align_res, chunk0_global_align_res_data, pre_vec_size,
                                                d_seq, d_len, d_pre_sum, h_seq, h_len, h_pre_sum,
                                                d_todo_cnt, device_query_ptr, device_ref_ptr, d_query_offset, d_ref_offset, batch_read_num, batch_total_read_len, rescue_threshold);
        cudaMemcpy(h_todo_cnt, d_todo_cnt, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_query_offset, d_query_offset, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_ref_offset, d_ref_offset, sizeof(int), cudaMemcpyDeviceToHost);
        assert(*h_query_offset <= mx_device_query_size);
        assert(*h_ref_offset <= mx_device_ref_size);
        time1_2 += GetTime() - t_1;

        // construct todo_info from align_res
        t_1 = GetTime();
#ifdef use_device_mem
        todo_querys.resize(*h_todo_cnt);
        todo_refs.resize(*h_todo_cnt);
        chunk0_h_todo_querys.resize(*h_todo_cnt);
        chunk0_h_todo_refs.resize(*h_todo_cnt);
#else
        chunk0_h_todo_querys.clear();
        chunk0_h_todo_refs.clear();
#endif
        int cal_todo_cnt = 0;
        for (int i = 0; i < chunk0_datas.size(); i++) {
            GPUAlignTmpRes &align_tmp_res = chunk0_global_align_res[i];
            for (int j = 0; j < align_tmp_res.todo_infos.size(); j++) {
                TODOInfos& todo_info = align_tmp_res.todo_infos[j];
                int global_id = todo_info.global_id;
                uint32_t info = todo_info.read_info;
                int is_read1 = (info >> 31) & 0x1;
                int is_rc    = (info >> 30) & 0x1;
                int q_begin  = (info >> 15) & 0x7FFF;
                int q_len    = info & 0x7FFF;
                const auto& h_query_seq = is_rc ? chunk0_datas[i].rc : (char*)chunk0_datas[i].read.base + chunk0_datas[i].read.pseq;
                const auto& h_ref_seq = references.sequences[todo_info.ref_id];
#ifdef use_device_mem
                todo_querys[global_id] = q_len;
                todo_refs[global_id] = todo_info.r_len;
                chunk0_h_todo_querys[global_id] = std::string_view(h_query_seq + q_begin, q_len);
                chunk0_h_todo_refs[global_id] = std::string_view(h_ref_seq.c_str() + todo_info.r_begin, todo_info.r_len);
#else
                todo_info.global_id = cal_todo_cnt;
                chunk0_h_todo_querys.push_back(std::string_view(h_query_seq + q_begin, q_len));
                chunk0_h_todo_refs.push_back(std::string_view(h_ref_seq.c_str() + todo_info.r_begin, todo_info.r_len));
#endif
                cal_todo_cnt++;
            }
        }
        time2_1 += GetTime() - t_1;
#ifdef use_device_mem
        assert(cal_todo_cnt == *h_todo_cnt);
        assert(todo_querys.size() == todo_refs.size());
#endif
        // ssw on GPU
        t_1 = GetTime();
        std::thread gpu_ssw_async;
        gpu_ssw_async = std::thread([&] (){
            chunk0_gasal_results.clear();
            cudaSetDevice(gpu_id);
#ifdef use_device_mem
            char* batch_query_ptr = device_query_ptr;
            char* batch_ref_ptr = device_ref_ptr;
            for (size_t i = 0; i + STREAM_BATCH_SIZE_GPU <= todo_querys.size(); i += STREAM_BATCH_SIZE_GPU) {
                auto query_start = todo_querys.begin() + i;
                auto query_end = query_start + STREAM_BATCH_SIZE_GPU;
                std::vector<int> query_batch(query_start, query_end);
                auto ref_start = todo_refs.begin() + i;
                auto ref_end = ref_start + STREAM_BATCH_SIZE_GPU;
                std::vector<int> ref_batch(ref_start, ref_end);
                int batch_query_size = 0;
                int batch_ref_size = 0;
                for (int j = 0; j < query_batch.size(); j++) {
                    //printf("[%d %d]\n", query_batch[j], ref_batch[j]);
                    batch_query_size += (query_batch[j] + 7) & ~7;
                    batch_ref_size += (ref_batch[j] + 7) & ~7;
                }
                solve_ssw_on_gpu_pre_copy(
                        thread_id, gasal_results_tmp, query_batch, ref_batch, aln_params.match,
                        aln_params.mismatch, aln_params.gap_open, aln_params.gap_extend, batch_query_ptr, batch_ref_ptr
                );
                batch_query_ptr += batch_query_size;
                batch_ref_ptr += batch_ref_size;
                chunk0_gasal_results.insert(chunk0_gasal_results.end(), gasal_results_tmp.begin(), gasal_results_tmp.end());
            }
            size_t remaining = todo_querys.size() % STREAM_BATCH_SIZE_GPU;
            if (remaining > 0) {
                auto query_start = todo_querys.end() - remaining;
                std::vector<int> query_batch(query_start, todo_querys.end());
                auto ref_start = todo_refs.end() - remaining;
                std::vector<int> ref_batch(ref_start, todo_refs.end());
                int batch_query_size = 0;
                int batch_ref_size = 0;
                for (int j = 0; j < query_batch.size(); j++) {
                    //printf("[%d %d]\n", query_batch[j], ref_batch[j]);
                    batch_query_size += (query_batch[j] + 7) & ~7;
                    batch_ref_size += (ref_batch[j] + 7) & ~7;
                }
                solve_ssw_on_gpu_pre_copy(
                        thread_id, gasal_results_tmp, query_batch, ref_batch, aln_params.match,
                        aln_params.mismatch, aln_params.gap_open, aln_params.gap_extend, batch_query_ptr, batch_ref_ptr
                );
                batch_query_ptr += batch_query_size;
                batch_ref_ptr += batch_ref_size;
                chunk0_gasal_results.insert(chunk0_gasal_results.end(), gasal_results_tmp.begin(), gasal_results_tmp.end());
            }
            assert(batch_query_ptr - device_query_ptr == *h_query_offset);
            assert(batch_ref_ptr - device_ref_ptr == *h_ref_offset);
#else
            for (size_t i = 0; i + STREAM_BATCH_SIZE_GPU <= chunk0_h_todo_querys.size(); i += STREAM_BATCH_SIZE_GPU) {
                auto query_start = chunk0_h_todo_querys.begin() + i;
                auto query_end = query_start + STREAM_BATCH_SIZE_GPU;
                std::vector<std::string_view> query_batch(query_start, query_end);
                auto ref_start = chunk0_h_todo_refs.begin() + i;
                auto ref_end = ref_start + STREAM_BATCH_SIZE_GPU;
                std::vector<std::string_view> ref_batch(ref_start, ref_end);
                solve_ssw_on_gpu(
                        thread_id, gasal_results_tmp, query_batch, ref_batch, aln_params.match,
                        aln_params.mismatch, aln_params.gap_open, aln_params.gap_extend
                );
                chunk0_gasal_results.insert(chunk0_gasal_results.end(), gasal_results_tmp.begin(), gasal_results_tmp.end());
            }
            size_t remaining = chunk0_h_todo_querys.size() % STREAM_BATCH_SIZE_GPU;
            if (remaining > 0) {
                auto query_start = chunk0_h_todo_querys.end() - remaining;
                std::vector<std::string_view> query_batch(query_start, chunk0_h_todo_querys.end());
                auto ref_start = chunk0_h_todo_refs.end() - remaining;
                std::vector<std::string_view> ref_batch(ref_start, chunk0_h_todo_refs.end());
                solve_ssw_on_gpu(
                        thread_id, gasal_results_tmp, query_batch, ref_batch, aln_params.match,
                        aln_params.mismatch, aln_params.gap_open, aln_params.gap_extend
                );
                chunk0_gasal_results.insert(chunk0_gasal_results.end(), gasal_results_tmp.begin(), gasal_results_tmp.end());
            }
#endif
        });
        gpu_ssw_async.join();
        time2_2 += GetTime() - t_1;

    }


    std::thread cpu_async_thread;
//    printf("se thread %d bind to %d - %d\n", thread_id, thread_id, async_thread_id);

    while (true) {

        if (chunk0_datas.size() == 0) break;
        cpu_async_thread = std::thread([&] () {
            double t_start;
            unset_thread_affinity();
            set_thread_affinity(async_thread_id);
            cudaSetDevice(gpu_id);
            // step: p_0
            {
                // post-ssw on CPU
                info_results.clear();
                t_start = GetTime();
                for (size_t i = 0; i < chunk0_h_todo_querys.size(); i++) {
                    AlignmentInfo info;
                    const auto &todo_q = chunk0_h_todo_querys[i];
                    const auto &todo_r = chunk0_h_todo_refs[i];
                    total_ssw++;
                    if (gasal_fail(todo_q, todo_r, chunk0_gasal_results[i])) {
                        double ta = GetTime();
                        info = aligner.align(todo_q, todo_r);
                        time2_3_1 += GetTime() - ta;
                    } else {
                        gpu_ssw++;
                        double ta = GetTime();
                        info = aligner.align_gpu(todo_q, todo_r, chunk0_gasal_results[i]);
                        time2_3_2 += GetTime() - ta;
                    }
                    info_results.push_back(info);
                }
                time2_3 += GetTime() - t_start;

                // update align_res using ssw result
                t_start = GetTime();
                for (size_t i = 0; i < chunk0_datas.size(); i++) {
                    const auto mu = isize_est->mu;
                    const auto sigma = isize_est->sigma;
                    GPUAlignTmpRes &align_tmp_res = chunk0_global_align_res[i];
                    size_t todo_size = align_tmp_res.todo_nams.size();
                    if (align_tmp_res.type == 1 || align_tmp_res.type == 2) {
                        assert(0);
                    } else if (align_tmp_res.type == 3) {
                        assert(0);
                    } else if (align_tmp_res.type == 4) {
                        int pos = 0;
                        for (size_t j = 0; j < todo_size; j++) {
                            if (!align_tmp_res.done_align[j]) {
                                if (align_tmp_res.is_extend_seed[j]) {
                                    GPU_part2_extend_seed_store_res(
                                            align_tmp_res, j, chunk0_datas[i], chunk0_datas[i], references,
                                            info_results[align_tmp_res.todo_infos[pos++].global_id]
                                    );
                                } else {
                                    assert(0);
                                }
                            }
                        }
                        assert(pos == align_tmp_res.todo_infos.size());
                    }
                }
                time2_4 += GetTime() - t_start;

                // format align_res to SAM item
                int base_read_num = 0;
                for (int chunk_id = 0; chunk_id < chunk0_real_chunk_num; chunk_id++) {
                    int fx_chunk_id = chunk0_real_chunk_ids[chunk_id];
                    int this_chunk_read_num = chunk0_real_chunk_nums[chunk_id];
                    t_start = GetTime();
                    std::string sam_out;
                    sam_out.reserve(7 * map_param.r * this_chunk_read_num);
                    Sam sam{sam_out, references, map_param.cigar_ops, read_group_id, map_param.output_unmapped, map_param.details};
                    for (size_t i = base_read_num; i < base_read_num + this_chunk_read_num; ++i) {
                        GPU_align_SE_read_last(chunk0_global_align_res[i], chunk0_datas[i], sam, sam_out, aligner,
                                               map_param, index_parameters, references, index, random_engine,
                                               time3_1_1, time3_1_2, time3_1_3, time3_1_4
                        );
                        GPUAlignTmpRes& align_tmp_res = chunk0_global_align_res[i];
                        for (int j = 0; j < align_tmp_res.todo_nams.size(); j++) {
                            if (align_tmp_res.cigar_info[j].has_realloc == 1) {
//                            printf("free %d %d %d\n", i, j, align_tmp_res.cigar_info[j].cigar[0]);
                                align_tmp_res.cigar_info[j].has_realloc = 0;
                                free(align_tmp_res.cigar_info[j].cigar);
                                align_tmp_res.cigar_info[j].cigar = align_tmp_res.cigar_info[j].gpu_cigar;
                            }
                        }
                    }
                    time3_1 += GetTime() - t_start;

                    // output SAM data
                    t_start = GetTime();
                    if (sam_out.length() > 0) output_buffer.output_records(std::move(sam_out), fx_chunk_id, unordered_output);
                    time3_2 += GetTime() - t_start;

                    base_read_num += this_chunk_read_num;
                }
                assert(base_read_num == chunk0_datas.size());

                // release rabbitfx chunk
                t_start = GetTime();
                for (int chunk_id = 0; chunk_id < chunk0_real_chunk_num; chunk_id++) {
                    fastqPool.Release(chunk0_fqdatachunks[chunk_id]);
                }
                time3_3 += GetTime() - t_start;
            }

            // step: f_2
            {
                t_start = GetTime();
                bool res;
                // format data
                int rc_pos = 0;
                chunk2_chunk_num = rand() % small_chunk_num + small_chunk_num + 1;
                chunk2_real_chunk_num = 0;
                chunk2_datas.clear();
                for (int chunk_id = 0; chunk_id < chunk2_chunk_num; chunk_id++) {
                    res = dq.Pop(id, chunk2_fqdatachunks[chunk_id]);
                    if (res) {
                        double t_3 = GetTime();
                        neo_datas.clear();
                        rabbit::fq::chunkFormat(chunk2_fqdatachunks[chunk_id], neo_datas);
                        time1_1_1 += GetTime() - t_3;

                        t_3 = GetTime();
                        for (int i = 0; i < neo_datas.size(); i++) {
                            char *name = (char *) neo_datas[i].base + neo_datas[i].pname;
                            if (neo_datas[i].lname > 0 && name[0] == '@') {
                                neo_datas[i].pname++;
                                neo_datas[i].lname--;
                                name++;
                            }
                            for (int j = 0; j < neo_datas[i].lname; j++) {
                                if (name[j] == ' ') {
                                    neo_datas[i].lname = j;
                                    break;
                                }
                            }
                            char *seq = (char *) neo_datas[i].base + neo_datas[i].pseq;
                            chunk2_datas.push_back({neo_datas[i], chunk2_rc_data + rc_pos});
                            for (int j = 0; j < neo_datas[i].lseq; j++) {
                                chunk2_rc_data[rc_pos++] = rc_gpu_nt2nt[seq[neo_datas[i].lseq - 1 - j]];
                            }
                        }
                        time1_1_2 += GetTime() - t_3;
                        chunk2_real_chunk_nums[chunk_id] = neo_datas.size();
                        chunk2_real_chunk_ids[chunk_id] = id;
                        chunk2_real_chunk_num++;
                    } else break;
                }
                assert(rc_pos <= batch_total_read_len);
                time1_1 += GetTime() - t_start;
            }
        });

        // step: s+e_1
        {
            // seeding on GPU
            t_1 = GetTime();
            chunk_index = id;
            random_engine.seed(chunk_index);
            cudaMemset(d_todo_cnt, 0, sizeof(int));
            cudaMemset(d_query_offset, 0, sizeof(int));
            cudaMemset(d_ref_offset, 0, sizeof(int));
            if (!chunk1_datas.empty()) GPU_align_SE(chunk1_datas,
                                                    ctx,
                                                    align_tmp_results,
                                                    global_hits_num, global_nams_info, global_align_info,
                                                    index, d_aligner, d_map_param, d_index_para,
                                                    global_references[gpu_id], d_randstrobes[gpu_id], d_randstrobe_start_indices[gpu_id],
                                                    global_randstrobes, global_todo_ids, global_randstrobe_sizes, global_hashes_value,
                                                    global_hits_per_ref0s, global_hits_per_ref1s, global_nams,
                                                    chunk1_global_align_res, chunk1_global_align_res_data, pre_vec_size,
                                                    d_seq, d_len, d_pre_sum, h_seq, h_len, h_pre_sum,
                                                    d_todo_cnt, device_query_ptr, device_ref_ptr, d_query_offset, d_ref_offset, batch_read_num, batch_total_read_len, rescue_threshold);
            cudaMemcpy(h_todo_cnt, d_todo_cnt, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_query_offset, d_query_offset, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_ref_offset, d_ref_offset, sizeof(int), cudaMemcpyDeviceToHost);
            assert(*h_query_offset <= mx_device_query_size);
            assert(*h_ref_offset <= mx_device_ref_size);
            time1_2 += GetTime() - t_1;

            // construct todo_info from align_res
            t_1 = GetTime();
#ifdef use_device_mem
            todo_querys.resize(*h_todo_cnt);
            todo_refs.resize(*h_todo_cnt);
            chunk1_h_todo_querys.resize(*h_todo_cnt);
            chunk1_h_todo_refs.resize(*h_todo_cnt);
#else
            chunk1_h_todo_querys.clear();
            chunk1_h_todo_refs.clear();
#endif
            int cal_todo_cnt = 0;
            for (int i = 0; i < chunk1_datas.size(); i++) {
                GPUAlignTmpRes &align_tmp_res = chunk1_global_align_res[i];
                for (int j = 0; j < align_tmp_res.todo_infos.size(); j++) {
                    TODOInfos& todo_info = align_tmp_res.todo_infos[j];
                    int global_id = todo_info.global_id;
                    uint32_t info = todo_info.read_info;
                    int is_read1 = (info >> 31) & 0x1;
                    int is_rc    = (info >> 30) & 0x1;
                    int q_begin  = (info >> 15) & 0x7FFF;
                    int q_len    = info & 0x7FFF;
                    const auto& h_query_seq = is_rc ? chunk1_datas[i].rc : (char*)chunk1_datas[i].read.base + chunk1_datas[i].read.pseq;
                    const auto& h_ref_seq = references.sequences[todo_info.ref_id];
#ifdef use_device_mem
                    todo_querys[global_id] = q_len;
                    todo_refs[global_id] = todo_info.r_len;
                    chunk1_h_todo_querys[global_id] = std::string_view(h_query_seq + q_begin, q_len);
                    chunk1_h_todo_refs[global_id] = std::string_view(h_ref_seq.c_str() + todo_info.r_begin, todo_info.r_len);
#else
                    chunk1_h_todo_querys.push_back(std::string_view(h_query_seq + q_begin, q_len));
                    chunk1_h_todo_refs.push_back(std::string_view(h_ref_seq.c_str() + todo_info.r_begin, todo_info.r_len));
                    todo_info.global_id = cal_todo_cnt;
#endif
                    cal_todo_cnt++;
                }
            }
            time2_1 += GetTime() - t_1;
#ifdef use_device_mem
            assert(cal_todo_cnt == *h_todo_cnt);
            assert(todo_querys.size() == todo_refs.size());
#endif

            // ssw on GPU
            t_1 = GetTime();
            std::thread gpu_ssw_async;
            gpu_ssw_async = std::thread([&] (){
                chunk1_gasal_results.clear();
                cudaSetDevice(gpu_id);
#ifdef use_device_mem
                char* batch_query_ptr = device_query_ptr;
                char* batch_ref_ptr = device_ref_ptr;
                //printf("todo size: %d\n", todo_querys.size());
                for (size_t i = 0; i + STREAM_BATCH_SIZE_GPU <= todo_querys.size(); i += STREAM_BATCH_SIZE_GPU) {
                    auto query_start = todo_querys.begin() + i;
                    auto query_end = query_start + STREAM_BATCH_SIZE_GPU;
                    std::vector<int> query_batch(query_start, query_end);
                    auto ref_start = todo_refs.begin() + i;
                    auto ref_end = ref_start + STREAM_BATCH_SIZE_GPU;
                    std::vector<int> ref_batch(ref_start, ref_end);
                    int batch_query_size = 0;
                    int batch_ref_size = 0;
                    for (int j = 0; j < query_batch.size(); j++) {
                        //printf("[%d %d]\n", query_batch[j], ref_batch[j]);
                        batch_query_size += (query_batch[j] + 7) & ~7;
                        batch_ref_size += (ref_batch[j] + 7) & ~7;
                    }
                    solve_ssw_on_gpu_pre_copy(
                            thread_id, gasal_results_tmp, query_batch, ref_batch, aln_params.match,
                            aln_params.mismatch, aln_params.gap_open, aln_params.gap_extend, batch_query_ptr, batch_ref_ptr
                    );
                    batch_query_ptr += batch_query_size;
                    batch_ref_ptr += batch_ref_size;
                    chunk1_gasal_results.insert(chunk1_gasal_results.end(), gasal_results_tmp.begin(), gasal_results_tmp.end());
                }
                size_t remaining = todo_querys.size() % STREAM_BATCH_SIZE_GPU;
                if (remaining > 0) {
                    auto query_start = todo_querys.end() - remaining;
                    std::vector<int> query_batch(query_start, todo_querys.end());
                    auto ref_start = todo_refs.end() - remaining;
                    std::vector<int> ref_batch(ref_start, todo_refs.end());
                    int batch_query_size = 0;
                    int batch_ref_size = 0;
                    for (int j = 0; j < query_batch.size(); j++) {
                        //printf("[%d %d]\n", query_batch[j], ref_batch[j]);
                        batch_query_size += (query_batch[j] + 7) & ~7;
                        batch_ref_size += (ref_batch[j] + 7) & ~7;
                    }
                    solve_ssw_on_gpu_pre_copy(
                            thread_id, gasal_results_tmp, query_batch, ref_batch, aln_params.match,
                            aln_params.mismatch, aln_params.gap_open, aln_params.gap_extend, batch_query_ptr, batch_ref_ptr
                    );
                    batch_query_ptr += batch_query_size;
                    batch_ref_ptr += batch_ref_size;
                    chunk1_gasal_results.insert(chunk1_gasal_results.end(), gasal_results_tmp.begin(), gasal_results_tmp.end());
                }
                assert(batch_query_ptr - device_query_ptr == *h_query_offset);
                assert(batch_ref_ptr - device_ref_ptr == *h_ref_offset);
#else
                for (size_t i = 0; i + STREAM_BATCH_SIZE_GPU <= chunk1_h_todo_querys.size(); i += STREAM_BATCH_SIZE_GPU) {
                    auto query_start = chunk1_h_todo_querys.begin() + i;
                    auto query_end = query_start + STREAM_BATCH_SIZE_GPU;
                    std::vector<std::string_view> query_batch(query_start, query_end);
                    auto ref_start = chunk1_h_todo_refs.begin() + i;
                    auto ref_end = ref_start + STREAM_BATCH_SIZE_GPU;
                    std::vector<std::string_view> ref_batch(ref_start, ref_end);
                    solve_ssw_on_gpu(
                            thread_id, gasal_results_tmp, query_batch, ref_batch, aln_params.match,
                            aln_params.mismatch, aln_params.gap_open, aln_params.gap_extend
                    );
                    chunk1_gasal_results.insert(chunk1_gasal_results.end(), gasal_results_tmp.begin(), gasal_results_tmp.end());
                }
                size_t remaining = chunk1_h_todo_querys.size() % STREAM_BATCH_SIZE_GPU;
                if (remaining > 0) {
                    auto query_start = chunk1_h_todo_querys.end() - remaining;
                    std::vector<std::string_view> query_batch(query_start, chunk1_h_todo_querys.end());
                    auto ref_start = chunk1_h_todo_refs.end() - remaining;
                    std::vector<std::string_view> ref_batch(ref_start, chunk1_h_todo_refs.end());
                    solve_ssw_on_gpu(
                            thread_id, gasal_results_tmp, query_batch, ref_batch, aln_params.match,
                            aln_params.mismatch, aln_params.gap_open, aln_params.gap_extend
                    );
                    chunk1_gasal_results.insert(chunk1_gasal_results.end(), gasal_results_tmp.begin(), gasal_results_tmp.end());
                }
#endif
            });
            gpu_ssw_async.join();
            time2_2 += GetTime() - t_1;
        }

        if (cpu_async_thread.joinable()) {
            cpu_async_thread.join();
        }

        // swap data
        {
            t_1 = GetTime();
            std::vector<neoRcRef> temp_datas = std::move(chunk0_datas);
            chunk0_datas = std::move(chunk1_datas);
            chunk1_datas = std::move(chunk2_datas);
            chunk2_datas = std::move(temp_datas);

            std::vector<std::string_view> temp_h_todo_querys = std::move(chunk0_h_todo_querys);
            chunk0_h_todo_querys = std::move(chunk1_h_todo_querys);
            chunk1_h_todo_querys = std::move(chunk2_h_todo_querys);
            chunk2_h_todo_querys = std::move(temp_h_todo_querys);

            std::vector<std::string_view> temp_h_todo_refs = std::move(chunk0_h_todo_refs);
            chunk0_h_todo_refs = std::move(chunk1_h_todo_refs);
            chunk1_h_todo_refs = std::move(chunk2_h_todo_refs);
            chunk2_h_todo_refs = std::move(temp_h_todo_refs);

            std::vector<gasal_tmp_res> temp_gasal_results = std::move(chunk0_gasal_results);
            chunk0_gasal_results = std::move(chunk1_gasal_results);
            chunk1_gasal_results = std::move(chunk2_gasal_results);
            chunk2_gasal_results = std::move(temp_gasal_results);

            char* temp_rc_data = chunk0_rc_data;
            chunk0_rc_data = chunk1_rc_data;
            chunk1_rc_data = chunk2_rc_data;
            chunk2_rc_data = temp_rc_data;

            int temp_real_chunk_num = chunk0_real_chunk_num;
            chunk0_real_chunk_num = chunk1_real_chunk_num;
            chunk1_real_chunk_num = chunk2_real_chunk_num;
            chunk2_real_chunk_num = temp_real_chunk_num;

            int temp_chunk_num = chunk0_chunk_num;
            chunk0_chunk_num = chunk1_chunk_num;
            chunk1_chunk_num = chunk2_chunk_num;
            chunk2_chunk_num = temp_chunk_num;

            int* temp_chunk_nums = chunk0_real_chunk_nums;
            chunk0_real_chunk_nums = chunk1_real_chunk_nums;
            chunk1_real_chunk_nums = chunk2_real_chunk_nums;
            chunk2_real_chunk_nums = temp_chunk_nums;

            int* temp_chunk_ids = chunk0_real_chunk_ids;
            chunk0_real_chunk_ids = chunk1_real_chunk_ids;
            chunk1_real_chunk_ids = chunk2_real_chunk_ids;
            chunk2_real_chunk_ids = temp_chunk_ids;

            GPUAlignTmpRes* temp_global_align_res = chunk0_global_align_res;
            chunk0_global_align_res = chunk1_global_align_res;
            chunk1_global_align_res = chunk2_global_align_res;
            chunk2_global_align_res = temp_global_align_res;

            char* temp_global_align_res_data = chunk0_global_align_res_data;
            chunk0_global_align_res_data = chunk1_global_align_res_data;
            chunk1_global_align_res_data = chunk2_global_align_res_data;
            chunk2_global_align_res_data = temp_global_align_res_data;

            for (int i = 0; i < MAX_RABBITFX_CHUNK_NUM; i++) {
                rabbit::fq::FastqDataChunk * temp_fqdatachunk = chunk0_fqdatachunks[i];
                chunk0_fqdatachunks[i] = chunk1_fqdatachunks[i];
                chunk1_fqdatachunks[i] = chunk2_fqdatachunks[i];
                chunk2_fqdatachunks[i] = temp_fqdatachunk;
            }

            time3_5 += GetTime() - t_1;
        }
    }
    done = true;

#ifdef PRINT_GPU_TIMER
    //    // The following cout block is added as requested.
    std::cout << "--------------- GPU Kernel Timers ---------------" << std::endl;
    std::cout << "pack read on host cost: " << gpu_copy1 << " s" << std::endl;
    std::cout << "H2D read cost: " << gpu_copy2 << " s" << std::endl;
    std::cout << "get_randstrobes cost (gpu_cost1): " << gpu_cost1 << " s" << std::endl;
    std::cout << "get_hits cost (gpu_cost2): " << gpu_cost2 << " s" << std::endl;
    std::cout << "filter normal read cost (gpu_init1): " << gpu_init1 << " s" << std::endl;
    std::cout << "sort_hits cost (gpu_cost3): " << gpu_cost3 << " s" << std::endl;
    std::cout << "merge_hits_get_nams cost (gpu_cost4): " << gpu_cost4 << " s" << std::endl;
    std::cout << "filter rescue read cost (gpu_init2): " << gpu_init2 << " s" << std::endl;
    std::cout << "rescue_get_hits cost (gpu_cost5): " << gpu_cost5 << " s" << std::endl;
    std::cout << "rescue_sort_hits cost (gpu_cost6): " << gpu_cost6 << " s" << std::endl;
    std::cout << "rescue_merge_hits_get_nams cost (gpu_cost7): " << gpu_cost7 << " s" << std::endl;
    std::cout << "sort_nams cost (gpu_cost8): " << gpu_cost8 << " s" << std::endl;
    std::cout << "pre_align cost (gpu_cost9): " << gpu_cost9 << " s" << std::endl;
    std::cout << "alloc align_tmp_res cost (gpu_init3): " << gpu_init3 << " s" << std::endl;
    std::cout << "align_SE cost (gpu_cost10): " << gpu_cost10 << " s" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "Total function cost (tot_cost): " << tot_cost << " s" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
#endif

    t_1 = GetTime();
    cudaFree(d_aligner);
    cudaFree(d_map_param);
    cudaFree(d_index_para);

    cudaFree(global_randstrobes);
    cudaFree(global_todo_ids);
    cudaFree(global_randstrobe_sizes);
    cudaFree(global_hashes_value);
    cudaFree(global_hits_per_ref0s);
    cudaFree(global_hits_per_ref1s);
    cudaFree(global_nams);

    cudaFree(d_seq);
    cudaFree(d_len);
    cudaFree(d_pre_sum);

    cudaFree(global_hits_num);
    cudaFree(global_nams_info);
    cudaFree(global_align_info);

    cudaFree(device_query_ptr);
    cudaFree(device_ref_ptr);

    cudaFree(d_todo_cnt);
    cudaFree(d_query_offset);
    cudaFree(d_ref_offset);

    cudaFreeHost(h_seq);
    cudaFreeHost(h_len);
    cudaFreeHost(h_pre_sum);

    cudaFreeHost(h_todo_cnt);
    cudaFreeHost(h_query_offset);
    cudaFreeHost(h_ref_offset);


    delete[] chunk0_rc_data;
    delete[] chunk1_rc_data;
    delete[] chunk2_rc_data;

    delete[] chunk0_real_chunk_nums;
    delete[] chunk1_real_chunk_nums;
    delete[] chunk2_real_chunk_nums;
    delete[] chunk0_real_chunk_ids;
    delete[] chunk1_real_chunk_ids;
    delete[] chunk2_real_chunk_ids;


    time4 += GetTime() - t_1;


    time_tot = GetTime() - t_0;
#ifdef PRINT_CPU_TIMER
    printf("tot ssw %d, gpu ssw %d, %.2f\n", total_ssw, gpu_ssw, 1.0 * gpu_ssw / total_ssw);
    fprintf(
            stderr, "cost time0:%.2f(%.2f %.2f %.2f %.2f [%.2f %.2f %.2f %.2f]) time1:(%.2f[%.2f %.2f] %.2f %.2f) time2:(%.2f[%.2f] %.2f %.2f[%.2f %.2f] %.2f) time3:(%.2f[%.2f %.2f %.2f %.2f] %.2f %.2f %.2f %.2f), time4:%.2f tot time:%.2f\n",
            time0, time0_1, time0_2, time0_3, time0_4, time0_4_1, time0_4_2, time0_4_3, time0_4_4,
            time1_1, time1_1_1, time1_1_2, time1_2, time1_3,
            time2_1, time2_1_1, time2_2, time2_3, time2_3_1, time2_3_2, time2_4,
            time3_1, time3_1_1, time3_1_2, time3_1_3, time3_1_4, time3_2, time3_3, time3_4, time3_5,
            time4, time_tot
    );
#endif

    cudaStreamSynchronize(ctx.stream);
}
