#include "gpu_seeding.h"

__device__ inline randstrobe_hash_t gpu_get_hash(const RefRandstrobe *d_randstrobes, size_t d_randstrobes_size, my_bucket_index_t position) {
    if (position < d_randstrobes_size) {
        return d_randstrobes[position].hash;
    } else {
        return static_cast<randstrobe_hash_t>(-1);
    }
}

__device__ inline bool gpu_is_filtered(const RefRandstrobe *d_randstrobes, size_t d_randstrobes_size, my_bucket_index_t position,
                                       unsigned int filter_cutoff) {
    return gpu_get_hash(d_randstrobes, d_randstrobes_size, position) ==
           gpu_get_hash(d_randstrobes, d_randstrobes_size, position + filter_cutoff);
}

__device__ int gpu_get_count(
        const RefRandstrobe *d_randstrobes,
        const my_bucket_index_t *d_randstrobe_start_indices,
        my_bucket_index_t position,
        int bits
) {
    const auto key = d_randstrobes[position].hash;
    const unsigned int top_N = key >> (64 - bits);
    int64_t position_end = d_randstrobe_start_indices[top_N + 1];
    int64_t position_start = position;

    if(position_end == 0) return 0;
    int64_t low = position_start, high = position_end - 1, ans = 0;
    while (low <= high) {
        int64_t mid = (low + high) / 2;
        if (d_randstrobes[mid].hash == key) {
            low = mid + 1;
            ans = mid;
        } else {
            high = mid - 1;
        }
    }
    return ans - position_start + 1;
}

__device__ inline size_t gpu_find(
        const RefRandstrobe *d_randstrobes,
        const my_bucket_index_t *d_randstrobe_start_indices,
        const randstrobe_hash_t key,
        int bits
) {
    const unsigned int top_N = key >> (64 - bits);
    my_bucket_index_t position_start = d_randstrobe_start_indices[top_N];
    my_bucket_index_t position_end = d_randstrobe_start_indices[top_N + 1];
    if(position_end - position_start < 64) {
        for (my_bucket_index_t i = position_start; i < position_end; ++i) {
            if (d_randstrobes[i].hash == key) {
                return i;
            }
        }
        return static_cast<size_t>(-1);
    } else {
        my_bucket_index_t low = position_start, high = position_end;
        while (low < high) {
            my_bucket_index_t mid = low + (high - low) / 2;
            if (d_randstrobes[mid].hash < key) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        if (d_randstrobes[low].hash == key) {
            return low;
        } else return static_cast<size_t>(-1);
    }
}

__device__ void add_to_hits_per_ref(
        my_vector<my_pair<int, Hit>>& hits_per_ref,
        int query_start,
        int query_end,
        size_t position,
        const RefRandstrobe *d_randstrobes,
        size_t d_randstrobes_size,
        int k
) {
    int min_diff = 1 << 30;
    for (const auto hash = gpu_get_hash(d_randstrobes, d_randstrobes_size, position); gpu_get_hash(d_randstrobes, d_randstrobes_size, position) == hash; ++position) {
        int ref_start = d_randstrobes[position].position;
        int ref_end = ref_start + d_randstrobes[position].strobe2_offset() + k;
        int diff = std::abs((query_end - query_start) - (ref_end - ref_start));
        if (diff <= min_diff) {
            hits_per_ref.push_back({d_randstrobes[position].reference_index(), Hit{query_start, query_end, ref_start, ref_end}});
            min_diff = diff;
        }
    }
}

__global__ void gpu_get_randstrobes(
        int num_tasks,
        int read_num,
        int base_read_num,
        int *pre_sum,
        int *lens,
        char *all_seqs,
        IndexParameters *index_para,
        int *randstrobe_sizes,
        uint64_t *hashes,
        my_vector<QueryRandstrobe>* global_randstrobes
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num_tasks) {
        int read_id = (id + base_read_num) % read_num;
        int is_read2 = (id + base_read_num) / read_num;
        size_t len;
        char *seq, *rc;
        if (is_read2 == 0) {
            len = lens[read_id];
            seq = all_seqs + pre_sum[read_id];
            rc = all_seqs + pre_sum[read_id + read_num];
        } else {
            len = lens[read_id + read_num * 2];
            seq = all_seqs + pre_sum[read_id + read_num * 2];
            rc = all_seqs + pre_sum[read_id + read_num * 3];
        }

        my_vector<Syncmer> syncmers(len);

        const int k = index_para->syncmer.k;
        const int s = index_para->syncmer.s;
        const int t = index_para->syncmer.t_syncmer;

        const uint64_t kmask = (1ULL << 2 * k) - 1;
        const uint64_t smask = (1ULL << 2 * s) - 1;
        const uint64_t kshift = (k - 1) * 2;
        const uint64_t sshift = (s - 1) * 2;
        my_vector<uint64_t> gpu_qs(len * 2);
        int l_pos = 0;
        int r_pos = 0;
        uint64_t qs_min_val = UINT64_MAX;
        int qs_min_pos = -1;
        int l = 0;
        uint64_t xk[2] = {0, 0};
        uint64_t xs[2] = {0, 0};
        for (size_t i = 0; i < len; i++) {
            int c = gpu_seq_nt4_table[(uint8_t) seq[i]];
            if (c < 4) { // not an "N" base
                xk[0] = (xk[0] << 2 | c) & kmask;                  // forward strand
                xk[1] = xk[1] >> 2 | (uint64_t)(3 - c) << kshift;  // reverse strand
                xs[0] = (xs[0] << 2 | c) & smask;                  // forward strand
                xs[1] = xs[1] >> 2 | (uint64_t)(3 - c) << sshift;  // reverse strand
                if (++l < s) {
                    continue;
                }
                // we find an s-mer
                uint64_t ys = xs[0] < xs[1] ? xs[0] : xs[1];
                uint64_t hash_s = gpu_syncmer_smer_hash(ys);
                gpu_qs[r_pos++] = hash_s;
                //assert(r_pos < len * 2);
                // not enough hashes in the queue, yet
                if (r_pos - l_pos < k - s + 1) {
                    continue;
                }
                if (r_pos - l_pos == k - s + 1) { // We are at the last s-mer within the first k-mer, need to decide if we add it
                    for (int j = l_pos; j < r_pos; j++) {
                        if (gpu_qs[j] < qs_min_val) {
                            qs_min_val = gpu_qs[j];
                            qs_min_pos = i - k + j - l_pos + 1;
                        }
                    }
                } else {
                    // update queue and current minimum and position
                    l_pos++;
                    if (qs_min_pos == i - k) { // we popped the previous minimizer, find new brute force
                        qs_min_val = UINT64_MAX;
                        qs_min_pos = i - s + 1;
                        for (int j = r_pos - 1; j >= l_pos; j--) { //Iterate in reverse to choose the rightmost minimizer in a window
                            if (gpu_qs[j] < qs_min_val) {
                                qs_min_val = gpu_qs[j];
                                qs_min_pos = i - k + j - l_pos + 1;
                            }
                        }
                    } else if (hash_s < qs_min_val) { // the new value added to queue is the new minimum
                        qs_min_val = hash_s;
                        qs_min_pos = i - s + 1;
                    }
                }
                if (qs_min_pos == i - k + t) { // occurs at t:th position in k-mer
                    uint64_t yk = xk[0] < xk[1] ? xk[0] : xk[1];
                    syncmers.push_back(Syncmer{gpu_syncmer_kmer_hash(yk), i - k + 1});
                }
            } else {
                // if there is an "N", restart
                qs_min_val = UINT64_MAX;
                qs_min_pos = -1;
                l = xs[0] = xs[1] = xk[0] = xk[1] = 0;
                r_pos = 0;
                l_pos = 0;
            }
        }


        const int w_min = index_para->randstrobe.w_min;
        const int w_max = index_para->randstrobe.w_max;
        const uint64_t q = index_para->randstrobe.q;
        const int max_dist = index_para->randstrobe.max_dist;

        global_randstrobes[id].init((my_max(syncmers.size() - w_min, 0)) * 2);

        for (int strobe1_index = 0; strobe1_index + w_min < syncmers.size(); strobe1_index++) {
            unsigned int w_end = (strobe1_index + w_max < syncmers.size() - 1) ? (strobe1_index + w_max) : syncmers.size() - 1;
            auto strobe1 = syncmers[strobe1_index];
            auto max_position = strobe1.position + max_dist;
            unsigned int w_start = strobe1_index + w_min;
            uint64_t min_val = 0xFFFFFFFFFFFFFFFF;
            Syncmer strobe2 = strobe1;
            for (auto i = w_start; i <= w_end && syncmers[i].position <= max_position; i++) {
                uint64_t hash_diff = (strobe1.hash ^ syncmers[i].hash) & q;
                uint64_t res = __popcll(hash_diff);
                if (res < min_val) {
                    min_val = res;
                    strobe2 = syncmers[i];
                }
            }
            global_randstrobes[id].push_back(
                    QueryRandstrobe{
                            gpu_randstrobe_hash(strobe1.hash, strobe2.hash), static_cast<uint32_t>(strobe1.position),
                            static_cast<uint32_t>(strobe2.position) + index_para->syncmer.k, false
                    }
            );
        }


        for (int i = 0; i < syncmers.size() / 2; i++) {
            my_swap(syncmers[i], syncmers[syncmers.size() - i - 1]);
        }
        for (size_t i = 0; i < syncmers.size(); i++) {
            syncmers[i].position = len - syncmers[i].position - (*index_para).syncmer.k;
        }

        for (int strobe1_index = 0; strobe1_index + w_min < syncmers.size(); strobe1_index++) {
            unsigned int w_end = (strobe1_index + w_max < syncmers.size() - 1) ? (strobe1_index + w_max) : syncmers.size() - 1;
            auto strobe1 = syncmers[strobe1_index];
            auto max_position = strobe1.position + max_dist;
            unsigned int w_start = strobe1_index + w_min;
            uint64_t min_val = 0xFFFFFFFFFFFFFFFF;
            Syncmer strobe2 = strobe1;
            for (auto i = w_start; i <= w_end && syncmers[i].position <= max_position; i++) {
                uint64_t hash_diff = (strobe1.hash ^ syncmers[i].hash) & q;
                uint64_t res = __popcll(hash_diff);
                if (res < min_val) {
                    min_val = res;
                    strobe2 = syncmers[i];
                }
            }
            global_randstrobes[id].push_back(
                    QueryRandstrobe{
                            gpu_randstrobe_hash(strobe1.hash, strobe2.hash), static_cast<uint32_t>(strobe1.position),
                            static_cast<uint32_t>(strobe2.position) + index_para->syncmer.k, true
                    }
            );
        }
    }
}


__global__ void gpu_get_hits_pre(
        int bits,
        unsigned int filter_cutoff,
        int rescue_cutoff,
        const RefRandstrobe *d_randstrobes,
        size_t d_randstrobes_size,
        const my_bucket_index_t *d_randstrobe_start_indices,
        int num_tasks,
        IndexParameters *index_para,
        uint64_t *global_hits_num,
        my_vector<QueryRandstrobe>* global_randstrobes,
        my_vector<my_pair<int, Hit>>* hits_per_ref0s,
        my_vector<my_pair<int, Hit>>* hits_per_ref1s
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num_tasks) {
        for (int i = 0; i < global_randstrobes[id].size(); i++) {
            QueryRandstrobe q = global_randstrobes[id][i];
            size_t position = gpu_find(d_randstrobes, d_randstrobe_start_indices, q.hash, bits);
            global_randstrobes[id][i].hash = position;
        }
    }
}

__global__ void gpu_get_hits_after(
        int bits,
        unsigned int filter_cutoff,
        int rescue_cutoff,
        const RefRandstrobe *d_randstrobes,
        size_t d_randstrobes_size,
        const my_bucket_index_t *d_randstrobe_start_indices,
        int num_tasks,
        IndexParameters *index_para,
        uint64_t *global_hits_num,
        my_vector<QueryRandstrobe>* global_randstrobes,
        my_vector<my_pair<int, Hit>>* hits_per_ref0s,
        my_vector<my_pair<int, Hit>>* hits_per_ref1s
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num_tasks) {
        uint64_t local_total_hits = 0;
        uint64_t local_nr_good_hits = 0;
        for (int i = 0; i < global_randstrobes[id].size(); i++) {
            QueryRandstrobe q = global_randstrobes[id][i];
            size_t position = q.hash;
            if (position != static_cast<size_t>(-1)) {
                local_total_hits++;
                if (!gpu_is_filtered(d_randstrobes, d_randstrobes_size, position, filter_cutoff)) {
                    local_nr_good_hits++;
                }
            }
        }
        float nonrepetitive_fraction = local_total_hits > 0 ? ((float) local_nr_good_hits) / ((float) local_total_hits) : 1.0;

        if (nonrepetitive_fraction < 0.7) return;

        hits_per_ref0s[id].init(8);
        hits_per_ref1s[id].init(8);
        for (int i = 0; i < global_randstrobes[id].size(); i++) {
            QueryRandstrobe q = global_randstrobes[id][i];
            size_t position = q.hash;
            if (position != static_cast<size_t>(-1)) {
                if (gpu_is_filtered(d_randstrobes, d_randstrobes_size, position, filter_cutoff)) continue;
                if(q.is_reverse) {
                    add_to_hits_per_ref(hits_per_ref1s[id], q.start, q.end, position, d_randstrobes, d_randstrobes_size, index_para->syncmer.k);
                } else {
                    add_to_hits_per_ref(hits_per_ref0s[id], q.start, q.end, position, d_randstrobes, d_randstrobes_size, index_para->syncmer.k);
                }
            }
        }
        global_randstrobes[id].release();
    }
}


__global__ void gpu_rescue_get_hits(
        int bits,
        unsigned int filter_cutoff,
        int rescue_cutoff,
        const RefRandstrobe *d_randstrobes,
        size_t d_randstrobes_size,
        const my_bucket_index_t *d_randstrobe_start_indices,
        int num_tasks,
        IndexParameters *index_para,
        uint64_t *global_hits_num,
        my_vector<QueryRandstrobe>* global_randstrobes,
        my_vector<my_pair<int, Hit>>* hits_per_ref0s,
        my_vector<my_pair<int, Hit>>* hits_per_ref1s,
        int* global_todo_ids,
        int rescue_threshold
)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num_tasks) {
        int real_id = global_todo_ids[id];
        my_vector<RescueHit> hits_t0;
        my_vector<RescueHit> hits_t1;
        for (int i = 0; i < global_randstrobes[real_id].size(); i++) {
            QueryRandstrobe q = global_randstrobes[real_id][i];
            size_t position = q.hash;
            if (position != static_cast<size_t>(-1)) {
                unsigned int count = gpu_get_count(d_randstrobes, d_randstrobe_start_indices, position, bits);
                RescueHit rh{position, count, q.start, q.end};
                if(q.is_reverse) hits_t1.push_back(rh);
                else hits_t0.push_back(rh);
            }
        }
        global_randstrobes[real_id].release();

        quick_sort(&(hits_t0[0]), hits_t0.size());
        quick_sort(&(hits_t1[0]), hits_t1.size());

        int cnt0 = 0, cnt1 = 0;
        for (int i = 0; i < hits_t0.size(); i++) {
            RescueHit &rh = hits_t0[i];
            if ((rh.count > rescue_cutoff && cnt0 >= 5) || rh.count > rescue_threshold) break;
            cnt0++;
        }
        for (int i = 0; i < hits_t1.size(); i++) {
            RescueHit &rh = hits_t1[i];
            if ((rh.count > rescue_cutoff && cnt1 >= 5) || rh.count > rescue_threshold) break;
            cnt1++;
        }

        quick_sort_iterative(&(hits_t0[0]), 0, cnt0 - 1, [](const RescueHit &r1, const RescueHit &r2) {
            return r1.query_start < r2.query_start;
        });
        quick_sort_iterative(&(hits_t1[0]), 0, cnt1 - 1, [](const RescueHit &r1, const RescueHit &r2) {
            return r1.query_start < r2.query_start;
        });

        for (int i = 0; i < cnt0; i++) {
            RescueHit &rh = hits_t0[i];
            add_to_hits_per_ref(hits_per_ref0s[real_id], rh.query_start, rh.query_end, rh.position, d_randstrobes, d_randstrobes_size, index_para->syncmer.k);
        }
        for (int i = 0; i < cnt1; i++) {
            RescueHit &rh = hits_t1[i];
            add_to_hits_per_ref(hits_per_ref1s[real_id], rh.query_start, rh.query_end, rh.position, d_randstrobes, d_randstrobes_size, index_para->syncmer.k);
        }
    }
}