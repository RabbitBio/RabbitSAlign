#include "gpu_merging.h"
#include <cub/cub.cuh>

__device__ size_t my_lower_bound(my_pair<int, Hit>* hits, size_t i_start, size_t i_end, int target) {
    size_t left = i_start, right = i_end;
    while (left < right) {
        size_t mid = left + (right - left) / 2;
        if (hits[mid].second.ref_start < target) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

__device__ void check_hits(my_vector<my_pair<int, Hit>> &hits_per_ref) {
    if (hits_per_ref.size() < 2) return;
    for(int i = 0; i < hits_per_ref.size() - 1; i++) {
        //if(hits_per_ref[i].first > hits_per_ref[i + 1].first) {
        //    printf("sort error [%d,%d] [%d,%d]\n", hits_per_ref[i].first, hits_per_ref[i].second.query_start, hits_per_ref[i + 1].first, hits_per_ref[i + 1].second.query_start);
        //    assert(false);
        //}
        if(hits_per_ref[i].first == hits_per_ref[i + 1].first && hits_per_ref[i].second.query_start > hits_per_ref[i + 1].second.query_start) {
            printf("sort error [%d,%d] [%d,%d]\n", hits_per_ref[i].first, hits_per_ref[i].second.query_start, hits_per_ref[i + 1].first, hits_per_ref[i + 1].second.query_start);
            assert(false);
        }
        if(hits_per_ref[i].first == hits_per_ref[i + 1].first && hits_per_ref[i].second.query_start == hits_per_ref[i + 1].second.query_start &&
           hits_per_ref[i].second.ref_start > hits_per_ref[i + 1].second.ref_start) {
            printf("sort error [%d,%d,%d] [%d,%d,%d]\n", hits_per_ref[i].first, hits_per_ref[i].second.query_start, hits_per_ref[i].second.ref_start,
                   hits_per_ref[i + 1].first, hits_per_ref[i + 1].second.query_start, hits_per_ref[i + 1].second.ref_start);
            assert(false);
        }
    }
}

__device__ void sort_hits_single(
        my_vector<my_pair<int, Hit>>& hits_per_ref
) {
    quick_sort(&(hits_per_ref[0]), hits_per_ref.size());
}

#define key_mod_val 29

__device__ int find_ref_ids(int ref_id, int* head, ref_ids_edge* edges) {
    int key = ref_id % key_mod_val;
    for (int i = head[key]; i != -1; i = edges[i].pre) {
        if (edges[i].ref_id == ref_id) return i;
    }
    return -1;
}

__device__ size_t indirect_lower_bound(
        const my_vector<my_pair<int, Hit>>& original_hits,
        const int* sorted_indices,
        size_t global_range_start_idx,
        size_t count,
        int value
) {
    size_t start = 0;
    while (count > 0) {
        size_t step = count / 2;
        size_t it = start + step;
        int original_idx = sorted_indices[global_range_start_idx + it];
        if (original_hits.data[original_idx].second.ref_start < value) {
            start = it + 1;
            count -= step + 1;
        } else {
            count = step;
        }
    }
    return start; // Returns a relative index [0, count)
}


__device__ void salign_merge_hits_seg(
        const my_vector<my_pair<int, Hit>>& original_hits,
        const int* sorted_indices,
        int task_start_offset,
        int task_end_offset,
        int k,
        bool is_revcomp,
        my_vector<Nam>& nams
) {
    int num_hits_in_task = task_end_offset - task_start_offset;
    assert(num_hits_in_task == original_hits.size());
    if (num_hits_in_task == 0) return;

    int ref_num = 0;
    my_vector<int> each_ref_size(8);

    int first_hit_original_idx = sorted_indices[task_start_offset];
    int pre_ref_id = original_hits.data[first_hit_original_idx].first;
    int now_ref_num = 1;

    for (int i = 1; i < num_hits_in_task; i++) {
        int global_idx = task_start_offset + i;
        int original_idx = sorted_indices[global_idx];
        int ref_id = original_hits.data[original_idx].first;

        if (ref_id != pre_ref_id) {
            ref_num++;
            pre_ref_id = ref_id;
            each_ref_size.push_back(now_ref_num);
            now_ref_num = 1;
        } else {
            now_ref_num++;
        }
    }
    ref_num++;
    each_ref_size.push_back(now_ref_num);

    // --- Step 2: Apply the original `salign_merge_hits` algorithm logic. ---
    my_vector<Nam> open_nams;
    my_vector<bool> is_added(32);
    int now_vec_pos = 0; // Local offset within this task's hits [0, num_hits_in_task)

    for (int rid = 0; rid < ref_num; rid++) {
        if(rid != 0) now_vec_pos += each_ref_size[rid - 1];

        int first_hit_global_idx = task_start_offset + now_vec_pos;
        int first_hit_original_idx = sorted_indices[first_hit_global_idx];
        int ref_id = original_hits.data[first_hit_original_idx].first;

        open_nams.clear();
        unsigned int prev_q_start = 0;
        size_t hits_size = each_ref_size[rid];

        for (size_t i = 0; i < hits_size; ) {
            int i_start_original_idx = sorted_indices[task_start_offset + now_vec_pos + i];
            int current_query_start = original_hits.data[i_start_original_idx].second.query_start;

            size_t i_start = i;
            size_t i_end = i + 1;
            while(i_end < hits_size) {
                int next_hit_original_idx = sorted_indices[task_start_offset + now_vec_pos + i_end];
                if (original_hits.data[next_hit_original_idx].second.query_start == current_query_start) {
                    i_end++;
                } else {
                    break;
                }
            }
            i = i_end;
            size_t i_size = i_end - i_start;

            is_added.clear();
            for(size_t j = 0; j < i_size; j++) is_added.push_back(false);

            int query_start = current_query_start;
            int cnt_done = 0;
            for (int k = 0; k < open_nams.size(); k++) {
                Nam& o = open_nams[k];
                if ( query_start > o.query_end ) continue;

                size_t global_range_start_idx = task_start_offset + now_vec_pos + i_start;
                size_t range_count = i_end - i_start;

                size_t lower_rel_idx = indirect_lower_bound(original_hits, sorted_indices, global_range_start_idx, range_count, o.ref_prev_hit_startpos + 1);
                size_t upper_rel_idx = indirect_lower_bound(original_hits, sorted_indices, global_range_start_idx, range_count, o.ref_end + 1);

                for (size_t j_rel = lower_rel_idx; j_rel < upper_rel_idx; j_rel++) {
                    size_t j = i_start + j_rel; // Convert relative index to local index
                    if(is_added[j - i_start]) continue;

                    int hit_original_idx = sorted_indices[task_start_offset + now_vec_pos + j];
                    const Hit& h = original_hits.data[hit_original_idx].second;

                    if (o.ref_prev_hit_startpos < h.ref_start && h.ref_start <= o.ref_end) {
                        if ((h.query_end > o.query_end) && (h.ref_end > o.ref_end)) {
                            o.query_end = h.query_end;
                            o.ref_end = h.ref_end;
                            o.query_prev_hit_startpos = h.query_start;
                            o.ref_prev_hit_startpos = h.ref_start;
                            o.n_hits++;
                            is_added[j - i_start] = true;
                            cnt_done++;
                            break;
                        } else if ((h.query_end <= o.query_end) && (h.ref_end <= o.ref_end)) {
                            o.query_prev_hit_startpos = h.query_start;
                            o.ref_prev_hit_startpos = h.ref_start;
                            o.n_hits++;
                            is_added[j - i_start] = true;
                            cnt_done++;
                            break;
                        }
                    }
                }
                if(cnt_done == i_size) break;
            }

            // Add new NAMs for hits not added to existing ones.
            for(size_t j = 0; j < i_size; j++) {
                if (!is_added[j]){
                    int hit_original_idx = sorted_indices[task_start_offset + now_vec_pos + i_start + j];
                    const Hit& h = original_hits.data[hit_original_idx].second;
                    Nam n;
                    n.query_start = h.query_start;
                    n.query_end = h.query_end;
                    n.ref_start = h.ref_start;
                    n.ref_end = h.ref_end;
                    n.ref_id = ref_id;
                    n.query_prev_hit_startpos = h.query_start;
                    n.ref_prev_hit_startpos = h.ref_start;
                    n.n_hits = 1;
                    n.is_rc = is_revcomp;
                    open_nams.push_back(n);
                }
            }


            // Only filter if we have advanced at least k nucleotides
            if (query_start > prev_q_start + k) {

                // Output all NAMs from open_matches to final_nams that the current hit have passed
                for (int k = 0; k < open_nams.size(); k++) {
                    Nam& n = open_nams[k];
                    if (n.query_end < query_start) {
                        int n_max_span = my_max(n.query_span(), n.ref_span());
                        int n_min_span = my_min(n.query_span(), n.ref_span());
                        float n_score;
                        n_score = ( 2*n_min_span -  n_max_span) > 0 ? (float) (n.n_hits * ( 2*n_min_span -  n_max_span) ) : 1;   // this is really just n_hits * ( min_span - (offset_in_span) ) );
                        //                        n_score = n.n_hits * n.query_span();
                        n.score = n_score;
                        n.nam_id = nams.size();
                        nams.push_back(n);
                    }
                }

                // Remove all NAMs from open_matches that the current hit have passed
                auto c = query_start;
                int old_open_size = open_nams.size();
                open_nams.clear();
                for (int in = 0; in < old_open_size; ++in) {
                    if (!(open_nams[in].query_end < c)) {
                        open_nams.push_back(open_nams[in]);
                    }
                }
                prev_q_start = query_start;
            }
        }
        // Add all current open_matches to final NAMs
        for (int k = 0; k < open_nams.size(); k++) {
            Nam& n = open_nams[k];
            int n_max_span = my_max(n.query_span(), n.ref_span());
            int n_min_span = my_min(n.query_span(), n.ref_span());
            float n_score;
            n_score = ( 2*n_min_span -  n_max_span) > 0 ? (float) (n.n_hits * ( 2*n_min_span -  n_max_span) ) : 1;   // this is really just n_hits * ( min_span - (offset_in_span) ) );
            //            n_score = n.n_hits * n.query_span();
            n.score = n_score;
            n.nam_id = nams.size();
            nams.push_back(n);
        }
    }
}


__device__ void salign_merge_hits(
        my_vector<my_pair<int, Hit>>& hits_per_ref,
        int k,
        bool is_revcomp,
        my_vector<Nam>& nams
) {
    if(hits_per_ref.size() == 0) return;
    int ref_num = 0;
    my_vector<int> each_ref_size;
    int pre_ref_id = hits_per_ref[0].first;
    int now_ref_num = 1;
    for(int i = 1; i < hits_per_ref.size(); i++) {
        int ref_id = hits_per_ref[i].first;
        Hit hit = hits_per_ref[i].second;
        if(ref_id != pre_ref_id) {
            ref_num++;
            pre_ref_id = ref_id;
            each_ref_size.push_back(now_ref_num);
            now_ref_num = 1;
        } else {
            now_ref_num++;
        }
    }
    ref_num++;
    each_ref_size.push_back(now_ref_num);

    my_vector<Nam> open_nams;
    my_vector<bool> is_added(32);
    int now_vec_pos = 0;
    for (int rid = 0; rid < ref_num; rid++) {
        if(rid != 0) now_vec_pos += each_ref_size[rid - 1];
        int ref_id = hits_per_ref[now_vec_pos].first;
        open_nams.clear();
        unsigned int prev_q_start = 0;
        size_t hits_size = each_ref_size[rid];
        my_pair<int, Hit>* hits = &(hits_per_ref[now_vec_pos]);
        for (size_t i = 0; i < hits_size; ) {
            size_t i_start = i;
            size_t i_end = i + 1;
            size_t i_size;
            while(i_end < hits_size && hits[i_end].second.query_start == hits[i].second.query_start) i_end++;
            i = i_end;
            i_size = i_end - i_start;
            //for(int j = 0; j < i_size - 1; j++) {
            //    assert(hits[i_start + j].second.ref_start <= hits[i_start + j + 1].second.ref_start);
            //}
            //quick_sort(&(hits[i_start]), i_size);
            is_added.clear();
            for(size_t j = 0; j < i_size; j++) is_added.push_back(false);
            int query_start = hits[i_start].second.query_start;
            int cnt_done = 0;
            for (int k = 0; k < open_nams.size(); k++) {
                Nam& o = open_nams[k];
                if ( query_start > o.query_end ) continue;
                size_t lower = my_lower_bound(hits, i_start, i_end, o.ref_prev_hit_startpos + 1);
                size_t upper = my_lower_bound(hits, i_start, i_end, o.ref_end + 1);
                for (size_t j = lower; j < upper; j++) {
                    if(is_added[j - i_start]) continue;
                    Hit& h = hits[j].second;
                    {
                        if (o.ref_prev_hit_startpos < h.ref_start && h.ref_start <= o.ref_end) {
                            if ((h.query_end > o.query_end) && (h.ref_end > o.ref_end)) {
                                o.query_end = h.query_end;
                                o.ref_end = h.ref_end;
                                //                        o.previous_query_start = h.query_s;
                                //                        o.previous_ref_start = h.ref_s; // keeping track so that we don't . Can be caused by interleaved repeats.
                                o.query_prev_hit_startpos = h.query_start;
                                o.ref_prev_hit_startpos = h.ref_start;
                                o.n_hits++;
                                //                        o.score += (float)1/ (float)h.count;
                                is_added[j - i_start] = true;
                                cnt_done++;
                                break;
                            } else if ((h.query_end <= o.query_end) && (h.ref_end <= o.ref_end)) {
                                //                        o.previous_query_start = h.query_s;
                                //                        o.previous_ref_start = h.ref_s; // keeping track so that we don't . Can be caused by interleaved repeats.
                                o.query_prev_hit_startpos = h.query_start;
                                o.ref_prev_hit_startpos = h.ref_start;
                                o.n_hits++;
                                //                        o.score += (float)1/ (float)h.count;
                                is_added[j - i_start] = true;
                                cnt_done++;
                                break;
                            }
                        }
                    }
                }
                if(cnt_done == i_size) break;
            }

            // Add the hit to open matches
            for(size_t j = 0; j < i_size; j++) {
                if (!is_added[j]){
                    Nam n;
                    n.query_start = hits[i_start + j].second.query_start;
                    n.query_end = hits[i_start + j].second.query_end;
                    n.ref_start = hits[i_start + j].second.ref_start;
                    n.ref_end = hits[i_start + j].second.ref_end;
                    n.ref_id = ref_id;
                    //                n.previous_query_start = h.query_s;
                    //                n.previous_ref_start = h.ref_s;
                    n.query_prev_hit_startpos = hits[i_start + j].second.query_start;
                    n.ref_prev_hit_startpos = hits[i_start + j].second.ref_start;
                    n.n_hits = 1;
                    n.is_rc = is_revcomp;
                    //                n.score += (float)1 / (float)h.count;
                    open_nams.push_back(n);
                }
            }

            // Only filter if we have advanced at least k nucleotides
            if (query_start > prev_q_start + k) {

                // Output all NAMs from open_matches to final_nams that the current hit have passed
                for (int k = 0; k < open_nams.size(); k++) {
                    Nam& n = open_nams[k];
                    if (n.query_end < query_start) {
                        int n_max_span = my_max(n.query_span(), n.ref_span());
                        int n_min_span = my_min(n.query_span(), n.ref_span());
                        float n_score;
                        n_score = ( 2*n_min_span -  n_max_span) > 0 ? (float) (n.n_hits * ( 2*n_min_span -  n_max_span) ) : 1;   // this is really just n_hits * ( min_span - (offset_in_span) ) );
                        //                        n_score = n.n_hits * n.query_span();
                        n.score = n_score;
                        n.nam_id = nams.size();
                        nams.push_back(n);
                    }
                }

                // Remove all NAMs from open_matches that the current hit have passed
                auto c = query_start;
                int old_open_size = open_nams.size();
                open_nams.clear();
                for (int in = 0; in < old_open_size; ++in) {
                    if (!(open_nams[in].query_end < c)) {
                        open_nams.push_back(open_nams[in]);
                    }
                }
                prev_q_start = query_start;
            }
        }
        // Add all current open_matches to final NAMs
        for (int k = 0; k < open_nams.size(); k++) {
            Nam& n = open_nams[k];
            int n_max_span = my_max(n.query_span(), n.ref_span());
            int n_min_span = my_min(n.query_span(), n.ref_span());
            float n_score;
            n_score = ( 2*n_min_span -  n_max_span) > 0 ? (float) (n.n_hits * ( 2*n_min_span -  n_max_span) ) : 1;   // this is really just n_hits * ( min_span - (offset_in_span) ) );
            //            n_score = n.n_hits * n.query_span();
            n.score = n_score;
            n.nam_id = nams.size();
            nams.push_back(n);
        }
    }
}

__device__ void merge_hits_seg(
        const my_vector<my_pair<int, Hit>>& original_hits,
        const int* sorted_indices,
        int task_start_offset,
        int task_end_offset,
        int k,
        bool is_revcomp,
        my_vector<Nam>& nams
) {
    int num_hits_in_task = task_end_offset - task_start_offset;
    assert(num_hits_in_task == original_hits.size());
    if (num_hits_in_task == 0) return;

    // --- Step 1: Group hits by ref_id (same logic as before, but with indirect access) ---
    int ref_num = 0;
    my_vector<int> each_ref_size(8);

    // Get the first hit to initialize the grouping
    int first_hit_original_idx = sorted_indices[task_start_offset];
    int pre_ref_id = original_hits.data[first_hit_original_idx].first;
    int now_ref_num = 1;

    for (int i = 1; i < num_hits_in_task; i++) {
        int global_idx = task_start_offset + i;
        int original_idx = sorted_indices[global_idx];
        int ref_id = original_hits.data[original_idx].first;

        if (ref_id != pre_ref_id) {
            ref_num++;
            pre_ref_id = ref_id;
            each_ref_size.push_back(now_ref_num);
            now_ref_num = 1;
        } else {
            now_ref_num++;
        }
    }
    ref_num++;
    each_ref_size.push_back(now_ref_num);

    // --- Step 2: Iterate through groups and merge hits into NAMs ---
    my_vector<Nam> open_nams;
    int now_vec_pos = 0; // This is a local offset within this task's hits [0, num_hits_in_task)

    for (int i = 0; i < ref_num; i++) {
        if (i != 0) now_vec_pos += each_ref_size[i - 1];

        // Get ref_id for the current group
        int first_hit_global_idx = task_start_offset + now_vec_pos;
        int first_hit_original_idx = sorted_indices[first_hit_global_idx];
        int ref_id = original_hits.data[first_hit_original_idx].first;

        open_nams.clear();
        unsigned int prev_q_start = 0;

        for (int j = 0; j < each_ref_size[i]; j++) {
            // Indirectly access the Hit object in sorted order
            int current_hit_local_idx_in_task = now_vec_pos + j;
            int current_hit_global_idx = task_start_offset + current_hit_local_idx_in_task;
            int original_idx = sorted_indices[current_hit_global_idx];
            const Hit& h = original_hits.data[original_idx].second;
            bool is_added = false;

            for (int k = 0; k < open_nams.size(); k++) {
                Nam& o = open_nams[k];

                // Extend NAM
                if ((o.query_prev_hit_startpos < h.query_start) && (h.query_start <= o.query_end ) && (o.ref_prev_hit_startpos < h.ref_start) && (h.ref_start <= o.ref_end) ){
                    if ( (h.query_end > o.query_end) && (h.ref_end > o.ref_end) ) {
                        o.query_end = h.query_end;
                        o.ref_end = h.ref_end;
                        //                        o.previous_query_start = h.query_s;
                        //                        o.previous_ref_start = h.ref_s; // keeping track so that we don't . Can be caused by interleaved repeats.
                        o.query_prev_hit_startpos = h.query_start;
                        o.ref_prev_hit_startpos = h.ref_start;
                        o.n_hits ++;
                        //                        o.score += (float)1/ (float)h.count;
                        is_added = true;
                        break;
                    }
                    else if ((h.query_end <= o.query_end) && (h.ref_end <= o.ref_end)) {
                        //                        o.previous_query_start = h.query_s;
                        //                        o.previous_ref_start = h.ref_s; // keeping track so that we don't . Can be caused by interleaved repeats.
                        o.query_prev_hit_startpos = h.query_start;
                        o.ref_prev_hit_startpos = h.ref_start;
                        o.n_hits ++;
                        //                        o.score += (float)1/ (float)h.count;
                        is_added = true;
                        break;
                    }
                }

            }

            // Add the hit to open matches
            if (!is_added){
                Nam n;
                n.query_start = h.query_start;
                n.query_end = h.query_end;
                n.ref_start = h.ref_start;
                n.ref_end = h.ref_end;
                n.ref_id = ref_id;
                //                n.previous_query_start = h.query_s;
                //                n.previous_ref_start = h.ref_s;
                n.query_prev_hit_startpos = h.query_start;
                n.ref_prev_hit_startpos = h.ref_start;
                n.n_hits = 1;
                n.is_rc = is_revcomp;
                //                n.score += (float)1 / (float)h.count;
                open_nams.push_back(n);
            }

            // Only filter if we have advanced at least k nucleotides
            if (h.query_start > prev_q_start + k) {
                // Output all NAMs from open_matches to final_nams that the current hit have passed
                for (int k = 0; k < open_nams.size(); k++) {
                    Nam& n = open_nams[k];
                    if (n.query_end < h.query_start) {
                        int n_max_span = my_max(n.query_span(), n.ref_span());
                        int n_min_span = my_min(n.query_span(), n.ref_span());
                        float n_score;
                        n_score = ( 2*n_min_span -  n_max_span) > 0 ? (float) (n.n_hits * ( 2*n_min_span -  n_max_span) ) : 1;   // this is really just n_hits * ( min_span - (offset_in_span) ) );
                        //                        n_score = n.n_hits * n.query_span();
                        n.score = n_score;
                        n.nam_id = nams.size();
                        nams.push_back(n);
                    }
                }

                // Remove all NAMs from open_matches that the current hit have passed
                auto c = h.query_start;
                int old_open_size = open_nams.size();
                open_nams.clear();
                for (int in = 0; in < old_open_size; ++in) {
                    if (!(open_nams[in].query_end < c)) {
                        open_nams.push_back(open_nams[in]);
                    }
                }
                prev_q_start = h.query_start;
            }
        }

        // Add all current open_matches to final NAMs
        for (int k = 0; k < open_nams.size(); k++) {
            Nam& n = open_nams[k];
            int n_max_span = my_max(n.query_span(), n.ref_span());
            int n_min_span = my_min(n.query_span(), n.ref_span());
            float n_score;
            n_score = ( 2*n_min_span -  n_max_span) > 0 ? (float) (n.n_hits * ( 2*n_min_span -  n_max_span) ) : 1;   // this is really just n_hits * ( min_span - (offset_in_span) ) );
            //            n_score = n.n_hits * n.query_span();
            n.score = n_score;
            n.nam_id = nams.size();
            nams.push_back(n);
        }
    }
}

__device__ void merge_hits(
        my_vector<my_pair<int, Hit>>& hits_per_ref,
        int k,
        bool is_revcomp,
        my_vector<Nam>& nams
) {
    if(hits_per_ref.size() == 0) return;
    int num_hits = hits_per_ref.size();

    int ref_num = 0;
    my_vector<int> each_ref_size(8);
    int pre_ref_id = hits_per_ref[0].first;
    int now_ref_num = 1;
    for(int i = 1; i < hits_per_ref.size(); i++) {
        int ref_id = hits_per_ref[i].first;
        Hit hit = hits_per_ref[i].second;
        if(ref_id != pre_ref_id) {
//            assert(ref_id > pre_ref_id);
            ref_num++;
            pre_ref_id = ref_id;
            each_ref_size.push_back(now_ref_num);
            now_ref_num = 1;
        } else {
            now_ref_num++;
        }
    }
    ref_num++;
    each_ref_size.push_back(now_ref_num);

    my_vector<Nam> open_nams;

    int now_vec_pos = 0;
    for (int i = 0; i < ref_num; i++) {

        if(i != 0) now_vec_pos += each_ref_size[i - 1];
        int ref_id = hits_per_ref[now_vec_pos].first;
        open_nams.clear();
        unsigned int prev_q_start = 0;

        for (int j = 0; j < each_ref_size[i]; j++) {
            Hit& h = hits_per_ref[now_vec_pos + j].second;
            bool is_added = false;
            for (int k = 0; k < open_nams.size(); k++) {
                Nam& o = open_nams[k];

                // Extend NAM
                if ((o.query_prev_hit_startpos < h.query_start) && (h.query_start <= o.query_end ) && (o.ref_prev_hit_startpos < h.ref_start) && (h.ref_start <= o.ref_end) ){
                    if ( (h.query_end > o.query_end) && (h.ref_end > o.ref_end) ) {
                        o.query_end = h.query_end;
                        o.ref_end = h.ref_end;
                        //                        o.previous_query_start = h.query_s;
                        //                        o.previous_ref_start = h.ref_s; // keeping track so that we don't . Can be caused by interleaved repeats.
                        o.query_prev_hit_startpos = h.query_start;
                        o.ref_prev_hit_startpos = h.ref_start;
                        o.n_hits ++;
                        //                        o.score += (float)1/ (float)h.count;
                        is_added = true;
                        break;
                    }
                    else if ((h.query_end <= o.query_end) && (h.ref_end <= o.ref_end)) {
                        //                        o.previous_query_start = h.query_s;
                        //                        o.previous_ref_start = h.ref_s; // keeping track so that we don't . Can be caused by interleaved repeats.
                        o.query_prev_hit_startpos = h.query_start;
                        o.ref_prev_hit_startpos = h.ref_start;
                        o.n_hits ++;
                        //                        o.score += (float)1/ (float)h.count;
                        is_added = true;
                        break;
                    }
                }

            }

            // Add the hit to open matches
            if (!is_added){
                Nam n;
                n.query_start = h.query_start;
                n.query_end = h.query_end;
                n.ref_start = h.ref_start;
                n.ref_end = h.ref_end;
                n.ref_id = ref_id;
                //                n.previous_query_start = h.query_s;
                //                n.previous_ref_start = h.ref_s;
                n.query_prev_hit_startpos = h.query_start;
                n.ref_prev_hit_startpos = h.ref_start;
                n.n_hits = 1;
                n.is_rc = is_revcomp;
                //                n.score += (float)1 / (float)h.count;
                open_nams.push_back(n);
            }

            // Only filter if we have advanced at least k nucleotides
            if (h.query_start > prev_q_start + k) {
                // Output all NAMs from open_matches to final_nams that the current hit have passed
                for (int k = 0; k < open_nams.size(); k++) {
                    Nam& n = open_nams[k];
                    if (n.query_end < h.query_start) {
                        int n_max_span = my_max(n.query_span(), n.ref_span());
                        int n_min_span = my_min(n.query_span(), n.ref_span());
                        float n_score;
                        n_score = ( 2*n_min_span -  n_max_span) > 0 ? (float) (n.n_hits * ( 2*n_min_span -  n_max_span) ) : 1;   // this is really just n_hits * ( min_span - (offset_in_span) ) );
                        //                        n_score = n.n_hits * n.query_span();
                        n.score = n_score;
                        n.nam_id = nams.size();
                        nams.push_back(n);
                    }
                }

                // Remove all NAMs from open_matches that the current hit have passed
                auto c = h.query_start;
                int old_open_size = open_nams.size();
                open_nams.clear();
                for (int in = 0; in < old_open_size; ++in) {
                    if (!(open_nams[in].query_end < c)) {
                        open_nams.push_back(open_nams[in]);
                    }
                }
                prev_q_start = h.query_start;
            }
        }

        // Add all current open_matches to final NAMs
        for (int k = 0; k < open_nams.size(); k++) {
            Nam& n = open_nams[k];
            int n_max_span = my_max(n.query_span(), n.ref_span());
            int n_min_span = my_min(n.query_span(), n.ref_span());
            float n_score;
            n_score = ( 2*n_min_span -  n_max_span) > 0 ? (float) (n.n_hits * ( 2*n_min_span -  n_max_span) ) : 1;   // this is really just n_hits * ( min_span - (offset_in_span) ) );
            //            n_score = n.n_hits * n.query_span();
            n.score = n_score;
            n.nam_id = nams.size();
            nams.push_back(n);
        }
    }
}


__device__ void gpu_shuffle_top_nams(my_vector<Nam>& nams) {
#ifdef GPU_ACC_TAG
    return;
#endif
    unsigned int seed = 1234567u;
    if (nams.empty()) {
        return;
    }
    auto best_score = nams[0].score;
    int top_cnt = 1;
    while (top_cnt < nams.size() && nams[top_cnt].score == best_score)
        ++top_cnt;
    auto next_rand = [&seed]() {
        seed = seed * 1664525u + 1013904223u;
        return seed;
    };

    for (int i = top_cnt - 1; i > 0; --i) {
        unsigned int j = next_rand() % (i + 1);
        Nam tmp = nams[i];
        nams[i] = nams[j];
        nams[j] = tmp;
    }
}


__device__ void sort_hits_by_refid(
        my_vector<my_pair<int, Hit>>& hits_per_ref
) {

    int *head = (int*)my_malloc(key_mod_val * sizeof(int));
    my_vector<ref_ids_edge> edges;
    for(int i = 0; i < key_mod_val; i++) head[i] = -1;
    int ref_ids_num = 0;
    for(int i = 0; i < hits_per_ref.size(); i++) {
        int ref_id = hits_per_ref[i].first;
        int find_ref_id_rank = find_ref_ids(ref_id, head, edges.data);
        if (find_ref_id_rank == -1) {
            find_ref_id_rank = ref_ids_num;
            int key = ref_id % key_mod_val;
            edges.push_back({head[key], ref_id});
            head[key] = ref_ids_num++;
        }
    }
    if (ref_ids_num <= 1) {
        my_free(head);
        return;
    }
    my_vector<my_pair<int, my_vector<Hit>*>> all_hits(ref_ids_num);
    all_hits.length = ref_ids_num;
    my_vector<Hit>* all_vecs = (my_vector<Hit>*)my_malloc(ref_ids_num * sizeof(my_vector<Hit>));
    for (int i = 0; i < ref_ids_num; i++) {
        all_hits[i].first = -1;
        all_hits[i].second = &all_vecs[i];
        all_hits[i].second->init();
    }
    for (int i = 0; i < hits_per_ref.size(); i++) {
        int ref_id = hits_per_ref[i].first;
        int find_ref_id_rank = find_ref_ids(ref_id, head, edges.data);
        assert(find_ref_id_rank >= 0 && find_ref_id_rank < ref_ids_num);
        all_hits[find_ref_id_rank].first = ref_id;
        all_hits[find_ref_id_rank].second->push_back(hits_per_ref[i].second);
    }
    hits_per_ref.clear();
    //quick_sort_iterative(&(all_hits[0]), 0, all_hits.size() - 1,
    //                     [](const my_pair<int, my_vector<Hit>*>& a, const my_pair<int, my_vector<Hit>*>& b) {
    //                         return a.first < b.first;
    //                     });
    for(int i = 0; i < all_hits.size(); i++) {
        for(int j = 0; j < all_hits[i].second->size(); j++) {
            hits_per_ref.push_back({all_hits[i].first, (*all_hits[i].second)[j]});
        }
        all_hits[i].second->release();
    }
    my_free(head);
    my_free(all_vecs);
}

__device__ void sort_nams_single_check(
        my_vector<Nam>& nams
) {
    //bubble_sort(&(hits_per_ref[0]), hits_per_ref.size());
    quick_sort_iterative(&(nams[0]), 0, nams.size() - 1, [](const Nam &n1, const Nam &n2) {
        if(n1.score != n2.score) return n1.score > n2.score;
        if(n1.n_hits != n2.n_hits) return n1.n_hits > n2.n_hits;
        if(n1.query_end != n2.query_end) return n1.query_end < n2.query_end;
        if(n1.query_start != n2.query_start) return n1.query_start < n2.query_start;
        if(n1.ref_end != n2.ref_end) return n1.ref_end < n2.ref_end;
        if(n1.ref_start != n2.ref_start) return n1.ref_start < n2.ref_start;
        if(n1.ref_id != n2.ref_id) return n1.ref_id < n2.ref_id;
        return n1.is_rc < n2.is_rc;
    });
}

__device__ void sort_nam_pairs_by_score(my_vector<gpu_NamPair>& joint_nam_scores, int mx_num) {
    int* head = (int*)my_malloc(key_mod_val * sizeof(int));
    my_vector<ref_ids_edge> edges;
    for (int i = 0; i < key_mod_val; i++) head[i] = -1;
    int score_group_num = 0;
    for (int i = 0; i < joint_nam_scores.size(); i++) {
        int score_key = (int)(joint_nam_scores[i].score);
        int score_rank = find_ref_ids(score_key, head, edges.data);
        if (score_rank == -1) {
            score_rank = score_group_num;
            int key = score_key % key_mod_val;
            edges.push_back({head[key], score_key});
            head[key] = score_group_num++;
        }
    }
    if (score_group_num <= 1) {
        my_free(head);
        return;
    }
    my_vector<my_pair<int, my_vector<gpu_NamPair>*>> all_nams(score_group_num);
    all_nams.length = score_group_num;
    my_vector<gpu_NamPair>* all_vecs = (my_vector<gpu_NamPair>*)my_malloc(score_group_num * sizeof(my_vector<gpu_NamPair>));
    for (int i = 0; i < score_group_num; i++) {
        all_nams[i].first = -1;
        all_nams[i].second = &all_vecs[i];
        all_nams[i].second->init();
    }
    for (int i = 0; i < joint_nam_scores.size(); i++) {
        int score_key = (int)(joint_nam_scores[i].score);
        int score_rank = find_ref_ids(score_key, head, edges.data);
        assert(score_rank >= 0 && score_rank < score_group_num);
        all_nams[score_rank].first = score_key;
        all_nams[score_rank].second->push_back(joint_nam_scores[i]);
    }
    joint_nam_scores.clear();
    quick_sort_iterative(&(all_nams[0]), 0, all_nams.size() - 1,
                         [](const my_pair<int, my_vector<gpu_NamPair>*>& a, const my_pair<int, my_vector<gpu_NamPair>*>& b) {
                             return a.first > b.first;
                         });
    for (int i = 0; i < all_nams.size(); i++) {
        for (int j = 0; j < all_nams[i].second->size(); j++) {
            if (joint_nam_scores.size() == mx_num) break;
            joint_nam_scores.push_back((*all_nams[i].second)[j]);
        }
        all_nams[i].second->release();
    }
    my_free(head);
    my_free(all_vecs);
}


__device__ void sort_nams_by_score(my_vector<Nam>& nams, int mx_num) {
    int* head = (int*)my_malloc(key_mod_val * sizeof(int));
    my_vector<ref_ids_edge> edges;
    for (int i = 0; i < key_mod_val; i++) head[i] = -1;
    int score_group_num = 0;
    for (int i = 0; i < nams.size(); i++) {
        int score_key = (int)(nams[i].score);
        int score_rank = find_ref_ids(score_key, head, edges.data);
        if (score_rank == -1) {
            score_rank = score_group_num;
            int key = score_key % key_mod_val;
            edges.push_back({head[key], score_key});
            head[key] = score_group_num++;
        }
    }
    if (score_group_num <= 1) {
        my_free(head);
        return;
    }
    my_vector<my_pair<int, my_vector<Nam>*>> all_nams(score_group_num);
    all_nams.length = score_group_num;
    my_vector<Nam>* all_vecs = (my_vector<Nam>*)my_malloc(score_group_num * sizeof(my_vector<Nam>));
    for (int i = 0; i < score_group_num; i++) {
        all_nams[i].first = -1;
        all_nams[i].second = &all_vecs[i];
        all_nams[i].second->init();
    }
    for (int i = 0; i < nams.size(); i++) {
        int score_key = (int)(nams[i].score);
        int score_rank = find_ref_ids(score_key, head, edges.data);
        assert(score_rank >= 0 && score_rank < score_group_num);
        all_nams[score_rank].first = score_key;
        all_nams[score_rank].second->push_back(nams[i]);
    }
    nams.clear();
    quick_sort_iterative(&(all_nams[0]), 0, all_nams.size() - 1,
                         [](const my_pair<int, my_vector<Nam>*>& a, const my_pair<int, my_vector<Nam>*>& b) {
                             return a.first > b.first;
                         });
    for (int i = 0; i < all_nams.size(); i++) {
        for (int j = 0; j < all_nams[i].second->size(); j++) {
            if (nams.size() == mx_num) break;
            nams.push_back((*all_nams[i].second)[j]);
        }
        all_nams[i].second->release();
    }
    my_free(head);
    my_free(all_vecs);
}

__device__ void sort_nams_by_hits(my_vector<Nam>& nams, int mx_num) {
    int* head = (int*)my_malloc(key_mod_val * sizeof(int));
    my_vector<ref_ids_edge> edges;
    for (int i = 0; i < key_mod_val; i++) head[i] = -1;
    int score_group_num = 0;
    for (int i = 0; i < nams.size(); i++) {
        int score_key = (int)(nams[i].n_hits);
        int score_rank = find_ref_ids(score_key, head, edges.data);
        if (score_rank == -1) {
            score_rank = score_group_num;
            int key = score_key % key_mod_val;
            edges.push_back({head[key], score_key});
            head[key] = score_group_num++;
        }
    }
    if (score_group_num <= 1) {
        my_free(head);
        return;
    }
    my_vector<my_pair<int, my_vector<Nam>*>> all_nams(score_group_num);
    all_nams.length = score_group_num;
    my_vector<Nam>* all_vecs = (my_vector<Nam>*)my_malloc(score_group_num * sizeof(my_vector<Nam>));
    for (int i = 0; i < score_group_num; i++) {
        all_nams[i].first = -1;
        all_nams[i].second = &all_vecs[i];
        all_nams[i].second->init();
    }
    for (int i = 0; i < nams.size(); i++) {
        int score_key = (int)(nams[i].n_hits);
        int score_rank = find_ref_ids(score_key, head, edges.data);
        assert(score_rank >= 0 && score_rank < score_group_num);
        all_nams[score_rank].first = score_key;
        all_nams[score_rank].second->push_back(nams[i]);
    }
    nams.clear();
    quick_sort_iterative(&(all_nams[0]), 0, all_nams.size() - 1,
                         [](const my_pair<int, my_vector<Nam>*>& a, const my_pair<int, my_vector<Nam>*>& b) {
                             return a.first > b.first;
                         });
    for (int i = 0; i < all_nams.size(); i++) {
        for (int j = 0; j < all_nams[i].second->size(); j++) {
            if (nams.size() == mx_num) break;
            nams.push_back((*all_nams[i].second)[j]);
        }
        all_nams[i].second->release();
    }
    my_free(head);
    my_free(all_vecs);
}

__global__ void get_task_sizes_kernel(
        int num_tasks,
        const my_vector<my_pair<int, Hit>>* all_task_vectors,
        const int* global_todo_ids,
        int* out_task_sizes)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id < num_tasks) {
        int real_id = global_todo_ids[global_id];
        out_task_sizes[global_id] = all_task_vectors[real_id].size();
    }
}

__global__ void marshal_data_for_sort_kernel(
        int num_tasks,
        const my_vector<my_pair<int, Hit>>* all_task_vectors,
        const int* global_todo_ids,
        const int* task_offsets, // Calculated by prefix sum of sizes
        int* out_keys,           // Destination for ref_ids
        int* out_values)         // Destination for original indices
{
    int task_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (task_id < num_tasks) {
        int real_id = global_todo_ids[task_id];
        const my_vector<my_pair<int, Hit>>& current_vector = all_task_vectors[real_id];
        int start_offset = task_offsets[task_id];

        for (int i = 0; i < current_vector.size(); ++i) {
            out_keys[start_offset + i]   = current_vector.data[i].first; // The ref_id
            out_values[start_offset + i] = i;                           // The original index within this task
        }
    }
}

__global__ void reorder_hits_kernel(
        int num_tasks,
        const my_vector<my_pair<int, Hit>>* original_vectors,
        const int* global_todo_ids,
        const int* task_offsets,
        const int* sorted_indices, // This is the d_values array after sorting
        my_pair<int, Hit>* reordered_buffer) // A temporary buffer to write results to
{
    int global_hit_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_hit_idx < task_offsets[num_tasks]) {
        int upper_bound_idx = cub::UpperBound(task_offsets, num_tasks + 1, global_hit_idx);
        int task_id = upper_bound_idx - 1;

        int real_id = global_todo_ids[task_id];
        const my_vector<my_pair<int, Hit>>& source_vector = original_vectors[real_id];

        // The original index of the hit to copy is given by the sorted_indices array.
        int original_hit_local_index = sorted_indices[global_hit_idx];

        // Copy the original pair to the new sorted location in the buffer.
        reordered_buffer[global_hit_idx] = source_vector.data[original_hit_local_index];
    }
}

__global__ void update_vector_pointers_kernel(
        int num_tasks,
        my_vector<my_pair<int, Hit>>* all_task_vectors,
        const int* global_todo_ids,
        const int* task_offsets,
        my_pair<int, Hit>* reordered_buffer)
{
    int task_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (task_id < num_tasks) {
        int real_id = global_todo_ids[task_id];
        my_vector<my_pair<int, Hit>>& current_vector = all_task_vectors[real_id];
        assert(current_vector.length == task_offsets[task_id + 1] - task_offsets[task_id]);
        current_vector.release();
        current_vector.data = reordered_buffer + task_offsets[task_id];
        current_vector.length = task_offsets[task_id + 1] - task_offsets[task_id];
        current_vector.capacity = current_vector.length;
    }
}

my_pair<int*, int*> sort_all_hits_with_cub_radix(
        int todo_cnt,
        my_vector<my_pair<int, Hit>>* hits_per_refs,
        int* global_todo_ids,
        cudaStream_t stream,
        SegSortGpuResources& buffers,
        double *gpu_cost1,
        double *gpu_cost2,
        double *gpu_cost3,
        double *gpu_cost4)
{
    my_pair<int*, int*> res({nullptr, nullptr});
    if (todo_cnt == 0) return res;

    int threads_per_block = 256;
    int blocks_per_grid = (todo_cnt + threads_per_block - 1) / threads_per_block;

    // --- Part 1: Prepare Segment Information with Re-allocation Logic ---
    double t0 = GetTime();

    size_t required_task_sizes_bytes = todo_cnt * sizeof(int);
    if (buffers.task_sizes_bytes < required_task_sizes_bytes) {
        printf("Allocating task sizes buffer: %zu (%zu) bytes\n", required_task_sizes_bytes, buffers.task_sizes_bytes);
        if (buffers.task_sizes_ptr) CUDA_CHECK(cudaFreeAsync(buffers.task_sizes_ptr, stream));
        CUDA_CHECK(cudaMallocAsync(&buffers.task_sizes_ptr, required_task_sizes_bytes * 2, stream));
        buffers.task_sizes_bytes = required_task_sizes_bytes * 2;
    }
    get_task_sizes_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(todo_cnt, hits_per_refs, global_todo_ids, buffers.task_sizes_ptr);

    size_t required_seg_offsets_bytes = (todo_cnt + 1) * sizeof(int);
    if (buffers.seg_offsets_bytes < required_seg_offsets_bytes) {
        printf("Allocating segment offsets buffer: %zu (%zu) bytes\n", required_seg_offsets_bytes, buffers.seg_offsets_bytes);
        if (buffers.seg_offsets_ptr) CUDA_CHECK(cudaFreeAsync(buffers.seg_offsets_ptr, stream));
        CUDA_CHECK(cudaMallocAsync(&buffers.seg_offsets_ptr, required_seg_offsets_bytes * 2, stream));
        buffers.seg_offsets_bytes = required_seg_offsets_bytes * 2;
    }

    void* d_scan_temp_storage = nullptr;
    size_t scan_temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_scan_temp_storage, scan_temp_storage_bytes, buffers.task_sizes_ptr, buffers.seg_offsets_ptr, todo_cnt + 1, stream);

    if (buffers.scan_temp_bytes < scan_temp_storage_bytes) {
        printf("Allocating scan temp storage: %zu (%zu) bytes\n", scan_temp_storage_bytes, buffers.scan_temp_bytes);
        if (buffers.scan_temp_ptr) CUDA_CHECK(cudaFreeAsync(buffers.scan_temp_ptr, stream));
        CUDA_CHECK(cudaMallocAsync(&buffers.scan_temp_ptr, scan_temp_storage_bytes * 2, stream));
        buffers.scan_temp_bytes = scan_temp_storage_bytes * 2;
    }
    cub::DeviceScan::ExclusiveSum(buffers.scan_temp_ptr, scan_temp_storage_bytes, buffers.task_sizes_ptr, buffers.seg_offsets_ptr, todo_cnt + 1, stream);

    int total_hits;
    CUDA_CHECK(cudaMemcpyAsync(&total_hits, buffers.seg_offsets_ptr + todo_cnt, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    *gpu_cost1 += GetTime() - t0;

    // --- Part 2: Marshal Data into Pre-allocated Buffers ---
    t0 = GetTime();

    if (buffers.key_value_capacity < total_hits) {
        fprintf(stderr, "FATAL ERROR: Pre-allocated sort buffer is too small. Required: %d, Allocated: %zu\n", total_hits, buffers.key_value_capacity);
        exit(EXIT_FAILURE);
    }

    marshal_data_for_sort_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(todo_cnt, hits_per_refs, global_todo_ids, buffers.seg_offsets_ptr, buffers.key_ptr, buffers.value_ptr);
    *gpu_cost2 += GetTime() - t0;

    // --- Part 3: Sort with CUB (Replaced with DeviceSegmentedRadixSort) ---
    t0 = GetTime();


    void* d_sort_temp_storage = nullptr;
    size_t sort_temp_storage_bytes = 0;

    cub::DeviceSegmentedRadixSort::SortPairs(
            d_sort_temp_storage, sort_temp_storage_bytes,
            buffers.key_ptr, buffers.key_alt_ptr,         // Keys_in, Keys_out
            buffers.value_ptr, buffers.value_alt_ptr,     // Values_in, Values_out
            total_hits, todo_cnt,
            buffers.seg_offsets_ptr, buffers.seg_offsets_ptr + 1,
            0, sizeof(int) * 8,                           // begin_bit, end_bit for 32-bit int
            stream);

    if (buffers.sort_temp_bytes < sort_temp_storage_bytes) {
        if (buffers.sort_temp_ptr) CUDA_CHECK(cudaFreeAsync(buffers.sort_temp_ptr, stream));
        CUDA_CHECK(cudaMallocAsync(&buffers.sort_temp_ptr, sort_temp_storage_bytes * 2, stream));
        buffers.sort_temp_bytes = sort_temp_storage_bytes * 2;
    }

    cub::DeviceSegmentedRadixSort::SortPairs(
            buffers.sort_temp_ptr, sort_temp_storage_bytes,
            buffers.key_ptr, buffers.key_alt_ptr,         // Keys_in, Keys_out
            buffers.value_ptr, buffers.value_alt_ptr,     // Values_in, Values_out
            total_hits, todo_cnt,
            buffers.seg_offsets_ptr, buffers.seg_offsets_ptr + 1,
            0, sizeof(int) * 8,                           // begin_bit, end_bit
            stream);

    *gpu_cost3 += GetTime() - t0;

    res.first = buffers.seg_offsets_ptr;
    res.second = buffers.value_alt_ptr;
    return res;
}

my_pair<int*, int*> sort_all_hits_with_cub(
        int todo_cnt,
        my_vector<my_pair<int, Hit>>* hits_per_refs,
        int* global_todo_ids,
        cudaStream_t stream,
        SegSortGpuResources& buffers,
        double *gpu_cost1,
        double *gpu_cost2,
        double *gpu_cost3,
        double *gpu_cost4)
{
    my_pair<int*, int*> res({nullptr, nullptr});
    if (todo_cnt == 0) return res;

    int threads_per_block = 256;
    int blocks_per_grid = (todo_cnt + threads_per_block - 1) / threads_per_block;

    // --- Part 1: Prepare Segment Information with Re-allocation Logic ---
    double t0 = GetTime();

    size_t required_task_sizes_bytes = todo_cnt * sizeof(int);
    if (buffers.task_sizes_bytes < required_task_sizes_bytes) {
        printf("Allocating task sizes buffer: %zu (%zu) bytes\n", required_task_sizes_bytes, buffers.task_sizes_bytes);
        if (buffers.task_sizes_ptr) CUDA_CHECK(cudaFreeAsync(buffers.task_sizes_ptr, stream));
        CUDA_CHECK(cudaMallocAsync(&buffers.task_sizes_ptr, required_task_sizes_bytes * 2, stream));
        buffers.task_sizes_bytes = required_task_sizes_bytes * 2;
    }
    get_task_sizes_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(todo_cnt, hits_per_refs, global_todo_ids, buffers.task_sizes_ptr);

    size_t required_seg_offsets_bytes = (todo_cnt + 1) * sizeof(int);
    if (buffers.seg_offsets_bytes < required_seg_offsets_bytes) {
        printf("Allocating segment offsets buffer: %zu (%zu) bytes\n", required_seg_offsets_bytes, buffers.seg_offsets_bytes);
        if (buffers.seg_offsets_ptr) CUDA_CHECK(cudaFreeAsync(buffers.seg_offsets_ptr, stream));
        CUDA_CHECK(cudaMallocAsync(&buffers.seg_offsets_ptr, required_seg_offsets_bytes * 2, stream));
        buffers.seg_offsets_bytes = required_seg_offsets_bytes * 2;
    }

    void* d_scan_temp_storage = nullptr;
    size_t scan_temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_scan_temp_storage, scan_temp_storage_bytes, buffers.task_sizes_ptr, buffers.seg_offsets_ptr, todo_cnt + 1, stream);

    if (buffers.scan_temp_bytes < scan_temp_storage_bytes) {
        printf("Allocating scan temp storage: %zu (%zu) bytes\n", scan_temp_storage_bytes, buffers.scan_temp_bytes);
        if (buffers.scan_temp_ptr) CUDA_CHECK(cudaFreeAsync(buffers.scan_temp_ptr, stream));
        CUDA_CHECK(cudaMallocAsync(&buffers.scan_temp_ptr, scan_temp_storage_bytes * 2, stream));
        buffers.scan_temp_bytes = scan_temp_storage_bytes * 2;
    }
    cub::DeviceScan::ExclusiveSum(buffers.scan_temp_ptr, scan_temp_storage_bytes, buffers.task_sizes_ptr, buffers.seg_offsets_ptr, todo_cnt + 1, stream);

    int total_hits;
    CUDA_CHECK(cudaMemcpyAsync(&total_hits, buffers.seg_offsets_ptr + todo_cnt, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    *gpu_cost1 += GetTime() - t0;

    // --- Part 2: Marshal Data into Pre-allocated Buffers ---
    t0 = GetTime();

    if (buffers.key_value_capacity < total_hits) {
        fprintf(stderr, "FATAL ERROR: Pre-allocated sort buffer is too small. Required: %d, Allocated: %zu\n", total_hits, buffers.key_value_capacity);
        exit(EXIT_FAILURE);
    }

    marshal_data_for_sort_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(todo_cnt, hits_per_refs, global_todo_ids, buffers.seg_offsets_ptr, buffers.key_ptr, buffers.value_ptr);
    *gpu_cost2 += GetTime() - t0;

    // --- Part 3: Sort with CUB ---
    t0 = GetTime();
    cub::DoubleBuffer<int> d_keys_buffer(buffers.key_ptr, buffers.key_alt_ptr);
    cub::DoubleBuffer<int> d_values_buffer(buffers.value_ptr, buffers.value_alt_ptr);

    void* d_sort_temp_storage = nullptr;
    size_t sort_temp_storage_bytes = 0;
    cub::DeviceSegmentedSort::SortPairs(d_sort_temp_storage, sort_temp_storage_bytes, d_keys_buffer, d_values_buffer, total_hits, todo_cnt, buffers.seg_offsets_ptr, buffers.seg_offsets_ptr + 1, stream);

    if (buffers.sort_temp_bytes < sort_temp_storage_bytes) {
        if (buffers.sort_temp_ptr) CUDA_CHECK(cudaFreeAsync(buffers.sort_temp_ptr, stream));
        CUDA_CHECK(cudaMallocAsync(&buffers.sort_temp_ptr, sort_temp_storage_bytes * 2, stream));
        buffers.sort_temp_bytes = sort_temp_storage_bytes * 2;
    }
    cub::DeviceSegmentedSort::SortPairs(buffers.sort_temp_ptr, sort_temp_storage_bytes, d_keys_buffer, d_values_buffer, total_hits, todo_cnt, buffers.seg_offsets_ptr, buffers.seg_offsets_ptr + 1, stream);
    *gpu_cost3 += GetTime() - t0;

    res.first = buffers.seg_offsets_ptr;
    res.second = d_values_buffer.Current();
    return res;
}

__global__ void get_nam_sizes_kernel(int num_tasks, my_vector<Nam>* nams_per_task, int* global_todo_ids, int* task_sizes) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id < num_tasks) {
        int real_id = global_todo_ids[global_id];
        task_sizes[global_id] = nams_per_task[real_id].size();
    }
}

__global__ void marshal_data_for_nam_sort_kernel(
        int num_tasks,
        my_vector<Nam>* nams_per_task,
        int* global_todo_ids,
        const int* seg_offsets,
        int* d_keys,    // Output: scores (as integers)
        int* d_values   // Output: original indices 0, 1, 2...
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id < num_tasks) {
        int real_id = global_todo_ids[global_id];
        const my_vector<Nam>& nams = nams_per_task[real_id];
        int task_start_offset = seg_offsets[global_id];

        for (int i = 0; i < nams.size(); i++) {
            // Key is the score. Cast to int for sorting. For descending order, negate the score.
            d_keys[task_start_offset + i] = static_cast<int>(-nams.data[i].score);
            // Value is the original index within this specific my_vector<Nam>.
            d_values[task_start_offset + i] = i;
        }
    }
}

__global__ void gather_sorted_nams_kernel(
        int num_tasks,
        const my_vector<Nam>* nams_per_task,
        const int* global_todo_ids,
        const int* seg_offsets,
        const int* sorted_indices,
        Nam* temp_nams_output
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id >= num_tasks) return;

    int real_id = global_todo_ids[global_id];
    const my_vector<Nam>& nams = nams_per_task[real_id];
    int task_start = seg_offsets[global_id];
    int num_nams_in_task = seg_offsets[global_id + 1] - task_start;

    for (int i = 0; i < num_nams_in_task; ++i) {
        int original_idx = sorted_indices[task_start + i];
        temp_nams_output[task_start + i] = nams.data[original_idx];
    }
}

__global__ void scatter_sorted_nams_kernel(
        int num_tasks,
        const Nam* temp_nams_input,
        const int* global_todo_ids,
        const int* seg_offsets,
        my_vector<Nam>* nams_per_task
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id >= num_tasks) return;

    int real_id = global_todo_ids[global_id];
    my_vector<Nam>& nams = nams_per_task[real_id];
    int task_start = seg_offsets[global_id];
    int num_nams_in_task = seg_offsets[global_id + 1] - task_start;

    for (int i = 0; i < num_nams_in_task; ++i) {
        nams.data[i] = temp_nams_input[task_start + i];
    }
}

void sort_nams_by_score_in_place_with_cub(
        int todo_cnt,
        my_vector<Nam>* nams_per_task,
        int* global_todo_ids,
        cudaStream_t stream,
        SegSortGpuResources& buffers,
        double *gpu_cost1,
        double *gpu_cost2,
        double *gpu_cost3,
        double *gpu_cost4)
{
    if (todo_cnt == 0) return;

    int threads_per_block = 256;
    int blocks_per_grid = (todo_cnt + threads_per_block - 1) / threads_per_block;

    // --- Part 1: Get sizes and offsets for each segment ---
    double t0 = GetTime();

    size_t required_task_sizes_bytes = todo_cnt * sizeof(int);
    if (buffers.task_sizes_bytes < required_task_sizes_bytes) {
        printf("Allocating NAM task sizes buffer: %zu (%zu) bytes\n", required_task_sizes_bytes, buffers.task_sizes_bytes);
        if (buffers.task_sizes_ptr) CUDA_CHECK(cudaFreeAsync(buffers.task_sizes_ptr, stream));
        CUDA_CHECK(cudaMallocAsync(&buffers.task_sizes_ptr, required_task_sizes_bytes * 2, stream));
        buffers.task_sizes_bytes = required_task_sizes_bytes * 2;
    }
    get_nam_sizes_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(todo_cnt, nams_per_task, global_todo_ids, buffers.task_sizes_ptr);

    size_t required_seg_offsets_bytes = (todo_cnt + 1) * sizeof(int);
    if (buffers.seg_offsets_bytes < required_seg_offsets_bytes) {
        printf("Allocating NAM segment offsets buffer: %zu (%zu) bytes\n", required_seg_offsets_bytes, buffers.seg_offsets_bytes);
        if (buffers.seg_offsets_ptr) CUDA_CHECK(cudaFreeAsync(buffers.seg_offsets_ptr, stream));
        CUDA_CHECK(cudaMallocAsync(&buffers.seg_offsets_ptr, required_seg_offsets_bytes * 2, stream));
        buffers.seg_offsets_bytes = required_seg_offsets_bytes * 2;
    }

    void* d_scan_temp_storage = nullptr;
    size_t scan_temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_scan_temp_storage, scan_temp_storage_bytes, buffers.task_sizes_ptr, buffers.seg_offsets_ptr, todo_cnt + 1, stream);

    if (buffers.scan_temp_bytes < scan_temp_storage_bytes) {
        printf("Allocating NAM scan temp storage: %zu (%zu) bytes\n", scan_temp_storage_bytes, buffers.scan_temp_bytes);
        if (buffers.scan_temp_ptr) CUDA_CHECK(cudaFreeAsync(buffers.scan_temp_ptr, stream));
        CUDA_CHECK(cudaMallocAsync(&buffers.scan_temp_ptr, scan_temp_storage_bytes * 2, stream));
        buffers.scan_temp_bytes = scan_temp_storage_bytes * 2;
    }
    cub::DeviceScan::ExclusiveSum(buffers.scan_temp_ptr, scan_temp_storage_bytes, buffers.task_sizes_ptr, buffers.seg_offsets_ptr, todo_cnt + 1, stream);

    int total_nams;
    CUDA_CHECK(cudaMemcpyAsync(&total_nams, buffers.seg_offsets_ptr + todo_cnt, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    *gpu_cost1 += GetTime() - t0;

    if (total_nams == 0) {
        return;
    }

    // --- Part 2: Marshal scores (keys) and original indices (values) ---
    t0 = GetTime();

    if (buffers.key_value_capacity < total_nams) {
        printf("Allocating NAM key-value buffers: %d (%zu) bytes\n", total_nams, buffers.key_value_capacity);
        if (buffers.key_ptr)       CUDA_CHECK(cudaFreeAsync(buffers.key_ptr, stream));
        if (buffers.value_ptr)     CUDA_CHECK(cudaFreeAsync(buffers.value_ptr, stream));
        if (buffers.key_alt_ptr)   CUDA_CHECK(cudaFreeAsync(buffers.key_alt_ptr, stream));
        if (buffers.value_alt_ptr) CUDA_CHECK(cudaFreeAsync(buffers.value_alt_ptr, stream));
        size_t new_capacity = total_nams * 2;
        size_t new_bytes = new_capacity * sizeof(int);
        CUDA_CHECK(cudaMallocAsync(&buffers.key_ptr,       new_bytes, stream));
        CUDA_CHECK(cudaMallocAsync(&buffers.value_ptr,     new_bytes, stream));
        CUDA_CHECK(cudaMallocAsync(&buffers.key_alt_ptr,   new_bytes, stream));
        CUDA_CHECK(cudaMallocAsync(&buffers.value_alt_ptr, new_bytes, stream));
        buffers.key_value_capacity = new_capacity;
    }

    marshal_data_for_nam_sort_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(todo_cnt, nams_per_task, global_todo_ids, buffers.seg_offsets_ptr, buffers.key_ptr, buffers.value_ptr);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    *gpu_cost2 += GetTime() - t0;

    // --- Part 3: Sort key-value pairs to get the sorted order ---
    t0 = GetTime();
    cub::DoubleBuffer<int> d_keys_buffer(buffers.key_ptr, buffers.key_alt_ptr);
    cub::DoubleBuffer<int> d_values_buffer(buffers.value_ptr, buffers.value_alt_ptr);

    void* d_sort_temp_storage = nullptr;
    size_t sort_temp_storage_bytes = 0;
    cub::DeviceSegmentedSort::SortPairs(d_sort_temp_storage, sort_temp_storage_bytes, d_keys_buffer, d_values_buffer, total_nams, todo_cnt, buffers.seg_offsets_ptr, buffers.seg_offsets_ptr + 1, stream);

    if (buffers.sort_temp_bytes < sort_temp_storage_bytes) {
        printf("Allocating NAM sort temp storage: %zu (%zu) bytes\n", sort_temp_storage_bytes, buffers.sort_temp_bytes);
        if (buffers.sort_temp_ptr) CUDA_CHECK(cudaFreeAsync(buffers.sort_temp_ptr, stream));
        CUDA_CHECK(cudaMallocAsync(&buffers.sort_temp_ptr, sort_temp_storage_bytes * 2, stream));
        buffers.sort_temp_bytes = sort_temp_storage_bytes * 2;
    }
    cub::DeviceSegmentedSort::SortPairs(buffers.sort_temp_ptr, sort_temp_storage_bytes, d_keys_buffer, d_values_buffer, total_nams, todo_cnt, buffers.seg_offsets_ptr, buffers.seg_offsets_ptr + 1, stream);
    const int* sorted_indices = d_values_buffer.Current();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    *gpu_cost3 += GetTime() - t0;


    // --- Part 4: Reorder the actual NAM objects in-place ---
    t0 = GetTime();

    if (buffers.nam_temp_capacity < total_nams) {
        printf("Allocating NAM temporary buffer: %d (%zu) bytes\n", total_nams * 2, buffers.nam_temp_capacity);
        if (buffers.nam_temp_ptr) CUDA_CHECK(cudaFreeAsync(buffers.nam_temp_ptr, stream));
        CUDA_CHECK(cudaMallocAsync(&buffers.nam_temp_ptr, total_nams * 2 * sizeof(Nam), stream));
        buffers.nam_temp_capacity = total_nams * 2;
    }

    // Step A: Gather NAMs into the temporary buffer in sorted order.
    gather_sorted_nams_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
            todo_cnt, nams_per_task, global_todo_ids, buffers.seg_offsets_ptr, sorted_indices, buffers.nam_temp_ptr
    );

    // Step B: Scatter the sorted NAMs from the temp buffer back to the original vectors.
    scatter_sorted_nams_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
            todo_cnt, buffers.nam_temp_ptr, global_todo_ids, buffers.seg_offsets_ptr, nams_per_task
    );
    CUDA_CHECK(cudaStreamSynchronize(stream));
    *gpu_cost4 += GetTime() - t0;
}


my_pair<int*, int*> sort_nams_by_score_with_cub(
        int todo_cnt,
        my_vector<Nam>* nams_per_task,
        int* global_todo_ids,
        cudaStream_t stream,
        SegSortGpuResources& buffers,
        double *gpu_cost1,
        double *gpu_cost2,
        double *gpu_cost3,
        double *gpu_cost4)
{
    my_pair<int*, int*> res({nullptr, nullptr});
    if (todo_cnt == 0) return res;

    int threads_per_block = 256;
    int blocks_per_grid = (todo_cnt + threads_per_block - 1) / threads_per_block;

    // --- Part 1: Prepare Segment Information with Re-allocation Logic ---
    double t0 = GetTime();

    size_t required_task_sizes_bytes = todo_cnt * sizeof(int);
    if (buffers.task_sizes_bytes < required_task_sizes_bytes) {
        printf("Allocating NAM task sizes buffer: %zu (%zu) bytes\n", required_task_sizes_bytes, buffers.task_sizes_bytes);
        if (buffers.task_sizes_ptr) CUDA_CHECK(cudaFreeAsync(buffers.task_sizes_ptr, stream));
        CUDA_CHECK(cudaMallocAsync(&buffers.task_sizes_ptr, required_task_sizes_bytes * 2, stream));
        buffers.task_sizes_bytes = required_task_sizes_bytes * 2;
    }
    get_nam_sizes_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(todo_cnt, nams_per_task, global_todo_ids, buffers.task_sizes_ptr);

    size_t required_seg_offsets_bytes = (todo_cnt + 1) * sizeof(int);
    if (buffers.seg_offsets_bytes < required_seg_offsets_bytes) {
        printf("Allocating NAM segment offsets buffer: %zu (%zu) bytes\n", required_seg_offsets_bytes, buffers.seg_offsets_bytes);
        if (buffers.seg_offsets_ptr) CUDA_CHECK(cudaFreeAsync(buffers.seg_offsets_ptr, stream));
        CUDA_CHECK(cudaMallocAsync(&buffers.seg_offsets_ptr, required_seg_offsets_bytes * 2, stream));
        buffers.seg_offsets_bytes = required_seg_offsets_bytes * 2;
    }

    void* d_scan_temp_storage = nullptr;
    size_t scan_temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_scan_temp_storage, scan_temp_storage_bytes, buffers.task_sizes_ptr, buffers.seg_offsets_ptr, todo_cnt + 1, stream);

    if (buffers.scan_temp_bytes < scan_temp_storage_bytes) {
        printf("Allocating NAM scan temp storage: %zu (%zu) bytes\n", scan_temp_storage_bytes, buffers.scan_temp_bytes);
        if (buffers.scan_temp_ptr) CUDA_CHECK(cudaFreeAsync(buffers.scan_temp_ptr, stream));
        CUDA_CHECK(cudaMallocAsync(&buffers.scan_temp_ptr, scan_temp_storage_bytes * 2, stream));
        buffers.scan_temp_bytes = scan_temp_storage_bytes * 2;
    }
    cub::DeviceScan::ExclusiveSum(buffers.scan_temp_ptr, scan_temp_storage_bytes, buffers.task_sizes_ptr, buffers.seg_offsets_ptr, todo_cnt + 1, stream);

    int total_nams;
    CUDA_CHECK(cudaMemcpyAsync(&total_nams, buffers.seg_offsets_ptr + todo_cnt, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    *gpu_cost1 += GetTime() - t0;

    if (total_nams == 0) {
        res.first = buffers.seg_offsets_ptr;
        return res;
    }

    // --- Part 2: Marshal Data into Pre-allocated Buffers ---
    t0 = GetTime();

    if (buffers.key_value_capacity < total_nams) {
        printf("Allocating NAM key-value buffers: %d (%zu) bytes\n", total_nams, buffers.key_value_capacity);
        if (buffers.key_ptr)       CUDA_CHECK(cudaFreeAsync(buffers.key_ptr, stream));
        if (buffers.value_ptr)     CUDA_CHECK(cudaFreeAsync(buffers.value_ptr, stream));
        if (buffers.key_alt_ptr)   CUDA_CHECK(cudaFreeAsync(buffers.key_alt_ptr, stream));
        if (buffers.value_alt_ptr) CUDA_CHECK(cudaFreeAsync(buffers.value_alt_ptr, stream));
        size_t new_capacity = total_nams * 2;
        size_t new_bytes = new_capacity * sizeof(int);
        CUDA_CHECK(cudaMallocAsync(&buffers.key_ptr,       new_bytes, stream));
        CUDA_CHECK(cudaMallocAsync(&buffers.value_ptr,     new_bytes, stream));
        CUDA_CHECK(cudaMallocAsync(&buffers.key_alt_ptr,   new_bytes, stream));
        CUDA_CHECK(cudaMallocAsync(&buffers.value_alt_ptr, new_bytes, stream));
        buffers.key_value_capacity = new_capacity;
    }

    marshal_data_for_nam_sort_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(todo_cnt, nams_per_task, global_todo_ids, buffers.seg_offsets_ptr, buffers.key_ptr, buffers.value_ptr);
    *gpu_cost2 += GetTime() - t0;

    // --- Part 3: Sort with CUB ---
    t0 = GetTime();
    cub::DoubleBuffer<int> d_keys_buffer(buffers.key_ptr, buffers.key_alt_ptr);
    cub::DoubleBuffer<int> d_values_buffer(buffers.value_ptr, buffers.value_alt_ptr);

    void* d_sort_temp_storage = nullptr;
    size_t sort_temp_storage_bytes = 0;
    cub::DeviceSegmentedSort::SortPairs(d_sort_temp_storage, sort_temp_storage_bytes, d_keys_buffer, d_values_buffer, total_nams, todo_cnt, buffers.seg_offsets_ptr, buffers.seg_offsets_ptr + 1, stream);

    if (buffers.sort_temp_bytes < sort_temp_storage_bytes) {
        printf("Allocating NAM sort temp storage: %zu (%zu) bytes\n", sort_temp_storage_bytes, buffers.sort_temp_bytes);
        if (buffers.sort_temp_ptr) CUDA_CHECK(cudaFreeAsync(buffers.sort_temp_ptr, stream));
        CUDA_CHECK(cudaMallocAsync(&buffers.sort_temp_ptr, sort_temp_storage_bytes * 2, stream));
        buffers.sort_temp_bytes = sort_temp_storage_bytes * 2;
    }
    cub::DeviceSegmentedSort::SortPairs(buffers.sort_temp_ptr, sort_temp_storage_bytes, d_keys_buffer, d_values_buffer, total_nams, todo_cnt, buffers.seg_offsets_ptr, buffers.seg_offsets_ptr + 1, stream);
    *gpu_cost3 += GetTime() - t0;

    // The function does not free any pre-allocated memory.
    res.first = buffers.seg_offsets_ptr;
    res.second = d_values_buffer.Current(); // This is the pointer to the sorted indices.
    return res;
}


__global__ void gpu_sort_hits(
        int num_tasks,
        my_vector<my_pair<int, Hit>>* hits_per_ref0s,
        my_vector<my_pair<int, Hit>>* hits_per_ref1s,
        int* global_todo_ids
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num_tasks) {
        int real_id = global_todo_ids[id];
#ifdef GPU_ACC_TAG
        sort_hits_single(hits_per_ref0s[real_id]);
        sort_hits_single(hits_per_ref1s[real_id]);
#else
        sort_hits_by_refid(hits_per_ref0s[real_id]);
        sort_hits_by_refid(hits_per_ref1s[real_id]);
#endif
        //check_hits(hits_per_ref0s[real_id]);
        //check_hits(hits_per_ref1s[real_id]);
    }
}

__global__ void gpu_rescue_sort_hits(
        int num_tasks,
        my_vector<my_pair<int, Hit>>* hits_per_ref0s,
        my_vector<my_pair<int, Hit>>* hits_per_ref1s,
        int* global_todo_ids
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num_tasks) {
        int real_id = global_todo_ids[id];
#ifdef GPU_ACC_TAG
        sort_hits_single(hits_per_ref0s[real_id]);
        sort_hits_single(hits_per_ref1s[real_id]);
#else
        sort_hits_by_refid(hits_per_ref0s[real_id]);
        sort_hits_by_refid(hits_per_ref1s[real_id]);
#endif
        //check_hits(hits_per_ref0s[real_id]);
        //check_hits(hits_per_ref1s[real_id]);
    }
}

__global__ void gpu_merge_hits_get_nams_seg(
        int num_tasks,
        IndexParameters *index_para,
        uint64_t *global_nams_info,
        my_vector<my_pair<int, Hit>>* hits_per_ref0s,
        my_vector<my_pair<int, Hit>>* hits_per_ref1s,
        const int* seg_offsets0, const int* sorted_indices0,
        const int* seg_offsets1, const int* sorted_indices1,
        my_vector<Nam> *global_nams,
        int* global_todo_ids
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id < num_tasks) {
        int real_id = global_todo_ids[global_id];
        global_nams[real_id].init(8);

        // Process hits for read 1 (forward strand)
        const my_vector<my_pair<int, Hit>>& original_hits0 = hits_per_ref0s[real_id];
        int task_start0 = seg_offsets0[global_id];
        int task_end0   = seg_offsets0[global_id + 1];
        merge_hits_seg(original_hits0, sorted_indices0, task_start0, task_end0, index_para->syncmer.k, 0, global_nams[real_id]);

        // Process hits for read 2 (reverse strand)
        const my_vector<my_pair<int, Hit>>& original_hits1 = hits_per_ref1s[real_id];
        int task_start1 = seg_offsets1[global_id];
        int task_end1   = seg_offsets1[global_id + 1];
        merge_hits_seg(original_hits1, sorted_indices1, task_start1, task_end1, index_para->syncmer.k, 1, global_nams[real_id]);

        hits_per_ref0s[real_id].release();
        hits_per_ref1s[real_id].release();
    }
}

__global__ void gpu_merge_hits_get_nams(
        int num_tasks,
        IndexParameters *index_para,
        uint64_t *global_nams_info,
        my_vector<my_pair<int, Hit>>* hits_per_ref0s,
        my_vector<my_pair<int, Hit>>* hits_per_ref1s,
        my_vector<Nam> *global_nams,
        int* global_todo_ids
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num_tasks) {
        int real_id = global_todo_ids[id];
        global_nams[real_id].init(8);
        merge_hits(hits_per_ref0s[real_id], index_para->syncmer.k, 0, global_nams[real_id]);
        merge_hits(hits_per_ref1s[real_id], index_para->syncmer.k, 1, global_nams[real_id]);
        hits_per_ref0s[real_id].release();
        hits_per_ref1s[real_id].release();
    }
}

__global__ void gpu_rescue_merge_hits_get_nams_seg(
        int num_tasks,
        IndexParameters *index_para,
        uint64_t *global_nams_info,
        my_vector<my_pair<int, Hit>>* hits_per_ref0s,
        my_vector<my_pair<int, Hit>>* hits_per_ref1s,
        const int* seg_offsets0, const int* sorted_indices0,
        const int* seg_offsets1, const int* sorted_indices1,
        my_vector<Nam> *global_nams,
        int* global_todo_ids
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id < num_tasks) {
        int real_id = global_todo_ids[global_id];
        global_nams[real_id].init(8);

        // Process hits for read 1 (forward strand)
        const my_vector<my_pair<int, Hit>>& original_hits0 = hits_per_ref0s[real_id];
        int task_start0 = seg_offsets0[global_id];
        int task_end0   = seg_offsets0[global_id + 1];
        salign_merge_hits_seg(original_hits0, sorted_indices0, task_start0, task_end0, index_para->syncmer.k, 0, global_nams[real_id]);

        // Process hits for read 2 (reverse strand)
        const my_vector<my_pair<int, Hit>>& original_hits1 = hits_per_ref1s[real_id];
        int task_start1 = seg_offsets1[global_id];
        int task_end1   = seg_offsets1[global_id + 1];
        salign_merge_hits_seg(original_hits1, sorted_indices1, task_start1, task_end1, index_para->syncmer.k, 1, global_nams[real_id]);

        hits_per_ref0s[real_id].release();
        hits_per_ref1s[real_id].release();
    }
}

__global__ void gpu_rescue_merge_hits_get_nams(
        int num_tasks,
        IndexParameters *index_para,
        uint64_t *global_nams_info,
        my_vector<my_pair<int, Hit>>* hits_per_ref0s,
        my_vector<my_pair<int, Hit>>* hits_per_ref1s,
        my_vector<Nam> *global_nams,
        int* global_todo_ids
)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num_tasks){
        int real_id = global_todo_ids[id];
        global_nams[real_id].init(8);
        salign_merge_hits(hits_per_ref0s[real_id], index_para->syncmer.k, 0, global_nams[real_id]);
        salign_merge_hits(hits_per_ref1s[real_id], index_para->syncmer.k, 1, global_nams[real_id]);
        hits_per_ref0s[real_id].release();
        hits_per_ref1s[real_id].release();
    }
}


__global__ void gpu_sort_nams(
        int num_tasks,
        my_vector<Nam> *global_nams,
        MappingParameters *mapping_parameters,
        int is_se
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num_tasks) {
        int max_tries = mapping_parameters->max_tries;
        if (is_se) {
#ifdef GPU_ACC_TAG
            sort_nams_single_check(global_nams[id]);
#else
            sort_nams_by_score(global_nams[id], max_tries);
            global_nams[id].length = my_min(global_nams[id].length, max_tries);
#endif
        } else {
#ifdef GPU_ACC_TAG
            sort_nams_single_check(global_nams[id]);
#else
//            sort_nams_by_score(global_nams[id], max_tries * 2);
//            global_nams[id].length = my_min(global_nams[id].length, max_tries * 2);
            sort_nams_by_score(global_nams[id], 1e9);
#endif
        }
        gpu_shuffle_top_nams(global_nams[id]);
    }
}
