#include "gpu_alignment.h"

__device__ void cigar_push(my_vector<uint32_t>& m_ops, uint8_t op, int len) {
    assert(op < 16);
    if (m_ops.empty() || (m_ops.back() & 0xf) != op) {
        m_ops.push_back(len << 4 | op);
    } else {
        m_ops.back() += len << 4;
    }
}

__device__ bool gpu_has_shared_substring(const my_string& read_seq, const my_string& ref_seq, int k) {
    // TODO test hash version
    int sub_size = 2 * k / 3;
    int step_size = k / 3;

    my_string submer;
    for (size_t i = 0; i + sub_size < read_seq.size(); i += step_size) {
        submer = read_seq.substr(i, sub_size);
        if (ref_seq.find(submer) != -1) {
            return true;
        }
    }
    return false;
}

__device__ void my_hamming_align(const my_string &query, const my_string &ref, int match, int mismatch, int end_bonus, GPUAlignmentInfo& aln) {
    if (query.length() != ref.length()) {
        return;
    }
    size_t n = query.length();

    size_t start = 0; // start of the current segment
    int score = end_bonus; // accumulated score so far in the current segment

    size_t best_start = 0;
    size_t best_end = 0;
    int best_score = 0;
    for (size_t i = 0; i < n; ++i) {
        if (query[i] == ref[i]) {
            score += match;
        } else {
            score -= mismatch;
        }
        if (score < 0) {
            start = i + 1;
            score = 0;
        }
        if (score > best_score) {
            best_start = start;
            best_score = score;
            best_end = i + 1;
        }
    }
    if (score + end_bonus > best_score) {
        best_score = score + end_bonus;
        best_end = query.length();
        best_start = start;
    }

    size_t segment_start = best_start;
    size_t segment_end = best_end;
    score = best_score;

    if (segment_start > 0) {
        cigar_push(aln.cigar, CIGAR_SOFTCLIP, segment_start);
    }

    // Create CIGAR string and count mismatches
    int counter = 0;
    bool prev_is_match = false;
    int mismatches = 0;
    bool first = true;
    for (size_t i = segment_start; i < segment_end; i++) {
        bool is_match = query[i] == ref[i];
        mismatches += is_match ? 0 : 1;
        if (!first && is_match != prev_is_match) {
            cigar_push(aln.cigar, prev_is_match ? CIGAR_EQ : CIGAR_X, counter);
            counter = 0;
        }
        counter++;
        prev_is_match = is_match;
        first = false;
    }
    if (!first) {
        cigar_push(aln.cigar, prev_is_match ? CIGAR_EQ : CIGAR_X, counter);
    }

    int soft_right = query.length() - segment_end;
    if (soft_right > 0) {
        cigar_push(aln.cigar, CIGAR_SOFTCLIP, soft_right);
    }

    aln.sw_score = score;
    aln.edit_distance = mismatches;
    aln.ref_start = segment_start;
    aln.ref_end = segment_end;
    aln.query_start = segment_start;
    aln.query_end = segment_end;
    return;
}

__device__ inline int gpu_hamming_distance(const my_string s, const my_string t) {
    if (s.length() != t.length()){
        return -1;
    }
    int mismatches = 0;
    for (size_t i = 0; i < s.length(); i++) {
        if (s[i] != t[i]) {
            mismatches++;
        }
    }
    return mismatches;
}

__device__ bool gpu_extend_seed_part(
        GPUAlignTmpRes& align_tmp_res,
        const AlignmentParameters& aligner_parameters,
        const Nam& nam,
        const GPUReferences& references,
        const GPURead& read,
        bool consistent_nam
) {
    const my_string query(nam.is_rc ? read.rc : read.seq, read.length);
    const my_string ref = references.sequences[nam.ref_id];

    const auto projected_ref_start = my_max(0, nam.ref_start - nam.query_start);
    const auto projected_ref_end = my_min(nam.ref_end + query.size() - nam.query_end, ref.size());

    GPUAlignmentInfo info;
    int result_ref_start;
    bool gapped = true;
    if (projected_ref_end - projected_ref_start == query.size() && consistent_nam) {
        my_string ref_segm_ham = ref.substr(projected_ref_start, query.size());
        int hamming_dist = gpu_hamming_distance(query, ref_segm_ham);
        if (hamming_dist >= 0 && (((float) hamming_dist / query.size()) < 0.05)) {  //Hamming distance worked fine, no need to ksw align
            my_hamming_align(
                    query, ref_segm_ham, aligner_parameters.match, aligner_parameters.mismatch,
                    aligner_parameters.end_bonus, info
            );
            result_ref_start = projected_ref_start + info.ref_start;
            gapped = false; // Tentatively mark as non-gapped (Hamming passed)

            // MODIFICATION: Check if the CIGAR is too long.
            // If so, revert to gapped alignment to trigger CPU-side handling.
            if (info.cigar.size() + 1 > MAX_CIGAR_ITEM) {
                //printf("Warning: CIGAR too long -- %d, reverting to gapped alignment.\n", info.cigar.size() + 1);
                gapped = true;
            }
        }
    }

    align_tmp_res.todo_nams.push_back(nam);
    align_tmp_res.is_extend_seed.push_back(true);
    if (gapped) {
        // This path is now taken if Hamming failed OR if the resulting CIGAR was too long.
        GPUAlignment alignment;
        align_tmp_res.done_align.push_back(false);
        align_tmp_res.align_res.push_back(alignment);
        align_tmp_res.cigar_info.length++;
        align_tmp_res.cigar_info.back().cigar = align_tmp_res.cigar_info.back().gpu_cigar;
        align_tmp_res.cigar_info.back().cigar[0] = 0;
        align_tmp_res.cigar_info.back().has_realloc = 0;
    } else {
        // This path is now only taken if Hamming passed AND the CIGAR fits.
        align_tmp_res.done_align.push_back(true);
        int softclipped = info.query_start + (query.size() - info.query_end);
        GPUAlignment alignment;
        alignment.edit_distance = info.edit_distance;
        alignment.global_ed = info.edit_distance + softclipped;
        alignment.score = info.sw_score;
        alignment.ref_start = result_ref_start;
        alignment.length = info.ref_span();
        alignment.is_rc = nam.is_rc;
        alignment.is_unaligned = false;
        alignment.ref_id = nam.ref_id;
        alignment.gapped = gapped;
        align_tmp_res.align_res.push_back(alignment);

        align_tmp_res.cigar_info.length++;
        align_tmp_res.cigar_info.back().cigar = align_tmp_res.cigar_info.back().gpu_cigar;
        align_tmp_res.cigar_info.back().has_realloc = 0;
        align_tmp_res.cigar_info.back().cigar[0] = info.cigar.size();
        for (int i = 0; i < info.cigar.size(); i++) {
            align_tmp_res.cigar_info.back().cigar[i + 1] = info.cigar[i];
        }
    }
    return gapped;
}

__device__ bool gpu_rescue_mate_part(
        GPUAlignTmpRes& align_tmp_res,
        const AlignmentParameters& aligner_parameters,
        const Nam& nam,
        const GPUReferences& references,
        const GPURead& read,
        float mu,
        float sigma,
        int k
) {
    GPUAlignment alignment;
    int a, b;
    my_string r_tmp;
    auto read_len = read.size();

    if (nam.is_rc) {
        r_tmp = my_string(read.seq, read_len);
        a = nam.ref_start - nam.query_start - (mu + 5 * sigma);
        b = nam.ref_start - nam.query_start + read_len / 2;  // at most half read overlap
    } else {
        r_tmp = my_string(read.rc, read_len);                                             // mate is rc since fr orientation
        a = nam.ref_end + (read_len - nam.query_end) - read_len / 2;  // at most half read overlap
        b = nam.ref_end + (read_len - nam.query_end) + (mu + 5 * sigma);
    }

    auto ref_len = references.lengths[nam.ref_id];
    auto ref_start = my_max(0, my_min(a, ref_len));
    auto ref_end = my_min(ref_len, my_max(0, b));

    align_tmp_res.todo_nams.push_back(nam);
    align_tmp_res.is_extend_seed.push_back(false);
    if (ref_end < ref_start + k) {
        //        alignment.cigar = Cigar();
        alignment.edit_distance = read_len;
        alignment.score = 0;
        alignment.ref_start = 0;
        alignment.is_rc = nam.is_rc;
        alignment.ref_id = nam.ref_id;
        alignment.is_unaligned = true;
        align_tmp_res.done_align.push_back(true);
        align_tmp_res.align_res.push_back(alignment);
        align_tmp_res.cigar_info.length++;
        align_tmp_res.cigar_info.back().cigar = align_tmp_res.cigar_info.back().gpu_cigar;
        align_tmp_res.cigar_info.back().cigar[0] = 0;
        align_tmp_res.cigar_info.back().has_realloc = 0;
        return true;
    }
    my_string ref_segm = references.sequences[nam.ref_id].substr(ref_start, ref_end - ref_start);

    if (!gpu_has_shared_substring(r_tmp, ref_segm, k)) {
        //        alignment.cigar = Cigar();
        alignment.edit_distance = read_len;
        alignment.score = 0;
        alignment.ref_start = 0;
        alignment.is_rc = nam.is_rc;
        alignment.ref_id = nam.ref_id;
        alignment.is_unaligned = true;
        align_tmp_res.done_align.push_back(true);
        align_tmp_res.align_res.push_back(alignment);
        align_tmp_res.cigar_info.length++;
        align_tmp_res.cigar_info.back().cigar = align_tmp_res.cigar_info.back().gpu_cigar;
        align_tmp_res.cigar_info.back().cigar[0] = 0;
        align_tmp_res.cigar_info.back().has_realloc = 0;
        return true;
    }

    align_tmp_res.done_align.push_back(false);
    align_tmp_res.align_res.push_back(alignment);
    align_tmp_res.cigar_info.length++;
    align_tmp_res.cigar_info.back().cigar = align_tmp_res.cigar_info.back().gpu_cigar;
    align_tmp_res.cigar_info.back().cigar[0] = 0;
    align_tmp_res.cigar_info.back().has_realloc = 0;
    return false;
}

__device__ bool gpu_reverse_nam_if_needed(Nam& nam, const GPURead& read, const GPUReferences& references, int k) {
    auto read_len = read.size();
    my_string ref_start_kmer = references.sequences[nam.ref_id].substr(nam.ref_start, k);
    my_string ref_end_kmer = references.sequences[nam.ref_id].substr(nam.ref_end - k, k);


    my_string seq, seq_rc;
    if (nam.is_rc) {
        seq = my_string(read.rc, read_len);
        seq_rc = my_string(read.seq, read_len);
    } else {
        seq = my_string(read.seq, read_len);
        seq_rc = my_string(read.rc, read_len);
    }
    my_string read_start_kmer = seq.substr(nam.query_start, k);
    my_string read_end_kmer = seq.substr(nam.query_end - k, k);
    if (ref_start_kmer == read_start_kmer && ref_end_kmer == read_end_kmer) {
        return true;
    }

    // False forward or false reverse (possible due to symmetrical hash values)
    //    we need two extra checks for this - hopefully this will remove all the false hits we see (true hash collisions should be very few)
    int q_start_tmp = read_len - nam.query_end;
    int q_end_tmp = read_len - nam.query_start;
    // false reverse hit, change coordinates in nam to forward
    read_start_kmer = seq_rc.substr(q_start_tmp, k);
    read_end_kmer = seq_rc.substr(q_end_tmp - k, k);
    if (ref_start_kmer == read_start_kmer && ref_end_kmer == read_end_kmer) {
        nam.is_rc = !nam.is_rc;
        nam.query_start = q_start_tmp;
        nam.query_end = q_end_tmp;
        return true;
    }
    return false;
}

__device__ uint8_t gpu_get_mapq_seg(const my_vector<Nam>& nams, const Nam& n_max, const int* sorted_indices) {
    if (nams.size() <= 1) {
        return 60;
    }
    const float s1 = n_max.score;
    const float s2 = nams[sorted_indices[1]].score;
    // from minimap2: MAPQ = 40(1−s2/s1) ·min{1,|M|/10} · log s1
    const float min_matches = my_min(n_max.n_hits / 10.0, 1.0);
    const int uncapped_mapq = 40 * (1 - s2 / s1) * min_matches * log(s1);
    return my_min(uncapped_mapq, 60);
}


__device__ uint8_t gpu_get_mapq(const my_vector<Nam>& nams, const Nam& n_max) {
    if (nams.size() <= 1) {
        return 60;
    }
    const float s1 = n_max.score;
    const float s2 = nams[1].score;
    // from minimap2: MAPQ = 40(1−s2/s1) ·min{1,|M|/10} · log s1
    const float min_matches = my_min(n_max.n_hits / 10.0, 1.0);
    const int uncapped_mapq = 40 * (1 - s2 / s1) * min_matches * log(s1);
    return my_min(uncapped_mapq, 60);
}

__device__ float gpu_top_dropoff(my_vector<Nam>& nams) {
    auto& n_max = nams[0];
    if (n_max.n_hits <= 2) {
        return 1.0;
    }
    if (nams.size() > 1) {
        return (float) nams[1].n_hits / n_max.n_hits;
    }
    return 0.0;
}

__device__ void gpu_part2_extend_seed_get_str(
        GPUAlignTmpRes& align_tmp_res,
        int j,
        const GPURead& read1,
        const GPURead& read2,
        const GPUReferences& references,
        int* d_todo_cnt,
        char* d_query_ptr, char* d_ref_ptr,
        int* d_query_offset, int* d_ref_offset
) {
    Nam nam = align_tmp_res.todo_nams[j];
    GPURead read = align_tmp_res.is_read1[j] ? read1 : read2;
    const my_string query = nam.is_rc ? my_string(read.rc, read.length) : my_string(read.seq, read.length);
    const my_string ref = references.sequences[nam.ref_id];

    const auto projected_ref_start = my_max(0, nam.ref_start - nam.query_start);
    const auto projected_ref_end = my_max(nam.ref_end + query.size() - nam.query_end, ref.size());

    const int diff = my_abs(nam.ref_span() - nam.query_span());
    const int ext_left = my_min(50, projected_ref_start);
    const int ref_start = projected_ref_start - ext_left;
    const int ext_right = my_min(50, ref.size() - nam.ref_end);
    auto ref_segm_size = read.size() + diff + ext_left + ext_right;
    if (ref_start + ref_segm_size > references.lengths[nam.ref_id]) ref_segm_size = references.lengths[nam.ref_id] - ref_start;
    int query_segm_size = read.length;
    int query_start = 0;

    uint32_t packed = (static_cast<uint32_t>(align_tmp_res.is_read1[j]) << 31) |
                      (static_cast<uint32_t>(nam.is_rc) << 30) |
                      (static_cast<uint32_t>(query_start) << 15) |
                      (static_cast<uint32_t>(query_segm_size));
    int global_id = 0;
#ifdef use_device_mem
    int q_len = query_segm_size;
    int r_len = ref_segm_size;
    q_len = ((q_len + 7) & ~7);
    r_len = ((r_len + 7) & ~7);

    global_id = atomicAdd(d_todo_cnt, 1);
    int query_offset = atomicAdd(d_query_offset, q_len);
    int ref_offset = atomicAdd(d_ref_offset, r_len);

    memcpy(d_query_ptr + query_offset, query.data + query_start, query_segm_size);
    memcpy(d_ref_ptr + ref_offset, ref.data + ref_start, ref_segm_size);
    memset(d_query_ptr + query_offset + query_segm_size, 0x4E, q_len - query_segm_size);
    memset(d_ref_ptr + ref_offset + ref_segm_size, 0x4E, r_len - ref_segm_size);
#endif

    align_tmp_res.todo_infos.push_back({global_id, packed, nam.ref_id, ref_start, ref_segm_size});
}

__device__ void gpu_part2_rescue_mate_get_str(
        GPUAlignTmpRes& align_tmp_res,
        int j,
        const GPURead& read1,
        const GPURead& read2,
        const GPUReferences& references,
        float mu,
        float sigma,
        int* d_todo_cnt,
        char* d_query_ptr, char* d_ref_ptr,
        int* d_query_offset, int* d_ref_offset
) {
    Nam nam = align_tmp_res.todo_nams[j];
    GPURead read = align_tmp_res.is_read1[j] ? read1 : read2;
    int a, b;
    my_string r_tmp;
    auto read_len = read.size();

    if (nam.is_rc) {
        r_tmp = my_string(read.seq, read.length);
        a = nam.ref_start - nam.query_start - (mu + 5 * sigma);
        b = nam.ref_start - nam.query_start + read_len / 2;  // at most half read overlap
    } else {
        r_tmp = my_string(read.rc, read.length);                                              // mate is rc since fr orientation
        a = nam.ref_end + (read_len - nam.query_end) - read_len / 2;  // at most half read overlap
        b = nam.ref_end + (read_len - nam.query_end) + (mu + 5 * sigma);
    }

    auto ref_len = references.lengths[nam.ref_id];
    auto ref_start = my_max(0, my_min(a, ref_len));
    auto ref_end = my_min(ref_len, my_max(0, b));
    int ref_segm_size = ref_end - ref_start;
    int query_segm_size = read.length;
    int query_start = 0;

    uint32_t packed = (static_cast<uint32_t>(align_tmp_res.is_read1[j]) << 31) |
                      (static_cast<uint32_t>(!nam.is_rc) << 30) |
                      (static_cast<uint32_t>(query_start) << 15) |
                      (static_cast<uint32_t>(query_segm_size));

    int global_id = 0;
#ifdef use_device_mem
    int q_len = query_segm_size;
    int r_len = ref_segm_size;
    q_len = ((q_len + 7) & ~7);
    r_len = ((r_len + 7) & ~7);

    global_id = atomicAdd(d_todo_cnt, 1);
    int query_offset = atomicAdd(d_query_offset, q_len);
    int ref_offset = atomicAdd(d_ref_offset, r_len);

    memcpy(d_query_ptr + query_offset, r_tmp.data + query_start, query_segm_size);
    memcpy(d_ref_ptr + ref_offset, references.sequences[nam.ref_id].data + ref_start, ref_segm_size);
    memset(d_query_ptr + query_offset + query_segm_size, 0x4E, q_len - query_segm_size);
    memset(d_ref_ptr + ref_offset + ref_segm_size, 0x4E, r_len - ref_segm_size);
#endif

    align_tmp_res.todo_infos.push_back({global_id, packed, nam.ref_id, ref_start, ref_segm_size});
}
