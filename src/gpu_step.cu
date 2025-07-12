
#include <cuda_runtime.h>
#include "gpu_step.h"
//#define assert(x) ((void)0)


#define my_bucket_index_t StrobemerIndex::bucket_index_t

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
        return static_cast<size_t>(-1); // No match
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
        } else return static_cast<size_t>(-1); // No match
    }
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
__device__ void bubble_sort(T* data, int size) {
    for (int i = 0; i < size - 1; ++i) {
        for (int j = 0; j < size - i - 1; ++j) {
            if (data[j + 1] < data[j]) {
                T temp = data[j];
                data[j] = data[j + 1];
                data[j + 1] = temp;
            }
        }
    }
}

template <typename T>
__device__ void quick_sort(T* data, int size) {
    quick_sort_iterative(data, 0, size - 1);
}

struct Rescue_Seeds {
    int read_id;
    int read_fr;
    int seeds_num;
    QueryRandstrobe* seeds;
};

__device__ int lock = 0;

__device__ void acquire_lock() {
    while (atomicCAS(&lock, 0, 1) != 0) {
    }
}

__device__ void release_lock() {
    atomicExch(&lock, 0);
}

__device__ void print_nam(Nam nam) {
    printf("nam_id: %d, ref_id: %d, ref_start: %d, ref_end: %d, query_start: %d, query_end: %d, n_hits: %d, is_rc: %d\n",
           nam.nam_id, nam.ref_id, nam.ref_start, nam.ref_end, nam.query_start, nam.query_end, nam.n_hits, nam.is_rc);
}

__device__ void print_str(my_string str) {
    for(int i = 0; i < str.size(); i++) {
        printf("%c", str[i]);
    }
    printf("\n");
}


#define MAX_TRIES_LIMIT (d_map_param->max_tries * 2 + 2)


__device__ void cigar_push(my_vector<uint32_t>& m_ops, uint8_t op, int len) {
    assert(op < 16);
    if (m_ops.empty() || (m_ops.back() & 0xf) != op) {
        m_ops.push_back(len << 4 | op);
    } else {
        m_ops.back() += len << 4;
    }
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
            gapped = false;
        }
    }

    align_tmp_res.todo_nams.push_back(nam);
    align_tmp_res.is_extend_seed.push_back(true);
    if (gapped) {
        // not pass hamming, pending to do align on GPU, tag is false
        GPUAlignment alignment;
        align_tmp_res.done_align.push_back(false);
        align_tmp_res.align_res.push_back(alignment);
        align_tmp_res.cigar_info.length++;
        align_tmp_res.cigar_info.back().cigar = align_tmp_res.cigar_info.back().gpu_cigar;
        align_tmp_res.cigar_info.back().cigar[0] = 0;
        align_tmp_res.cigar_info.back().has_realloc = 0;
    } else {
        // pass hamming, store result, tag is true
        align_tmp_res.done_align.push_back(true);
        int softclipped = info.query_start + (query.size() - info.query_end);
        GPUAlignment alignment;
        //alignment.cigar.move_from(info.cigar);
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
        if (info.cigar.size() + 1 > MAX_CIGAR_ITEM) {
            printf("gpu gg cigar size %d\n", info.cigar.size());
        }
        assert(info.cigar.size() + 1 <= MAX_CIGAR_ITEM);
        align_tmp_res.cigar_info.length++;
        align_tmp_res.cigar_info.back().cigar = align_tmp_res.cigar_info.back().gpu_cigar;
        align_tmp_res.cigar_info.back().cigar[0] = info.cigar.size();
        align_tmp_res.cigar_info.back().has_realloc = 0;
        for (int i = 0; i < info.cigar.size(); i++) {
            align_tmp_res.cigar_info.back().cigar[i + 1] = info.cigar[i];
        }
    }
    return gapped;
}


__device__ bool gpu_has_shared_substring(const my_string& read_seq, const my_string& ref_seq, int k) {
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

    //my_vector<uint32_t> hash0;
    __shared__ uint32_t g_hash0[50 * 32];
    uint32_t *hash0 = &(g_hash0[threadIdx.x * 50]);
    int N = 0;
    for (int i = 0; i + sub_size < read_seq.size(); i += step_size) {
        uint32_t h = 0;
        for (int j = 0; j < sub_size; ++j) {
            unsigned char base = read_seq[i + j];
            uint8_t code = gpu_nt2int_mod8[base % 8];
            h = (h << 2) | code;
        }
        hash0[N++] = h;
        //assert(N <= 50);
        //hash0.push_back(h);
        //N++;
    }
    quick_sort(&(hash0[0]), N);
    for (int i = 0; i + sub_size < ref_seq.size(); i++) {
        uint32_t h = 0;
        for (int j = 0; j < sub_size; ++j) {
            unsigned char base = ref_seq[i + j];
            uint8_t code = gpu_nt2int_mod8[base % 8];
            h = (h << 2) | code;
        }
        int left = 0, right = N - 1;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (hash0[mid] == h) return true;
            else if (hash0[mid] < h) left = mid + 1;
            else right = mid - 1;
        }
    }
    return false;
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


__device__ void gpu_part2_extend_seed_get_str(
        GPUAlignTmpRes& align_tmp_res,
        int j,
        const GPURead& read1,
        const GPURead& read2,
        const GPUReferences& references,
        int read_id,
        my_vector<TmpTODOInfos>& tmp_todo_infos
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

    tmp_todo_infos.push_back({query_segm_size, ref_segm_size, query.data + query_start, ref.data + ref_start, {0, packed, nam.ref_id, ref_start, ref_segm_size}});
}


__device__ void gpu_part2_rescue_mate_get_str(
        GPUAlignTmpRes& align_tmp_res,
        int j,
        const GPURead& read1,
        const GPURead& read2,
        const GPUReferences& references,
        float mu,
        float sigma,
        int read_id,
        my_vector<TmpTODOInfos>& tmp_todo_infos
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

    tmp_todo_infos.push_back({query_segm_size, ref_segm_size, r_tmp.data + query_start, references.sequences[nam.ref_id].data + ref_start, {0, packed, nam.ref_id, ref_start, ref_segm_size}});
}

__device__ void gpu_part2_extend_seed_get_str(
        GPUAlignTmpRes& align_tmp_res,
        int j,
        const GPURead& read1,
        const GPURead& read2,
        const GPUReferences& references,
        int read_id,
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
        int read_id,
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

__device__ void gpu_rescue_read_part(
        int flag,
        GPUAlignTmpRes& align_tmp_res,
        const GPURead& read2,  // read to be rescued
        const GPURead& read1,  // read that has NAMs
        const AlignmentParameters& aligner_parameters,
        const GPUReferences& references,
        my_vector<Nam>& nams1,
        int max_tries,
        float dropoff,
        int k,
        float mu,
        float sigma,
        size_t max_secondary,
        double secondary_dropoff,
        bool swap_r1r2
) {
    //align_tmp_res.type = flag;
    Nam n_max1 = nams1[0];
    int tries = 0;
    // this loop is safe, loop size is stable
    for (int i = 0; i < nams1.size(); i++) {
        Nam &nam = nams1[i];
        float score_dropoff1 = (float) nam.n_hits / n_max1.n_hits;
        // only consider top hits (as minimap2 does) and break if below dropoff cutoff.
        if (tries >= max_tries || score_dropoff1 < dropoff) {
            break;
        }

        const bool consistent_nam = gpu_reverse_nam_if_needed(nam, read1, references, k);
        // reserve extend and store info
        if(flag == 1) align_tmp_res.is_read1.push_back(true);
        else align_tmp_res.is_read1.push_back(false);
        bool gapped = gpu_extend_seed_part(align_tmp_res, aligner_parameters, nam, references, read1, consistent_nam);

        // Force SW alignment to rescue mate
        if(flag == 1) align_tmp_res.is_read1.push_back(false);
        else align_tmp_res.is_read1.push_back(true);
        bool is_unaligned = gpu_rescue_mate_part(align_tmp_res, aligner_parameters, nam, references, read2, mu, sigma, k);
        tries++;
    }
}

__device__ inline bool gpu_is_proper_nam_pair3(const Nam nam1, const Nam nam2, float mu, float sigma) {
    int a = my_max(0, nam1.ref_start - nam1.query_start);
    int b = my_max(0, nam2.ref_start - nam2.query_start);

    // r1 ---> <---- r2
    bool r1_r2 = nam2.is_rc && (a <= b) && (b - a < mu + 10 * sigma);
    if(r1_r2) return true;

    // r2 ---> <---- r1
    bool r2_r1 = nam1.is_rc && (b <= a) && (a - b < mu + 10 * sigma);
    if(r2_r1) return true;
    return false;
}


__device__ inline bool gpu_is_proper_nam_pair2(const Nam nam1, const Nam nam2, float mu, float sigma) {
    if (nam1.is_rc == nam2.is_rc) {
        return false;
    }
    int a = my_max(0, nam1.ref_start - nam1.query_start);
    int b = my_max(0, nam2.ref_start - nam2.query_start);

    // r1 ---> <---- r2
    bool r1_r2 = nam2.is_rc && (a <= b) && (b - a < mu + 10 * sigma);
    if(r1_r2) return true;

    // r2 ---> <---- r1
    bool r2_r1 = nam1.is_rc && (b <= a) && (a - b < mu + 10 * sigma);
    if(r2_r1) return true;
    return false;
}

__device__ inline bool gpu_is_proper_nam_pair(const Nam nam1, const Nam nam2, float mu, float sigma) {
    if (nam1.ref_id != nam2.ref_id || nam1.is_rc == nam2.is_rc) {
        return false;
    }
    int a = my_max(0, nam1.ref_start - nam1.query_start);
    int b = my_max(0, nam2.ref_start - nam2.query_start);

    // r1 ---> <---- r2
    bool r1_r2 = nam2.is_rc && (a <= b) && (b - a < mu + 10 * sigma);
    //    if(r1_r2) return 1;

    // r2 ---> <---- r1
    bool r2_r1 = nam1.is_rc && (b <= a) && (a - b < mu + 10 * sigma);
    //    if(r2_r1) return 1;
    //    return 0;

    return r1_r2 || r2_r1;
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

struct GPUInsertSizeDistribution {
    float sample_size = 1;
    float mu = 300;
    float sigma = 100;
    float V = 10000;
    float SSE = 10000;

    // Add a new observation
    __device__ __host__ void update(int dist) {
        if (dist >= 2000) {
            return;
        }
        const float e = dist - mu;
        mu += e / sample_size;  // (1.0/(sample_size +1.0)) * (sample_size*mu + d);
        SSE += e * (dist - mu);
        if (sample_size > 1) {
            //d < 1000 ? ((sample_size +1.0)/sample_size) * ( (V*sample_size/(sample_size +1)) + ((mu-d)*(mu-d))/sample_size ) : V;
            V = SSE / (sample_size - 1.0);
        } else {
            V = SSE;
        }
        sigma = sqrtf(V);
        sample_size = sample_size + 1.0;
        if (mu < 0) {
            printf("mu negative, mu: %f sigma: %f SSE: %f sample size: %f\n", mu, sigma, SSE, sample_size);
            assert(false);
        }
        if (SSE < 0) {
            printf("SSE negative, mu: %f sigma: %f SSE: %f sample size: %f\n", mu, sigma, SSE, sample_size);
            assert(false);
        }
    }
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

#define key_mod_val 29

__device__ void gpu_get_best_scoring_nam_pairs(
        my_vector<gpu_NamPair>& joint_nam_scores,
        my_vector<Nam>& nams1,
        my_vector<Nam>& nams2,
        float mu,
        float sigma,
        int max_tries,
        int tid
) {
    int nams1_len = nams1.size();
    int nams2_len = nams2.size();
    my_vector<bool> added_n1(nams1_len);
    my_vector<bool> added_n2(nams2_len);
    for(int i = 0; i < nams1_len; i++) added_n1.push_back(false);
    for(int i = 0; i < nams2_len; i++) added_n2.push_back(false);

    int best_joint_hits = 0;
    for (int i = 0; i < nams1_len; i++) {
        const Nam &nam1 = nams1[i];
        for (int j = 0; j < nams2_len; j++) {
            const Nam &nam2 = nams2[j];
            int joint_hits = nam1.n_hits + nam2.n_hits;
            if (joint_hits < 0.5 * best_joint_hits || joint_nam_scores.size() > max_tries * 2) {
            //if (joint_hits < best_joint_hits / 2) {
                break;
            }
            if (gpu_is_proper_nam_pair(nam1, nam2, mu, sigma)) {
                joint_nam_scores.push_back(gpu_NamPair{joint_hits, &nams1, &nams2, i, j});
                added_n1[i] = 1;
                added_n2[j] = 1;
                best_joint_hits = my_max(joint_hits, best_joint_hits);
            }
        }
        if (joint_nam_scores.size() > max_tries * 2) break;
    }

    // Find high-scoring R1 NAMs that are not part of a proper pair
    Nam dummy_nam;
    dummy_nam.ref_start = -1;
    int best_joint_hits1 = best_joint_hits > 0 ? best_joint_hits : nams1[0].n_hits;
    for(int i = 0; i < my_min(nams1.size(), max_tries); i++) {
    //for(int i = 0; i < nams1.size(); i++) {
        Nam nam1 = nams1[i];
        if (nam1.n_hits < best_joint_hits1 / 2) {
            break;
        }
        if (added_n1[i]) {
            continue;
        }
        joint_nam_scores.push_back(gpu_NamPair{nam1.n_hits, &nams1, &nams2, i, -1});
    }

    // Find high-scoring R2 NAMs that are not part of a proper pair
    int best_joint_hits2 = best_joint_hits > 0 ? best_joint_hits : nams2[0].n_hits;
    for(int i = 0; i < my_min(nams2.size(), max_tries); i++) {
    //for(int i = 0; i < nams2.size(); i++) {
        Nam nam2 = nams2[i];
        if (nam2.n_hits < best_joint_hits2 / 2) {
            break;
        }
        if (added_n2[i]) {
            continue;
        }
        joint_nam_scores.push_back(gpu_NamPair{nam2.n_hits, &nams1, &nams2, -1, i});
    }
    quick_sort_iterative(&(joint_nam_scores[0]), 0, joint_nam_scores.size() - 1, 
            [](const gpu_NamPair &n1, const gpu_NamPair &n2) {
            return n1.score > n2.score;
            }
    );

    //quick_sort_iterative(&(joint_nam_scores[0]), 0, joint_nam_scores.size() - 1, 
    //        [](const gpu_NamPair &n1, const gpu_NamPair &n2) {
    //        if (n1.score != n2.score) return n1.score > n2.score;

    //        // Initialize the dummy nam here
    //        Nam dummy_nam;
    //        dummy_nam.ref_start = -1;

    //        // Ensure that nams1 and nams2 are valid before dereferencing
    //        const my_vector<Nam>& lnams1 = *(n1.nams1);
    //        const my_vector<Nam>& lnams2 = *(n1.nams2);

    //        // Safely access Nam1 and Nam2 objects based on valid indices
    //        const Nam n1_nam1 = (n1.i1 >= 0 && n1.i1 < lnams1.size()) ? lnams1[n1.i1] : dummy_nam;
    //        const Nam n1_nam2 = (n1.i2 >= 0 && n1.i2 < lnams2.size()) ? lnams2[n1.i2] : dummy_nam;
    //        const Nam n2_nam1 = (n2.i1 >= 0 && n2.i1 < lnams1.size()) ? lnams1[n2.i1] : dummy_nam;
    //        const Nam n2_nam2 = (n2.i2 >= 0 && n2.i2 < lnams2.size()) ? lnams2[n2.i2] : dummy_nam;

    //        if (n1_nam1.nam_id != n2_nam1.nam_id) {
    //        return n1_nam1.nam_id < n2_nam1.nam_id;
    //        }
    //        if (n1_nam2.nam_id != n2_nam2.nam_id) {
    //            return n1_nam2.nam_id < n2_nam2.nam_id;
    //        }
    //        return n1_nam1.ref_start < n2_nam1.ref_start;
    //        }
    //);

    return;
}

__device__ static unsigned char gpu_revcomp_table[256] = {
        'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
        'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
        'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
        'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
        'N', 'T', 'N', 'G',  'N', 'N', 'N', 'C',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
        'N', 'N', 'N', 'N',  'A', 'A', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
        'N', 'T', 'N', 'G',  'N', 'N', 'N', 'C',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
        'N', 'N', 'N', 'N',  'A', 'A', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
        'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
        'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
        'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
        'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
        'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
        'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
        'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
        'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N'
};


__device__ void align_PE_part12(
        GPUAlignTmpRes& align_tmp_res,
        const AlignmentParameters& aligner_parameters,
        my_vector<Nam>& nams1,
        my_vector<Nam>& nams2,
        char* seq1, char* rc1, int seq_len1,
        char* seq2, char* rc2, int seq_len2,
        int k,
        const GPUReferences& references,
        float dropoff,
        GPUInsertSizeDistribution& isize_est,
        unsigned max_tries,
        size_t max_secondary,
        int type,
        int read_id,
        int* d_todo_cnt,
        char* d_query_ptr, char* d_ref_ptr,
        int* d_query_offset, int* d_ref_offset
) {
    //assert(!nams1.empty() && nams2.empty());
    const auto mu = isize_est.mu;
    const auto sigma = isize_est.sigma;
    GPURead read1{seq1, rc1, seq_len1};
    GPURead read2{seq2, rc2, seq_len2};
    double secondary_dropoff = 2 * aligner_parameters.mismatch + aligner_parameters.gap_open;
//    align_tmp_res.type = 1;
    gpu_rescue_read_part(
            type, align_tmp_res, type == 1 ? read2 : read1, type == 1 ? read1 : read2, aligner_parameters, references, type == 1 ? nams1 : nams2, max_tries, dropoff, k, mu,
            sigma, max_secondary, secondary_dropoff, type == 1 ? false : true
    );
//    my_vector<TmpTODOInfos> tmp_todo_infos;
    for (size_t j = 0; j < align_tmp_res.todo_nams.size(); j += 2) {
        assert(align_tmp_res.is_extend_seed[j]);
        if (align_tmp_res.type == 1)
            assert(align_tmp_res.is_read1[j]);
        else
            assert(!align_tmp_res.is_read1[j]);
        if (!align_tmp_res.done_align[j]) {
            gpu_part2_extend_seed_get_str(
                    align_tmp_res, j, read1, read2, references, read_id,
//                    tmp_todo_infos
                    d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset
            );
        }
        assert(!align_tmp_res.is_extend_seed[j + 1]);
        if (align_tmp_res.type == 1)
            assert(!align_tmp_res.is_read1[j + 1]);
        else
            assert(align_tmp_res.is_read1[j + 1]);
        if (!align_tmp_res.done_align[j + 1]) {
            gpu_part2_rescue_mate_get_str(
                    align_tmp_res, j + 1, read1, read2, references, mu, sigma, read_id,
                    d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset
//                    tmp_todo_infos
            );
        }
    }
//    int global_id_start = atomicAdd(d_todo_cnt, tmp_todo_infos.size());
//    int total_query_offset = 0;
//    int total_ref_offset = 0;
//    for (int i = 0; i < tmp_todo_infos.size(); i++) {
//        TmpTODOInfos& tmp_todo_info = tmp_todo_infos[i];
//        int q_len = tmp_todo_info.q_len;
//        q_len = ((q_len + 7) & ~7);
//        int r_len = tmp_todo_info.r_len;
//        r_len = ((r_len + 7) & ~7);
//        total_query_offset += q_len;
//        total_ref_offset += r_len;
//        align_tmp_res.todo_infos.push_back({
//                                                   global_id_start + i,
//                                                   tmp_todo_info.todo_info.read_info,
//                                                   tmp_todo_info.todo_info.ref_id,
//                                                   tmp_todo_info.todo_info.r_begin,
//                                                   tmp_todo_info.todo_info.r_len
//                                           });
//    }
//    int query_offset = atomicAdd(d_query_offset, total_query_offset);
//    int ref_offset = atomicAdd(d_ref_offset, total_ref_offset);
//    for (int i = 0; i < tmp_todo_infos.size(); i++) {
//        TmpTODOInfos& tmp_todo_info = tmp_todo_infos[i];
//        int query_segm_size = tmp_todo_info.q_len;
//        int ref_segm_size = tmp_todo_info.r_len;
//        int q_len = ((query_segm_size + 7) & ~7);
//        int r_len = ((ref_segm_size + 7) & ~7);
//        memcpy(d_query_ptr + query_offset, tmp_todo_info.q_ptr, query_segm_size);
//        memcpy(d_ref_ptr + ref_offset, tmp_todo_info.r_ptr, ref_segm_size);
//        memset(d_query_ptr + query_offset + query_segm_size, 0x4E, q_len - query_segm_size);
//        memset(d_ref_ptr + ref_offset + ref_segm_size, 0x4E, r_len - ref_segm_size);
//        query_offset += q_len;
//        ref_offset += r_len;
//    }
    return;
}


__device__ void align_PE_part3(
        GPUAlignTmpRes& align_tmp_res,
        const AlignmentParameters& aligner_parameters,
        my_vector<Nam>& nams1,
        my_vector<Nam>& nams2,
        char* seq1, char* rc1, int seq_len1,
        char* seq2, char* rc2, int seq_len2,
        int k,
        const GPUReferences& references,
        float dropoff,
        GPUInsertSizeDistribution& isize_est,
        unsigned max_tries,
        size_t max_secondary,
        int read_id,
        int* d_todo_cnt,
        char* d_query_ptr, char* d_ref_ptr,
        int* d_query_offset, int* d_ref_offset
) {
    assert(!nams1.empty() && !nams2.empty());
    const auto mu = isize_est.mu;
    const auto sigma = isize_est.sigma;
    GPURead read1{seq1, rc1, seq_len1};
    GPURead read2{seq2, rc2, seq_len2};
    double secondary_dropoff = 2 * aligner_parameters.mismatch + aligner_parameters.gap_open;

    // Deal with the typical case that both reads map uniquely and form a proper pair
    assert(gpu_top_dropoff(nams1) < dropoff && gpu_top_dropoff(nams2) < dropoff && gpu_is_proper_nam_pair(nams1[0], nams2[0], mu, sigma));
//    align_tmp_res.type = 3;
    Nam n_max1 = nams1[0];
    Nam n_max2 = nams2[0];

    bool consistent_nam1 = gpu_reverse_nam_if_needed(n_max1, read1, references, k);
    bool consistent_nam2 = gpu_reverse_nam_if_needed(n_max2, read2, references, k);

    align_tmp_res.is_read1.push_back(true);
    bool gapped1 = gpu_extend_seed_part(align_tmp_res, aligner_parameters, n_max1, references, read1, consistent_nam1);


    align_tmp_res.is_read1.push_back(false);
    bool gapped2 = gpu_extend_seed_part(align_tmp_res, aligner_parameters, n_max2, references, read2, consistent_nam2);

    int mapq1 = gpu_get_mapq(nams1, n_max1);
    int mapq2 = gpu_get_mapq(nams2, n_max2);
    align_tmp_res.mapq1 = mapq1;
    align_tmp_res.mapq2 = mapq2;

    align_tmp_res.type3_isize_val = -1;
    if(!gapped1 && !gapped2) {
        int res_size = align_tmp_res.align_res.size();
        if(res_size < 2) {
            printf("align_tmp_res.align_res.size error %d\n", res_size);
        }
        auto alignment1 = align_tmp_res.align_res[res_size - 2];
        auto alignment2 = align_tmp_res.align_res[res_size - 1];
        if(alignment1.gapped || alignment2.gapped) {
            printf("alignment gapped error\n");
        }
        bool is_proper = gpu_is_proper_pair(alignment1, alignment2, mu, sigma);
        if ((isize_est.sample_size < 400) && (alignment1.edit_distance + alignment2.edit_distance < 3) && is_proper) {
            align_tmp_res.type3_isize_val = my_abs(alignment1.ref_start - alignment2.ref_start);
            //isize_est.update(std::abs(alignment1.ref_start - alignment2.ref_start));
        }
    }

    assert(align_tmp_res.is_extend_seed[0]);
    assert(align_tmp_res.is_read1[0]);
//    my_vector<TmpTODOInfos> tmp_todo_infos;
    if (!align_tmp_res.done_align[0]) {
        gpu_part2_extend_seed_get_str(
                align_tmp_res, 0, read1, read2, references, read_id,
                d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset
//                tmp_todo_infos
        );
    }
    assert(align_tmp_res.is_extend_seed[1]);
    assert(!align_tmp_res.is_read1[1]);
    if (!align_tmp_res.done_align[1]) {
        gpu_part2_extend_seed_get_str(
                align_tmp_res, 1, read1, read2, references, read_id,
                d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset
//                tmp_todo_infos
        );
    }
//    int global_id_start = atomicAdd(d_todo_cnt, tmp_todo_infos.size());
//    int total_query_offset = 0;
//    int total_ref_offset = 0;
//    for (int i = 0; i < tmp_todo_infos.size(); i++) {
//        TmpTODOInfos& tmp_todo_info = tmp_todo_infos[i];
//        int q_len = tmp_todo_info.q_len;
//        q_len = ((q_len + 7) & ~7);
//        int r_len = tmp_todo_info.r_len;
//        r_len = ((r_len + 7) & ~7);
//        total_query_offset += q_len;
//        total_ref_offset += r_len;
//        align_tmp_res.todo_infos.push_back({
//                                                   global_id_start + i,
//                                                   tmp_todo_info.todo_info.read_info,
//                                                   tmp_todo_info.todo_info.ref_id,
//                                                   tmp_todo_info.todo_info.r_begin,
//                                                   tmp_todo_info.todo_info.r_len
//                                           });
//    }
//    int query_offset = atomicAdd(d_query_offset, total_query_offset);
//    int ref_offset = atomicAdd(d_ref_offset, total_ref_offset);
//    for (int i = 0; i < tmp_todo_infos.size(); i++) {
//        TmpTODOInfos& tmp_todo_info = tmp_todo_infos[i];
//        int query_segm_size = tmp_todo_info.q_len;
//        int ref_segm_size = tmp_todo_info.r_len;
//        int q_len = ((query_segm_size + 7) & ~7);
//        int r_len = ((ref_segm_size + 7) & ~7);
//        memcpy(d_query_ptr + query_offset, tmp_todo_info.q_ptr, query_segm_size);
//        memcpy(d_ref_ptr + ref_offset, tmp_todo_info.r_ptr, ref_segm_size);
//        memset(d_query_ptr + query_offset + query_segm_size, 0x4E, q_len - query_segm_size);
//        memset(d_ref_ptr + ref_offset + ref_segm_size, 0x4E, r_len - ref_segm_size);
//        query_offset += q_len;
//        ref_offset += r_len;
//    }
//    return;

}

__device__ void align_PE_part4(
        GPUAlignTmpRes& align_tmp_res,
        const AlignmentParameters& aligner_parameters,
        my_vector<Nam>& nams1,
        my_vector<Nam>& nams2,
        char* seq1, char* rc1, int seq_len1,
        char* seq2, char* rc2, int seq_len2,
        int k,
        const GPUReferences& references,
        float dropoff,
        GPUInsertSizeDistribution& isize_est,
        int max_tries,
        size_t max_secondary,
        int tid,
        int read_id,
        int* d_todo_cnt,
        char* d_query_ptr, char* d_ref_ptr,
        int* d_query_offset, int* d_ref_offset
) {
    assert(!nams1.empty() && !nams2.empty());

    const auto mu = isize_est.mu;
    const auto sigma = isize_est.sigma;
    GPURead read1{seq1, rc1, seq_len1};
    GPURead read2{seq2, rc2, seq_len2};
    double secondary_dropoff = 2 * aligner_parameters.mismatch + aligner_parameters.gap_open;

    Nam dummy_nam;
    dummy_nam.ref_start = -1;

    my_vector<gpu_NamPair> joint_nam_scores(nams1.size() + nams2.size());
    gpu_get_best_scoring_nam_pairs(joint_nam_scores, nams1, nams2, mu, sigma, max_tries, tid);

    if (joint_nam_scores.size() > max_tries) joint_nam_scores.length = max_tries;

    int nams1_len = nams1.size();
    int nams2_len = nams2.size();
    my_vector<bool> is_aligned1(nams1_len + 1);
    my_vector<bool> is_aligned2(nams2_len + 1);
    for (int i = 0; i <= nams1_len; i++) is_aligned1.push_back(false);
    for (int i = 0; i <= nams2_len; i++) is_aligned2.push_back(false);

    {
        Nam n1_max = nams1[0];
        bool consistent_nam1 = gpu_reverse_nam_if_needed(n1_max, read1, references, k);
        align_tmp_res.is_read1.push_back(true);
        bool gapped1 = gpu_extend_seed_part(align_tmp_res, aligner_parameters, n1_max, references, read1, consistent_nam1);
        is_aligned1[0] = 1;

        Nam n2_max = nams2[0];
        bool consistent_nam2 = gpu_reverse_nam_if_needed(n2_max, read2, references, k);
        align_tmp_res.is_read1.push_back(false);
        bool gapped2 = gpu_extend_seed_part(align_tmp_res, aligner_parameters, n2_max, references, read2, consistent_nam2);
        is_aligned2[0] = 1;
    }

    // Turn pairs of high-scoring NAMs into pairs of alignments
    int high_scores_size = 0;
    double max_score = joint_nam_scores[0].score;
    align_tmp_res.type4_loop_size = 0;
    for(int i = 0; i < joint_nam_scores.size(); i++) {
        double score_ = joint_nam_scores[i].score;
        int id1 = joint_nam_scores[i].i1 == -1 ? nams1_len : joint_nam_scores[i].i1;
        int id2 = joint_nam_scores[i].i2 == -1 ? nams2_len : joint_nam_scores[i].i2;
        Nam n1 = joint_nam_scores[i].i1 == -1 ? dummy_nam : nams1[joint_nam_scores[i].i1];
        Nam n2 = joint_nam_scores[i].i2 == -1 ? dummy_nam : nams2[joint_nam_scores[i].i2];

        float score_dropoff = (float) score_ / max_score;
        if (high_scores_size >= max_tries || score_dropoff < dropoff) {
            break;
        }

        align_tmp_res.type4_nams.push_back(n1);
        align_tmp_res.type4_nams.push_back(n2);
        align_tmp_res.type4_loop_size++;

        if (n1.ref_start >= 0) {
            if (is_aligned1[id1] == 1) {

            } else {
                bool consistent_nam = gpu_reverse_nam_if_needed(n1, read1, references, k);
                align_tmp_res.is_read1.push_back(true);
                bool gapped = gpu_extend_seed_part(align_tmp_res, aligner_parameters, n1, references, read1, consistent_nam);
                is_aligned1[id1] = 1;
            }
        } else {
            gpu_reverse_nam_if_needed(n2, read2, references, k);
            align_tmp_res.is_read1.push_back(true);
            bool is_unaligned = gpu_rescue_mate_part(align_tmp_res, aligner_parameters, n2, references, read1, mu, sigma, k);
        }

        if (n2.ref_start >= 0) {
            if (is_aligned2[id2] == 1) {

            } else {
                bool consistent_nam = gpu_reverse_nam_if_needed(n2, read2, references, k);
                align_tmp_res.is_read1.push_back(false);
                bool gapped = gpu_extend_seed_part(align_tmp_res, aligner_parameters, n2, references, read2, consistent_nam);
                is_aligned2[id2] = 1;
            }
        } else {
            gpu_reverse_nam_if_needed(n1, read1, references, k);
            align_tmp_res.is_read1.push_back(false);
            bool is_unaligned = gpu_rescue_mate_part(align_tmp_res, aligner_parameters, n1, references, read2, mu, sigma, k);
        }
        high_scores_size++;
    }
//    my_vector<TmpTODOInfos> tmp_todo_infos;
    for (size_t j = 0; j < align_tmp_res.todo_nams.size(); j++) {
        if (!align_tmp_res.done_align[j]) {
            if (align_tmp_res.is_extend_seed[j]) {
                gpu_part2_extend_seed_get_str(
                        align_tmp_res, j, read1, read2, references, read_id,
                        d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset
//                        tmp_todo_infos
                );
            } else {
                gpu_part2_rescue_mate_get_str(
                        align_tmp_res, j, read1, read2, references, mu, sigma, read_id,
                        d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset
//                        tmp_todo_infos
                );
            }
        }
    }
//    int global_id_start = atomicAdd(d_todo_cnt, tmp_todo_infos.size());
//    int total_query_offset = 0;
//    int total_ref_offset = 0;
//    for (int i = 0; i < tmp_todo_infos.size(); i++) {
//        TmpTODOInfos& tmp_todo_info = tmp_todo_infos[i];
//        int q_len = tmp_todo_info.q_len;
//        q_len = ((q_len + 7) & ~7);
//        int r_len = tmp_todo_info.r_len;
//        r_len = ((r_len + 7) & ~7);
//        total_query_offset += q_len;
//        total_ref_offset += r_len;
//        align_tmp_res.todo_infos.push_back({
//                                                   global_id_start + i,
//                                                   tmp_todo_info.todo_info.read_info,
//                                                   tmp_todo_info.todo_info.ref_id,
//                                                   tmp_todo_info.todo_info.r_begin,
//                                                   tmp_todo_info.todo_info.r_len
//                                           });
//    }
//    int query_offset = atomicAdd(d_query_offset, total_query_offset);
//    int ref_offset = atomicAdd(d_ref_offset, total_ref_offset);
//    for (int i = 0; i < tmp_todo_infos.size(); i++) {
//        TmpTODOInfos& tmp_todo_info = tmp_todo_infos[i];
//        int query_segm_size = tmp_todo_info.q_len;
//        int ref_segm_size = tmp_todo_info.r_len;
//        int q_len = ((query_segm_size + 7) & ~7);
//        int r_len = ((ref_segm_size + 7) & ~7);
//        memcpy(d_query_ptr + query_offset, tmp_todo_info.q_ptr, query_segm_size);
//        memcpy(d_ref_ptr + ref_offset, tmp_todo_info.r_ptr, ref_segm_size);
//        memset(d_query_ptr + query_offset + query_segm_size, 0x4E, q_len - query_segm_size);
//        memset(d_ref_ptr + ref_offset + ref_segm_size, 0x4E, r_len - ref_segm_size);
//        query_offset += q_len;
//        ref_offset += r_len;
//    }
    return;
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
        int tid,
        int read_id,
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
                        align_tmp_res, j, read, read, references, read_id,
                        d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset
//                        tmp_todo_infos
                );
            } else {
                assert(0);
            }
        }
    }

    return;
}

__device__ void check_hits(my_vector<my_pair<int, Hit>> &hits_per_ref) {
    // check if sort is correct
    if (hits_per_ref.size() < 2) return;
    for(int i = 0; i < hits_per_ref.size() - 1; i++) {
        //        if(hits_per_ref[i].first > hits_per_ref[i + 1].first) {
        //            printf("sort error [%d,%d] [%d,%d]\n", hits_per_ref[i].first, hits_per_ref[i].second.query_start, hits_per_ref[i + 1].first, hits_per_ref[i + 1].second.query_start);
        //            assert(false);
        //        }
        if(hits_per_ref[i].first == hits_per_ref[i + 1].first && hits_per_ref[i].second.query_start > hits_per_ref[i + 1].second.query_start) {
            printf("sort error [%d,%d] [%d,%d]\n", hits_per_ref[i].first, hits_per_ref[i].second.query_start, hits_per_ref[i + 1].first, hits_per_ref[i + 1].second.query_start);
            assert(false);
        }
    }
}

__device__ void check_nams(my_vector<Nam> &nams) {
    // check if sort is correct
    if (nams.size() < 2) return;
    for(int i = 0; i < nams.size() - 1; i++) {
        if(nams[i].score < nams[i + 1].score) {
            printf("sort error [%lf,%d] [%lf,%d]\n", nams[i].score, nams[i].query_end, nams[i + 1].score, nams[i + 1].query_end);
            assert(false);
        }
        //        if(nams[i].score == nams[i + 1].score && nams[i].query_end > nams[i + 1].query_end) {
        //            printf("sort error [%lf,%d] [%lf,%d]\n", nams[i].score, nams[i].query_end, nams[i + 1].score, nams[i + 1].query_end);
        //            assert(false);
        //        }
    }
}

__device__ void sort_hits_single(
        my_vector<my_pair<int, Hit>>& hits_per_ref
) {
    //bubble_sort(&(hits_per_ref[0]), hits_per_ref.size());
    quick_sort(&(hits_per_ref[0]), hits_per_ref.size());
}

__device__ void topk_quick_sort(my_vector<Nam>& nams, int mx_num) {
    if (nams.size() == 0) return;
    //const int MAX_STACK = 64;
    //int left_stack[MAX_STACK];
    //int right_stack[MAX_STACK];
    if (nams.size() < mx_num) mx_num = nams.size();
    my_vector<int>ll(nams.size() * 2);
    my_vector<int>rr(nams.size() * 2);
    int *left_stack = ll.data;
    int *right_stack = rr.data;
    int top = -1;

    int left = 0, right = nams.size() - 1;
    left_stack[++top] = left;
    right_stack[top] = right;

    while (top >= 0) {
        left = left_stack[top];
        right = right_stack[top--];

        if (left >= right) continue;

        Nam pivot = nams[right];
        int i = left - 1;

        for (int j = left; j < right; ++j) {
            if (nams[j].score > pivot.score) {
                ++i;
                Nam tmp = nams[i];
                nams[i] = nams[j];
                nams[j] = tmp;
            }
        }

        Nam tmp = nams[i + 1];
        nams[i + 1] = nams[right];
        nams[right] = tmp;

        int pivot_index = i + 1;

        if (pivot_index > mx_num) {
            left_stack[++top] = left;
            right_stack[top] = pivot_index - 1;
        } else if (pivot_index < mx_num - 1) {
            left_stack[++top] = pivot_index + 1;
            right_stack[top] = right;
        } else {
        }
    }
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

__device__ void gpu_shuffle_top_nams(my_vector<Nam>& nams) {
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

__device__ void sort_nams_single(
        my_vector<Nam>& nams
) {
    //bubble_sort(&(hits_per_ref[0]), hits_per_ref.size());
    quick_sort_iterative(&(nams[0]), 0, nams.size() - 1, [](const Nam &n1, const Nam &n2) {
        //if(n1.score != n2.score) return n1.score > n2.score;
        if(n1.n_hits != n2.n_hits) return n1.n_hits > n2.n_hits;
        if(n1.query_end != n2.query_end) return n1.query_end < n2.query_end;
        if(n1.query_start != n2.query_start) return n1.query_start < n2.query_start;
        if(n1.ref_end != n2.ref_end) return n1.ref_end < n2.ref_end;
        if(n1.ref_start != n2.ref_start) return n1.ref_start < n2.ref_start;
        if(n1.ref_id != n2.ref_id) return n1.ref_id < n2.ref_id;
        return n1.is_rc < n2.is_rc;
    });
}

__device__ void sort_nams_single2(
        my_vector<Nam>& nams
) {
    //quick_sort(&(nams[0]), nams.size());
    quick_sort_iterative(&(nams[0]), 0, nams.size() - 1, [](const Nam &n1, const Nam &n2) {
        if(n1.is_rc != n2.is_rc) return n1.is_rc < n2.is_rc;
        if(n1.ref_id != n2.ref_id) return n1.ref_id < n2.ref_id;
        //if(n1.score != n2.score) return n1.score > n2.score;
        if(n1.n_hits != n2.n_hits) return n1.n_hits > n2.n_hits;
        if(n1.query_end != n2.query_end) return n1.query_end < n2.query_end;
        if(n1.query_start != n2.query_start) return n1.query_start < n2.query_start;
        if(n1.ref_end != n2.ref_end) return n1.ref_end < n2.ref_end;
        if(n1.ref_start != n2.ref_start) return n1.ref_start < n2.ref_start;
        //return n1.is_rc < n2.is_rc;
        return true;
    });


}

__device__ int find_ref_ids(int ref_id, int* head, ref_ids_edge* edges) {
    int key = ref_id % key_mod_val;
    for (int i = head[key]; i != -1; i = edges[i].pre) {
        if (edges[i].ref_id == ref_id) return i;
    }
    return -1;
}

__device__ void sort_nams_get_k(my_vector<Nam>& nams, int mx_num) {
    int limit = mx_num;
    if (limit > nams.size()) limit = nams.size();
    for (int i = 0; i < limit; ++i) {
        int max_idx = i;
        for (int j = i + 1; j < nams.size(); ++j) {
            if (nams[j].score > nams[max_idx].score) {
                max_idx = j;
            }
        }
        if (max_idx != i) {
            Nam tmp = nams[i];
            nams[i] = nams[max_idx];
            nams[max_idx] = tmp;
        }
    }
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

__device__ void sort_nams_by_score2(my_vector<Nam>& nams, int mx_num) {
    my_vector<my_pair<int, my_vector<Nam>*>> all_nams;
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
            my_vector<Nam>* bucket = (my_vector<Nam>*)my_malloc(sizeof(my_vector<Nam>));
            bucket->init(32);
            all_nams.push_back({score_key, bucket});
        }
        all_nams[score_rank].second->push_back(nams[i]);
    }
    nams.clear();
    quick_sort_iterative(&(all_nams[0]), 0, all_nams.size() - 1,
                         [] (const my_pair<int, my_vector<Nam>*>& a, const my_pair<int, my_vector<Nam>*>& b) {
                             return a.first > b.first;
                         });
    for (int i = 0; i < all_nams.size(); i++) {
        for (int j = 0; j < all_nams[i].second->size(); j++) {
            if (nams.size() == mx_num) break;
            nams.push_back((*all_nams[i].second)[j]);
        }
        all_nams[i].second->release();
        my_free(all_nams[i].second);
    }
    my_free(head);
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
    for(int i = 0; i < all_hits.size(); i++) {
        for(int j = 0; j < all_hits[i].second->size(); j++) {
            hits_per_ref.push_back({all_hits[i].first, (*all_hits[i].second)[j]});
        }
        all_hits[i].second->release();
    }
    my_free(head);
    my_free(all_vecs);
}

__device__ void sort_hits_by_refid2(
        my_vector<my_pair<int, Hit>>& hits_per_ref
) {
    my_vector<my_pair<int, my_vector<Hit>*>> all_hits;
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
            my_vector<Hit>* hits = (my_vector<Hit>*)my_malloc(sizeof(my_vector<Hit>));
            hits->init(32);
            all_hits.push_back({ref_id, hits});
        }
        all_hits[find_ref_id_rank].second->push_back(hits_per_ref[i].second);
    }
//    quick_sort(&(all_hits[0]), all_hits.size());
    hits_per_ref.clear();
    for(int i = 0; i < all_hits.size(); i++) {
        for(int j = 0; j < all_hits[i].second->size(); j++) {
            hits_per_ref.push_back({all_hits[i].first, (*all_hits[i].second)[j]});
        }
        all_hits[i].second->release();
        my_free(all_hits[i].second);
    }
    my_free(head);
}

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
    //int mx_hits_per_ref = 0;
    //for (int i = 0; i < each_ref_size.size(); i++) {
    //    mx_hits_per_ref = my_max(mx_hits_per_ref, each_ref_size[i]);
    //}

    my_vector<Nam> open_nams;
    //(mx_hits_per_ref);

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
            my_vector<bool> is_added(i_size);
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


__device__ void merge_hits(
        my_vector<my_pair<int, Hit>>& hits_per_ref,
        int k,
        bool is_revcomp,
        my_vector<Nam>& nams
) {
    if(hits_per_ref.size() == 0) return;
    int num_hits = hits_per_ref.size();

    int ref_num = 0;
    my_vector<int> each_ref_size;
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
//        if (diff < min_diff || diff == 0) {
            hits_per_ref.push_back({d_randstrobes[position].reference_index(), Hit{query_start, query_end, ref_start, ref_end}});
            min_diff = diff;
        }
    }
}


#define GPU_thread_task_size 1

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
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int l_range = global_id * GPU_thread_task_size;
    int r_range = l_range + GPU_thread_task_size;
    if (r_range > num_tasks) r_range = num_tasks;
    for (int id = l_range; id < r_range; id++) {
        int real_id = global_todo_ids[id];
        my_vector<my_pair<int, Hit>>* hits_per_ref0;
        my_vector<my_pair<int, Hit>>* hits_per_ref1;
        hits_per_ref0 = (my_vector<my_pair<int, Hit>>*)my_malloc(sizeof(my_vector<my_pair<int, Hit>>));
        hits_per_ref1 = (my_vector<my_pair<int, Hit>>*)my_malloc(sizeof(my_vector<my_pair<int, Hit>>));
        hits_per_ref0->init(32);
        hits_per_ref1->init(32);

        my_vector<RescueHit> hits_t0;
        my_vector<RescueHit> hits_t1;
        for (int i = 0; i < global_randstrobes[real_id].size(); i++) {
            QueryRandstrobe q = global_randstrobes[real_id][i];
            //size_t position = gpu_find(d_randstrobes, d_randstrobe_start_indices, q.hash, bits);
            size_t position = q.hash;
            if (position != static_cast<size_t>(-1)) {
                if(position >= d_randstrobes_size) {
                    printf("position > d_randstrobes_size : %llu %llu\n", position, d_randstrobes_size);
                    assert(false);
                }
                unsigned int count = gpu_get_count(d_randstrobes, d_randstrobe_start_indices, position, bits);
                RescueHit rh{position, count, q.start, q.end};
                if(q.is_reverse) hits_t1.push_back(rh);
                else hits_t0.push_back(rh);
            }
        }
        global_randstrobes[real_id].release();
        quick_sort(&(hits_t0[0]), hits_t0.size());
        quick_sort(&(hits_t1[0]), hits_t1.size());

#define pre_sort

#ifdef pre_sort
        int cnt0 = 0, cnt1 = 0;
        for (int i = 0; i < hits_t0.size(); i++) {
            RescueHit &rh = hits_t0[i];
            if ((rh.count > rescue_cutoff && cnt0 >= 5) || rh.count > 100) {
                break;
            }
            cnt0++;
        }
        for (int i = 0; i < hits_t1.size(); i++) {
            RescueHit &rh = hits_t1[i];
            if ((rh.count > rescue_cutoff && cnt1 >= 5) || rh.count > 100) {
                break;
            }
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
            add_to_hits_per_ref(*hits_per_ref0, rh.query_start, rh.query_end, rh.position, d_randstrobes, d_randstrobes_size, index_para->syncmer.k);
        }
        for (int i = 0; i < cnt1; i++) {
            RescueHit &rh = hits_t1[i];
            add_to_hits_per_ref(*hits_per_ref1, rh.query_start, rh.query_end, rh.position, d_randstrobes, d_randstrobes_size, index_para->syncmer.k);
        }
#else
        int cnt = 0;
        for (int i = 0; i < hits_t0.size(); i++) {
            RescueHit &rh = hits_t0[i];
            if ((rh.count > rescue_cutoff && cnt >= 5) || rh.count > 100) {
                break;
            }
            add_to_hits_per_ref(*hits_per_ref0, rh.query_start, rh.query_end, rh.position, d_randstrobes, d_randstrobes_size, index_para->syncmer.k);
            cnt++;
        }
        cnt = 0;
        for (int i = 0; i < hits_t1.size(); i++) {
            RescueHit &rh = hits_t1[i];
            if ((rh.count > rescue_cutoff && cnt >= 5) || rh.count > 100) {
                break;
            }
            add_to_hits_per_ref(*hits_per_ref1, rh.query_start, rh.query_end, rh.position, d_randstrobes, d_randstrobes_size, index_para->syncmer.k);
            cnt++;
        }
#endif
//        global_hits_num[real_id] = hits_per_ref0->size() + hits_per_ref1->size();
        hits_per_ref0s[real_id] = *hits_per_ref0;
        hits_per_ref1s[real_id] = *hits_per_ref1;
        my_free(hits_per_ref0);
        my_free(hits_per_ref1);
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
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int l_range = global_id * GPU_thread_task_size;
    int r_range = l_range + GPU_thread_task_size;
    if (r_range > num_tasks) r_range = num_tasks;
    for (int id = l_range; id < r_range; id++) {
        int real_id = global_todo_ids[id];
        my_vector<Nam> *nams = (my_vector<Nam>*)my_malloc(sizeof(my_vector<Nam>));
        nams->init(32);
        salign_merge_hits(hits_per_ref0s[real_id], index_para->syncmer.k, 0, *nams);
        salign_merge_hits(hits_per_ref1s[real_id], index_para->syncmer.k, 1, *nams);
        //merge_hits(hits_per_ref0s[real_id], index_para->syncmer.k, 0, *nams);
        //merge_hits(hits_per_ref1s[real_id], index_para->syncmer.k, 1, *nams);
        global_nams[real_id] = *nams;
        my_free(nams);
        hits_per_ref0s[real_id].release();
        hits_per_ref1s[real_id].release();
    }
}


__global__ void gpu_get_randstrobes(
        int num_tasks,
        int read_num,
        int *pre_sum,
        int *lens,
        char *all_seqs,
        IndexParameters *index_para,
        int *randstrobe_sizes,
        uint64_t *hashes,
        my_vector<QueryRandstrobe>* global_randstrobes
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int l_range = global_id * GPU_thread_task_size;
    int r_range = l_range + GPU_thread_task_size;
    if (r_range > num_tasks) r_range = num_tasks;
    for (int id = l_range; id < r_range; id++) {
        int read_id = id % read_num;
        int is_read2 = id / read_num;
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
        my_vector<uint64_t>gpu_qs(len * 2);
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

        my_vector<QueryRandstrobe> *randstrobes;
        randstrobes = (my_vector<QueryRandstrobe>*)my_malloc(sizeof(my_vector<QueryRandstrobe>));
        randstrobes->init((my_max(syncmers.size() - w_min, 0)) * 2);


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
            randstrobes->push_back(
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
            randstrobes->push_back(
                    QueryRandstrobe{
                            gpu_randstrobe_hash(strobe1.hash, strobe2.hash), static_cast<uint32_t>(strobe1.position),
                            static_cast<uint32_t>(strobe2.position) + index_para->syncmer.k, true
                    }
            );
        }
        global_randstrobes[id] = *randstrobes;
        my_free(randstrobes);
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
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int l_range = global_id * GPU_thread_task_size;
    int r_range = l_range + GPU_thread_task_size;
    if (r_range > num_tasks) r_range = num_tasks;
    for (int id = l_range; id < r_range; id++) {
        int sum_seeds0 = 0;
        int sum_seeds1 = 0;
        for (int i = 0; i < global_randstrobes[id].size(); i++) {
            if (global_randstrobes[id][i].is_reverse) {
                sum_seeds1++;
            } else {
                sum_seeds0++;
            }
        }
        my_vector<my_pair<int, Hit>>* hits_per_ref0;
        my_vector<my_pair<int, Hit>>* hits_per_ref1;
        hits_per_ref0 = (my_vector<my_pair<int, Hit>>*)my_malloc(sizeof(my_vector<my_pair<int, Hit>>));
        hits_per_ref1 = (my_vector<my_pair<int, Hit>>*)my_malloc(sizeof(my_vector<my_pair<int, Hit>>));
        hits_per_ref0->init(32);
        hits_per_ref1->init(32);

        uint64_t local_total_hits = 0;
        uint64_t local_nr_good_hits = 0;
        for (int i = 0; i < global_randstrobes[id].size(); i++) {
            QueryRandstrobe q = global_randstrobes[id][i];
            size_t position = q.hash;
            if (position != static_cast<size_t>(-1)) {
                local_total_hits++;
                bool res = gpu_is_filtered(d_randstrobes, d_randstrobes_size, position, filter_cutoff);
                if (res) continue;
                local_nr_good_hits++;
                if(q.is_reverse) {
                    add_to_hits_per_ref(*hits_per_ref1, q.start, q.end, position, d_randstrobes, d_randstrobes_size, index_para->syncmer.k);
                } else {
                    add_to_hits_per_ref(*hits_per_ref0, q.start, q.end, position, d_randstrobes, d_randstrobes_size, index_para->syncmer.k);
                }
            }
        }
        float nonrepetitive_fraction = local_total_hits > 0 ? ((float) local_nr_good_hits) / ((float) local_total_hits) : 1.0;

        if (nonrepetitive_fraction < 0.7 || hits_per_ref0->size() + hits_per_ref1->size() == 0) {
            hits_per_ref0->release();
            hits_per_ref1->release();
        } else {
            global_randstrobes[id].release();
        }
//        global_hits_num[id] = hits_per_ref0->size() + hits_per_ref1->size();
        hits_per_ref0s[id] = *hits_per_ref0;
        hits_per_ref1s[id] = *hits_per_ref1;
        my_free(hits_per_ref0);
        my_free(hits_per_ref1);
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
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int l_range = global_id * GPU_thread_task_size;
    int r_range = l_range + GPU_thread_task_size;
    if (r_range > num_tasks) r_range = num_tasks;
    for (int id = l_range; id < r_range; id++) {
        for (int i = 0; i < global_randstrobes[id].size(); i++) {
            QueryRandstrobe q = global_randstrobes[id][i];
            size_t position = gpu_find(d_randstrobes, d_randstrobe_start_indices, q.hash, bits);
            //if(position != static_cast<size_t>(-1) && position >= d_randstrobes_size) {
            //    printf("position GG %zu %zu\n", position, d_randstrobes_size);
            //    assert(false);
            //}
            global_randstrobes[id][i].hash = position;
        }
    }
}

__global__ void gpu_sort_nams(
        int num_tasks,
        my_vector<Nam> *global_nams,
        MappingParameters *mapping_parameters,
        int is_se
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int l_range = global_id * GPU_thread_task_size;
    int r_range = l_range + GPU_thread_task_size;
    if (r_range > num_tasks) r_range = num_tasks;
    int read_num = num_tasks / 2;
    for (int id = l_range; id < r_range; id++) {
        int max_tries = mapping_parameters->max_tries;
        if (is_se) {
            sort_nams_by_score(global_nams[id], max_tries);
            global_nams[id].length = my_min(global_nams[id].length, max_tries);
        } else {
            //sort_nams_by_score(global_nams[id], max_tries * 2);
            //global_nams[id].length = my_min(global_nams[id].length, max_tries * 2);
            sort_nams_by_score(global_nams[id], 1e9);
        }
//        sort_nams_single_check(global_nams[id]);
        gpu_shuffle_top_nams(global_nams[id]);
    }
}

__global__ void gpu_rescue_sort_hits(
        int num_tasks,
        my_vector<my_pair<int, Hit>>* hits_per_ref0s,
        my_vector<my_pair<int, Hit>>* hits_per_ref1s,
        int* global_todo_ids
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int l_range = global_id * GPU_thread_task_size;
    int r_range = l_range + GPU_thread_task_size;
    if (r_range > num_tasks) r_range = num_tasks;
    for (int id = l_range; id < r_range; id++) {
        int real_id = global_todo_ids[id];
//        sort_hits_single(hits_per_ref0s[real_id]);
//        sort_hits_single(hits_per_ref1s[real_id]);
        sort_hits_by_refid(hits_per_ref0s[real_id]);
        sort_hits_by_refid(hits_per_ref1s[real_id]);
        //        check_hits(hits_per_ref0s[real_id]);
        //        check_hits(hits_per_ref1s[real_id]);
    }
}

__global__ void gpu_sort_hits(
        int num_tasks,
        my_vector<my_pair<int, Hit>>* hits_per_ref0s,
        my_vector<my_pair<int, Hit>>* hits_per_ref1s,
        int* global_todo_ids
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int l_range = global_id * GPU_thread_task_size;
    int r_range = l_range + GPU_thread_task_size;
    if (r_range > num_tasks) r_range = num_tasks;
    for (int id = l_range; id < r_range; id++) {
        int real_id = global_todo_ids[id];
//        sort_hits_single(hits_per_ref0s[real_id]);
//        sort_hits_single(hits_per_ref1s[real_id]);
        sort_hits_by_refid(hits_per_ref0s[real_id]);
        sort_hits_by_refid(hits_per_ref1s[real_id]);
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
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int l_range = global_id * GPU_thread_task_size;
    int r_range = l_range + GPU_thread_task_size;
    if (r_range > num_tasks) r_range = num_tasks;
    for (int id = l_range; id < r_range; id++) {
        int real_id = global_todo_ids[id];
        my_vector<Nam> *nams = (my_vector<Nam>*)my_malloc(sizeof(my_vector<Nam>));
        nams->init(32);
//        salign_merge_hits(hits_per_ref0s[real_id], index_para->syncmer.k, 0, *nams);
//        salign_merge_hits(hits_per_ref1s[real_id], index_para->syncmer.k, 1, *nams);
        merge_hits(hits_per_ref0s[real_id], index_para->syncmer.k, 0, *nams);
        merge_hits(hits_per_ref1s[real_id], index_para->syncmer.k, 1, *nams);
        hits_per_ref0s[real_id].release();
        hits_per_ref1s[real_id].release();
        global_nams[real_id] = *nams;
        my_free(nams);
    }
}


__global__ void gpu_pre_cal_type(
        int num_tasks,
        float dropoff_threshold,
        my_vector<Nam> *global_nams,
        GPUInsertSizeDistribution& isize_est,
        int *global_todo_ids) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int l_range = global_id * GPU_thread_task_size;
    int r_range = l_range + GPU_thread_task_size;
    if (r_range > num_tasks) r_range = num_tasks;
    for (int id = l_range; id < r_range; id++) {
        //GPUInsertSizeDistribution isize_est;
        my_vector<Nam> &nams1 = global_nams[id];
        my_vector<Nam> &nams2 = global_nams[id + num_tasks];
        if (nams1.empty() && nams2.empty()) {
            global_todo_ids[id] = 0;
        } else if (!nams1.empty() && nams2.empty()) {
            global_todo_ids[id] = 1;
        } else if (nams1.empty() && !nams2.empty()) {
            global_todo_ids[id] = 2;
        } else if (gpu_top_dropoff(nams1) < dropoff_threshold && gpu_top_dropoff(nams2) < dropoff_threshold && gpu_is_proper_nam_pair(nams1[0], nams2[0], isize_est.mu, isize_est.sigma)) {
            global_todo_ids[id] = 3;
        } else {
            global_todo_ids[id] = 4;
        }
    }
}

__global__ void gpu_pre_align_SE(
        int num_tasks,
        int s_len,
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
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int l_range = global_id * GPU_thread_task_size;
    int r_range = l_range + GPU_thread_task_size;
    if (r_range > num_tasks) r_range = num_tasks;
    for (int id = l_range; id < r_range; id++) {
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

__global__ void gpu_align_SE(
        int num_tasks,
        int s_len,
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
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int l_range = global_id * GPU_thread_task_size;
    int r_range = l_range + GPU_thread_task_size;
    if (r_range > num_tasks) r_range = num_tasks;
    for (int id = l_range; id < r_range; id++) {
        int real_id = id;
        int type = global_align_res[real_id].type;
        assert(type <= 4);
        if (type == 0) {
            global_nams[real_id].release();
        } else if (type == 4) {
            size_t seq_len;
            seq_len = lens[real_id];
            char *seq, *rc;
            seq = all_seqs + pre_sum[real_id];
            rc = all_seqs + pre_sum[real_id + s_len];

            GPUAlignTmpRes* align_tmp_res = &global_align_res[real_id];
            align_SE_part(*align_tmp_res, *aligner_parameters, global_nams[real_id],
                           seq, rc, seq_len, index_para->syncmer.k, *global_references,
                           mapping_parameters->dropoff_threshold, mapping_parameters->max_tries, mapping_parameters->max_secondary, tid, real_id,
                           d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset);
            global_nams[real_id].release();
        } else {
            assert(false);
        }
    }

}

__global__ void gpu_align_PE01234(
        int num_tasks,
        int s_len,
        IndexParameters *index_para,
        uint64_t *global_align_info,
        AlignmentParameters* aligner_parameters,
        int *pre_sum,
        int *lens,
        char *all_seqs,
        GPUReferences *global_references,
        MappingParameters *mapping_parameters,
        my_vector<Nam> *global_nams,
        GPUInsertSizeDistribution& isize_est,
        int *global_todo_ids,
        GPUAlignTmpRes *global_align_res,
        int* d_todo_cnt,
        char* d_query_ptr, char* d_ref_ptr,
        int* d_query_offset, int* d_ref_offset
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int l_range = global_id * GPU_thread_task_size;
    int r_range = l_range + GPU_thread_task_size;
    if (r_range > num_tasks) r_range = num_tasks;
    for (int id = l_range; id < r_range; id++) {
        int real_id = global_todo_ids[id];
        int type = global_align_res[real_id].type;
        assert(type <= 4);
        if (type == 0) {
            global_nams[real_id].release();
            global_nams[real_id + s_len].release();
        } else if (type == 1) {
            size_t seq_len1, seq_len2;
            seq_len1 = lens[real_id];
            seq_len2 = lens[real_id + s_len * 2];
            char *seq1, *seq2, *rc1, *rc2;
            seq1 = all_seqs + pre_sum[real_id];
            rc1 = all_seqs + pre_sum[real_id + s_len];
            seq2 = all_seqs + pre_sum[real_id + s_len * 2];
            rc2 = all_seqs + pre_sum[real_id + s_len * 3];

            GPUAlignTmpRes* align_tmp_res = &global_align_res[real_id];
            //GPUInsertSizeDistribution isize_est;
            align_PE_part12(*align_tmp_res, *aligner_parameters, global_nams[real_id], global_nams[real_id + s_len],
                            seq1, rc1, seq_len1, seq2, rc2, seq_len2, index_para->syncmer.k, *global_references,
                            mapping_parameters->dropoff_threshold, isize_est, mapping_parameters->max_tries, mapping_parameters->max_secondary, 1, real_id,
                            d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset);
            global_nams[real_id].release();
            global_nams[real_id + s_len].release();
        } else if (type == 2) {
            size_t seq_len1, seq_len2;
            seq_len1 = lens[real_id];
            seq_len2 = lens[real_id + s_len * 2];
            char *seq1, *seq2, *rc1, *rc2;
            seq1 = all_seqs + pre_sum[real_id];
            rc1 = all_seqs + pre_sum[real_id + s_len];
            seq2 = all_seqs + pre_sum[real_id + s_len * 2];
            rc2 = all_seqs + pre_sum[real_id + s_len * 3];

            GPUAlignTmpRes* align_tmp_res = &global_align_res[real_id];
            //GPUInsertSizeDistribution isize_est;
            align_PE_part12(*align_tmp_res, *aligner_parameters, global_nams[real_id], global_nams[real_id + s_len],
                            seq1, rc1, seq_len1, seq2, rc2, seq_len2, index_para->syncmer.k, *global_references,
                            mapping_parameters->dropoff_threshold, isize_est, mapping_parameters->max_tries, mapping_parameters->max_secondary, 2, real_id,
                            d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset);
            global_nams[real_id].release();
            global_nams[real_id + s_len].release();
        } else if (type == 3) {
            size_t seq_len1, seq_len2;
            seq_len1 = lens[real_id];
            seq_len2 = lens[real_id + s_len * 2];
            char *seq1, *seq2, *rc1, *rc2;
            seq1 = all_seqs + pre_sum[real_id];
            rc1 = all_seqs + pre_sum[real_id + s_len];
            seq2 = all_seqs + pre_sum[real_id + s_len * 2];
            rc2 = all_seqs + pre_sum[real_id + s_len * 3];

            GPUAlignTmpRes* align_tmp_res = &global_align_res[real_id];
            //GPUInsertSizeDistribution isize_est;
            align_PE_part3(*align_tmp_res, *aligner_parameters, global_nams[real_id], global_nams[real_id + s_len],
                           seq1, rc1, seq_len1, seq2, rc2, seq_len2, index_para->syncmer.k, *global_references,
                           mapping_parameters->dropoff_threshold, isize_est, mapping_parameters->max_tries, mapping_parameters->max_secondary, real_id,
                           d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset);
            global_nams[real_id].release();
            global_nams[real_id + s_len].release();
        } else if (type == 4) {
            size_t seq_len1, seq_len2;
            seq_len1 = lens[real_id];
            seq_len2 = lens[real_id + s_len * 2];
            char *seq1, *seq2, *rc1, *rc2;
            seq1 = all_seqs + pre_sum[real_id];
            rc1 = all_seqs + pre_sum[real_id + s_len];
            seq2 = all_seqs + pre_sum[real_id + s_len * 2];
            rc2 = all_seqs + pre_sum[real_id + s_len * 3];

            GPUAlignTmpRes* align_tmp_res = &global_align_res[real_id];
            //GPUInsertSizeDistribution isize_est;
            align_PE_part4(*align_tmp_res, *aligner_parameters, global_nams[real_id], global_nams[real_id + s_len],
                           seq1, rc1, seq_len1, seq2, rc2, seq_len2, index_para->syncmer.k, *global_references,
                           mapping_parameters->dropoff_threshold, isize_est, mapping_parameters->max_tries, mapping_parameters->max_secondary, tid, real_id,
                           d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset);
            global_nams[real_id].release();
            global_nams[real_id + s_len].release();
        } else {
            assert(false);
        }
    }

}

__global__ void gpu_free_align_res(int num_tasks, GPUAlignTmpRes *global_align_res) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int l_range = global_id * GPU_thread_task_size;
    int r_range = l_range + GPU_thread_task_size;
    if (r_range > num_tasks) r_range = num_tasks;
    for (int id = l_range; id < r_range; id++) {
        global_align_res[id].is_extend_seed.release();
        global_align_res[id].consistent_nam.release();
        global_align_res[id].is_read1.release();
        global_align_res[id].type4_nams.release();
        global_align_res[id].todo_nams.release();
        global_align_res[id].done_align.release();
        global_align_res[id].align_res.release();
    }
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

// GPU timer
thread_local double gpu_copy1 = 0;
thread_local double gpu_copy2 = 0;
thread_local double gpu_init1 = 0;
thread_local double gpu_init2 = 0;
thread_local double gpu_init3 = 0;
thread_local double gpu_init4 = 0;
thread_local double gpu_cost1 = 0;
thread_local double gpu_cost2 = 0;
thread_local double gpu_cost2_1 = 0;
thread_local double gpu_cost2_2 = 0;
thread_local double gpu_cost3 = 0;
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

struct ThreadContext {
    int device_id;
    cudaStream_t stream;

    ThreadContext(int tid, int gpuid) {
        device_id = gpuid;
        cudaSetDevice(device_id);
        cudaStreamCreate(&stream);
    }

    ~ThreadContext() {
        cudaSetDevice(device_id);
        cudaStreamDestroy(stream);
    }
};

void GPU_part2_rescue_mate_get_str(
        std::vector<std::string>& todo_querys,
        std::vector<std::string>& todo_refs,
        GPUAlignTmpRes& align_tmp_res,
        int j,
        Read &read1,
        Read &read2,
        const References& references,
        const Aligner& aligner,
        float mu,
        float sigma
) {
    Nam nam = align_tmp_res.todo_nams[j];
    Read read = align_tmp_res.is_read1[j] ? read1 : read2;
    int a, b;
    std::string r_tmp;
    auto read_len = read.size();

    if (nam.is_rc) {
        r_tmp = read.seq;
        a = nam.ref_start - nam.query_start - (mu + 5 * sigma);
        b = nam.ref_start - nam.query_start + read_len / 2;  // at most half read overlap
    } else {
        r_tmp = read.rc;                                              // mate is rc since fr orientation
        a = nam.ref_end + (read_len - nam.query_end) - read_len / 2;  // at most half read overlap
        b = nam.ref_end + (read_len - nam.query_end) + (mu + 5 * sigma);
    }

    auto ref_len = static_cast<int>(references.lengths[nam.ref_id]);
    auto ref_start = std::max(0, std::min(a, ref_len));
    auto ref_end = std::min(ref_len, std::max(0, b));

    std::string ref_segm = references.sequences[nam.ref_id].substr(ref_start, ref_end - ref_start);
    todo_querys.push_back(r_tmp);
    todo_refs.push_back(ref_segm);
}

void GPU_part2_extend_seed_get_str(
        std::vector<std::string>& todo_querys,
        std::vector<std::string>& todo_refs,
        GPUAlignTmpRes& align_tmp_res,
        int j,
        Read &read1,
        Read &read2,
        const References& references,
        const Aligner& aligner
) {
    Nam nam = align_tmp_res.todo_nams[j];
    Read read = align_tmp_res.is_read1[j] ? read1 : read2;
    AlignmentInfo info;
    int result_ref_start;
    const std::string query = nam.is_rc ? read.rc : read.seq;
    const std::string& ref = references.sequences[nam.ref_id];

    const auto projected_ref_start = std::max(0, nam.ref_start - nam.query_start);
    const auto projected_ref_end = std::min(nam.ref_end + query.size() - nam.query_end, ref.size());

    const int diff = std::abs(nam.ref_span() - nam.query_span());
    const int ext_left = std::min(50, projected_ref_start);
    const int ref_start = projected_ref_start - ext_left;
    const int ext_right = std::min(std::size_t(50), ref.size() - nam.ref_end);
    const auto ref_segm_size = read.size() + diff + ext_left + ext_right;
    const auto ref_segm = ref.substr(ref_start, ref_segm_size);
    todo_querys.push_back(query);
    todo_refs.push_back(ref_segm);
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
//    alignment.cigar = std::move(info.cigar);
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
        //printf("host cigar too big %d %d\n", j, info.cigar.m_ops.size());
        align_tmp_res.cigar_info[j].has_realloc = 1;
        uint32_t* tmp_cigar = (uint32_t*)malloc((info.cigar.m_ops.size() + 1) * sizeof(uint32_t));
        align_tmp_res.cigar_info[j].cigar = tmp_cigar;
    }
    align_tmp_res.cigar_info[j].cigar[0] = info.cigar.m_ops.size();
    for (int k = 0; k < info.cigar.m_ops.size(); k++) {
        align_tmp_res.cigar_info[j].cigar[k + 1] = info.cigar.m_ops[k];
    }
}

void GPU_part2_rescue_mate_store_res(
        GPUAlignTmpRes& align_tmp_res,
        int j,
        const neoRcRef &read1,
        const neoRcRef &read2,
        const References& references,
        const AlignmentInfo& info,
        float mu,
        float sigma
) {
    Nam nam = align_tmp_res.todo_nams[j];
    const neoRcRef &read = align_tmp_res.is_read1[j] ? read1 : read2;
    int a, b;
    auto read_len = read.read.lseq;

    if (nam.is_rc) {
        a = nam.ref_start - nam.query_start - (mu + 5 * sigma);
        b = nam.ref_start - nam.query_start + read_len / 2;  // at most half read overlap
    } else {
        a = nam.ref_end + (read_len - nam.query_end) - read_len / 2;  // at most half read overlap
        b = nam.ref_end + (read_len - nam.query_end) + (mu + 5 * sigma);
    }

    auto ref_len = static_cast<int>(references.lengths[nam.ref_id]);
    auto ref_start = std::max(0, std::min(a, ref_len));
    auto ref_end = std::min(ref_len, std::max(0, b));

    GPUAlignment& alignment = align_tmp_res.align_res[j];
//    alignment.cigar = info.cigar;
    alignment.edit_distance = info.edit_distance;
    alignment.score = info.sw_score;
    alignment.ref_start = ref_start + info.ref_start;
    alignment.is_rc = !nam.is_rc;
    alignment.ref_id = nam.ref_id;
    alignment.is_unaligned = info.cigar.empty();
    alignment.length = info.ref_span();

    align_tmp_res.cigar_info[j].has_realloc = 0;
    align_tmp_res.cigar_info[j].cigar = align_tmp_res.cigar_info[j].gpu_cigar;
    if (info.cigar.m_ops.size() + 1 > MAX_CIGAR_ITEM) {
        //printf("host cigar too big %d %d\n", j, info.cigar.m_ops.size());
        align_tmp_res.cigar_info[j].has_realloc = 1;
        uint32_t* tmp_cigar = (uint32_t*)malloc((info.cigar.m_ops.size() + 1) * sizeof(uint32_t));
        align_tmp_res.cigar_info[j].cigar = tmp_cigar;
    }
    align_tmp_res.cigar_info[j].cigar[0] = info.cigar.m_ops.size();
    for (int k = 0; k < info.cigar.m_ops.size(); k++) {
        align_tmp_res.cigar_info[j].cigar[k + 1] = info.cigar.m_ops[k];
    }

}

struct GPUScoredAlignmentPair {
    double score;
    std::pair<GPUAlignment, CigarData> alignment1;
    std::pair<GPUAlignment, CigarData> alignment2;
};

static inline float GPU_normal_pdf(float x, float mu, float sigma) {
    static const float inv_sqrt_2pi = 0.3989422804014327;
    const float a = (x - mu) / sigma;

    return inv_sqrt_2pi / sigma * std::exp(-0.5f * a * a);
}

static inline std::vector<GPUScoredAlignmentPair> GPU_get_best_scoring_pairs(
        const std::vector<std::pair<GPUAlignment, CigarData>>& alignments1,
        const std::vector<std::pair<GPUAlignment, CigarData>>& alignments2,
        float mu,
        float sigma
) {
    std::vector<GPUScoredAlignmentPair> pairs;
    for (auto& aa1 : alignments1) {
        for (auto& aa2 : alignments2) {
            GPUAlignment a1 = aa1.first;
            GPUAlignment a2 = aa2.first;
            float dist = std::abs(a1.ref_start - a2.ref_start);
            double score = a1.score + a2.score;
            if ((a1.is_rc ^ a2.is_rc) && (dist < mu + 4 * sigma)) {
                score += log(GPU_normal_pdf(dist, mu, sigma));
            } else {  // individual score
                // 10 corresponds to a value of log(GPU_normal_pdf(dist, mu, sigma)) of more than 4 stddevs away
                score -= 10;
            }
            pairs.push_back(GPUScoredAlignmentPair{score, aa1, aa2});
        }
    }

    return pairs;
}

void GPU_deduplicate_scored_pairs(std::vector<GPUScoredAlignmentPair>& pairs) {
    int prev_ref_start1 = pairs[0].alignment1.first.ref_start;
    int prev_ref_start2 = pairs[0].alignment2.first.ref_start;
    int prev_ref_id1 = pairs[0].alignment1.first.ref_id;
    int prev_ref_id2 = pairs[0].alignment2.first.ref_id;
    size_t j = 1;
    for (size_t i = 1; i < pairs.size(); i++) {
        int ref_start1 = pairs[i].alignment1.first.ref_start;
        int ref_start2 = pairs[i].alignment2.first.ref_start;
        int ref_id1 = pairs[i].alignment1.first.ref_id;
        int ref_id2 = pairs[i].alignment2.first.ref_id;
        if (ref_start1 != prev_ref_start1 || ref_start2 != prev_ref_start2 || ref_id1 != prev_ref_id1 ||
            ref_id2 != prev_ref_id2) {
            prev_ref_start1 = ref_start1;
            prev_ref_start2 = ref_start2;
            prev_ref_id1 = ref_id1;
            prev_ref_id2 = ref_id2;
            pairs[j] = pairs[i];
            j++;
        }
    }
    pairs.resize(j);
}

void GPU_pick_random_top_pair(std::vector<GPUScoredAlignmentPair>& high_scores, std::minstd_rand& random_engine) {
    size_t i = 1;
    for (; i < high_scores.size(); ++i) {
        if (high_scores[i].score != high_scores[0].score) {
            break;
        }
    }
    if (i > 1) {
        size_t random_index = std::uniform_int_distribution<>(0, i - 1)(random_engine);
        if (random_index != 0) {
            std::swap(high_scores[0], high_scores[random_index]);
        }
    }
}

static std::pair<int, int> GPU_joint_mapq_from_high_scores(const std::vector<GPUScoredAlignmentPair>& pairs) {
    if (pairs.size() <= 1) {
        return std::make_pair(60, 60);
    }
    auto score1 = pairs[0].score;
    auto score2 = pairs[1].score;
    if (score1 == score2) {
        return std::make_pair(0, 0);
    }
    int mapq;
    const int diff = score1 - score2;  // (1.0 - (S1 - S2) / S1);
    //  float log10_p = diff > 6 ? -6.0 : -diff; // Corresponds to: p_error= 0.1^diff // change in sw score times rough illumina error rate. This is highly heauristic, but so seem most computations of mapq scores
    if (score1 > 0 && score2 > 0) {
        mapq = std::min(60, diff);
        //            mapq1 = -10 * log10_p < 60 ? -10 * log10_p : 60;
    } else if (score1 > 0 && score2 <= 0) {
        mapq = 60;
    } else {  // both negative SW one is better
        mapq = 1;
    }
    return std::make_pair(mapq, mapq);
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

void GPU_rescue_read_last(
        int flag,
        GPUAlignTmpRes& align_tmp_res,
        const Read& read2,  // read to be rescued
        const Read& read1,  // read that has NAMs
        const Aligner& aligner,
        const References& references,
        std::array<Details, 2>& details,
        float mu,
        float sigma,
        size_t max_secondary,
        double secondary_dropoff,
        Sam& sam,
        const klibpp::KSeq& record1,
        const klibpp::KSeq& record2,
        bool swap_r1r2,  // TODO get rid of this
        std::minstd_rand& random_engine
) {
    std::vector<std::pair<GPUAlignment, CigarData>> alignments1;
    std::vector<std::pair<GPUAlignment, CigarData>> alignments2;
    int res_num = align_tmp_res.todo_nams.size();
    assert(res_num % 2 == 0);
    for (int i = 0; i < res_num; i += 2) {
        alignments1.push_back(std::make_pair(align_tmp_res.align_res[i], align_tmp_res.cigar_info[i]));
        alignments2.push_back(std::make_pair(align_tmp_res.align_res[i + 1], align_tmp_res.cigar_info[i + 1]));
        details[1].mate_rescue += !align_tmp_res.align_res[i + 1].is_unaligned;
        //        fprintf(stderr, "3 a1 score %d\n", align_tmp_res.align_res[i].score);
        //        fprintf(stderr, "3 a2 score %d\n", align_tmp_res.align_res[i + 1].score);
    }
    std::sort(alignments1.begin(), alignments1.end(),
              [](const std::pair<GPUAlignment, CigarData>& a,
                 const std::pair<GPUAlignment, CigarData>& b) {
                  return a.first.score > b.first.score;
              });
    std::sort(alignments2.begin(), alignments2.end(),
              [](const std::pair<GPUAlignment, CigarData>& a,
                 const std::pair<GPUAlignment, CigarData>& b) {
                  return a.first.score > b.first.score;
              });

    // Calculate best combined score here
    auto high_scores = GPU_get_best_scoring_pairs(alignments1, alignments2, mu, sigma);

    std::sort(high_scores.begin(), high_scores.end(),
              [](const GPUScoredAlignmentPair& a,
                 const GPUScoredAlignmentPair& b) {
                  return a.score > b.score;
              });
    GPU_deduplicate_scored_pairs(high_scores);
    GPU_pick_random_top_pair(high_scores, random_engine);

    auto [mapq1, mapq2] = GPU_joint_mapq_from_high_scores(high_scores);

    // append both alignments to string here
    if (max_secondary == 0) {
        auto best_aln_pair = high_scores[0];
        std::pair<GPUAlignment, CigarData> alignment1 = best_aln_pair.alignment1;
        std::pair<GPUAlignment, CigarData> alignment2 = best_aln_pair.alignment2;
        if (swap_r1r2) {
            sam.add_pair(
                    alignment2, alignment1, record2, record1, read2.rc, read1.rc, mapq2, mapq1,
                    GPU_is_proper_pair(alignment2, alignment1, mu, sigma), true, details
            );
        } else {
            sam.add_pair(
                    alignment1, alignment2, record1, record2, read1.rc, read2.rc, mapq1, mapq2,
                    GPU_is_proper_pair(alignment1, alignment2, mu, sigma), true, details
            );
        }
    } else {
        auto max_out = std::min(high_scores.size(), (size_t)max_secondary);
        bool is_primary = true;
        auto best_aln_pair = high_scores[0];
        auto s_max = best_aln_pair.score;
        for (size_t i = 0; i < max_out; ++i) {
            if (i > 0) {
                is_primary = false;
                mapq1 = 0;
                mapq2 = 0;
            }
            auto aln_pair = high_scores[i];
            auto s_score = aln_pair.score;
            auto alignment1 = aln_pair.alignment1;
            auto alignment2 = aln_pair.alignment2;
            if (s_max - s_score < secondary_dropoff) {
                if (swap_r1r2) {
                    bool is_proper = GPU_is_proper_pair(alignment2, alignment1, mu, sigma);
                    std::array<Details, 2> swapped_details{details[1], details[0]};
                    sam.add_pair(
                            alignment2, alignment1, record2, record1, read2.rc, read1.rc, mapq2, mapq1,
                            is_proper, is_primary, swapped_details
                    );
                } else {
                    bool is_proper = GPU_is_proper_pair(alignment1, alignment2, mu, sigma);
                    sam.add_pair(
                            alignment1, alignment2, record1, record2, read1.rc, read2.rc, mapq1, mapq2,
                            is_proper, is_primary, details
                    );
                }
            } else {
                break;
            }
        }
    }
}

Alignment gpu_ConverAlignment(GPUAlignment gpu_alignment, CigarData cigar_data) {
    Alignment alignment;
    alignment.ref_start = gpu_alignment.ref_start;
    alignment.ref_id = gpu_alignment.ref_id;
    alignment.is_rc = gpu_alignment.is_rc;
    alignment.is_unaligned = gpu_alignment.is_unaligned;
    alignment.gapped = gpu_alignment.gapped;
    alignment.length = gpu_alignment.length;
    alignment.edit_distance = gpu_alignment.edit_distance;
    alignment.global_ed = gpu_alignment.global_ed;
    alignment.score = gpu_alignment.score;

    alignment.cigar = Cigar();
    for (int i = 0; i < cigar_data.cigar[0]; i++) {
        alignment.cigar.m_ops.push_back(cigar_data.cigar[i + 1]);
    }
    return alignment;
}

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

                // Pick one randomly using reservoir sampling
                std::uniform_int_distribution<> distrib(1, alignments_with_best_score);
                if (distrib(random_engine) == 1) {
                    update_best = true;
                }
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

void GPU_align_SE_read_last2(
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
    auto record = gpu_ConvertNeo2KSeq(data.read);
    if (align_tmp_res.type == 0) {
        sam.add_unmapped(record);
        t1 += GetTime() - t0;
        return;
    }

    Read read(record.seq);
    std::vector<Alignment> alignments;
    int tries = 0;
    Nam n_max = align_tmp_res.todo_nams[0];

    int best_edit_distance = std::numeric_limits<int>::max();
    int best_score = 0;
    int second_best_score = 0;
    int alignments_with_best_score = 0;
    size_t best_index = 0;

    Alignment best_alignment;
    best_alignment.is_unaligned = true;
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
        auto alignment = gpu_ConverAlignment(align_tmp_res.align_res[i], align_tmp_res.cigar_info[i]);
        details.tried_alignment++;
        details.gapped += alignment.gapped;

        if (max_secondary > 0) {
            alignments.emplace_back(alignment);
        }

        if (alignment.score >= best_score) {
            second_best_score = best_score;
            bool update_best = false;
            if (alignment.score > best_score) {
                alignments_with_best_score = 1;
                update_best = true;
            } else {
                assert(alignment.score == best_score);
                // Two or more alignments have the same best score - count them
                alignments_with_best_score++;

                // Pick one randomly using reservoir sampling
                std::uniform_int_distribution<> distrib(1, alignments_with_best_score);
                if (distrib(random_engine) == 1) {
                    update_best = true;
                }
            }
            if (update_best) {
                best_score = alignment.score;
                best_alignment = std::move(alignment);
                best_index = tries;
                if (max_secondary == 0) {
                    best_edit_distance = best_alignment.global_ed;
                }
            }
        } else if (alignment.score > second_best_score) {
            second_best_score = alignment.score;
        }
        tries++;
    }
    t2 += GetTime() - t0;

    t0 = GetTime();
    uint8_t mapq = (60.0 * (best_score - second_best_score) + best_score - 1) / best_score;
    bool is_primary = true;
    sam.add(best_alignment, record, read.rc, mapq, is_primary, details);

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
    std::sort(alignments.begin(), alignments.end(), [](const Alignment& a, const Alignment& b) -> bool {
        return a.score > b.score;
    });
    t3 += GetTime() - t0;

    // Output secondary alignments
    t0 = GetTime();
    size_t n = 0;
    for (const auto& alignment : alignments) {
        if (n >= max_secondary ||
            alignment.score - best_score > 2 * aligner.parameters.mismatch + aligner.parameters.gap_open) {
            break;
        }
        bool is_primary = false;
        sam.add(alignment, record, read.rc, mapq, is_primary, details);
        n++;
    }
    t4 += GetTime() - t0;

}

void GPU_align_PE_read_last(
        GPUAlignTmpRes& align_tmp_res,
        const neoRcRef &data1,
        const neoRcRef &data2,
        Sam& sam,
        std::string& outstring,
        GPUInsertSizeDistribution& isize_est,
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
    std::array<Details, 2> details;
    const auto mu = isize_est.mu;
    const auto sigma = isize_est.sigma;
    double secondary_dropoff = 2 * aligner.parameters.mismatch + aligner.parameters.gap_open;

    //    fprintf(stderr, "type %d\n", align_tmp_res.type);
    if (align_tmp_res.type == 0) {
        // None of the reads have any NAMs
        sam.add_unmapped_pair(data1.read, data2.read);
    } else if (align_tmp_res.type == 1) {
        double t_start = GetTime();
        auto record1 = gpu_ConvertNeo2KSeq(data1.read);
        auto record2 = gpu_ConvertNeo2KSeq(data2.read);
        Read read1(record1.seq);
        Read read2(record2.seq);
        GPU_rescue_read_last(
                1, align_tmp_res, read2, read1, aligner, references, details, mu,
                sigma, map_param.max_secondary, secondary_dropoff, sam, record1, record2, false, random_engine
        );
        t1 += GetTime() - t_start;
    } else if (align_tmp_res.type == 2) {
        double t_start = GetTime();
        auto record1 = gpu_ConvertNeo2KSeq(data1.read);
        auto record2 = gpu_ConvertNeo2KSeq(data2.read);
        Read read1(record1.seq);
        Read read2(record2.seq);
        GPU_rescue_read_last(
                2, align_tmp_res, read1, read2, aligner, references, details, mu,
                sigma, map_param.max_secondary, secondary_dropoff, sam, record2, record1, true, random_engine
        );
        t2 += GetTime() - t_start;
    } else if (align_tmp_res.type == 3) {
        double t_start = GetTime();
        assert(align_tmp_res.todo_nams.size() == 2);
        int mapq1 = align_tmp_res.mapq1;
        int mapq2 = align_tmp_res.mapq2;
        std::pair<GPUAlignment, CigarData> alignment1 = std::make_pair(align_tmp_res.align_res[0], align_tmp_res.cigar_info[0]);
        std::pair<GPUAlignment, CigarData> alignment2 = std::make_pair(align_tmp_res.align_res[1], align_tmp_res.cigar_info[1]);
        bool is_proper = GPU_is_proper_pair(alignment1, alignment2, mu, sigma);
        bool is_primary = true;
        sam.add_pair(
                alignment1, alignment2, data1.read, data2.read, data1.rc, data2.rc, mapq1, mapq2, is_proper, is_primary,
                details
        );
        t3 += GetTime() - t_start;
    } else if (align_tmp_res.type == 4) {
        double t_start = GetTime();
        int pos = 0;
        robin_hood::unordered_map<int, std::pair<GPUAlignment, CigarData>> is_aligned1;
        robin_hood::unordered_map<int, std::pair<GPUAlignment, CigarData>> is_aligned2;

        std::pair<GPUAlignment, CigarData> a1_indv_max, a2_indv_max;
        {

            auto n1_max = align_tmp_res.todo_nams[pos];
            //            fprintf(stderr, "get n1 %d from %d\n", n1_max.nam_id, pos);
            a1_indv_max = std::make_pair(align_tmp_res.align_res[pos], align_tmp_res.cigar_info[pos]);
            is_aligned1[n1_max.nam_id] = a1_indv_max;

            pos++;

            auto n2_max = align_tmp_res.todo_nams[pos];
            //            fprintf(stderr, "get n2 %d from %d\n", n2_max.nam_id, pos);
            a2_indv_max = std::make_pair(align_tmp_res.align_res[pos], align_tmp_res.cigar_info[pos]);
            is_aligned2[n2_max.nam_id] = a2_indv_max;

            pos++;
        }

        std::vector<GPUScoredAlignmentPair> high_scores;
        assert(align_tmp_res.type4_loop_size * 2 == align_tmp_res.type4_nams.size());

        for(int i = 0; i < align_tmp_res.type4_loop_size; i++) {
            Nam n1 = align_tmp_res.type4_nams[i * 2];
            Nam n2 = align_tmp_res.type4_nams[i * 2 + 1];

            std::pair<GPUAlignment, CigarData> a1;
            // ref_start == -1 is a marker for a dummy NAM
            if (n1.ref_start >= 0) {
                if (is_aligned1.find(n1.nam_id) != is_aligned1.end()) {
                    a1 = is_aligned1[n1.nam_id];
                } else {
                    a1 = std::make_pair(align_tmp_res.align_res[pos], align_tmp_res.cigar_info[pos]);
                    assert(n1.nam_id == align_tmp_res.todo_nams[pos].nam_id);
                    pos++;
                    is_aligned1[n1.nam_id] = a1;
                }
            } else {
                a1 = std::make_pair(align_tmp_res.align_res[pos], align_tmp_res.cigar_info[pos]);
                assert(n2.nam_id == align_tmp_res.todo_nams[pos].nam_id);
                pos++;
                details[0].mate_rescue += !a1.first.is_unaligned;
            }
            if (a1.first.score > a1_indv_max.first.score) {
                a1_indv_max = a1;
            }

            std::pair<GPUAlignment, CigarData> a2;
            // ref_start == -1 is a marker for a dummy NAM
            if (n2.ref_start >= 0) {
                if (is_aligned2.find(n2.nam_id) != is_aligned2.end()) {
                    //                    fprintf(stderr, "find n2 %d\n", n2.nam_id);
                    a2 = is_aligned2[n2.nam_id];
                } else {
                    //                    fprintf(stderr, "get n2 %d from %d\n", n2.nam_id, pos);
                    a2 = std::make_pair(align_tmp_res.align_res[pos], align_tmp_res.cigar_info[pos]);
                    assert(n2.nam_id == align_tmp_res.todo_nams[pos].nam_id);
                    pos++;
                    is_aligned2[n2.nam_id] = a2;
                }
            } else {
                a2 = std::make_pair(align_tmp_res.align_res[pos], align_tmp_res.cigar_info[pos]);
                assert(n1.nam_id == align_tmp_res.todo_nams[pos].nam_id);
                pos++;
                details[1].mate_rescue += !a2.first.is_unaligned;
            }
            if (a2.first.score > a2_indv_max.first.score) {
                a2_indv_max = a2;
            }

            bool r1_r2 = a2.first.is_rc && (a1.first.ref_start <= a2.first.ref_start) &&
                         ((a2.first.ref_start - a1.first.ref_start) < mu + 10 * sigma);  // r1 ---> <---- r2
            bool r2_r1 = a1.first.is_rc && (a2.first.ref_start <= a1.first.ref_start) &&
                         ((a1.first.ref_start - a2.first.ref_start) < mu + 10 * sigma);  // r2 ---> <---- r1

            double combined_score;
            if (r1_r2 || r2_r1) {
                // Treat a1/a2 as a pair
                float x = std::abs(a1.first.ref_start - a2.first.ref_start);
                combined_score = (double) a1.first.score + (double) a2.first.score +
                                 std::max(-20.0f + 0.001f, log(GPU_normal_pdf(x, mu, sigma)));
                //* (1 - s2 / s1) * min_matches * log(s1);
            } else {
                // Treat a1/a2 as two single-end reads
                // 20 corresponds to a value of log(GPU_normal_pdf(x, mu, sigma)) of more than 5 stddevs away (for most reasonable values of stddev)
                combined_score = (double) a1.first.score + (double) a2.first.score - 20;
            }

            GPUScoredAlignmentPair aln_pair{combined_score, a1, a2};
            high_scores.push_back(aln_pair);

        }
        assert(pos == align_tmp_res.todo_nams.size());

        // Finally, add highest scores of both mates as individually mapped
        double combined_score =
                (double) a1_indv_max.first.score + (double) a2_indv_max.first.score -
                20;  // 20 corresponds to  a value of log( GPU_normal_pdf(x, mu, sigma ) ) of more than 5 stddevs away (for most reasonable values of stddev)
        GPUScoredAlignmentPair aln_tuple{combined_score, a1_indv_max, a2_indv_max};
        high_scores.push_back(aln_tuple);

        std::sort(high_scores.begin(), high_scores.end(),
                  [](const GPUScoredAlignmentPair& a,
                     const GPUScoredAlignmentPair& b) {
                      return a.score > b.score;
                  });
        GPU_deduplicate_scored_pairs(high_scores);
        GPU_pick_random_top_pair(high_scores, random_engine);

        auto [mapq1, mapq2] = GPU_joint_mapq_from_high_scores(high_scores);
        auto best_aln_pair = high_scores[0];
        auto alignment1 = best_aln_pair.alignment1;
        auto alignment2 = best_aln_pair.alignment2;
        if (map_param.max_secondary == 0) {
            bool is_proper = GPU_is_proper_pair(alignment1, alignment2, mu, sigma);
            sam.add_pair(
                    alignment1, alignment2, data1.read, data2.read, data1.rc, data2.rc, mapq1, mapq2, is_proper, true,
                    details
            );

        } else {
            auto max_out = std::min(high_scores.size(), (size_t)map_param.max_secondary);
            // remove eventual duplicates - comes from, e.g., adding individual best alignments above (if identical to joint best alignment)
            float s_max = best_aln_pair.score;
            bool is_primary = true;
            for (size_t i = 0; i < max_out; ++i) {
                auto aln_pair = high_scores[i];
                alignment1 = aln_pair.alignment1;
                alignment2 = aln_pair.alignment2;
                float s_score = aln_pair.score;
                if (i > 0) {
                    is_primary = false;
                    mapq1 = 255;
                    mapq2 = 255;
                }

                if (s_max - s_score < secondary_dropoff) {
                    bool is_proper = GPU_is_proper_pair(alignment1, alignment2, mu, sigma);
                    sam.add_pair(
                            alignment1, alignment2, data1.read, data2.read, data1.rc, data2.rc, mapq1, mapq2, is_proper,
                            is_primary, details
                    );
                } else {
                    break;
                }
            }
        }
        t4 += GetTime() - t_start;
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
    int l_id, r_id, s_len;

    // pack read on host
    t1 = GetTime();
    uint64_t tot_len = 0;
    h_pre_sum[0] = 0;
    s_len = datas.size();
    for (int i = 0; i < s_len * 2; i++) {
        int read_id = i % s_len;
        if (i < s_len) { // read1 seq
            h_len[i] = datas[read_id].read.lseq;
            h_pre_sum[i + 1] = h_pre_sum[i] + h_len[i];
            char* seq_ptr = (char*)datas[read_id].read.base + datas[read_id].read.pseq;
            memcpy(h_seq + h_pre_sum[i], seq_ptr, h_len[i]);
        } else { // read1 rc
            h_len[i] = datas[read_id].read.lseq;
            h_pre_sum[i + 1] = h_pre_sum[i] + h_len[i];
            char* seq_ptr = datas[read_id].rc;
            memcpy(h_seq + h_pre_sum[i], seq_ptr, h_len[i]);
        }
    }
    tot_len = h_pre_sum[s_len * 2];
    assert(tot_len <= batch_total_read_len * 2);
//    printf("se cal tot len %llu\n", tot_len);
    gpu_copy1 += GetTime() - t1;

    // transfer read to GPU
    t1 = GetTime();
    cudaMemcpy(d_seq, h_seq, tot_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_len, h_len, s_len * sizeof(int) * 2 + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pre_sum, h_pre_sum, s_len * sizeof(int) * 2 + 1, cudaMemcpyHostToDevice);
    gpu_copy2 += GetTime() - t1;

    for (l_id = 0; l_id < datas.size(); l_id += batch_read_num) {
        r_id = l_id + batch_read_num;
        if (r_id > datas.size()) r_id = datas.size();
        s_len = r_id - l_id;

        char* local_d_seq = d_seq;
        int* local_d_len = d_len + l_id;
        int* local_d_pre_sum = d_pre_sum + l_id;

        // get randstrobes
        t1 = GetTime();
        int threads_per_block;
        int reads_per_block;
        int blocks_per_grid;
        threads_per_block = 4;
        reads_per_block = threads_per_block * GPU_thread_task_size;
        blocks_per_grid = (s_len + reads_per_block - 1) / reads_per_block;
        gpu_get_randstrobes<<<blocks_per_grid, threads_per_block, 0, ctx.stream>>>(s_len, s_len, local_d_pre_sum, local_d_len, local_d_seq, d_index_para,
                                                                                   global_randstrobe_sizes, global_hashes_value, global_randstrobes);
        cudaDeviceSynchronize();
        gpu_cost1 += GetTime() - t1;
        //printf("get randstrobe done\n");

        // query database and get hits
        t1 = GetTime();
        threads_per_block = 4;
        reads_per_block = threads_per_block * GPU_thread_task_size;
        blocks_per_grid = (s_len + reads_per_block - 1) / reads_per_block;
        gpu_get_hits_pre<<<blocks_per_grid, threads_per_block, 0, ctx.stream>>>(index.bits, index.filter_cutoff, d_map_param->rescue_cutoff, d_randstrobes, index.randstrobes.size(), d_randstrobe_start_indices,
                                                                                s_len, d_index_para, global_hits_num, global_randstrobes,
                                                                                global_hits_per_ref0s, global_hits_per_ref1s);
        cudaDeviceSynchronize();
        //printf("get hits pre done\n");
        threads_per_block = 4;
        reads_per_block = threads_per_block * GPU_thread_task_size;
        blocks_per_grid = (s_len + reads_per_block - 1) / reads_per_block;
        gpu_get_hits_after<<<blocks_per_grid, threads_per_block, 0, ctx.stream>>>(index.bits, index.filter_cutoff, d_map_param->rescue_cutoff, d_randstrobes, index.randstrobes.size(), d_randstrobe_start_indices,
                                                                                  s_len, d_index_para, global_hits_num, global_randstrobes,
                                                                                  global_hits_per_ref0s, global_hits_per_ref1s);
        cudaDeviceSynchronize();
        //printf("get hits after done\n");
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

        // sort hits by ref_id
        t1 = GetTime();
        threads_per_block = 4;
        reads_per_block = threads_per_block * GPU_thread_task_size;
        blocks_per_grid = (todo_cnt + reads_per_block - 1) / reads_per_block;
        gpu_sort_hits<<<blocks_per_grid, threads_per_block, 0, ctx.stream>>>(todo_cnt, global_hits_per_ref0s, global_hits_per_ref1s, global_todo_ids);
        cudaDeviceSynchronize();
        gpu_cost3 += GetTime() - t1;
        //printf("sort hits done\n");

        // merge hits to NAMs
        t1 = GetTime();
        threads_per_block = 4;
        reads_per_block = threads_per_block * GPU_thread_task_size;
        blocks_per_grid = (todo_cnt + reads_per_block - 1) / reads_per_block;
        gpu_merge_hits_get_nams<<<blocks_per_grid, threads_per_block, 0, ctx.stream>>>(todo_cnt, d_index_para, global_nams_info,
                                                                                       global_hits_per_ref0s, global_hits_per_ref1s, global_nams, global_todo_ids);
        cudaDeviceSynchronize();
        gpu_cost4 += GetTime() - t1;

        // filter GG read for rescue
        t1 = GetTime();
        todo_cnt = 0;
        for (int i = 0; i < s_len; i++) {
            if (global_randstrobes[i].data != nullptr) {
                global_todo_ids[todo_cnt] = i;
                todo_cnt++;
            }
        }
        gpu_init2 += GetTime() - t1;

        // rescue mode get hits
        t1 = GetTime();
        threads_per_block = 4;
        reads_per_block = threads_per_block * GPU_thread_task_size;
        blocks_per_grid = (todo_cnt + reads_per_block - 1) / reads_per_block;
        gpu_rescue_get_hits<<<blocks_per_grid, threads_per_block, 0, ctx.stream>>>(index.bits, index.filter_cutoff, d_map_param->rescue_cutoff, d_randstrobes, index.randstrobes.size(), d_randstrobe_start_indices,
                                                                                   todo_cnt, d_index_para, global_hits_num, global_randstrobes,
                                                                                   global_hits_per_ref0s, global_hits_per_ref1s, global_todo_ids, rescue_threshold);
        cudaDeviceSynchronize();
        gpu_cost5 += GetTime() - t1;

        // rescue mode sort hits by ref_id
        t1 = GetTime();
        threads_per_block = 4;
        reads_per_block = threads_per_block * GPU_thread_task_size;
        blocks_per_grid = (todo_cnt + reads_per_block - 1) / reads_per_block;
        gpu_rescue_sort_hits<<<blocks_per_grid, threads_per_block, 0, ctx.stream>>>(todo_cnt, global_hits_per_ref0s, global_hits_per_ref1s, global_todo_ids);
        cudaDeviceSynchronize();
        gpu_cost6 += GetTime() - t1;
        //printf("rescue sort hits done\n");

        // rescue mode merge hits to NAMs
        t1 = GetTime();
        threads_per_block = 4;
        reads_per_block = threads_per_block * GPU_thread_task_size;
        blocks_per_grid = (todo_cnt + reads_per_block - 1) / reads_per_block;
        gpu_rescue_merge_hits_get_nams<<<blocks_per_grid, threads_per_block, 0, ctx.stream>>>(todo_cnt, d_index_para, global_nams_info,
                                                                                              global_hits_per_ref0s, global_hits_per_ref1s, global_nams, global_todo_ids);
        cudaDeviceSynchronize();
        gpu_cost7 += GetTime() - t1;

        // sort nams for all reads
        t1 = GetTime();
        threads_per_block = 4;
        reads_per_block = threads_per_block * GPU_thread_task_size;
        blocks_per_grid = (s_len + reads_per_block - 1) / reads_per_block;
        gpu_sort_nams<<<blocks_per_grid, threads_per_block, 0, ctx.stream>>>(s_len, global_nams, d_map_param, 1);
        cudaDeviceSynchronize();
        gpu_cost8 += GetTime() - t1;

        for (int i = 0; i < s_len; i++) global_todo_ids[i] = -1;

        // pre align reads
        threads_per_block = 4;
        reads_per_block = threads_per_block * GPU_thread_task_size;
        blocks_per_grid = (s_len + reads_per_block - 1) / reads_per_block;
        gpu_pre_align_SE<<<blocks_per_grid, threads_per_block, 0, ctx.stream>>>(s_len, s_len, d_index_para, global_align_info, d_aligner, local_d_pre_sum, local_d_len, local_d_seq,
                                                                            global_references, d_map_param, global_nams, global_todo_ids, global_align_res,
                                                                            d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset);
        cudaDeviceSynchronize();
        gpu_cost9 += GetTime() - t1;

        // allocate memory for align_tmp_res
        t1 = GetTime();
        char* base_ptr = global_align_res_data;
        uint64_t big_size = 0;
//        printf("MAX_TRIES_LIMIT %d\n", MAX_TRIES_LIMIT);
        for (int i = 0; i < s_len; i++) {
            int type = 4;
            if (global_nams[i].length == 0) type = 0;
            if (global_todo_ids[i] == -1 || global_todo_ids[i] > d_map_param->max_tries) {
                printf("global_todo_ids[%d] %d, type %d\n", i, global_todo_ids[i], type);
            }
//            int tries_num = MAX_TRIES_LIMIT / 2 + 1;
            int tries_num = global_todo_ids[i] * 2;
            if (tries_num > MAX_TRIES_LIMIT) {
                tries_num = MAX_TRIES_LIMIT;
            }
            if (type == 0) tries_num = 0;

            big_size += (MAX_TRIES_LIMIT / 2 + 1) * (sizeof(int) * 4 + sizeof(Nam) * 2 + sizeof(GPUAlignment) + sizeof(CigarData) + sizeof(TODOInfos));

            GPUAlignTmpRes *tmp = global_align_res + i;
            tmp->type = type;
            tmp->mapq1 = 0, tmp->mapq2 = 0, tmp->type4_loop_size = 0;

            tmp->is_extend_seed.data = (int*)base_ptr;
            tmp->is_extend_seed.length = 0;
            tmp->is_extend_seed.capacity = tries_num;
            base_ptr += tries_num * sizeof(int);

            tmp->consistent_nam.data = (int*)base_ptr;
            tmp->consistent_nam.length = 0;
            tmp->consistent_nam.capacity = tries_num;
            base_ptr += tries_num * sizeof(int);

            tmp->is_read1.data = (int*)base_ptr;
            tmp->is_read1.length = 0;
            tmp->is_read1.capacity = tries_num;
            base_ptr += tries_num * sizeof(int);

            tmp->type4_nams.data = (Nam*)base_ptr;
            tmp->type4_nams.length = 0;
            tmp->type4_nams.capacity = tries_num;
            base_ptr += tries_num * sizeof(Nam);

            tmp->todo_nams.data = (Nam*)base_ptr;
            tmp->todo_nams.length = 0;
            tmp->todo_nams.capacity = tries_num;
            base_ptr += tries_num * sizeof(Nam);

            tmp->done_align.data = (int*)base_ptr;
            tmp->done_align.length = 0;
            tmp->done_align.capacity = tries_num;
            base_ptr += tries_num * sizeof(int);

            tmp->align_res.data = (GPUAlignment*)base_ptr;
            tmp->align_res.length = 0;
            tmp->align_res.capacity = tries_num;
            base_ptr += tries_num * sizeof(GPUAlignment);

            tmp->cigar_info.data = (CigarData*)base_ptr;
            tmp->cigar_info.length = 0;
            tmp->cigar_info.capacity = tries_num;
            base_ptr += tries_num * sizeof(CigarData);

            tmp->todo_infos.data = (TODOInfos*)base_ptr;
            tmp->todo_infos.length = 0;
            tmp->todo_infos.capacity = tries_num;
            base_ptr += tries_num * sizeof(TODOInfos);
        }
//        printf("allocate memory for align_tmp_res done, size %llu - %llu, %lf\n", base_ptr - global_align_res_data, big_size, 1.0 * (base_ptr - global_align_res_data) / big_size);
        gpu_init3 += GetTime() - t1;
        //printf("types: %d %d %d %d %d\n", types[0].size(), types[1].size(), types[2].size(), types[3].size(), types[4].size());

        // align reads
        threads_per_block = 4;
        reads_per_block = threads_per_block * GPU_thread_task_size;
        blocks_per_grid = (s_len + reads_per_block - 1) / reads_per_block;
        gpu_align_SE<<<blocks_per_grid, threads_per_block, 0, ctx.stream>>>(s_len, s_len, d_index_para, global_align_info, d_aligner, local_d_pre_sum, local_d_len, local_d_seq,
                                                                                 global_references, d_map_param, global_nams, global_todo_ids, global_align_res,
                                                                                 d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset);
        cudaDeviceSynchronize();
        gpu_cost10 += GetTime() - t1;

    }

    tot_cost += GetTime() - t0;

}

void GPU_align_PE(std::vector<neoRcRef> &data1s, std::vector<neoRcRef> &data2s,
                  ThreadContext& ctx, std::vector<AlignTmpRes> &align_tmp_results,
                  uint64_t* global_hits_num, uint64_t* global_nams_info, uint64_t* global_align_info,
                  const StrobemerIndex& index, AlignmentParameters *d_aligner, MappingParameters* d_map_param, IndexParameters *d_index_para,
                  GPUReferences *global_references, RefRandstrobe *d_randstrobes, my_bucket_index_t *d_randstrobe_start_indices,
                  my_vector<QueryRandstrobe> *global_randstrobes, int *global_todo_ids, int *global_randstrobe_sizes, uint64_t * global_hashes_value,
                  my_vector<my_pair<int, Hit>> *global_hits_per_ref0s, my_vector<my_pair<int, Hit>> *global_hits_per_ref1s, my_vector<Nam> *global_nams, GPUInsertSizeDistribution& isize_est,
                  GPUAlignTmpRes *global_align_res, char *global_align_res_data, uint64_t pre_vec_size,
                  char *d_seq, int *d_len, int *d_pre_sum, char *h_seq, int *h_len, int *h_pre_sum,
                  int* d_todo_cnt, char* d_query_ptr, char* d_ref_ptr, int* d_query_offset, int* d_ref_offset, const int batch_read_num, const int batch_total_read_len, int rescue_threshold) {

    assert(data1s.size() == data2s.size());
    assert(data1s.size() <= batch_read_num);

    double t0, t1;
    t0 = GetTime();
    int l_id, r_id, s_len;

    // pack read on host
    t1 = GetTime();
    uint64_t tot_len = 0;
    h_pre_sum[0] = 0;
    s_len = data1s.size();
    for (int i = 0; i < s_len * 4; i++) {
        int read_id = i % s_len;
        if (i < s_len) { // read1 seq
            h_len[i] = data1s[read_id].read.lseq;
            h_pre_sum[i + 1] = h_pre_sum[i] + h_len[i];
            char* seq_ptr = (char*)data1s[read_id].read.base + data1s[read_id].read.pseq;
            memcpy(h_seq + h_pre_sum[i], seq_ptr, h_len[i]);
        } else if (i < s_len * 2) { // read1 rc
            h_len[i] = data1s[read_id].read.lseq;
            h_pre_sum[i + 1] = h_pre_sum[i] + h_len[i];
            char* seq_ptr = data1s[read_id].rc;
            memcpy(h_seq + h_pre_sum[i], seq_ptr, h_len[i]);
        } else if (i < s_len * 3) { // read2 seq
            h_len[i] = data2s[read_id].read.lseq;
            h_pre_sum[i + 1] = h_pre_sum[i] + h_len[i];
            char* seq_ptr = (char*)data2s[read_id].read.base + data2s[read_id].read.pseq;
            memcpy(h_seq + h_pre_sum[i], seq_ptr, h_len[i]);
        } else { // read2 rc
            h_len[i] = data2s[read_id].read.lseq;
            h_pre_sum[i + 1] = h_pre_sum[i] + h_len[i];
            char* seq_ptr = data2s[read_id].rc;
            memcpy(h_seq + h_pre_sum[i], seq_ptr, h_len[i]);
        }
    }
    tot_len = h_pre_sum[s_len * 4];
    assert(tot_len <= batch_total_read_len * 4);
    //printf("cal tot len %llu\n", tot_len);
    gpu_copy1 += GetTime() - t1;

    // transfer read to GPU
    t1 = GetTime();
    cudaMemcpy(d_seq, h_seq, tot_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_len, h_len, s_len * sizeof(int) * 4 + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pre_sum, h_pre_sum, s_len * sizeof(int) * 4 + 1, cudaMemcpyHostToDevice);
    gpu_copy2 += GetTime() - t1;

    for (l_id = 0; l_id < data1s.size(); l_id += batch_read_num) {
        r_id = l_id + batch_read_num;
        if (r_id > data1s.size()) r_id = data1s.size();
        s_len = r_id - l_id;

        char* local_d_seq = d_seq;
        int* local_d_len = d_len + l_id;
        int* local_d_pre_sum = d_pre_sum + l_id;

        // get randstrobes
        t1 = GetTime();
        int threads_per_block;
        int reads_per_block;
        int blocks_per_grid;
        threads_per_block = 4;
        reads_per_block = threads_per_block * GPU_thread_task_size;
        blocks_per_grid = (s_len * 2 + reads_per_block - 1) / reads_per_block;
        gpu_get_randstrobes<<<blocks_per_grid, threads_per_block, 0, ctx.stream>>>(s_len * 2, s_len, local_d_pre_sum, local_d_len, local_d_seq, d_index_para,
                                                                                   global_randstrobe_sizes, global_hashes_value, global_randstrobes);
        cudaDeviceSynchronize();
        gpu_cost1 += GetTime() - t1;
        //printf("get randstrobe done\n");

        // query database and get hits
        t1 = GetTime();
        threads_per_block = 4;
        reads_per_block = threads_per_block * GPU_thread_task_size;
        blocks_per_grid = (s_len * 2 + reads_per_block - 1) / reads_per_block;
        gpu_get_hits_pre<<<blocks_per_grid, threads_per_block, 0, ctx.stream>>>(index.bits, index.filter_cutoff, d_map_param->rescue_cutoff, d_randstrobes, index.randstrobes.size(), d_randstrobe_start_indices,
                                                                                s_len * 2, d_index_para, global_hits_num, global_randstrobes,
                                                                                global_hits_per_ref0s, global_hits_per_ref1s);
        cudaDeviceSynchronize();
        //printf("get hits pre done\n");
        threads_per_block = 4;
        reads_per_block = threads_per_block * GPU_thread_task_size;
        blocks_per_grid = (s_len * 2 + reads_per_block - 1) / reads_per_block;
        gpu_get_hits_after<<<blocks_per_grid, threads_per_block, 0, ctx.stream>>>(index.bits, index.filter_cutoff, d_map_param->rescue_cutoff, d_randstrobes, index.randstrobes.size(), d_randstrobe_start_indices,
                                                                                  s_len * 2, d_index_para, global_hits_num, global_randstrobes,
                                                                                  global_hits_per_ref0s, global_hits_per_ref1s);
        cudaDeviceSynchronize();
        //printf("get hits after done\n");
        gpu_cost2 += GetTime() - t1;

        // reads which pass the filter, normal mode
        t1 = GetTime();
        int todo_cnt = 0;
        for (int i = 0; i < s_len * 2; i++) {
            if (global_randstrobes[i].data == nullptr) {
                global_todo_ids[todo_cnt] = i;
                todo_cnt++;
            }
        }
        gpu_init1 += GetTime() - t1;

        // sort hits by ref_id
        t1 = GetTime();
        threads_per_block = 4;
        reads_per_block = threads_per_block * GPU_thread_task_size;
        blocks_per_grid = (todo_cnt + reads_per_block - 1) / reads_per_block;
        gpu_sort_hits<<<blocks_per_grid, threads_per_block, 0, ctx.stream>>>(todo_cnt, global_hits_per_ref0s, global_hits_per_ref1s, global_todo_ids);
        cudaDeviceSynchronize();
        gpu_cost3 += GetTime() - t1;
        //printf("sort hits done\n");

        // merge hits to NAMs
        t1 = GetTime();
        threads_per_block = 4;
        reads_per_block = threads_per_block * GPU_thread_task_size;
        blocks_per_grid = (todo_cnt + reads_per_block - 1) / reads_per_block;
        gpu_merge_hits_get_nams<<<blocks_per_grid, threads_per_block, 0, ctx.stream>>>(todo_cnt, d_index_para, global_nams_info,
                                                                                       global_hits_per_ref0s, global_hits_per_ref1s, global_nams, global_todo_ids);
        cudaDeviceSynchronize();
        gpu_cost4 += GetTime() - t1;

        // filter GG read for rescue
        t1 = GetTime();
        todo_cnt = 0;
        for (int i = 0; i < s_len * 2; i++) {
            if (global_randstrobes[i].data != nullptr) {
                global_todo_ids[todo_cnt] = i;
                todo_cnt++;
            }
        }
        gpu_init2 += GetTime() - t1;

        // rescue mode get hits
        t1 = GetTime();
        threads_per_block = 4;
        reads_per_block = threads_per_block * GPU_thread_task_size;
        blocks_per_grid = (todo_cnt + reads_per_block - 1) / reads_per_block;
        gpu_rescue_get_hits<<<blocks_per_grid, threads_per_block, 0, ctx.stream>>>(index.bits, index.filter_cutoff, d_map_param->rescue_cutoff, d_randstrobes, index.randstrobes.size(), d_randstrobe_start_indices,
                                                                                   todo_cnt, d_index_para, global_hits_num, global_randstrobes,
                                                                                   global_hits_per_ref0s, global_hits_per_ref1s, global_todo_ids, rescue_threshold);
        cudaDeviceSynchronize();
        gpu_cost5 += GetTime() - t1;

        // rescue mode sort hits by ref_id
        t1 = GetTime();
        threads_per_block = 4;
        reads_per_block = threads_per_block * GPU_thread_task_size;
        blocks_per_grid = (todo_cnt + reads_per_block - 1) / reads_per_block;
        gpu_rescue_sort_hits<<<blocks_per_grid, threads_per_block, 0, ctx.stream>>>(todo_cnt, global_hits_per_ref0s, global_hits_per_ref1s, global_todo_ids);
        cudaDeviceSynchronize();
        gpu_cost6 += GetTime() - t1;
        //printf("rescue sort hits done\n");

        // rescue mode merge hits to NAMs
        t1 = GetTime();
        threads_per_block = 4;
        reads_per_block = threads_per_block * GPU_thread_task_size;
        blocks_per_grid = (todo_cnt + reads_per_block - 1) / reads_per_block;
        gpu_rescue_merge_hits_get_nams<<<blocks_per_grid, threads_per_block, 0, ctx.stream>>>(todo_cnt, d_index_para, global_nams_info,
                                                                                              global_hits_per_ref0s, global_hits_per_ref1s, global_nams, global_todo_ids);
        cudaDeviceSynchronize();
        gpu_cost7 += GetTime() - t1;

        // sort nams for all reads
        t1 = GetTime();
        threads_per_block = 4;
        reads_per_block = threads_per_block * GPU_thread_task_size;
        blocks_per_grid = (s_len * 2 + reads_per_block - 1) / reads_per_block;
        gpu_sort_nams<<<blocks_per_grid, threads_per_block, 0, ctx.stream>>>(s_len * 2, global_nams, d_map_param, 0);
        cudaDeviceSynchronize();
        gpu_cost8 += GetTime() - t1;


        // pre-classification of reads
        t1 = GetTime();
        threads_per_block = 4;
        reads_per_block = threads_per_block * GPU_thread_task_size;
        blocks_per_grid = (s_len + reads_per_block - 1) / reads_per_block;
        gpu_pre_cal_type<<<blocks_per_grid, threads_per_block, 0, ctx.stream>>>(s_len, d_map_param->dropoff_threshold, global_nams, isize_est, global_todo_ids);
        cudaDeviceSynchronize();
        gpu_cost9 += GetTime() - t1;

        // allocate memory for align_tmp_res
        t1 = GetTime();
        std::vector<int> types[5];
        char* base_ptr = global_align_res_data;
        for (int i = 0; i < s_len; i++) {
            if (global_todo_ids[i] > 4) printf("GG type %d\n", global_todo_ids[i]);
            assert(global_todo_ids[i] <= 4);
            types[global_todo_ids[i]].push_back(i);

            int tries_num = MAX_TRIES_LIMIT;
            if (global_todo_ids[i] == 3 || global_todo_ids[i] == 0)  tries_num = 2;

            GPUAlignTmpRes *tmp = global_align_res + i;
            tmp->type = global_todo_ids[i];
            tmp->mapq1 = 0, tmp->mapq2 = 0, tmp->type4_loop_size = 0;

            tmp->is_extend_seed.data = (int*)base_ptr;
            tmp->is_extend_seed.length = 0;
            tmp->is_extend_seed.capacity = tries_num;
            base_ptr += tries_num * sizeof(int);

            tmp->consistent_nam.data = (int*)base_ptr;
            tmp->consistent_nam.length = 0;
            tmp->consistent_nam.capacity = tries_num;
            base_ptr += tries_num * sizeof(int);

            tmp->is_read1.data = (int*)base_ptr;
            tmp->is_read1.length = 0;
            tmp->is_read1.capacity = tries_num;
            base_ptr += tries_num * sizeof(int);

            tmp->type4_nams.data = (Nam*)base_ptr;
            tmp->type4_nams.length = 0;
            tmp->type4_nams.capacity = tries_num;
            base_ptr += tries_num * sizeof(Nam);

            tmp->todo_nams.data = (Nam*)base_ptr;
            tmp->todo_nams.length = 0;
            tmp->todo_nams.capacity = tries_num;
            base_ptr += tries_num * sizeof(Nam);

            tmp->done_align.data = (int*)base_ptr;
            tmp->done_align.length = 0;
            tmp->done_align.capacity = tries_num;
            base_ptr += tries_num * sizeof(int);

            tmp->align_res.data = (GPUAlignment*)base_ptr;
            tmp->align_res.length = 0;
            tmp->align_res.capacity = tries_num;
            base_ptr += tries_num * sizeof(GPUAlignment);

            tmp->cigar_info.data = (CigarData*)base_ptr;
            tmp->cigar_info.length = 0;
            tmp->cigar_info.capacity = tries_num;
            base_ptr += tries_num * sizeof(CigarData);
            //for (int j = 0; j < tries_num; j++) {
            //    memset(tmp->cigar_info[j].gpu_cigar, 0, sizeof(tmp->cigar_info[j].gpu_cigar));
            //    tmp->cigar_info[j].cigar = tmp->cigar_info[j].gpu_cigar;
            //    tmp->cigar_info[j].has_realloc = 0;
            //}

            tmp->todo_infos.data = (TODOInfos*)base_ptr;
            tmp->todo_infos.length = 0;
            tmp->todo_infos.capacity = tries_num;
            base_ptr += tries_num * sizeof(TODOInfos);
        }
        gpu_init3 += GetTime() - t1;
//        printf("types: %d %d %d %d %d\n", types[0].size(), types[1].size(), types[2].size(), types[3].size(), types[4].size());

        // align reads
        t1 = GetTime();
        int r_pos = 0;
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < types[i].size(); j++) {
                global_todo_ids[r_pos++] = types[i][j];
            }
        }

        assert(r_pos == s_len);
        threads_per_block = 4;
        reads_per_block = threads_per_block * GPU_thread_task_size;
        blocks_per_grid = (s_len + reads_per_block - 1) / reads_per_block;
        gpu_align_PE01234<<<blocks_per_grid, threads_per_block, 0, ctx.stream>>>(s_len, s_len, d_index_para, global_align_info, d_aligner, local_d_pre_sum, local_d_len, local_d_seq,
                                                                                 global_references, d_map_param, global_nams, isize_est, global_todo_ids, global_align_res,
                                                                                 d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset);
        cudaDeviceSynchronize();
        gpu_cost10 += GetTime() - t1;

    }

    tot_cost += GetTime() - t0;

}

std::once_flag init_flag_ref[GPU_NUM_MAX];
std::once_flag init_flag_pool[GPU_NUM_MAX];

GPUReferences *global_references[GPU_NUM_MAX];
RefRandstrobe *d_randstrobes[GPU_NUM_MAX];
my_bucket_index_t *d_randstrobe_start_indices[GPU_NUM_MAX];

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


void PrintStr(const char* str, int len) {
    for(int i = 0; i < len; i++) printf("%c", str[i]);
    printf("\n");
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


GPUAlignTmpRes *g_chunk0_global_align_res[THREAD_NUM_MAX];
GPUAlignTmpRes *g_chunk1_global_align_res[THREAD_NUM_MAX];
GPUAlignTmpRes *g_chunk2_global_align_res[THREAD_NUM_MAX];
char *g_chunk0_global_align_res_data[THREAD_NUM_MAX];
char *g_chunk1_global_align_res_data[THREAD_NUM_MAX];
char *g_chunk2_global_align_res_data[THREAD_NUM_MAX];

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

void perform_task_async_pe_fx_GPU(
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
        rabbit::core::TDataQueue<rabbit::fq::FastqDataPairChunk> &dq,
        const bool use_good_numa,
        const int gpu_id,
        const int async_thread_id,
        const int batch_read_num,
        const int batch_total_read_len,
        const int chunk_num
) {

    if(use_good_numa) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(thread_id, &cpuset);
        pthread_t current_thread = pthread_self();
//        std::cout << "Setting thread affinity to CPU " << thread_id << std::endl;
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

    time0_1 += GetTime() - t_2;

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
    cudaMallocManaged(&global_randstrobes, batch_read_num * 2 * sizeof(my_vector<QueryRandstrobe>));
    meta_data_size + batch_read_num * 2 * sizeof(my_vector<QueryRandstrobe>);
    cudaMemset(global_randstrobes, 0, batch_read_num * 2 * sizeof(my_vector<QueryRandstrobe>));
    int *global_todo_ids;
    cudaMallocManaged(&global_todo_ids, batch_read_num * 2 * sizeof(int));
    meta_data_size += batch_read_num * 2 * sizeof(int);
    cudaMemset(global_todo_ids, 0, batch_read_num * 2 * sizeof(int));
    int *global_randstrobe_sizes;
    cudaMallocManaged(&global_randstrobe_sizes, batch_read_num * 2 * sizeof(int));
    meta_data_size += batch_read_num * 2 * sizeof(int);
    cudaMemset(global_randstrobe_sizes, 0, batch_read_num * 2 * sizeof(int));
    uint64_t * global_hashes_value;
    cudaMallocManaged(&global_hashes_value, batch_read_num * 2 * sizeof(uint64_t));
    meta_data_size += batch_read_num * 2 * sizeof(uint64_t);
    cudaMemset(global_hashes_value, 0, batch_read_num * 2 * sizeof(uint64_t));
    my_vector<my_pair<int, Hit>> *global_hits_per_ref0s;
    cudaMallocManaged(&global_hits_per_ref0s, batch_read_num * 2 * sizeof(my_vector<my_pair<int, Hit>>));
    meta_data_size += batch_read_num * 2 * sizeof(my_vector<my_pair<int, Hit>>);
    cudaMemset(global_hits_per_ref0s, 0, batch_read_num * 2 * sizeof(my_vector<my_pair<int, Hit>>));
    my_vector<my_pair<int, Hit>> *global_hits_per_ref1s;
    cudaMallocManaged(&global_hits_per_ref1s, batch_read_num * 2 * sizeof(my_vector<my_pair<int, Hit>>));
    meta_data_size += batch_read_num * 2 * sizeof(my_vector<my_pair<int, Hit>>);
    cudaMemset(global_hits_per_ref1s, 0, batch_read_num * 2 * sizeof(my_vector<my_pair<int, Hit>>));
    my_vector<Nam> *global_nams;
    cudaMallocManaged(&global_nams, batch_read_num * 2 * sizeof(my_vector<Nam>));
    meta_data_size += batch_read_num * 2 * sizeof(my_vector<Nam>);
    cudaMemset(global_nams, 0, batch_read_num * 2 * sizeof(my_vector<Nam>));

    uint64_t * global_hits_num;
    cudaMallocManaged(&global_hits_num, batch_read_num * 2 * sizeof(uint64_t));
    meta_data_size += batch_read_num * 2 * sizeof(uint64_t);
    cudaMemset(global_hits_num, 0, batch_read_num * 2 * sizeof(uint64_t));

    uint64_t * global_nams_info;
    cudaMallocManaged(&global_nams_info, batch_read_num * 2 * sizeof(uint64_t));
    meta_data_size += batch_read_num * 2 * sizeof(uint64_t);
    cudaMemset(global_nams_info, 0, batch_read_num * 2 * sizeof(uint64_t));

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
    cudaMalloc(&d_seq, seq_size_alloc * 4);
    cudaMemset(d_seq, 0, seq_size_alloc * 4);
    cudaMalloc(&d_len, (batch_read_num + 1) * sizeof(int) * 4);
    cudaMemset(d_len, 0, (batch_read_num + 1) * sizeof(int) * 4);
    cudaMalloc(&d_pre_sum, (batch_read_num + 1) * sizeof(int) * 4);
    cudaMemset(d_pre_sum, 0, (batch_read_num + 1) * sizeof(int) * 4);

    if (gpu_id == 0) printf("--- seq GPU mem alloc %llu\n", seq_size_alloc * 4 + (batch_read_num + 1) * sizeof(int) * 4 * 2);

    int *h_len;
    int *h_pre_sum;
    char *h_seq;
    cudaHostAlloc(&h_seq, seq_size_alloc * 4, cudaHostAllocDefault);
    cudaMemset(h_seq, 0, seq_size_alloc * 4);
    cudaHostAlloc(&h_len, (batch_read_num + 1) * sizeof(int) * 4, cudaHostAllocDefault);
    cudaMemset(h_len, 0, (batch_read_num + 1) * sizeof(int) * 4);
    cudaHostAlloc(&h_pre_sum, (batch_read_num + 1) * sizeof(int) * 4, cudaHostAllocDefault);
    cudaMemset(h_pre_sum, 0, (batch_read_num + 1) * sizeof(int) * 4);

    const int mx_device_query_size = chunk_num * DEVICE_TODO_SIZE_PER_CHUNK;
    const int mx_device_ref_size = mx_device_query_size * 2;

    char* device_query_ptr;
    char* device_ref_ptr;
#ifdef use_device_mem
    cudaMalloc(&device_query_ptr, mx_device_query_size);
    cudaMalloc(&device_ref_ptr, mx_device_ref_size);
    if (gpu_id == 0) printf("--- todo GPU mem alloc %llu\n", mx_device_query_size + mx_device_ref_size);
#endif


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
    std::vector<neoReference> neo_data1s;
    std::vector<neoReference> neo_data2s;
    std::vector<AlignmentInfo> info_results;
#ifdef use_device_mem
    std::vector<int> todo_querys;
    std::vector<int> todo_refs;
    todo_querys.reserve(batch_read_num * 2);
    todo_refs.reserve(batch_read_num * 2);
#endif

    char* chunk0_rc_data1 = new char[batch_total_read_len];
    char* chunk0_rc_data2 = new char[batch_total_read_len];
    rabbit::fq::FastqDataPairChunk *chunk0_fqdatachunks[MAX_RABBITFX_CHUNK_NUM];
    int chunk0_real_chunk_num;
    int chunk0_chunk_num;
    std::vector<neoRcRef> chunk0_data1s;
    std::vector<neoRcRef> chunk0_data2s;
    std::vector<std::string_view> chunk0_h_todo_querys;
    std::vector<std::string_view> chunk0_h_todo_refs;
    std::vector<gasal_tmp_res> chunk0_gasal_results;
    chunk0_h_todo_querys.reserve(batch_read_num * 2);
    chunk0_h_todo_refs.reserve(batch_read_num * 2);
    chunk0_gasal_results.reserve(batch_read_num * 2);


    char* chunk1_rc_data1 = new char[batch_total_read_len];
    char* chunk1_rc_data2 = new char[batch_total_read_len];
    rabbit::fq::FastqDataPairChunk *chunk1_fqdatachunks[MAX_RABBITFX_CHUNK_NUM];
    int chunk1_real_chunk_num;
    int chunk1_chunk_num;
    std::vector<neoRcRef> chunk1_data1s;
    std::vector<neoRcRef> chunk1_data2s;
    std::vector<std::string_view> chunk1_h_todo_querys;
    std::vector<std::string_view> chunk1_h_todo_refs;
    std::vector<gasal_tmp_res> chunk1_gasal_results;
    chunk1_h_todo_querys.reserve(batch_read_num * 2);
    chunk1_h_todo_refs.reserve(batch_read_num * 2);
    chunk1_gasal_results.reserve(batch_read_num * 2);


    char* chunk2_rc_data1 = new char[batch_total_read_len];
    char* chunk2_rc_data2 = new char[batch_total_read_len];
    rabbit::fq::FastqDataPairChunk *chunk2_fqdatachunks[MAX_RABBITFX_CHUNK_NUM];
    int chunk2_real_chunk_num;
    int chunk2_chunk_num;
    std::vector<neoRcRef> chunk2_data1s;
    std::vector<neoRcRef> chunk2_data2s;
    std::vector<std::string_view> chunk2_h_todo_querys;
    std::vector<std::string_view> chunk2_h_todo_refs;
    std::vector<gasal_tmp_res> chunk2_gasal_results;
    chunk2_h_todo_querys.reserve(batch_read_num * 2);
    chunk2_h_todo_refs.reserve(batch_read_num * 2);
    chunk2_gasal_results.reserve(batch_read_num * 2);

    rabbit::int64 id;

    time0_4 += GetTime() - t_2;

    time0 += GetTime() - t_1;

    const int small_chunk_num = chunk_num / 2;

    int read_len = 150;


    // step: f_0
    {
        bool res;
        // format data
        t_1 = GetTime();
        int rc_pos1 = 0, rc_pos2 = 0;
        chunk0_chunk_num = rand() % small_chunk_num + small_chunk_num + 1;
//        chunk0_chunk_num = chunk_num;
        chunk0_real_chunk_num = 0;
        chunk0_data1s.clear();
        chunk0_data2s.clear();
//        printf("chunk0_chunk_num %d\n", chunk0_chunk_num);
        for (int chunk_id = 0; chunk_id < chunk0_chunk_num; chunk_id++) {
            res = dq.Pop(id, chunk0_fqdatachunks[chunk_id]);
            if (res) {
                double t_3 = GetTime();
                neo_data1s.clear();
                neo_data2s.clear();
                rabbit::fq::chunkFormat((rabbit::fq::FastqDataChunk * )(chunk0_fqdatachunks[chunk_id]->left_part), neo_data1s);
                rabbit::fq::chunkFormat((rabbit::fq::FastqDataChunk * )(chunk0_fqdatachunks[chunk_id]->right_part), neo_data2s);
                time1_1_1 += GetTime() - t_3;

                assert(neo_data1s.size() == neo_data2s.size());
                t_3 = GetTime();
                for (int i = 0; i < neo_data1s.size(); i++) {
                    char *name1 = (char *) neo_data1s[i].base + neo_data1s[i].pname;
                    if (neo_data1s[i].lname > 0 && name1[0] == '@') {
                        neo_data1s[i].pname++;
                        neo_data1s[i].lname--;
                        name1++;
                    }
                    for (int j = 0; j < neo_data1s[i].lname; j++) {
                        if (name1[j] == ' ') {
                            neo_data1s[i].lname = j;
                            break;
                        }
                    }
                    char *name2 = (char *) neo_data2s[i].base + neo_data2s[i].pname;
                    if (neo_data2s[i].lname > 0 && name2[0] == '@') {
                        neo_data2s[i].pname++;
                        neo_data2s[i].lname--;
                        name2++;
                    }
                    for (int j = 0; j < neo_data2s[i].lname; j++) {
                        if (name2[j] == ' ') {
                            neo_data2s[i].lname = j;
                            break;
                        }
                    }
                    read_len = std::max(read_len, (int)neo_data1s[i].lseq);
                    char *seq1 = (char *) neo_data1s[i].base + neo_data1s[i].pseq;
                    chunk0_data1s.push_back({neo_data1s[i], chunk0_rc_data1 + rc_pos1});
                    for (int j = 0; j < neo_data1s[i].lseq; j++) {
                        chunk0_rc_data1[rc_pos1++] = rc_gpu_nt2nt[seq1[neo_data1s[i].lseq - 1 - j]];
                    }
                    char *seq2 = (char *) neo_data2s[i].base + neo_data2s[i].pseq;
                    chunk0_data2s.push_back({neo_data2s[i], chunk0_rc_data2 + rc_pos2});
                    for (int j = 0; j < neo_data2s[i].lseq; j++) {
                        chunk0_rc_data2[rc_pos2++] = rc_gpu_nt2nt[seq2[neo_data2s[i].lseq - 1 - j]];
                    }
                }
                time1_1_2 += GetTime() - t_3;
                chunk0_real_chunk_nums[chunk_id] = neo_data1s.size();
                chunk0_real_chunk_ids[chunk_id] = id;
                chunk0_real_chunk_num++;
            } else break;
        }
        assert(rc_pos1 <= batch_total_read_len && rc_pos2 <= batch_total_read_len);
        time1_1 += GetTime() - t_1;
    }

    //int rescue_threshold = read_len;
    int rescue_threshold = 100;
//    printf("rescue_threshold %d\n", rescue_threshold);


    // step: f_1
    {
        bool res;
        // format data
        t_1 = GetTime();
        int rc_pos1 = 0, rc_pos2 = 0;
        chunk1_chunk_num = rand() % small_chunk_num + small_chunk_num + 1;
//        chunk1_chunk_num = chunk_num;
        chunk1_real_chunk_num = 0;
        chunk1_data1s.clear();
        chunk1_data2s.clear();
        for (int chunk_id = 0; chunk_id < chunk1_chunk_num; chunk_id++) {
            res = dq.Pop(id, chunk1_fqdatachunks[chunk_id]);
            if (res) {
                double t_3 = GetTime();
                neo_data1s.clear();
                neo_data2s.clear();
                rabbit::fq::chunkFormat((rabbit::fq::FastqDataChunk * )(chunk1_fqdatachunks[chunk_id]->left_part), neo_data1s);
                rabbit::fq::chunkFormat((rabbit::fq::FastqDataChunk * )(chunk1_fqdatachunks[chunk_id]->right_part), neo_data2s);
                time1_1_1 += GetTime() - t_3;

//                printf("neo %d %d\n", neo_data1s.size(), neo_data2s.size());
                assert(neo_data1s.size() == neo_data2s.size());
                t_3 = GetTime();
                for (int i = 0; i < neo_data1s.size(); i++) {
                    char *name1 = (char *) neo_data1s[i].base + neo_data1s[i].pname;
                    if (neo_data1s[i].lname > 0 && name1[0] == '@') {
                        neo_data1s[i].pname++;
                        neo_data1s[i].lname--;
                        name1++;
                    }
                    for (int j = 0; j < neo_data1s[i].lname; j++) {
                        if (name1[j] == ' ') {
                            neo_data1s[i].lname = j;
                            break;
                        }
                    }
                    char *name2 = (char *) neo_data2s[i].base + neo_data2s[i].pname;
                    if (neo_data2s[i].lname > 0 && name2[0] == '@') {
                        neo_data2s[i].pname++;
                        neo_data2s[i].lname--;
                        name2++;
                    }
                    for (int j = 0; j < neo_data2s[i].lname; j++) {
                        if (name2[j] == ' ') {
                            neo_data2s[i].lname = j;
                            break;
                        }
                    }
                    char *seq1 = (char *) neo_data1s[i].base + neo_data1s[i].pseq;
                    chunk1_data1s.push_back({neo_data1s[i], chunk1_rc_data1 + rc_pos1});
                    for (int j = 0; j < neo_data1s[i].lseq; j++) {
                        chunk1_rc_data1[rc_pos1++] = rc_gpu_nt2nt[seq1[neo_data1s[i].lseq - 1 - j]];
                    }
                    char *seq2 = (char *) neo_data2s[i].base + neo_data2s[i].pseq;
                    chunk1_data2s.push_back({neo_data2s[i], chunk1_rc_data2 + rc_pos2});
                    for (int j = 0; j < neo_data2s[i].lseq; j++) {
                        chunk1_rc_data2[rc_pos2++] = rc_gpu_nt2nt[seq2[neo_data2s[i].lseq - 1 - j]];
                    }
                }
                time1_1_2 += GetTime() - t_3;
                chunk1_real_chunk_nums[chunk_id] = neo_data1s.size();
                chunk1_real_chunk_ids[chunk_id] = id;
                chunk1_real_chunk_num++;
            } else break;
        }
        assert(rc_pos1 <= batch_total_read_len && rc_pos2 <= batch_total_read_len);
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
        if (!chunk0_data1s.empty()) GPU_align_PE(chunk0_data1s, chunk0_data2s,
                                ctx,
                                align_tmp_results,
                                global_hits_num, global_nams_info, global_align_info,
                                index, d_aligner, d_map_param, d_index_para,
                                global_references[gpu_id], d_randstrobes[gpu_id], d_randstrobe_start_indices[gpu_id],
                                global_randstrobes, global_todo_ids, global_randstrobe_sizes, global_hashes_value,
                                global_hits_per_ref0s, global_hits_per_ref1s, global_nams, *isize_est,
                                chunk0_global_align_res, chunk0_global_align_res_data, pre_vec_size,
                                d_seq, d_len, d_pre_sum, h_seq, h_len, h_pre_sum,
                                d_todo_cnt, device_query_ptr, device_ref_ptr, d_query_offset, d_ref_offset, batch_read_num, batch_total_read_len, rescue_threshold);
        cudaMemcpy(h_todo_cnt, d_todo_cnt, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_query_offset, d_query_offset, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_ref_offset, d_ref_offset, sizeof(int), cudaMemcpyDeviceToHost);
        assert(*h_query_offset <= mx_device_query_size);
        assert(*h_ref_offset <= mx_device_ref_size);
        time1_2 += GetTime() - t_1;

//        printf("todo cnt is %d, %d, %d\n", *h_todo_cnt, *h_query_offset, *h_ref_offset);

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
        for (int i = 0; i < chunk0_data1s.size(); i++) {
            GPUAlignTmpRes &align_tmp_res = chunk0_global_align_res[i];
            //printf("read %d, type %d, todo size %d\n", i, align_tmp_res.type, align_tmp_res.todo_infos.size());
            for (int j = 0; j < align_tmp_res.todo_infos.size(); j++) {
                TODOInfos& todo_info = align_tmp_res.todo_infos[j];
                int global_id = todo_info.global_id;
                uint32_t info = todo_info.read_info;
                int is_read1 = (info >> 31) & 0x1;
                int is_rc    = (info >> 30) & 0x1;
                int q_begin  = (info >> 15) & 0x7FFF;
                int q_len    = info & 0x7FFF;
                const auto& h_query_seq = is_read1 ? (is_rc ? chunk0_data1s[i].rc : (char*)chunk0_data1s[i].read.base + chunk0_data1s[i].read.pseq) :
                                          (is_rc ? chunk0_data2s[i].rc : (char*)chunk0_data2s[i].read.base + chunk0_data2s[i].read.pseq);
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
//        printf("cal_todo_cnt %d, h_todo_cnt %d\n", cal_todo_cnt, *h_todo_cnt);
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
//    printf("thread %d bind to %d - %d\n", thread_id, thread_id, async_thread_id);

    while (true) {

		// update isize_est
		for (int i = 0; i < chunk0_data1s.size(); i++) {
		    GPUAlignTmpRes &align_tmp_res = chunk0_global_align_res[i];
            size_t todo_size = align_tmp_res.todo_nams.size();
            if (align_tmp_res.type != 3) continue;
            if (align_tmp_res.type3_isize_val == -1) continue;
            isize_est->update(align_tmp_res.type3_isize_val);
		}

        if (chunk0_data1s.size() == 0) break;
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
                    if (gasal_fail(todo_q, todo_r, chunk0_gasal_results[i])) {
                        double ta = GetTime();
                        info = aligner.align(todo_q, todo_r);
                        time2_3_1 += GetTime() - ta;
                    } else {
                        double ta = GetTime();
                        info = aligner.align_gpu(todo_q, todo_r, chunk0_gasal_results[i]);
                        time2_3_2 += GetTime() - ta;
                    }
                    info_results.push_back(info);
                }
                time2_3 += GetTime() - t_start;

                // update align_res using ssw result
                t_start = GetTime();
                for (size_t i = 0; i < chunk0_data1s.size(); i++) {
                    const auto mu = isize_est->mu;
                    const auto sigma = isize_est->sigma;
                    GPUAlignTmpRes &align_tmp_res = chunk0_global_align_res[i];
                    size_t todo_size = align_tmp_res.todo_nams.size();
                    if (align_tmp_res.type == 1 || align_tmp_res.type == 2) {
                        int pos = 0;
                        for (size_t j = 0; j < todo_size; j += 2) {
                            if (!align_tmp_res.done_align[j]) {
                                GPU_part2_extend_seed_store_res(
                                        align_tmp_res, j, chunk0_data1s[i], chunk0_data2s[i], references,
                                        info_results[align_tmp_res.todo_infos[pos++].global_id]
                                );
                            }
                            if (!align_tmp_res.done_align[j + 1]) {
                                GPU_part2_rescue_mate_store_res(
                                        align_tmp_res, j + 1, chunk0_data1s[i], chunk0_data2s[i], references,
                                        info_results[align_tmp_res.todo_infos[pos++].global_id], mu, sigma
                                );
                            }
                        }
                        assert(pos == align_tmp_res.todo_infos.size());
                    } else if (align_tmp_res.type == 3) {
                        int pos = 0;
                        if (!align_tmp_res.done_align[0]) {
                            GPU_part2_extend_seed_store_res(
                                    align_tmp_res, 0, chunk0_data1s[i], chunk0_data2s[i], references,
                                    info_results[align_tmp_res.todo_infos[pos++].global_id]
                            );
                        }
                        if (!align_tmp_res.done_align[1]) {
                            GPU_part2_extend_seed_store_res(
                                    align_tmp_res, 1, chunk0_data1s[i], chunk0_data2s[i], references,
                                    info_results[align_tmp_res.todo_infos[pos++].global_id]
                            );
                        }
                        assert(pos == align_tmp_res.todo_infos.size());
                    } else if (align_tmp_res.type == 4) {
                        int pos = 0;
                        for (size_t j = 0; j < todo_size; j++) {
                            if (!align_tmp_res.done_align[j]) {
                                if (align_tmp_res.is_extend_seed[j]) {
                                    GPU_part2_extend_seed_store_res(
                                            align_tmp_res, j, chunk0_data1s[i], chunk0_data2s[i], references,
                                            info_results[align_tmp_res.todo_infos[pos++].global_id]
                                    );
                                } else {
                                    GPU_part2_rescue_mate_store_res(
                                            align_tmp_res, j, chunk0_data1s[i], chunk0_data2s[i], references,
                                            info_results[align_tmp_res.todo_infos[pos++].global_id], mu, sigma
                                    );
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
                        GPU_align_PE_read_last(chunk0_global_align_res[i], chunk0_data1s[i], chunk0_data2s[i], sam, sam_out,
                                               *isize_est, aligner,
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
                    if (sam_out.length() > 0) output_buffer.output_records(std::move(sam_out), fx_chunk_id);
                    time3_2 += GetTime() - t_start;

                    base_read_num += this_chunk_read_num;
                }
                assert(base_read_num == chunk0_data1s.size());

                // release rabbitfx chunk
                t_start = GetTime();
                for (int chunk_id = 0; chunk_id < chunk0_real_chunk_num; chunk_id++) {
                    fastqPool.Release(chunk0_fqdatachunks[chunk_id]->left_part);
                    fastqPool.Release(chunk0_fqdatachunks[chunk_id]->right_part);
                }
                time3_3 += GetTime() - t_start;
            }

            // step: f_2
            {
                t_start = GetTime();
                bool res;
                // format data
                int rc_pos1 = 0, rc_pos2 = 0;
                chunk2_chunk_num = rand() % small_chunk_num + small_chunk_num + 1;
//                chunk2_chunk_num = chunk_num;
                chunk2_real_chunk_num = 0;
                chunk2_data1s.clear();
                chunk2_data2s.clear();
                for (int chunk_id = 0; chunk_id < chunk2_chunk_num; chunk_id++) {
                    res = dq.Pop(id, chunk2_fqdatachunks[chunk_id]);
                    if (res) {
                        double t_3 = GetTime();
                        neo_data1s.clear();
                        neo_data2s.clear();
                        rabbit::fq::chunkFormat((rabbit::fq::FastqDataChunk * )(chunk2_fqdatachunks[chunk_id]->left_part), neo_data1s);
                        rabbit::fq::chunkFormat((rabbit::fq::FastqDataChunk * )(chunk2_fqdatachunks[chunk_id]->right_part), neo_data2s);
                        time1_1_1 += GetTime() - t_3;

                        assert(neo_data1s.size() == neo_data2s.size());
                        t_3 = GetTime();
                        for (int i = 0; i < neo_data1s.size(); i++) {
                            char *name1 = (char *) neo_data1s[i].base + neo_data1s[i].pname;
                            if (neo_data1s[i].lname > 0 && name1[0] == '@') {
                                neo_data1s[i].pname++;
                                neo_data1s[i].lname--;
                                name1++;
                            }
                            for (int j = 0; j < neo_data1s[i].lname; j++) {
                                if (name1[j] == ' ') {
                                    neo_data1s[i].lname = j;
                                    break;
                                }
                            }
                            char *name2 = (char *) neo_data2s[i].base + neo_data2s[i].pname;
                            if (neo_data2s[i].lname > 0 && name2[0] == '@') {
                                neo_data2s[i].pname++;
                                neo_data2s[i].lname--;
                                name2++;
                            }
                            for (int j = 0; j < neo_data2s[i].lname; j++) {
                                if (name2[j] == ' ') {
                                    neo_data2s[i].lname = j;
                                    break;
                                }
                            }
                            char *seq1 = (char *) neo_data1s[i].base + neo_data1s[i].pseq;
                            chunk2_data1s.push_back({neo_data1s[i], chunk2_rc_data1 + rc_pos1});
                            for (int j = 0; j < neo_data1s[i].lseq; j++) {
                                chunk2_rc_data1[rc_pos1++] = rc_gpu_nt2nt[seq1[neo_data1s[i].lseq - 1 - j]];
                            }
                            char *seq2 = (char *) neo_data2s[i].base + neo_data2s[i].pseq;
                            chunk2_data2s.push_back({neo_data2s[i], chunk2_rc_data2 + rc_pos2});
                            for (int j = 0; j < neo_data2s[i].lseq; j++) {
                                chunk2_rc_data2[rc_pos2++] = rc_gpu_nt2nt[seq2[neo_data2s[i].lseq - 1 - j]];
                            }
                        }
                        time1_1_2 += GetTime() - t_3;
                        chunk2_real_chunk_nums[chunk_id] = neo_data1s.size();
                        chunk2_real_chunk_ids[chunk_id] = id;
                        chunk2_real_chunk_num++;
                    } else break;
                }
                assert(rc_pos1 <= batch_total_read_len && rc_pos2 <= batch_total_read_len);
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
            if (!chunk1_data1s.empty()) GPU_align_PE(chunk1_data1s, chunk1_data2s,
                                    ctx,
                                    align_tmp_results,
                                    global_hits_num, global_nams_info, global_align_info,
                                    index, d_aligner, d_map_param, d_index_para,
                                    global_references[gpu_id], d_randstrobes[gpu_id], d_randstrobe_start_indices[gpu_id],
                                    global_randstrobes, global_todo_ids, global_randstrobe_sizes, global_hashes_value,
                                    global_hits_per_ref0s, global_hits_per_ref1s, global_nams, *isize_est,
                                    chunk1_global_align_res, chunk1_global_align_res_data, pre_vec_size,
                                    d_seq, d_len, d_pre_sum, h_seq, h_len, h_pre_sum,
                                    d_todo_cnt, device_query_ptr, device_ref_ptr, d_query_offset, d_ref_offset, batch_read_num, batch_total_read_len, rescue_threshold);
            cudaMemcpy(h_todo_cnt, d_todo_cnt, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_query_offset, d_query_offset, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_ref_offset, d_ref_offset, sizeof(int), cudaMemcpyDeviceToHost);
            assert(*h_query_offset <= mx_device_query_size);
            assert(*h_ref_offset <= mx_device_ref_size);
            time1_2 += GetTime() - t_1;

//            printf("todo cnt is %d, %d, %d\n", *h_todo_cnt, *h_query_offset, *h_ref_offset);

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
            for (int i = 0; i < chunk1_data1s.size(); i++) {
                GPUAlignTmpRes &align_tmp_res = chunk1_global_align_res[i];
                //printf("read %d, type %d, todo size %d\n", i, align_tmp_res.type, align_tmp_res.todo_infos.size());
                for (int j = 0; j < align_tmp_res.todo_infos.size(); j++) {
                    TODOInfos& todo_info = align_tmp_res.todo_infos[j];
                    int global_id = todo_info.global_id;
                    uint32_t info = todo_info.read_info;
                    int is_read1 = (info >> 31) & 0x1;
                    int is_rc    = (info >> 30) & 0x1;
                    int q_begin  = (info >> 15) & 0x7FFF;
                    int q_len    = info & 0x7FFF;
                    const auto& h_query_seq = is_read1 ? (is_rc ? chunk1_data1s[i].rc : (char*)chunk1_data1s[i].read.base + chunk1_data1s[i].read.pseq) :
                                              (is_rc ? chunk1_data2s[i].rc : (char*)chunk1_data2s[i].read.base + chunk1_data2s[i].read.pseq);
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
//            printf("cal_todo_cnt %d, h_todo_cnt %d\n", cal_todo_cnt, *h_todo_cnt);
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
            std::vector<neoRcRef> temp_data1s = std::move(chunk0_data1s);
            chunk0_data1s = std::move(chunk1_data1s);
            chunk1_data1s = std::move(chunk2_data1s);
            chunk2_data1s = std::move(temp_data1s);

            std::vector<neoRcRef> temp_data2s = std::move(chunk0_data2s);
            chunk0_data2s = std::move(chunk1_data2s);
            chunk1_data2s = std::move(chunk2_data2s);
            chunk2_data2s = std::move(temp_data2s);

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

            char* temp_rc_data1 = chunk0_rc_data1;
            chunk0_rc_data1 = chunk1_rc_data1;
            chunk1_rc_data1 = chunk2_rc_data1;
            chunk2_rc_data1 = temp_rc_data1;

            char* temp_rc_data2 = chunk0_rc_data2;
            chunk0_rc_data2 = chunk1_rc_data2;
            chunk1_rc_data2 = chunk2_rc_data2;
            chunk2_rc_data2 = temp_rc_data2;

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
                rabbit::fq::FastqDataPairChunk * temp_fqdatachunk = chunk0_fqdatachunks[i];
                chunk0_fqdatachunks[i] = chunk1_fqdatachunks[i];
                chunk1_fqdatachunks[i] = chunk2_fqdatachunks[i];
                chunk2_fqdatachunks[i] = temp_fqdatachunk;
            }

            time3_5 += GetTime() - t_1;
        }
    }
    done = true;

#ifdef PRINT_GPU_TIMER
    std::cout << "gpu cost " << gpu_copy1 << " " << gpu_copy2 << " " << gpu_cost1 << " " << gpu_cost2 << " [" << gpu_cost2_1 << " " << gpu_cost2_2 << "] " << gpu_cost3 << " " << gpu_cost4 << std::endl;
    std::cout << gpu_cost5 << " " << gpu_cost6 << " " << gpu_cost7 << " " << gpu_cost8 << " " << gpu_cost9 << " " << gpu_cost10 << std::endl;
//    std::cout << "[" << gpu_cost10_0 << " " << gpu_cost10_1 << " " << gpu_cost10_2 << " " << gpu_cost10_3 << " " << gpu_cost10_4 << "]" << std::endl;
//    std::cout << "copy data to host cost " << gpu_cost11 << " [" << gpu_cost11_copy1 << ", " << gpu_cost11_copy2 << "]" << std::endl;
    std::cout << "init data " << gpu_init1 << " " << gpu_init2 << " " << gpu_init3 << " " << gpu_init4 << std::endl;
    std::cout << "total cost " << tot_cost << std::endl;
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


    delete[] chunk0_rc_data1;
    delete[] chunk0_rc_data2;
    delete[] chunk1_rc_data1;
    delete[] chunk1_rc_data2;
    delete[] chunk2_rc_data1;
    delete[] chunk2_rc_data2;

    delete[] chunk0_real_chunk_nums;
    delete[] chunk1_real_chunk_nums;
    delete[] chunk2_real_chunk_nums;
    delete[] chunk0_real_chunk_ids;
    delete[] chunk1_real_chunk_ids;
    delete[] chunk2_real_chunk_ids;


    time4 += GetTime() - t_1;


    time_tot = GetTime() - t_0;
#ifdef PRINT_CPU_TIMER
    fprintf(
            stderr, "cost time0:%.2f(%.2f %.2f %.2f %.2f) time1:(%.2f[%.2f %.2f] %.2f %.2f) time2:(%.2f[%.2f] %.2f %.2f[%.2f %.2f] %.2f) time3:(%.2f[%.2f %.2f %.2f %.2f] %.2f %.2f %.2f %.2f), time4:%.2f tot time:%.2f\n",
            time0, time0_1, time0_2, time0_3, time0_4,
            time1_1, time1_1_1, time1_1_2, time1_2, time1_3,
            time2_1, time2_1_1, time2_2, time2_3, time2_3_1, time2_3_2, time2_4,
            time3_1, time3_1_1, time3_1_2, time3_1_3, time3_1_4, time3_2, time3_3, time3_4, time3_5,
            time4, time_tot
    );
#endif

    cudaStreamSynchronize(ctx.stream);
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
        const int chunk_num
) {

    if(use_good_numa) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(thread_id, &cpuset);
        pthread_t current_thread = pthread_self();
//        std::cout << "Setting thread affinity to CPU " << thread_id << std::endl;
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
    cudaMallocManaged(&global_randstrobes, batch_read_num * sizeof(my_vector<QueryRandstrobe>));
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
    cudaMallocManaged(&global_hits_per_ref0s, batch_read_num * sizeof(my_vector<my_pair<int, Hit>>));
    meta_data_size += batch_read_num * sizeof(my_vector<my_pair<int, Hit>>);
    cudaMemset(global_hits_per_ref0s, 0, batch_read_num * sizeof(my_vector<my_pair<int, Hit>>));
    my_vector<my_pair<int, Hit>> *global_hits_per_ref1s;
    cudaMallocManaged(&global_hits_per_ref1s, batch_read_num * sizeof(my_vector<my_pair<int, Hit>>));
    meta_data_size += batch_read_num * sizeof(my_vector<my_pair<int, Hit>>);
    cudaMemset(global_hits_per_ref1s, 0, batch_read_num * sizeof(my_vector<my_pair<int, Hit>>));
    my_vector<Nam> *global_nams;
    cudaMallocManaged(&global_nams, batch_read_num * sizeof(my_vector<Nam>));
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
//        chunk0_chunk_num = chunk_num;
        chunk0_real_chunk_num = 0;
        chunk0_datas.clear();
//        printf("chunk0_chunk_num %d\n", chunk0_chunk_num);
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
    int rescue_threshold = 100;
//    printf("rescue_threshold %d\n", rescue_threshold);


    // step: f_1
    {
        bool res;
        // format data
        t_1 = GetTime();
        int rc_pos = 0;
        chunk1_chunk_num = rand() % small_chunk_num + small_chunk_num + 1;
//        chunk1_chunk_num = chunk_num;
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

//        printf("todo cnt is %d, %d, %d\n", *h_todo_cnt, *h_query_offset, *h_ref_offset);

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
//            printf("read %d, type %d, todo size %d %d\n", i, align_tmp_res.type, align_tmp_res.todo_nams.size(), align_tmp_res.todo_infos.size());
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
//        printf("cal_todo_cnt %d, h_todo_cnt %d\n", cal_todo_cnt, *h_todo_cnt);
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
                    if (sam_out.length() > 0) output_buffer.output_records(std::move(sam_out), fx_chunk_id);
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
//                chunk2_chunk_num = chunk_num;
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

//            printf("todo cnt is %d, %d, %d\n", *h_todo_cnt, *h_query_offset, *h_ref_offset);

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
//                printf("read %d, type %d, todo size %d %d\n", i, align_tmp_res.type, align_tmp_res.todo_nams.size(), align_tmp_res.todo_infos.size());
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
//            printf("cal_todo_cnt %d, h_todo_cnt %d\n", cal_todo_cnt, *h_todo_cnt);
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
    std::cout << "gpu cost " << gpu_copy1 << " " << gpu_copy2 << " " << gpu_cost1 << " " << gpu_cost2 << " [" << gpu_cost2_1 << " " << gpu_cost2_2 << "] " << gpu_cost3 << " " << gpu_cost4 << std::endl;
    std::cout << gpu_cost5 << " " << gpu_cost6 << " " << gpu_cost7 << " " << gpu_cost8 << " " << gpu_cost9 << " " << gpu_cost10 << std::endl;
//    std::cout << "[" << gpu_cost10_0 << " " << gpu_cost10_1 << " " << gpu_cost10_2 << " " << gpu_cost10_3 << " " << gpu_cost10_4 << "]" << std::endl;
//    std::cout << "copy data to host cost " << gpu_cost11 << " [" << gpu_cost11_copy1 << ", " << gpu_cost11_copy2 << "]" << std::endl;
    std::cout << "init data " << gpu_init1 << " " << gpu_init2 << " " << gpu_init3 << " " << gpu_init4 << std::endl;
    std::cout << "total cost " << tot_cost << std::endl;
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
