#include "aln.hpp"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <sstream>
#include "aligner.hpp"
#include "nam.hpp"
#include "paf.hpp"
#include "revcomp.hpp"
#include "timer.hpp"

using namespace klibpp;

struct NamPair {
    int score;
    Nam nam1;
    Nam nam2;
};

struct ScoredAlignmentPair {
    double score;
    Alignment alignment1;
    Alignment alignment2;
};

static inline Alignment extend_seed(
    const Aligner& aligner,
    const Nam& nam,
    const References& references,
    const Read& read,
    bool consistent_nam
);


static inline bool extend_seed_part(
    AlignTmpRes& align_tmp_res,
    const Aligner& aligner,
    const Nam& nam,
    const References& references,
    const Read& read,
    bool consistent_nam
); 

template <typename T>
bool by_score(const T& a, const T& b) {
    return a.score > b.score;
}

/*
 * Determine whether the NAM represents a match to the forward or
 * reverse-complemented sequence by checking in which orientation the
 * first and last strobe in the NAM match
 *
 * - If first and last strobe match in forward orientation, return true.
 * - If first and last strobe match in reverse orientation, update the NAM
 *   in place and return true.
 * - If first and last strobe do not match consistently, return false.
 */
bool reverse_nam_if_needed(Nam& nam, const Read& read, const References& references, int k) {
    auto read_len = read.size();
    std::string ref_start_kmer = references.sequences[nam.ref_id].substr(nam.ref_start, k);
    std::string ref_end_kmer = references.sequences[nam.ref_id].substr(nam.ref_end - k, k);

    std::string seq, seq_rc;
    if (nam.is_rc) {
        seq = read.rc;
        seq_rc = read.seq;
    } else {
        seq = read.seq;
        seq_rc = read.rc;
    }
    std::string read_start_kmer = seq.substr(nam.query_start, k);
    std::string read_end_kmer = seq.substr(nam.query_end - k, k);
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

static inline void align_SE_part(
    AlignTmpRes& align_tmp_res,
    const Aligner& aligner,
    std::vector<Nam>& nams,
    const KSeq& record,
    int k,
    const References& references,
    Details& details,
    float dropoff_threshold,
    int max_tries,
    unsigned max_secondary,
    std::minstd_rand& random_engine
) {
    if (nams.empty()) {
        align_tmp_res.type = 0;
        return;
    }

    Read read(record.seq);
    int tries = 0;
    Nam n_max = nams[0];

    align_tmp_res.type = 4;

    for (auto& nam : nams) {
        float score_dropoff = (float) nam.n_hits / n_max.n_hits;
        if (tries >= max_tries || score_dropoff < dropoff_threshold) {
            break;
        }
        bool consistent_nam = reverse_nam_if_needed(nam, read, references, k);
        align_tmp_res.consistent_nam.push_back(consistent_nam);
        align_tmp_res.is_read1.push_back(true);
        bool gapped = extend_seed_part(align_tmp_res, aligner, nam, references, read, consistent_nam);
        tries++;
    }
}

void align_SE_read_last(
    AlignTmpRes& align_tmp_res,
    const KSeq& record,
    Sam& sam,
    std::string& outstring,
    AlignmentStatistics& statistics,
    const Aligner& aligner,
    const MappingParameters& map_param,
    const IndexParameters& index_parameters,
    const References& references,
    const StrobemerIndex& index,
    std::minstd_rand& random_engine
) {
    Timer extend_timer;
    auto dropoff_threshold = map_param.dropoff_threshold;
    auto max_tries = map_param.max_tries;
    auto max_secondary = map_param.max_secondary;
    auto k = index_parameters.syncmer.k;
    Details details;
    if (align_tmp_res.type == 0) {
        sam.add_unmapped(record);
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
        auto alignment = align_tmp_res.align_res[i];
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
    uint8_t mapq = (60.0 * (best_score - second_best_score) + best_score - 1) / best_score;
    bool is_primary = true;
    sam.add(best_alignment, record, read.rc, mapq, is_primary, details);

    if (max_secondary == 0) {
        statistics.tot_extend += extend_timer.duration();
        statistics += details;
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

    // Output secondary alignments
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
    statistics.tot_extend += extend_timer.duration();
    statistics += details;
        
}

static inline void align_SE(
    const Aligner& aligner,
    Sam& sam,
    std::vector<Nam>& nams,
    const KSeq& record,
    int k,
    const References& references,
    Details& details,
    float dropoff_threshold,
    int max_tries,
    unsigned max_secondary,
    std::minstd_rand& random_engine
) {
    if (nams.empty()) {
        sam.add_unmapped(record);
        return;
    }

    Read read(record.seq);
    std::vector<Alignment> alignments;
    int tries = 0;
    Nam n_max = nams[0];

    int best_edit_distance = std::numeric_limits<int>::max();
    int best_score = 0;
    int second_best_score = 0;
    int alignments_with_best_score = 0;
    size_t best_index = 0;

    Alignment best_alignment;
    best_alignment.is_unaligned = true;

    for (auto& nam : nams) {
        float score_dropoff = (float) nam.n_hits / n_max.n_hits;
        if (tries >= max_tries || (tries > 1 && best_edit_distance == 0) ||
            score_dropoff < dropoff_threshold) {
            break;
        }
        bool consistent_nam = reverse_nam_if_needed(nam, read, references, k);
        details.nam_inconsistent += !consistent_nam;
        auto alignment = extend_seed(aligner, nam, references, read, consistent_nam);
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
    uint8_t mapq = (60.0 * (best_score - second_best_score) + best_score - 1) / best_score;
    bool is_primary = true;
    sam.add(best_alignment, record, read.rc, mapq, is_primary, details);

    if (max_secondary == 0) {
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

    // Output secondary alignments
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
}

static inline bool extend_seed_part(
    AlignTmpRes& align_tmp_res,
    const Aligner& aligner,
    const Nam& nam,
    const References& references,
    const Read& read,
    bool consistent_nam
) {
    const std::string query = nam.is_rc ? read.rc : read.seq;
    const std::string& ref = references.sequences[nam.ref_id];

    const auto projected_ref_start = std::max(0, nam.ref_start - nam.query_start);
    const auto projected_ref_end = std::min(nam.ref_end + query.size() - nam.query_end, ref.size());

    AlignmentInfo info;
    int result_ref_start;
    bool gapped = true;
    if (projected_ref_end - projected_ref_start == query.size() && consistent_nam) {
        std::string ref_segm_ham = ref.substr(projected_ref_start, query.size());
        auto hamming_dist = hamming_distance(query, ref_segm_ham);

        if (hamming_dist >= 0 && (((float) hamming_dist / query.size()) < 0.05
                                 )) {  //Hamming distance worked fine, no need to ksw align
            info = hamming_align(
                query, ref_segm_ham, aligner.parameters.match, aligner.parameters.mismatch,
                aligner.parameters.end_bonus
            );
            result_ref_start = projected_ref_start + info.ref_start;
            gapped = false;
        }
    }

    align_tmp_res.todo_nams.push_back(nam);
    align_tmp_res.is_extend_seed.push_back(true);
    if (gapped) {
        // not pass hamming, pending to do align on GPU, tag is false
        Alignment alignment;
        align_tmp_res.done_align.push_back(false);
        align_tmp_res.align_res.push_back(alignment);
    } else {
        // pass hamming, store result, tag is true
        align_tmp_res.done_align.push_back(true);
        int softclipped = info.query_start + (query.size() - info.query_end);
        Alignment alignment;
        alignment.cigar = std::move(info.cigar);
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
    }
    return gapped;
}

/*
 Extend a NAM so that it covers the entire read and return the resulting
 alignment.
*/
static inline Alignment extend_seed(
    const Aligner& aligner,
    const Nam& nam,
    const References& references,
    const Read& read,
    bool consistent_nam
) {
    const std::string query = nam.is_rc ? read.rc : read.seq;
    const std::string& ref = references.sequences[nam.ref_id];

    const auto projected_ref_start = std::max(0, nam.ref_start - nam.query_start);
    const auto projected_ref_end = std::min(nam.ref_end + query.size() - nam.query_end, ref.size());

    AlignmentInfo info;
    int result_ref_start;
    bool gapped = true;
    if (projected_ref_end - projected_ref_start == query.size() && consistent_nam) {
        std::string ref_segm_ham = ref.substr(projected_ref_start, query.size());
        auto hamming_dist = hamming_distance(query, ref_segm_ham);

        if (hamming_dist >= 0 && (((float) hamming_dist / query.size()) < 0.05
                                 )) {  //Hamming distance worked fine, no need to ksw align
            info = hamming_align(
                query, ref_segm_ham, aligner.parameters.match, aligner.parameters.mismatch,
                aligner.parameters.end_bonus
            );
            result_ref_start = projected_ref_start + info.ref_start;
            gapped = false;
        }
    }
    if (gapped) {
        const int diff = std::abs(nam.ref_span() - nam.query_span());
        const int ext_left = std::min(50, projected_ref_start);
        const int ref_start = projected_ref_start - ext_left;
        const int ext_right = std::min(std::size_t(50), ref.size() - nam.ref_end);
        const auto ref_segm_size = read.size() + diff + ext_left + ext_right;
        const auto ref_segm = ref.substr(ref_start, ref_segm_size);
        info = aligner.align(query, ref_segm);
        result_ref_start = ref_start + info.ref_start;
    }
    int softclipped = info.query_start + (query.size() - info.query_end);
    Alignment alignment;
    alignment.cigar = std::move(info.cigar);
    alignment.edit_distance = info.edit_distance;
    alignment.global_ed = info.edit_distance + softclipped;
    alignment.score = info.sw_score;
    alignment.ref_start = result_ref_start;
    alignment.length = info.ref_span();
    alignment.is_rc = nam.is_rc;
    alignment.is_unaligned = false;
    alignment.ref_id = nam.ref_id;
    alignment.gapped = gapped;

    return alignment;
}

static inline uint8_t get_mapq(const std::vector<Nam>& nams, const Nam& n_max) {
    if (nams.size() <= 1) {
        return 60;
    }
    const float s1 = n_max.score;
    const float s2 = nams[1].score;
    // from minimap2: MAPQ = 40(1−s2/s1) ·min{1,|M|/10} · log s1
    const float min_matches = std::min(n_max.n_hits / 10.0, 1.0);
    const int uncapped_mapq = 40 * (1 - s2 / s1) * min_matches * log(s1);
    return std::min(uncapped_mapq, 60);
}

/* Compute paired-end mapping score given best alignments (sorted by score) */
static std::pair<int, int> joint_mapq_from_high_scores(const std::vector<ScoredAlignmentPair>& pairs) {
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

static inline float normal_pdf(float x, float mu, float sigma) {
    static const float inv_sqrt_2pi = 0.3989422804014327;
    const float a = (x - mu) / sigma;

    return inv_sqrt_2pi / sigma * std::exp(-0.5f * a * a);
}

static inline std::vector<ScoredAlignmentPair> get_best_scoring_pairs(
    const std::vector<Alignment>& alignments1,
    const std::vector<Alignment>& alignments2,
    float mu,
    float sigma
) {
    std::vector<ScoredAlignmentPair> pairs;
    for (auto& a1 : alignments1) {
        for (auto& a2 : alignments2) {
            float dist = std::abs(a1.ref_start - a2.ref_start);
            double score = a1.score + a2.score;
            if ((a1.is_rc ^ a2.is_rc) && (dist < mu + 4 * sigma)) {
                score += log(normal_pdf(dist, mu, sigma));
            } else {  // individual score
                // 10 corresponds to a value of log(normal_pdf(dist, mu, sigma)) of more than 4 stddevs away
                score -= 10;
            }
            pairs.push_back(ScoredAlignmentPair{score, a1, a2});
        }
    }

    return pairs;
}

bool is_proper_nam_pair(const Nam nam1, const Nam nam2, float mu, float sigma) {
    if (nam1.ref_id != nam2.ref_id || nam1.is_rc == nam2.is_rc) {
        return false;
    }
    int a = std::max(0, nam1.ref_start - nam1.query_start);
    int b = std::max(0, nam2.ref_start - nam2.query_start);

    // r1 ---> <---- r2
    bool r1_r2 = nam2.is_rc && (a <= b) && (b - a < mu + 10 * sigma);
    if(r1_r2) return 1;

    // r2 ---> <---- r1
    bool r2_r1 = nam1.is_rc && (b <= a) && (a - b < mu + 10 * sigma);
    if(r2_r1) return 1;
    return 0;

//    return r1_r2 || r2_r1;
}

inline uint64_t GetTime() {
    unsigned int lo, hi;
    asm volatile ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}

/*
 * Find high-scoring NAMs and NAM pairs. Proper pairs are preferred, but also
 * high-scoring NAMs that could not be paired up are returned (these get a
 * "dummy" NAM as partner in the returned vector).
 */
//#define timer1
static inline std::vector<NamPair> get_best_scoring_nam_pairs(
    const std::vector<Nam>& nams1,
    const std::vector<Nam>& nams2,
    float mu,
    float sigma
) {
#ifdef timer1
    thread_local int cnt = 0;
    cnt++;
    thread_local long long cnt0 = 0;
    thread_local long long cnt1 = 0;
    thread_local long long cnt2 = 0;
    thread_local long long cnt3 = 0;
    thread_local long long cnt4 = 0;
    thread_local uint64_t time_tot = 0;
    thread_local uint64_t time1 = 0;
    thread_local uint64_t time2 = 0;
    thread_local uint64_t time3 = 0;
    thread_local uint64_t time4 = 0;
    thread_local uint64_t time5 = 0;
    double t_tot = GetTime();
    double t0;
#endif
    std::vector<NamPair> joint_nam_scores;
    if (nams1.empty() && nams2.empty()) {
        return joint_nam_scores;
    }

#ifdef timer1
    t0 = GetTime();
#endif
    // Find NAM pairs that appear to be proper pairs
//    std::vector<bool> added_n1(nams1.size(), false);
//    std::vector<bool> added_n2(nams2.size(), false);
    joint_nam_scores.reserve(nams1.size() + nams2.size());
    robin_hood::unordered_set<int> added_n1;
    robin_hood::unordered_set<int> added_n2;
    int best_joint_hits = 0;
#define use_fast_loop3
//#define use_fast_loop2
//#define use_init_loop
//

#ifdef use_fast_loop3

    std::vector<Nam> nams2_sorted[2];
    for(auto nam2 : nams2) {
        nams2_sorted[nam2.is_rc].push_back(nam2);
    }
    for(int i = 0; i < 2; i++) {
        std::sort(nams2_sorted[i].begin(), nams2_sorted[i].end(), [](const Nam &n1, const Nam &n2) {
                int val1 = std::max(0, n1.ref_start - n1.query_start);
                int val2 = std::max(0, n2.ref_start - n2.query_start);
                return val1 < val2;
                });
    }

    for (auto& nam1 : nams1) {
        int nam1_val = std::max(0, nam1.ref_start - nam1.query_start);
        if(nam1.is_rc == 1) {
            float L_val = nam1_val - (mu + 10 * sigma);
            float R_val = nam1_val;
            int ll = 0, rr = nams2_sorted[0].size() - 1, ans_pos = nams2_sorted[0].size();
            while(ll <= rr) {
                int mid  = (ll + rr) / 2;
                int now_val = std::max(0, nams2_sorted[0][mid].ref_start - nams2_sorted[0][mid].query_start);
                if(now_val > L_val) {
                    rr = mid - 1;
                    ans_pos = mid;
                } else {
                    ll = mid + 1;
                }
            }

			for (int id = ans_pos; id < nams2_sorted[0].size(); id++) {
				Nam nam2 = nams2_sorted[0][id];
                //cnt0++;
                int joint_hits = nam1.n_hits + nam2.n_hits;
                //cnt1++;
                if(nam1.ref_id != nam2.ref_id) continue;
                //cnt2++;
                //if(nam1.is_rc == nam2.is_rc) {
                //    fprintf(stderr, "GGGGG\n");
                //    exit(0);
                //}
                //cnt3++;

                int a = std::max(0, nam1.ref_start - nam1.query_start);
                int b = std::max(0, nam2.ref_start - nam2.query_start);
                if(b > R_val - 1e-6) break;

                // nam1 is rec, nam2 is fwd
                
                //// r1 ---> <---- r2
                //bool r1_r2 = nam2.is_rc && (a <= b) && (b - a < mu + 10 * sigma);

                // r2 ---> <---- r1
                bool r2_r1 = (a - b >= 0) && (a - b < mu + 10 * sigma);

                //if(a - b >= mu + 10 * sigma) {
                //    fprintf(stderr, "GGG2 _ 1\n");
                //    exit(0);
                //}

                //if(a - b < 0) {
                //    fprintf(stderr, "GGG2 _ 2\n");
                //    exit(0);
                //}

                if (r2_r1) {
                    //cnt4++;
                    joint_nam_scores.emplace_back(NamPair{joint_hits, nam1, nam2});
                    //                added_n1[nam1.nam_id] = 1;
                    //                added_n2[nam2.nam_id] = 1;
                    added_n1.insert(nam1.nam_id);
                    added_n2.insert(nam2.nam_id);
                    //best_joint_hits = std::max(joint_hits, best_joint_hits);
                }
            }
        } else{
            float L_val = nam1_val;
            float R_val = nam1_val + mu + 10 * sigma;
            int ll = 0, rr = nams2_sorted[1].size() - 1, ans_pos = nams2_sorted[1].size();
            while(ll <= rr) {
                int mid  = (ll + rr) / 2;
                int now_val = std::max(0, nams2_sorted[1][mid].ref_start - nams2_sorted[1][mid].query_start);
                if(now_val >= L_val) {
                    rr = mid - 1;
                    ans_pos = mid;
                } else {
                    ll = mid + 1;
                }
            }

			for (int id = ans_pos; id < nams2_sorted[1].size(); id++) {
				Nam nam2 = nams2_sorted[1][id];
                //cnt0++;
                int joint_hits = nam1.n_hits + nam2.n_hits;
                //cnt1++;
                if(nam1.ref_id != nam2.ref_id) continue;
                //cnt2++;
                //if(nam1.is_rc == nam2.is_rc) {
                //    fprintf(stderr, "GGGGG\n");
                //    exit(0);
                //}
                //cnt3++;

                int a = std::max(0, nam1.ref_start - nam1.query_start);
                int b = std::max(0, nam2.ref_start - nam2.query_start);
                if(b >= R_val - 1e-6) break;

                // nam1 is fwd, nam2 is rec

                // r1 ---> <---- r2
                bool r1_r2 = (b - a >= 0) && (b - a < mu + 10 * sigma);

                //// r2 ---> <---- r1
                //bool r2_r1 = nam1.is_rc && (b <= a) && (a - b < mu + 10 * sigma);
                //if(b - a >= mu + 10 * sigma) {
                //    fprintf(stderr, "GGG3 _ 1\n");
                //    exit(0);
                //}
                //if(b - a < 0) {
                //    fprintf(stderr, "GGG3 _ 2\n");
                //    exit(0);
                //}
                if (r1_r2) {
                    //cnt4++;
                    joint_nam_scores.emplace_back(NamPair{joint_hits, nam1, nam2});
                    //                added_n1[nam1.nam_id] = 1;
                    //                added_n2[nam2.nam_id] = 1;
                    added_n1.insert(nam1.nam_id);
                    added_n2.insert(nam2.nam_id);
                    //best_joint_hits = std::max(joint_hits, best_joint_hits);
                }
            }
        }
		
    }

#endif


#ifdef use_fast_loop

    std::vector<Nam> nams2_sorted[2];
    for(auto nam2 : nams2) {
        nams2_sorted[nam2.is_rc].push_back(nam2);
    }
    for(int i = 0; i < 2; i++) {
        std::sort(nams2_sorted[i].begin(), nams2_sorted[i].end(), [](const Nam &a, const Nam &b) {
                return a.ref_id < b.ref_id;
                });
    }

    for (auto& nam1 : nams1) {

		auto lower = std::lower_bound(nams2_sorted[!nam1.is_rc].begin(), nams2_sorted[!nam1.is_rc].end(), nam1, 
				[](const Nam& nam2, const Nam& nam1) {
				return nam2.ref_id < nam1.ref_id;
				});

		auto upper = std::upper_bound(nams2_sorted[!nam1.is_rc].begin(), nams2_sorted[!nam1.is_rc].end(), nam1, 
				[](const Nam& nam1, const Nam& nam2) {
				return nam1.ref_id < nam2.ref_id;
				});
		for (auto it = lower; it != upper; ++it) {
            Nam nam2 = *it;
            cnt0++;
            int joint_hits = nam1.n_hits + nam2.n_hits;
            //if (joint_hits < best_joint_hits / 2) {
            //    break;
            //}
            cnt1++;
            assert(nam1.ref_id != nam2.ref_id);
            cnt2++;
            assert(nam1.is_rc == nam2.is_rc);
            cnt3++;

            int a = std::max(0, nam1.ref_start - nam2.query_start);
            int b = std::max(0, nam2.ref_start - nam2.query_start);

            // r1 ---> <---- r2
            bool r1_r2 = nam2.is_rc && (a <= b) && (b - a < mu + 10 * sigma);

            // r2 ---> <---- r1
            bool r2_r1 = nam1.is_rc && (b <= a) && (a - b < mu + 10 * sigma);
  
            if (r1_r2 || r2_r1) {
                cnt4++;
                joint_nam_scores.emplace_back(NamPair{joint_hits, nam1, nam2});
//                added_n1[nam1.nam_id] = 1;
//                added_n2[nam2.nam_id] = 1;
                added_n1.insert(nam1.nam_id);
                added_n2.insert(nam2.nam_id);
                //best_joint_hits = std::max(joint_hits, best_joint_hits);
            }
        }
    }

#endif

#ifdef use_fast_loop2

    std::vector<Nam> nams2_sorted[2];
    for(auto nam2 : nams2) {
        nams2_sorted[nam2.is_rc].push_back(nam2);
    }

    for (auto& nam1 : nams1) {

		for (auto& nam2 : nams2_sorted[!nam1.is_rc]) {
            cnt0++;
            int joint_hits = nam1.n_hits + nam2.n_hits;
            if (joint_hits < best_joint_hits / 2) {
                break;
            }
            cnt1++;
            if(nam1.ref_id != nam2.ref_id) continue;
            //assert(nam1.ref_id != nam2.ref_id);
            cnt2++;
            assert(nam1.is_rc == nam2.is_rc);
            cnt3++;

            int a = std::max(0, nam1.ref_start - nam2.query_start);
            int b = std::max(0, nam2.ref_start - nam2.query_start);

            // r1 ---> <---- r2
            bool r1_r2 = nam2.is_rc && (a <= b) && (b - a < mu + 10 * sigma);

            // r2 ---> <---- r1
            bool r2_r1 = nam1.is_rc && (b <= a) && (a - b < mu + 10 * sigma);
  
            if (r1_r2 || r2_r1) {
                cnt4++;
                joint_nam_scores.emplace_back(NamPair{joint_hits, nam1, nam2});
//                added_n1[nam1.nam_id] = 1;
//                added_n2[nam2.nam_id] = 1;
                added_n1.insert(nam1.nam_id);
                added_n2.insert(nam2.nam_id);
                //best_joint_hits = std::max(joint_hits, best_joint_hits);
            }
        }
    }
#endif

#ifdef use_init_loop

    for (auto& nam1 : nams1) {
        for (auto& nam2 : nams2) {
            cnt0++;
            int joint_hits = nam1.n_hits + nam2.n_hits;
            if (joint_hits < best_joint_hits / 2) {
                break;
            }
            cnt1++;
            if(nam1.ref_id != nam2.ref_id) continue;
            cnt2++;
            if(nam1.is_rc == nam2.is_rc) continue;
            cnt3++;

            //int a = std::max(0, nam1.ref_start - nam2.query_start);
            //int b = std::max(0, nam2.ref_start - nam2.query_start);

            //// r1 ---> <---- r2
            //bool r1_r2 = nam2.is_rc && (a <= b) && (b - a < mu + 10 * sigma);
            //if(r1_r2) return 1;

            //// r2 ---> <---- r1
            //bool r2_r1 = nam1.is_rc && (b <= a) && (a - b < mu + 10 * sigma);
            //if(r2_r1) return 1;
  
            if (is_proper_nam_pair(nam1, nam2, mu, sigma)) {
                cnt4++;
                joint_nam_scores.emplace_back(NamPair{joint_hits, nam1, nam2});
//                added_n1[nam1.nam_id] = 1;
//                added_n2[nam2.nam_id] = 1;
                added_n1.insert(nam1.nam_id);
                added_n2.insert(nam2.nam_id);
                best_joint_hits = std::max(joint_hits, best_joint_hits);
            }
        }
    }
#endif

#ifdef timer1
    time1 += GetTime() - t0;

    t0 = GetTime();
#endif

    // Find high-scoring R1 NAMs that are not part of a proper pair
    Nam dummy_nam;
    dummy_nam.ref_start = -1;
    if (!nams1.empty()) {
        int best_joint_hits1 = best_joint_hits > 0 ? best_joint_hits : nams1[0].n_hits;
        for (auto& nam1 : nams1) {
            if (nam1.n_hits < best_joint_hits1 / 2) {
                break;
            }
            if (added_n1.find(nam1.nam_id) != added_n1.end()) {
//            if (added_n1[nam1.nam_id]) {
                continue;
            }
            //            int n1_penalty = std::abs(nam1.query_span() - nam1.ref_span());
            joint_nam_scores.push_back(NamPair{nam1.n_hits, nam1, dummy_nam});
        }
    }

#ifdef timer1
    time2 += GetTime() - t0;

    t0 = GetTime();
#endif

    // Find high-scoring R2 NAMs that are not part of a proper pair
    if (!nams2.empty()) {
        int best_joint_hits2 = best_joint_hits > 0 ? best_joint_hits : nams2[0].n_hits;
        for (auto& nam2 : nams2) {
            if (nam2.n_hits < best_joint_hits2 / 2) {
                break;
            }
            if (added_n2.find(nam2.nam_id) != added_n2.end()) {
//            if (added_n2[nam2.nam_id]) {
                continue;
            }
            //            int n2_penalty = std::abs(nam2.query_span() - nam2.ref_span());
            joint_nam_scores.push_back(NamPair{nam2.n_hits, dummy_nam, nam2});
        }
    }

#ifdef timer1
    time3 += GetTime() - t0;

    t0 = GetTime();
#endif

    added_n1.clear();
    added_n2.clear();

#ifdef timer1
    time4 += GetTime() - t0;

    t0 = GetTime();
#endif

    std::sort(
        joint_nam_scores.begin(), joint_nam_scores.end(),
        [](const NamPair& a, const NamPair& b) -> bool { return a.score > b.score; }
    );  // Sort by highest score first

#ifdef timer1
    time5 += GetTime() - t0;
    time_tot += GetTime() - t_tot;

    if(cnt % 100000 == 0) {
        std::thread::id mainthreadid = std::this_thread::get_id();
        std::ostringstream ss;
        ss << mainthreadid;
        unsigned long mainthreadidvalue = std::stoul(ss.str());
        fprintf(stderr, "get_best_scoring_nam_pairs [%lu] tot_time:%lf time1:%lf time2:%lf time3:%lf time4:%lf time5:%lf [%lld %lld %lld %lld %lld]\n",
                mainthreadidvalue, time_tot * 1e-8, time1 * 1e-8, time2 * 1e-8, time3 * 1e-8, time4 * 1e-8, time5 * 1e-8, cnt0, cnt1, cnt2, cnt3, cnt4);
    }
#endif
    return joint_nam_scores;
}

/*
 * Determine (roughly) whether the read sequence has some l-mer (with l = k*2/3)
 * in common with the reference sequence
 */
bool has_shared_substring(const std::string& read_seq, const std::string& ref_seq, int k) {
    int sub_size = 2 * k / 3;
    int step_size = k / 3;
    std::string submer;
    for (size_t i = 0; i + sub_size < read_seq.size(); i += step_size) {
        submer = read_seq.substr(i, sub_size);
        if (ref_seq.find(submer) != std::string::npos) {
            return true;
        }
    }
    return false;
}

static inline bool rescue_mate_part(
    AlignTmpRes& align_tmp_res,
    const Aligner& aligner,
    const Nam& nam,
    const References& references,
    const Read& read,
    float mu,
    float sigma,
    int k
) {
    Alignment alignment;
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

    align_tmp_res.todo_nams.push_back(nam);
    align_tmp_res.is_extend_seed.push_back(false);
    if (ref_end < ref_start + k) {
        alignment.cigar = Cigar();
        alignment.edit_distance = read_len;
        alignment.score = 0;
        alignment.ref_start = 0;
        alignment.is_rc = nam.is_rc;
        alignment.ref_id = nam.ref_id;
        alignment.is_unaligned = true;
        align_tmp_res.done_align.push_back(true);
        align_tmp_res.align_res.push_back(alignment);
        return true;
    }
    std::string ref_segm = references.sequences[nam.ref_id].substr(ref_start, ref_end - ref_start);

    if (!has_shared_substring(r_tmp, ref_segm, k)) {
        alignment.cigar = Cigar();
        alignment.edit_distance = read_len;
        alignment.score = 0;
        alignment.ref_start = 0;
        alignment.is_rc = nam.is_rc;
        alignment.ref_id = nam.ref_id;
        alignment.is_unaligned = true;
        align_tmp_res.done_align.push_back(true);
        align_tmp_res.align_res.push_back(alignment);
        return true;
    }

    align_tmp_res.done_align.push_back(false);
    align_tmp_res.align_res.push_back(alignment);
    return false;
}

/* Return true iff rescue by alignment was actually attempted */
static inline Alignment rescue_mate(
    const Aligner& aligner,
    const Nam& nam,
    const References& references,
    const Read& read,
    float mu,
    float sigma,
    int k
) {
    Alignment alignment;
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

    if (ref_end < ref_start + k) {
        alignment.cigar = Cigar();
        alignment.edit_distance = read_len;
        alignment.score = 0;
        alignment.ref_start = 0;
        alignment.is_rc = nam.is_rc;
        alignment.ref_id = nam.ref_id;
        alignment.is_unaligned = true;
        //        std::cerr << "RESCUE: Caught Bug3! ref start: " << ref_start << " ref end: " << ref_end << " ref len:  " << ref_len << std::endl;
        return alignment;
    }
    std::string ref_segm = references.sequences[nam.ref_id].substr(ref_start, ref_end - ref_start);

    if (!has_shared_substring(r_tmp, ref_segm, k)) {
        alignment.cigar = Cigar();
        alignment.edit_distance = read_len;
        alignment.score = 0;
        alignment.ref_start = 0;
        alignment.is_rc = nam.is_rc;
        alignment.ref_id = nam.ref_id;
        alignment.is_unaligned = true;
        //        std::cerr << "Avoided!" << std::endl;
        return alignment;
    }
    auto info = aligner.align(r_tmp, ref_segm);

    alignment.cigar = info.cigar;
    alignment.edit_distance = info.edit_distance;
    alignment.score = info.sw_score;
    alignment.ref_start = ref_start + info.ref_start;
    alignment.is_rc = !nam.is_rc;
    alignment.ref_id = nam.ref_id;
    alignment.is_unaligned = info.cigar.empty();
    alignment.length = info.ref_span();

    return alignment;
}

/*
 * Turn a vector of scored alignment pairs into one that
 * is sorted by score (highest first) and contains only unique entries.
 */
void deduplicate_scored_pairs(std::vector<ScoredAlignmentPair>& pairs) {
    int prev_ref_start1 = pairs[0].alignment1.ref_start;
    int prev_ref_start2 = pairs[0].alignment2.ref_start;
    int prev_ref_id1 = pairs[0].alignment1.ref_id;
    int prev_ref_id2 = pairs[0].alignment2.ref_id;
    size_t j = 1;
    for (size_t i = 1; i < pairs.size(); i++) {
        int ref_start1 = pairs[i].alignment1.ref_start;
        int ref_start2 = pairs[i].alignment2.ref_start;
        int ref_id1 = pairs[i].alignment1.ref_id;
        int ref_id2 = pairs[i].alignment2.ref_id;
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

/*
 * If there are multiple top-scoring alignments (all with the same score),
 * pick one randomly and move it to the front.
 */
void pick_random_top_pair(std::vector<ScoredAlignmentPair>& high_scores, std::minstd_rand& random_engine) {
    size_t i = 1;
    for (; i < high_scores.size(); ++i) {
        if (high_scores[i].score != high_scores[0].score) {
            break;
        }
    }
//    fprintf(stderr, "pick end %zu, ", i);
    if (i > 1) {
        size_t random_index = std::uniform_int_distribution<>(0, i - 1)(random_engine);
//        fprintf(stderr, "pick swap %zu", random_index);
        if (random_index != 0) {
            std::swap(high_scores[0], high_scores[random_index]);
        }
    }
//    fprintf(stderr, "\n");
}

void rescue_read_part(
    int flag,
    AlignTmpRes& align_tmp_res,
    const Read& read2,  // read to be rescued
    const Read& read1,  // read that has NAMs
    const Aligner& aligner,
    const References& references,
    std::vector<Nam>& nams1,
    int max_tries,
    float dropoff,
    std::array<Details, 2>& details,
    int k,
    float mu,
    float sigma,
    size_t max_secondary,
    double secondary_dropoff,
    const klibpp::KSeq& record1,
    const klibpp::KSeq& record2,
    bool swap_r1r2,  // TODO get rid of this
    std::minstd_rand& random_engine
) {
    align_tmp_res.type = flag;
    Nam n_max1 = nams1[0];
    int tries = 0;
    // this loop is safe, loop size is stable
    for (auto& nam : nams1) {
        float score_dropoff1 = (float) nam.n_hits / n_max1.n_hits;
        // only consider top hits (as minimap2 does) and break if below dropoff cutoff.
        if (tries >= max_tries || score_dropoff1 < dropoff) {
            break;
        }

        const bool consistent_nam = reverse_nam_if_needed(nam, read1, references, k);
        details[0].nam_inconsistent += !consistent_nam;
        // reserve extend and store info
        if(flag == 1) align_tmp_res.is_read1.push_back(true);
        else align_tmp_res.is_read1.push_back(false);
        bool gapped = extend_seed_part(align_tmp_res, aligner, nam, references, read1, consistent_nam);
        details[0].gapped += gapped;
        details[0].tried_alignment++;

        // Force SW alignment to rescue mate
        if(flag == 1) align_tmp_res.is_read1.push_back(false);
        else align_tmp_res.is_read1.push_back(true);
        bool is_unaligned = rescue_mate_part(align_tmp_res, aligner, nam, references, read2, mu, sigma, k);
//        details[1].mate_rescue += !is_unaligned;
        tries++;
    }
}

/*
 * Align a pair of reads for which only one has NAMs. For the other, rescue
 * is attempted by aligning it locally.
 */
void rescue_read(
    const Read& read2,  // read to be rescued
    const Read& read1,  // read that has NAMs
    const Aligner& aligner,
    const References& references,
    std::vector<Nam>& nams1,
    int max_tries,
    float dropoff,
    std::array<Details, 2>& details,
    int k,
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
    Nam n_max1 = nams1[0];
    int tries = 0;

    std::vector<Alignment> alignments1;
    std::vector<Alignment> alignments2;
    for (auto& nam : nams1) {
        float score_dropoff1 = (float) nam.n_hits / n_max1.n_hits;
        // only consider top hits (as minimap2 does) and break if below dropoff cutoff.
        if (tries >= max_tries || score_dropoff1 < dropoff) {
            break;
        }

        const bool consistent_nam = reverse_nam_if_needed(nam, read1, references, k);
        details[0].nam_inconsistent += !consistent_nam;
        auto alignment = extend_seed(aligner, nam, references, read1, consistent_nam);
        details[0].gapped += alignment.gapped;
        alignments1.emplace_back(alignment);
        details[0].tried_alignment++;

        // Force SW alignment to rescue mate
        Alignment a2 = rescue_mate(aligner, nam, references, read2, mu, sigma, k);
        details[1].mate_rescue += !a2.is_unaligned;
        alignments2.emplace_back(a2);

        tries++;
    }
    std::sort(alignments1.begin(), alignments1.end(), by_score<Alignment>);
    std::sort(alignments2.begin(), alignments2.end(), by_score<Alignment>);

    // Calculate best combined score here
    auto high_scores = get_best_scoring_pairs(alignments1, alignments2, mu, sigma);

    std::sort(high_scores.begin(), high_scores.end(), by_score<ScoredAlignmentPair>);
    deduplicate_scored_pairs(high_scores);
    pick_random_top_pair(high_scores, random_engine);

    auto [mapq1, mapq2] = joint_mapq_from_high_scores(high_scores);

    // append both alignments to string here
    if (max_secondary == 0) {
        auto best_aln_pair = high_scores[0];
        Alignment alignment1 = best_aln_pair.alignment1;
        Alignment alignment2 = best_aln_pair.alignment2;
        if (swap_r1r2) {
            sam.add_pair(
                alignment2, alignment1, record2, record1, read2.rc, read1.rc, mapq2, mapq1,
                is_proper_pair(alignment2, alignment1, mu, sigma), true, details
            );
        } else {
            sam.add_pair(
                alignment1, alignment2, record1, record2, read1.rc, read2.rc, mapq1, mapq2,
                is_proper_pair(alignment1, alignment2, mu, sigma), true, details
            );
        }
    } else {
        auto max_out = std::min(high_scores.size(), max_secondary);
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
            Alignment alignment1 = aln_pair.alignment1;
            Alignment alignment2 = aln_pair.alignment2;
            if (s_max - s_score < secondary_dropoff) {
                if (swap_r1r2) {
                    bool is_proper = is_proper_pair(alignment2, alignment1, mu, sigma);
                    std::array<Details, 2> swapped_details{details[1], details[0]};
                    sam.add_pair(
                        alignment2, alignment1, record2, record1, read2.rc, read1.rc, mapq2, mapq1, is_proper,
                        is_primary, swapped_details
                    );
                } else {
                    bool is_proper = is_proper_pair(alignment1, alignment2, mu, sigma);
                    sam.add_pair(
                        alignment1, alignment2, record1, record2, read1.rc, read2.rc, mapq1, mapq2, is_proper,
                        is_primary, details
                    );
                }
            } else {
                break;
            }
        }
    }
}

// compute dropoff of the first (top) NAM
float top_dropoff(std::vector<Nam>& nams) {
    auto& n_max = nams[0];
    if (n_max.n_hits <= 2) {
        return 1.0;
    }
    if (nams.size() > 1) {
        return (float) nams[1].n_hits / n_max.n_hits;
    }
    return 0.0;
}

inline void align_PE_part(
    AlignTmpRes& align_tmp_res,
    const Aligner& aligner,
    std::vector<Nam>& nams1,
    std::vector<Nam>& nams2,
    const KSeq& record1,
    const KSeq& record2,
    int k,
    const References& references,
    std::array<Details, 2>& details,
    float dropoff,
    InsertSizeDistribution& isize_est,
    unsigned max_tries,
    size_t max_secondary,
    std::minstd_rand& random_engine
) {
    const auto mu = isize_est.mu;
    const auto sigma = isize_est.sigma;
    Read read1(record1.seq);
    Read read2(record2.seq);
//    fprintf(stderr, "nam1 size %zu, nam2 size %zu\n", nams1.size(), nams2.size());
    double secondary_dropoff = 2 * aligner.parameters.mismatch + aligner.parameters.gap_open;

    if (nams1.empty() && nams2.empty()) {
        // None of the reads have any NAMs
        align_tmp_res.type = 0;
        return;
    }

    if (!nams1.empty() && nams2.empty()) {
        // Only read 1 has NAMS: attempt to rescue read 2
        rescue_read_part(
            1, align_tmp_res, read2, read1, aligner, references, nams1, max_tries, dropoff, details, k, mu,
            sigma, max_secondary, secondary_dropoff, record1, record2, false, random_engine
        );
        return;
    }

    if (nams1.empty() && !nams2.empty()) {
        // Only read 2 has NAMS: attempt to rescue read 1
        rescue_read_part(
            2, align_tmp_res, read1, read2, aligner, references, nams2, max_tries, dropoff, details, k, mu,
            sigma, max_secondary, secondary_dropoff, record2, record1, true, random_engine
        );
        return;
    }

    // If we get here, both reads have NAMs
    assert(!nams1.empty() && !nams2.empty());

    // Deal with the typical case that both reads map uniquely and form a proper pair
    if (top_dropoff(nams1) < dropoff && top_dropoff(nams2) < dropoff &&
        is_proper_nam_pair(nams1[0], nams2[0], mu, sigma)) {
        align_tmp_res.type = 3;
        Nam n_max1 = nams1[0];
        Nam n_max2 = nams2[0];

        bool consistent_nam1 = reverse_nam_if_needed(n_max1, read1, references, k);
        details[0].nam_inconsistent += !consistent_nam1;
        bool consistent_nam2 = reverse_nam_if_needed(n_max2, read2, references, k);
        details[1].nam_inconsistent += !consistent_nam2;

        align_tmp_res.is_read1.push_back(true);
        bool gapped1 = extend_seed_part(align_tmp_res, aligner, n_max1, references, read1, consistent_nam1);
        details[0].tried_alignment++;
        details[0].gapped += gapped1;


        align_tmp_res.is_read1.push_back(false);
        bool gapped2 = extend_seed_part(align_tmp_res, aligner, n_max2, references, read2, consistent_nam2);
        details[1].tried_alignment++;
        details[1].gapped += gapped2;

        int mapq1 = get_mapq(nams1, n_max1);
        int mapq2 = get_mapq(nams2, n_max2);
        align_tmp_res.mapq1 = mapq1;
        align_tmp_res.mapq2 = mapq2;

        if(!gapped1 && !gapped2) {
            int res_size = align_tmp_res.align_res.size();
            if(res_size < 2) {
                fprintf(stderr, "align_tmp_res.align_res.size error %d\n", res_size);
                exit(0);
            }
            auto alignment1 = align_tmp_res.align_res[res_size - 2];
            auto alignment2 = align_tmp_res.align_res[res_size - 1];
            if(alignment1.gapped || alignment2.gapped) {
                fprintf(stderr, "alignment gapped error\n");
                exit(0);
            }
            bool is_proper = is_proper_pair(alignment1, alignment2, mu, sigma);
            if ((isize_est.sample_size < 400) && (alignment1.edit_distance + alignment2.edit_distance < 3) && is_proper) {
                isize_est.update(std::abs(alignment1.ref_start - alignment2.ref_start));
            }
        }

        return;
    }

    // Do a full search for highest-scoring pair
    // Get top hit counts for all locations. The joint hit count is the sum of hits of the two mates. Then align as long as score dropoff or cnt < 20

    align_tmp_res.type = 4;
    std::vector<NamPair> joint_nam_scores = get_best_scoring_nam_pairs(nams1, nams2, mu, sigma);

    // Cache for already computed alignments. Maps NAM ids to alignments.
    robin_hood::unordered_map<int, bool> is_aligned1;
    robin_hood::unordered_map<int, bool> is_aligned2;

    // These keep track of the alignments that would be best if we treated
    // the paired-end read as two single-end reads.
    {
        auto n1_max = nams1[0];
        bool consistent_nam1 = reverse_nam_if_needed(n1_max, read1, references, k);
        details[0].nam_inconsistent += !consistent_nam1;
        align_tmp_res.is_read1.push_back(true);
        bool gapped1 = extend_seed_part(align_tmp_res, aligner, n1_max, references, read1, consistent_nam1);
        is_aligned1[n1_max.nam_id] = 1;
        details[0].tried_alignment++;
        details[0].gapped += gapped1;
//        fprintf(stderr, "cal n1 %d\n", n1_max.nam_id);


        auto n2_max = nams2[0];
        bool consistent_nam2 = reverse_nam_if_needed(n2_max, read2, references, k);
        details[1].nam_inconsistent += !consistent_nam2;
        align_tmp_res.is_read1.push_back(false);
        bool gapped2 = extend_seed_part(align_tmp_res, aligner, n2_max, references, read2, consistent_nam2);
        is_aligned2[n2_max.nam_id] = 1;
        details[1].tried_alignment++;
        details[1].gapped += gapped2;
//        fprintf(stderr, "cal n2 %d\n", n2_max.nam_id);

    }

    // Turn pairs of high-scoring NAMs into pairs of alignments
    std::vector<ScoredAlignmentPair> high_scores;
    auto max_score = joint_nam_scores[0].score;
    align_tmp_res.type4_loop_size = 0;
    for (auto& [score_, n1, n2] : joint_nam_scores) {
        float score_dropoff = (float) score_ / max_score;
        if (high_scores.size() >= max_tries || score_dropoff < dropoff) {
            break;
        }

        align_tmp_res.type4_nams.push_back(n1);
        align_tmp_res.type4_nams.push_back(n2);
        align_tmp_res.type4_loop_size++;
//        fprintf(stderr, "n1 %d, n2 %d\n", n1.nam_id, n2.nam_id);


        // Get alignments for the two NAMs, either by computing the alignment,
        // retrieving it from the cache or by attempting a rescue (if the NAM
        // actually is a dummy, that is, only the partner is available)
        // ref_start == -1 is a marker for a dummy NAM
        if (n1.ref_start >= 0) {
            if (is_aligned1.find(n1.nam_id) != is_aligned1.end()) {
//                fprintf(stderr, "dup nam : %d\n", n1.nam_id);

            } else {
                bool consistent_nam = reverse_nam_if_needed(n1, read1, references, k);
                details[0].nam_inconsistent += !consistent_nam;
                align_tmp_res.is_read1.push_back(true);
                bool gapped = extend_seed_part(align_tmp_res, aligner, n1, references, read1, consistent_nam);
                is_aligned1[n1.nam_id] = 1;
                details[0].tried_alignment++;
                details[0].gapped += gapped;
//                fprintf(stderr, "add extend nam : %d\n", n1.nam_id);
            }
        } else {
            details[1].nam_inconsistent += !reverse_nam_if_needed(n2, read2, references, k);
            align_tmp_res.is_read1.push_back(true);
            bool is_unaligned = rescue_mate_part(align_tmp_res, aligner, n2, references, read1, mu, sigma, k);
//            details[0].mate_rescue += !is_unaligned;
            details[0].tried_alignment++;
//            fprintf(stderr, "add rescue nam : %d\n", n2.nam_id);

        }

        // ref_start == -1 is a marker for a dummy NAM
        if (n2.ref_start >= 0) {
            if (is_aligned2.find(n2.nam_id) != is_aligned2.end()) {
//                fprintf(stderr, "dup nam : %d\n", n2.nam_id);

            } else {
                bool consistent_nam = reverse_nam_if_needed(n2, read2, references, k);
                details[1].nam_inconsistent += !consistent_nam;
                align_tmp_res.is_read1.push_back(false);
                bool gapped = extend_seed_part(align_tmp_res, aligner, n2, references, read2, consistent_nam);
                is_aligned2[n2.nam_id] = 1;
                details[1].tried_alignment++;
                details[1].gapped += gapped;
//                fprintf(stderr, "add extend nam : %d\n", n2.nam_id);

            }
        } else {
            details[0].nam_inconsistent += !reverse_nam_if_needed(n1, read1, references, k);
            align_tmp_res.is_read1.push_back(false);
            bool is_unaligned = rescue_mate_part(align_tmp_res, aligner, n1, references, read2, mu, sigma, k);
//            details[1].mate_rescue += !is_unaligned;
            details[1].tried_alignment++;
//            fprintf(stderr, "add rescue nam : %d\n", n1.nam_id);

        }

        ScoredAlignmentPair aln_pair;
        high_scores.push_back(aln_pair);
    }
}

inline void align_PE(
    const Aligner& aligner,
    Sam& sam,
    std::vector<Nam>& nams1,
    std::vector<Nam>& nams2,
    const KSeq& record1,
    const KSeq& record2,
    int k,
    const References& references,
    std::array<Details, 2>& details,
    float dropoff,
    InsertSizeDistribution& isize_est,
    unsigned max_tries,
    size_t max_secondary,
    std::minstd_rand& random_engine
) {
    const auto mu = isize_est.mu;
    const auto sigma = isize_est.sigma;
    Read read1(record1.seq);
    Read read2(record2.seq);
    double secondary_dropoff = 2 * aligner.parameters.mismatch + aligner.parameters.gap_open;

    if (nams1.empty() && nams2.empty()) {
        // None of the reads have any NAMs
        sam.add_unmapped_pair(record1, record2);
        return;
    }

    if (!nams1.empty() && nams2.empty()) {
        // Only read 1 has NAMS: attempt to rescue read 2
        rescue_read(
            read2, read1, aligner, references, nams1, max_tries, dropoff, details, k, mu, sigma,
            max_secondary, secondary_dropoff, sam, record1, record2, false, random_engine
        );
        return;
    }

    if (nams1.empty() && !nams2.empty()) {
        // Only read 2 has NAMS: attempt to rescue read 1
        rescue_read(
            read1, read2, aligner, references, nams2, max_tries, dropoff, details, k, mu, sigma,
            max_secondary, secondary_dropoff, sam, record2, record1, true, random_engine
        );
        return;
    }

    // If we get here, both reads have NAMs
    assert(!nams1.empty() && !nams2.empty());

    // Deal with the typical case that both reads map uniquely and form a proper pair
    if (top_dropoff(nams1) < dropoff && top_dropoff(nams2) < dropoff &&
        is_proper_nam_pair(nams1[0], nams2[0], mu, sigma)) {
        Nam n_max1 = nams1[0];
        Nam n_max2 = nams2[0];

        bool consistent_nam1 = reverse_nam_if_needed(n_max1, read1, references, k);
        details[0].nam_inconsistent += !consistent_nam1;
        bool consistent_nam2 = reverse_nam_if_needed(n_max2, read2, references, k);
        details[1].nam_inconsistent += !consistent_nam2;

        auto alignment1 = extend_seed(aligner, n_max1, references, read1, consistent_nam1);
        details[0].tried_alignment++;
        details[0].gapped += alignment1.gapped;
        auto alignment2 = extend_seed(aligner, n_max2, references, read2, consistent_nam2);
        details[1].tried_alignment++;
        details[1].gapped += alignment2.gapped;
        int mapq1 = get_mapq(nams1, n_max1);
        int mapq2 = get_mapq(nams2, n_max2);
        bool is_proper = is_proper_pair(alignment1, alignment2, mu, sigma);
        bool is_primary = true;
        sam.add_pair(
            alignment1, alignment2, record1, record2, read1.rc, read2.rc, mapq1, mapq2, is_proper, is_primary,
            details
        );

        if ((isize_est.sample_size < 400) && (alignment1.edit_distance + alignment2.edit_distance < 3) &&
            is_proper) {
            isize_est.update(std::abs(alignment1.ref_start - alignment2.ref_start));
        }
        return;
    }

    // Do a full search for highest-scoring pair
    // Get top hit counts for all locations. The joint hit count is the sum of hits of the two mates. Then align as long as score dropoff or cnt < 20

    std::vector<NamPair> joint_nam_scores = get_best_scoring_nam_pairs(nams1, nams2, mu, sigma);

    // Cache for already computed alignments. Maps NAM ids to alignments.
    robin_hood::unordered_map<int, Alignment> is_aligned1;
    robin_hood::unordered_map<int, Alignment> is_aligned2;

    // These keep track of the alignments that would be best if we treated
    // the paired-end read as two single-end reads.
    Alignment a1_indv_max, a2_indv_max;
    {
        auto n1_max = nams1[0];
        bool consistent_nam1 = reverse_nam_if_needed(n1_max, read1, references, k);
        details[0].nam_inconsistent += !consistent_nam1;
        a1_indv_max = extend_seed(aligner, n1_max, references, read1, consistent_nam1);
        is_aligned1[n1_max.nam_id] = a1_indv_max;
        details[0].tried_alignment++;
        details[0].gapped += a1_indv_max.gapped;

        auto n2_max = nams2[0];
        bool consistent_nam2 = reverse_nam_if_needed(n2_max, read2, references, k);
        details[1].nam_inconsistent += !consistent_nam2;
        a2_indv_max = extend_seed(aligner, n2_max, references, read2, consistent_nam2);
        is_aligned2[n2_max.nam_id] = a2_indv_max;
        details[1].tried_alignment++;
        details[1].gapped += a2_indv_max.gapped;
    }

    // Turn pairs of high-scoring NAMs into pairs of alignments
    std::vector<ScoredAlignmentPair> high_scores;
    auto max_score = joint_nam_scores[0].score;
    for (auto& [score_, n1, n2] : joint_nam_scores) {
        float score_dropoff = (float) score_ / max_score;

        if (high_scores.size() >= max_tries || score_dropoff < dropoff) {
            break;
        }

        // Get alignments for the two NAMs, either by computing the alignment,
        // retrieving it from the cache or by attempting a rescue (if the NAM
        // actually is a dummy, that is, only the partner is available)
        Alignment a1;
        // ref_start == -1 is a marker for a dummy NAM
        if (n1.ref_start >= 0) {
            if (is_aligned1.find(n1.nam_id) != is_aligned1.end()) {
                a1 = is_aligned1[n1.nam_id];
            } else {
                bool consistent_nam = reverse_nam_if_needed(n1, read1, references, k);
                details[0].nam_inconsistent += !consistent_nam;
                a1 = extend_seed(aligner, n1, references, read1, consistent_nam);
                is_aligned1[n1.nam_id] = a1;
                details[0].tried_alignment++;
                details[0].gapped += a1.gapped;
            }
        } else {
            details[1].nam_inconsistent += !reverse_nam_if_needed(n2, read2, references, k);
            a1 = rescue_mate(aligner, n2, references, read1, mu, sigma, k);
            details[0].mate_rescue += !a1.is_unaligned;
            details[0].tried_alignment++;
        }
        if (a1.score > a1_indv_max.score) {
            a1_indv_max = a1;
        }

        Alignment a2;
        // ref_start == -1 is a marker for a dummy NAM
        if (n2.ref_start >= 0) {
            if (is_aligned2.find(n2.nam_id) != is_aligned2.end()) {
                a2 = is_aligned2[n2.nam_id];
            } else {
                bool consistent_nam = reverse_nam_if_needed(n2, read2, references, k);
                details[1].nam_inconsistent += !consistent_nam;
                a2 = extend_seed(aligner, n2, references, read2, consistent_nam);
                is_aligned2[n2.nam_id] = a2;
                details[1].tried_alignment++;
                details[1].gapped += a2.gapped;
            }
        } else {
            details[0].nam_inconsistent += !reverse_nam_if_needed(n1, read1, references, k);
            a2 = rescue_mate(aligner, n1, references, read2, mu, sigma, k);
            details[1].mate_rescue += !a2.is_unaligned;
            details[1].tried_alignment++;
        }
        if (a2.score > a2_indv_max.score) {
            a2_indv_max = a2;
        }

        bool r1_r2 = a2.is_rc && (a1.ref_start <= a2.ref_start) &&
                     ((a2.ref_start - a1.ref_start) < mu + 10 * sigma);  // r1 ---> <---- r2
        bool r2_r1 = a1.is_rc && (a2.ref_start <= a1.ref_start) &&
                     ((a1.ref_start - a2.ref_start) < mu + 10 * sigma);  // r2 ---> <---- r1

        double combined_score;
        if (r1_r2 || r2_r1) {
            // Treat a1/a2 as a pair
            float x = std::abs(a1.ref_start - a2.ref_start);
            combined_score = (double) a1.score + (double) a2.score +
                             std::max(-20.0f + 0.001f, log(normal_pdf(x, mu, sigma)));
            //* (1 - s2 / s1) * min_matches * log(s1);
        } else {
            // Treat a1/a2 as two single-end reads
            // 20 corresponds to a value of log(normal_pdf(x, mu, sigma)) of more than 5 stddevs away (for most reasonable values of stddev)
            combined_score = (double) a1.score + (double) a2.score - 20;
        }

        ScoredAlignmentPair aln_pair{combined_score, a1, a2};
        high_scores.push_back(aln_pair);
    }

    // Finally, add highest scores of both mates as individually mapped
    double combined_score =
        (double) a1_indv_max.score + (double) a2_indv_max.score -
        20;  // 20 corresponds to  a value of log( normal_pdf(x, mu, sigma ) ) of more than 5 stddevs away (for most reasonable values of stddev)
    ScoredAlignmentPair aln_tuple{combined_score, a1_indv_max, a2_indv_max};
    high_scores.push_back(aln_tuple);

    std::sort(high_scores.begin(), high_scores.end(), by_score<ScoredAlignmentPair>);
    deduplicate_scored_pairs(high_scores);
    pick_random_top_pair(high_scores, random_engine);

    auto [mapq1, mapq2] = joint_mapq_from_high_scores(high_scores);
    auto best_aln_pair = high_scores[0];
    auto alignment1 = best_aln_pair.alignment1;
    auto alignment2 = best_aln_pair.alignment2;
    if (max_secondary == 0) {
        bool is_proper = is_proper_pair(alignment1, alignment2, mu, sigma);
        sam.add_pair(
            alignment1, alignment2, record1, record2, read1.rc, read2.rc, mapq1, mapq2, is_proper, true,
            details
        );
    } else {
        auto max_out = std::min(high_scores.size(), max_secondary);
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
                bool is_proper = is_proper_pair(alignment1, alignment2, mu, sigma);
                sam.add_pair(
                    alignment1, alignment2, record1, record2, read1.rc, read2.rc, mapq1, mapq2, is_proper,
                    is_primary, details
                );
            } else {
                break;
            }
        }
    }
}

// Only used for PAF output
inline void get_best_map_location(
    std::vector<Nam>& nams1,
    std::vector<Nam>& nams2,
    InsertSizeDistribution& isize_est,
    Nam& best_nam1,
    Nam& best_nam2
) {
    std::vector<NamPair> joint_nam_scores =
        get_best_scoring_nam_pairs(nams1, nams2, isize_est.mu, isize_est.sigma);
    Nam n1_joint_max, n2_joint_max, n1_indiv_max, n2_indiv_max;
    float score_joint = 0;
    float score_indiv = 0;
    best_nam1.ref_start = -1;  //Unmapped until proven mapped
    best_nam2.ref_start = -1;  //Unmapped until proven mapped

    if (joint_nam_scores.empty()) {
        return;
    }
    // get best joint score
    for (auto& t : joint_nam_scores) {  // already sorted by descending score
        auto& n1 = t.nam1;
        auto& n2 = t.nam2;
        if (n1.ref_start >= 0 && n2.ref_start >= 0) {  // Valid pair
            score_joint = n1.score + n2.score;
            n1_joint_max = n1;
            n2_joint_max = n2;
            break;
        }
    }

    // get individual best scores
    if (!nams1.empty()) {
        auto n1_indiv_max = nams1[0];
        score_indiv +=
            n1_indiv_max.score - (n1_indiv_max.score / 2.0);  //Penalty for being mapped individually
        best_nam1 = n1_indiv_max;
    }
    if (!nams2.empty()) {
        auto n2_indiv_max = nams2[0];
        score_indiv +=
            n2_indiv_max.score - (n2_indiv_max.score / 2.0);  //Penalty for being mapped individually
        best_nam2 = n2_indiv_max;
    }
    if (score_joint > score_indiv) {  // joint score is better than individual
        best_nam1 = n1_joint_max;
        best_nam2 = n2_joint_max;
    }

    if (isize_est.sample_size < 400 && score_joint > score_indiv) {
        isize_est.update(std::abs(n1_joint_max.ref_start - n2_joint_max.ref_start));
    }
}

/* Add a new observation */
void InsertSizeDistribution::update(int dist) {
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
    sigma = std::sqrt(V);
    sample_size = sample_size + 1.0;
    if (mu < 0) {
        std::cerr << "mu negative, mu: " << mu << " sigma: " << sigma << " SSE: " << SSE
                  << " sample size: " << sample_size << std::endl;
    }
    if (SSE < 0) {
        std::cerr << "SSE negative, mu: " << mu << " sigma: " << sigma << " SSE: " << SSE
                  << " sample size: " << sample_size << std::endl;
    }
}

/* Shuffle the top-scoring NAMs. Input must be sorted by score.
 *
 * This helps to ensure we pick a random location in case there are multiple
 * equally good ones.
 */
void shuffle_top_nams(std::vector<Nam>& nams, std::minstd_rand& random_engine) {
    if (nams.empty()) {
        return;
    }
    auto best_score = nams[0].score;
    auto it = std::find_if(nams.begin(), nams.end(), [&](const Nam& nam) { return nam.score != best_score; });
    if (it != nams.end()) {
        std::shuffle(nams.begin(), it, random_engine);
    }
//    fprintf(stderr, "shuffle : ");
//    for(int i = 0; i < nams.size(); i++) {
//        if(nams[i].score != best_score) break;
//        fprintf(stderr, "%d ", nams[i].nam_id);
//    }
//    fprintf(stderr, "\n\n");
}

void align_PE_read_part(
    AlignTmpRes& align_tmp_res,
    const KSeq& record1,
    const KSeq& record2,
    AlignmentStatistics& statistics,
    InsertSizeDistribution& isize_est,
    const Aligner& aligner,
    const MappingParameters& map_param,
    const IndexParameters& index_parameters,
    const References& references,
    const StrobemerIndex& index,
    std::minstd_rand& random_engine
) {
    std::array<Details, 2> details;
    std::array<std::vector<Nam>, 2> nams_pair;

    for (size_t is_revcomp : {0, 1}) {
        const auto& record = is_revcomp == 0 ? record1 : record2;

        Timer strobe_timer;
        auto query_randstrobes = randstrobes_query(record.seq, index_parameters);
        statistics.tot_construct_strobemers += strobe_timer.duration();

        // Find NAMs
        Timer nam_timer;
        auto [nonrepetitive_fraction, nams] = find_nams(query_randstrobes, index);
        statistics.tot_find_nams += nam_timer.duration();

        if (map_param.rescue_level > 1) {
            Timer rescue_timer;
            if (nams.empty() || nonrepetitive_fraction < 0.7) {
                nams = find_nams_rescue(query_randstrobes, index, map_param.rescue_cutoff);
                details[is_revcomp].nam_rescue = true;
            }
            statistics.tot_time_rescue += rescue_timer.duration();
        }
        details[is_revcomp].nams = nams.size();
        Timer nam_sort_timer;
        std::sort(nams.begin(), nams.end(), by_score<Nam>);
        shuffle_top_nams(nams, random_engine);
        statistics.tot_sort_nams += nam_sort_timer.duration();
        nams_pair[is_revcomp] = nams;
    }

    Timer extend_timer;
    // call xx_part func to filter nams and store align info
    align_PE_part(
        align_tmp_res, aligner, nams_pair[0], nams_pair[1], record1, record2, index_parameters.syncmer.k,
        references, details, map_param.dropoff_threshold, isize_est, map_param.max_tries,
        map_param.max_secondary, random_engine
    );
    statistics.tot_extend += extend_timer.duration();
    statistics += details[0];
    statistics += details[1];
}

void rescue_read_last(
    int flag,
    AlignTmpRes& align_tmp_res,
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
    std::vector<Alignment> alignments1;
    std::vector<Alignment> alignments2;
    int res_num = align_tmp_res.todo_nams.size();
    assert(res_num % 2 == 0);
    for (int i = 0; i < res_num; i += 2) {
        alignments1.push_back(align_tmp_res.align_res[i]);
        alignments2.push_back(align_tmp_res.align_res[i + 1]);
        details[1].mate_rescue += !align_tmp_res.align_res[i + 1].is_unaligned;
//        fprintf(stderr, "3 a1 score %d\n", align_tmp_res.align_res[i].score);
//        fprintf(stderr, "3 a2 score %d\n", align_tmp_res.align_res[i + 1].score);
    }
    std::sort(alignments1.begin(), alignments1.end(), by_score<Alignment>);
    std::sort(alignments2.begin(), alignments2.end(), by_score<Alignment>);

    // Calculate best combined score here
    auto high_scores = get_best_scoring_pairs(alignments1, alignments2, mu, sigma);

    std::sort(high_scores.begin(), high_scores.end(), by_score<ScoredAlignmentPair>);
    deduplicate_scored_pairs(high_scores);
//    fprintf(stderr, "rescue read : ");
//    for(auto item : high_scores)
//        fprintf(stderr, "%f ,", item.score);
//    fprintf(stderr, "\n\n");
    pick_random_top_pair(high_scores, random_engine);

    auto [mapq1, mapq2] = joint_mapq_from_high_scores(high_scores);

    // append both alignments to string here
    if (max_secondary == 0) {
        auto best_aln_pair = high_scores[0];
        Alignment alignment1 = best_aln_pair.alignment1;
        Alignment alignment2 = best_aln_pair.alignment2;
//        fprintf(stderr, "good a1 score %d, a2 score %d\n", alignment1.score, alignment2.score);
        if (swap_r1r2) {
            sam.add_pair(
                alignment2, alignment1, record2, record1, read2.rc, read1.rc, mapq2, mapq1,
                is_proper_pair(alignment2, alignment1, mu, sigma), true, details
            );
        } else {
            sam.add_pair(
                alignment1, alignment2, record1, record2, read1.rc, read2.rc, mapq1, mapq2,
                is_proper_pair(alignment1, alignment2, mu, sigma), true, details
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
            Alignment alignment1 = aln_pair.alignment1;
            Alignment alignment2 = aln_pair.alignment2;
            if (s_max - s_score < secondary_dropoff) {
                if (swap_r1r2) {
                    bool is_proper = is_proper_pair(alignment2, alignment1, mu, sigma);
                    std::array<Details, 2> swapped_details{details[1], details[0]};
                    sam.add_pair(
                        alignment2, alignment1, record2, record1, read2.rc, read1.rc, mapq2, mapq1,
                        is_proper, is_primary, swapped_details
                    );
                } else {
                    bool is_proper = is_proper_pair(alignment1, alignment2, mu, sigma);
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


void align_PE_read_last(
    AlignTmpRes& align_tmp_res,
    const KSeq& record1,
    const KSeq& record2,
    Sam& sam,
    std::string& outstring,
    AlignmentStatistics& statistics,
    InsertSizeDistribution& isize_est,
    const Aligner& aligner,
    const MappingParameters& map_param,
    const IndexParameters& index_parameters,
    const References& references,
    const StrobemerIndex& index,
    std::minstd_rand& random_engine
) {
    std::array<Details, 2> details;
    const auto mu = isize_est.mu;
    const auto sigma = isize_est.sigma;
    Read read1(record1.seq);
    Read read2(record2.seq);
    double secondary_dropoff = 2 * aligner.parameters.mismatch + aligner.parameters.gap_open;

    Timer extend_timer;
//    fprintf(stderr, "type %d\n", align_tmp_res.type);
    if (align_tmp_res.type == 0) {
        // None of the reads have any NAMs
        sam.add_unmapped_pair(record1, record2);
    } else if (align_tmp_res.type == 1) {
        rescue_read_last(
            1, align_tmp_res, read2, read1, aligner, references, details, mu,
            sigma, map_param.max_secondary, secondary_dropoff, sam, record1, record2, false, random_engine
        );
    } else if (align_tmp_res.type == 2) {
        rescue_read_last(
            2, align_tmp_res, read1, read2, aligner, references, details, mu,
            sigma, map_param.max_secondary, secondary_dropoff, sam, record2, record1, true, random_engine
        );
    } else if (align_tmp_res.type == 3) {
        assert(align_tmp_res.todo_nams.size() == 2);
        int mapq1 = align_tmp_res.mapq1;
        int mapq2 = align_tmp_res.mapq2;
        Alignment alignment1 = align_tmp_res.align_res[0];
        Alignment alignment2 = align_tmp_res.align_res[1];
        bool is_proper = is_proper_pair(alignment1, alignment2, mu, sigma);
        bool is_primary = true;
        sam.add_pair(
            alignment1, alignment2, record1, record2, read1.rc, read2.rc, mapq1, mapq2, is_proper, is_primary,
            details
        );
        //TODO update
        //if ((isize_est.sample_size < 400) && (alignment1.edit_distance + alignment2.edit_distance < 3) &&
        //    is_proper) {
        //    isize_est.update(std::abs(alignment1.ref_start - alignment2.ref_start));
        //}

    } else if (align_tmp_res.type == 4) {
        int pos = 0;
        robin_hood::unordered_map<int, Alignment> is_aligned1;
        robin_hood::unordered_map<int, Alignment> is_aligned2;
//        fprintf(stderr, "size1 %zu\n", align_tmp_res.todo_nams.size());
//        fprintf(stderr, "size2 %d\n", align_tmp_res.type4_loop_size);

        Alignment a1_indv_max, a2_indv_max;
        {

            auto n1_max = align_tmp_res.todo_nams[pos];
//            fprintf(stderr, "get n1 %d from %d\n", n1_max.nam_id, pos);
            a1_indv_max = align_tmp_res.align_res[pos];
            is_aligned1[n1_max.nam_id] = a1_indv_max;

            pos++;


            auto n2_max = align_tmp_res.todo_nams[pos];
//            fprintf(stderr, "get n2 %d from %d\n", n2_max.nam_id, pos);
            a2_indv_max = align_tmp_res.align_res[pos];
            is_aligned2[n2_max.nam_id] = a2_indv_max;

            pos++;
        }

        std::vector<ScoredAlignmentPair> high_scores;
        assert(align_tmp_res.type4_loop_size * 2 == align_tmp_res.type4_nams.size());

        for(int i = 0; i < align_tmp_res.type4_loop_size; i++) {
            Nam n1 = align_tmp_res.type4_nams[i * 2];
            Nam n2 = align_tmp_res.type4_nams[i * 2 + 1];
//            fprintf(stderr, "n1 %d, n2 %d\n", n1.nam_id, n2.nam_id);

            Alignment a1;
            // ref_start == -1 is a marker for a dummy NAM
            if (n1.ref_start >= 0) {
                if (is_aligned1.find(n1.nam_id) != is_aligned1.end()) {
//                    fprintf(stderr, "find n1 %d\n", n1.nam_id);
                    a1 = is_aligned1[n1.nam_id];
                } else {
//                    fprintf(stderr, "get n1 %d from %d\n", n1.nam_id, pos);
                    a1 = align_tmp_res.align_res[pos];
                    assert(n1.nam_id == align_tmp_res.todo_nams[pos].nam_id);
                    pos++;
                    is_aligned1[n1.nam_id] = a1;
                }
            } else {
//                fprintf(stderr, "get gg n1 %d from %d\n", n1.nam_id, pos);
                a1 = align_tmp_res.align_res[pos];
                assert(n2.nam_id == align_tmp_res.todo_nams[pos].nam_id);
                pos++;
                details[0].mate_rescue += !a1.is_unaligned;
            }
            if (a1.score > a1_indv_max.score) {
                a1_indv_max = a1;
            }

            Alignment a2;
            // ref_start == -1 is a marker for a dummy NAM
            if (n2.ref_start >= 0) {
                if (is_aligned2.find(n2.nam_id) != is_aligned2.end()) {
//                    fprintf(stderr, "find n2 %d\n", n2.nam_id);
                    a2 = is_aligned2[n2.nam_id];
                } else {
//                    fprintf(stderr, "get n2 %d from %d\n", n2.nam_id, pos);
                    a2 = align_tmp_res.align_res[pos];
                    assert(n2.nam_id == align_tmp_res.todo_nams[pos].nam_id);
                    pos++;
                    is_aligned2[n2.nam_id] = a2;
                }
            } else {
//                fprintf(stderr, "get gg n2 %d from %d\n", n2.nam_id, pos);
                a2 = align_tmp_res.align_res[pos];
                assert(n1.nam_id == align_tmp_res.todo_nams[pos].nam_id);
                pos++;
                details[1].mate_rescue += !a2.is_unaligned;
            }
            if (a2.score > a2_indv_max.score) {
                a2_indv_max = a2;
            }

            bool r1_r2 = a2.is_rc && (a1.ref_start <= a2.ref_start) &&
                         ((a2.ref_start - a1.ref_start) < mu + 10 * sigma);  // r1 ---> <---- r2
            bool r2_r1 = a1.is_rc && (a2.ref_start <= a1.ref_start) &&
                         ((a1.ref_start - a2.ref_start) < mu + 10 * sigma);  // r2 ---> <---- r1

            double combined_score;
            if (r1_r2 || r2_r1) {
                // Treat a1/a2 as a pair
                float x = std::abs(a1.ref_start - a2.ref_start);
                combined_score = (double) a1.score + (double) a2.score +
                                 std::max(-20.0f + 0.001f, log(normal_pdf(x, mu, sigma)));
                //* (1 - s2 / s1) * min_matches * log(s1);
            } else {
                // Treat a1/a2 as two single-end reads
                // 20 corresponds to a value of log(normal_pdf(x, mu, sigma)) of more than 5 stddevs away (for most reasonable values of stddev)
                combined_score = (double) a1.score + (double) a2.score - 20;
            }

            ScoredAlignmentPair aln_pair{combined_score, a1, a2};
            high_scores.push_back(aln_pair);

        }
        assert(pos == align_tmp_res.todo_nams.size());
//        fprintf(stderr, "111\n");

        // Finally, add highest scores of both mates as individually mapped
        double combined_score =
            (double) a1_indv_max.score + (double) a2_indv_max.score -
            20;  // 20 corresponds to  a value of log( normal_pdf(x, mu, sigma ) ) of more than 5 stddevs away (for most reasonable values of stddev)
        ScoredAlignmentPair aln_tuple{combined_score, a1_indv_max, a2_indv_max};
        high_scores.push_back(aln_tuple);

        std::sort(high_scores.begin(), high_scores.end(), by_score<ScoredAlignmentPair>);
        deduplicate_scored_pairs(high_scores);
//        fprintf(stderr, "part4 : ");
//        for(auto item : high_scores)
//            fprintf(stderr, "%f ,", item.score);
//        fprintf(stderr, "\n\n");
        pick_random_top_pair(high_scores, random_engine);
//        fprintf(stderr, "111\n");

        auto [mapq1, mapq2] = joint_mapq_from_high_scores(high_scores);
        auto best_aln_pair = high_scores[0];
        auto alignment1 = best_aln_pair.alignment1;
        auto alignment2 = best_aln_pair.alignment2;
        if (map_param.max_secondary == 0) {
            bool is_proper = is_proper_pair(alignment1, alignment2, mu, sigma);
//            fprintf(stderr, "111\n");

            sam.add_pair(
                alignment1, alignment2, record1, record2, read1.rc, read2.rc, mapq1, mapq2, is_proper, true,
                details
            );
//            fprintf(stderr, "111\n");

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
                    bool is_proper = is_proper_pair(alignment1, alignment2, mu, sigma);
                    sam.add_pair(
                        alignment1, alignment2, record1, record2, read1.rc, read2.rc, mapq1, mapq2, is_proper,
                        is_primary, details
                    );
                } else {
                    break;
                }
            }
        }
    }
    statistics.tot_extend += extend_timer.duration();
    statistics += details[0];
    statistics += details[1];
}

void align_PE_read(
    const KSeq& record1,
    const KSeq& record2,
    Sam& sam,
    std::string& outstring,
    AlignmentStatistics& statistics,
    InsertSizeDistribution& isize_est,
    const Aligner& aligner,
    const MappingParameters& map_param,
    const IndexParameters& index_parameters,
    const References& references,
    const StrobemerIndex& index,
    std::minstd_rand& random_engine
) {
    std::array<Details, 2> details;
    std::array<std::vector<Nam>, 2> nams_pair;

    for (size_t is_revcomp : {0, 1}) {
        const auto& record = is_revcomp == 0 ? record1 : record2;

        Timer strobe_timer;
        auto query_randstrobes = randstrobes_query(record.seq, index_parameters);
        statistics.tot_construct_strobemers += strobe_timer.duration();

        // Find NAMs
        Timer nam_timer;
        auto [nonrepetitive_fraction, nams] = find_nams(query_randstrobes, index);
        statistics.tot_find_nams += nam_timer.duration();

        if (map_param.rescue_level > 1) {
            Timer rescue_timer;
            if (nams.empty() || nonrepetitive_fraction < 0.7) {
                nams = find_nams_rescue(query_randstrobes, index, map_param.rescue_cutoff);
                details[is_revcomp].nam_rescue = true;
            }
            statistics.tot_time_rescue += rescue_timer.duration();
        }
        details[is_revcomp].nams = nams.size();
        Timer nam_sort_timer;
        std::sort(nams.begin(), nams.end(), by_score<Nam>);
        shuffle_top_nams(nams, random_engine);
        statistics.tot_sort_nams += nam_sort_timer.duration();
        nams_pair[is_revcomp] = nams;
    }

    Timer extend_timer;
    if (!map_param.is_sam_out) {
        Nam nam_read1;
        Nam nam_read2;
        get_best_map_location(nams_pair[0], nams_pair[1], isize_est, nam_read1, nam_read2);
        output_hits_paf_PE(outstring, nam_read1, record1.name, references, record1.seq.length());
        output_hits_paf_PE(outstring, nam_read2, record2.name, references, record2.seq.length());
    } else {
        align_PE(
            aligner, sam, nams_pair[0], nams_pair[1], record1, record2, index_parameters.syncmer.k,
            references, details, map_param.dropoff_threshold, isize_est, map_param.max_tries,
            map_param.max_secondary, random_engine
        );
    }
    statistics.tot_extend += extend_timer.duration();
    statistics += details[0];
    statistics += details[1];
}

void align_SE_read_part(
    AlignTmpRes& align_tmp_res,
    const KSeq& record,
    AlignmentStatistics& statistics,
    const Aligner& aligner,
    const MappingParameters& map_param,
    const IndexParameters& index_parameters,
    const References& references,
    const StrobemerIndex& index,
    std::minstd_rand& random_engine
) {
    Details details;
    Timer strobe_timer;
    auto query_randstrobes = randstrobes_query(record.seq, index_parameters);
    statistics.tot_construct_strobemers += strobe_timer.duration();

    // Find NAMs
    Timer nam_timer;
    auto [nonrepetitive_fraction, nams] = find_nams(query_randstrobes, index);
    statistics.tot_find_nams += nam_timer.duration();

    if (map_param.rescue_level > 1) {
        Timer rescue_timer;
        if (nams.empty() || nonrepetitive_fraction < 0.7) {
            details.nam_rescue = true;
            nams = find_nams_rescue(query_randstrobes, index, map_param.rescue_cutoff);
        }
        statistics.tot_time_rescue += rescue_timer.duration();
    }
    details.nams = nams.size();

    Timer nam_sort_timer;
    std::sort(nams.begin(), nams.end(), by_score<Nam>);
    shuffle_top_nams(nams, random_engine);
    statistics.tot_sort_nams += nam_sort_timer.duration();

    Timer extend_timer;

    align_SE_part(
        align_tmp_res, aligner, nams, record, index_parameters.syncmer.k, references, details,
        map_param.dropoff_threshold, map_param.max_tries, map_param.max_secondary, random_engine);

    statistics.tot_extend += extend_timer.duration();
    statistics += details;
}


void align_SE_read(
    const KSeq& record,
    Sam& sam,
    std::string& outstring,
    AlignmentStatistics& statistics,
    const Aligner& aligner,
    const MappingParameters& map_param,
    const IndexParameters& index_parameters,
    const References& references,
    const StrobemerIndex& index,
    std::minstd_rand& random_engine
) {
    Details details;
    Timer strobe_timer;
    auto query_randstrobes = randstrobes_query(record.seq, index_parameters);
    statistics.tot_construct_strobemers += strobe_timer.duration();

    // Find NAMs
    Timer nam_timer;
    auto [nonrepetitive_fraction, nams] = find_nams(query_randstrobes, index);
    statistics.tot_find_nams += nam_timer.duration();

    if (map_param.rescue_level > 1) {
        Timer rescue_timer;
        if (nams.empty() || nonrepetitive_fraction < 0.7) {
            details.nam_rescue = true;
            nams = find_nams_rescue(query_randstrobes, index, map_param.rescue_cutoff);
        }
        statistics.tot_time_rescue += rescue_timer.duration();
    }
    details.nams = nams.size();

    Timer nam_sort_timer;
    std::sort(nams.begin(), nams.end(), by_score<Nam>);
    shuffle_top_nams(nams, random_engine);
    statistics.tot_sort_nams += nam_sort_timer.duration();

    Timer extend_timer;
    if (!map_param.is_sam_out) {
        output_hits_paf(outstring, nams, record.name, references, record.seq.length());
    } else {
        align_SE(
            aligner, sam, nams, record, index_parameters.syncmer.k, references, details,
            map_param.dropoff_threshold, map_param.max_tries, map_param.max_secondary, random_engine
        );
    }
    statistics.tot_extend += extend_timer.duration();
    statistics += details;
}
