#include "gpu_pipeline_pe.h"
#include "gpu_seeding.h"
#include "gpu_merging.h"
#include "gpu_alignment.h"
#include <cuda_runtime.h>

__device__ void gpu_rescue_read_part_seg(
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
        bool swap_r1r2,
        const int* sorted_indices
) {
    //align_tmp_res.type = flag;
    Nam n_max1 = nams1[sorted_indices[0]];
    int tries = 0;
    // this loop is safe, loop size is stable
    for (int i = 0; i < nams1.size(); i++) {
        Nam &nam = nams1[sorted_indices[i]];
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
#ifdef GPU_ACC_TAG
    return;
#endif
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
        bool swap_r1r2,
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
            a1_indv_max = std::make_pair(align_tmp_res.align_res[pos], align_tmp_res.cigar_info[pos]);
            is_aligned1[n1_max.nam_id] = a1_indv_max;
            pos++;

            auto n2_max = align_tmp_res.todo_nams[pos];
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
                    a2 = is_aligned2[n2.nam_id];
                } else {
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

#define MAX_NAM_SEARCH 30

__device__ void gpu_get_best_scoring_nam_pairs_optimized(
        my_vector<gpu_NamPair>& joint_nam_scores,
        my_vector<Nam>& nams1,
        my_vector<Nam>& nams2,
        float mu,
        float sigma,
        int max_tries
) {
    int n1_search_len = my_min((int)nams1.size(), MAX_NAM_SEARCH);
    int n2_search_len = my_min((int)nams2.size(), MAX_NAM_SEARCH);

    bool added_n1[MAX_NAM_SEARCH] = {false};
    bool added_n2[MAX_NAM_SEARCH] = {false};

    int best_joint_hits = 0;

    for (int i = 0; i < n1_search_len; i++) {
        const Nam &nam1 = nams1[i];

        if (n2_search_len > 0 && (nam1.n_hits + nams2[0].n_hits) < best_joint_hits * 0.5f) {
            break;
        }

        for (int j = 0; j < n2_search_len; j++) {
            const Nam &nam2 = nams2[j];
            int joint_hits = nam1.n_hits + nam2.n_hits;

            if (joint_hits < best_joint_hits * 0.5f) {
                break;
            }

            if (gpu_is_proper_nam_pair(nam1, nam2, mu, sigma)) {
                if (joint_nam_scores.size() < joint_nam_scores.capacity) {
                    joint_nam_scores.push_back(gpu_NamPair{joint_hits, &nams1, &nams2, i, j});
                    added_n1[i] = true;
                    added_n2[j] = true;
                    if (joint_hits > best_joint_hits) {
                        best_joint_hits = joint_hits;
                    }
                } else {
                    goto end_pair_search;
                }
            }
        }
    }
    end_pair_search:;

    for (int i = 0; i < n1_search_len; i++) {
        if (joint_nam_scores.size() >= joint_nam_scores.capacity) break;

        if (added_n1[i]) {
            continue;
        }

        const Nam &nam1 = nams1[i];
        if (best_joint_hits > 0 && nam1.n_hits < best_joint_hits * 0.5f) {
            break;
        }
        joint_nam_scores.push_back(gpu_NamPair{nam1.n_hits, &nams1, &nams2, i, -1});
    }

    for (int i = 0; i < n2_search_len; i++) {
        if (joint_nam_scores.size() >= joint_nam_scores.capacity) break;

        if (added_n2[i]) {
            continue;
        }

        const Nam &nam2 = nams2[i];
        if (best_joint_hits > 0 && nam2.n_hits < best_joint_hits * 0.5f) {
            break;
        }
        joint_nam_scores.push_back(gpu_NamPair{nam2.n_hits, &nams1, &nams2, -1, i});
    }

    if (!joint_nam_scores.empty()) {
        sort_nam_pairs_by_score(joint_nam_scores, max_tries);
    }
}

__device__ void gpu_get_best_scoring_nam_pairs_seg(
        my_vector<gpu_NamPair>& joint_nam_scores,
        my_vector<Nam>& nams1,
        my_vector<Nam>& nams2,
        float mu,
        float sigma,
        int max_tries,
        const int* sorted_indices1,
        const int* sorted_indices2
) {
    int nams1_len = nams1.size();
    int nams2_len = nams2.size();
    my_vector<bool> added_n1(nams1_len);
    my_vector<bool> added_n2(nams2_len);
    for(int i = 0; i < nams1_len; i++) added_n1.push_back(false);
    for(int i = 0; i < nams2_len; i++) added_n2.push_back(false);

    int best_joint_hits = 0;
    for (int i = 0; i < nams1_len; i++) {
        const Nam &nam1 = nams1[sorted_indices1[i]];
        for (int j = 0; j < nams2_len; j++) {
            const Nam &nam2 = nams2[sorted_indices2[j]];
            int joint_hits = nam1.n_hits + nam2.n_hits;
            //if (joint_hits < 0.5 * best_joint_hits || joint_nam_scores.size() > max_tries * 2) {
            if (joint_hits < best_joint_hits / 2) {
                break;
            }
            if (gpu_is_proper_nam_pair(nam1, nam2, mu, sigma)) {
                joint_nam_scores.push_back(gpu_NamPair{joint_hits, &nams1, &nams2, i, j});
                added_n1[i] = 1;
                added_n2[j] = 1;
                best_joint_hits = my_max(joint_hits, best_joint_hits);
            }
        }
        //if (joint_nam_scores.size() > max_tries * 2) break;
    }

    // Find high-scoring R1 NAMs that are not part of a proper pair
    Nam dummy_nam;
    dummy_nam.ref_start = -1;
    int best_joint_hits1 = best_joint_hits > 0 ? best_joint_hits : nams1[sorted_indices1[0]].n_hits;
    //for(int i = 0; i < my_min(nams1_len, max_tries); i++) {
    for(int i = 0; i < nams1_len; i++) {
        Nam nam1 = nams1[sorted_indices1[i]];
        if (nam1.n_hits < best_joint_hits1 / 2) {
            break;
        }
        if (added_n1[i]) {
            continue;
        }
        joint_nam_scores.push_back(gpu_NamPair{nam1.n_hits, &nams1, &nams2, i, -1});
    }

    // Find high-scoring R2 NAMs that are not part of a proper pair
    int best_joint_hits2 = best_joint_hits > 0 ? best_joint_hits : nams2[sorted_indices2[0]].n_hits;
    //for(int i = 0; i < my_min(nams2_len, max_tries); i++) {
    for(int i = 0; i < nams2_len; i++) {
        Nam nam2 = nams2[sorted_indices2[i]];
        if (nam2.n_hits < best_joint_hits2 / 2) {
            break;
        }
        if (added_n2[i]) {
            continue;
        }
        joint_nam_scores.push_back(gpu_NamPair{nam2.n_hits, &nams1, &nams2, -1, i});
    }

#ifdef GPU_ACC_TAG
    quick_sort_iterative(&(joint_nam_scores[0]), 0, joint_nam_scores.size() - 1,
            [](const gpu_NamPair &n1, const gpu_NamPair &n2) {
                if (n1.score != n2.score) return n1.score > n2.score;
                // Initialize the dummy nam here
                Nam dummy_nam;
                dummy_nam.ref_start = -1;
                // Ensure that nams1 and nams2 are valid before dereferencing
                const my_vector<Nam>& lnams1 = *(n1.nams1);
                const my_vector<Nam>& lnams2 = *(n1.nams2);
                // Safely access Nam1 and Nam2 objects based on valid indices
                const Nam n1_nam1 = (n1.i1 >= 0 && n1.i1 < lnams1.size()) ? lnams1[n1.i1] : dummy_nam;
                const Nam n1_nam2 = (n1.i2 >= 0 && n1.i2 < lnams2.size()) ? lnams2[n1.i2] : dummy_nam;
                const Nam n2_nam1 = (n2.i1 >= 0 && n2.i1 < lnams1.size()) ? lnams1[n2.i1] : dummy_nam;
                const Nam n2_nam2 = (n2.i2 >= 0 && n2.i2 < lnams2.size()) ? lnams2[n2.i2] : dummy_nam;
                if (n1_nam1.query_start != n2_nam1.query_start)
                    return n1_nam1.query_start < n2_nam1.query_start;
                if (n1_nam1.query_end != n2_nam1.query_end)
                    return n1_nam1.query_end < n2_nam1.query_end;
                if (n1_nam1.ref_start != n2_nam1.ref_start)
                    return n1_nam1.ref_start < n2_nam1.ref_start;
                if (n1_nam1.ref_end != n2_nam1.ref_end)
                    return n1_nam1.ref_end < n2_nam1.ref_end;
                if (n1_nam1.n_hits != n2_nam1.n_hits)
                    return n1_nam1.n_hits < n2_nam1.n_hits;
                if (n1_nam1.ref_id != n2_nam1.ref_id)
                    return n1_nam1.ref_id < n2_nam1.ref_id;
                if (n1_nam2.query_start != n2_nam2.query_start)
                    return n1_nam2.query_start < n2_nam2.query_start;
                if (n1_nam2.query_end != n2_nam2.query_end)
                    return n1_nam2.query_end < n2_nam2.query_end;
                if (n1_nam2.ref_start != n2_nam2.ref_start)
                    return n1_nam2.ref_start < n2_nam2.ref_start;
                if (n1_nam2.ref_end != n2_nam2.ref_end)
                    return n1_nam2.ref_end < n2_nam2.ref_end;
                if (n1_nam2.n_hits != n2_nam2.n_hits)
                    return n1_nam2.n_hits < n2_nam2.n_hits;
                if (n1_nam2.ref_id != n2_nam2.ref_id)
                    return n1_nam2.ref_id < n2_nam2.ref_id;
                return false;
            }
    );
#else
    sort_nam_pairs_by_score(joint_nam_scores, max_tries);
//    quick_sort_iterative(&(joint_nam_scores[0]), 0, joint_nam_scores.size() - 1,
//                         [](const gpu_NamPair &n1, const gpu_NamPair &n2) {
//                             return n1.score > n2.score;
//                         }
//    );
#endif

    return;
}


__device__ void gpu_get_best_scoring_nam_pairs(
        my_vector<gpu_NamPair>& joint_nam_scores,
        my_vector<Nam>& nams1,
        my_vector<Nam>& nams2,
        float mu,
        float sigma,
        int max_tries
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
            //if (joint_hits < 0.5 * best_joint_hits || joint_nam_scores.size() > max_tries * 2) {
            if (joint_hits < best_joint_hits / 2) {
                break;
            }
            if (gpu_is_proper_nam_pair(nam1, nam2, mu, sigma)) {
                joint_nam_scores.push_back(gpu_NamPair{joint_hits, &nams1, &nams2, i, j});
                added_n1[i] = 1;
                added_n2[j] = 1;
                best_joint_hits = my_max(joint_hits, best_joint_hits);
            }
        }
        //if (joint_nam_scores.size() > max_tries * 2) break;
    }

    // Find high-scoring R1 NAMs that are not part of a proper pair
    Nam dummy_nam;
    dummy_nam.ref_start = -1;
    int best_joint_hits1 = best_joint_hits > 0 ? best_joint_hits : nams1[0].n_hits;
    //for(int i = 0; i < my_min(nams1_len, max_tries); i++) {
    for(int i = 0; i < nams1_len; i++) {
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
    //for(int i = 0; i < my_min(nams2_len, max_tries); i++) {
    for(int i = 0; i < nams2_len; i++) {
        Nam nam2 = nams2[i];
        if (nam2.n_hits < best_joint_hits2 / 2) {
            break;
        }
        if (added_n2[i]) {
            continue;
        }
        joint_nam_scores.push_back(gpu_NamPair{nam2.n_hits, &nams1, &nams2, -1, i});
    }

#ifdef GPU_ACC_TAG
    quick_sort_iterative(&(joint_nam_scores[0]), 0, joint_nam_scores.size() - 1,
            [](const gpu_NamPair &n1, const gpu_NamPair &n2) {
                if (n1.score != n2.score) return n1.score > n2.score;
                // Initialize the dummy nam here
                Nam dummy_nam;
                dummy_nam.ref_start = -1;
                // Ensure that nams1 and nams2 are valid before dereferencing
                const my_vector<Nam>& lnams1 = *(n1.nams1);
                const my_vector<Nam>& lnams2 = *(n1.nams2);
                // Safely access Nam1 and Nam2 objects based on valid indices
                const Nam n1_nam1 = (n1.i1 >= 0 && n1.i1 < lnams1.size()) ? lnams1[n1.i1] : dummy_nam;
                const Nam n1_nam2 = (n1.i2 >= 0 && n1.i2 < lnams2.size()) ? lnams2[n1.i2] : dummy_nam;
                const Nam n2_nam1 = (n2.i1 >= 0 && n2.i1 < lnams1.size()) ? lnams1[n2.i1] : dummy_nam;
                const Nam n2_nam2 = (n2.i2 >= 0 && n2.i2 < lnams2.size()) ? lnams2[n2.i2] : dummy_nam;
                if (n1_nam1.query_start != n2_nam1.query_start)
                    return n1_nam1.query_start < n2_nam1.query_start;
                if (n1_nam1.query_end != n2_nam1.query_end)
                    return n1_nam1.query_end < n2_nam1.query_end;
                if (n1_nam1.ref_start != n2_nam1.ref_start)
                    return n1_nam1.ref_start < n2_nam1.ref_start;
                if (n1_nam1.ref_end != n2_nam1.ref_end)
                    return n1_nam1.ref_end < n2_nam1.ref_end;
                if (n1_nam1.n_hits != n2_nam1.n_hits)
                    return n1_nam1.n_hits < n2_nam1.n_hits;
                if (n1_nam1.ref_id != n2_nam1.ref_id)
                    return n1_nam1.ref_id < n2_nam1.ref_id;
                if (n1_nam2.query_start != n2_nam2.query_start)
                    return n1_nam2.query_start < n2_nam2.query_start;
                if (n1_nam2.query_end != n2_nam2.query_end)
                    return n1_nam2.query_end < n2_nam2.query_end;
                if (n1_nam2.ref_start != n2_nam2.ref_start)
                    return n1_nam2.ref_start < n2_nam2.ref_start;
                if (n1_nam2.ref_end != n2_nam2.ref_end)
                    return n1_nam2.ref_end < n2_nam2.ref_end;
                if (n1_nam2.n_hits != n2_nam2.n_hits)
                    return n1_nam2.n_hits < n2_nam2.n_hits;
                if (n1_nam2.ref_id != n2_nam2.ref_id)
                    return n1_nam2.ref_id < n2_nam2.ref_id;
                return false;
            }
    );
#else
    sort_nam_pairs_by_score(joint_nam_scores, max_tries);
//    quick_sort_iterative(&(joint_nam_scores[0]), 0, joint_nam_scores.size() - 1,
//                         [](const gpu_NamPair &n1, const gpu_NamPair &n2) {
//                             return n1.score > n2.score;
//                         }
//    );
#endif

    return;
}


__device__ void align_PE_part12_seg(
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
        int* d_query_offset, int* d_ref_offset,
        const int* sorted_indices1,
        const int* sorted_indices2
) {
    //assert(!nams1.empty() && nams2.empty());
    const auto mu = isize_est.mu;
    const auto sigma = isize_est.sigma;
    GPURead read1{seq1, rc1, seq_len1};
    GPURead read2{seq2, rc2, seq_len2};
    double secondary_dropoff = 2 * aligner_parameters.mismatch + aligner_parameters.gap_open;
//    align_tmp_res.type = 1;
    gpu_rescue_read_part_seg(
            type, align_tmp_res, type == 1 ? read2 : read1, type == 1 ? read1 : read2, aligner_parameters, references, type == 1 ? nams1 : nams2, max_tries, dropoff, k, mu,
            sigma, max_secondary, secondary_dropoff, type == 1 ? false : true, type == 1 ? sorted_indices1 : sorted_indices2
    );
    for (size_t j = 0; j < align_tmp_res.todo_nams.size(); j += 2) {
        assert(align_tmp_res.is_extend_seed[j]);
        if (align_tmp_res.type == 1)
            assert(align_tmp_res.is_read1[j]);
        else
            assert(!align_tmp_res.is_read1[j]);
        if (!align_tmp_res.done_align[j]) {
            gpu_part2_extend_seed_get_str(
                    align_tmp_res, j, read1, read2, references,
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
                    align_tmp_res, j + 1, read1, read2, references, mu, sigma,
                    d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset
            );
        }
    }
    return;
}

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
    for (size_t j = 0; j < align_tmp_res.todo_nams.size(); j += 2) {
        assert(align_tmp_res.is_extend_seed[j]);
        if (align_tmp_res.type == 1)
            assert(align_tmp_res.is_read1[j]);
        else
            assert(!align_tmp_res.is_read1[j]);
        if (!align_tmp_res.done_align[j]) {
            gpu_part2_extend_seed_get_str(
                    align_tmp_res, j, read1, read2, references,
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
                    align_tmp_res, j + 1, read1, read2, references, mu, sigma,
                    d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset
            );
        }
    }
    return;
}


__device__ void align_PE_part3_seg(
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
        int* d_query_offset, int* d_ref_offset,
        const int* sorted_indices1,
        const int* sorted_indices2
) {
    assert(!nams1.empty() && !nams2.empty());
    const auto mu = isize_est.mu;
    const auto sigma = isize_est.sigma;
    GPURead read1{seq1, rc1, seq_len1};
    GPURead read2{seq2, rc2, seq_len2};
    double secondary_dropoff = 2 * aligner_parameters.mismatch + aligner_parameters.gap_open;

    // Deal with the typical case that both reads map uniquely and form a proper pair
//    align_tmp_res.type = 3;
    Nam n_max1 = nams1[sorted_indices1[0]];
    Nam n_max2 = nams2[sorted_indices2[0]];

    bool consistent_nam1 = gpu_reverse_nam_if_needed(n_max1, read1, references, k);
    bool consistent_nam2 = gpu_reverse_nam_if_needed(n_max2, read2, references, k);

    align_tmp_res.is_read1.push_back(true);
    bool gapped1 = gpu_extend_seed_part(align_tmp_res, aligner_parameters, n_max1, references, read1, consistent_nam1);


    align_tmp_res.is_read1.push_back(false);
    bool gapped2 = gpu_extend_seed_part(align_tmp_res, aligner_parameters, n_max2, references, read2, consistent_nam2);

    int mapq1 = gpu_get_mapq_seg(nams1, n_max1, sorted_indices1);
    int mapq2 = gpu_get_mapq_seg(nams2, n_max2, sorted_indices2);
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
    if (!align_tmp_res.done_align[0]) {
        gpu_part2_extend_seed_get_str(
                align_tmp_res, 0, read1, read2, references,
                d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset
        );
    }
    assert(align_tmp_res.is_extend_seed[1]);
    assert(!align_tmp_res.is_read1[1]);
    if (!align_tmp_res.done_align[1]) {
        gpu_part2_extend_seed_get_str(
                align_tmp_res, 1, read1, read2, references,
                d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset
        );
    }
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
    if (!align_tmp_res.done_align[0]) {
        gpu_part2_extend_seed_get_str(
                align_tmp_res, 0, read1, read2, references,
                d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset
        );
    }
    assert(align_tmp_res.is_extend_seed[1]);
    assert(!align_tmp_res.is_read1[1]);
    if (!align_tmp_res.done_align[1]) {
        gpu_part2_extend_seed_get_str(
                align_tmp_res, 1, read1, read2, references,
                d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset
        );
    }
}

__device__ void align_PE_part4_seg(
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
        int read_id,
        int* d_todo_cnt,
        char* d_query_ptr, char* d_ref_ptr,
        int* d_query_offset, int* d_ref_offset,
        const int* sorted_indices1,
        const int* sorted_indices2
) {
    assert(!nams1.empty() && !nams2.empty());

    const auto mu = isize_est.mu;
    const auto sigma = isize_est.sigma;
    GPURead read1{seq1, rc1, seq_len1};
    GPURead read2{seq2, rc2, seq_len2};
    double secondary_dropoff = 2 * aligner_parameters.mismatch + aligner_parameters.gap_open;

    Nam dummy_nam;
    dummy_nam.ref_start = -1;

    my_vector<gpu_NamPair> joint_nam_scores(my_max(max_tries, nams1.size() + nams2.size()));
//    gpu_get_best_scoring_nam_pairs_optimized(joint_nam_scores, nams1, nams2, mu, sigma, max_tries);
    gpu_get_best_scoring_nam_pairs_seg(joint_nam_scores, nams1, nams2, mu, sigma, max_tries, sorted_indices1, sorted_indices2);
    //if (joint_nam_scores.size() > max_tries) joint_nam_scores.length = max_tries;

    int nams1_len = nams1.size();
    int nams2_len = nams2.size();
    my_vector<bool> is_aligned1(nams1_len + 1);
    my_vector<bool> is_aligned2(nams2_len + 1);
    for (int i = 0; i <= nams1_len; i++) is_aligned1.push_back(false);
    for (int i = 0; i <= nams2_len; i++) is_aligned2.push_back(false);

    {
        Nam n1_max = nams1[sorted_indices1[0]];
        bool consistent_nam1 = gpu_reverse_nam_if_needed(n1_max, read1, references, k);
        align_tmp_res.is_read1.push_back(true);
        bool gapped1 = gpu_extend_seed_part(align_tmp_res, aligner_parameters, n1_max, references, read1, consistent_nam1);
        is_aligned1[0] = 1;

        Nam n2_max = nams2[sorted_indices2[0]];
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
    for (size_t j = 0; j < align_tmp_res.todo_nams.size(); j++) {
        if (!align_tmp_res.done_align[j]) {
            if (align_tmp_res.is_extend_seed[j]) {
                gpu_part2_extend_seed_get_str(
                        align_tmp_res, j, read1, read2, references,
                        d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset
                );
            } else {
                gpu_part2_rescue_mate_get_str(
                        align_tmp_res, j, read1, read2, references, mu, sigma,
                        d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset
                );
            }
        }
    }
    return;
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

    my_vector<gpu_NamPair> joint_nam_scores(my_max(max_tries, nams1.size() + nams2.size()));
//    gpu_get_best_scoring_nam_pairs_optimized(joint_nam_scores, nams1, nams2, mu, sigma, max_tries);
    gpu_get_best_scoring_nam_pairs(joint_nam_scores, nams1, nams2, mu, sigma, max_tries);
    //if (joint_nam_scores.size() > max_tries) joint_nam_scores.length = max_tries;

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
    for (size_t j = 0; j < align_tmp_res.todo_nams.size(); j++) {
        if (!align_tmp_res.done_align[j]) {
            if (align_tmp_res.is_extend_seed[j]) {
                gpu_part2_extend_seed_get_str(
                        align_tmp_res, j, read1, read2, references,
                        d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset
                );
            } else {
                gpu_part2_rescue_mate_get_str(
                        align_tmp_res, j, read1, read2, references, mu, sigma,
                        d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset
                );
            }
        }
    }
    return;
}

__device__ float gpu_top_dropoff_seg(
        int num_nams,
        const my_vector<Nam>& nams,
        const int* sorted_nam_indices,
        int task_start_offset)
{
    if (num_nams < 2) {
        return 0.0f; // Not enough NAMs to calculate a drop-off.
    }
    int top_nam_idx = sorted_nam_indices[task_start_offset];
    float top_score = nams.data[top_nam_idx].score;

    int second_nam_idx = sorted_nam_indices[task_start_offset + 1];
    float second_score = nams.data[second_nam_idx].score;

    if (top_score > 0) {
        return (top_score - second_score) / top_score;
    }
    return 0.0f;
}


__global__ void gpu_pre_cal_type_seg(
        int num_tasks,
        float dropoff_threshold,
        my_vector<Nam>* global_nams,
        GPUInsertSizeDistribution* isize_est,
        int* global_todo_ids,
        const int* nam_seg_offsets,   // Input: Segment offsets from the sort
        const int* sorted_nam_indices // Input: Sorted indices from the sort
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_tasks) return;

    int task_id1 = id;
    int task_id2 = id + num_tasks;

    const my_vector<Nam>& nams1 = global_nams[task_id1];
    const my_vector<Nam>& nams2 = global_nams[task_id2];

    int task_start1 = nam_seg_offsets[task_id1];
    int task_end1   = nam_seg_offsets[task_id1 + 1];
    int num_nams1   = task_end1 - task_start1;

    int task_start2 = nam_seg_offsets[task_id2];
    int task_end2   = nam_seg_offsets[task_id2 + 1];
    int num_nams2   = task_end2 - task_start2;

    if (num_nams1 == 0 && num_nams2 == 0) {
        global_todo_ids[id] = 0;
    } else if (num_nams1 > 0 && num_nams2 == 0) {
        global_todo_ids[id] = 1;
    } else if (num_nams1 == 0 && num_nams2 > 0) {
        global_todo_ids[id] = 2;
    } else { // Both reads have at least one NAM.
        int best_nam1_idx = sorted_nam_indices[task_start1];
        const Nam& best_nam1 = nams1.data[best_nam1_idx];

        int best_nam2_idx = sorted_nam_indices[task_start2];
        const Nam& best_nam2 = nams2.data[best_nam2_idx];

        float dropoff1 = gpu_top_dropoff_seg(num_nams1, nams1, sorted_nam_indices, task_start1);
        float dropoff2 = gpu_top_dropoff_seg(num_nams2, nams2, sorted_nam_indices, task_start2);

        if (dropoff1 < dropoff_threshold && dropoff2 < dropoff_threshold && gpu_is_proper_nam_pair(best_nam1, best_nam2, isize_est->mu, isize_est->sigma)) {
            global_todo_ids[id] = 3;
        } else {
            global_todo_ids[id] = 4;
        }
    }
}


__global__ void gpu_pre_cal_type(
        int num_tasks,
        float dropoff_threshold,
        my_vector<Nam> *global_nams,
        GPUInsertSizeDistribution& isize_est,
        int *global_todo_ids) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num_tasks) {
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

__global__ void gpu_align_PE01234_seg(
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
        GPUInsertSizeDistribution& isize_est,
        int *global_todo_ids,
        GPUAlignTmpRes *global_align_res,
        int* d_todo_cnt,
        char* d_query_ptr, char* d_ref_ptr,
        int* d_query_offset, int* d_ref_offset,
        const int* sorted_nam_indices,
        const int* nam_seg_offsets
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num_tasks) {
        int real_id = global_todo_ids[id];
        int read_id = real_id + base_read_num;
        int type = global_align_res[real_id].type;
        assert(type <= 4);
        size_t seq_len1, seq_len2;
        seq_len1 = lens[read_id];
        seq_len2 = lens[read_id + read_num * 2];
        char *seq1, *seq2, *rc1, *rc2;
        seq1 = all_seqs + pre_sum[read_id];
        rc1 = all_seqs + pre_sum[read_id + read_num];
        seq2 = all_seqs + pre_sum[read_id + read_num * 2];
        rc2 = all_seqs + pre_sum[read_id + read_num * 3];

        const int* sorted_indices1 = sorted_nam_indices + nam_seg_offsets[real_id];
        assert(global_nams[real_id].size() == nam_seg_offsets[real_id + 1] - nam_seg_offsets[real_id]);
        const int* sorted_indices2 = sorted_nam_indices + nam_seg_offsets[real_id + num_tasks];
        assert(global_nams[real_id + num_tasks].size() == nam_seg_offsets[real_id + num_tasks + 1] - nam_seg_offsets[real_id + num_tasks]);

        if (type == 0) {
            global_nams[real_id].release();
            global_nams[real_id + num_tasks].release();
        } else if (type == 1) {
            GPUAlignTmpRes* align_tmp_res = &global_align_res[real_id];
            align_PE_part12_seg(*align_tmp_res, *aligner_parameters, global_nams[real_id], global_nams[real_id + num_tasks],
                            seq1, rc1, seq_len1, seq2, rc2, seq_len2, index_para->syncmer.k, *global_references,
                            mapping_parameters->dropoff_threshold, isize_est, mapping_parameters->max_tries, mapping_parameters->max_secondary, 1, real_id,
                            d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset, sorted_indices1, sorted_indices2);
            global_nams[real_id].release();
            global_nams[real_id + num_tasks].release();
        } else if (type == 2) {
            GPUAlignTmpRes* align_tmp_res = &global_align_res[real_id];
            align_PE_part12_seg(*align_tmp_res, *aligner_parameters, global_nams[real_id], global_nams[real_id + num_tasks],
                            seq1, rc1, seq_len1, seq2, rc2, seq_len2, index_para->syncmer.k, *global_references,
                            mapping_parameters->dropoff_threshold, isize_est, mapping_parameters->max_tries, mapping_parameters->max_secondary, 2, real_id,
                            d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset, sorted_indices1, sorted_indices2);
            global_nams[real_id].release();
            global_nams[real_id + num_tasks].release();
        } else if (type == 3) {
            GPUAlignTmpRes* align_tmp_res = &global_align_res[real_id];
            align_PE_part3_seg(*align_tmp_res, *aligner_parameters, global_nams[real_id], global_nams[real_id + num_tasks],
                           seq1, rc1, seq_len1, seq2, rc2, seq_len2, index_para->syncmer.k, *global_references,
                           mapping_parameters->dropoff_threshold, isize_est, mapping_parameters->max_tries, mapping_parameters->max_secondary, real_id,
                           d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset, sorted_indices1, sorted_indices2);
            global_nams[real_id].release();
            global_nams[real_id + num_tasks].release();
        } else if (type == 4) {
            GPUAlignTmpRes* align_tmp_res = &global_align_res[real_id];
            align_PE_part4_seg(*align_tmp_res, *aligner_parameters, global_nams[real_id], global_nams[real_id + num_tasks],
                           seq1, rc1, seq_len1, seq2, rc2, seq_len2, index_para->syncmer.k, *global_references,
                           mapping_parameters->dropoff_threshold, isize_est, mapping_parameters->max_tries, mapping_parameters->max_secondary, real_id,
                           d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset, sorted_indices1, sorted_indices2);
            global_nams[real_id].release();
            global_nams[real_id + num_tasks].release();
        } else {
            assert(false);
        }
    }
}


__global__ void gpu_align_PE01234(
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
        GPUInsertSizeDistribution& isize_est,
        int *global_todo_ids,
        GPUAlignTmpRes *global_align_res,
        int* d_todo_cnt,
        char* d_query_ptr, char* d_ref_ptr,
        int* d_query_offset, int* d_ref_offset
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num_tasks) {
        int real_id = global_todo_ids[id];
        int read_id = real_id + base_read_num;
        int type = global_align_res[real_id].type;
        assert(type <= 4);
        size_t seq_len1, seq_len2;
        seq_len1 = lens[read_id];
        seq_len2 = lens[read_id + read_num * 2];
        char *seq1, *seq2, *rc1, *rc2;
        seq1 = all_seqs + pre_sum[read_id];
        rc1 = all_seqs + pre_sum[read_id + read_num];
        seq2 = all_seqs + pre_sum[read_id + read_num * 2];
        rc2 = all_seqs + pre_sum[read_id + read_num * 3];
        if (type == 0) {
            global_nams[real_id].release();
            global_nams[real_id + num_tasks].release();
        } else if (type == 1) {
            GPUAlignTmpRes* align_tmp_res = &global_align_res[real_id];
            align_PE_part12(*align_tmp_res, *aligner_parameters, global_nams[real_id], global_nams[real_id + num_tasks],
                            seq1, rc1, seq_len1, seq2, rc2, seq_len2, index_para->syncmer.k, *global_references,
                            mapping_parameters->dropoff_threshold, isize_est, mapping_parameters->max_tries, mapping_parameters->max_secondary, 1, real_id,
                            d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset);
            global_nams[real_id].release();
            global_nams[real_id + num_tasks].release();
        } else if (type == 2) {
            GPUAlignTmpRes* align_tmp_res = &global_align_res[real_id];
            align_PE_part12(*align_tmp_res, *aligner_parameters, global_nams[real_id], global_nams[real_id + num_tasks],
                            seq1, rc1, seq_len1, seq2, rc2, seq_len2, index_para->syncmer.k, *global_references,
                            mapping_parameters->dropoff_threshold, isize_est, mapping_parameters->max_tries, mapping_parameters->max_secondary, 2, real_id,
                            d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset);
            global_nams[real_id].release();
            global_nams[real_id + num_tasks].release();
        } else if (type == 3) {
            GPUAlignTmpRes* align_tmp_res = &global_align_res[real_id];
            align_PE_part3(*align_tmp_res, *aligner_parameters, global_nams[real_id], global_nams[real_id + num_tasks],
                           seq1, rc1, seq_len1, seq2, rc2, seq_len2, index_para->syncmer.k, *global_references,
                           mapping_parameters->dropoff_threshold, isize_est, mapping_parameters->max_tries, mapping_parameters->max_secondary, real_id,
                           d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset);
            global_nams[real_id].release();
            global_nams[real_id + num_tasks].release();
        } else if (type == 4) {
            GPUAlignTmpRes* align_tmp_res = &global_align_res[real_id];
            align_PE_part4(*align_tmp_res, *aligner_parameters, global_nams[real_id], global_nams[real_id + num_tasks],
                           seq1, rc1, seq_len1, seq2, rc2, seq_len2, index_para->syncmer.k, *global_references,
                           mapping_parameters->dropoff_threshold, isize_est, mapping_parameters->max_tries, mapping_parameters->max_secondary, real_id,
                           d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset);
            global_nams[real_id].release();
            global_nams[real_id + num_tasks].release();
        } else {
            assert(false);
        }
    }
}

__global__ void gpu_align_PE0(
        int num_tasks,
        int s_len,
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
        GPUInsertSizeDistribution& isize_est,
        int *global_todo_ids,
        GPUAlignTmpRes *global_align_res,
        int* d_todo_cnt,
        char* d_query_ptr, char* d_ref_ptr,
        int* d_query_offset, int* d_ref_offset
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num_tasks) {
        int real_id = global_todo_ids[id];
        int type = global_align_res[real_id].type;
        assert(type == 0);
        global_nams[real_id].release();
        global_nams[real_id + s_len].release();
    }
}

__global__ void gpu_align_PE12(
        int num_tasks,
        int s_len,
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
        GPUInsertSizeDistribution& isize_est,
        int *global_todo_ids,
        GPUAlignTmpRes *global_align_res,
        int* d_todo_cnt,
        char* d_query_ptr, char* d_ref_ptr,
        int* d_query_offset, int* d_ref_offset
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num_tasks) {
        int real_id = global_todo_ids[id];
        int read_id = real_id + base_read_num;
        int type = global_align_res[real_id].type;
        assert(type == 1 || type == 2);
        size_t seq_len1, seq_len2;
        seq_len1 = lens[read_id];
        seq_len2 = lens[read_id + read_num * 2];
        char *seq1, *seq2, *rc1, *rc2;
        seq1 = all_seqs + pre_sum[read_id];
        rc1 = all_seqs + pre_sum[read_id + read_num];
        seq2 = all_seqs + pre_sum[read_id + read_num * 2];
        rc2 = all_seqs + pre_sum[read_id + read_num * 3];
        if (type == 1) {
            GPUAlignTmpRes* align_tmp_res = &global_align_res[real_id];
            align_PE_part12(*align_tmp_res, *aligner_parameters, global_nams[real_id], global_nams[real_id + s_len],
                            seq1, rc1, seq_len1, seq2, rc2, seq_len2, index_para->syncmer.k, *global_references,
                            mapping_parameters->dropoff_threshold, isize_est, mapping_parameters->max_tries, mapping_parameters->max_secondary, 1, real_id,
                            d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset);
            global_nams[real_id].release();
            global_nams[real_id + s_len].release();
        } else {
            GPUAlignTmpRes* align_tmp_res = &global_align_res[real_id];
            align_PE_part12(*align_tmp_res, *aligner_parameters, global_nams[real_id], global_nams[real_id + s_len],
                            seq1, rc1, seq_len1, seq2, rc2, seq_len2, index_para->syncmer.k, *global_references,
                            mapping_parameters->dropoff_threshold, isize_est, mapping_parameters->max_tries, mapping_parameters->max_secondary, 2, real_id,
                            d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset);
            global_nams[real_id].release();
            global_nams[real_id + s_len].release();
        }
    }
}


__global__ void gpu_align_PE3(
        int num_tasks,
        int s_len,
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
        GPUInsertSizeDistribution& isize_est,
        int *global_todo_ids,
        GPUAlignTmpRes *global_align_res,
        int* d_todo_cnt,
        char* d_query_ptr, char* d_ref_ptr,
        int* d_query_offset, int* d_ref_offset
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num_tasks) {
        int real_id = global_todo_ids[id];
        int read_id = real_id + base_read_num;
        int type = global_align_res[real_id].type;
        assert(type == 3);
        size_t seq_len1, seq_len2;
        seq_len1 = lens[read_id];
        seq_len2 = lens[read_id + read_num * 2];
        char *seq1, *seq2, *rc1, *rc2;
        seq1 = all_seqs + pre_sum[read_id];
        rc1 = all_seqs + pre_sum[read_id + read_num];
        seq2 = all_seqs + pre_sum[read_id + read_num * 2];
        rc2 = all_seqs + pre_sum[read_id + read_num * 3];

        GPUAlignTmpRes* align_tmp_res = &global_align_res[real_id];
        align_PE_part3(*align_tmp_res, *aligner_parameters, global_nams[real_id], global_nams[real_id + s_len],
                       seq1, rc1, seq_len1, seq2, rc2, seq_len2, index_para->syncmer.k, *global_references,
                       mapping_parameters->dropoff_threshold, isize_est, mapping_parameters->max_tries, mapping_parameters->max_secondary, real_id,
                       d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset);
        global_nams[real_id].release();
        global_nams[real_id + s_len].release();

    }

}


__global__ void gpu_align_PE4(
        int num_tasks,
        int s_len,
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
        GPUInsertSizeDistribution& isize_est,
        int *global_todo_ids,
        GPUAlignTmpRes *global_align_res,
        int* d_todo_cnt,
        char* d_query_ptr, char* d_ref_ptr,
        int* d_query_offset, int* d_ref_offset
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num_tasks) {
        int real_id = global_todo_ids[id];
        int read_id = real_id + base_read_num;
        int type = global_align_res[real_id].type;
        assert(type == 4);
        size_t seq_len1, seq_len2;
        seq_len1 = lens[read_id];
        seq_len2 = lens[read_id + read_num * 2];
        char *seq1, *seq2, *rc1, *rc2;
        seq1 = all_seqs + pre_sum[read_id];
        rc1 = all_seqs + pre_sum[read_id + read_num];
        seq2 = all_seqs + pre_sum[read_id + read_num * 2];
        rc2 = all_seqs + pre_sum[read_id + read_num * 3];

        GPUAlignTmpRes* align_tmp_res = &global_align_res[real_id];
        align_PE_part4(*align_tmp_res, *aligner_parameters, global_nams[real_id], global_nams[real_id + s_len],
                       seq1, rc1, seq_len1, seq2, rc2, seq_len2, index_para->syncmer.k, *global_references,
                       mapping_parameters->dropoff_threshold, isize_est, mapping_parameters->max_tries, mapping_parameters->max_secondary, real_id,
                       d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset);
        global_nams[real_id].release();
        global_nams[real_id + s_len].release();

    }

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
                  SegSortGpuResources& buffers0, SegSortGpuResources& buffers1,
                  int* d_todo_cnt, char* d_query_ptr, char* d_ref_ptr, int* d_query_offset, int* d_ref_offset, const int batch_read_num, const int batch_total_read_len, int rescue_threshold) {

    assert(data1s.size() == data2s.size());
    assert(data1s.size() <= batch_read_num);

    double t0, t1;
    t0 = GetTime();

    // pack read on host
    t1 = GetTime();
    uint64_t tot_len = 0;
    h_pre_sum[0] = 0;
    int total_data_size = data1s.size();
    for (int i = 0; i < total_data_size * 4; i++) {
        int read_id = i % total_data_size;
        if (i < total_data_size) { // read1 seq
            h_len[i] = data1s[read_id].read.lseq;
            h_pre_sum[i + 1] = h_pre_sum[i] + h_len[i];
            char* seq_ptr = (char*)data1s[read_id].read.base + data1s[read_id].read.pseq;
            memcpy(h_seq + h_pre_sum[i], seq_ptr, h_len[i]);
        } else if (i < total_data_size * 2) { // read1 rc
            h_len[i] = data1s[read_id].read.lseq;
            h_pre_sum[i + 1] = h_pre_sum[i] + h_len[i];
            char* seq_ptr = data1s[read_id].rc;
            memcpy(h_seq + h_pre_sum[i], seq_ptr, h_len[i]);
        } else if (i < total_data_size * 3) { // read2 seq
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
    tot_len = h_pre_sum[total_data_size * 4];
    assert(tot_len <= batch_total_read_len * 4);
    gpu_copy1 += GetTime() - t1;

    // transfer read to GPU
    t1 = GetTime();
    cudaMemcpy(d_seq, h_seq, tot_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_len, h_len, total_data_size * sizeof(int) * 4 + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pre_sum, h_pre_sum, total_data_size * sizeof(int) * 4 + 1, cudaMemcpyHostToDevice);
    gpu_copy2 += GetTime() - t1;

    const int small_batch_size = (batch_read_num / SMALL_CHUNK_FAC > 0) ? (batch_read_num / SMALL_CHUNK_FAC) : 1;
    uint64_t global_data_offset = 0;

    for (int l_id = 0; l_id < total_data_size; l_id += small_batch_size) {
        int r_id = l_id + small_batch_size;
        if (r_id > total_data_size) r_id = total_data_size;
        int s_len = r_id - l_id;
        if (s_len <= 0) continue;

        // get randstrobes
        t1 = GetTime();
        int total_tasks = s_len * 2;
        int blocks_per_grid = (total_tasks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        gpu_get_randstrobes<<<blocks_per_grid, THREADS_PER_BLOCK, 0, ctx.stream>>>(total_tasks, total_data_size, l_id, d_pre_sum, d_len, d_seq, d_index_para,
                                                                                   global_randstrobe_sizes, global_hashes_value, global_randstrobes);
        cudaStreamSynchronize(ctx.stream);
        gpu_cost1 += GetTime() - t1;

//        for (int i = 0; i < s_len; i++) {
//            printf("read %d has %d-%d seeds\n", i, global_randstrobes[i].size(), global_randstrobes[i + s_len].size());
//        }

        // query database and get hits
        t1 = GetTime();
        blocks_per_grid = (total_tasks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        gpu_get_hits_pre<<<blocks_per_grid, THREADS_PER_BLOCK, 0, ctx.stream>>>(index.bits, index.filter_cutoff, d_map_param->rescue_cutoff, d_randstrobes, index.randstrobes.size(), d_randstrobe_start_indices,
                                                                                total_tasks, d_index_para, global_hits_num, global_randstrobes,
                                                                                global_hits_per_ref0s, global_hits_per_ref1s);
        cudaStreamSynchronize(ctx.stream);

        blocks_per_grid = (total_tasks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        gpu_get_hits_after<<<blocks_per_grid, THREADS_PER_BLOCK, 0, ctx.stream>>>(index.bits, index.filter_cutoff, d_map_param->rescue_cutoff, d_randstrobes, index.randstrobes.size(), d_randstrobe_start_indices,
                                                                                  total_tasks, d_index_para, global_hits_num, global_randstrobes,
                                                                                  global_hits_per_ref0s, global_hits_per_ref1s);
        cudaStreamSynchronize(ctx.stream);
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

//        for (int i = 0; i < todo_cnt; i++) {
//            printf("normal read %d-[%d] has %d-%d hits\n", i, global_todo_ids[i], global_hits_per_ref0s[global_todo_ids[i]].size(), global_hits_per_ref1s[global_todo_ids[i]].size());
//        }

        // sort hits by ref_id
        t1 = GetTime();
        auto sort_res0 = sort_all_hits_with_cub(todo_cnt, global_hits_per_ref0s, global_todo_ids, ctx.stream, buffers0,
                                                &gpu_cost3_1, &gpu_cost3_2, &gpu_cost3_3, &gpu_cost3_4);
        auto sort_res1 = sort_all_hits_with_cub(todo_cnt, global_hits_per_ref1s, global_todo_ids, ctx.stream, buffers1,
                                                &gpu_cost3_1, &gpu_cost3_2, &gpu_cost3_3, &gpu_cost3_4);
        cudaStreamSynchronize(ctx.stream);
        gpu_cost3 += GetTime() - t1;

        // merge hits to NAMs
        t1 = GetTime();
        blocks_per_grid = (todo_cnt + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        gpu_merge_hits_get_nams_seg<<<blocks_per_grid, THREADS_PER_BLOCK, 0, ctx.stream>>>(todo_cnt, d_index_para, global_nams_info,
                                                                                           global_hits_per_ref0s, global_hits_per_ref1s,
                                                                                           sort_res0.first, sort_res0.second,
                                                                                           sort_res1.first, sort_res1.second,
                                                                                           global_nams, global_todo_ids);
        cudaStreamSynchronize(ctx.stream);
        gpu_cost4 += GetTime() - t1;

//        for (int i = 0; i < todo_cnt; i++) {
//            printf("normal read %d-[%d] has %d nams\n", i, global_todo_ids[i], global_nams[global_todo_ids[i]].size());
//        }

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
        blocks_per_grid = (todo_cnt + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        gpu_rescue_get_hits<<<blocks_per_grid, THREADS_PER_BLOCK, 0, ctx.stream>>>(index.bits, index.filter_cutoff, d_map_param->rescue_cutoff, d_randstrobes, index.randstrobes.size(), d_randstrobe_start_indices,
                                                                                   todo_cnt, d_index_para, global_hits_num, global_randstrobes,
                                                                                   global_hits_per_ref0s, global_hits_per_ref1s, global_todo_ids, rescue_threshold);
        cudaStreamSynchronize(ctx.stream);
        gpu_cost5 += GetTime() - t1;

//        for (int i = 0; i < todo_cnt; i++) {
//            printf("rescue read %d-[%d] has %d-%d hits\n", i, global_todo_ids[i], global_hits_per_ref0s[global_todo_ids[i]].size(), global_hits_per_ref1s[global_todo_ids[i]].size());
//        }

        // rescue mode sort hits by ref_id
        t1 = GetTime();
        auto rescue_sort_res0 = sort_all_hits_with_cub(todo_cnt, global_hits_per_ref0s, global_todo_ids, ctx.stream, buffers0,
                                                       &gpu_cost6_1, &gpu_cost6_2, &gpu_cost6_3, &gpu_cost6_4);
        auto rescue_sort_res1 = sort_all_hits_with_cub(todo_cnt, global_hits_per_ref1s, global_todo_ids, ctx.stream, buffers1,
                                                       &gpu_cost6_1, &gpu_cost6_2, &gpu_cost6_3, &gpu_cost6_4);
        cudaStreamSynchronize(ctx.stream);
        gpu_cost6 += GetTime() - t1;

        t1 = GetTime();
        blocks_per_grid = (todo_cnt + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        gpu_rescue_merge_hits_get_nams_seg<<<blocks_per_grid, THREADS_PER_BLOCK, 0, ctx.stream>>>(todo_cnt, d_index_para, global_nams_info,
                                                                                                  global_hits_per_ref0s, global_hits_per_ref1s,
                                                                                                  rescue_sort_res0.first, rescue_sort_res0.second,
                                                                                                  rescue_sort_res1.first, rescue_sort_res1.second,
                                                                                                  global_nams, global_todo_ids);
        cudaStreamSynchronize(ctx.stream);
        gpu_cost7 += GetTime() - t1;

        // sort nams for all reads
        t1 = GetTime();
        total_tasks = s_len * 2;
        blocks_per_grid = (total_tasks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
//        gpu_sort_nams<<<blocks_per_grid, THREADS_PER_BLOCK, 0, ctx.stream>>>(total_tasks, global_nams, d_map_param, 0);
        sort_nams_by_score_in_place_with_cub(total_tasks, global_nams, global_todo_ids, ctx.stream, buffers0,
                                                    &gpu_cost8_1, &gpu_cost8_2, &gpu_cost8_3, &gpu_cost8_4);
        cudaStreamSynchronize(ctx.stream);
        gpu_cost8 += GetTime() - t1;

//        printf("s_len %d, todo_cnt %d, -- %.2f\n", s_len, todo_cnt, 1.0 * todo_cnt / total_tasks);

        // pre-classification of reads
        t1 = GetTime();
        blocks_per_grid = (s_len + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
//        gpu_pre_cal_type_seg<<<blocks_per_grid, THREADS_PER_BLOCK, 0, ctx.stream>>>(s_len, d_map_param->dropoff_threshold, global_nams, isize_est,
//                                                                                global_todo_ids, nam_sort_res.first, nam_sort_res.second);
        gpu_pre_cal_type<<<blocks_per_grid, THREADS_PER_BLOCK, 0, ctx.stream>>>(s_len, d_map_param->dropoff_threshold, global_nams, isize_est, global_todo_ids);
        cudaStreamSynchronize(ctx.stream);
        gpu_cost9 += GetTime() - t1;

        // allocate memory for align_tmp_res (This CPU part remains the same)
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
        gpu_init3 += GetTime() - t1;

        const int align1234_threads = THREADS_PER_BLOCK;

        // align reads
        t1 = GetTime();

        int r_pos = 0;
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < types[i].size(); j++) {
                global_todo_ids[r_pos++] = types[i][j];
            }
        }
        assert(r_pos == s_len);

        blocks_per_grid = (s_len + align1234_threads - 1) / align1234_threads;
        gpu_align_PE01234<<<blocks_per_grid, align1234_threads, 0, ctx.stream>>>(s_len, total_data_size, l_id, d_index_para, global_align_info, d_aligner, d_pre_sum, d_len, d_seq,
                                                                                 global_references, d_map_param, global_nams, isize_est, global_todo_ids, global_align_res,
                                                                                 d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset);
        cudaStreamSynchronize(ctx.stream);
//        printf("types %d %d %d %d\n", types[0].size(), types[1].size() + types[2].size(), types[3].size(), types[4].size());
//        double t2 = GetTime();
//        int r_pos = 0;
//        for (int i = 0; i < types[0].size(); i++) {
//            global_todo_ids[r_pos++] = types[0][i];
//        }
//        blocks_per_grid = (s_len + align1234_threads - 1) / align1234_threads;
//        gpu_align_PE0<<<blocks_per_grid, align1234_threads, 0, ctx.stream>>>(r_pos, s_len, total_data_size, l_id, d_index_para, global_align_info, d_aligner, d_pre_sum, d_len, d_seq,
//                                                                             global_references, d_map_param, global_nams, isize_est, global_todo_ids, global_align_res,
//                                                                             d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset);
//        cudaStreamSynchronize(ctx.stream);
//        gpu_cost10_1 += GetTime() - t2;
//
//
//        t2 = GetTime();
//        r_pos = 0;
//        for (int i = 0; i < types[1].size(); i++) {
//            global_todo_ids[r_pos++] = types[1][i];
//        }
//        for (int i = 0; i < types[2].size(); i++) {
//            global_todo_ids[r_pos++] = types[2][i];
//        }
//        blocks_per_grid = (s_len + align1234_threads - 1) / align1234_threads;
//        gpu_align_PE12<<<blocks_per_grid, align1234_threads, 0, ctx.stream>>>(r_pos, s_len, total_data_size, l_id, d_index_para, global_align_info, d_aligner, d_pre_sum, d_len, d_seq,
//                                                                              global_references, d_map_param, global_nams, isize_est, global_todo_ids, global_align_res,
//                                                                              d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset);
//        cudaStreamSynchronize(ctx.stream);
//        gpu_cost10_2 += GetTime() - t2;
//
//        t2 = GetTime();
//        r_pos = 0;
//        for (int i = 0; i < types[3].size(); i++) {
//            global_todo_ids[r_pos++] = types[3][i];
//        }
//        blocks_per_grid = (s_len + align1234_threads - 1) / align1234_threads;
//        gpu_align_PE3<<<blocks_per_grid, align1234_threads, 0, ctx.stream>>>(r_pos, s_len, total_data_size, l_id, d_index_para, global_align_info, d_aligner, d_pre_sum, d_len, d_seq,
//                                                                             global_references, d_map_param, global_nams, isize_est, global_todo_ids, global_align_res,
//                                                                             d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset);
//        cudaStreamSynchronize(ctx.stream);
//        gpu_cost10_3 += GetTime() - t2;
//
//        t2 = GetTime();
//        r_pos = 0;
//        for (int i = 0; i < types[4].size(); i++) {
//            global_todo_ids[r_pos++] = types[4][i];
//        }
//        blocks_per_grid = (s_len + align1234_threads - 1) / align1234_threads;
//        gpu_align_PE4<<<blocks_per_grid, align1234_threads, 0, ctx.stream>>>(r_pos, s_len, total_data_size, l_id, d_index_para, global_align_info, d_aligner, d_pre_sum, d_len, d_seq,
//                                                                             global_references, d_map_param, global_nams, isize_est, global_todo_ids, global_align_res,
//                                                                             d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset);
//        cudaStreamSynchronize(ctx.stream);
//        gpu_cost10_4 += GetTime() - t2;

        gpu_cost10 += GetTime() - t1;

//        for (int i = 0; i < s_len; i++) {
//            printf("read %d has %d align res\n", i, global_align_res[i].align_res.size());
//        }
    }

    tot_cost += GetTime() - t0;
}


void GPU_align_PE_init(std::vector<neoRcRef> &data1s, std::vector<neoRcRef> &data2s,
                  ThreadContext& ctx, std::vector<AlignTmpRes> &align_tmp_results,
                  uint64_t* global_hits_num, uint64_t* global_nams_info, uint64_t* global_align_info,
                  const StrobemerIndex& index, AlignmentParameters *d_aligner, MappingParameters* d_map_param, IndexParameters *d_index_para,
                  GPUReferences *global_references, RefRandstrobe *d_randstrobes, my_bucket_index_t *d_randstrobe_start_indices,
                  my_vector<QueryRandstrobe> *global_randstrobes, int *global_todo_ids, int *global_randstrobe_sizes, uint64_t * global_hashes_value,
                  my_vector<my_pair<int, Hit>> *global_hits_per_ref0s, my_vector<my_pair<int, Hit>> *global_hits_per_ref1s, my_vector<Nam> *global_nams, GPUInsertSizeDistribution& isize_est,
                  GPUAlignTmpRes *global_align_res, char *global_align_res_data, uint64_t pre_vec_size,
                  char *d_seq, int *d_len, int *d_pre_sum, char *h_seq, int *h_len, int *h_pre_sum,
                  SegSortGpuResources& buffers0, SegSortGpuResources& buffers1,
                  int* d_todo_cnt, char* d_query_ptr, char* d_ref_ptr, int* d_query_offset, int* d_ref_offset, const int batch_read_num, const int batch_total_read_len, int rescue_threshold) {

    assert(data1s.size() == data2s.size());
    assert(data1s.size() <= batch_read_num);

    double t0, t1;
    t0 = GetTime();

    // pack read on host
    t1 = GetTime();
    uint64_t tot_len = 0;
    h_pre_sum[0] = 0;
    int total_data_size = data1s.size();
    for (int i = 0; i < total_data_size * 4; i++) {
        int read_id = i % total_data_size;
        if (i < total_data_size) { // read1 seq
            h_len[i] = data1s[read_id].read.lseq;
            h_pre_sum[i + 1] = h_pre_sum[i] + h_len[i];
            char* seq_ptr = (char*)data1s[read_id].read.base + data1s[read_id].read.pseq;
            memcpy(h_seq + h_pre_sum[i], seq_ptr, h_len[i]);
        } else if (i < total_data_size * 2) { // read1 rc
            h_len[i] = data1s[read_id].read.lseq;
            h_pre_sum[i + 1] = h_pre_sum[i] + h_len[i];
            char* seq_ptr = data1s[read_id].rc;
            memcpy(h_seq + h_pre_sum[i], seq_ptr, h_len[i]);
        } else if (i < total_data_size * 3) { // read2 seq
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
    tot_len = h_pre_sum[total_data_size * 4];
    assert(tot_len <= batch_total_read_len * 4);
    gpu_copy1 += GetTime() - t1;

    // transfer read to GPU
    t1 = GetTime();
    cudaMemcpy(d_seq, h_seq, tot_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_len, h_len, total_data_size * sizeof(int) * 4 + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pre_sum, h_pre_sum, total_data_size * sizeof(int) * 4 + 1, cudaMemcpyHostToDevice);
    gpu_copy2 += GetTime() - t1;

    const int small_batch_size = (batch_read_num / SMALL_CHUNK_FAC > 0) ? (batch_read_num / SMALL_CHUNK_FAC) : 1;
    uint64_t global_data_offset = 0;

    for (int l_id = 0; l_id < total_data_size; l_id += small_batch_size) {
        int r_id = l_id + small_batch_size;
        if (r_id > total_data_size) r_id = total_data_size;
        int s_len = r_id - l_id;
        if (s_len <= 0) continue;

        // get randstrobes
        t1 = GetTime();
        int total_tasks = s_len * 2;
        int blocks_per_grid = (total_tasks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        gpu_get_randstrobes<<<blocks_per_grid, THREADS_PER_BLOCK, 0, ctx.stream>>>(total_tasks, total_data_size, l_id, d_pre_sum, d_len, d_seq, d_index_para,
                                                                                   global_randstrobe_sizes, global_hashes_value, global_randstrobes);
        cudaStreamSynchronize(ctx.stream);
        gpu_cost1 += GetTime() - t1;

//        for (int i = 0; i < s_len; i++) {
//            printf("read %d has %d-%d seeds\n", i, global_randstrobes[i].size(), global_randstrobes[i + s_len].size());
//        }

        // query database and get hits
        t1 = GetTime();
        blocks_per_grid = (total_tasks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        gpu_get_hits_pre<<<blocks_per_grid, THREADS_PER_BLOCK, 0, ctx.stream>>>(index.bits, index.filter_cutoff, d_map_param->rescue_cutoff, d_randstrobes, index.randstrobes.size(), d_randstrobe_start_indices,
                                                                                total_tasks, d_index_para, global_hits_num, global_randstrobes,
                                                                                global_hits_per_ref0s, global_hits_per_ref1s);
        cudaStreamSynchronize(ctx.stream);

        blocks_per_grid = (total_tasks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        gpu_get_hits_after<<<blocks_per_grid, THREADS_PER_BLOCK, 0, ctx.stream>>>(index.bits, index.filter_cutoff, d_map_param->rescue_cutoff, d_randstrobes, index.randstrobes.size(), d_randstrobe_start_indices,
                                                                                  total_tasks, d_index_para, global_hits_num, global_randstrobes,
                                                                                  global_hits_per_ref0s, global_hits_per_ref1s);
        cudaStreamSynchronize(ctx.stream);
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

//        for (int i = 0; i < todo_cnt; i++) {
//            printf("normal read %d-[%d] has %d-%d hits\n", i, global_todo_ids[i], global_hits_per_ref0s[global_todo_ids[i]].size(), global_hits_per_ref1s[global_todo_ids[i]].size());
//        }

        // sort hits by ref_id
        t1 = GetTime();
        blocks_per_grid = (todo_cnt + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        gpu_sort_hits<<<blocks_per_grid, THREADS_PER_BLOCK, 0, ctx.stream>>>(todo_cnt, global_hits_per_ref0s, global_hits_per_ref1s, global_todo_ids);
        cudaStreamSynchronize(ctx.stream);
        gpu_cost3 += GetTime() - t1;

        // merge hits to NAMs
        t1 = GetTime();
        blocks_per_grid = (todo_cnt + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        gpu_merge_hits_get_nams<<<blocks_per_grid, THREADS_PER_BLOCK, 0, ctx.stream>>>(todo_cnt, d_index_para, global_nams_info,
                                                                                        global_hits_per_ref0s, global_hits_per_ref1s,
                                                                                        global_nams, global_todo_ids);
        cudaStreamSynchronize(ctx.stream);
        gpu_cost4 += GetTime() - t1;

//        for (int i = 0; i < todo_cnt; i++) {
//            printf("normal read %d-[%d] has %d nams\n", i, global_todo_ids[i], global_nams[global_todo_ids[i]].size());
//        }

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
        blocks_per_grid = (todo_cnt + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        gpu_rescue_get_hits<<<blocks_per_grid, THREADS_PER_BLOCK, 0, ctx.stream>>>(index.bits, index.filter_cutoff, d_map_param->rescue_cutoff, d_randstrobes, index.randstrobes.size(), d_randstrobe_start_indices,
                                                                                   todo_cnt, d_index_para, global_hits_num, global_randstrobes,
                                                                                   global_hits_per_ref0s, global_hits_per_ref1s, global_todo_ids, rescue_threshold);
        cudaStreamSynchronize(ctx.stream);
        gpu_cost5 += GetTime() - t1;

//        for (int i = 0; i < todo_cnt; i++) {
//            printf("rescue read %d-[%d] has %d-%d hits\n", i, global_todo_ids[i], global_hits_per_ref0s[global_todo_ids[i]].size(), global_hits_per_ref1s[global_todo_ids[i]].size());
//        }

        // rescue mode sort hits by ref_id
        t1 = GetTime();
        blocks_per_grid = (todo_cnt + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        gpu_rescue_sort_hits<<<blocks_per_grid, THREADS_PER_BLOCK, 0, ctx.stream>>>(todo_cnt, global_hits_per_ref0s, global_hits_per_ref1s, global_todo_ids);
        cudaStreamSynchronize(ctx.stream);
        gpu_cost6 += GetTime() - t1;

        // rescue mode merge hits to NAMs
        t1 = GetTime();
        blocks_per_grid = (todo_cnt + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        gpu_rescue_merge_hits_get_nams<<<blocks_per_grid, THREADS_PER_BLOCK, 0, ctx.stream>>>(todo_cnt, d_index_para, global_nams_info,
                                                                                              global_hits_per_ref0s, global_hits_per_ref1s, global_nams, global_todo_ids);
        cudaStreamSynchronize(ctx.stream);
        gpu_cost7 += GetTime() - t1;

//        for (int i = 0; i < todo_cnt; i++) {
//            printf("rescue read %d-[%d] has %d nams\n", i, global_todo_ids[i], global_nams[global_todo_ids[i]].size());
//        }

        // sort nams for all reads
        t1 = GetTime();
        total_tasks = s_len * 2;
        blocks_per_grid = (total_tasks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        gpu_sort_nams<<<blocks_per_grid, THREADS_PER_BLOCK, 0, ctx.stream>>>(total_tasks, global_nams, d_map_param, 0);
        cudaStreamSynchronize(ctx.stream);
        gpu_cost8 += GetTime() - t1;

//        printf("s_len %d, todo_cnt %d, -- %.2f\n", s_len, todo_cnt, 1.0 * todo_cnt / total_tasks);

        // pre-classification of reads
        t1 = GetTime();
        blocks_per_grid = (s_len + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        gpu_pre_cal_type<<<blocks_per_grid, THREADS_PER_BLOCK, 0, ctx.stream>>>(s_len, d_map_param->dropoff_threshold, global_nams, isize_est, global_todo_ids);
        cudaStreamSynchronize(ctx.stream);
        gpu_cost9 += GetTime() - t1;

        // allocate memory for align_tmp_res (This CPU part remains the same)
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
        gpu_init3 += GetTime() - t1;

        const int align1234_threads = THREADS_PER_BLOCK;

        // align reads
        t1 = GetTime();

//        int r_pos = 0;
//        for (int i = 0; i < 5; i++) {
//            for (int j = 0; j < types[i].size(); j++) {
//                global_todo_ids[r_pos++] = types[i][j];
//            }
//        }
//        assert(r_pos == s_len);
//
//        blocks_per_grid = (s_len + align1234_threads - 1) / align1234_threads;
//        gpu_align_PE01234<<<blocks_per_grid, align1234_threads, 0, ctx.stream>>>(s_len, total_data_size, l_id, d_index_para, global_align_info, d_aligner, d_pre_sum, d_len, d_seq,
//                                                                                 global_references, d_map_param, global_nams, isize_est, global_todo_ids, global_align_res,
//                                                                                 d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset);
//        cudaStreamSynchronize(ctx.stream);
//        printf("types %d %d %d %d\n", types[0].size(), types[1].size() + types[2].size(), types[3].size(), types[4].size());
        double t2 = GetTime();
        int r_pos = 0;
        for (int i = 0; i < types[0].size(); i++) {
            global_todo_ids[r_pos++] = types[0][i];
        }
        blocks_per_grid = (s_len + align1234_threads - 1) / align1234_threads;
        gpu_align_PE0<<<blocks_per_grid, align1234_threads, 0, ctx.stream>>>(r_pos, s_len, total_data_size, l_id, d_index_para, global_align_info, d_aligner, d_pre_sum, d_len, d_seq,
                                                                                 global_references, d_map_param, global_nams, isize_est, global_todo_ids, global_align_res,
                                                                                 d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset);
        cudaStreamSynchronize(ctx.stream);
        gpu_cost10_1 += GetTime() - t2;


        t2 = GetTime();
        r_pos = 0;
        for (int i = 0; i < types[1].size(); i++) {
            global_todo_ids[r_pos++] = types[1][i];
        }
        for (int i = 0; i < types[2].size(); i++) {
            global_todo_ids[r_pos++] = types[2][i];
        }
        blocks_per_grid = (s_len + align1234_threads - 1) / align1234_threads;
        gpu_align_PE12<<<blocks_per_grid, align1234_threads, 0, ctx.stream>>>(r_pos, s_len, total_data_size, l_id, d_index_para, global_align_info, d_aligner, d_pre_sum, d_len, d_seq,
                                                                             global_references, d_map_param, global_nams, isize_est, global_todo_ids, global_align_res,
                                                                             d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset);
        cudaStreamSynchronize(ctx.stream);
        gpu_cost10_2 += GetTime() - t2;

        t2 = GetTime();
        r_pos = 0;
        for (int i = 0; i < types[3].size(); i++) {
            global_todo_ids[r_pos++] = types[3][i];
        }
        blocks_per_grid = (s_len + align1234_threads - 1) / align1234_threads;
        gpu_align_PE3<<<blocks_per_grid, align1234_threads, 0, ctx.stream>>>(r_pos, s_len, total_data_size, l_id, d_index_para, global_align_info, d_aligner, d_pre_sum, d_len, d_seq,
                                                                             global_references, d_map_param, global_nams, isize_est, global_todo_ids, global_align_res,
                                                                             d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset);
        cudaStreamSynchronize(ctx.stream);
        gpu_cost10_3 += GetTime() - t2;

        t2 = GetTime();
        r_pos = 0;
        for (int i = 0; i < types[4].size(); i++) {
            global_todo_ids[r_pos++] = types[4][i];
        }
        blocks_per_grid = (s_len + align1234_threads - 1) / align1234_threads;
        gpu_align_PE4<<<blocks_per_grid, align1234_threads, 0, ctx.stream>>>(r_pos, s_len, total_data_size, l_id, d_index_para, global_align_info, d_aligner, d_pre_sum, d_len, d_seq,
                                                                             global_references, d_map_param, global_nams, isize_est, global_todo_ids, global_align_res,
                                                                             d_todo_cnt, d_query_ptr, d_ref_ptr, d_query_offset, d_ref_offset);
        cudaStreamSynchronize(ctx.stream);
        gpu_cost10_4 += GetTime() - t2;

        gpu_cost10 += GetTime() - t1;

//        for (int i = 0; i < s_len; i++) {
//            printf("read %d has %d align res\n", i, global_align_res[i].align_res.size());
//        }
    }

    tot_cost += GetTime() - t0;
}

void init_seg_sort_resources(
        SegSortGpuResources& resources,
        size_t initial_capacity,
        size_t max_todo_cnt,
        size_t initial_scan_temp_bytes,
        size_t initial_sort_temp_bytes,
        cudaStream_t stream
) {
    resources.key_value_capacity = initial_capacity;
    size_t initial_bytes = initial_capacity * sizeof(int);
    CUDA_CHECK(cudaMallocAsync(&resources.key_ptr,       initial_bytes, stream));
    CUDA_CHECK(cudaMallocAsync(&resources.value_ptr,     initial_bytes, stream));
    CUDA_CHECK(cudaMallocAsync(&resources.key_alt_ptr,   initial_bytes, stream));
    CUDA_CHECK(cudaMallocAsync(&resources.value_alt_ptr, initial_bytes, stream));

    resources.task_sizes_bytes = max_todo_cnt * sizeof(int);
    resources.seg_offsets_bytes = (max_todo_cnt + 1) * sizeof(int);
    CUDA_CHECK(cudaMallocAsync(&resources.task_sizes_ptr, resources.task_sizes_bytes, stream));
    CUDA_CHECK(cudaMallocAsync(&resources.seg_offsets_ptr, resources.seg_offsets_bytes, stream));

    resources.scan_temp_bytes = initial_scan_temp_bytes;
    resources.sort_temp_bytes = initial_sort_temp_bytes;
    CUDA_CHECK(cudaMallocAsync(&resources.scan_temp_ptr, resources.scan_temp_bytes, stream));
    CUDA_CHECK(cudaMallocAsync(&resources.sort_temp_ptr, resources.sort_temp_bytes, stream));
}

void free_seg_sort_resources(SegSortGpuResources& resources, cudaStream_t stream) {
    if (resources.key_ptr)       CUDA_CHECK(cudaFreeAsync(resources.key_ptr, stream));
    if (resources.value_ptr)     CUDA_CHECK(cudaFreeAsync(resources.value_ptr, stream));
    if (resources.key_alt_ptr)   CUDA_CHECK(cudaFreeAsync(resources.key_alt_ptr, stream));
    if (resources.value_alt_ptr) CUDA_CHECK(cudaFreeAsync(resources.value_alt_ptr, stream));

    if (resources.task_sizes_ptr) CUDA_CHECK(cudaFreeAsync(resources.task_sizes_ptr, stream));
    if (resources.seg_offsets_ptr) CUDA_CHECK(cudaFreeAsync(resources.seg_offsets_ptr, stream));
    if (resources.scan_temp_ptr)  CUDA_CHECK(cudaFreeAsync(resources.scan_temp_ptr, stream));
    if (resources.sort_temp_ptr)  CUDA_CHECK(cudaFreeAsync(resources.sort_temp_ptr, stream));
    resources = {};
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
    cudaMallocManaged(&global_randstrobes, batch_read_num * 2  / SMALL_CHUNK_FAC * sizeof(my_vector<QueryRandstrobe>));
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
    cudaMallocManaged(&global_hits_per_ref0s, batch_read_num * 2 / SMALL_CHUNK_FAC * sizeof(my_vector<my_pair<int, Hit>>));
    meta_data_size += batch_read_num * 2 * sizeof(my_vector<my_pair<int, Hit>>);
    cudaMemset(global_hits_per_ref0s, 0, batch_read_num * 2 * sizeof(my_vector<my_pair<int, Hit>>));
    my_vector<my_pair<int, Hit>> *global_hits_per_ref1s;
    cudaMallocManaged(&global_hits_per_ref1s, batch_read_num * 2 / SMALL_CHUNK_FAC * sizeof(my_vector<my_pair<int, Hit>>));
    meta_data_size += batch_read_num * 2 * sizeof(my_vector<my_pair<int, Hit>>);
    cudaMemset(global_hits_per_ref1s, 0, batch_read_num * 2 * sizeof(my_vector<my_pair<int, Hit>>));
    my_vector<Nam> *global_nams;
    cudaMallocManaged(&global_nams, batch_read_num * 2 / SMALL_CHUNK_FAC * sizeof(my_vector<Nam>));
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

    SegSortGpuResources buffers0;
    SegSortGpuResources buffers1;
#ifdef use_seg_sort
    size_t pre_alloc_elements = 1024 * 1024 * chunk_num * 2;
    size_t max_tasks = batch_read_num;
    size_t scan_buffer_size = 1024 * 1024;
    size_t sort_buffer_size = 1024 * 1024 * 8;
    init_seg_sort_resources(buffers0, pre_alloc_elements, max_tasks, scan_buffer_size, sort_buffer_size, ctx.stream);
    init_seg_sort_resources(buffers1, pre_alloc_elements, max_tasks, scan_buffer_size, sort_buffer_size, ctx.stream);

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
        chunk0_real_chunk_num = 0;
        chunk0_data1s.clear();
        chunk0_data2s.clear();
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
    int rescue_threshold = RESCUE_THRESHOLD;
    printf("rescue_threshold %d\n", rescue_threshold);


    // step: f_1
    {
        bool res;
        // format data
        t_1 = GetTime();
        int rc_pos1 = 0, rc_pos2 = 0;
        chunk1_chunk_num = rand() % small_chunk_num + small_chunk_num + 1;
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
                                                 buffers0, buffers1,
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
        for (int i = 0; i < chunk0_data1s.size(); i++) {
            GPUAlignTmpRes &align_tmp_res = chunk0_global_align_res[i];
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
//        for (int i = 0; i < chunk0_data1s.size(); i++) {
//            GPUAlignTmpRes &align_tmp_res = chunk0_global_align_res[i];
//            if (align_tmp_res.type != 3) continue;
//            if (align_tmp_res.type3_isize_val == -1) continue;
//            isize_est->update(align_tmp_res.type3_isize_val);
//        }

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
                    if (sam_out.length() > 0) output_buffer.output_records(std::move(sam_out), fx_chunk_id, unordered_output);
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
                                                     buffers0, buffers1,
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
            for (int i = 0; i < chunk1_data1s.size(); i++) {
                GPUAlignTmpRes &align_tmp_res = chunk1_global_align_res[i];
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
    //    // The following cout block is added as requested.
    std::cout << "--------------- GPU Kernel Timers ---------------" << std::endl;
    std::cout << "pack read on host cost: " << gpu_copy1 << " s" << std::endl;
    std::cout << "H2D read cost: " << gpu_copy2 << " s" << std::endl;
    std::cout << "get_randstrobes cost (gpu_cost1): " << gpu_cost1 << " s" << std::endl;
    std::cout << "get_hits cost (gpu_cost2): " << gpu_cost2 << " s" << std::endl;
    std::cout << "filter normal read cost (gpu_init1): " << gpu_init1 << " s" << std::endl;
    std::cout << "sort_hits cost (gpu_cost3): " << gpu_cost3 << " [" << gpu_cost3_1 << " " << gpu_cost3_2 << " " << gpu_cost3_3 << " " << gpu_cost3_4 << "] " << " s" << std::endl;
    std::cout << "merge_hits_get_nams cost (gpu_cost4): " << gpu_cost4 << " s" << std::endl;
    std::cout << "filter rescue read cost (gpu_init2): " << gpu_init2 << " s" << std::endl;
    std::cout << "rescue_get_hits cost (gpu_cost5): " << gpu_cost5 << " s" << std::endl;
    std::cout << "rescue_sort_hits cost (gpu_cost6): " << gpu_cost6 << " [" << gpu_cost6_1 << " " << gpu_cost6_2 << " " << gpu_cost6_3 << " " << gpu_cost6_4 << "] " << " s" << std::endl;
    std::cout << "rescue_merge_hits_get_nams cost (gpu_cost7): " << gpu_cost7 << " s" << std::endl;
    std::cout << "sort_nams cost (gpu_cost8): " << gpu_cost8 << " [" << gpu_cost8_1 << " " << gpu_cost8_2 << " " << gpu_cost8_3 << " " << gpu_cost8_4 << "] " << " s" << std::endl;
    std::cout << "pre_cal_type cost (gpu_cost9): " << gpu_cost9 << " s" << std::endl;
    std::cout << "alloc align_tmp_res cost (gpu_init3): " << gpu_init3 << " s" << std::endl;
    std::cout << "align_PE cost (gpu_cost10): " << gpu_cost10 << " [" << gpu_cost10_1 << " " << gpu_cost10_2 << " " << gpu_cost10_3 << " " << gpu_cost10_4 << "] " << " s" << std::endl;
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

#ifdef use_device_mem
    cudaFree(device_query_ptr);
    cudaFree(device_ref_ptr);
#endif

#ifdef use_seg_sort
    free_seg_sort_resources(buffers0, ctx.stream);
    free_seg_sort_resources(buffers1, ctx.stream);
#endif

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
