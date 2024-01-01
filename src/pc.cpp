//
// Created by Kristoffer Sahlin on 3/22/22.
//

// Using initial base format of Buffer classed from: https://andrew128.github.io/ProducerConsumer/

#include "pc.hpp"
#include <pthread.h>
#include <chrono>
#include <iostream>
#include <mutex>
#include <queue>

#include "index.hpp"
#include "kseq++/kseq++.hpp"
#include "revcomp.hpp"
#include "robin_hood.h"
#include "sam.hpp"
#include "timer.hpp"

// checks if two read names are the same ignoring /1 suffix on the first one
// and /2 on the second one (if present)
bool same_name(const std::string& n1, const std::string& n2) {
    if (n1.length() != n2.length())
        return false;
    if (n1.length() <= 2)
        return n1 == n2;
    size_t i = 0;
    for (; i < n1.length() - 1; ++i) {
        if (n1[i] != n2[i])
            return false;
    }
    if (n1[i - 1] == '/' && n1[i] == '1' && n2[i] == '2')
        return true;
    return n1[i] == n2[i];
}

// distribute_interleaved implements the 'interleaved' format:
// If two consequent reads have the same name, they are considered to be a pair.
// Otherwise, they are considered to be single-end reads.

void distribute_interleaved(
    std::vector<klibpp::KSeq>& records,
    std::vector<klibpp::KSeq>& records1,
    std::vector<klibpp::KSeq>& records2,
    std::vector<klibpp::KSeq>& records3,
    std::optional<klibpp::KSeq>& lookahead1
) {
    auto it = records.begin();
    if (lookahead1) {
        if (it != records.end() && same_name(lookahead1->name, it->name)) {
            records1.push_back(*lookahead1);
            records2.push_back(*it);
            ++it;
        } else {
            records3.push_back(*lookahead1);
        }
        lookahead1 = std::nullopt;
    }
    for (; it != records.end(); ++it) {
        if (it + 1 != records.end() && same_name(it->name, (it + 1)->name)) {
            records1.push_back(*it);
            records2.push_back(*(it + 1));
            ++it;
        } else {
            records3.push_back(*it);
        }
    }
    if (it != records.end()) {
        lookahead1 = *it;
    }
}

size_t InputBuffer::read_records(
    std::vector<klibpp::KSeq>& records1,
    std::vector<klibpp::KSeq>& records2,
    std::vector<klibpp::KSeq>& records3,
    int to_read
) {
    records1.clear();
    records2.clear();
    records3.clear();
    // Acquire a unique lock on the mutex
    std::unique_lock<std::mutex> unique_lock(mtx);
    if (to_read == -1) {
        to_read = chunk_size;
    }
    if (this->is_interleaved) {
        auto records = ks1->stream().read(to_read * 2);
        distribute_interleaved(records, records1, records2, records3, lookahead1);
    } else if (!ks2) {
        records3 = ks1->stream().read(to_read);
    } else {
        records1 = ks1->stream().read(to_read);
        records2 = ks2->stream().read(to_read);
    }
    size_t current_chunk_index = chunk_index;
    chunk_index++;

    if (records1.empty() && records3.empty()) {
        finished_reading = true;
    }

    unique_lock.unlock();

    return current_chunk_index;
}

void InputBuffer::rewind_reset() {
    std::unique_lock<std::mutex> unique_lock(mtx);
    ks1->rewind();
    if (ks2) {
        ks2->rewind();
    }
    finished_reading = false;
    chunk_index = 0;
}

void OutputBuffer::output_records(std::string chunk, size_t chunk_index) {
    std::unique_lock<std::mutex> unique_lock(mtx);

    // Ensure we print the chunks in the order in which they were read
    assert(chunks.count(chunk_index) == 0);
    chunks.emplace(std::make_pair(chunk_index, chunk));
    while (true) {
        const auto& item = chunks.find(next_chunk_index);
        if (item == chunks.end()) {
            break;
        }
        out << item->second;
        chunks.erase(item);
        next_chunk_index++;
    }
    unique_lock.unlock();
}

inline void part2_extend_seed(AlignTmpRes &align_tmp_res, int j, Read read1, Read read2, const References& references, const Aligner& aligner) {
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
    info = aligner.align(query, ref_segm);
    result_ref_start = ref_start + info.ref_start;
    int softclipped = info.query_start + (query.size() - info.query_end);
    Alignment &alignment = align_tmp_res.align_res[j];
    alignment.cigar = std::move(info.cigar);
    alignment.edit_distance = info.edit_distance;
    alignment.global_ed = info.edit_distance + softclipped;
    alignment.score = info.sw_score;
    alignment.ref_start = result_ref_start;
    alignment.length = info.ref_span();
    alignment.is_rc = nam.is_rc;
    alignment.is_unaligned = false;
    alignment.ref_id = nam.ref_id;
    alignment.gapped = true;
}

inline void part2_rescue_mate(AlignTmpRes &align_tmp_res, int j, Read read1, Read read2, const References& references, const Aligner& aligner, float mu, float sigma) {
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

    auto info = aligner.align(r_tmp, ref_segm);

    Alignment &alignment = align_tmp_res.align_res[j];

    alignment.cigar = info.cigar;
    alignment.edit_distance = info.edit_distance;
    alignment.score = info.sw_score;
    alignment.ref_start = ref_start + info.ref_start;
    alignment.is_rc = !nam.is_rc;
    alignment.ref_id = nam.ref_id;
    alignment.is_unaligned = info.cigar.empty();
    alignment.length = info.ref_span();
}

void perform_task(
    InputBuffer &input_buffer,
    OutputBuffer &output_buffer,
    AlignmentStatistics& statistics,
    int& done,
    const AlignmentParameters &aln_params,
    const MappingParameters &map_param,
    const IndexParameters& index_parameters,
    const References& references,
    const StrobemerIndex& index,
    const std::string& read_group_id,
    const int thread_id
) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(thread_id, &cpuset);

    pthread_t current_thread = pthread_self();
    if (pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) != 0) {
        std::cerr << "Error setting thread affinity" << std::endl;
    }

    bool eof = false;
    Aligner aligner{aln_params};
    std::minstd_rand random_engine;
    while (!eof) {
        std::vector<klibpp::KSeq> records1;
        std::vector<klibpp::KSeq> records2;
        std::vector<klibpp::KSeq> records3;
        Timer timer;
        auto chunk_index = input_buffer.read_records(records1, records2, records3);
        statistics.tot_read_file += timer.duration();
        assert(records1.size() == records2.size());
        if (records1.empty()
            && records3.empty()
            && input_buffer.finished_reading){
            break;
        }

        std::string sam_out;
        sam_out.reserve(7*map_param.r * (records1.size() + records3.size()));
        Sam sam{sam_out, references, map_param.cigar_ops, read_group_id, map_param.output_unmapped, map_param.details};
        InsertSizeDistribution isize_est;
        // Use chunk index as random seed for reproducibility
        random_engine.seed(chunk_index);
        for (size_t i = 0; i < records1.size(); ++i) {
            auto record1 = records1[i];
            auto record2 = records2[i];
            to_uppercase(record1.seq);
            to_uppercase(record2.seq);
            align_PE_read(record1, record2, sam, sam_out, statistics, isize_est, aligner,
                          map_param, index_parameters, references, index, random_engine);
            statistics.n_reads += 2;
        }
        for (size_t i = 0; i < records3.size(); ++i) {
            auto record = records3[i];
            align_SE_read(record, sam, sam_out, statistics, aligner, map_param, index_parameters, references, index, random_engine);
            statistics.n_reads++;
        }
        output_buffer.output_records(std::move(sam_out), chunk_index);
        assert(sam_out == "");
    }
    statistics.tot_aligner_calls += aligner.calls_count();
    done = true;
}
#include <sys/time.h>
inline double GetTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double) tv.tv_sec + (double) tv.tv_usec / 1000000;
}
void perform_task_sync(
    InputBuffer& input_buffer,
    OutputBuffer& output_buffer,
    AlignmentStatistics& statistics,
    int& done,
    const AlignmentParameters& aln_params,
    const MappingParameters& map_param,
    const IndexParameters& index_parameters,
    const References& references,
    const StrobemerIndex& index,
    const std::string& read_group_id,
    const int thread_id
) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(thread_id, &cpuset);

    pthread_t current_thread = pthread_self();
    if (pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) != 0) {
        std::cerr << "Error setting thread affinity" << std::endl;
    }

    bool eof = false;
    Aligner aligner{aln_params};
    std::minstd_rand random_engine;
    std::vector<klibpp::KSeq> records1;
    std::vector<klibpp::KSeq> records2;
    std::vector<klibpp::KSeq> records3;
    std::vector<klibpp::KSeq> pre_records1;
    std::vector<klibpp::KSeq> pre_records2;
    std::vector<klibpp::KSeq> pre_records3;
    size_t chunk_index;
    size_t pre_chunk_index;
    std::vector<AlignTmpRes> align_tmp_results;
    std::vector<AlignTmpRes> pre_align_tmp_results;
    thread_local double time0 = 0;
    thread_local double time1 = 0;
    thread_local double time2 = 0;
    thread_local double time3 = 0;

    double t00 = GetTime();
    double t0 = GetTime();

    //chunk0_part1
    Timer timer;
    pre_chunk_index = input_buffer.read_records(pre_records1, pre_records2, pre_records3);
    statistics.tot_read_file += timer.duration();
    assert(pre_records1.size() == pre_records2.size());
    if (pre_records1.empty() && pre_records3.empty() && input_buffer.finished_reading) {
        eof = true;
    }

    InsertSizeDistribution isize_est;
    // Use chunk index as random seed for reproducibility
    random_engine.seed(pre_chunk_index);
    for (size_t i = 0; i < pre_records1.size(); ++i) {
        auto record1 = pre_records1[i];
        auto record2 = pre_records2[i];
        to_uppercase(record1.seq);
        to_uppercase(record2.seq);
        AlignTmpRes align_tmp_res;
        align_PE_read_part(
            align_tmp_res, record1, record2, statistics, isize_est, aligner, map_param, index_parameters,
            references, index, random_engine
        );
        pre_align_tmp_results.push_back(align_tmp_res);
        statistics.n_reads += 2;
//        fprintf(stderr, "ii %d %d %zu %d\n", i, align_tmp_res.type, align_tmp_res.todo_nams.size(), align_tmp_res.type4_loop_size);
    }
//    fprintf(stderr, "111\n");
    time1 += GetTime() - t0;

    while (!eof) {
        //chunk0_part2
        //process todo_nams
        t0 = GetTime();
        for (size_t i = 0; i < pre_records1.size(); i++) {
            auto record1 = pre_records1[i];
            auto record2 = pre_records2[i];
            to_uppercase(record1.seq);
            to_uppercase(record2.seq);
            Read read1(record1.seq);
            Read read2(record2.seq);
            const auto mu = isize_est.mu;
            const auto sigma = isize_est.sigma;
            auto& align_tmp_res = pre_align_tmp_results[i];
            size_t todo_size = align_tmp_res.todo_nams.size();
            assert(todo_size == align_tmp_res.done_align.size());
            assert(todo_size == align_tmp_res.align_res.size());
//            fprintf(stderr, "type %d\n", align_tmp_res.type);
//            fprintf(stderr, "size1 %zu\n", todo_size);
//            fprintf(stderr, "size2 %d\n", align_tmp_res.type4_loop_size);

            if(align_tmp_res.type == 1 || align_tmp_res.type == 2) {
                assert(todo_size % 2 == 0);
                for(size_t j = 0; j < todo_size; j += 2) {
                    assert(align_tmp_res.is_extend_seed[j]);
                    if(align_tmp_res.type == 1) assert(align_tmp_res.is_read1[j]);
                    else assert(!align_tmp_res.is_read1[j]);
                    if(!align_tmp_res.done_align[j]) {
                        // solve extend_seed for good read1
                        part2_extend_seed(align_tmp_res, j, read1, read2, references, aligner);
                    }
                    assert(!align_tmp_res.is_extend_seed[j + 1]);
                    if(align_tmp_res.type == 1) assert(!align_tmp_res.is_read1[j + 1]);
                    else assert(align_tmp_res.is_read1[j + 1]);
                    if(!align_tmp_res.done_align[j + 1]) {
                        // solve rescue_mate for bad read2
                        part2_rescue_mate(align_tmp_res, j + 1, read1, read2, references, aligner, mu, sigma);
                    }
                }
            } else if(align_tmp_res.type == 3) {
                assert(todo_size == 2);
                assert(align_tmp_res.is_extend_seed[0]);
                assert(align_tmp_res.is_read1[0]);
                if(!align_tmp_res.done_align[0]) {
                    // solve extend_seed for read1
                    part2_extend_seed(align_tmp_res, 0, read1, read2, references, aligner);
                }
                assert(align_tmp_res.is_extend_seed[1]);
                assert(!align_tmp_res.is_read1[1]);
                if(!align_tmp_res.done_align[1]) {
                    // solve extend_seed for read2
                    part2_extend_seed(align_tmp_res, 1, read1, read2, references, aligner);
                }
                //TODO
//                bool is_proper = is_proper_pair(align_tmp_res.align_res[0], align_tmp_res.align_res[1], mu, sigma);
//                if ((isize_est.sample_size < 400) && (align_tmp_res.align_res[0].edit_distance + align_tmp_res.align_res[1].edit_distance < 3) &&
//                    is_proper) {
//                    isize_est.update(std::abs(align_tmp_res.align_res[0].ref_start - align_tmp_res.align_res[1].ref_start));
//                }
            } else if(align_tmp_res.type == 4) {
                for(size_t j = 0; j < todo_size; j++) {
                    if(!align_tmp_res.done_align[j]) {
                        if(align_tmp_res.is_extend_seed[j]) {
                            part2_extend_seed(align_tmp_res, j, read1, read2, references, aligner);
                        } else {
                            part2_rescue_mate(align_tmp_res, j, read1, read2, references, aligner, mu, sigma);
                        }
                    }
                }
            }
        }
        time2 = GetTime() - t0;


        t0 = GetTime();
        //chunk1_part2
        Timer timer;
        auto chunk_index = input_buffer.read_records(records1, records2, records3);
        statistics.tot_read_file += timer.duration();
        assert(records1.size() == records2.size());
        if (records1.empty() && records3.empty() && input_buffer.finished_reading) {
            eof = true;
        }
//        fprintf(stderr, "eof %d\n", eof);

        InsertSizeDistribution isize_est;
        // Use chunk index as random seed for reproducibility
        random_engine.seed(chunk_index);
        for (size_t i = 0; i < records1.size(); ++i) {
            auto record1 = records1[i];
            auto record2 = records2[i];
            to_uppercase(record1.seq);
            to_uppercase(record2.seq);
            AlignTmpRes align_tmp_res;
            align_PE_read_part(
                align_tmp_res, record1, record2, statistics, isize_est, aligner, map_param, index_parameters,
                references, index, random_engine
            );
            align_tmp_results.push_back(align_tmp_res);
            statistics.n_reads += 2;
        }
        time1 += GetTime() - t0;

        //chunk0_part3

        t0 = GetTime();
        std::string sam_out;
        sam_out.reserve(7 * map_param.r * (pre_records1.size() + pre_records3.size()));
        Sam sam{sam_out, references, map_param.cigar_ops, read_group_id, map_param.output_unmapped, map_param.details};
        for (size_t i = 0; i < pre_records1.size(); ++i) {
            auto record1 = pre_records1[i];
            auto record2 = pre_records2[i];
            to_uppercase(record1.seq);
            to_uppercase(record2.seq);
            align_PE_read_last(
                pre_align_tmp_results[i], record1, record2, sam, sam_out, statistics, isize_est, aligner,
                map_param, index_parameters, references, index, random_engine
            );
//            fprintf(stderr, "last %zu\n", i);

        }

        output_buffer.output_records(std::move(sam_out), pre_chunk_index);
        time3 += GetTime() - t0;
        //assert(sam_out == "");

        //change data
        pre_align_tmp_results = std::move(align_tmp_results);
        pre_records1 = std::move(records1);
        pre_records2 = std::move(records2);
        pre_records3 = std::move(records3);
        pre_chunk_index = chunk_index;
    }
    statistics.tot_aligner_calls += aligner.calls_count();
    done = true;
    fprintf(stderr, "cost time1:%.2f time2:%.2f time3:%.2f, tot time:%.2f\n", time1, time2, time3, GetTime() - t00);
}
