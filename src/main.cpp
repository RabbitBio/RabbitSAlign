#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <string>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <thread>
#include <cassert>
#include <iomanip>
#include <chrono>
#include <random>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#include "refs.hpp"
#include "exceptions.hpp"
#include "cmdline.hpp"
#include "index.hpp"
#include "pc.hpp"
#include "aln.hpp"
#include "logger.hpp"
#include "timer.hpp"
#include "readlen.hpp"
#include "version.hpp"
#include "buildconfig.hpp"
#include "gpu_step.h"

#ifdef RABBIT_FX
#include "FastxStream.h"
#include "FastxChunk.h"
#include "DataQueue.h"
#include "Formater.h"
#endif

#define use_gpu_align

//#define use_device_mem

#include <sys/time.h>
inline double GetTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double) tv.tv_sec + (double) tv.tv_usec / 1000000;
}

int getNumaNodeCount() {
    FILE* pipe = popen("lscpu | grep 'NUMA node(s)' | awk '{print $3}'", "r");
    if (!pipe) return -1;
    char buffer[128];
    std::string result = "";
    while (!feof(pipe)) {
        if (fgets(buffer, 128, pipe) != nullptr)
            result += buffer;
    }
    pclose(pipe);
    return std::stoi(result);
}

long long getAvailableMemory() {
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    long long availableMemory = -1;
    while (std::getline(meminfo, line)) {
        std::istringstream iss(line);
        std::string key;
        long long value;
        std::string unit;
        iss >> key >> value >> unit;
        if (key == "MemAvailable:") {
            availableMemory = value * 1024;  // Convert from kB to Bytes
            break;
        }
    }
    return availableMemory;
}


static Logger& logger = Logger::get();

/*
 * Return formatted SAM header as a string
 */
std::string sam_header(const References& references, const std::string& read_group_id, const std::vector<std::string>& read_group_fields, const std::string& cmd_line) {
    std::stringstream out;
    out << "@HD\tVN:1.6\tSO:unsorted\n";
    for (size_t i = 0; i < references.size(); ++i) {
        out << "@SQ\tSN:" << references.names[i] << "\tLN:" << references.lengths[i] << "\n";
    }
    if (!read_group_id.empty()) {
        out << "@RG\tID:" << read_group_id;
        for (const auto& field : read_group_fields) {
           out << '\t' << field;
        }
        out << '\n';
    }
    out << "@PG\tID:rabbitsalign\tPN:rabbitsalign\tVN:" << version_string() << "\tCL:" << cmd_line << std::endl;
    return out.str();
}

void warn_if_no_optimizations() {
    if (std::string(CMAKE_BUILD_TYPE) == "Debug") {
        logger.info() << "\n    ***** Binary was compiled without optimizations - this will be very slow *****\n\n";
    }
}

void log_parameters(const IndexParameters& index_parameters, const MappingParameters& map_param, const AlignmentParameters& aln_params) {
    logger.debug() << "Using" << std::endl
        << "k: " << index_parameters.syncmer.k << std::endl
        << "s: " << index_parameters.syncmer.s << std::endl
        << "w_min: " << index_parameters.randstrobe.w_min << std::endl
        << "w_max: " << index_parameters.randstrobe.w_max << std::endl
        << "Read length (r): " << map_param.r << std::endl
        << "Maximum seed length: " << index_parameters.randstrobe.max_dist + index_parameters.syncmer.k << std::endl
        << "R: " << map_param.rescue_level << std::endl
        << "Expected [w_min, w_max] in #syncmers: [" << index_parameters.randstrobe.w_min << ", " << index_parameters.randstrobe.w_max << "]" << std::endl
        << "Expected [w_min, w_max] in #nucleotides: [" << (index_parameters.syncmer.k - index_parameters.syncmer.s + 1) * index_parameters.randstrobe.w_min << ", " << (index_parameters.syncmer.k - index_parameters.syncmer.s + 1) * index_parameters.randstrobe.w_max << "]" << std::endl
        << "A: " << aln_params.match << std::endl
        << "B: " << aln_params.mismatch << std::endl
        << "O: " << aln_params.gap_open << std::endl
        << "E: " << aln_params.gap_extend << std::endl
        << "end bonus: " << aln_params.end_bonus << '\n';
}

bool avx2_enabled() {
#ifdef __AVX2__
    return true;
#else
    return false;
#endif
}

InputBuffer get_input_buffer(const CommandLineOptions& opt) {
    if (opt.is_SE) {
        return InputBuffer(opt.reads_filename1, "", opt.chunk_size, false);
    } else if (opt.is_interleaved) {
        if (opt.reads_filename2 != "") {
            throw BadParameter("Cannot specify both --interleaved and specify two read files");
        }
        return InputBuffer(opt.reads_filename1, "", opt.chunk_size, true);
    } else {
        return InputBuffer(opt.reads_filename1, opt.reads_filename2, opt.chunk_size, false);
    }
}

void show_progress_until_done(std::vector<int>& worker_done, std::vector<AlignmentStatistics>& stats) {
    Timer timer;
    bool reported = false;
    bool done = false;
    // Waiting time between progress updates
    // Start with a small value so that there’s no delay if there are very few
    // reads to align.
    auto time_to_wait = std::chrono::milliseconds(1);
    while (!done) {
        std::this_thread::sleep_for(time_to_wait);
        // Ramp up waiting time
        time_to_wait = std::min(time_to_wait * 2, std::chrono::milliseconds(1000));
        done = true;
        for (auto is_done : worker_done) {
            if (!is_done) {
                done = false;
                continue;
            }
        }
        auto n_reads = 0ull;
        for (auto& stat : stats) {
            n_reads += stat.n_reads;
        }
        auto elapsed = timer.elapsed();
        if (elapsed >= 1.0) {
            std::cerr
                << " Mapped "
                << std::setw(12) << (n_reads / 1E6) << " M reads @ "
                << std::setw(8) << (timer.elapsed() * 1E6 / n_reads) << " us/read                   \r";
            reported = true;
        }
    }
    if (reported) {
        std::cerr << '\n';
    }
}


int totalCPUs;
void setThreadAffinity(int cpu_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);

    pthread_t current_thread = pthread_self();
    if (pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) != 0) {
        std::cerr << "Error setting thread affinity for CPU " << cpu_id << std::endl;
    }
}

void readIndexOnCPU(StrobemerIndex& index, const std::string& sti_path, int cpu_id) {
    setThreadAffinity(cpu_id);
    index.read(sti_path);
}

#ifdef RABBIT_FX

int producer_pe_fastq_task(std::string file, std::string file2, rabbit::fq::FastqDataPool &fastqPool, rabbit::core::TDataQueue<rabbit::fq::FastqDataPairChunk> &dq) {
	rabbit::fq::FastqFileReader *fqFileReader;
	fqFileReader = new rabbit::fq::FastqFileReader(file, fastqPool, false, file2, 1 << 12);
	int n_chunks = 0;
	int line_sum = 0;
	while (true) {
		rabbit::fq::FastqDataPairChunk *fqdatachunk = new rabbit::fq::FastqDataPairChunk;
		fqdatachunk = fqFileReader->readNextPairChunk();
		if (fqdatachunk == NULL) break;
		//std::cout << "readed chunk: " << n_chunks << std::endl;
		dq.Push(n_chunks, fqdatachunk);
		n_chunks++;
	}

	dq.SetCompleted();
	delete fqFileReader;
	std::cerr << "file " << file << " has " << n_chunks << " chunks" << std::endl;
	return 0;
}

int producer_se_fastq_task(std::string file, rabbit::fq::FastqDataPool& fastqPool, rabbit::core::TDataQueue<rabbit::fq::FastqDataChunk> &dq){
	rabbit::fq::FastqFileReader fqFileReader(file, fastqPool, false, "", 1 << 12);
	rabbit::int64 n_chunks = 0; 
	while(true){ 
		rabbit::fq::FastqDataChunk* fqdatachunk;// = new rabbit::fq::FastqDataChunk;
		fqdatachunk = fqFileReader.readNextChunk(); 
		if (fqdatachunk == NULL) break;
		//std::cout << "readed chunk: " << n_chunks << std::endl;
		dq.Push(n_chunks, fqdatachunk);
		n_chunks++;
	}
	dq.SetCompleted();
	std::cerr << "file " << file << " has " << n_chunks << " chunks" << std::endl;
	return 0;
}
#endif

struct ThreadAssignment {
    int thread_id;
    int cpu_core;
    int numa_node;
    int gpu_id;
    int flag;
    int pass;
};

std::vector<ThreadAssignment> assign_threads_fixed_with_flags() {
    std::vector<ThreadAssignment> assignments;

    for (int thread_id = 0; thread_id < 72; ++thread_id) {
        int numa_node = 0;
        int gpu_id = 0;
        
        if (thread_id <= 35) {               // 0-35
            numa_node = 0;
            if (thread_id <= 17)
                gpu_id = 0;                   // 0-17 -> GPU0
            else
                gpu_id = 1;                   // 18-35 -> GPU1
        }
        else if (thread_id <= 71) {           // 36-71
            numa_node = 1;
            if (thread_id <= 53)
                gpu_id = 2;                   // 36-53 -> GPU2
            else
                gpu_id = 3;                   // 54-71 -> GPU3
        }

        int cpu_core = thread_id;

        assignments.push_back({thread_id, cpu_core, numa_node, gpu_id, 0, 0});
    }
    
    for (int base = 0; base < 72; base += 18) {
        if (base + 0 < 72) assignments[base + 0].flag = 1;
        if (base + 9 < 72) assignments[base + 9].flag = 1;
        if (base + 1 < 72) {
            assignments[base + 1].flag = 1;
            assignments[base + 1].pass = 1;
        }
        if (base + 10 < 72) {
            assignments[base + 10].flag = 1;
            assignments[base + 10].pass = 1;
        }
    }


    return assignments;
}

void set_scores(int gpu_id, int match_score, int mismatch_score, int gap_open_score, int gap_extend_score) {
    cudaSetDevice(gpu_id);
    gasal_subst_scores sub_scores;
    sub_scores.match = match_score;
    sub_scores.mismatch = mismatch_score;
    sub_scores.gap_open = gap_open_score - 1;
    sub_scores.gap_extend = gap_extend_score;
    gasal_copy_subst_scores(&sub_scores);
}

int run_rabbitsalign(int argc, char **argv) {
    auto opt = parse_command_line_arguments(argc, argv);

    logger.set_level(opt.verbose ? LOG_DEBUG : LOG_INFO);
    logger.info() << std::setprecision(2) << std::fixed;
    logger.info() << "This is rabbitsalign " << version_string() << '\n';
    logger.debug() << "Build type: " << CMAKE_BUILD_TYPE << '\n';
    warn_if_no_optimizations();
    logger.debug() << "AVX2 enabled: " << (avx2_enabled() ? "yes" : "no") << '\n';

    if (opt.c >= 64 || opt.c <= 0) {
        throw BadParameter("c must be greater than 0 and less than 64");
    }

    InputBuffer input_buffer = get_input_buffer(opt);
    if (!opt.r_set && !opt.reads_filename1.empty()) {
        opt.r = estimate_read_length(input_buffer);
        logger.info() << "Estimated read length: " << opt.r << " bp\n";
    }

#ifdef RABBIT_FX
    rabbit::fq::FastqDataPool fastqPool(4096, 1 << 20);
    rabbit::core::TDataQueue<rabbit::fq::FastqDataChunk> queue_se(4096, 1);
    rabbit::core::TDataQueue<rabbit::fq::FastqDataPairChunk> queue_pe(4096, 1);
    std::thread *producer;
    if(!opt.only_gen_index) {
        if(opt.is_SE) {
            producer = new std::thread(producer_se_fastq_task, opt.reads_filename1, std::ref(fastqPool), std::ref(queue_se));
        } else if(opt.is_interleaved) {
            producer = new std::thread(producer_se_fastq_task, opt.reads_filename1, std::ref(fastqPool), std::ref(queue_se));
        } else {
            producer = new std::thread(producer_pe_fastq_task, opt.reads_filename1, opt.reads_filename2, std::ref(fastqPool), std::ref(queue_pe));
        }
    }
#else
    input_buffer.rewind_reset();
#endif
    IndexParameters index_parameters = IndexParameters::from_read_length(
        opt.r,
        opt.k_set ? opt.k : IndexParameters::DEFAULT,
        opt.s_set ? opt.s : IndexParameters::DEFAULT,
        opt.l_set ? opt.l : IndexParameters::DEFAULT,
        opt.u_set ? opt.u : IndexParameters::DEFAULT,
        opt.c_set ? opt.c : IndexParameters::DEFAULT,
        opt.max_seed_len_set ? opt.max_seed_len : IndexParameters::DEFAULT
    );
    logger.debug() << index_parameters << '\n';
    AlignmentParameters aln_params;
    aln_params.match = opt.A;
    aln_params.mismatch = opt.B;
    aln_params.gap_open = opt.O;
    aln_params.gap_extend = opt.E;
    aln_params.end_bonus = opt.end_bonus;

    MappingParameters map_param;
    map_param.r = opt.r;
    map_param.max_secondary = opt.max_secondary;
    map_param.dropoff_threshold = opt.dropoff_threshold;
    map_param.rescue_level = opt.rescue_level;
    map_param.max_tries = opt.max_tries;
    map_param.is_sam_out = opt.is_sam_out;
    map_param.cigar_ops = opt.cigar_eqx ? CigarOps::EQX : CigarOps::M;
    map_param.output_unmapped = opt.output_unmapped;
    map_param.details = opt.details;
    map_param.verify();


    const int gpu_num = 4;

    std::vector<ThreadAssignment> assignments = assign_threads_fixed_with_flags();

//    std::vector<gasal_tmp_res> gasal_results_tmp;
//    std::vector<std::string> query_batch;
//    std::vector<std::string> ref_batch;
//    std::vector<std::string_view> query_batch_v;
//    std::vector<std::string_view> ref_batch_v;
//    std::string query_test = "AAA\n";
//    std::string ref_test = "AAA\n";
//    for (int i = 0; i < STREAM_BATCH_SIZE; i++) {
//        query_batch.push_back(query_test);
//        ref_batch.push_back(ref_test);
//    }
//    printf("init gasal2\n");
//
//    //assert(opt.n_threads == 144);
//    for (int i = 0; i < opt.n_threads; i++) {
//        cudaSetDevice(assignments[i].gpu_id);
//        char* d_seq_demo;
//        char* d_ref_demo;
//        cudaMalloc(&d_seq_demo, 4);
//        cudaMemset(d_seq_demo, 'A', 4);
//        cudaMalloc(&d_ref_demo, 4);
//        cudaMemset(d_ref_demo, 'A', 4);
//        query_batch_v.clear();
//        ref_batch_v.clear();
//        for (int i = 0; i < STREAM_BATCH_SIZE_GPU; i++) {
//#ifdef use_device_mem
//            query_batch_v.push_back(std::string_view(d_seq_demo, 4));
//            ref_batch_v.push_back(std::string_view(d_ref_demo, 4));
//#else
//            query_batch_v.push_back(std::string_view(query_test));
//            ref_batch_v.push_back(std::string_view(ref_test));
//#endif
//        }
//        if(assignments[i].flag) {
//            solve_ssw_on_gpu2(i, gasal_results_tmp, query_batch_v, ref_batch_v, aln_params.match,
//                             aln_params.mismatch, aln_params.gap_open, aln_params.gap_extend);
//        } else {
//            solve_ssw_on_gpu(i, gasal_results_tmp, query_batch, ref_batch, aln_params.match,
//                             aln_params.mismatch, aln_params.gap_open, aln_params.gap_extend);
//        }
//        cudaFree(d_seq_demo);
//        cudaFree(d_ref_demo);
//    }
//    printf("init done\n");

    log_parameters(index_parameters, map_param, aln_params);
    logger.debug() << "Threads: " << opt.n_threads << std::endl;

//    assert(k <= (w/2)*w_min && "k should be smaller than (w/2)*w_min to avoid creating short strobemers");

    // Create index
    References references;
    Timer read_refs_timer;
    references = References::from_fasta(opt.ref_filename);
//    for (int i = 0; i < references.size(); i++) {
//        for (int j = 0; j < references.sequences[i].length(); j++) {
//            auto c = references.sequences[i][j];
//            bool res = (c == 'A' || c == 'N' || c == 'C' || c == 'G' || c == 'T');
//            if (!res) references.sequences[i][j] = 'A';
//        }
//    }
    logger.info() << "Time reading reference: " << read_refs_timer.elapsed() << " s\n";

    logger.info() << "Reference size: " << references.total_length() / 1E6 << " Mbp ("
        << references.size() << " contig" << (references.size() == 1 ? "" : "s")
        << "; largest: "
        << (*std::max_element(references.lengths.begin(), references.lengths.end()) / 1E6) << " Mbp)\n";
    if (references.total_length() == 0) {
        throw InvalidFasta("No reference sequences found");
    }

    int numa_num = getNumaNodeCount();
    long long mem_avali = getAvailableMemory();
    bool use_good_numa = 1;
#ifdef OPT_NUMA_CLOSE
    use_good_numa = 0;
#endif
    if(numa_num * (15ll << 30) > mem_avali * 0.8) {
        use_good_numa = 0;
    }
    if(numa_num == 1) {
        use_good_numa = 0;
    }
    if(!opt.use_index) {
        use_good_numa = 0;
    }
    fprintf(stderr, "use_good_numa is %d\n", use_good_numa);

    StrobemerIndex index(references, index_parameters, opt.bits);
    StrobemerIndex index2(references, index_parameters, opt.bits);
    if (opt.use_index) {
        // Read the index from a file
        assert(!opt.only_gen_index);
        Timer read_index_timer;
        std::string sti_path = opt.ref_filename + index_parameters.filename_extension();
        logger.info() << "Reading index from " << sti_path << '\n';
        //totalCPUs = std::thread::hardware_concurrency();
        totalCPUs = 72;
    
		fprintf(stderr, "read index1\n");
        std::thread thread1(readIndexOnCPU, std::ref(index), sti_path, 0);

        if(use_good_numa) {
            fprintf(stderr, "read index2\n");
            std::thread thread2(readIndexOnCPU, std::ref(index2), sti_path, totalCPUs / 2);
            thread2.join();
        }
        thread1.join();


        logger.debug() << "Bits used to index buckets: " << index.get_bits() << "\n";
        logger.info() << "Total time reading index: " << read_index_timer.elapsed() << " s\n";
    } else {
        logger.debug() << "Bits used to index buckets: " << index.get_bits() << "\n";
        logger.info() << "Indexing ...\n";
        Timer index_timer;
        index.populate(opt.f, opt.n_threads);
        //index.populate_fast(opt.f, opt.n_threads);
        
        logger.info() << "  Time counting seeds: " << index.stats.elapsed_counting_hashes.count() << " s" <<  std::endl;
        logger.info() << "  Time generating seeds: " << index.stats.elapsed_generating_seeds.count() << " s" <<  std::endl;
        logger.info() << "  Time sorting seeds: " << index.stats.elapsed_sorting_seeds.count() << " s" <<  std::endl;
        logger.info() << "  Time generating hash table index: " << index.stats.elapsed_hash_index.count() << " s" <<  std::endl;
        logger.info() << "Total time indexing: " << index_timer.elapsed() << " s\n";

        logger.debug()
            << "Index statistics\n"
            << "  Total strobemers:    " << std::setw(14) << index.stats.tot_strobemer_count << '\n'
            << "  Distinct strobemers: " << std::setw(14) << index.stats.distinct_strobemers << " (100.00%)\n"
            << "    1 occurrence:      " << std::setw(14) << index.stats.tot_occur_once
                << " (" << std::setw(6) << (100.0 * index.stats.tot_occur_once / index.stats.distinct_strobemers) << "%)\n"
            << "    2..100 occurrences:" << std::setw(14) << index.stats.tot_mid_ab
                << " (" << std::setw(6) << (100.0 * index.stats.tot_mid_ab / index.stats.distinct_strobemers) << "%)\n"
            << "    >100 occurrences:  " << std::setw(14) << index.stats.tot_high_ab
                << " (" << std::setw(6) << (100.0 * index.stats.tot_high_ab / index.stats.distinct_strobemers) << "%)\n"
            ;
        if (index.stats.tot_high_ab >= 1) {
            logger.debug() << "Ratio distinct to highly abundant: " << index.stats.distinct_strobemers / index.stats.tot_high_ab << std::endl;
        }
        if (index.stats.tot_mid_ab >= 1) {
            logger.debug() << "Ratio distinct to non distinct: " << index.stats.distinct_strobemers / (index.stats.tot_high_ab + index.stats.tot_mid_ab) << std::endl;
        }
        logger.debug() << "Filtered cutoff index: " << index.stats.index_cutoff << std::endl;
        logger.debug() << "Filtered cutoff count: " << index.stats.filter_cutoff << std::endl;
        
        if (!opt.logfile_name.empty()) {
            index.print_diagnostics(opt.logfile_name, index_parameters.syncmer.k);
            logger.debug() << "Finished printing log stats" << std::endl;
        }
        if (opt.only_gen_index) {
            Timer index_writing_timer;
            std::string sti_path = opt.ref_filename + index_parameters.filename_extension();
            logger.info() << "Writing index to " << sti_path << '\n';
            index.write(opt.ref_filename + index_parameters.filename_extension());
            logger.info() << "Total time writing index: " << index_writing_timer.elapsed() << " s\n";
            return EXIT_SUCCESS;
        }
    }

    // Map/align reads
        
    Timer map_align_timer;
    map_param.rescue_cutoff = map_param.rescue_level < 100 ? map_param.rescue_level * index.filter_cutoff : 1000;
    logger.debug() << "Using rescue cutoff: " << map_param.rescue_cutoff << std::endl;

    std::streambuf* buf;
    std::ofstream of;

    if (!opt.write_to_stdout) {
        of.open(opt.output_file_name);
        buf = of.rdbuf();
    }
    else {
        buf = std::cout.rdbuf();
    }

    std::ostream out(buf);

    if (map_param.is_sam_out) {
        std::stringstream cmd_line;
        for(int i = 0; i < argc; ++i) {
            cmd_line << argv[i] << " ";
        }

        out << sam_header(references, opt.read_group_id, opt.read_group_fields, cmd_line.str());
    }

    std::vector<AlignmentStatistics> log_stats_vec(opt.n_threads);

    logger.info() << "Running in " << (opt.is_SE ? "single-end" : "paired-end") << " mode" << std::endl;

    OutputBuffer output_buffer(out);

    uint64_t num_bytes = 24 * 1024ll * 1024ll * 1024ll;
    uint64_t seed = 13;
    for (int i = 0; i < gpu_num && i < ceil(opt.n_threads / 18.0); i++) {
        set_scores(i, aln_params.match, aln_params.mismatch, aln_params.gap_open, aln_params.gap_extend);
    }
#ifdef use_gpu_align
    for (int i = 0; i < gpu_num && i < ceil(opt.n_threads / 18.0); i++) {
        init_shared_data(references, index, i, 0);
        init_mm_safe(num_bytes, seed, i);
    }
    for (int i = 0; i < opt.n_threads; i++) {
        if (assignments[i].flag == 1 && assignments[i].pass == 0) {
            init_global_big_data(i, assignments[i].gpu_id, map_param.max_tries);
        }
    }
#endif

    double tt0 = GetTime();

    std::vector<std::thread> workers;
    std::vector<int> worker_done(opt.n_threads);  // each thread sets its entry to 1 when it’s done



#ifdef RABBIT_FX

    if(opt.is_SE) {
        //SE
        fprintf(stderr, "SE module RABBIT_FX\n");
        if(use_good_numa) {
            for (int i = 0; i < opt.n_threads / 2; ++i) {
                std::thread consumer(perform_task_async_se_fx, std::ref(input_buffer), std::ref(output_buffer),
                        std::ref(log_stats_vec[i]), std::ref(worker_done[i]), std::ref(aln_params),
                        std::ref(map_param), std::ref(index_parameters), std::ref(references),
                        std::ref(index), std::ref(opt.read_group_id), i, 
                        std::ref(fastqPool), std::ref(queue_se), use_good_numa);
                workers.push_back(std::move(consumer));
            }
            for (int i = opt.n_threads / 2; i < opt.n_threads; ++i) {
                std::thread consumer(perform_task_async_se_fx, std::ref(input_buffer), std::ref(output_buffer),
                        std::ref(log_stats_vec[i]), std::ref(worker_done[i]), std::ref(aln_params),
                        std::ref(map_param), std::ref(index_parameters), std::ref(references),
                        std::ref(index2), std::ref(opt.read_group_id), totalCPUs / 2 + i - opt.n_threads / 2, 
                        std::ref(fastqPool), std::ref(queue_se), use_good_numa);
                workers.push_back(std::move(consumer));
            }
        } else {
            for (int i = 0; i < opt.n_threads; ++i) {
                std::thread consumer(perform_task_async_se_fx, std::ref(input_buffer), std::ref(output_buffer),
                        std::ref(log_stats_vec[i]), std::ref(worker_done[i]), std::ref(aln_params),
                        std::ref(map_param), std::ref(index_parameters), std::ref(references),
                        std::ref(index), std::ref(opt.read_group_id), i, 
                        std::ref(fastqPool), std::ref(queue_se), use_good_numa);
                workers.push_back(std::move(consumer));
            }
        }

    } else {
        //PE
        fprintf(stderr, "PE module RABBIT_FX\n");

        if(use_good_numa) {

#ifdef use_gpu_align
            for (int i = 0; i < opt.n_threads * 1 / 2; ++i) {
                if (assignments[i].flag) {
                    if (assignments[i].pass) continue;
                    std::thread consumer(perform_task_async_pe_fx_GPU, std::ref(input_buffer), std::ref(output_buffer),
                            std::ref(log_stats_vec[i]), std::ref(worker_done[i]), std::ref(aln_params),
                            std::ref(map_param), std::ref(index_parameters), std::ref(references),
                            std::ref(index), std::ref(opt.read_group_id), i,
                            std::ref(fastqPool), std::ref(queue_pe), use_good_numa, assignments[i].gpu_id);
                    workers.push_back(std::move(consumer));
                } else {
                    std::thread consumer(perform_task_async_pe_fx, std::ref(input_buffer), std::ref(output_buffer),
                            std::ref(log_stats_vec[i]), std::ref(worker_done[i]), std::ref(aln_params),
                            std::ref(map_param), std::ref(index_parameters), std::ref(references),
                            std::ref(index), std::ref(opt.read_group_id), i,
                            std::ref(fastqPool), std::ref(queue_pe), use_good_numa, assignments[i].gpu_id);
                    workers.push_back(std::move(consumer));
                }
            }
            for (int i = opt.n_threads * 1 / 2; i < opt.n_threads; ++i) {
                if (assignments[i].flag) {
                    if (assignments[i].pass) continue;
                    std::thread consumer(perform_task_async_pe_fx_GPU, std::ref(input_buffer), std::ref(output_buffer),
                            std::ref(log_stats_vec[i]), std::ref(worker_done[i]), std::ref(aln_params),
                            std::ref(map_param), std::ref(index_parameters), std::ref(references),
                            std::ref(index2), std::ref(opt.read_group_id), i,
                            std::ref(fastqPool), std::ref(queue_pe), use_good_numa, assignments[i].gpu_id);
                    workers.push_back(std::move(consumer));
                } else {
                    std::thread consumer(perform_task_async_pe_fx, std::ref(input_buffer), std::ref(output_buffer),
                            std::ref(log_stats_vec[i]), std::ref(worker_done[i]), std::ref(aln_params),
                            std::ref(map_param), std::ref(index_parameters), std::ref(references),
                            std::ref(index2), std::ref(opt.read_group_id), i,
                            std::ref(fastqPool), std::ref(queue_pe), use_good_numa, assignments[i].gpu_id);
                    workers.push_back(std::move(consumer));
                }
            }
#else
            for (int i = 0; i < opt.n_threads * 1 / 2; ++i) {
                std::thread consumer(perform_task_async_pe_fx, std::ref(input_buffer), std::ref(output_buffer),
                        std::ref(log_stats_vec[i]), std::ref(worker_done[i]), std::ref(aln_params),
                        std::ref(map_param), std::ref(index_parameters), std::ref(references),
                        std::ref(index), std::ref(opt.read_group_id), i,
                        std::ref(fastqPool), std::ref(queue_pe), use_good_numa, assignments[i].gpu_id);
                workers.push_back(std::move(consumer));
            }
            for (int i = opt.n_threads * 1 / 2; i < opt.n_threads; ++i) {
                std::thread consumer(perform_task_async_pe_fx, std::ref(input_buffer), std::ref(output_buffer),
                        std::ref(log_stats_vec[i]), std::ref(worker_done[i]), std::ref(aln_params),
                        std::ref(map_param), std::ref(index_parameters), std::ref(references),
                        std::ref(index2), std::ref(opt.read_group_id), i,
                        std::ref(fastqPool), std::ref(queue_pe), use_good_numa, assignments[i].gpu_id);
                workers.push_back(std::move(consumer));
            }
        
#endif
        } else {
#ifdef use_gpu_align
            assert(false);
#else
            assert(false);
#endif
        }
    }
    if (opt.show_progress && isatty(2)) {
        show_progress_until_done(worker_done, log_stats_vec);
    }
    for (auto& worker : workers) {
        worker.join();
    }
    producer->join();
    delete producer;


#else

    if(opt.is_SE) {
        //SE
        fprintf(stderr, "SE module\n");
        if(use_good_numa) {
            for (int i = 0; i < opt.n_threads / 2; ++i) {
                std::thread consumer(perform_task_async_se, std::ref(input_buffer), std::ref(output_buffer),
                        std::ref(log_stats_vec[i]), std::ref(worker_done[i]), std::ref(aln_params),
                        std::ref(map_param), std::ref(index_parameters), std::ref(references),
                        std::ref(index), std::ref(opt.read_group_id), i, use_good_numa);
                workers.push_back(std::move(consumer));
            }
            for (int i = opt.n_threads / 2; i < opt.n_threads; ++i) {
                std::thread consumer(perform_task_async_se, std::ref(input_buffer), std::ref(output_buffer),
                        std::ref(log_stats_vec[i]), std::ref(worker_done[i]), std::ref(aln_params),
                        std::ref(map_param), std::ref(index_parameters), std::ref(references),
                        std::ref(index2), std::ref(opt.read_group_id), totalCPUs / 2 + i - opt.n_threads / 2, use_good_numa);
                workers.push_back(std::move(consumer));
            }
        } else {
            for (int i = 0; i < opt.n_threads; ++i) {
                std::thread consumer(perform_task_async_se, std::ref(input_buffer), std::ref(output_buffer),
                        std::ref(log_stats_vec[i]), std::ref(worker_done[i]), std::ref(aln_params),
                        std::ref(map_param), std::ref(index_parameters), std::ref(references),
                        std::ref(index), std::ref(opt.read_group_id), i, use_good_numa);
                workers.push_back(std::move(consumer));
            }
        }
    } else {
        //PE
        fprintf(stderr, "PE module\n");
        if(use_good_numa) {
            for (int i = 0; i < opt.n_threads / 2; ++i) {
                std::thread consumer(perform_task_async_pe, std::ref(input_buffer), std::ref(output_buffer),
                        std::ref(log_stats_vec[i]), std::ref(worker_done[i]), std::ref(aln_params),
                        std::ref(map_param), std::ref(index_parameters), std::ref(references),
                        std::ref(index), std::ref(opt.read_group_id), i, use_good_numa);
                workers.push_back(std::move(consumer));
            }
            for (int i = opt.n_threads / 2; i < opt.n_threads; ++i) {
                std::thread consumer(perform_task_async_pe, std::ref(input_buffer), std::ref(output_buffer),
                        std::ref(log_stats_vec[i]), std::ref(worker_done[i]), std::ref(aln_params),
                        std::ref(map_param), std::ref(index_parameters), std::ref(references),
                        std::ref(index2), std::ref(opt.read_group_id), totalCPUs / 2 + i - opt.n_threads / 2, use_good_numa);
                workers.push_back(std::move(consumer));
            }
        } else{
            for (int i = 0; i < opt.n_threads; ++i) {
                std::thread consumer(perform_task_async_pe, std::ref(input_buffer), std::ref(output_buffer),
                        std::ref(log_stats_vec[i]), std::ref(worker_done[i]), std::ref(aln_params),
                        std::ref(map_param), std::ref(index_parameters), std::ref(references),
                        std::ref(index), std::ref(opt.read_group_id), i, use_good_numa);
                workers.push_back(std::move(consumer));
            }
        }
    }
    if (opt.show_progress && isatty(2)) {
        show_progress_until_done(worker_done, log_stats_vec);
    }
    for (auto& worker : workers) {
        worker.join();
    }

#endif

    logger.info() << "Done!\n";
    fprintf(stderr, "consumer cost %.2f\n", GetTime() - tt0);

    AlignmentStatistics tot_statistics;
    for (auto& it : log_stats_vec) {
        tot_statistics += it;
    }

    logger.info() << "Total mapping sites tried: " << tot_statistics.tot_all_tried << std::endl
        << "Total calls to ssw: " << tot_statistics.tot_aligner_calls << std::endl
        << "Inconsistent NAM ends: " << tot_statistics.inconsistent_nams << std::endl
        << "Tried NAM rescue: " << tot_statistics.nam_rescue << std::endl
        << "Mates rescued by alignment: " << tot_statistics.tot_rescued << std::endl
        << "Total time mapping: " << map_align_timer.elapsed() << " s." << std::endl
        << "Total time reading read-file(s): " << tot_statistics.tot_read_file.count() / opt.n_threads << " s." << std::endl
        << "Total time creating strobemers: " << tot_statistics.tot_construct_strobemers.count() / opt.n_threads << " s." << std::endl
        << "Total time finding NAMs (non-rescue mode): " << tot_statistics.tot_find_nams.count() / opt.n_threads << " s." << std::endl
        << "Total time finding NAMs (rescue mode): " << tot_statistics.tot_time_rescue.count() / opt.n_threads << " s." << std::endl;
    //<< "Total time finding NAMs ALTERNATIVE (candidate sites): " << tot_find_nams_alt.count()/opt.n_threads  << " s." <<  std::endl;
    logger.info() << "Total time sorting NAMs (candidate sites): " << tot_statistics.tot_sort_nams.count() / opt.n_threads << " s." << std::endl
        << "Total time base level alignment (ssw): " << tot_statistics.tot_extend.count() / opt.n_threads << " s." << std::endl
        << "Total time writing alignment to files: " << tot_statistics.tot_write_file.count() << " s." << std::endl;
    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    try {
        return run_rabbitsalign(argc, argv);
    } catch (BadParameter& e) {
        logger.error() << "A parameter is invalid: " << e.what() << std::endl;
    } catch (const std::runtime_error& e) {
        logger.error() << "rabbitsalign: " << e.what() << std::endl;
    }
    return EXIT_FAILURE;
}
