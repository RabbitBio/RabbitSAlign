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
#include <unordered_set>

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
#include "gpu_pipeline.h"
#include "FastxStream.h"
#include "FastxChunk.h"
#include "DataQueue.h"
#include "Formater.h"

// --- Static variables and constants ---

// Global logger instance
static Logger& logger = Logger::get();

// Constant definitions for memory sizes and buffer settings
constexpr uint64_t GB_BYTE = 1024 * 1024 * 1024;
constexpr int META_DATA_SIZE = 168; // Metadata size per read in GPU memory
constexpr double GALLATIN_BASE_SIZE = 2.0 * GB_BYTE; // Base size for Gallatin GPU memory
constexpr double GALLATIN_CHUNK_SIZE = 1.0 * GB_BYTE; // Size of each chunk in Gallatin GPU memory
constexpr int OVERLAP_SIZE = 3; // Using triple buffering
int G_num = 2; // Number of type B threads per GPU

// Configurable stream processing sizes
uint64_t STREAM_BATCH_SIZE = 1024ll;
uint64_t STREAM_BATCH_SIZE_GPU = 4096ll;
uint64_t MAX_QUERY_LEN = 600ll;
uint64_t MAX_TARGET_LEN = 1000ll;


// --- System Information Functions ---

/**
 * @brief Gets the number of NUMA nodes in the system.
 * @details Reads /sys/devices/system/node/possible to determine the NUMA configuration.
 * @return The number of NUMA nodes. Returns 1 if detection fails.
 */
int getNumaNodeCount() {
    std::ifstream file("/sys/devices/system/node/possible");
    if (!file.is_open()) {
        return 1; // Assume a single NUMA node if the file cannot be opened
    }

    std::string line;
    if (std::getline(file, line)) {
        size_t hyphen_pos = line.find('-');
        if (hyphen_pos == std::string::npos) {
            try {
                // Format is a single number, e.g., "0"
                int max_node = std::stoi(line);
                return max_node + 1;
            } catch (const std::exception&) {
                return 1;
            }
        } else {
            try {
                // Format is a range, e.g., "0-3"
                int max_node = std::stoi(line.substr(hyphen_pos + 1));
                return max_node + 1;
            } catch (const std::exception&) {
                return 1;
            }
        }
    }
    return 1;
}

/**
 * @brief Gets the amount of available memory in the system.
 * @details Reads /proc/meminfo and parses the "MemAvailable" field.
 * @return The available memory in bytes. Returns -1 on failure.
 */
long long getAvailableMemory() {
    std::ifstream meminfo("/proc/meminfo");
    if (!meminfo.is_open()) {
        return -1;
    }
    std::string line;
    while (std::getline(meminfo, line)) {
        if (line.rfind("MemAvailable:", 0) == 0) {
            std::istringstream iss(line);
            std::string key;
            long long value;
            iss >> key >> value;
            return value * 1024; // Convert from kB to Bytes
        }
    }
    return -1;
}


// --- SAM Formatting and Parameter Logging ---

/**
 * @brief Generates a SAM format header string.
 * @param references The reference sequences.
 * @param read_group_id The read group ID.
 * @param read_group_fields Additional read group fields.
 * @param cmd_line The command line used to execute the program.
 * @return A formatted SAM header as a string.
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

/**
 * @brief Issues a warning if the binary was compiled in Debug mode without optimizations.
 */
void warn_if_no_optimizations() {
    if (std::string(CMAKE_BUILD_TYPE) == "Debug") {
        logger.info() << "\n    ***** Binary was compiled without optimizations - this will be very slow *****\n\n";
    }
}

/**
 * @brief Logs the indexing, mapping, and alignment parameters.
 * @param index_parameters Parameters for indexing.
 * @param map_param Parameters for mapping.
 * @param aln_params Parameters for alignment.
 */
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


// --- Thread and Task Management ---

/**
 * @brief Checks if AVX2 instruction set is enabled at compile time.
 * @return True if AVX2 is enabled, false otherwise.
 */
bool avx2_enabled() {
#ifdef __AVX2__
    return true;
#else
    return false;
#endif
}

/**
 * @brief Creates an InputBuffer based on command-line options.
 * @param opt The parsed command line options.
 * @return An initialized InputBuffer object.
 * @throws BadParameter if options are inconsistent (e.g., --interleaved with two files).
 */
InputBuffer get_input_buffer(const CommandLineOptions& opt) {
    if (opt.is_SE) {
        return InputBuffer(opt.reads_filename1, "", opt.chunk_size, false);
    } else if (opt.is_interleaved) {
        if (!opt.reads_filename2.empty()) {
            throw BadParameter("Cannot specify both --interleaved and specify two read files");
        }
        return InputBuffer(opt.reads_filename1, "", opt.chunk_size, true);
    } else {
        return InputBuffer(opt.reads_filename1, opt.reads_filename2, opt.chunk_size, false);
    }
}

/**
 * @brief Displays a progress indicator to stderr until all worker threads are done.
 * @param worker_done A vector of flags indicating if each worker thread has finished.
 * @param stats A vector of alignment statistics from each thread.
 */
void show_progress_until_done(const std::vector<int>& worker_done, const std::vector<AlignmentStatistics>& stats) {
    Timer timer;
    bool reported = false;
    auto time_to_wait = std::chrono::milliseconds(1);

    while (true) {
        std::this_thread::sleep_for(time_to_wait);
        // Exponentially increase wait time up to a maximum of 1 second
        time_to_wait = std::min(time_to_wait * 2, std::chrono::milliseconds(1000));

        bool all_done = true;
        for (int is_done : worker_done) {
            if (!is_done) {
                all_done = false;
                break;
            }
        }
        if (all_done) {
            break;
        }

        uint64_t n_reads = 0;
        for (const auto& stat : stats) {
            n_reads += stat.n_reads;
        }

        auto elapsed = timer.elapsed();
        if (elapsed >= 1.0 && n_reads > 0) {
            std::cerr << " Mapped " << std::setw(12) << (n_reads / 1E6) << " M reads @ "
                      << std::setw(8) << (elapsed * 1E6 / n_reads) << " us/read                   \r";
            reported = true;
        }
    }

    if (reported) {
        std::cerr << '\n';
    }
}

/**
 * @brief Sets the CPU affinity for the current thread.
 * @param cpu_id The ID of the CPU core to bind the thread to.
 */
void setThreadAffinity(int cpu_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);

    pthread_t current_thread = pthread_self();
    if (pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) != 0) {
        std::cerr << "Error setting thread affinity for CPU " << cpu_id << std::endl;
    }
}

/**
 * @brief Reads the strobemer index on a specific CPU core.
 * @param index The StrobemerIndex object to populate.
 * @param sti_path The path to the index file.
 * @param cpu_id The ID of the CPU to run this task on.
 */
void readIndexOnCPU(StrobemerIndex& index, const std::string& sti_path, int cpu_id) {
    setThreadAffinity(cpu_id);
    index.read(sti_path);
}

/**
 * @brief Producer task for reading paired-end FASTQ data.
 * @details Reads chunks of paired-end reads and pushes them to a data queue.
 * @return 0 on completion.
 */
int producer_pe_fastq_task(std::string file, std::string file2, rabbit::fq::FastqDataPool& fastqPool, rabbit::core::TDataQueue<rabbit::fq::FastqDataPairChunk>& dq, int nxtSize) {
    rabbit::fq::FastqFileReader fqFileReader(file, fastqPool, false, file2, nxtSize);
    int n_chunks = 0;
    while (true) {
        // Read the next chunk of paired-end data
        rabbit::fq::FastqDataPairChunk* fqdatachunk = fqFileReader.readNextPairChunk();
        if (fqdatachunk == nullptr) {
            break; // End of file
        }
        dq.Push(n_chunks, fqdatachunk);
        n_chunks++;
    }
    dq.SetCompleted();
    std::cerr << "file " << file << " has " << n_chunks << " chunks" << std::endl;
    return 0;
}

/**
 * @brief Producer task for reading single-end FASTQ data.
 * @details Reads chunks of single-end reads and pushes them to a data queue.
 * @return 0 on completion.
 */
int producer_se_fastq_task(std::string file, rabbit::fq::FastqDataPool& fastqPool, rabbit::core::TDataQueue<rabbit::fq::FastqDataChunk>& dq, int nxtSize) {
    rabbit::fq::FastqFileReader fqFileReader(file, fastqPool, false, "", nxtSize);
    rabbit::int64 n_chunks = 0;
    while (true) {
        // Read the next chunk of single-end data
        rabbit::fq::FastqDataChunk* fqdatachunk = fqFileReader.readNextChunk();
        if (fqdatachunk == nullptr) {
            break; // End of file
        }
        dq.Push(n_chunks, fqdatachunk);
        n_chunks++;
    }
    dq.SetCompleted();
    std::cerr << "file " << file << " has " << n_chunks << " chunks" << std::endl;
    return 0;
}


// --- Memory Usage Calculation ---

/**
 * @brief Calculates the estimated memory usage for single-end mode.
 * @return The estimated memory usage in bytes.
 */
uint64_t calculateMemoryUsageSE(int total_cpu_num, int gpu_num, int chunk_num, int eval_read_len, int chunk_size, int max_tries, uint64_t ref_index_size) {
    int total_read_num = chunk_num * chunk_size / 2 / eval_read_len;
    int total_read_len = total_read_num * eval_read_len;

    uint64_t align_res_item_size = 4 * sizeof(int) + 2 * sizeof(Nam) + sizeof(GPUAlignment) + sizeof(CigarData) + sizeof(TODOInfos);
    uint64_t align_res_vec_size = total_read_num * (max_tries * 2 + 2) * align_res_item_size;
    uint64_t align_res_total_size = total_read_num * 2 * sizeof(GPUAlignTmpRes) * OVERLAP_SIZE + align_res_vec_size * OVERLAP_SIZE;
    uint64_t meta_data_size = total_read_num * META_DATA_SIZE;
    uint64_t seq_data_size = total_read_num * 4 + total_read_len * 2;
    uint64_t device_todo_mem_size = chunk_num * DEVICE_TODO_SIZE_PER_CHUNK * 3;

    int C_num = total_cpu_num / gpu_num - G_num;
    double SIZE0 = G_num * MAX_GPU_SSW_SIZE + C_num * MAX_CPU_SSW_SIZE;
    double SIZE1 = G_num * (align_res_total_size + meta_data_size + seq_data_size + device_todo_mem_size);
    double SIZE2 = GALLATIN_BASE_SIZE + chunk_num * GALLATIN_CHUNK_SIZE * 0.75;
    double SIZE3 = ref_index_size;

    return static_cast<uint64_t>(SIZE0 + SIZE1 + SIZE2 + SIZE3);
}

/**
 * @brief Calculates the estimated memory usage for paired-end mode.
 * @return The estimated memory usage in bytes.
 */
uint64_t calculateMemoryUsagePE(int total_cpu_num, int gpu_num, int chunk_num, int eval_read_len, int chunk_size, int max_tries, uint64_t ref_index_size) {
    int total_read_num = chunk_num * chunk_size / 2 / eval_read_len;
    int total_read_len = total_read_num * eval_read_len;

    uint64_t align_res_item_size = 4 * sizeof(int) + 2 * sizeof(Nam) + sizeof(GPUAlignment) + sizeof(CigarData) + sizeof(TODOInfos);
    uint64_t align_res_vec_size = total_read_num * (max_tries * 2 + 2) * align_res_item_size;
    uint64_t align_res_total_size = total_read_num * 2 * sizeof(GPUAlignTmpRes) * OVERLAP_SIZE + align_res_vec_size * OVERLAP_SIZE;
    uint64_t meta_data_size = total_read_num * META_DATA_SIZE;
    uint64_t seq_data_size = total_read_num * 8 + total_read_len * 4;
    uint64_t device_todo_mem_size = chunk_num * DEVICE_TODO_SIZE_PER_CHUNK * 3;

    int C_num = total_cpu_num / gpu_num - G_num;
    double SIZE0 = G_num * MAX_GPU_SSW_SIZE + C_num * MAX_CPU_SSW_SIZE;
    double SIZE1 = G_num * (align_res_total_size + meta_data_size + seq_data_size + device_todo_mem_size);
    double SIZE2 = GALLATIN_BASE_SIZE + chunk_num * GALLATIN_CHUNK_SIZE;
    double SIZE3 = ref_index_size;

    return static_cast<uint64_t>(SIZE0 + SIZE1 + SIZE2 + SIZE3);
}

/**
 * @brief Calculates the maximum number of chunks that can be processed given the available GPU memory.
 * @return The maximum valid number of chunks.
 */
int calculateMaxChunkNum(int total_cpu_num, int gpu_num, uint64_t GPU_mem_size, bool is_se, int eval_read_len, int chunk_size, int max_tries, uint64_t ref_index_size) {
    printf("Calculating max chunk_num for total_cpu_num: %d, gpu_num: %d, GPU_mem_size: %llu GB, is_se: %d, eval_read_len: %d, chunk_size: %d, max_tries: %d, ref_index_size: %llu GB\n",
           total_cpu_num, gpu_num, GPU_mem_size / GB_BYTE, is_se, eval_read_len, chunk_size, max_tries, ref_index_size / GB_BYTE);

    for (int chunk_num = 1; chunk_num <= 96; ++chunk_num) {
        uint64_t memory_usage = is_se
                                ? calculateMemoryUsageSE(total_cpu_num, gpu_num, chunk_num, eval_read_len, chunk_size, max_tries, ref_index_size)
                                : calculateMemoryUsagePE(total_cpu_num, gpu_num, chunk_num, eval_read_len, chunk_size, max_tries, ref_index_size);

        if (memory_usage > GPU_mem_size) {
            return chunk_num - 1;
        }
    }
    return 96; // Return the maximum tested value if all fit
}


// --- Thread Assignment ---

/**
 * @brief Represents the assignment of a thread to specific resources (CPU, GPU, NUMA node).
 */
struct ThreadAssignment {
    int thread_id;
    int async_thread_id;
    int numa_node;
    int gpu_id;
    int flag; // Flag to indicate special roles (e.g., GPU thread)
    int pass; // Flag to indicate if the thread should be skipped (e.g., auxiliary GPU thread)
};

/**
 * @brief Evenly selects a specified number of elements from a total range.
 * @param total The total number of items to select from.
 * @param count The number of items to select.
 * @return A vector of selected indices.
 */
std::vector<int> evenly_select(int total, int count) {
    std::vector<int> result;
    if (count <= 0 || total <= 0) {
        return result;
    }
    if (count > total) {
        count = total;
    }

    double step = static_cast<double>(total) / count;
    for (int i = 0; i < count; ++i) {
        result.push_back(static_cast<int>(i * step));
    }
    return result;
}

/**
 * @brief Assigns threads to CPUs and GPUs based on system topology and user requests.
 * @details This function creates a mapping of threads to resources, flagging some for GPU tasks.
 * @return A vector of ThreadAssignment structs.
 */
std::vector<ThreadAssignment> assign_threads_fixed_with_flags(int total_cpu_num, int total_gpu_num, int cpu_num, int gpu_num, int numa_num) {
    std::cout << "total_cpu_num: " << total_cpu_num << ", use " << cpu_num << std::endl;
    std::cout << "total_gpu_num: " << total_gpu_num << ", use " << gpu_num << std::endl;

    std::vector<ThreadAssignment> assignments;
    std::vector<int> selected_threads = evenly_select(total_cpu_num, cpu_num);
    std::vector<int> main_gpu_indices = evenly_select(cpu_num, gpu_num * G_num);
    std::unordered_set<int> main_gpu_tids;
    std::unordered_set<int> aux_gpu_tids;

    for (int i : main_gpu_indices) {
        if (i < cpu_num) {
            int main_tid = selected_threads[i];
            main_gpu_tids.insert(main_tid);
            if (i + 1 < cpu_num) {
                int aux_tid = selected_threads[i + 1];
                aux_gpu_tids.insert(aux_tid);
            }
        }
    }

    int gpu_id_counter = 0;

    for (int i = 0; i < selected_threads.size(); ++i) {
        int tid = selected_threads[i];
        int numa_node = tid / (total_cpu_num / numa_num);
        int gpu_id = tid / (total_cpu_num / gpu_num);

        ThreadAssignment ta = {tid, -1, numa_node, gpu_id, 0, 0};

        if (main_gpu_tids.count(tid)) {
            ta.flag = 1;
            ta.async_thread_id = selected_threads[i + 1];
            ta.gpu_id = gpu_id_counter / 2;
            gpu_id_counter++;
//            std::cout << "Assign GPU main thread " << tid << " (GPU " << ta.gpu_id << ") with aux " << ta.async_thread_id << "\n";
        } else if (aux_gpu_tids.count(tid)) {
            ta.flag = 1;
            ta.pass = 1;
//            std::cout << "Assign GPU aux thread " << tid << "\n";
        }

        assignments.push_back(ta);
    }

    return assignments;
}


// --- Main Application Logic ---

/**
 * @brief Sets the alignment scores for a specific GPU.
 * @param gpu_id The ID of the target GPU.
 * @param match_score Score for a match.
 * @param mismatch_score Penalty for a mismatch.
 * @param gap_open_score Penalty for opening a gap.
 * @param gap_extend_score Penalty for extending a gap.
 */
void set_scores(int gpu_id, int match_score, int mismatch_score, int gap_open_score, int gap_extend_score) {
    cudaSetDevice(gpu_id);
    gasal_subst_scores sub_scores;
    sub_scores.match = match_score;
    sub_scores.mismatch = mismatch_score;
    sub_scores.gap_open = gap_open_score - 1;
    sub_scores.gap_extend = gap_extend_score;
    gasal_copy_subst_scores(&sub_scores);
}

/**
 * @brief Main function to run the rabbitsalign pipeline.
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on error.
 */
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

    int eval_read_len = opt.r;
#if !defined(CPU_ACC_TAG) && !defined(GPU_ACC_TAG)
    MAX_QUERY_LEN = eval_read_len + 100;
//    MAX_TARGET_LEN = eval_read_len * 4;
#endif
    logger.info() << "MAX_QUERY_LEN: " << MAX_QUERY_LEN << ", MAX_TARGET_LEN: " << MAX_TARGET_LEN << std::endl;

    G_num = opt.threads_per_gpu;

    int fx_batch_size = 1 << 22;
    int nxt_size = fx_batch_size / 4;
    if (eval_read_len <= 500) {
        nxt_size = 1 << 12;
    }

    rabbit::fq::FastqDataPool fastqPool(4096, fx_batch_size);
    rabbit::core::TDataQueue<rabbit::fq::FastqDataChunk> queue_se(4096, 1);
    rabbit::core::TDataQueue<rabbit::fq::FastqDataPairChunk> queue_pe(4096, 1);

    std::thread producer_thread;
    if (!opt.only_gen_index) {
        if (opt.is_SE || opt.is_interleaved) {
            producer_thread = std::thread(producer_se_fastq_task, opt.reads_filename1, std::ref(fastqPool), std::ref(queue_se), nxt_size);
        } else {
            producer_thread = std::thread(producer_pe_fastq_task, opt.reads_filename1, opt.reads_filename2, std::ref(fastqPool), std::ref(queue_pe), nxt_size);
        }
    }

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

    log_parameters(index_parameters, map_param, aln_params);
    logger.info() << "CPU Threads: " << opt.n_threads << std::endl;
    logger.info() << "GPU Numbers: " << opt.n_gpus << std::endl;

    if (opt.n_gpus <= 0) {
        logger.error() << "No GPUs specified. Please specify at least one GPU using -g option.\n";
        return 1;
    }

    // Load reference genome
    References references;
    Timer read_refs_timer;
    references = References::from_fasta(opt.ref_filename);
    logger.info() << "Time reading reference: " << read_refs_timer.elapsed() << " s\n";

    logger.info() << "Reference size: " << references.total_length() / 1E6 << " Mbp ("
                  << references.size() << " contig" << (references.size() == 1 ? "" : "s")
                  << "; largest: "
                  << (*std::max_element(references.lengths.begin(), references.lengths.end()) / 1E6) << " Mbp)\n";
    if (references.total_length() == 0) {
        throw InvalidFasta("No reference sequences found");
    }

    // NUMA-awareness logic
    int numa_num = getNumaNodeCount();
    long long mem_avail = getAvailableMemory();
    bool use_good_numa = true;
#ifdef OPT_NUMA_CLOSE
    use_good_numa = false;
#endif
    if (numa_num * (15ll << 30) > mem_avail * 0.8) {
        use_good_numa = false;
    }
    if (numa_num == 1) {
        use_good_numa = false;
    }
    if (!opt.use_index) {
        use_good_numa = false;
    }
    fprintf(stderr, "use_good_numa is %d\n", use_good_numa);

    StrobemerIndex index(references, index_parameters, opt.bits);
    StrobemerIndex index2(references, index_parameters, opt.bits);
    int totalCPUs = std::thread::hardware_concurrency();
    int totalGPUs = 0;
    cudaError_t err = cudaGetDeviceCount(&totalGPUs);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    size_t free_memory, total_memory;
    cudaSetDevice(0); // Check memory on first GPU
    err = cudaMemGetInfo(&free_memory, &total_memory);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    logger.info() << "Total CPUs: " << totalCPUs << ", Total GPUs: " << totalGPUs << std::endl;
    logger.info() << "Available GPU memory: " << free_memory / (1024 * 1024) << " MB, Total GPU memory: " << total_memory / (1024 * 1024) << " MB\n";

    if (opt.use_index) {
        // Read the index from a file
        assert(!opt.only_gen_index);
        Timer read_index_timer;
        std::string sti_path = opt.ref_filename + index_parameters.filename_extension();
        logger.info() << "Reading index from " << sti_path << '\n';

        fprintf(stderr, "read index1\n");
        std::thread thread1(readIndexOnCPU, std::ref(index), sti_path, 0);

        if (use_good_numa) {
            fprintf(stderr, "read index2\n");
            std::thread thread2(readIndexOnCPU, std::ref(index2), sti_path, totalCPUs / 2);
            thread2.join();
        }
        thread1.join();

        logger.debug() << "Bits used to index buckets: " << index.get_bits() << "\n";
        logger.info() << "Total time reading index: " << read_index_timer.elapsed() << " s\n";
    } else {
        // Build the index from the reference
        logger.debug() << "Bits used to index buckets: " << index.get_bits() << "\n";
        logger.info() << "Indexing ...\n";
        Timer index_timer;
        index.populate(opt.f, opt.n_threads);

        logger.info() << "  Time counting seeds: " << index.stats.elapsed_counting_hashes.count() << " s" << std::endl;
        logger.info() << "  Time generating seeds: " << index.stats.elapsed_generating_seeds.count() << " s" << std::endl;
        logger.info() << "  Time sorting seeds: " << index.stats.elapsed_sorting_seeds.count() << " s" << std::endl;
        logger.info() << "  Time generating hash table index: " << index.stats.elapsed_hash_index.count() << " s" << std::endl;
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
            index.write(sti_path);
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
    } else {
        buf = std::cout.rdbuf();
    }
    std::ostream out(buf);

    if (map_param.is_sam_out) {
        std::stringstream cmd_line;
        for (int i = 0; i < argc; ++i) {
            cmd_line << argv[i] << " ";
        }
        out << sam_header(references, opt.read_group_id, opt.read_group_fields, cmd_line.str());
    }

    OutputBuffer output_buffer(out);

    std::vector<AlignmentStatistics> log_stats_vec(opt.n_threads);
    logger.info() << "Running in " << (opt.is_SE ? "single-end" : "paired-end") << " mode" << std::endl;

    uint64_t ref_size = references.size() * sizeof(my_string) + references.total_length() * sizeof(char) + references.size() * sizeof(int);
    uint64_t index_size = index.randstrobes.size() * sizeof(RefRandstrobe) + index.randstrobe_start_indices.size() * sizeof(StrobemerIndex::bucket_index_t);

    int max_chunk_num = calculateMaxChunkNum(opt.n_threads, opt.n_gpus, free_memory, opt.is_SE, eval_read_len, fx_batch_size, map_param.max_tries, ref_size + index_size);
    if (max_chunk_num < 1) {
        logger.error() << "Not enough memory to run rabbitsalign. Please reduce the number of threads or increase available memory.\n";
        return -1;
    }
    int chunk_num = max_chunk_num - (max_chunk_num % 2); // Ensure even number

    uint64_t last_mem_size = 0;
    if (opt.is_SE) {
        auto res = calculateMemoryUsageSE(opt.n_threads, opt.n_gpus, chunk_num, eval_read_len, fx_batch_size, map_param.max_tries, ref_size + index_size);
        last_mem_size = free_memory - 0.5 - res;
    } else {
        auto res = calculateMemoryUsagePE(opt.n_threads, opt.n_gpus, chunk_num, eval_read_len, fx_batch_size, map_param.max_tries, ref_size + index_size);
        last_mem_size = free_memory - 0.5 - res;
    }
    if (last_mem_size < 0) last_mem_size = 0;
    uint64_t gallatin_num_bytes = GALLATIN_BASE_SIZE + chunk_num * GALLATIN_CHUNK_SIZE + last_mem_size;
    uint64_t limited_gallatin_num_bytes = 24ULL * GB_BYTE; // limit to 24 GB
    gallatin_num_bytes = std::min(limited_gallatin_num_bytes, gallatin_num_bytes); // limit to 24 GB

    while (gallatin_num_bytes < 4 * GB_BYTE) {
        STREAM_BATCH_SIZE = STREAM_BATCH_SIZE / 2;
        STREAM_BATCH_SIZE_GPU = STREAM_BATCH_SIZE_GPU / 2;
        max_chunk_num = calculateMaxChunkNum(opt.n_threads, opt.n_gpus, free_memory, opt.is_SE, eval_read_len, fx_batch_size, map_param.max_tries, ref_size + index_size);
        chunk_num = max_chunk_num - (max_chunk_num % 2);
        if (opt.is_SE) {
            auto res = calculateMemoryUsageSE(opt.n_threads, opt.n_gpus, chunk_num, eval_read_len, fx_batch_size, map_param.max_tries, ref_size + index_size);
            last_mem_size = free_memory - 0.5 - res;
        } else {
            auto res = calculateMemoryUsagePE(opt.n_threads, opt.n_gpus, chunk_num, eval_read_len, fx_batch_size, map_param.max_tries, ref_size + index_size);
            last_mem_size = free_memory - 0.5 - res;
        }
        if (last_mem_size < 0) last_mem_size = 0;
        gallatin_num_bytes = GALLATIN_BASE_SIZE + chunk_num * GALLATIN_CHUNK_SIZE + last_mem_size;
        logger.info() << "resize BATCH size to " << STREAM_BATCH_SIZE << " and GPU BATCH size to " << STREAM_BATCH_SIZE_GPU << std::endl;
        logger.info() << "now chunk num is " << chunk_num << " and gallatin_num_bytes is " << gallatin_num_bytes / GB_BYTE << " GB" << std::endl;
    }

    logger.info() << "BATCH size: " << STREAM_BATCH_SIZE << " and GPU BATCH size: " << STREAM_BATCH_SIZE_GPU << std::endl;
    logger.info() << "Additional memory for Gallatin: " << last_mem_size / (1024 * 1024) << " MB\n";

    logger.info() << "Gallatin memory usage: " << gallatin_num_bytes / (1024 * 1024) << " MB\n";

    logger.info() << "Using chunk size: " << chunk_num << std::endl;
    uint64_t gallatin_seed = 13;
    for (int i = 0; i < opt.n_gpus; i++) {
        set_scores(i, aln_params.match, aln_params.mismatch, aln_params.gap_open, aln_params.gap_extend);
    }

    int batch_read_num = chunk_num * fx_batch_size / 2 / eval_read_len;
    int batch_total_read_len = batch_read_num * eval_read_len;
    logger.info() << "Batch read number: " << batch_read_num << std::endl;
    logger.info() << "Batch total read length: " << batch_total_read_len << std::endl;

    std::vector<ThreadAssignment> assignments = assign_threads_fixed_with_flags(totalCPUs, totalGPUs, opt.n_threads, opt.n_gpus, numa_num);

    for (int i = 0; i < opt.n_gpus; i++) {
        init_shared_data(references, index, i, 0);
        init_mm_safe(gallatin_num_bytes, gallatin_seed, i);
    }
    for (size_t i = 0; i < assignments.size(); i++) {
        if (assignments[i].flag == 1 && assignments[i].pass == 0) {
            init_global_big_data(assignments[i].thread_id, assignments[i].gpu_id, map_param.max_tries, batch_read_num);
        }
    }

    double tt0 = GetTime();
    std::vector<std::thread> workers;
    std::vector<int> worker_done(opt.n_threads, 0);

    // Main consumer thread creation loop
    if(opt.is_SE) {
        //SE
        fprintf(stderr, "SE module RABBIT_FX\n");
        if(use_good_numa) {
            for (int i = 0; i < opt.n_threads / 2; ++i) {
                if (assignments[i].flag) {
                    if (assignments[i].pass) {
//                        printf("gpu thread %d skip\n", i);
                        continue;
                    }
                    std::thread consumer(perform_task_async_se_fx_GPU, std::ref(input_buffer), std::ref(output_buffer),
                                         std::ref(log_stats_vec[i]), std::ref(worker_done[i]), std::ref(aln_params),
                                         std::ref(map_param), std::ref(index_parameters), std::ref(references),
                                         std::ref(index), std::ref(opt.read_group_id), assignments[i].thread_id,
                                         std::ref(fastqPool), std::ref(queue_se), use_good_numa, assignments[i].gpu_id, assignments[i].async_thread_id,
                                         batch_read_num, batch_total_read_len, chunk_num, opt.unordered_output);
                    workers.push_back(std::move(consumer));
                } else if (!opt.only_gpu) {
                    std::thread consumer(perform_task_async_se_fx, std::ref(input_buffer), std::ref(output_buffer),
                                         std::ref(log_stats_vec[i]), std::ref(worker_done[i]), std::ref(aln_params),
                                         std::ref(map_param), std::ref(index_parameters), std::ref(references),
                                         std::ref(index), std::ref(opt.read_group_id), assignments[i].thread_id,
                                         std::ref(fastqPool), std::ref(queue_se), use_good_numa, assignments[i].gpu_id);
                    workers.push_back(std::move(consumer));
                }
            }
            for (int i = opt.n_threads / 2; i < opt.n_threads; ++i) {
                if (assignments[i].flag) {
                    if (assignments[i].pass) {
//                        printf("gpu thread %d skip\n", i);
                        continue;
                    }
                    std::thread consumer(perform_task_async_se_fx_GPU, std::ref(input_buffer), std::ref(output_buffer),
                                         std::ref(log_stats_vec[i]), std::ref(worker_done[i]), std::ref(aln_params),
                                         std::ref(map_param), std::ref(index_parameters), std::ref(references),
                                         std::ref(index2), std::ref(opt.read_group_id), assignments[i].thread_id,
                                         std::ref(fastqPool), std::ref(queue_se), use_good_numa, assignments[i].gpu_id, assignments[i].async_thread_id,
                                         batch_read_num, batch_total_read_len, chunk_num, opt.unordered_output);
                    workers.push_back(std::move(consumer));
                } else if (!opt.only_gpu) {
                    std::thread consumer(perform_task_async_se_fx, std::ref(input_buffer), std::ref(output_buffer),
                                         std::ref(log_stats_vec[i]), std::ref(worker_done[i]), std::ref(aln_params),
                                         std::ref(map_param), std::ref(index_parameters), std::ref(references),
                                         std::ref(index2), std::ref(opt.read_group_id), assignments[i].thread_id,
                                         std::ref(fastqPool), std::ref(queue_se), use_good_numa, assignments[i].gpu_id);
                    workers.push_back(std::move(consumer));
                }
            }
        } else {
            for (int i = 0; i < opt.n_threads; ++i) {
                if (assignments[i].flag) {
                    if (assignments[i].pass) {
//                        printf("gpu thread %d skip\n", i);
                        continue;
                    }
                    std::thread consumer(perform_task_async_se_fx_GPU, std::ref(input_buffer), std::ref(output_buffer),
                                         std::ref(log_stats_vec[i]), std::ref(worker_done[i]), std::ref(aln_params),
                                         std::ref(map_param), std::ref(index_parameters), std::ref(references),
                                         std::ref(index), std::ref(opt.read_group_id), assignments[i].thread_id,
                                         std::ref(fastqPool), std::ref(queue_se), use_good_numa, assignments[i].gpu_id, assignments[i].async_thread_id,
                                         batch_read_num, batch_total_read_len, chunk_num, opt.unordered_output);
                    workers.push_back(std::move(consumer));
                } else if (!opt.only_gpu) {
                    std::thread consumer(perform_task_async_se_fx, std::ref(input_buffer), std::ref(output_buffer),
                                         std::ref(log_stats_vec[i]), std::ref(worker_done[i]), std::ref(aln_params),
                                         std::ref(map_param), std::ref(index_parameters), std::ref(references),
                                         std::ref(index), std::ref(opt.read_group_id), assignments[i].thread_id,
                                         std::ref(fastqPool), std::ref(queue_se), use_good_numa, assignments[i].gpu_id);
                    workers.push_back(std::move(consumer));
                }
            }
        }

    } else {
        //PE
        fprintf(stderr, "PE module RABBIT_FX\n");

        if(use_good_numa) {
            for (int i = 0; i < opt.n_threads * 1 / 2; ++i) {
                if (assignments[i].flag) {
                    if (assignments[i].pass) {
//                        printf("gpu thread %d skip\n", i);
                        continue;
                    }
                    std::thread consumer(perform_task_async_pe_fx_GPU, std::ref(input_buffer), std::ref(output_buffer),
                                         std::ref(log_stats_vec[i]), std::ref(worker_done[i]), std::ref(aln_params),
                                         std::ref(map_param), std::ref(index_parameters), std::ref(references),
                                         std::ref(index), std::ref(opt.read_group_id), assignments[i].thread_id,
                                         std::ref(fastqPool), std::ref(queue_pe), use_good_numa, assignments[i].gpu_id, assignments[i].async_thread_id,
                                         batch_read_num, batch_total_read_len, chunk_num, opt.unordered_output);
                    workers.push_back(std::move(consumer));
                } else if (!opt.only_gpu) {
                    std::thread consumer(perform_task_async_pe_fx, std::ref(input_buffer), std::ref(output_buffer),
                                         std::ref(log_stats_vec[i]), std::ref(worker_done[i]), std::ref(aln_params),
                                         std::ref(map_param), std::ref(index_parameters), std::ref(references),
                                         std::ref(index), std::ref(opt.read_group_id), assignments[i].thread_id,
                                         std::ref(fastqPool), std::ref(queue_pe), use_good_numa, assignments[i].gpu_id);
                    workers.push_back(std::move(consumer));
                }
            }
            for (int i = opt.n_threads * 1 / 2; i < opt.n_threads; ++i) {
                if (assignments[i].flag) {
                    if (assignments[i].pass) {
//                        printf("gpu thread %d skip\n", i);
                        continue;
                    }
                    std::thread consumer(perform_task_async_pe_fx_GPU, std::ref(input_buffer), std::ref(output_buffer),
                                         std::ref(log_stats_vec[i]), std::ref(worker_done[i]), std::ref(aln_params),
                                         std::ref(map_param), std::ref(index_parameters), std::ref(references),
                                         std::ref(index2), std::ref(opt.read_group_id), assignments[i].thread_id,
                                         std::ref(fastqPool), std::ref(queue_pe), use_good_numa, assignments[i].gpu_id, assignments[i].async_thread_id,
                                         batch_read_num, batch_total_read_len, chunk_num, opt.unordered_output);
                    workers.push_back(std::move(consumer));
                } else if (!opt.only_gpu) {
                    std::thread consumer(perform_task_async_pe_fx, std::ref(input_buffer), std::ref(output_buffer),
                                         std::ref(log_stats_vec[i]), std::ref(worker_done[i]), std::ref(aln_params),
                                         std::ref(map_param), std::ref(index_parameters), std::ref(references),
                                         std::ref(index2), std::ref(opt.read_group_id), assignments[i].thread_id,
                                         std::ref(fastqPool), std::ref(queue_pe), use_good_numa, assignments[i].gpu_id);
                    workers.push_back(std::move(consumer));
                }
            }
        } else {
            for (int i = 0; i < opt.n_threads; ++i) {
                if (assignments[i].flag) {
                    if (assignments[i].pass) {
//                        printf("gpu thread %d skip\n", i);
                        continue;
                    }
                    std::thread consumer(perform_task_async_pe_fx_GPU, std::ref(input_buffer), std::ref(output_buffer),
                                         std::ref(log_stats_vec[i]), std::ref(worker_done[i]), std::ref(aln_params),
                                         std::ref(map_param), std::ref(index_parameters), std::ref(references),
                                         std::ref(index), std::ref(opt.read_group_id), assignments[i].thread_id,
                                         std::ref(fastqPool), std::ref(queue_pe), use_good_numa, assignments[i].gpu_id, assignments[i].async_thread_id,
                                         batch_read_num, batch_total_read_len, chunk_num, opt.unordered_output);
                    workers.push_back(std::move(consumer));
                } else if (!opt.only_gpu) {
                    std::thread consumer(perform_task_async_pe_fx, std::ref(input_buffer), std::ref(output_buffer),
                                         std::ref(log_stats_vec[i]), std::ref(worker_done[i]), std::ref(aln_params),
                                         std::ref(map_param), std::ref(index_parameters), std::ref(references),
                                         std::ref(index), std::ref(opt.read_group_id), assignments[i].thread_id,
                                         std::ref(fastqPool), std::ref(queue_pe), use_good_numa, assignments[i].gpu_id);
                    workers.push_back(std::move(consumer));
                }
            }
        }
    }

    if (opt.show_progress && isatty(fileno(stderr))) {
        show_progress_until_done(worker_done, log_stats_vec);
    }

    for (auto& worker : workers) {
        worker.join();
    }
    if (producer_thread.joinable()) {
        producer_thread.join();
    }

    logger.info() << "Done!\n";
    fprintf(stderr, "consumer cost %.2f\n", GetTime() - tt0);

    AlignmentStatistics tot_statistics;
    for (const auto& it : log_stats_vec) {
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

/**
 * @brief Main entry point of the application.
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return Exit code.
 */
int main(int argc, char **argv) {
    try {
        return run_rabbitsalign(argc, argv);
    } catch (const BadParameter& e) {
        logger.error() << "A parameter is invalid: " << e.what() << std::endl;
    } catch (const std::runtime_error& e) {
        logger.error() << "rabbitsalign runtime error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        logger.error() << "An unexpected error occurred: " << e.what() << std::endl;
    }
    return EXIT_FAILURE;
}
