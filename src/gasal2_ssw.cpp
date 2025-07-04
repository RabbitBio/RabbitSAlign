#include "gasal2_ssw.h"
#include "gpu_packing.h"
#include <sys/time.h>

#define use_device_mem

void writeToFasta(
        const std::vector<std::string>& sequences,
        const std::vector<std::string>& headers,
        const std::string& filename
        ) {
    std::ofstream file(filename);
    for (size_t i = 0; i < sequences.size(); ++i) {
        file << (headers[i].empty() ? std::to_string(i) : headers[i]) << "\n";
        file << sequences[i] << "\n";
    }
    file.close();
}

inline double GetTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double) tv.tv_sec + (double) tv.tv_usec / 1000000;
}

void solve_ssw_on_gpu(
        int thread_id,
        std::vector<gasal_tmp_res>& gasal_results,
        std::vector<std::string>& query_seqs,
        std::vector<std::string>& target_seqs,
        int match_score,
        int mismatch_score,
        int gap_open_score,
        int gap_extend_score
        ) {

    static thread_local double time_pre = 0;
    static thread_local double time_fill = 0;
    static thread_local double time_fill1 = 0;
    static thread_local double time_fill2 = 0;
    static thread_local double time_cal = 0;


    double t0;
    t0 = GetTime();
    static int cnt[THREAD_NUM_MAX] = {0};
    cnt[thread_id]++;
    assert(query_seqs.size() == target_seqs.size());
    assert(query_seqs.size() <= STREAM_BATCH_SIZE);
    gasal_results.resize(query_seqs.size());
    //gasal_set_device(GPU_SELECT);

    //-y local -s -p -t
    int argc = 18;
    const char* argv[] = {"A",  "-r", "1",  "-q",    "11", "-a", "2",  "-b",         "8",
        "-n", "1",  "-y", "local", "-s", "-p", "-t", "/dev/stdin", "/dev/stdin"};
    static Parameters* args[THREAD_NUM_MAX];
    if(cnt[thread_id] == 1) {
        args[thread_id] = new Parameters(argc, const_cast<char**>(argv));
        args[thread_id]->parse();
    }

    int print_out = args[thread_id]->print_out;


    //-------------------------------------------------------------------

    int total_seqs = query_seqs.size();

    static std::vector<std::string> query_headers(total_seqs, ">name");
    static std::vector<std::string> target_headers(total_seqs, ">name");
    static std::vector<uint8_t> query_seq_mod_vec(total_seqs, 0);
    static std::vector<uint8_t> target_seq_mod_vec(total_seqs, 0);
    static uint8_t* query_seq_mod = &query_seq_mod_vec[0];
    static uint8_t* target_seq_mod = &target_seq_mod_vec[0];

    int maximum_sequence_length_query = 0;
    int maximum_sequence_length_target = 0;

    for (size_t i = 0; i < query_seqs.size(); i++) {
        //        query_headers[i] += std::to_string(i);
        if(query_seqs[i].length() > MAX_QUERY_LEN) query_seqs[i] = query_seqs[i].substr(0, MAX_QUERY_LEN);
        maximum_sequence_length_query = MAX(maximum_sequence_length_query, query_seqs[i].length());
    }
    for (size_t i = 0; i < target_seqs.size(); i++) {
        //        target_headers[i] += std::to_string(i);
        if(target_seqs[i].length() > MAX_TARGET_LEN) target_seqs[i] = target_seqs[i].substr(0, MAX_TARGET_LEN);
        maximum_sequence_length_target = MAX(maximum_sequence_length_target, target_seqs[i].length());
    }

    if(maximum_sequence_length_query > MAX_QUERY_LEN) {
        std::cerr << "gasal2 : read size is too big, " <<  maximum_sequence_length_query << " > " << MAX_QUERY_LEN << std::endl;
        exit(0);
    }
    assert(maximum_sequence_length_query <= MAX_QUERY_LEN);
    assert(maximum_sequence_length_target <= MAX_TARGET_LEN);


    static gasal_gpu_storage_v gpu_storage_vecs[THREAD_NUM_MAX];
    if (cnt[thread_id] == 1) {
        gpu_storage_vecs[thread_id] = gasal_init_gpu_storage_v(NB_STREAMS);
        gasal_init_streams(
                &(gpu_storage_vecs[thread_id]),
                (MAX_QUERY_LEN + 7),  //TODO: remove maximum_sequence_length_query
                (MAX_TARGET_LEN + 7),
                STREAM_BATCH_SIZE,  //device
                args[thread_id]
                );
    }

    gpu_batch gpu_batch_arr[gpu_storage_vecs[thread_id].n];

    for (int z = 0; z < gpu_storage_vecs[thread_id].n; z++) {
        gpu_batch_arr[z].gpu_storage = &(gpu_storage_vecs[thread_id].a[z]);
    }

    uint32_t query_batch_idx = 0;
    uint32_t target_batch_idx = 0;
    int gpu_batch_arr_idx = 0;

    time_pre += GetTime() - t0;

    t0 = GetTime();

    for (int i = 0; i < total_seqs; i++) {
        gpu_batch_arr[gpu_batch_arr_idx].gpu_storage->current_n_alns++;

        if (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage->current_n_alns >
                gpu_batch_arr[gpu_batch_arr_idx].gpu_storage->host_max_n_alns) {
            gasal_host_alns_resize(
                    gpu_batch_arr[gpu_batch_arr_idx].gpu_storage,
                    gpu_batch_arr[gpu_batch_arr_idx].gpu_storage->host_max_n_alns * 2, args[thread_id]
                    );
        }

        (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_query_batch_offsets[i] = query_batch_idx;
        (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_target_batch_offsets[i] = target_batch_idx;

        query_batch_idx = gasal_host_batch_fill(
                gpu_batch_arr[gpu_batch_arr_idx].gpu_storage, query_batch_idx, query_seqs[i].data(),
                query_seqs[i].size(), QUERY
                );

        target_batch_idx = gasal_host_batch_fill(
                gpu_batch_arr[gpu_batch_arr_idx].gpu_storage, target_batch_idx, target_seqs[i].data(),
                target_seqs[i].size(), TARGET
                );

        (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_query_batch_lens[i] = query_seqs[i].size();
        (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_target_batch_lens[i] = target_seqs[i].size();
    }

    gasal_op_fill(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage, query_seq_mod, total_seqs, QUERY);
    gasal_op_fill(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage, target_seq_mod, total_seqs, TARGET);


    time_fill += GetTime() - t0;

    t0 = GetTime();

    gpu_batch_arr[gpu_batch_arr_idx].n_seqs_batch = total_seqs;
    uint32_t query_batch_bytes = query_batch_idx;
    uint32_t target_batch_bytes = target_batch_idx;
    gpu_batch_arr[gpu_batch_arr_idx].batch_start = 0;

    gasal_aln_async(
            gpu_batch_arr[gpu_batch_arr_idx].gpu_storage, query_batch_bytes, target_batch_bytes,
            gpu_batch_arr[gpu_batch_arr_idx].n_seqs_batch, args[thread_id]
            );
    gpu_batch_arr[gpu_batch_arr_idx].gpu_storage->current_n_alns = 0;

    gpu_batch_arr_idx = 0;

    while (true) {
        if (gasal_is_aln_async_done(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage) == 0) {
            if (print_out) {
                for (int j = 0, i = gpu_batch_arr[gpu_batch_arr_idx].batch_start;
                        j < gpu_batch_arr[gpu_batch_arr_idx].n_seqs_batch; i++, j++) {
                    std::ostringstream oss;
                    if (args[thread_id]->start_pos == WITH_TB) {
                        int u;
                        int offset = (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_query_batch_offsets[j];
                        int n_cigar_ops = (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->n_cigar_ops[j];
                        int last_op = ((gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->cigar[offset + n_cigar_ops - 1]) & 3;
                        int count = ((gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->cigar[offset + n_cigar_ops - 1]) >> 2;
                        for (u = n_cigar_ops - 2; u >= 0; u--) {
                            int curr_op = ((gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->cigar[offset + u]) & 3;
                            if (curr_op == last_op) {
                                count += ((gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->cigar[offset + u]) >> 2;
                            } else {
                                char op;
                                switch (last_op) {
                                    case 0:
                                        op = 'M';
                                        break;
                                    case 1:
                                        op = 'X';
                                        break;
                                    case 2:
                                        op = 'D';
                                        break;
                                    case 3:
                                        op = 'I';
                                        break;
                                    default:
                                        op = 'E';
                                        break;
                                }
                                oss << count << op;
                                count = ((gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->cigar[offset + u]) >> 2;
                            }
                            last_op = curr_op;
                        }
                        char op;
                        switch (last_op) {
                            case 0:
                                op = 'M';
                                break;
                            case 1:
                                op = 'X';
                                break;
                            case 2:
                                op = 'D';
                                break;
                            case 3:
                                op = 'I';
                                break;
                        }
                        oss << count << op;
                    }
                    gasal_results[i] = {
                        (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->aln_score[j],
                        (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->query_batch_start[j],
                        (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->query_batch_end[j],
                        (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->target_batch_start[j],
                        (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->target_batch_end[j],
                        oss.str()
                    };
                }
            }
            break;
        }
        usleep(100);
    }

    time_cal += GetTime() - t0;

    //    if (cnt[thread_id] % 100 == 0) {
    //        printf("gasal2 cpu main timer : %.2f %.2f %.2f\n", time_pre, time_fill, time_cal);
    //    }
}

void solve_ssw_on_gpu(
        int thread_id,
        std::vector<gasal_tmp_res>& gasal_results,
        std::vector<std::string_view>& query_seqs,
        std::vector<std::string_view>& target_seqs,
        int match_score,
        int mismatch_score,
        int gap_open_score,
        int gap_extend_score
        ) {
    static thread_local double time_pre = 0;
    static thread_local double time_fill = 0;
    static thread_local double time_fill1 = 0;
    static thread_local double time_fill2 = 0;
    static thread_local double time_cal = 0;


    double t0;
    t0 = GetTime();
    static int cnt[THREAD_NUM_MAX] = {0};
    cnt[thread_id]++;
    assert(query_seqs.size() == target_seqs.size());
    assert(query_seqs.size() <= STREAM_BATCH_SIZE_GPU);
    gasal_results.resize(query_seqs.size());
    //gasal_set_device(GPU_SELECT);

    std::ifstream f(".tmp1");
    if (!f) std::ofstream(".tmp1");
    //-y local -s -p -t
    int argc = 18;
    const char* argv[] = {"A",  "-r", "1",  "-q",    "11", "-a", "2",  "-b",         "8",
        "-n", "1",  "-y", "local", "-s", "-p", "-t", ".tmp1", ".tmp1"};
    static Parameters* args[THREAD_NUM_MAX];
    if(cnt[thread_id] == 1) {
        args[thread_id] = new Parameters(argc, const_cast<char**>(argv));
        args[thread_id]->parse();
    }

    int print_out = args[thread_id]->print_out;

    //-------------------------------------------------------------------

    int total_seqs = query_seqs.size();

    static std::vector<std::string> query_headers(total_seqs, ">name");
    static std::vector<std::string> target_headers(total_seqs, ">name");
    static std::vector<uint8_t> query_seq_mod_vec(total_seqs, 0);
    static std::vector<uint8_t> target_seq_mod_vec(total_seqs, 0);
    static uint8_t* query_seq_mod = &query_seq_mod_vec[0];
    static uint8_t* target_seq_mod = &target_seq_mod_vec[0];

    int maximum_sequence_length_query = 0;
    int maximum_sequence_length_target = 0;

    for (size_t i = 0; i < query_seqs.size(); i++) {
        //        query_headers[i] += std::to_string(i);
        if(query_seqs[i].length() > MAX_QUERY_LEN) query_seqs[i] = query_seqs[i].substr(0, MAX_QUERY_LEN);
        maximum_sequence_length_query = MAX(maximum_sequence_length_query, query_seqs[i].length());
    }
    for (size_t i = 0; i < target_seqs.size(); i++) {
        //        target_headers[i] += std::to_string(i);
        if(target_seqs[i].length() > MAX_TARGET_LEN) target_seqs[i] = target_seqs[i].substr(0, MAX_TARGET_LEN);
        maximum_sequence_length_target = MAX(maximum_sequence_length_target, target_seqs[i].length());
    }

    if(maximum_sequence_length_query > MAX_QUERY_LEN) {
        std::cerr << "gasal2 : read size is too big, " <<  maximum_sequence_length_query << " > " << MAX_QUERY_LEN << std::endl;
        exit(0);
    }
    assert(maximum_sequence_length_query <= MAX_QUERY_LEN);
    assert(maximum_sequence_length_target <= MAX_TARGET_LEN);


    static gasal_gpu_storage_v gpu_storage_vecs[THREAD_NUM_MAX];
    if (cnt[thread_id] == 1) {
        gpu_storage_vecs[thread_id] = gasal_init_gpu_storage_v(NB_STREAMS);
        gasal_init_streams(
                &(gpu_storage_vecs[thread_id]),
                (MAX_QUERY_LEN + 7),  //TODO: remove maximum_sequence_length_query
                (MAX_TARGET_LEN + 7),
                STREAM_BATCH_SIZE_GPU,  //device
                args[thread_id]
                );
    }

    gpu_batch gpu_batch_arr[gpu_storage_vecs[thread_id].n];

    for (int z = 0; z < gpu_storage_vecs[thread_id].n; z++) {
        gpu_batch_arr[z].gpu_storage = &(gpu_storage_vecs[thread_id].a[z]);
    }

    uint32_t query_batch_idx = 0;
    uint32_t target_batch_idx = 0;
    int gpu_batch_arr_idx = 0;

    time_pre += GetTime() - t0;

    t0 = GetTime();

    for (int i = 0; i < total_seqs; i++) {
        gpu_batch_arr[gpu_batch_arr_idx].gpu_storage->current_n_alns++;

        if (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage->current_n_alns >
                gpu_batch_arr[gpu_batch_arr_idx].gpu_storage->host_max_n_alns) {
            gasal_host_alns_resize(
                    gpu_batch_arr[gpu_batch_arr_idx].gpu_storage,
                    gpu_batch_arr[gpu_batch_arr_idx].gpu_storage->host_max_n_alns * 2, args[thread_id]
                    );
        }

        (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_query_batch_offsets[i] = query_batch_idx;
        (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_target_batch_offsets[i] = target_batch_idx;

        query_batch_idx = gasal_host_batch_fill(
                gpu_batch_arr[gpu_batch_arr_idx].gpu_storage, query_batch_idx, query_seqs[i].data(),
                query_seqs[i].size(), QUERY
                );

        target_batch_idx = gasal_host_batch_fill(
                gpu_batch_arr[gpu_batch_arr_idx].gpu_storage, target_batch_idx, target_seqs[i].data(),
                target_seqs[i].size(), TARGET
                );

        (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_query_batch_lens[i] = query_seqs[i].size();
        (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_target_batch_lens[i] = target_seqs[i].size();
    }

    gasal_op_fill(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage, query_seq_mod, total_seqs, QUERY);
    gasal_op_fill(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage, target_seq_mod, total_seqs, TARGET);


    time_fill += GetTime() - t0;

    t0 = GetTime();

    gpu_batch_arr[gpu_batch_arr_idx].n_seqs_batch = total_seqs;
    uint32_t query_batch_bytes = query_batch_idx;
    uint32_t target_batch_bytes = target_batch_idx;
    gpu_batch_arr[gpu_batch_arr_idx].batch_start = 0;

    gasal_aln_async(
            gpu_batch_arr[gpu_batch_arr_idx].gpu_storage, query_batch_bytes, target_batch_bytes,
            gpu_batch_arr[gpu_batch_arr_idx].n_seqs_batch, args[thread_id]
            );
    gpu_batch_arr[gpu_batch_arr_idx].gpu_storage->current_n_alns = 0;

    gpu_batch_arr_idx = 0;

    while (true) {
        if (gasal_is_aln_async_done(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage) == 0) {
            if (print_out) {
                for (int j = 0, i = gpu_batch_arr[gpu_batch_arr_idx].batch_start;
                        j < gpu_batch_arr[gpu_batch_arr_idx].n_seqs_batch; i++, j++) {
                    std::ostringstream oss;
                    if (args[thread_id]->start_pos == WITH_TB) {
                        int u;
                        int offset = (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_query_batch_offsets[j];
                        int n_cigar_ops = (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->n_cigar_ops[j];
                        int last_op = ((gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->cigar[offset + n_cigar_ops - 1]) & 3;
                        int count = ((gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->cigar[offset + n_cigar_ops - 1]) >> 2;
                        for (u = n_cigar_ops - 2; u >= 0; u--) {
                            int curr_op = ((gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->cigar[offset + u]) & 3;
                            if (curr_op == last_op) {
                                count += ((gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->cigar[offset + u]) >> 2;
                            } else {
                                char op;
                                switch (last_op) {
                                    case 0:
                                        op = 'M';
                                        break;
                                    case 1:
                                        op = 'X';
                                        break;
                                    case 2:
                                        op = 'D';
                                        break;
                                    case 3:
                                        op = 'I';
                                        break;
                                    default:
                                        op = 'E';
                                        break;
                                }
                                oss << count << op;
                                count = ((gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->cigar[offset + u]) >> 2;
                            }
                            last_op = curr_op;
                        }
                        char op;
                        switch (last_op) {
                            case 0:
                                op = 'M';
                                break;
                            case 1:
                                op = 'X';
                                break;
                            case 2:
                                op = 'D';
                                break;
                            case 3:
                                op = 'I';
                                break;
                        }
                        oss << count << op;
                    }
                    gasal_results[i] = {
                        (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->aln_score[j],
                        (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->query_batch_start[j],
                        (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->query_batch_end[j],
                        (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->target_batch_start[j],
                        (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->target_batch_end[j],
                        oss.str()
                    };
                }
            }
            break;
        }
        usleep(100);
    }

    time_cal += GetTime() - t0;

    //    if (cnt[thread_id] % 100 == 0) {
    //        printf("gasal2 cpu main timer : %.2f %.2f %.2f\n", time_pre, time_fill, time_cal);
    //    }
}

void solve_ssw_on_gpu2(
    int thread_id,
    std::vector<gasal_tmp_res>& gasal_results,
    std::vector<std::string_view>& query_seqs,
    std::vector<std::string_view>& target_seqs,
    int match_score,
    int mismatch_score,
    int gap_open_score,
    int gap_extend_score
) {

    static double time_pre[256] = {0};
    static double time_fill[256] = {0};
    static double time_fill1[256] = {0};
    static double time_fill1_1[256] = {0};
    static double time_fill1_2[256] = {0};
    static double time_fill1_3[256] = {0};
    static double time_fill2[256] = {0};
    static double time_cal[256] = {0};

    double t0;
    t0 = GetTime();
    static int cnt[THREAD_NUM_MAX] = {0};
    cnt[thread_id]++;
    assert(query_seqs.size() == target_seqs.size());
    assert(query_seqs.size() <= STREAM_BATCH_SIZE_GPU);
    gasal_results.resize(query_seqs.size());
    //gasal_set_device(GPU_SELECT);

    //-y local -s -p -t
    int argc = 18;
    const char* argv[] = {"A",  "-r", "1",  "-q",    "11", "-a", "2",  "-b",         "8",
                          "-n", "1",  "-y", "local", "-s", "-p", "-t", "/dev/stdin", "/dev/stdin"};
    static Parameters* args[THREAD_NUM_MAX];
    if(cnt[thread_id] == 1) {
        args[thread_id] = new Parameters(argc, const_cast<char**>(argv));
        args[thread_id]->parse();
    }

    int print_out = args[thread_id]->print_out;

    //-------------------------------------------------------------------

    int total_seqs = query_seqs.size();

    static std::vector<std::string> query_headers(total_seqs, ">name");
    static std::vector<std::string> target_headers(total_seqs, ">name");
    static std::vector<uint8_t> query_seq_mod_vec(total_seqs, 0);
    static std::vector<uint8_t> target_seq_mod_vec(total_seqs, 0);
    static uint8_t* query_seq_mod = &query_seq_mod_vec[0];
    static uint8_t* target_seq_mod = &target_seq_mod_vec[0];

	static const char** global_query_ptrs[THREAD_NUM_MAX];
	static int* global_query_lens[THREAD_NUM_MAX];
	static uint32_t* global_query_offsets[THREAD_NUM_MAX];
	static const char** global_target_ptrs[THREAD_NUM_MAX];
	static int* global_target_lens[THREAD_NUM_MAX];
	static uint32_t* global_target_offsets[THREAD_NUM_MAX];

    if (cnt[thread_id] == 1) {
        cudaMallocManaged(&global_query_ptrs[thread_id], STREAM_BATCH_SIZE_GPU * sizeof(char*));
        cudaMallocManaged(&global_query_lens[thread_id], STREAM_BATCH_SIZE_GPU * sizeof(int));
        cudaMallocManaged(&global_query_offsets[thread_id], STREAM_BATCH_SIZE_GPU * sizeof(uint32_t));
        cudaMallocManaged(&global_target_ptrs[thread_id], STREAM_BATCH_SIZE_GPU * sizeof(char*));
        cudaMallocManaged(&global_target_lens[thread_id], STREAM_BATCH_SIZE_GPU * sizeof(int));
        cudaMallocManaged(&global_target_offsets[thread_id], STREAM_BATCH_SIZE_GPU * sizeof(uint32_t));
    }

    int maximum_sequence_length_query = 0;
    int maximum_sequence_length_target = 0;

    for (size_t i = 0; i < query_seqs.size(); i++) {
        //        query_headers[i] += std::to_string(i);
        if(query_seqs[i].length() > MAX_QUERY_LEN) query_seqs[i] = query_seqs[i].substr(0, MAX_QUERY_LEN);
        maximum_sequence_length_query = MAX(maximum_sequence_length_query, query_seqs[i].length());
    }
    for (size_t i = 0; i < target_seqs.size(); i++) {
        //        target_headers[i] += std::to_string(i);
        if(target_seqs[i].length() > MAX_TARGET_LEN) target_seqs[i] = target_seqs[i].substr(0, MAX_TARGET_LEN);
        maximum_sequence_length_target = MAX(maximum_sequence_length_target, target_seqs[i].length());
    }

    if(maximum_sequence_length_query > MAX_QUERY_LEN) {
        std::cerr << "gasal2 : read size is too big, " <<  maximum_sequence_length_query << " > " << MAX_QUERY_LEN << std::endl;
        exit(0);
    }
    assert(maximum_sequence_length_query <= MAX_QUERY_LEN);
    assert(maximum_sequence_length_target <= MAX_TARGET_LEN);


    static gasal_gpu_storage_v gpu_storage_vecs[THREAD_NUM_MAX];
    if (cnt[thread_id] == 1) {
        gpu_storage_vecs[thread_id] = gasal_init_gpu_storage_v(NB_STREAMS);
        gasal_init_streams(
            &(gpu_storage_vecs[thread_id]),
            (MAX_QUERY_LEN + 7),  //TODO: remove maximum_sequence_length_query
            (MAX_TARGET_LEN + 7),
            STREAM_BATCH_SIZE_GPU,  //device
            args[thread_id],
#ifdef use_device_mem
            1
#else
            0
#endif
        );
    }

    gpu_batch gpu_batch_arr[gpu_storage_vecs[thread_id].n];

    for (int z = 0; z < gpu_storage_vecs[thread_id].n; z++) {
        gpu_batch_arr[z].gpu_storage = &(gpu_storage_vecs[thread_id].a[z]);
    }

    uint32_t query_batch_idx = 0;
    uint32_t target_batch_idx = 0;
    int gpu_batch_arr_idx = 0;

    time_pre[thread_id] += GetTime() - t0;

    t0 = GetTime();

	auto gpu_storage = gpu_batch_arr[gpu_batch_arr_idx].gpu_storage;
#ifdef use_device_mem
    double t1 = GetTime();
    double t2;

	int total = query_seqs.size();
	uint32_t q_offset = 0, t_offset = 0;
    t2 = GetTime();
	for (int i = 0; i < total; i++) {
        gpu_storage->host_query_batch_offsets[i] = q_offset;
        gpu_storage->host_target_batch_offsets[i] = t_offset;

		global_query_ptrs[thread_id][i] = query_seqs[i].data();
		global_query_lens[thread_id][i] = query_seqs[i].size();
		global_query_offsets[thread_id][i] = q_offset;
		q_offset += ((global_query_lens[thread_id][i] + 7) & ~7);

		global_target_ptrs[thread_id][i] = target_seqs[i].data();
		global_target_lens[thread_id][i] = target_seqs[i].size();
		global_target_offsets[thread_id][i] = t_offset;
		t_offset += ((global_target_lens[thread_id][i] + 7) & ~7);

        gpu_storage->host_query_batch_lens[i] = query_seqs[i].size();
        gpu_storage->host_target_batch_lens[i] = target_seqs[i].size();
	}
    time_fill1_1[thread_id] += GetTime() - t2;
    gpu_storage->extensible_host_unpacked_query_batch->data_size = q_offset;
    gpu_storage->extensible_host_unpacked_target_batch->data_size = t_offset;
    query_batch_idx = q_offset;
    target_batch_idx = t_offset;

    t2 = GetTime();
	assert(gpu_storage->extensible_host_unpacked_query_batch->offset == 0);
	assert(gpu_storage->extensible_host_unpacked_target_batch->offset == 0);

	char* d_query_data = (char*)gpu_storage->extensible_host_unpacked_query_batch->data;
	char* d_target_data = (char*)gpu_storage->extensible_host_unpacked_target_batch->data;
    time_fill2[thread_id] += GetTime() - t1;

    t1 = GetTime();
	LaunchPackBatchesKernel(
			d_query_data, d_target_data,
			global_query_ptrs[thread_id], global_query_lens[thread_id], global_query_offsets[thread_id],
			global_target_ptrs[thread_id], global_target_lens[thread_id], global_target_offsets[thread_id],
            total
            );
    time_fill2[thread_id] += GetTime() - t1;

#else
	std::vector<uint32_t> pre_sum_query_batch_idx;
    pre_sum_query_batch_idx.resize(total_seqs + 1);
    std::vector<uint32_t> pre_sum_target_batch_idx;
    pre_sum_target_batch_idx.resize(total_seqs + 1);
    pre_sum_query_batch_idx[0] = 0;
    pre_sum_target_batch_idx[0] = 0;
    for (int i = 1; i <= total_seqs; i++) {
        int query_size = query_seqs[i - 1].size();
        query_size = (query_size + 7) & ~7;
        pre_sum_query_batch_idx[i] = pre_sum_query_batch_idx[i - 1] + query_size;

        int target_size = target_seqs[i - 1].size();
		target_size = (target_size + 7) & ~7;
		pre_sum_target_batch_idx[i] = pre_sum_target_batch_idx[i - 1] + target_size;
	}

    for (int i = 0; i < total_seqs; i++) {
        query_batch_idx = pre_sum_query_batch_idx[i];
        target_batch_idx = pre_sum_target_batch_idx[i];

        gpu_storage->host_query_batch_offsets[i] = query_batch_idx;
        gpu_storage->host_target_batch_offsets[i] = target_batch_idx;

        host_batch_t *cur_page = NULL;

        cur_page = (gpu_storage->extensible_host_unpacked_query_batch);
        uint32_t size = query_seqs[i].size();
        const char* data = query_seqs[i].data();
        int nbr_N = 0;
        while ((size + nbr_N) % 8) nbr_N++;
        uint32_t idx = query_batch_idx;
        memcpy(&(cur_page->data[idx - cur_page->offset]), data, size);
        for(int i = 0; i < nbr_N; i++) {
            cur_page->data[idx + size - cur_page->offset + i] = 0x4E;
        }
        cur_page->data_size += size + nbr_N;

        cur_page = (gpu_storage->extensible_host_unpacked_target_batch);
        size = target_seqs[i].size();
        data = target_seqs[i].data();
        nbr_N = 0;
        while ((size + nbr_N) % 8) nbr_N++;
        idx = target_batch_idx;
        memcpy(&(cur_page->data[idx - cur_page->offset]), data, size);
        for(int i = 0; i < nbr_N; i++) {
            cur_page->data[idx + size - cur_page->offset + i] = 0x4E;
        }
        cur_page->data_size += size + nbr_N;

        gpu_storage->host_query_batch_lens[i] = query_seqs[i].size();
        gpu_storage->host_target_batch_lens[i] = target_seqs[i].size();
    }
    query_batch_idx = pre_sum_query_batch_idx[total_seqs];
    target_batch_idx = pre_sum_target_batch_idx[total_seqs];
#endif
    gpu_batch_arr[gpu_batch_arr_idx].gpu_storage->current_n_alns = total_seqs;

    gasal_op_fill(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage, query_seq_mod, total_seqs, QUERY);
    gasal_op_fill(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage, target_seq_mod, total_seqs, TARGET);


    time_fill[thread_id] += GetTime() - t0;

    t0 = GetTime();

    gpu_batch_arr[gpu_batch_arr_idx].n_seqs_batch = total_seqs;
    uint32_t query_batch_bytes = query_batch_idx;
    uint32_t target_batch_bytes = target_batch_idx;
    gpu_batch_arr[gpu_batch_arr_idx].batch_start = 0;

    gasal_aln_async(
        gpu_batch_arr[gpu_batch_arr_idx].gpu_storage, query_batch_bytes, target_batch_bytes,
        gpu_batch_arr[gpu_batch_arr_idx].n_seqs_batch, args[thread_id],
#ifdef use_device_mem
        1
#else
        0
#endif
    );
    gpu_batch_arr[gpu_batch_arr_idx].gpu_storage->current_n_alns = 0;

    gpu_batch_arr_idx = 0;
    
    while (true) {
        if (gasal_is_aln_async_done(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage) == 0) {
            if (print_out) {
                for (int j = 0, i = gpu_batch_arr[gpu_batch_arr_idx].batch_start;
                     j < gpu_batch_arr[gpu_batch_arr_idx].n_seqs_batch; i++, j++) {
                    std::ostringstream oss;
                    if (args[thread_id]->start_pos == WITH_TB) {
                        int u;
                        int offset = (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_query_batch_offsets[j];
                        int n_cigar_ops = (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->n_cigar_ops[j];
                        int last_op = ((gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->cigar[offset + n_cigar_ops - 1]) & 3;
                        int count = ((gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->cigar[offset + n_cigar_ops - 1]) >> 2;
                        for (u = n_cigar_ops - 2; u >= 0; u--) {
                            int curr_op = ((gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->cigar[offset + u]) & 3;
                            if (curr_op == last_op) {
                                count += ((gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->cigar[offset + u]) >> 2;
                            } else {
                                char op;
                                switch (last_op) {
                                    case 0:
                                        op = 'M';
                                        break;
                                    case 1:
                                        op = 'X';
                                        break;
                                    case 2:
                                        op = 'D';
                                        break;
                                    case 3:
                                        op = 'I';
                                        break;
                                    default:
                                        op = 'E';
                                        break;
                                }
                                oss << count << op;
                                count = ((gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->cigar[offset + u]) >> 2;
                            }
                            last_op = curr_op;
                        }
                        char op;
                        switch (last_op) {
                            case 0:
                                op = 'M';
                                break;
                            case 1:
                                op = 'X';
                                break;
                            case 2:
                                op = 'D';
                                break;
                            case 3:
                                op = 'I';
                                break;
                        }
                        oss << count << op;
                    }
                    gasal_results[i] = {
                        (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->aln_score[j],
                        (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->query_batch_start[j],
                        (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->query_batch_end[j],
                        (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->target_batch_start[j],
                        (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->target_batch_end[j],
                        oss.str()
                    };
                }
            }
            break;
        }
        usleep(100);
    }

    time_cal[thread_id] += GetTime() - t0;

    if (cnt[thread_id] % 100 == 0) {
        printf("gasal2 gpu %d-%d main timer : %.2f %.2f[%.2f(%.2f %.2f %.2f) %.2f] %.2f\n", thread_id, cnt[thread_id], time_pre[thread_id], time_fill[thread_id], time_fill1[thread_id], time_fill1_1[thread_id], time_fill1_2[thread_id], time_fill1_3[thread_id], time_fill2[thread_id], time_cal[thread_id]);
    }
}

void solve_ssw_on_gpu_pre_copy(
        int thread_id,
        std::vector<gasal_tmp_res>& gasal_results,
        std::vector<int>& query_seqs,
        std::vector<int>& target_seqs,
        int match_score,
        int mismatch_score,
        int gap_open_score,
        int gap_extend_score,
        char* device_query_ptr,
        char* device_target_ptr
) {

    static double time_pre[256] = {0};
    static double time_fill[256] = {0};
    static double time_fill1[256] = {0};
    static double time_fill2[256] = {0};
    static double time_cal[256] = {0};

    double t0;
    t0 = GetTime();
    static int cnt[THREAD_NUM_MAX] = {0};
    cnt[thread_id]++;
    assert(query_seqs.size() == target_seqs.size());
    assert(query_seqs.size() <= STREAM_BATCH_SIZE_GPU);
    gasal_results.resize(query_seqs.size());
    //gasal_set_device(GPU_SELECT);

    //-y local -s -p -t
    int argc = 18;
    const char* argv[] = {"A",  "-r", "1",  "-q",    "11", "-a", "2",  "-b",         "8",
                          "-n", "1",  "-y", "local", "-s", "-p", "-t", "/dev/stdin", "/dev/stdin"};
    static Parameters* args[THREAD_NUM_MAX];
    if(cnt[thread_id] == 1) {
        args[thread_id] = new Parameters(argc, const_cast<char**>(argv));
        args[thread_id]->parse();
    }

    int print_out = args[thread_id]->print_out;

    //-------------------------------------------------------------------

    int total_seqs = query_seqs.size();

    static std::vector<std::string> query_headers(total_seqs, ">name");
    static std::vector<std::string> target_headers(total_seqs, ">name");
    static std::vector<uint8_t> query_seq_mod_vec(total_seqs, 0);
    static std::vector<uint8_t> target_seq_mod_vec(total_seqs, 0);
    static uint8_t* query_seq_mod = &query_seq_mod_vec[0];
    static uint8_t* target_seq_mod = &target_seq_mod_vec[0];

    int maximum_sequence_length_query = 0;
    int maximum_sequence_length_target = 0;

    for (size_t i = 0; i < query_seqs.size(); i++) {
        if(query_seqs[i] > MAX_QUERY_LEN) query_seqs[i] = MAX_QUERY_LEN;
        maximum_sequence_length_query = MAX(maximum_sequence_length_query, query_seqs[i]);
    }
    for (size_t i = 0; i < target_seqs.size(); i++) {
        if(target_seqs[i] > MAX_TARGET_LEN) target_seqs[i] = MAX_TARGET_LEN;
        maximum_sequence_length_target = MAX(maximum_sequence_length_target, target_seqs[i]);
    }

    if(maximum_sequence_length_query > MAX_QUERY_LEN) {
        std::cerr << "gasal2 : read size is too big, " <<  maximum_sequence_length_query << " > " << MAX_QUERY_LEN << std::endl;
        exit(0);
    }
    assert(maximum_sequence_length_query <= MAX_QUERY_LEN);
    assert(maximum_sequence_length_target <= MAX_TARGET_LEN);


    static gasal_gpu_storage_v gpu_storage_vecs[THREAD_NUM_MAX];
    if (cnt[thread_id] == 1) {
        gpu_storage_vecs[thread_id] = gasal_init_gpu_storage_v(NB_STREAMS);
        gasal_init_streams(
                &(gpu_storage_vecs[thread_id]),
                (MAX_QUERY_LEN + 7),  //TODO: remove maximum_sequence_length_query
                (MAX_TARGET_LEN + 7),
                STREAM_BATCH_SIZE_GPU,  //device
                args[thread_id],
                1
        );
    }

    gpu_batch gpu_batch_arr[gpu_storage_vecs[thread_id].n];

    for (int z = 0; z < gpu_storage_vecs[thread_id].n; z++) {
        gpu_batch_arr[z].gpu_storage = &(gpu_storage_vecs[thread_id].a[z]);
    }

    uint32_t query_batch_idx = 0;
    uint32_t target_batch_idx = 0;
    int gpu_batch_arr_idx = 0;

    time_pre[thread_id] += GetTime() - t0;

    t0 = GetTime();

    auto gpu_storage = gpu_batch_arr[gpu_batch_arr_idx].gpu_storage;
    double t1 = GetTime();
    int total = query_seqs.size();
    uint32_t q_offset = 0, t_offset = 0;
    for (int i = 0; i < total; i++) {
        gpu_storage->host_query_batch_offsets[i] = q_offset;
        gpu_storage->host_target_batch_offsets[i] = t_offset;

        q_offset += ((query_seqs[i] + 7) & ~7);
        t_offset += ((target_seqs[i] + 7) & ~7);

        gpu_storage->host_query_batch_lens[i] = query_seqs[i];
        gpu_storage->host_target_batch_lens[i] = target_seqs[i];
    }
    gpu_storage->extensible_host_unpacked_query_batch->data_size = q_offset;
    gpu_storage->extensible_host_unpacked_target_batch->data_size = t_offset;
    query_batch_idx = q_offset;
    target_batch_idx = t_offset;
    time_fill1[thread_id] += GetTime() - t1;

    t1 = GetTime();
    gpu_storage->extensible_host_unpacked_query_batch->data = (uint8_t*)device_query_ptr;
    gpu_storage->extensible_host_unpacked_target_batch->data = (uint8_t*)device_target_ptr;
//    cudaMemcpy(
//            gpu_storage->extensible_host_unpacked_query_batch->data, device_query_ptr,
//            gpu_storage->extensible_host_unpacked_query_batch->data_size, cudaMemcpyDeviceToDevice
//    );
//    cudaMemcpy(
//            gpu_storage->extensible_host_unpacked_target_batch->data, device_target_ptr,
//            gpu_storage->extensible_host_unpacked_target_batch->data_size, cudaMemcpyDeviceToDevice
//    );
    assert(gpu_storage->extensible_host_unpacked_query_batch->offset == 0);
    assert(gpu_storage->extensible_host_unpacked_target_batch->offset == 0);
    time_fill2[thread_id] += GetTime() - t1;

    gpu_batch_arr[gpu_batch_arr_idx].gpu_storage->current_n_alns = total_seqs;
    gasal_op_fill(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage, query_seq_mod, total_seqs, QUERY);
    gasal_op_fill(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage, target_seq_mod, total_seqs, TARGET);

    time_fill[thread_id] += GetTime() - t0;

    t0 = GetTime();

    gpu_batch_arr[gpu_batch_arr_idx].n_seqs_batch = total_seqs;
    uint32_t query_batch_bytes = query_batch_idx;
    uint32_t target_batch_bytes = target_batch_idx;
    gpu_batch_arr[gpu_batch_arr_idx].batch_start = 0;

    gasal_aln_async(
            gpu_batch_arr[gpu_batch_arr_idx].gpu_storage, query_batch_bytes, target_batch_bytes,
            gpu_batch_arr[gpu_batch_arr_idx].n_seqs_batch, args[thread_id],
            1
    );
    gpu_batch_arr[gpu_batch_arr_idx].gpu_storage->current_n_alns = 0;

    gpu_batch_arr_idx = 0;

    while (true) {
        if (gasal_is_aln_async_done(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage) == 0) {
            if (print_out) {
                for (int j = 0, i = gpu_batch_arr[gpu_batch_arr_idx].batch_start;
                     j < gpu_batch_arr[gpu_batch_arr_idx].n_seqs_batch; i++, j++) {
                    std::ostringstream oss;
                    if (args[thread_id]->start_pos == WITH_TB) {
                        int u;
                        int offset = (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_query_batch_offsets[j];
                        int n_cigar_ops = (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->n_cigar_ops[j];
                        int last_op = ((gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->cigar[offset + n_cigar_ops - 1]) & 3;
                        int count = ((gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->cigar[offset + n_cigar_ops - 1]) >> 2;
                        for (u = n_cigar_ops - 2; u >= 0; u--) {
                            int curr_op = ((gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->cigar[offset + u]) & 3;
                            if (curr_op == last_op) {
                                count += ((gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->cigar[offset + u]) >> 2;
                            } else {
                                char op;
                                switch (last_op) {
                                    case 0:
                                        op = 'M';
                                        break;
                                    case 1:
                                        op = 'X';
                                        break;
                                    case 2:
                                        op = 'D';
                                        break;
                                    case 3:
                                        op = 'I';
                                        break;
                                    default:
                                        op = 'E';
                                        break;
                                }
                                oss << count << op;
                                count = ((gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->cigar[offset + u]) >> 2;
                            }
                            last_op = curr_op;
                        }
                        char op;
                        switch (last_op) {
                            case 0:
                                op = 'M';
                                break;
                            case 1:
                                op = 'X';
                                break;
                            case 2:
                                op = 'D';
                                break;
                            case 3:
                                op = 'I';
                                break;
                        }
                        oss << count << op;
                    }
                    gasal_results[i] = {
                            (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->aln_score[j],
                            (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->query_batch_start[j],
                            (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->query_batch_end[j],
                            (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->target_batch_start[j],
                            (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->target_batch_end[j],
                            oss.str()
                    };
                }
            }
            break;
        }
        usleep(100);
    }

    time_cal[thread_id] += GetTime() - t0;

//    if (cnt[thread_id] % 100 == 0) {
//        printf("gasal2 gpu %d-%d main timer : %.2f %.2f[%.2f %.2f] %.2f\n", thread_id, cnt[thread_id], time_pre[thread_id], time_fill[thread_id], time_fill1[thread_id], time_fill2[thread_id], time_cal[thread_id]);
//    }
}
