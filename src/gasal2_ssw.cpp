//
// Created by ylf9811 on 2024/1/4.
//
#include "gasal2_ssw.h"

#define NB_STREAMS 1

//#define STREAM_BATCH_SIZE (262144)
// this gives each stream HALF of the sequences.
//#define STREAM_BATCH_SIZE ceil((double)target_seqs.size() / (double)(2))

#define STREAM_BATCH_SIZE 1000  //ceil((double)target_seqs.size() / (double)(2 * 2))

#define DEBUG

#define MAX(a, b) (a > b ? a : b)

void writeToFasta(const std::vector<std::string>& sequences,
                  const std::vector<std::string>& headers,
                  const std::string& filename) {
    std::ofstream file(filename);
    for (size_t i = 0; i < sequences.size(); ++i) {
        file << (headers[i].empty() ? std::to_string(i) : headers[i]) << "\n";
        file << sequences[i] << "\n";
    }
    file.close();
}

void solve_ssw_on_gpu(
    std::vector<gasal_tmp_res>& gasal_results,
    std::vector<std::string>& todo_querys,
    std::vector<std::string>& todo_refs,
    int match_score,
    int mismatch_score,
    int gap_open_score,
    int gap_extend_score
) {
    gasal_results.resize(todo_querys.size());
    //gasal_set_device(GPU_SELECT);

    //-y local -s -p -t
    int argc = 18;
    const char* argv[] = {"A", "-r", "1", "-q", "11", "-a", "2", "-b", "8", "-n", "1", "-y", "local", "-s", "-p", "-t", "/dev/stdin", "/dev/stdin"};
    Parameters* args;
    args = new Parameters(argc, const_cast<char**>(argv));
    args->parse();
//    args->print();

    int print_out = args->print_out;
    int n_threads = args->n_threads;

    //--------------copy substitution scores to GPU--------------------
    gasal_subst_scores sub_scores;

    sub_scores.match = match_score;
    sub_scores.mismatch = mismatch_score;
    sub_scores.gap_open = gap_open_score - 1;
    sub_scores.gap_extend = gap_extend_score;
//
//    sub_scores.match = args->sa;
//    sub_scores.mismatch = args->sb;
//    sub_scores.gap_open = args->gapo;
//    sub_scores.gap_extend = args->gape;

    gasal_copy_subst_scores(&sub_scores);

    //-------------------------------------------------------------------

    std::vector<std::string>& query_seqs = todo_querys;
    std::vector<std::string>& target_seqs = todo_refs;
    std::vector<std::string> query_headers(todo_querys.size(), ">name");
    std::vector<std::string> target_headers(todo_refs.size(), ">name");
    std::string query_batch_line, target_batch_line;


    int total_seqs = 0;
    uint32_t maximum_sequence_length = 0;
    uint32_t target_seqs_len = 0;
    uint32_t query_seqs_len = 0;

    for (size_t i = 0; i < query_seqs.size(); i++) {
//        query_headers[i] += std::to_string(i);
        maximum_sequence_length = MAX(maximum_sequence_length, query_seqs[i].length());
        query_seqs_len += query_seqs[i].length();
    }
    for (size_t i = 0; i < target_seqs.size(); i++) {
//        target_headers[i] += std::to_string(i);
        maximum_sequence_length = MAX(maximum_sequence_length, target_seqs[i].length());
        target_seqs_len += target_seqs[i].length();
    }

//    writeToFasta(query_seqs, query_headers, "small_query.fasta");
//    writeToFasta(target_seqs, target_headers, "small_target.fasta");

    total_seqs = query_seqs.size();
    //TODO
    int maximum_sequence_length_query = MAX((query_seqs.back()).length(), 0);

#ifdef DEBUG
    std::cerr << "[TEST_PROG DEBUG]: ";
    std::cerr << "Size of read batches are: query=" << query_seqs_len << ", target=" << target_seqs_len
              << ". maximum_sequence_length=" << maximum_sequence_length << std::endl;
#endif

    // transforming the _mod into a char* array (to be passed to GASAL, which deals with C types)
    uint8_t* target_seq_mod = (uint8_t*) malloc(total_seqs * sizeof(uint8_t));
    uint8_t* query_seq_mod = (uint8_t*) malloc(total_seqs * sizeof(uint8_t));
    uint32_t* target_seq_id = (uint32_t*) malloc(total_seqs * sizeof(uint32_t));
    uint32_t* query_seq_id = (uint32_t*) malloc(total_seqs * sizeof(uint32_t));

    for (int i = 0; i < total_seqs; i++) {
        query_seq_mod[i] = 0;
        query_seq_id[i] = i + 1;
    }

    for (int i = 0; i < total_seqs; i++) {
        target_seq_mod[i] = 0;
        target_seq_id[i] = i + 1;
    }


    int* thread_seqs_idx = (int*) malloc(n_threads * sizeof(int));
    int* thread_n_seqs = (int*) malloc(n_threads * sizeof(int));
    int* thread_n_batchs = (int*) malloc(n_threads * sizeof(int));

    int thread_batch_size = (int) ceil((double) total_seqs / n_threads);
    int n_seqs_alloc = 0;
    for (int i = 0; i < n_threads; i++) {  //distribute the sequences among the threads equally
        thread_seqs_idx[i] = n_seqs_alloc;
        if (n_seqs_alloc + thread_batch_size < total_seqs)
            thread_n_seqs[i] = thread_batch_size;
        else
            thread_n_seqs[i] = total_seqs - n_seqs_alloc;
        thread_n_batchs[i] = (int) ceil((double) thread_n_seqs[i] / (STREAM_BATCH_SIZE));
        n_seqs_alloc += thread_n_seqs[i];
    }

//    std::ofstream file("other_info.txt");
//    file << maximum_sequence_length << "\n";
//    file << maximum_sequence_length_query << "\n";
//    file << total_seqs << "\n";
//    file << sub_scores.match << "\n";
//    file << sub_scores.mismatch << "\n";
//    file << sub_scores.gap_open << "\n";
//    file << sub_scores.gap_extend << "\n";
//    file << thread_batch_size << "\n";
//    file << n_seqs_alloc << "\n";
//    for(int i = 0; i < n_threads; i++) {
//        file << thread_seqs_idx[i] << " " << thread_n_seqs[i] << " " << thread_n_batchs[i] << "\n";
//    }
//    for(int i = 0; i < total_seqs; i++) {
//        file << query_seq_mod[i] << " " << query_seq_id[i] << " " << target_seq_mod[i] << " " <<  target_seq_id[i] << "\n";
//    }
//    for(int i = 0; i < total_seqs; i++) {
//        file << query_seqs[i] << "\n";
//        file << target_seqs[i] << "\n";
//    }
//    file.close();


#ifdef DEBUG
    std::cerr << "[TEST_PROG DEBUG]: ";
    std::cerr << "Processing..." << std::endl;
#endif

    gasal_gpu_storage_v* gpu_storage_vecs =
        (gasal_gpu_storage_v*) calloc(n_threads, sizeof(gasal_gpu_storage_v));
    for (int z = 0; z < n_threads; z++) {
        gpu_storage_vecs[z] = gasal_init_gpu_storage_v(NB_STREAMS);  // creating NB_STREAMS streams per thread

        /*
            About memory sizes:
            The required memory is the total size of the batch + its padding, divided by the number of streams.
            The worst case would be that every sequence has to be padded with 7 'N', since they must have a length multiple of 8.
            Even though the memory can be dynamically expanded both for Host and Device, it is advised to start with a memory large enough so that these expansions rarely occur (for better performance.)
            Modifying the factor '1' in front of each size lets you see how GASAL2 expands the memory when needed.
        */
        /*
        // For exemple, this is exactly the memory needed to allocate to fit all sequences is a single GPU BATCH.
        gasal_init_streams(&(gpu_storage_vecs[z]),
                            1 * ceil((double)(query_seqs_len +7*total_seqs) / (double)(NB_STREAMS)) ,
                            1 * ceil((double)(query_seqs_len +7*total_seqs) / (double)(NB_STREAMS)) ,
                            1 * ceil((double)(query_seqs_len +7*total_seqs) / (double)(NB_STREAMS)) ,
                            1 * ceil((double)(query_seqs_len +7*total_seqs) / (double)(NB_STREAMS))  ,
                            ceil((double)target_seqs.size() / (double)(NB_STREAMS)), // maximum number of alignments is bigger on target than on query side.
                            ceil((double)target_seqs.size() / (double)(NB_STREAMS)),
                            args);
        */
        //initializing the streams by allocating the required CPU and GPU memory
        // note: the calculations of the detailed sizes to allocate could be done on the library side (to hide it from the user's perspective)
        gasal_init_streams(
            &(gpu_storage_vecs[z]),
            (maximum_sequence_length_query + 7),  //TODO: remove maximum_sequence_length_query
            (maximum_sequence_length + 7),
            STREAM_BATCH_SIZE,  //device
            args
        );
    }
#ifdef DEBUG
    std::cerr << "[TEST_PROG DEBUG]: ";
    std::cerr << "size of host_unpack_query is " << (query_seqs_len + 7 * total_seqs) / (NB_STREAMS)
              << std::endl;
#endif

    int n_seqs = thread_n_seqs[0];      //number of sequences allocated to this thread
    int curr_idx = thread_seqs_idx[0];  //number of sequences allocated to this thread
    int seqs_done = 0;
    int n_batchs_done = 0;

    struct gpu_batch {                     //a struct to hold data structures of a stream
        gasal_gpu_storage_t* gpu_storage;  //the struct that holds the GASAL2 data structures
        int n_seqs_batch;  //number of sequences in the batch (<= (target_seqs.size() / NB_STREAMS))
        int batch_start;   //starting index of batch
    };

#ifdef DEBUG
    std::cerr << "[TEST_PROG DEBUG]: ";
    std::cerr << "Number of gpu_batch in gpu_batch_arr : " << gpu_storage_vecs[0].n << std::endl;
    std::cerr << "[TEST_PROG DEBUG]: ";
    std::cerr << "Number of gpu_storage_vecs in a gpu_batch : " << 0 + 1 << std::endl;
#endif

    gpu_batch gpu_batch_arr[gpu_storage_vecs[0].n];

    for (int z = 0; z < gpu_storage_vecs[0].n; z++) {
        gpu_batch_arr[z].gpu_storage = &(gpu_storage_vecs[0].a[z]);
    }

    if (n_seqs > 0) {
        // Loop on streams
        while (n_batchs_done < thread_n_batchs[0]) {
//            std::cerr << "[TEST_PROG DEBUG]: " << n_batchs_done  << " " << thread_n_batchs[0] << std::endl;
            int gpu_batch_arr_idx = 0;
            //------------checking the availability of a "free" stream"-----------------
            while (gpu_batch_arr_idx < gpu_storage_vecs[0].n &&
                   (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->is_free != 1) {
                gpu_batch_arr_idx++;
            }

            //找到了一个可以用的stream，启动！
            if (seqs_done < n_seqs && gpu_batch_arr_idx < gpu_storage_vecs[0].n) {
                uint32_t query_batch_idx = 0;
                uint32_t target_batch_idx = 0;
                unsigned int j = 0;
                //-----------Create a batch of sequences to be aligned on the GPU. The batch contains (target_seqs.size() / NB_STREAMS) number of sequences-----------------------

                for (int i = curr_idx; seqs_done < n_seqs && j < (STREAM_BATCH_SIZE); i++, j++, seqs_done++) {
                    gpu_batch_arr[gpu_batch_arr_idx].gpu_storage->current_n_alns++;

                    if (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage->current_n_alns >
                        gpu_batch_arr[gpu_batch_arr_idx].gpu_storage->host_max_n_alns) {
                        gasal_host_alns_resize(
                            gpu_batch_arr[gpu_batch_arr_idx].gpu_storage,
                            gpu_batch_arr[gpu_batch_arr_idx].gpu_storage->host_max_n_alns * 2, args
                        );
                    }

                    (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_query_batch_offsets[j] =
                        query_batch_idx;
                    (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_target_batch_offsets[j] =
                        target_batch_idx;

                    /*
                            All the filling is moved on the library size, to take care of the memory size and expansions (when needed).
                            The function gasal_host_batch_fill takes care of how to fill, how much to pad with 'N', and how to deal with memory.
                            It's the same function for query and target, and you only need to set the final flag to either ; this avoides code duplication.
                            The way the host memory is filled changes the current _idx (it's increased by size, and by the padding). That's why it's returned by the function.
                        */

                    query_batch_idx = gasal_host_batch_fill(
                        gpu_batch_arr[gpu_batch_arr_idx].gpu_storage, query_batch_idx, query_seqs[i].c_str(),
                        query_seqs[i].size(), QUERY
                    );

                    target_batch_idx = gasal_host_batch_fill(
                        gpu_batch_arr[gpu_batch_arr_idx].gpu_storage, target_batch_idx,
                        target_seqs[i].c_str(), target_seqs[i].size(), TARGET
                    );

                    (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_query_batch_lens[j] =
                        query_seqs[i].size();
                    (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_target_batch_lens[j] =
                        target_seqs[i].size();
                }

#ifdef DEBUG
                std::cerr << "[TEST_PROG DEBUG]: ";
                std::cerr << "Stream " << gpu_batch_arr_idx << ": j = " << j << ", seqs_done = " << seqs_done
                          << ", query_batch_idx=" << query_batch_idx
                          << " , target_batch_idx=" << target_batch_idx << std::endl;
#endif

                // Here, we fill the operations arrays for the current batch to be processed by the stream
                gasal_op_fill(
                    gpu_batch_arr[gpu_batch_arr_idx].gpu_storage, query_seq_mod + seqs_done - j, j, QUERY
                );
                gasal_op_fill(
                    gpu_batch_arr[gpu_batch_arr_idx].gpu_storage, target_seq_mod + seqs_done - j, j, TARGET
                );

                gpu_batch_arr[gpu_batch_arr_idx].n_seqs_batch = j;
                uint32_t query_batch_bytes = query_batch_idx;
                uint32_t target_batch_bytes = target_batch_idx;
                gpu_batch_arr[gpu_batch_arr_idx].batch_start = curr_idx;
                curr_idx += (STREAM_BATCH_SIZE);

                //----------------------------------------------------------------------------------------------------
                //-----------------calling the GASAL2 non-blocking alignment function---------------------------------

                gasal_aln_async(
                    gpu_batch_arr[gpu_batch_arr_idx].gpu_storage, query_batch_bytes, target_batch_bytes,
                    gpu_batch_arr[gpu_batch_arr_idx].n_seqs_batch, args
                );
                gpu_batch_arr[gpu_batch_arr_idx].gpu_storage->current_n_alns = 0;
                //---------------------------------------------------------------------------------
            }
            //启动结束

            //-------------------------------print alignment results----------------------------------------

            //不管前面有没有启动，这里都找一个完成的stream做输出
            gpu_batch_arr_idx = 0;
            while (gpu_batch_arr_idx < gpu_storage_vecs[0].n) {
                //loop through all the streams and print the results
                //of the finished streams.
                if (gasal_is_aln_async_done(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage) == 0) {
                    int j = 0;
                    if (print_out) {
                        /*
alignment.cigar = info.cigar;
alignment.edit_distance = info.edit_distance;
alignment.score = info.sw_score;
alignment.ref_start = ref_start + info.ref_start;
alignment.is_rc = !nam.is_rc;
alignment.ref_id = nam.ref_id;
alignment.is_unaligned = info.cigar.empty();
alignment.length = info.ref_span();

alignment.global_ed = info.edit_distance + softclipped;
alignment.gapped = true;
 */


                        for (int i = gpu_batch_arr[gpu_batch_arr_idx].batch_start;
                             j < gpu_batch_arr[gpu_batch_arr_idx].n_seqs_batch; i++, j++) {
                            std::ostringstream oss;
                            if (args->start_pos == WITH_TB) {
                                int u;
                                int offset = (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)
                                                 ->host_query_batch_offsets[j];
                                int n_cigar_ops =
                                    (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_res->n_cigar_ops[j];
                                int last_op = ((gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)
                                                   ->host_res->cigar[offset + n_cigar_ops - 1]) &
                                              3;
                                int count = ((gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)
                                                 ->host_res->cigar[offset + n_cigar_ops - 1]) >>
                                            2;
                                for (u = n_cigar_ops - 2; u >= 0; u--) {
                                    int curr_op = ((gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)
                                                       ->host_res->cigar[offset + u]) &
                                                  3;
                                    if (curr_op == last_op) {
                                        count += ((gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)
                                                      ->host_res->cigar[offset + u]) >>
                                                 2;
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
                                        count = ((gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)
                                                     ->host_res->cigar[offset + u]) >>
                                                2;
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
                    n_batchs_done++;
                }
                gpu_batch_arr_idx++;
            }
        }
    }

//    for (int z = 0; z < n_threads; z++) {
//        gasal_destroy_streams(&(gpu_storage_vecs[z]), args);
//        gasal_destroy_gpu_storage_v(&(gpu_storage_vecs[z]));
//    }
//    free(gpu_storage_vecs);
//    /*
//    string algorithm = al_type;
//    string start_type[2] = {"without_start", "with_start"};
//    al_type += "_";
//    al_type += start_type[start_pos==WITH_START];
//    */
//    delete args;  // closes the files
//    //free(args); // closes the files
}