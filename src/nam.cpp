#include "nam.hpp"
#include <thread>
#include <sstream>
#include <sys/time.h>
#include <map>
#include <queue>
#include "my_struct.hpp"

inline double GetTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double) tv.tv_sec + (double) tv.tv_usec / 1000000;
}


namespace {

//struct Hit {
//    int query_start;
//    int query_end;
//    int ref_start;
//    int ref_end;
//    bool operator<(const Hit& other) const {
//        if(query_start == other.query_start) return ref_start < other.ref_start;
//        return query_start < other.query_start;
//    }
//
//    bool operator==(const Hit& other) const {
//        return query_start == other.query_start &&
//                query_end == other.query_end &&
//                ref_start == other.ref_start &&
//                ref_end == other.ref_end;
//    }
//};

inline void add_to_hits_per_ref_fast(
    robin_hood::unordered_map<unsigned int, std::vector<Hit>>& hits_per_ref,
    int query_start,
    int query_end,
    const StrobemerIndex& index,
    size_t position
) {
    int min_diff = std::numeric_limits<int>::max();
    std::vector<Hit> tmp_hits;
    int pre_ref_id = -1;
    for (const auto hash = index.get_hash(position); index.get_hash(position) == hash; ++position) {
        int ref_start = index.get_strobe1_position(position);
        int ref_end = ref_start + index.strobe2_offset(position) + index.k();
        int diff = std::abs((query_end - query_start) - (ref_end - ref_start));
        if (diff <= min_diff) {
            auto ref_id = index.reference_index(position);
            if (ref_id == pre_ref_id || pre_ref_id == -1) {
                tmp_hits.push_back(Hit{query_start, query_end, ref_start, ref_end});
            } else {
                std::sort(tmp_hits.begin(), tmp_hits.end());
                hits_per_ref[pre_ref_id].insert(hits_per_ref[pre_ref_id].end(), tmp_hits.begin(), tmp_hits.end());
                tmp_hits.clear();
                tmp_hits.push_back(Hit{query_start, query_end, ref_start, ref_end});
            }
            pre_ref_id = ref_id;
            min_diff = diff;
        }
    }
    if(!tmp_hits.empty() && pre_ref_id != -1) {
        std::sort(tmp_hits.begin(), tmp_hits.end());
        hits_per_ref[pre_ref_id].insert(hits_per_ref[pre_ref_id].end(), tmp_hits.begin(), tmp_hits.end());
    }
}

inline void add_to_hits_per_ref(
    robin_hood::unordered_map<unsigned int, std::vector<Hit>>& hits_per_ref,
    int query_start,
    int query_end,
    const StrobemerIndex& index,
    size_t position
) {
    int min_diff = std::numeric_limits<int>::max();
    for (const auto hash = index.get_hash(position); index.get_hash(position) == hash; ++position) {
        int ref_start = index.get_strobe1_position(position);
        int ref_end = ref_start + index.strobe2_offset(position) + index.k();
        int diff = std::abs((query_end - query_start) - (ref_end - ref_start));
        if (diff <= min_diff) {
            hits_per_ref[index.reference_index(position)].push_back(Hit{query_start, query_end, ref_start, ref_end});
            min_diff = diff;
        }
    }
}

inline void add_to_hits_per_ref_pre(
    robin_hood::unordered_map<unsigned int, std::vector<Hit>>& hits_per_ref,
    int query_start,
    int query_end,
    const StrobemerIndex& index,
    size_t position
) {
    int min_diff = std::numeric_limits<int>::max();
    for (const auto hash = index.get_hash(position); index.get_hash(position) == hash; ++position) {
        int ref_start = index.get_strobe1_position(position);
        int ref_end = ref_start + index.strobe2_offset(position) + index.k();
        int diff = std::abs((query_end - query_start) - (ref_end - ref_start));
        if (diff <= min_diff) {
            int refIndex = index.reference_index(position);
            if (hits_per_ref.find(refIndex) == hits_per_ref.end()) {
                hits_per_ref[refIndex] = std::vector<Hit>(); 
            }
            min_diff = diff;
        }
    }
}

inline uint64_t GetTime() {
    unsigned int lo, hi;
    asm volatile ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}

//#define timer

void merge_hits_into_nams_fast(
    robin_hood::unordered_map<unsigned int, std::vector<Hit>>& hits_per_ref,
    int k,
    bool sort,
    bool is_revcomp,
    std::vector<Nam>& nams  // inout
) {
#ifdef timer
    thread_local int cnt = 0;
    cnt++;
    thread_local uint64_t time1 = 0;
    thread_local uint64_t time2 = 0;
    thread_local uint64_t time3 = 0;
    thread_local uint64_t time4 = 0;
    thread_local uint64_t time5 = 0;
    thread_local uint64_t time0 = 0;
    thread_local uint64_t sum = 0;
    uint64_t t_tot = GetTime();
    uint64_t t0;
#endif

    for (auto &[ref_id, hits] : hits_per_ref) {
#ifdef timer
        t0 = GetTime();
#endif
        if (sort) {
            //for(int i = 1; i < hits.size(); i++) {
            //    if(hits[i].query_start < hits[i - 1].query_start) {
            //        fprintf(stderr, "GG1 sort %d %d\n", hits[i - 1].query_start, hits[i].query_start);
            //        exit(0);
            //    } else if(hits[i].query_start == hits[i - 1].query_start && hits[i].ref_start < hits[i - 1].ref_start) {
            //        //fprintf(stderr, "GG2 sort %d %d %d %d\n", hits[i - 1].query_start, hits[i].query_start, hits[i - 1].ref_start, hits[i].ref_start);
            //        //for(int j = 0; j < hits.size(); j++)
            //        //    fprintf(stderr, "%d %d\n", hits[j].query_start, hits[j].ref_start);
            //        //fprintf(stderr, "\n");
            //        //exit(0);
            //    }
            //}
            std::sort(hits.begin(), hits.end());
        }
#ifdef timer
        time1 += GetTime() - t0;
#endif


        std::vector<Nam> open_nams;
        unsigned int prev_q_start = 0;

        for (size_t i = 0; i < hits.size(); ) {
#ifdef timer
            t0 = GetTime();
#endif
            size_t i_start = i;
            size_t i_end = i + 1;
            size_t i_size;
            while(i_end < hits.size() && hits[i_end].query_start == hits[i].query_start) i_end++;
            i = i_end;
            i_size = i_end - i_start;

            bool is_added[i_size];
            for(size_t j = 0; j < i_size; j++) is_added[j] = false;

            int query_start = hits[i_start].query_start;
            int cnt_done = 0;

            std::sort(hits.begin() + i_start, hits.begin() + i_end);

            for (auto & o : open_nams) {
                auto lower = std::lower_bound(hits.begin() + i_start, hits.begin() + i_end, o.ref_prev_hit_startpos + 1, [](const Hit& h, int value){
                    return h.ref_start < value;
                }) - hits.begin();
                auto upper = std::lower_bound(hits.begin() + i_start, hits.begin() + i_end, o.ref_end + 1,[](const Hit& h, int value) {
                    return h.ref_start < value;
                }) - hits.begin();
                for (size_t j = lower; j < upper; j++) {
                    if(is_added[j - i_start])
                        continue;
                    if ( query_start <= o.query_end) {
                        auto& h = hits[j];
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


//            for (auto & o : open_nams) {
//                if(query_start <= o.query_prev_hit_startpos) {
//                    fprintf(stderr, "bound GG1\n");
//                    exit(0);
//                }
//                // Extend NAM
//                if (query_start <= o.query_end) {
//                    // valid range is [o.ref_prev_hit_startpos, o.ref_end]
//                    // h.ref_start is in ascending order
//                    //                    auto lower = std::lower_bound(hits.begin() + i_start, hits.begin() + i_end, o.ref_prev_hit_startpos + 1, [](const Hit& h, int value){
//                    //                        return h.ref_start < value;
//                    //                    }) - hits.begin();
//                    //                    auto upper = std::lower_bound(hits.begin() + i_start, hits.begin() + i_end, o.ref_end + 1,[](const Hit& h, int value) {
//                    //                        return h.ref_start < value;
//                    //                    }) - hits.begin();
//                    //
//                    //                    for(size_t j = lower; j < upper; j++) {
//                    for (size_t j = i_start; j < i_end; j++) {
//                        if(is_added[j - i_start]) continue;
//                        auto& h = hits[j];
//                        if (o.ref_prev_hit_startpos < h.ref_start && h.ref_start <= o.ref_end) {
//                            if ((h.query_end > o.query_end) && (h.ref_end > o.ref_end)) {
//                                o.query_end = h.query_end;
//                                o.ref_end = h.ref_end;
//                                //                        o.previous_query_start = h.query_s;
//                                //                        o.previous_ref_start = h.ref_s; // keeping track so that we don't . Can be caused by interleaved repeats.
//                                o.query_prev_hit_startpos = h.query_start;
//                                o.ref_prev_hit_startpos = h.ref_start;
//                                o.n_hits++;
//                                //                        o.score += (float)1/ (float)h.count;
//                                is_added[j - i_start] = true;
//                                cnt_done++;
//                            } else if ((h.query_end <= o.query_end) && (h.ref_end <= o.ref_end)) {
//                                //                        o.previous_query_start = h.query_s;
//                                //                        o.previous_ref_start = h.ref_s; // keeping track so that we don't . Can be caused by interleaved repeats.
//                                o.query_prev_hit_startpos = h.query_start;
//                                o.ref_prev_hit_startpos = h.ref_start;
//                                o.n_hits++;
//                                //                        o.score += (float)1/ (float)h.count;
//                                is_added[j - i_start] = true;
//                                cnt_done++;
//                            }
//                        }
//                    }
//                    if(cnt_done == i_size) break;
//                }
//
//            }
#ifdef timer
            time2 += GetTime() - t0;
            t0 = GetTime();
#endif

            // Add the hit to open matches
            for(size_t j = 0; j < i_size; j++) {
                if (!is_added[j]){
                    Nam n;
                    n.query_start = hits[i_start + j].query_start;
                    n.query_end = hits[i_start + j].query_end;
                    n.ref_start = hits[i_start + j].ref_start;
                    n.ref_end = hits[i_start + j].ref_end;
                    n.ref_id = ref_id;
                    //                n.previous_query_start = h.query_s;
                    //                n.previous_ref_start = h.ref_s;
                    n.query_prev_hit_startpos = hits[i_start + j].query_start;
                    n.ref_prev_hit_startpos = hits[i_start + j].ref_start;
                    n.n_hits = 1;
                    n.is_rc = is_revcomp;
                    //                n.score += (float)1 / (float)h.count;
                    open_nams.push_back(n);
                }
            }

#ifdef timer
            time3 += GetTime() - t0;
            t0 = GetTime();
#endif
            // Only filter if we have advanced at least k nucleotides
            if (query_start > prev_q_start + k) {
                //            if (1) {

                // Output all NAMs from open_matches to final_nams that the current hit have passed
                for (auto &n : open_nams) {
                    if (n.query_end < query_start) {
                        int n_max_span = std::max(n.query_span(), n.ref_span());
                        int n_min_span = std::min(n.query_span(), n.ref_span());
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
                auto predicate = [c](decltype(open_nams)::value_type const &nam) { return nam.query_end < c; };
                open_nams.erase(std::remove_if(open_nams.begin(), open_nams.end(), predicate), open_nams.end());
                prev_q_start = query_start;
            }
#ifdef timer
            time4 += GetTime() - t0;
#endif
        }
        //        time3 += GetTime() - t0;

#ifdef timer
        t0 = GetTime();
#endif
        // Add all current open_matches to final NAMs
        for (auto &n : open_nams) {
            int n_max_span = std::max(n.query_span(), n.ref_span());
            int n_min_span = std::min(n.query_span(), n.ref_span());
            float n_score;
            n_score = ( 2*n_min_span -  n_max_span) > 0 ? (float) (n.n_hits * ( 2*n_min_span -  n_max_span) ) : 1;   // this is really just n_hits * ( min_span - (offset_in_span) ) );
                                                                                                                   //            n_score = n.n_hits * n.query_span();
            n.score = n_score;
            n.nam_id = nams.size();
            nams.push_back(n);
        }
#ifdef timer
        time5 += GetTime() - t0;
#endif
    }

#ifdef timer
    time0 += GetTime() - t_tot;

    if(cnt % 100000 == 0) {
        std::thread::id mainthreadid = std::this_thread::get_id();
        std::ostringstream ss;
        ss << mainthreadid;
        unsigned long mainthreadidvalue = std::stoul(ss.str());
        fprintf(stderr, "merge_hits_into_nams [%lu] tot_time:%lf time1:%lf time2:%lf time3:%lf time4:%lf time5:%lf - %llu\n",
                mainthreadidvalue, time0 * 1e-8, time1 * 1e-8, time2 * 1e-8, time3 * 1e-8, time4 * 1e-8, time5 * 1e-8, sum);
    }
#endif
}



void merge_hits_into_nams(
    robin_hood::unordered_map<unsigned int, std::vector<Hit>>& hits_per_ref,
    int k,
    bool sort,
    bool is_revcomp,
    std::vector<Nam>& nams  // inout
) {
#ifdef timer
    thread_local int cnt = 0;
    cnt++;
    thread_local uint64_t time1 = 0;
    thread_local uint64_t time2 = 0;
    thread_local uint64_t time3 = 0;
    thread_local uint64_t time4 = 0;
    thread_local uint64_t time5 = 0;
    thread_local uint64_t time0 = 0;
    thread_local uint64_t sum = 0;
    uint64_t t_tot = GetTime();
    uint64_t t0;
#endif

    for (auto &[ref_id, hits] : hits_per_ref) {
#ifdef timer
        t0 = GetTime();
#endif
        if (sort) {
            std::sort(hits.begin(), hits.end());
            //            std::sort(hits.begin(), hits.end(), [](const Hit& a, const Hit& b) -> bool {
            //                // first sort on query starts, then on reference starts
            ////                return std::tie(a.query_start, a.ref_start, a.query_end, a.ref_end) <
            ////                       std::tie(b.query_start, b.ref_start, b.query_end, b.ref_end);
            //
            //                return (a.query_start < b.query_start) || ( (a.query_start == b.query_start) && (a.ref_start < b.ref_start) );
            ////                return a.query_start < b.query_start;
            //            }
            //            );
        }
#ifdef timer
        time1 += GetTime() - t0;
#endif


        std::vector<Nam> open_nams;
        unsigned int prev_q_start = 0;
        for (auto &h : hits) {
            bool is_added = false;
#ifdef timer
            t0 = GetTime();
#endif

            for (auto & o : open_nams) {

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
#ifdef timer
            time2 += GetTime() - t0;
            t0 = GetTime();
#endif

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
#ifdef timer
            time3 += GetTime() - t0;
            t0 = GetTime();
#endif
            // Only filter if we have advanced at least k nucleotides
            if (h.query_start > prev_q_start + k) {
//            if (1) {

                // Output all NAMs from open_matches to final_nams that the current hit have passed
                for (auto &n : open_nams) {
                    if (n.query_end < h.query_start) {
                        int n_max_span = std::max(n.query_span(), n.ref_span());
                        int n_min_span = std::min(n.query_span(), n.ref_span());
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
                auto predicate = [c](decltype(open_nams)::value_type const &nam) { return nam.query_end < c; };
                open_nams.erase(std::remove_if(open_nams.begin(), open_nams.end(), predicate), open_nams.end());
                prev_q_start = h.query_start;
            }
#ifdef timer
            time4 += GetTime() - t0;
#endif
        }
        //        time3 += GetTime() - t0;

#ifdef timer
        t0 = GetTime();
#endif
        // Add all current open_matches to final NAMs
        for (auto &n : open_nams) {
            int n_max_span = std::max(n.query_span(), n.ref_span());
            int n_min_span = std::min(n.query_span(), n.ref_span());
            float n_score;
            n_score = ( 2*n_min_span -  n_max_span) > 0 ? (float) (n.n_hits * ( 2*n_min_span -  n_max_span) ) : 1;   // this is really just n_hits * ( min_span - (offset_in_span) ) );
                                                                                                                   //            n_score = n.n_hits * n.query_span();
            n.score = n_score;
            n.nam_id = nams.size();
            nams.push_back(n);
        }
#ifdef timer
        time5 += GetTime() - t0;
#endif
    }

#ifdef timer
    time0 += GetTime() - t_tot;

    if(cnt % 100000 == 0) {
        std::thread::id mainthreadid = std::this_thread::get_id();
        std::ostringstream ss;
        ss << mainthreadid;
        unsigned long mainthreadidvalue = std::stoul(ss.str());
        fprintf(stderr, "merge_hits_into_nams [%lu] tot_time:%lf time1:%lf time2:%lf time3:%lf time4:%lf time5:%lf - %llu\n",
                mainthreadidvalue, time0 * 1e-8, time1 * 1e-8, time2 * 1e-8, time3 * 1e-8, time4 * 1e-8, time5 * 1e-8, sum);
    }
#endif
}

std::vector<Nam> merge_hits_into_nams_forward_and_reverse_fast(
    std::array<robin_hood::unordered_map<unsigned int, std::vector<Hit>>, 2>& hits_per_ref,
    int k,
    bool sort
) {
    std::vector<Nam> nams;
    for (size_t is_revcomp = 0; is_revcomp < 2; ++is_revcomp) {
        auto& hits_oriented = hits_per_ref[is_revcomp];
        merge_hits_into_nams_fast(hits_oriented, k, sort, is_revcomp, nams);
    }
    return nams;
}

std::vector<Nam> merge_hits_into_nams_forward_and_reverse(
    std::array<robin_hood::unordered_map<unsigned int, std::vector<Hit>>, 2>& hits_per_ref,
    int k,
    bool sort
) {
    std::vector<Nam> nams;
    for (size_t is_revcomp = 0; is_revcomp < 2; ++is_revcomp) {
        auto& hits_oriented = hits_per_ref[is_revcomp];
        merge_hits_into_nams(hits_oriented, k, sort, is_revcomp, nams);
    }
    return nams;
}

} // namespace

/*
 * Find a query’s NAMs, ignoring randstrobes that occur too often in the
 * reference (have a count above filter_cutoff).
 *
 * Return the fraction of nonrepetitive hits (those not above the filter_cutoff threshold)
 */
#define pre_sort
//#define Detail_timer
#define unROLL
#ifdef Detail_timer
std::pair<float, std::vector<Nam>> find_nams(
    const QueryRandstrobeVector &query_randstrobes,
    const StrobemerIndex& index
) {
    static thread_local double tot_time = 0;
    static thread_local double find_time = 0;
    static thread_local double filter_time = 0;
    static thread_local double add_time = 0;
    static thread_local double merge_time = 0;
    static thread_local int cnt = 0;
    double t0;
    double tt0 = GetTime();
    std::array<robin_hood::unordered_map<unsigned int, std::vector<Hit>>, 2> hits_per_ref;
    hits_per_ref[0].reserve(100);
    hits_per_ref[1].reserve(100);
    int nr_good_hits = 0, total_hits = 0;

#ifdef unROLL
    int i = 0;
    for (; i + 4 <= query_randstrobes.size(); i += 4) {
        t0 = GetTime();
        auto q0 = query_randstrobes[i + 0].hash;
        auto q1 = query_randstrobes[i + 1].hash;
        auto q2 = query_randstrobes[i + 2].hash;
        auto q3 = query_randstrobes[i + 3].hash;
        constexpr int MAX_LINEAR_SEARCH = 4;
        const unsigned int top_N0 = q0 >> (64 - index.bits);
        const unsigned int top_N1 = q1 >> (64 - index.bits);
        const unsigned int top_N2 = q2 >> (64 - index.bits);
        const unsigned int top_N3 = q3 >> (64 - index.bits);
        uint64_t position_start0 = index.randstrobe_start_indices[top_N0];
        uint64_t position_start1 = index.randstrobe_start_indices[top_N1];
        uint64_t position_start2 = index.randstrobe_start_indices[top_N2];
        uint64_t position_start3 = index.randstrobe_start_indices[top_N3];
        uint64_t position_end0 = index.randstrobe_start_indices[top_N0 + 1];
        uint64_t position_end1 = index.randstrobe_start_indices[top_N1 + 1];
        uint64_t position_end2 = index.randstrobe_start_indices[top_N2 + 1];
        uint64_t position_end3 = index.randstrobe_start_indices[top_N3 + 1];
        int positions[4];
        positions[0] = -1;
        positions[1] = -1;
        positions[2] = -1;
        positions[3] = -1;

        auto cmp = [](const RefRandstrobe lhs, const RefRandstrobe rhs) {return lhs.hash < rhs.hash; };

        if (position_start0 == position_end0) {
            positions[0] = -1;
        } else if (position_end0 - position_start0 < MAX_LINEAR_SEARCH) {
            for ( ; position_start0 < position_end0; ++position_start0) {
                if (index.randstrobes[position_start0].hash == q0) {
                    positions[0] = position_start0;
                    break;
                }
                if (index.randstrobes[position_start0].hash > q0) {
                    positions[0] = -1;
                    break;
                }
            }
        } else {
            auto pos0 = std::lower_bound(index.randstrobes.begin() + position_start0, index.randstrobes.begin() + position_end0, RefRandstrobe{q0, 0, 0}, cmp);
            if (pos0->hash == q0) positions[0] = pos0 - index.randstrobes.begin();
        }

        if (position_start1 == position_end1) {
            positions[1] = -1;
        } else if (position_end1 - position_start1 < MAX_LINEAR_SEARCH) {
            for ( ; position_start1 < position_end1; ++position_start1) {
                if (index.randstrobes[position_start1].hash == q1) {
                    positions[1] = position_start1;
                    break;
                }
                if (index.randstrobes[position_start1].hash > q1) {
                    positions[1] = -1;
                    break;
                }
            }
        } else {
            auto pos1 = std::lower_bound(index.randstrobes.begin() + position_start1, index.randstrobes.begin() + position_end1, RefRandstrobe{q1, 0, 0}, cmp);
            if (pos1->hash == q1) positions[1] = pos1 - index.randstrobes.begin();
        }

        if (position_start2 == position_end2) {
            positions[2] = -1;
        } else if (position_end2 - position_start2 < MAX_LINEAR_SEARCH) {
            for ( ; position_start2 < position_end2; ++position_start2) {
                if (index.randstrobes[position_start2].hash == q2) {
                    positions[2] = position_start2;
                    break;
                }
                if (index.randstrobes[position_start2].hash > q2) {
                    positions[2] = -1;
                    break;
                }
            }
        } else {
            auto pos2 = std::lower_bound(index.randstrobes.begin() + position_start2, index.randstrobes.begin() + position_end2, RefRandstrobe{q2, 0, 0}, cmp);
            if (pos2->hash == q2) positions[2] = pos2 - index.randstrobes.begin();
        }

        if (position_start3 == position_end3) {
            positions[3] = -1;
        } else if (position_end3 - position_start3 < MAX_LINEAR_SEARCH) {
            for ( ; position_start3 < position_end3; ++position_start3) {
                if (index.randstrobes[position_start3].hash == q3) {
                    positions[3] = position_start3;
                    break;
                }
                if (index.randstrobes[position_start3].hash > q3) {
                    positions[3] = -1;
                    break;
                }
            }
        } else {
            auto pos3 = std::lower_bound(index.randstrobes.begin() + position_start3, index.randstrobes.begin() + position_end3, RefRandstrobe{q3, 0, 0}, cmp);
            if (pos3->hash == q3) positions[3] = pos3 - index.randstrobes.begin();
        }
        find_time += GetTime() - t0;

        for(int j = 0; j < 4; j++) {
            auto q = query_randstrobes[i + j];
            if (positions[j] != index.end()){
                total_hits++;
                t0 = GetTime();
                auto res = index.is_filtered(positions[j]);
                filter_time += GetTime() - t0;
                if (res) {
                    continue;
                }
                nr_good_hits++;
                t0 = GetTime();
                add_to_hits_per_ref(hits_per_ref[q.is_reverse], q.start, q.end, index, positions[j]);
                add_time += GetTime() - t0;
            }

        }
    }
    for (; i < query_randstrobes.size(); i++) {
        auto q = query_randstrobes[i];
        t0 = GetTime();
        size_t position = index.find(q.hash);
        find_time += GetTime() - t0;
        if (position != index.end()){
            total_hits++;
            t0 = GetTime();
            auto res = index.is_filtered(position);
            filter_time += GetTime() - t0;
            if (res) {
                continue;
            }
            nr_good_hits++;
            t0 = GetTime();
            add_to_hits_per_ref(hits_per_ref[q.is_reverse], q.start, q.end, index, position);
            add_time += GetTime() - t0;
        }
    }
#else
    for (const auto &q : query_randstrobes) {
        t0 = GetTime();
        size_t position = index.find(q.hash);
        find_time += GetTime() - t0;
        if (position != index.end()){
            total_hits++;
            t0 = GetTime();
            auto res = index.is_filtered(position);
            filter_time += GetTime() - t0;
            if (res) {
                continue;
            }
            nr_good_hits++;
            t0 = GetTime();
            add_to_hits_per_ref(hits_per_ref[q.is_reverse], q.start, q.end, index, position);
            add_time += GetTime() - t0;
        }
    }
#endif
    t0 = GetTime();
    float nonrepetitive_fraction = total_hits > 0 ? ((float) nr_good_hits) / ((float) total_hits) : 1.0;
    auto nams = merge_hits_into_nams_forward_and_reverse(hits_per_ref, index.k(), false);
    merge_time += GetTime() - t0;

    tot_time += GetTime() - tt0;

    cnt++;

    if(cnt % 100000 == 0) {
        std::thread::id mainthreadid = std::this_thread::get_id();
        std::ostringstream ss;
        ss << mainthreadid;
        unsigned long mainthreadidvalue = std::stoul(ss.str());
        fprintf(stderr, "find_nams [%lld] tot_time:%lf find_time:%lf filter_time:%lf add_time:%lf merge_time:%lf\n", mainthreadidvalue, tot_time, find_time, filter_time, add_time, merge_time);
    }
    return make_pair(nonrepetitive_fraction, nams);
}
#else
std::pair<float, std::vector<Nam>> find_nams(
    const QueryRandstrobeVector &query_randstrobes,
    const StrobemerIndex& index
) {
    std::array<robin_hood::unordered_map<unsigned int, std::vector<Hit>>, 2> hits_per_ref;
    hits_per_ref[0].reserve(100);
    hits_per_ref[1].reserve(100);
    int nr_good_hits = 0, total_hits = 0;
#ifdef unROLL
    int i = 0;
    for (; i + 4 <= query_randstrobes.size(); i += 4) {
        auto q0 = query_randstrobes[i + 0].hash;
        auto q1 = query_randstrobes[i + 1].hash;
        auto q2 = query_randstrobes[i + 2].hash;
        auto q3 = query_randstrobes[i + 3].hash;
        constexpr int MAX_LINEAR_SEARCH = 4;
        const unsigned int top_N0 = q0 >> (64 - index.bits);
        const unsigned int top_N1 = q1 >> (64 - index.bits);
        const unsigned int top_N2 = q2 >> (64 - index.bits);
        const unsigned int top_N3 = q3 >> (64 - index.bits);
        uint64_t position_start0 = index.randstrobe_start_indices[top_N0];
        uint64_t position_start1 = index.randstrobe_start_indices[top_N1];
        uint64_t position_start2 = index.randstrobe_start_indices[top_N2];
        uint64_t position_start3 = index.randstrobe_start_indices[top_N3];
        uint64_t position_end0 = index.randstrobe_start_indices[top_N0 + 1];
        uint64_t position_end1 = index.randstrobe_start_indices[top_N1 + 1];
        uint64_t position_end2 = index.randstrobe_start_indices[top_N2 + 1];
        uint64_t position_end3 = index.randstrobe_start_indices[top_N3 + 1];
        int positions[4];
        positions[0] = -1;
        positions[1] = -1;
        positions[2] = -1;
        positions[3] = -1;

        auto cmp = [](const RefRandstrobe lhs, const RefRandstrobe rhs) {return lhs.hash < rhs.hash; };

        if (position_start0 == position_end0) {
            positions[0] = -1;
        } else if (position_end0 - position_start0 < MAX_LINEAR_SEARCH) {
            for ( ; position_start0 < position_end0; ++position_start0) {
                if (index.randstrobes[position_start0].hash == q0) {
                    positions[0] = position_start0;
                    break;
                }
                if (index.randstrobes[position_start0].hash > q0) {
                    positions[0] = -1;
                    break;
                }
            }
        } else {
            auto pos0 = std::lower_bound(index.randstrobes.begin() + position_start0, index.randstrobes.begin() + position_end0, RefRandstrobe{q0, 0, 0}, cmp);
            if (pos0->hash == q0) positions[0] = pos0 - index.randstrobes.begin();
        }

        if (position_start1 == position_end1) {
            positions[1] = -1;
        } else if (position_end1 - position_start1 < MAX_LINEAR_SEARCH) {
            for ( ; position_start1 < position_end1; ++position_start1) {
                if (index.randstrobes[position_start1].hash == q1) {
                    positions[1] = position_start1;
                    break;
                }
                if (index.randstrobes[position_start1].hash > q1) {
                    positions[1] = -1;
                    break;
                }
            }
        } else {
            auto pos1 = std::lower_bound(index.randstrobes.begin() + position_start1, index.randstrobes.begin() + position_end1, RefRandstrobe{q1, 0, 0}, cmp);
            if (pos1->hash == q1) positions[1] = pos1 - index.randstrobes.begin();
        }

        if (position_start2 == position_end2) {
            positions[2] = -1;
        } else if (position_end2 - position_start2 < MAX_LINEAR_SEARCH) {
            for ( ; position_start2 < position_end2; ++position_start2) {
                if (index.randstrobes[position_start2].hash == q2) {
                    positions[2] = position_start2;
                    break;
                }
                if (index.randstrobes[position_start2].hash > q2) {
                    positions[2] = -1;
                    break;
                }
            }
        } else {
            auto pos2 = std::lower_bound(index.randstrobes.begin() + position_start2, index.randstrobes.begin() + position_end2, RefRandstrobe{q2, 0, 0}, cmp);
            if (pos2->hash == q2) positions[2] = pos2 - index.randstrobes.begin();
        }

        if (position_start3 == position_end3) {
            positions[3] = -1;
        } else if (position_end3 - position_start3 < MAX_LINEAR_SEARCH) {
            for ( ; position_start3 < position_end3; ++position_start3) {
                if (index.randstrobes[position_start3].hash == q3) {
                    positions[3] = position_start3;
                    break;
                }
                if (index.randstrobes[position_start3].hash > q3) {
                    positions[3] = -1;
                    break;
                }
            }
        } else {
            auto pos3 = std::lower_bound(index.randstrobes.begin() + position_start3, index.randstrobes.begin() + position_end3, RefRandstrobe{q3, 0, 0}, cmp);
            if (pos3->hash == q3) positions[3] = pos3 - index.randstrobes.begin();
        }

        for(int j = 0; j < 4; j++) {
            auto q = query_randstrobes[i + j];
            if (positions[j] != index.end()){
                total_hits++;
                auto res = index.is_filtered(positions[j]);
                if (res) {
                    continue;
                }
                nr_good_hits++;
                add_to_hits_per_ref(hits_per_ref[q.is_reverse], q.start, q.end, index, positions[j]);
            }

        }
    }
    for (; i < query_randstrobes.size(); i++) {
        auto q = query_randstrobes[i];
        size_t position = index.find(q.hash);
        if (position != index.end()){
            total_hits++;
            auto res = index.is_filtered(position);
            if (res) {
                continue;
            }
            nr_good_hits++;
            add_to_hits_per_ref(hits_per_ref[q.is_reverse], q.start, q.end, index, position);
        }
    }
#else
    for (const auto &q : query_randstrobes) {
        size_t position = index.find(q.hash);
        if (position != index.end()){
            total_hits++;
            auto res = index.is_filtered(position);
            if (res) {
                continue;
            }
            nr_good_hits++;
            add_to_hits_per_ref(hits_per_ref[q.is_reverse], q.start, q.end, index, position);
        }
    }
#endif
    float nonrepetitive_fraction = total_hits > 0 ? ((float) nr_good_hits) / ((float) total_hits) : 1.0;
    //auto nams = merge_hits_into_nams_forward_and_reverse(hits_per_ref, index.k(), false);
    auto nams = merge_hits_into_nams_forward_and_reverse(hits_per_ref, index.k(), true);


    return make_pair(nonrepetitive_fraction, nams);
}
#endif


/*
 * Find a query’s NAMs, using also some of the randstrobes that occur more often
 * than filter_cutoff.
 *
 */
//struct RescueHit {
//    size_t position;
//    unsigned int count;
//    unsigned int query_start;
//    unsigned int query_end;
//
//};

bool cmp1 (const RescueHit& a, const RescueHit& b) {
    return std::tie(a.count, a.query_start, a.query_end)
           < std::tie(b.count, b.query_start, b.query_end);
}

bool cmp2 (const RescueHit& a, const RescueHit& b) {
    //return std::tie(a.query_start, a.position)
    //       < std::tie(b.query_start, b.position);
    return a.query_start < b.query_start;
}


std::vector<Nam> find_nams_rescue(
    const QueryRandstrobeVector &query_randstrobes,
    const StrobemerIndex& index,
    unsigned int rescue_cutoff
) {


#ifdef pre_sort

    std::array<robin_hood::unordered_map<unsigned int, std::vector<Hit>>, 2> hits_per_ref;
    std::vector<RescueHit> hits_fw;
    std::vector<RescueHit> hits_rc;
    //hits_per_ref[0].reserve(100);
    //hits_per_ref[1].reserve(100);
    //hits_fw.reserve(5000);
    //hits_rc.reserve(5000);

    for (auto &qr : query_randstrobes) {
        size_t position = index.find(qr.hash);
        if (position != index.end()) {
            unsigned int count = index.get_count(position);
            RescueHit rh{position, count, qr.start, qr.end};
            if (qr.is_reverse){
                hits_rc.push_back(rh);
            } else {
                hits_fw.push_back(rh);
            }
        }
    }

    std::sort(hits_fw.begin(), hits_fw.end(), cmp1);
    std::sort(hits_rc.begin(), hits_rc.end(), cmp1);
    std::vector<RescueHit> rhs[2];
    size_t is_revcomp = 0;
    for (auto& rescue_hits : {hits_fw, hits_rc}) {
        int cnt = 0;
        for (auto &rh : rescue_hits) {
            if ((rh.count > rescue_cutoff && cnt >= 5) || rh.count > 100) {
                break;
            }
            rhs[is_revcomp].push_back(rh);
            add_to_hits_per_ref_pre(hits_per_ref[is_revcomp], rh.query_start, rh.query_end, index, rh.position);
            cnt++;
        }
        is_revcomp++;
    }


    std::sort(rhs[0].begin(), rhs[0].end(), cmp2);
    std::sort(rhs[1].begin(), rhs[1].end(), cmp2);
    for(int i = 0; i < 2; i++) {
        for(auto &rh : rhs[i]) {
            add_to_hits_per_ref(hits_per_ref[i], rh.query_start, rh.query_end, index, rh.position);
//            add_to_hits_per_ref_fast(hits_per_ref[i], rh.query_start, rh.query_end, index, rh.position);
        }
    }
    return merge_hits_into_nams_forward_and_reverse_fast(hits_per_ref, index.k(), false);
    //return merge_hits_into_nams_forward_and_reverse_fast(hits_per_ref, index.k(), true);

#else

    std::array<robin_hood::unordered_map<unsigned int, std::vector<Hit>>, 2> hits_per_ref;
//    std::array<std::map<unsigned int, std::vector<Hit>>, 2> hits_per_ref;
    std::vector<RescueHit> hits_fw;
    std::vector<RescueHit> hits_rc;
    //hits_per_ref[0].reserve(100);
    //hits_per_ref[1].reserve(100);
    //hits_fw.reserve(5000);
    //hits_rc.reserve(5000);

    for (auto &qr : query_randstrobes) {
        size_t position = index.find(qr.hash);
        if (position != index.end()) {
            unsigned int count = index.get_count(position);
            RescueHit rh{position, count, qr.start, qr.end};
            if (qr.is_reverse){
                hits_rc.push_back(rh);
            } else {
                hits_fw.push_back(rh);
            }
        }
    }

    std::sort(hits_fw.begin(), hits_fw.end(), cmp1);
    std::sort(hits_rc.begin(), hits_rc.end(), cmp1);
    std::vector<RescueHit> rhs[2];
    size_t is_revcomp = 0;
    for (auto& rescue_hits : {hits_fw, hits_rc}) {
        int cnt = 0;
        for (auto &rh : rescue_hits) {
            if ((rh.count > rescue_cutoff && cnt >= 5) || rh.count > 100) {
                break;
            }
            add_to_hits_per_ref(hits_per_ref[is_revcomp], rh.query_start, rh.query_end, index, rh.position);
            cnt++;
        }
        is_revcomp++;
    }


    return merge_hits_into_nams_forward_and_reverse(hits_per_ref, index.k(), true);

#endif
}

std::ostream& operator<<(std::ostream& os, const Nam& n) {
    os << "Nam(ref_id=" << n.ref_id << ", query: " << n.query_start << ".." << n.query_end << ", ref: " << n.ref_start << ".." << n.ref_end << ", score=" << n.score << ")";
    return os;
}
