//
// Created by ylf9811 on 2024/1/4.
//

#ifndef STROBEALIGN_GASAL2_SSW_H
#define STROBEALIGN_GASAL2_SSW_H
#include "aligner.hpp"
#include "include/gasal_header.h"
#include <vector>
#include <cmath>
#include <sstream>
struct gasal_tmp_res{
    int score;
    int query_start;
    int query_end;
    int ref_start;
    int ref_end;
    std::string cigar_str;
};
void solve_ssw_on_gpu(std::vector<gasal_tmp_res> &gasal_results, std::vector<std::string> &todo_querys, std::vector<std::string> &todo_refs,
                      int match_score = 2, int mismatch_score = 8, int gap_open_score = 12, int gap_extend_score = 1);
#endif  //STROBEALIGN_GASAL2_SSW_H
