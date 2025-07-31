#ifndef STROBEALIGN_NAM_HPP
#define STROBEALIGN_NAM_HPP

#include <vector>
#include <set>
#include <array>
#include "index.hpp"
#include "randstrobes.hpp"
#include "my_struct.hpp"

std::pair<float, std::vector<Nam>> find_nams(
    const QueryRandstrobeVector &query_randstrobes,
    const StrobemerIndex& index
);

std::vector<Nam> find_nams_rescue(
    const QueryRandstrobeVector &query_randstrobes,
    const StrobemerIndex& index,
    unsigned int rescue_cutoff
);

std::ostream& operator<<(std::ostream& os, const Nam& nam);

#endif
