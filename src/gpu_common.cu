#include "gpu_common.h"
#include "my_struct.hpp" // For Nam definition

__device__ __host__ void GPUInsertSizeDistribution::update(int dist) {
    if (dist >= 2000) {
        return;
    }
    const float e = dist - mu;
    mu += e / sample_size;
    SSE += e * (dist - mu);
    if (sample_size > 1) {
        V = SSE / (sample_size - 1.0);
    } else {
        V = SSE;
    }
    sigma = sqrtf(V);
    sample_size = sample_size + 1.0;
    if (mu < 0) {
        printf("mu negative, mu: %f sigma: %f SSE: %f sample size: %f\n", mu, sigma, SSE, sample_size);
        assert(false);
    }
    if (SSE < 0) {
        printf("SSE negative, mu: %f sigma: %f SSE: %f sample size: %f\n", mu, sigma, SSE, sample_size);
        assert(false);
    }
}


__device__ void print_nam(Nam nam) {
    printf("nam_id: %d, ref_id: %d, ref_start: %d, ref_end: %d, query_start: %d, query_end: %d, n_hits: %d, is_rc: %d\n",
           nam.nam_id, nam.ref_id, nam.ref_start, nam.ref_end, nam.query_start, nam.query_end, nam.n_hits, nam.is_rc);
}

__device__ void print_str(my_string str) {
    for(int i = 0; i < str.size(); i++) {
        printf("%c", str[i]);
    }
    printf("\n");
}