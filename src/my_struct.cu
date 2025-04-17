#include "my_struct.hpp"

#include <gallatin/allocators/global_allocator.cuh>


__device__ void* my_malloc(size_t size) {
    //if(size < 128) size = 128;
    //assert(size <= 65536);
    void* ptr = gallatin::allocators::global_malloc(size);
    if (ptr == nullptr) {
        printf("mm_memory allocation failed %d\n", size);
        assert(false);
    }
    return ptr;
}

__device__ void my_free(void* ptr) {
    gallatin::allocators::global_free(ptr);
}

__host__ void init_mm(uint64_t num_bytes, uint64_t seed) {
    gallatin::allocators::init_global_allocator(num_bytes, seed);
}

__host__ void free_mm() {
    gallatin::allocators::free_global_allocator();
}

__host__ void print_mm() {
    gallatin::allocators::print_global_stats();
}
