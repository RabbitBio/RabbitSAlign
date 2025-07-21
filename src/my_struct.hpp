#ifndef MY_STRUCT_HPP
#define MY_STRUCT_HPP

#include <iostream>
#include <stdexcept>
#include <cassert>

#include "FastxStream.h"
#include "FastxChunk.h"
#include "DataQueue.h"
#include "Formater.h"

#ifdef __CUDACC__
#define CUDA_HOST __host__
#define CUDA_DEV __device__
#else
#define CUDA_HOST
#define CUDA_DEV
#endif

CUDA_DEV static const uint8_t gpu_nt2int_mod8[8] = {
    0, // 0
    0, // 1 → 'A'
    0, // 2
    1, // 3 → 'C'
    3, // 4 → 'T'
    0, // 5
    0, // 6
    2  // 7 → 'G'
};

CUDA_HOST CUDA_DEV static const char rc_gpu_nt2nt_mod8[8] = {
    'A', // 0
    'T', // 1 → 'A'
    'A', // 2
    'G', // 3 → 'C'
    'A', // 4 → 'T'
    'A', // 5
    'A', // 6
    'C'  // 7 → 'G'
};

CUDA_HOST CUDA_DEV static const char rc_gpu_nt2nt[256] = {
    'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
    'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
    'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
    'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
    'N', 'T', 'N', 'G',  'N', 'N', 'N', 'C',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
    'N', 'N', 'N', 'N',  'A', 'A', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
    'N', 'T', 'N', 'G',  'N', 'N', 'N', 'C',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
    'N', 'N', 'N', 'N',  'A', 'A', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
    'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
    'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
    'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
    'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
    'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
    'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
    'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',
    'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N',  'N', 'N', 'N', 'N'
};


CUDA_DEV static unsigned char gpu_seq_nt4_table[256] = {
    0, 1, 2, 3,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  3, 3, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  3, 3, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4
};
/*
 * A -- 65 -- 0
 * C -- 67 -- 1
 * G -- 71 -- 2
 * T -- 84 -- 3
 * U -- 85 -- 3
 */

struct neoRcRef{
    neoReference read;
    char* rc;
};


template <typename T>
CUDA_HOST CUDA_DEV T my_max(T a, T b) {
    return a > b ? a : b;
}

template <typename T>
CUDA_HOST CUDA_DEV T my_min(T a, T b) {
    return a < b ? a : b;
}

template <typename T>
CUDA_HOST CUDA_DEV T my_abs(T a) {
    return a < 0 ? -a : a;
}


// Non-overlapping approximate match
struct Nam {
    int nam_id{0};
    int query_start{0};
    int query_end{0};
    int query_prev_hit_startpos{0};
    int ref_start{0};
    int ref_end{0};
    int ref_prev_hit_startpos{0};
    int n_hits{0};
    int ref_id{0};
    float score{0};
//    unsigned int previous_query_start;
//    unsigned int previous_ref_start;
    bool is_rc{false};

    CUDA_HOST CUDA_DEV int ref_span() const {
        return ref_end - ref_start;
    }

    CUDA_HOST CUDA_DEV int query_span() const {
        return query_end - query_start;
    }

    bool operator < (const Nam& nn) const {
        if(query_end != nn.query_end) return query_end < nn.query_end;
        return nam_id < nn.nam_id;
    }
};

struct Hit {
    int query_start;
    int query_end;
    int ref_start;
    int ref_end;
    CUDA_HOST CUDA_DEV bool operator<(const Hit& other) const {
        if(query_start == other.query_start) return ref_start < other.ref_start;
        return query_start < other.query_start;
    }
    CUDA_HOST CUDA_DEV  bool operator==(const Hit& other) const {
        return query_start == other.query_start &&
               query_end == other.query_end &&
               ref_start == other.ref_start &&
               ref_end == other.ref_end;
    }
};

struct RescueHit {
    size_t position;
    unsigned int count;
    unsigned int query_start;
    unsigned int query_end;
    CUDA_HOST CUDA_DEV bool operator<(const RescueHit& other) const {
        if (count != other.count) return count < other.count;
        if (query_start != other.query_start) return query_start < other.query_start;
        return query_end < other.query_end;
    }
};

#define my_malloc(size) my_malloc_impl((size), __FILE__, __LINE__, __func__)

CUDA_DEV void* my_malloc_impl(size_t size, const char* file, int line, const char* func);

CUDA_DEV void my_free(void* ptr);

CUDA_HOST void init_mm(uint64_t num_bytes, uint64_t seed);
CUDA_HOST void free_mm();
CUDA_HOST void print_mm();


struct my_string {
    char* data = nullptr;
    int slen;
    CUDA_HOST CUDA_DEV my_string() : data(nullptr), slen(0) {}
    CUDA_DEV my_string(char* str, int len) {
        slen = len;
        data = str;
    }
    CUDA_HOST CUDA_DEV ~my_string() {
        data = nullptr;
    }
    CUDA_DEV int length() const {
        return slen;
    }
    CUDA_DEV int size() const {
        return slen;
    }
    CUDA_DEV const char* c_str() const {
        return data;
    }
    CUDA_DEV char operator[](int index) const {
        return data[index];
    }
    CUDA_DEV char& operator[](int index) {
        return data[index];
    }
    CUDA_DEV my_string substr(int start, int len) const {
        int real_len = my_min(len, slen - start);
        return my_string(data + start, real_len);
    }
    CUDA_DEV int find(const my_string& str, int start = 0) const {
        for (int i = start; i <= slen - str.slen; ++i) {
            bool found = true;
            for (int j = 0; j < str.slen; ++j) {
                if (data[i + j] != str[j]) {
                    found = false;
                    break;
                }
            }
            if (found) return i;
        }
        return -1;
    }
    CUDA_DEV bool operator== (const my_string& other) const {
        if (slen != other.slen) return false;
        for (int i = 0; i < slen; ++i) {
            if (data[i] != other[i]) return false;
        }
        return true;
    }
};

#define VEC_PUSH_BACK(item) VEC_PUSH_BACK_impl((item), __FILE__, __LINE__, __func__)

template <typename T>
struct my_vector {
    T* data = nullptr;
    int length;
    int capacity;

    CUDA_DEV my_vector(int N = 4, const char* file = nullptr, int line = 0, const char* func = nullptr) {
        capacity = N;
        length = 0;
        if (N > 0) data = (T*)my_malloc_impl(capacity * sizeof(T), file, line, func);
        else data = nullptr;
    }

    CUDA_DEV void VEC_INIT_impl(int N = 4, const char* file = nullptr, int line = 0, const char* func = nullptr) {
        capacity = N;
        length = 0;
        data = (T*)my_malloc_impl(capacity * sizeof(T), file, line, func);
    }

    CUDA_HOST CUDA_DEV ~my_vector() {
        if (data != nullptr) my_free(data);
        data = nullptr;
    }

    CUDA_DEV void resize(int new_capacity, const char* file = nullptr, int line = 0, const char* func = nullptr) {
        T* new_data;
        new_data = (T*)my_malloc_impl(new_capacity * sizeof(T), file, line, func);
        for (int i = 0; i < length; ++i) new_data[i] = data[i];
        if (data != nullptr) my_free(data);
        data = new_data;
        capacity = new_capacity;
    }

    CUDA_DEV void VEC_PUSH_BACK_impl(const T& value, const char* file, int line, const char* func) {
        if (length == capacity) {
            resize(capacity == 0 ? 1 : capacity * 2, file, line, func);
        }
        data[length++] = value;
    }
    CUDA_DEV void emplace_back() {
        if (length == capacity) {
            resize(capacity == 0 ? 1 : capacity * 2);
        }
        data[length++] = T();
    }

    CUDA_HOST CUDA_DEV int size() const {
        return length;
    }

    CUDA_HOST CUDA_DEV void clear() {
        length = 0;
    }

    CUDA_HOST CUDA_DEV void release() {
        if (data != nullptr) my_free(data);
        data = nullptr;
        length = 0;
        capacity = 0;
    }

    CUDA_HOST CUDA_DEV T& operator[](int index) {
        return data[index];
    }

    CUDA_HOST CUDA_DEV const T& operator[](int index) const {
        return data[index];
    }

    CUDA_HOST CUDA_DEV T& back() {
        return data[length - 1];
    }

    CUDA_HOST CUDA_DEV void move_from(my_vector<T>& src) {
        if (data != nullptr) my_free(data);
        data = src.data;
        length = src.length;
        capacity = src.capacity;
        src.data = nullptr;
        src.length = 0;
        src.capacity = 0;
    }

    CUDA_HOST CUDA_DEV bool empty() const {
        return length == 0;
    }
};

template <typename T>
CUDA_DEV void my_swap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}


template <typename T1, typename T2>
struct my_pair {
    T1 first;
    T2 second;
    CUDA_DEV bool operator<(const my_pair& other) const {
        if (first < other.first) return true;
        if (first > other.first) return false;
        return second < other.second;
    }
};


#endif // MY_STRUCT_HPP

