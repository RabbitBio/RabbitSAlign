#ifndef _H_MY_SHARED_MERGE_KERNELS
#define _H_MY_SHARED_MERGE_KERNELS

#include <limits>

#include "bb_exch.h"

// segsize 512
// 128 threads per block | 4 ppt
template<class K, class T>
__global__
void my_blk128_ppt4_shared(K *key, T *val, K *keyB, T *valB, int *segs, int *bin, int bin_size) {
	const int tid = threadIdx.x;
	const int bin_it = blockIdx.x;
	__shared__ K smem[512];
	__shared__ int tmem[512];
	const int bit1 = (tid>>0)&0x1;
	const int bit2 = (tid>>1)&0x1;
	const int bit3 = (tid>>2)&0x1;
	const int bit4 = (tid>>3)&0x1;
	const int bit5 = (tid>>4)&0x1;
	const int tid1 = threadIdx.x & 31;
	const int warp_id = threadIdx.x / 32;
	K rg_k0;
	K rg_k1;
	K rg_k2;
	K rg_k3;
	int rg_v0;
	int rg_v1;
	int rg_v2;
	int rg_v3;
	int k;
	int seg_size;
	int ext_seg_size;

	if(bin_it < bin_size) {
		k = segs[bin[bin_it]];
		seg_size = segs[bin[bin_it]+1] - segs[bin[bin_it]];
		// ext_seg_size is the seg_sizes next biggest multiple of [warp_lengt/2]
		ext_seg_size = ((seg_size + 63) / 64) * 64;
		// Calculate the number of big and small warps needed
		int big_wp = (ext_seg_size - blockDim.x * 2) / 64;
		int sml_wp = blockDim.x / 32 - big_wp;
		// sml_len is the total length of small warps
		int sml_len = sml_wp * 64;
		// big_warp_id is the id for a warp, if it's big; starting for 0 at the first big warp
		const int big_warp_id = (warp_id - sml_wp < 0)? 0: warp_id - sml_wp;
		bool sml_warp = warp_id < sml_wp;

		if (sml_warp) {
			rg_k0  = key[k+tid1+(warp_id<<6)+0	];
			rg_k1  = key[k+tid1+(warp_id<<6)+32	];
			rg_v0  = tid1+(warp_id<<6)+0	;
			rg_v1  = tid1+(warp_id<<6)+32	;

			// sort 64 elements
			// exch_intxn: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
			// exch_intxn: generate exch_intxn()
			exch_intxn(rg_k0, rg_k1, rg_v0, rg_v1, 0x1, bit1);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
			// exch_intxn: generate exch_intxn()
			exch_intxn(rg_k0, rg_k1, rg_v0, rg_v1, 0x3, bit2);
			// exch_paral: generate exch_paral()
			exch_paral(rg_k0, rg_k1, rg_v0, rg_v1, 0x1, bit1);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
			// exch_intxn: generate exch_intxn()
			exch_intxn(rg_k0, rg_k1, rg_v0, rg_v1, 0x7, bit3);
			// exch_paral: generate exch_paral()
			exch_paral(rg_k0, rg_k1, rg_v0, rg_v1, 0x2, bit2);
			// exch_paral: generate exch_paral()
			exch_paral(rg_k0, rg_k1, rg_v0, rg_v1, 0x1, bit1);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
			// exch_intxn: generate exch_intxn()
			exch_intxn(rg_k0, rg_k1, rg_v0, rg_v1, 0xf, bit4);
			// exch_paral: generate exch_paral()
			exch_paral(rg_k0, rg_k1, rg_v0, rg_v1, 0x4, bit3);
			// exch_paral: generate exch_paral()
			exch_paral(rg_k0, rg_k1, rg_v0, rg_v1, 0x2, bit2);
			// exch_paral: generate exch_paral()
			exch_paral(rg_k0, rg_k1, rg_v0, rg_v1, 0x1, bit1);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
			// exch_intxn: generate exch_intxn()
			exch_intxn(rg_k0, rg_k1, rg_v0, rg_v1, 0x1f, bit5);
			// exch_paral: generate exch_paral()
			exch_paral(rg_k0, rg_k1, rg_v0, rg_v1, 0x8, bit4);
			// exch_paral: generate exch_paral()
			exch_paral(rg_k0, rg_k1, rg_v0, rg_v1, 0x4, bit3);
			// exch_paral: generate exch_paral()
			exch_paral(rg_k0, rg_k1, rg_v0, rg_v1, 0x2, bit2);
			// exch_paral: generate exch_paral()
			exch_paral(rg_k0, rg_k1, rg_v0, rg_v1, 0x1, bit1);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
		} else {
			rg_k0  = (sml_len+tid1+(big_warp_id<<7)+0	<seg_size)?key[k+sml_len+tid1+(big_warp_id<<7)+0	]:std::numeric_limits<K>::max();
			rg_k1  = (sml_len+tid1+(big_warp_id<<7)+32	<seg_size)?key[k+sml_len+tid1+(big_warp_id<<7)+32	]:std::numeric_limits<K>::max();
			rg_k2  = (sml_len+tid1+(big_warp_id<<7)+64	<seg_size)?key[k+sml_len+tid1+(big_warp_id<<7)+64	]:std::numeric_limits<K>::max();
			rg_k3  = (sml_len+tid1+(big_warp_id<<7)+96	<seg_size)?key[k+sml_len+tid1+(big_warp_id<<7)+96	]:std::numeric_limits<K>::max();
			if (sml_len+tid1+(big_warp_id<<7)+0	<seg_size) rg_v0  = sml_len+tid1+(big_warp_id<<7)+0	;
			if (sml_len+tid1+(big_warp_id<<7)+32	<seg_size) rg_v1  = sml_len+tid1+(big_warp_id<<7)+32	;
			if (sml_len+tid1+(big_warp_id<<7)+64	<seg_size) rg_v2  = sml_len+tid1+(big_warp_id<<7)+64	;
			if (sml_len+tid1+(big_warp_id<<7)+96	<seg_size) rg_v3  = sml_len+tid1+(big_warp_id<<7)+96	;

			// sort 128 elements
			// exch_intxn: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
			CMP_SWP(K, rg_k2, rg_k3, int, rg_v2, rg_v3);
			// exch_intxn: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k3, int, rg_v0, rg_v3);
			CMP_SWP(K, rg_k1, rg_k2, int, rg_v1, rg_v2);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
			CMP_SWP(K, rg_k2, rg_k3, int, rg_v2, rg_v3);
			// exch_intxn: generate exch_intxn()
			exch_intxn(rg_k0, rg_k1, rg_k2, rg_k3, rg_v0, rg_v1, rg_v2, rg_v3, 0x1, bit1);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k2, int, rg_v0, rg_v2);
			CMP_SWP(K, rg_k1, rg_k3, int, rg_v1, rg_v3);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
			CMP_SWP(K, rg_k2, rg_k3, int, rg_v2, rg_v3);
			// exch_intxn: generate exch_intxn()
			exch_intxn(rg_k0, rg_k1, rg_k2, rg_k3, rg_v0, rg_v1, rg_v2, rg_v3, 0x3, bit2);
			// exch_paral: generate exch_paral()
			exch_paral(rg_k0, rg_k1, rg_k2, rg_k3, rg_v0, rg_v1, rg_v2, rg_v3, 0x1, bit1);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k2, int, rg_v0, rg_v2);
			CMP_SWP(K, rg_k1, rg_k3, int, rg_v1, rg_v3);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
			CMP_SWP(K, rg_k2, rg_k3, int, rg_v2, rg_v3);
			// exch_intxn: generate exch_intxn()
			exch_intxn(rg_k0, rg_k1, rg_k2, rg_k3, rg_v0, rg_v1, rg_v2, rg_v3, 0x7, bit3);
			// exch_paral: generate exch_paral()
			exch_paral(rg_k0, rg_k1, rg_k2, rg_k3, rg_v0, rg_v1, rg_v2, rg_v3, 0x2, bit2);
			// exch_paral: generate exch_paral()
			exch_paral(rg_k0, rg_k1, rg_k2, rg_k3, rg_v0, rg_v1, rg_v2, rg_v3, 0x1, bit1);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k2, int, rg_v0, rg_v2);
			CMP_SWP(K, rg_k1, rg_k3, int, rg_v1, rg_v3);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
			CMP_SWP(K, rg_k2, rg_k3, int, rg_v2, rg_v3);
			// exch_intxn: generate exch_intxn()
			exch_intxn(rg_k0, rg_k1, rg_k2, rg_k3, rg_v0, rg_v1, rg_v2, rg_v3, 0xf, bit4);
			// exch_paral: generate exch_paral()
			exch_paral(rg_k0, rg_k1, rg_k2, rg_k3, rg_v0, rg_v1, rg_v2, rg_v3, 0x4, bit3);
			// exch_paral: generate exch_paral()
			exch_paral(rg_k0, rg_k1, rg_k2, rg_k3, rg_v0, rg_v1, rg_v2, rg_v3, 0x2, bit2);
			// exch_paral: generate exch_paral()
			exch_paral(rg_k0, rg_k1, rg_k2, rg_k3, rg_v0, rg_v1, rg_v2, rg_v3, 0x1, bit1);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k2, int, rg_v0, rg_v2);
			CMP_SWP(K, rg_k1, rg_k3, int, rg_v1, rg_v3);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
			CMP_SWP(K, rg_k2, rg_k3, int, rg_v2, rg_v3);
			// exch_intxn: generate exch_intxn()
			exch_intxn(rg_k0, rg_k1, rg_k2, rg_k3, rg_v0, rg_v1, rg_v2, rg_v3, 0x1f, bit5);
			// exch_paral: generate exch_paral()
			exch_paral(rg_k0, rg_k1, rg_k2, rg_k3, rg_v0, rg_v1, rg_v2, rg_v3, 0x8, bit4);
			// exch_paral: generate exch_paral()
			exch_paral(rg_k0, rg_k1, rg_k2, rg_k3, rg_v0, rg_v1, rg_v2, rg_v3, 0x4, bit3);
			// exch_paral: generate exch_paral()
			exch_paral(rg_k0, rg_k1, rg_k2, rg_k3, rg_v0, rg_v1, rg_v2, rg_v3, 0x2, bit2);
			// exch_paral: generate exch_paral()
			exch_paral(rg_k0, rg_k1, rg_k2, rg_k3, rg_v0, rg_v1, rg_v2, rg_v3, 0x1, bit1);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k2, int, rg_v0, rg_v2);
			CMP_SWP(K, rg_k1, rg_k3, int, rg_v1, rg_v3);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
			CMP_SWP(K, rg_k2, rg_k3, int, rg_v2, rg_v3);
		}

		// Store register results to shared memory
		if (sml_warp) {
			smem[(warp_id<<6) + (tid1<<1) + 0] = rg_k0;
			smem[(warp_id<<6) + (tid1<<1) + 1] = rg_k1;
			tmem[(warp_id<<6) + (tid1<<1) + 0] = rg_v0;
			tmem[(warp_id<<6) + (tid1<<1) + 1] = rg_v1;
		} else {
			smem[sml_len+(big_warp_id<<7) + (tid1<<2) + 0] = rg_k0;
			smem[sml_len+(big_warp_id<<7) + (tid1<<2) + 1] = rg_k1;
			smem[sml_len+(big_warp_id<<7) + (tid1<<2) + 2] = rg_k2;
			smem[sml_len+(big_warp_id<<7) + (tid1<<2) + 3] = rg_k3;
			tmem[sml_len+(big_warp_id<<7) + (tid1<<2) + 0] = rg_v0;
			tmem[sml_len+(big_warp_id<<7) + (tid1<<2) + 1] = rg_v1;
			tmem[sml_len+(big_warp_id<<7) + (tid1<<2) + 2] = rg_v2;
			tmem[sml_len+(big_warp_id<<7) + (tid1<<2) + 3] = rg_v3;
		}
		__syncthreads();

		// Merge in 2 steps
		int grp_start_wp_id;
		int grp_start_off;
		int lhs_len;
		int rhs_len;
		int gran;
		int s_a;
		int s_b;
		bool p;
		K tmp_k0;
		K tmp_k1;
		int tmp_v0;
		int tmp_v1;
		K *start;

		// Step 0
		grp_start_wp_id = ((warp_id>>1)<<1);
		grp_start_off = (grp_start_wp_id<sml_wp)?grp_start_wp_id*64:sml_len+(grp_start_wp_id-sml_wp) * 128;
		lhs_len = ((grp_start_wp_id + 0 < sml_wp) ? 64 : 128);
		rhs_len = ((grp_start_wp_id + 1 < sml_wp) ? 64 : 128);
		// gran denotes where each thread will write its merged entries
		gran = (warp_id<sml_wp) ? (tid1<<1) : (tid1<<2);
		if ((warp_id & 1) == 0) {
			gran += 0;
		}
		if ((warp_id & 1) == 1) {
			gran += ((grp_start_wp_id + 0 < sml_wp) ? 64 : 128);
		}

		start = smem + grp_start_off;
		s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
		s_b = lhs_len + gran - s_a;
		if (sml_warp) {
			tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
			tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
			if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
			if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k0 = p ? tmp_k0 : tmp_k1;
			rg_v0 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k1 = p ? tmp_k0 : tmp_k1;
			rg_v1 = p ? tmp_v0 : tmp_v1;
		} else {
			tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
			tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
			if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
			if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k0 = p ? tmp_k0 : tmp_k1;
			rg_v0 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k1 = p ? tmp_k0 : tmp_k1;
			rg_v1 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k2 = p ? tmp_k0 : tmp_k1;
			rg_v2 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k3 = p ? tmp_k0 : tmp_k1;
			rg_v3 = p ? tmp_v0 : tmp_v1;
		}

		__syncthreads();
		// Store merged results back into shared memory
		if (sml_warp) {
			smem[grp_start_off + gran + 0] = rg_k0;
			smem[grp_start_off + gran + 1] = rg_k1;
			tmem[grp_start_off + gran + 0] = rg_v0;
			tmem[grp_start_off + gran + 1] = rg_v1;
		} else {
			smem[grp_start_off + gran + 0] = rg_k0;
			smem[grp_start_off + gran + 1] = rg_k1;
			smem[grp_start_off + gran + 2] = rg_k2;
			smem[grp_start_off + gran + 3] = rg_k3;
			tmem[grp_start_off + gran + 0] = rg_v0;
			tmem[grp_start_off + gran + 1] = rg_v1;
			tmem[grp_start_off + gran + 2] = rg_v2;
			tmem[grp_start_off + gran + 3] = rg_v3;
		}
		__syncthreads();

		// Step 1
		grp_start_wp_id = ((warp_id>>2)<<2);
		grp_start_off = (grp_start_wp_id<sml_wp)?grp_start_wp_id*64:sml_len+(grp_start_wp_id-sml_wp) * 128;
		lhs_len = ((grp_start_wp_id + 0 < sml_wp) ? 64 : 128)+
		          ((grp_start_wp_id + 1 < sml_wp) ? 64 : 128);
		rhs_len = ((grp_start_wp_id + 2 < sml_wp) ? 64 : 128)+
		          ((grp_start_wp_id + 3 < sml_wp) ? 64 : 128);
		// gran denotes where each thread will write its merged entries
		gran = (warp_id<sml_wp) ? (tid1<<1) : (tid1<<2);
		if ((warp_id & 3) == 0) {
			gran += 0;
		}
		if ((warp_id & 3) == 1) {
			gran += ((grp_start_wp_id + 0 < sml_wp) ? 64 : 128);
		}
		if ((warp_id & 3) == 2) {
			gran += ((grp_start_wp_id + 0 < sml_wp) ? 64 : 128)+
			        ((grp_start_wp_id + 1 < sml_wp) ? 64 : 128);
		}
		if ((warp_id & 3) == 3) {
			gran += ((grp_start_wp_id + 0 < sml_wp) ? 64 : 128)+
			        ((grp_start_wp_id + 1 < sml_wp) ? 64 : 128)+
			        ((grp_start_wp_id + 2 < sml_wp) ? 64 : 128);
		}

		start = smem + grp_start_off;
		s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
		s_b = lhs_len + gran - s_a;
		if (sml_warp) {
			tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
			tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
			if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
			if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k0 = p ? tmp_k0 : tmp_k1;
			rg_v0 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k1 = p ? tmp_k0 : tmp_k1;
			rg_v1 = p ? tmp_v0 : tmp_v1;
		} else {
			tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
			tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
			if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
			if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k0 = p ? tmp_k0 : tmp_k1;
			rg_v0 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k1 = p ? tmp_k0 : tmp_k1;
			rg_v1 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k2 = p ? tmp_k0 : tmp_k1;
			rg_v2 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k3 = p ? tmp_k0 : tmp_k1;
			rg_v3 = p ? tmp_v0 : tmp_v1;
		}

		if(sml_warp) {
			if((tid<<1)+0 <seg_size) keyB[k+(tid<<1)+0] = rg_k0;
			if((tid<<1)+1 <seg_size) keyB[k+(tid<<1)+1] = rg_k1;
			if((tid<<1)+0 <seg_size) valB[k+(tid<<1)+0] = val[k + rg_v0];
			if((tid<<1)+1 <seg_size) valB[k+(tid<<1)+1] = val[k + rg_v1];
		} else {
			if((tid<<2)+0 - sml_len <seg_size) keyB[k+(tid<<2)+0 - sml_len] = rg_k0;
			if((tid<<2)+1 - sml_len <seg_size) keyB[k+(tid<<2)+1 - sml_len] = rg_k1;
			if((tid<<2)+2 - sml_len <seg_size) keyB[k+(tid<<2)+2 - sml_len] = rg_k2;
			if((tid<<2)+3 - sml_len <seg_size) keyB[k+(tid<<2)+3 - sml_len] = rg_k3;
			if((tid<<2)+0 - sml_len <seg_size) valB[k+(tid<<2)+0 - sml_len] = val[k + rg_v0];
			if((tid<<2)+1 - sml_len <seg_size) valB[k+(tid<<2)+1 - sml_len] = val[k + rg_v1];
			if((tid<<2)+2 - sml_len <seg_size) valB[k+(tid<<2)+2 - sml_len] = val[k + rg_v2];
			if((tid<<2)+3 - sml_len <seg_size) valB[k+(tid<<2)+3 - sml_len] = val[k + rg_v3];
		}
	}
}

// segsize 4096
// 512 threads per block | 8 ppt | strided
template<class K, class T>
__global__
void my_blk512_ppt8_shared_strd(K *key, T *val, K *keyB, T *valB, int *segs, int *bin, int bin_size) {
	const int tid = threadIdx.x;
	const int bin_it = blockIdx.x;
	__shared__ K smem[4096];
	__shared__ int tmem[4096];
	const int bit1 = (tid>>0)&0x1;
	const int bit2 = (tid>>1)&0x1;
	const int bit3 = (tid>>2)&0x1;
	const int bit4 = (tid>>3)&0x1;
	const int bit5 = (tid>>4)&0x1;
	const int tid1 = threadIdx.x & 31;
	const int warp_id = threadIdx.x / 32;
	K rg_k0;
	K rg_k1;
	K rg_k2;
	K rg_k3;
	K rg_k4;
	K rg_k5;
	K rg_k6;
	K rg_k7;
	int rg_v0;
	int rg_v1;
	int rg_v2;
	int rg_v3;
	int rg_v4;
	int rg_v5;
	int rg_v6;
	int rg_v7;
	int k;
	int seg_size;
	int ext_seg_size;

	if(bin_it < bin_size) {
		k = segs[bin[bin_it]];
		seg_size = segs[bin[bin_it]+1] - segs[bin[bin_it]];
		// ext_seg_size is the seg_sizes next biggest multiple of [warp_lengt/2]
		ext_seg_size = ((seg_size + 127) / 128) * 128;
		// Calculate the number of big and small warps needed
		int big_wp = (ext_seg_size - blockDim.x * 4) / 128;
		int sml_wp = blockDim.x / 32 - big_wp;
		// sml_len is the total length of small warps
		int sml_len = sml_wp * 128;
		// big_warp_id is the id for a warp, if it's big; starting for 0 at the first big warp
		const int big_warp_id = (warp_id - sml_wp < 0)? 0: warp_id - sml_wp;
		bool sml_warp = warp_id < sml_wp;

		if (sml_warp) {
			rg_k0  = key[k+tid1+(warp_id<<7)+0	];
			rg_k1  = key[k+tid1+(warp_id<<7)+32	];
			rg_k2  = key[k+tid1+(warp_id<<7)+64	];
			rg_k3  = key[k+tid1+(warp_id<<7)+96	];
			rg_v0  = tid1+(warp_id<<7)+0	;
			rg_v1  = tid1+(warp_id<<7)+32	;
			rg_v2  = tid1+(warp_id<<7)+64	;
			rg_v3  = tid1+(warp_id<<7)+96	;

			// sort 128 elements
			// exch_intxn: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
			CMP_SWP(K, rg_k2, rg_k3, int, rg_v2, rg_v3);
			// exch_intxn: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k3, int, rg_v0, rg_v3);
			CMP_SWP(K, rg_k1, rg_k2, int, rg_v1, rg_v2);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
			CMP_SWP(K, rg_k2, rg_k3, int, rg_v2, rg_v3);
			// exch_intxn: generate exch_intxn()
			exch_intxn(rg_k0, rg_k1, rg_k2, rg_k3, rg_v0, rg_v1, rg_v2, rg_v3, 0x1, bit1);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k2, int, rg_v0, rg_v2);
			CMP_SWP(K, rg_k1, rg_k3, int, rg_v1, rg_v3);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
			CMP_SWP(K, rg_k2, rg_k3, int, rg_v2, rg_v3);
			// exch_intxn: generate exch_intxn()
			exch_intxn(rg_k0, rg_k1, rg_k2, rg_k3, rg_v0, rg_v1, rg_v2, rg_v3, 0x3, bit2);
			// exch_paral: generate exch_paral()
			exch_paral(rg_k0, rg_k1, rg_k2, rg_k3, rg_v0, rg_v1, rg_v2, rg_v3, 0x1, bit1);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k2, int, rg_v0, rg_v2);
			CMP_SWP(K, rg_k1, rg_k3, int, rg_v1, rg_v3);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
			CMP_SWP(K, rg_k2, rg_k3, int, rg_v2, rg_v3);
			// exch_intxn: generate exch_intxn()
			exch_intxn(rg_k0, rg_k1, rg_k2, rg_k3, rg_v0, rg_v1, rg_v2, rg_v3, 0x7, bit3);
			// exch_paral: generate exch_paral()
			exch_paral(rg_k0, rg_k1, rg_k2, rg_k3, rg_v0, rg_v1, rg_v2, rg_v3, 0x2, bit2);
			// exch_paral: generate exch_paral()
			exch_paral(rg_k0, rg_k1, rg_k2, rg_k3, rg_v0, rg_v1, rg_v2, rg_v3, 0x1, bit1);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k2, int, rg_v0, rg_v2);
			CMP_SWP(K, rg_k1, rg_k3, int, rg_v1, rg_v3);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
			CMP_SWP(K, rg_k2, rg_k3, int, rg_v2, rg_v3);
			// exch_intxn: generate exch_intxn()
			exch_intxn(rg_k0, rg_k1, rg_k2, rg_k3, rg_v0, rg_v1, rg_v2, rg_v3, 0xf, bit4);
			// exch_paral: generate exch_paral()
			exch_paral(rg_k0, rg_k1, rg_k2, rg_k3, rg_v0, rg_v1, rg_v2, rg_v3, 0x4, bit3);
			// exch_paral: generate exch_paral()
			exch_paral(rg_k0, rg_k1, rg_k2, rg_k3, rg_v0, rg_v1, rg_v2, rg_v3, 0x2, bit2);
			// exch_paral: generate exch_paral()
			exch_paral(rg_k0, rg_k1, rg_k2, rg_k3, rg_v0, rg_v1, rg_v2, rg_v3, 0x1, bit1);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k2, int, rg_v0, rg_v2);
			CMP_SWP(K, rg_k1, rg_k3, int, rg_v1, rg_v3);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
			CMP_SWP(K, rg_k2, rg_k3, int, rg_v2, rg_v3);
			// exch_intxn: generate exch_intxn()
			exch_intxn(rg_k0, rg_k1, rg_k2, rg_k3, rg_v0, rg_v1, rg_v2, rg_v3, 0x1f, bit5);
			// exch_paral: generate exch_paral()
			exch_paral(rg_k0, rg_k1, rg_k2, rg_k3, rg_v0, rg_v1, rg_v2, rg_v3, 0x8, bit4);
			// exch_paral: generate exch_paral()
			exch_paral(rg_k0, rg_k1, rg_k2, rg_k3, rg_v0, rg_v1, rg_v2, rg_v3, 0x4, bit3);
			// exch_paral: generate exch_paral()
			exch_paral(rg_k0, rg_k1, rg_k2, rg_k3, rg_v0, rg_v1, rg_v2, rg_v3, 0x2, bit2);
			// exch_paral: generate exch_paral()
			exch_paral(rg_k0, rg_k1, rg_k2, rg_k3, rg_v0, rg_v1, rg_v2, rg_v3, 0x1, bit1);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k2, int, rg_v0, rg_v2);
			CMP_SWP(K, rg_k1, rg_k3, int, rg_v1, rg_v3);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
			CMP_SWP(K, rg_k2, rg_k3, int, rg_v2, rg_v3);
		} else {
			rg_k0  = (sml_len+tid1+(big_warp_id<<8)+0	<seg_size)?key[k+sml_len+tid1+(big_warp_id<<8)+0	]:std::numeric_limits<K>::max();
			rg_k1  = (sml_len+tid1+(big_warp_id<<8)+32	<seg_size)?key[k+sml_len+tid1+(big_warp_id<<8)+32	]:std::numeric_limits<K>::max();
			rg_k2  = (sml_len+tid1+(big_warp_id<<8)+64	<seg_size)?key[k+sml_len+tid1+(big_warp_id<<8)+64	]:std::numeric_limits<K>::max();
			rg_k3  = (sml_len+tid1+(big_warp_id<<8)+96	<seg_size)?key[k+sml_len+tid1+(big_warp_id<<8)+96	]:std::numeric_limits<K>::max();
			rg_k4  = (sml_len+tid1+(big_warp_id<<8)+128	<seg_size)?key[k+sml_len+tid1+(big_warp_id<<8)+128	]:std::numeric_limits<K>::max();
			rg_k5  = (sml_len+tid1+(big_warp_id<<8)+160	<seg_size)?key[k+sml_len+tid1+(big_warp_id<<8)+160	]:std::numeric_limits<K>::max();
			rg_k6  = (sml_len+tid1+(big_warp_id<<8)+192	<seg_size)?key[k+sml_len+tid1+(big_warp_id<<8)+192	]:std::numeric_limits<K>::max();
			rg_k7  = (sml_len+tid1+(big_warp_id<<8)+224	<seg_size)?key[k+sml_len+tid1+(big_warp_id<<8)+224	]:std::numeric_limits<K>::max();
			if (sml_len+tid1+(big_warp_id<<8)+0	<seg_size) rg_v0  = sml_len+tid1+(big_warp_id<<8)+0	;
			if (sml_len+tid1+(big_warp_id<<8)+32	<seg_size) rg_v1  = sml_len+tid1+(big_warp_id<<8)+32	;
			if (sml_len+tid1+(big_warp_id<<8)+64	<seg_size) rg_v2  = sml_len+tid1+(big_warp_id<<8)+64	;
			if (sml_len+tid1+(big_warp_id<<8)+96	<seg_size) rg_v3  = sml_len+tid1+(big_warp_id<<8)+96	;
			if (sml_len+tid1+(big_warp_id<<8)+128	<seg_size) rg_v4  = sml_len+tid1+(big_warp_id<<8)+128	;
			if (sml_len+tid1+(big_warp_id<<8)+160	<seg_size) rg_v5  = sml_len+tid1+(big_warp_id<<8)+160	;
			if (sml_len+tid1+(big_warp_id<<8)+192	<seg_size) rg_v6  = sml_len+tid1+(big_warp_id<<8)+192	;
			if (sml_len+tid1+(big_warp_id<<8)+224	<seg_size) rg_v7  = sml_len+tid1+(big_warp_id<<8)+224	;

			// sort 256 elements
			// exch_intxn: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
			CMP_SWP(K, rg_k2, rg_k3, int, rg_v2, rg_v3);
			CMP_SWP(K, rg_k4, rg_k5, int, rg_v4, rg_v5);
			CMP_SWP(K, rg_k6, rg_k7, int, rg_v6, rg_v7);
			// exch_intxn: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k3, int, rg_v0, rg_v3);
			CMP_SWP(K, rg_k1, rg_k2, int, rg_v1, rg_v2);
			CMP_SWP(K, rg_k4, rg_k7, int, rg_v4, rg_v7);
			CMP_SWP(K, rg_k5, rg_k6, int, rg_v5, rg_v6);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
			CMP_SWP(K, rg_k2, rg_k3, int, rg_v2, rg_v3);
			CMP_SWP(K, rg_k4, rg_k5, int, rg_v4, rg_v5);
			CMP_SWP(K, rg_k6, rg_k7, int, rg_v6, rg_v7);
			// exch_intxn: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k7, int, rg_v0, rg_v7);
			CMP_SWP(K, rg_k1, rg_k6, int, rg_v1, rg_v6);
			CMP_SWP(K, rg_k2, rg_k5, int, rg_v2, rg_v5);
			CMP_SWP(K, rg_k3, rg_k4, int, rg_v3, rg_v4);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k2, int, rg_v0, rg_v2);
			CMP_SWP(K, rg_k1, rg_k3, int, rg_v1, rg_v3);
			CMP_SWP(K, rg_k4, rg_k6, int, rg_v4, rg_v6);
			CMP_SWP(K, rg_k5, rg_k7, int, rg_v5, rg_v7);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
			CMP_SWP(K, rg_k2, rg_k3, int, rg_v2, rg_v3);
			CMP_SWP(K, rg_k4, rg_k5, int, rg_v4, rg_v5);
			CMP_SWP(K, rg_k6, rg_k7, int, rg_v6, rg_v7);
			// exch_intxn: generate exch_intxn()
			exch_intxn(rg_k0, rg_k1, rg_k2, rg_k3, rg_k4, rg_k5, rg_k6, rg_k7, rg_v0, rg_v1, rg_v2, rg_v3, rg_v4, rg_v5, rg_v6, rg_v7, 0x1, bit1);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k4, int, rg_v0, rg_v4);
			CMP_SWP(K, rg_k1, rg_k5, int, rg_v1, rg_v5);
			CMP_SWP(K, rg_k2, rg_k6, int, rg_v2, rg_v6);
			CMP_SWP(K, rg_k3, rg_k7, int, rg_v3, rg_v7);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k2, int, rg_v0, rg_v2);
			CMP_SWP(K, rg_k1, rg_k3, int, rg_v1, rg_v3);
			CMP_SWP(K, rg_k4, rg_k6, int, rg_v4, rg_v6);
			CMP_SWP(K, rg_k5, rg_k7, int, rg_v5, rg_v7);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
			CMP_SWP(K, rg_k2, rg_k3, int, rg_v2, rg_v3);
			CMP_SWP(K, rg_k4, rg_k5, int, rg_v4, rg_v5);
			CMP_SWP(K, rg_k6, rg_k7, int, rg_v6, rg_v7);
			// exch_intxn: generate exch_intxn()
			exch_intxn(rg_k0, rg_k1, rg_k2, rg_k3, rg_k4, rg_k5, rg_k6, rg_k7, rg_v0, rg_v1, rg_v2, rg_v3, rg_v4, rg_v5, rg_v6, rg_v7, 0x3, bit2);
			// exch_paral: generate exch_paral()
			exch_paral(rg_k0, rg_k1, rg_k2, rg_k3, rg_k4, rg_k5, rg_k6, rg_k7, rg_v0, rg_v1, rg_v2, rg_v3, rg_v4, rg_v5, rg_v6, rg_v7, 0x1, bit1);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k4, int, rg_v0, rg_v4);
			CMP_SWP(K, rg_k1, rg_k5, int, rg_v1, rg_v5);
			CMP_SWP(K, rg_k2, rg_k6, int, rg_v2, rg_v6);
			CMP_SWP(K, rg_k3, rg_k7, int, rg_v3, rg_v7);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k2, int, rg_v0, rg_v2);
			CMP_SWP(K, rg_k1, rg_k3, int, rg_v1, rg_v3);
			CMP_SWP(K, rg_k4, rg_k6, int, rg_v4, rg_v6);
			CMP_SWP(K, rg_k5, rg_k7, int, rg_v5, rg_v7);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
			CMP_SWP(K, rg_k2, rg_k3, int, rg_v2, rg_v3);
			CMP_SWP(K, rg_k4, rg_k5, int, rg_v4, rg_v5);
			CMP_SWP(K, rg_k6, rg_k7, int, rg_v6, rg_v7);
			// exch_intxn: generate exch_intxn()
			exch_intxn(rg_k0, rg_k1, rg_k2, rg_k3, rg_k4, rg_k5, rg_k6, rg_k7, rg_v0, rg_v1, rg_v2, rg_v3, rg_v4, rg_v5, rg_v6, rg_v7, 0x7, bit3);
			// exch_paral: generate exch_paral()
			exch_paral(rg_k0, rg_k1, rg_k2, rg_k3, rg_k4, rg_k5, rg_k6, rg_k7, rg_v0, rg_v1, rg_v2, rg_v3, rg_v4, rg_v5, rg_v6, rg_v7, 0x2, bit2);
			// exch_paral: generate exch_paral()
			exch_paral(rg_k0, rg_k1, rg_k2, rg_k3, rg_k4, rg_k5, rg_k6, rg_k7, rg_v0, rg_v1, rg_v2, rg_v3, rg_v4, rg_v5, rg_v6, rg_v7, 0x1, bit1);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k4, int, rg_v0, rg_v4);
			CMP_SWP(K, rg_k1, rg_k5, int, rg_v1, rg_v5);
			CMP_SWP(K, rg_k2, rg_k6, int, rg_v2, rg_v6);
			CMP_SWP(K, rg_k3, rg_k7, int, rg_v3, rg_v7);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k2, int, rg_v0, rg_v2);
			CMP_SWP(K, rg_k1, rg_k3, int, rg_v1, rg_v3);
			CMP_SWP(K, rg_k4, rg_k6, int, rg_v4, rg_v6);
			CMP_SWP(K, rg_k5, rg_k7, int, rg_v5, rg_v7);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
			CMP_SWP(K, rg_k2, rg_k3, int, rg_v2, rg_v3);
			CMP_SWP(K, rg_k4, rg_k5, int, rg_v4, rg_v5);
			CMP_SWP(K, rg_k6, rg_k7, int, rg_v6, rg_v7);
			// exch_intxn: generate exch_intxn()
			exch_intxn(rg_k0, rg_k1, rg_k2, rg_k3, rg_k4, rg_k5, rg_k6, rg_k7, rg_v0, rg_v1, rg_v2, rg_v3, rg_v4, rg_v5, rg_v6, rg_v7, 0xf, bit4);
			// exch_paral: generate exch_paral()
			exch_paral(rg_k0, rg_k1, rg_k2, rg_k3, rg_k4, rg_k5, rg_k6, rg_k7, rg_v0, rg_v1, rg_v2, rg_v3, rg_v4, rg_v5, rg_v6, rg_v7, 0x4, bit3);
			// exch_paral: generate exch_paral()
			exch_paral(rg_k0, rg_k1, rg_k2, rg_k3, rg_k4, rg_k5, rg_k6, rg_k7, rg_v0, rg_v1, rg_v2, rg_v3, rg_v4, rg_v5, rg_v6, rg_v7, 0x2, bit2);
			// exch_paral: generate exch_paral()
			exch_paral(rg_k0, rg_k1, rg_k2, rg_k3, rg_k4, rg_k5, rg_k6, rg_k7, rg_v0, rg_v1, rg_v2, rg_v3, rg_v4, rg_v5, rg_v6, rg_v7, 0x1, bit1);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k4, int, rg_v0, rg_v4);
			CMP_SWP(K, rg_k1, rg_k5, int, rg_v1, rg_v5);
			CMP_SWP(K, rg_k2, rg_k6, int, rg_v2, rg_v6);
			CMP_SWP(K, rg_k3, rg_k7, int, rg_v3, rg_v7);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k2, int, rg_v0, rg_v2);
			CMP_SWP(K, rg_k1, rg_k3, int, rg_v1, rg_v3);
			CMP_SWP(K, rg_k4, rg_k6, int, rg_v4, rg_v6);
			CMP_SWP(K, rg_k5, rg_k7, int, rg_v5, rg_v7);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
			CMP_SWP(K, rg_k2, rg_k3, int, rg_v2, rg_v3);
			CMP_SWP(K, rg_k4, rg_k5, int, rg_v4, rg_v5);
			CMP_SWP(K, rg_k6, rg_k7, int, rg_v6, rg_v7);
			// exch_intxn: generate exch_intxn()
			exch_intxn(rg_k0, rg_k1, rg_k2, rg_k3, rg_k4, rg_k5, rg_k6, rg_k7, rg_v0, rg_v1, rg_v2, rg_v3, rg_v4, rg_v5, rg_v6, rg_v7, 0x1f, bit5);
			// exch_paral: generate exch_paral()
			exch_paral(rg_k0, rg_k1, rg_k2, rg_k3, rg_k4, rg_k5, rg_k6, rg_k7, rg_v0, rg_v1, rg_v2, rg_v3, rg_v4, rg_v5, rg_v6, rg_v7, 0x8, bit4);
			// exch_paral: generate exch_paral()
			exch_paral(rg_k0, rg_k1, rg_k2, rg_k3, rg_k4, rg_k5, rg_k6, rg_k7, rg_v0, rg_v1, rg_v2, rg_v3, rg_v4, rg_v5, rg_v6, rg_v7, 0x4, bit3);
			// exch_paral: generate exch_paral()
			exch_paral(rg_k0, rg_k1, rg_k2, rg_k3, rg_k4, rg_k5, rg_k6, rg_k7, rg_v0, rg_v1, rg_v2, rg_v3, rg_v4, rg_v5, rg_v6, rg_v7, 0x2, bit2);
			// exch_paral: generate exch_paral()
			exch_paral(rg_k0, rg_k1, rg_k2, rg_k3, rg_k4, rg_k5, rg_k6, rg_k7, rg_v0, rg_v1, rg_v2, rg_v3, rg_v4, rg_v5, rg_v6, rg_v7, 0x1, bit1);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k4, int, rg_v0, rg_v4);
			CMP_SWP(K, rg_k1, rg_k5, int, rg_v1, rg_v5);
			CMP_SWP(K, rg_k2, rg_k6, int, rg_v2, rg_v6);
			CMP_SWP(K, rg_k3, rg_k7, int, rg_v3, rg_v7);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k2, int, rg_v0, rg_v2);
			CMP_SWP(K, rg_k1, rg_k3, int, rg_v1, rg_v3);
			CMP_SWP(K, rg_k4, rg_k6, int, rg_v4, rg_v6);
			CMP_SWP(K, rg_k5, rg_k7, int, rg_v5, rg_v7);
			// exch_paral: switch to exch_local()
			CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
			CMP_SWP(K, rg_k2, rg_k3, int, rg_v2, rg_v3);
			CMP_SWP(K, rg_k4, rg_k5, int, rg_v4, rg_v5);
			CMP_SWP(K, rg_k6, rg_k7, int, rg_v6, rg_v7);
		}

		// Store register results to shared memory
		if (sml_warp) {
			smem[(warp_id<<7) + (tid1<<2) + 0] = rg_k0;
			smem[(warp_id<<7) + (tid1<<2) + 1] = rg_k1;
			smem[(warp_id<<7) + (tid1<<2) + 2] = rg_k2;
			smem[(warp_id<<7) + (tid1<<2) + 3] = rg_k3;
			tmem[(warp_id<<7) + (tid1<<2) + 0] = rg_v0;
			tmem[(warp_id<<7) + (tid1<<2) + 1] = rg_v1;
			tmem[(warp_id<<7) + (tid1<<2) + 2] = rg_v2;
			tmem[(warp_id<<7) + (tid1<<2) + 3] = rg_v3;
		} else {
			smem[sml_len+(big_warp_id<<8) + (tid1<<3) + 0] = rg_k0;
			smem[sml_len+(big_warp_id<<8) + (tid1<<3) + 1] = rg_k1;
			smem[sml_len+(big_warp_id<<8) + (tid1<<3) + 2] = rg_k2;
			smem[sml_len+(big_warp_id<<8) + (tid1<<3) + 3] = rg_k3;
			smem[sml_len+(big_warp_id<<8) + (tid1<<3) + 4] = rg_k4;
			smem[sml_len+(big_warp_id<<8) + (tid1<<3) + 5] = rg_k5;
			smem[sml_len+(big_warp_id<<8) + (tid1<<3) + 6] = rg_k6;
			smem[sml_len+(big_warp_id<<8) + (tid1<<3) + 7] = rg_k7;
			tmem[sml_len+(big_warp_id<<8) + (tid1<<3) + 0] = rg_v0;
			tmem[sml_len+(big_warp_id<<8) + (tid1<<3) + 1] = rg_v1;
			tmem[sml_len+(big_warp_id<<8) + (tid1<<3) + 2] = rg_v2;
			tmem[sml_len+(big_warp_id<<8) + (tid1<<3) + 3] = rg_v3;
			tmem[sml_len+(big_warp_id<<8) + (tid1<<3) + 4] = rg_v4;
			tmem[sml_len+(big_warp_id<<8) + (tid1<<3) + 5] = rg_v5;
			tmem[sml_len+(big_warp_id<<8) + (tid1<<3) + 6] = rg_v6;
			tmem[sml_len+(big_warp_id<<8) + (tid1<<3) + 7] = rg_v7;
		}
		__syncthreads();

		// Merge in 4 steps
		int grp_start_wp_id;
		int grp_start_off;
		int lhs_len;
		int rhs_len;
		int gran;
		int s_a;
		int s_b;
		bool p;
		K tmp_k0;
		K tmp_k1;
		int tmp_v0;
		int tmp_v1;
		K *start;

		// Step 0
		grp_start_wp_id = ((warp_id>>1)<<1);
		grp_start_off = (grp_start_wp_id<sml_wp)?grp_start_wp_id*128:sml_len+(grp_start_wp_id-sml_wp) * 256;
		lhs_len = ((grp_start_wp_id + 0 < sml_wp) ? 128 : 256);
		rhs_len = ((grp_start_wp_id + 1 < sml_wp) ? 128 : 256);
		// gran denotes where each thread will write its merged entries
		gran = (warp_id<sml_wp) ? (tid1<<2) : (tid1<<3);
		if ((warp_id & 1) == 0) {
			gran += 0;
		}
		if ((warp_id & 1) == 1) {
			gran += ((grp_start_wp_id + 0 < sml_wp) ? 128 : 256);
		}

		start = smem + grp_start_off;
		s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
		s_b = lhs_len + gran - s_a;
		if (sml_warp) {
			tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
			tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
			if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
			if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k0 = p ? tmp_k0 : tmp_k1;
			rg_v0 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k1 = p ? tmp_k0 : tmp_k1;
			rg_v1 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k2 = p ? tmp_k0 : tmp_k1;
			rg_v2 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k3 = p ? tmp_k0 : tmp_k1;
			rg_v3 = p ? tmp_v0 : tmp_v1;
		} else {
			tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
			tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
			if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
			if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k0 = p ? tmp_k0 : tmp_k1;
			rg_v0 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k1 = p ? tmp_k0 : tmp_k1;
			rg_v1 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k2 = p ? tmp_k0 : tmp_k1;
			rg_v2 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k3 = p ? tmp_k0 : tmp_k1;
			rg_v3 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k4 = p ? tmp_k0 : tmp_k1;
			rg_v4 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k5 = p ? tmp_k0 : tmp_k1;
			rg_v5 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k6 = p ? tmp_k0 : tmp_k1;
			rg_v6 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k7 = p ? tmp_k0 : tmp_k1;
			rg_v7 = p ? tmp_v0 : tmp_v1;
		}

		__syncthreads();
		// Store merged results back into shared memory
		if (sml_warp) {
			smem[grp_start_off + gran + 0] = rg_k0;
			smem[grp_start_off + gran + 1] = rg_k1;
			smem[grp_start_off + gran + 2] = rg_k2;
			smem[grp_start_off + gran + 3] = rg_k3;
			tmem[grp_start_off + gran + 0] = rg_v0;
			tmem[grp_start_off + gran + 1] = rg_v1;
			tmem[grp_start_off + gran + 2] = rg_v2;
			tmem[grp_start_off + gran + 3] = rg_v3;
		} else {
			smem[grp_start_off + gran + 0] = rg_k0;
			smem[grp_start_off + gran + 1] = rg_k1;
			smem[grp_start_off + gran + 2] = rg_k2;
			smem[grp_start_off + gran + 3] = rg_k3;
			smem[grp_start_off + gran + 4] = rg_k4;
			smem[grp_start_off + gran + 5] = rg_k5;
			smem[grp_start_off + gran + 6] = rg_k6;
			smem[grp_start_off + gran + 7] = rg_k7;
			tmem[grp_start_off + gran + 0] = rg_v0;
			tmem[grp_start_off + gran + 1] = rg_v1;
			tmem[grp_start_off + gran + 2] = rg_v2;
			tmem[grp_start_off + gran + 3] = rg_v3;
			tmem[grp_start_off + gran + 4] = rg_v4;
			tmem[grp_start_off + gran + 5] = rg_v5;
			tmem[grp_start_off + gran + 6] = rg_v6;
			tmem[grp_start_off + gran + 7] = rg_v7;
		}
		__syncthreads();

		// Step 1
		grp_start_wp_id = ((warp_id>>2)<<2);
		grp_start_off = (grp_start_wp_id<sml_wp)?grp_start_wp_id*128:sml_len+(grp_start_wp_id-sml_wp) * 256;
		lhs_len = ((grp_start_wp_id + 0 < sml_wp) ? 128 : 256)+
		          ((grp_start_wp_id + 1 < sml_wp) ? 128 : 256);
		rhs_len = ((grp_start_wp_id + 2 < sml_wp) ? 128 : 256)+
		          ((grp_start_wp_id + 3 < sml_wp) ? 128 : 256);
		// gran denotes where each thread will write its merged entries
		gran = (warp_id<sml_wp) ? (tid1<<2) : (tid1<<3);
		if ((warp_id & 3) == 0) {
			gran += 0;
		}
		if ((warp_id & 3) == 1) {
			gran += ((grp_start_wp_id + 0 < sml_wp) ? 128 : 256);
		}
		if ((warp_id & 3) == 2) {
			gran += ((grp_start_wp_id + 0 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 1 < sml_wp) ? 128 : 256);
		}
		if ((warp_id & 3) == 3) {
			gran += ((grp_start_wp_id + 0 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 1 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 2 < sml_wp) ? 128 : 256);
		}

		start = smem + grp_start_off;
		s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
		s_b = lhs_len + gran - s_a;
		if (sml_warp) {
			tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
			tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
			if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
			if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k0 = p ? tmp_k0 : tmp_k1;
			rg_v0 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k1 = p ? tmp_k0 : tmp_k1;
			rg_v1 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k2 = p ? tmp_k0 : tmp_k1;
			rg_v2 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k3 = p ? tmp_k0 : tmp_k1;
			rg_v3 = p ? tmp_v0 : tmp_v1;
		} else {
			tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
			tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
			if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
			if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k0 = p ? tmp_k0 : tmp_k1;
			rg_v0 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k1 = p ? tmp_k0 : tmp_k1;
			rg_v1 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k2 = p ? tmp_k0 : tmp_k1;
			rg_v2 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k3 = p ? tmp_k0 : tmp_k1;
			rg_v3 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k4 = p ? tmp_k0 : tmp_k1;
			rg_v4 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k5 = p ? tmp_k0 : tmp_k1;
			rg_v5 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k6 = p ? tmp_k0 : tmp_k1;
			rg_v6 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k7 = p ? tmp_k0 : tmp_k1;
			rg_v7 = p ? tmp_v0 : tmp_v1;
		}

		__syncthreads();
		// Store merged results back into shared memory
		if (sml_warp) {
			smem[grp_start_off + gran + 0] = rg_k0;
			smem[grp_start_off + gran + 1] = rg_k1;
			smem[grp_start_off + gran + 2] = rg_k2;
			smem[grp_start_off + gran + 3] = rg_k3;
			tmem[grp_start_off + gran + 0] = rg_v0;
			tmem[grp_start_off + gran + 1] = rg_v1;
			tmem[grp_start_off + gran + 2] = rg_v2;
			tmem[grp_start_off + gran + 3] = rg_v3;
		} else {
			smem[grp_start_off + gran + 0] = rg_k0;
			smem[grp_start_off + gran + 1] = rg_k1;
			smem[grp_start_off + gran + 2] = rg_k2;
			smem[grp_start_off + gran + 3] = rg_k3;
			smem[grp_start_off + gran + 4] = rg_k4;
			smem[grp_start_off + gran + 5] = rg_k5;
			smem[grp_start_off + gran + 6] = rg_k6;
			smem[grp_start_off + gran + 7] = rg_k7;
			tmem[grp_start_off + gran + 0] = rg_v0;
			tmem[grp_start_off + gran + 1] = rg_v1;
			tmem[grp_start_off + gran + 2] = rg_v2;
			tmem[grp_start_off + gran + 3] = rg_v3;
			tmem[grp_start_off + gran + 4] = rg_v4;
			tmem[grp_start_off + gran + 5] = rg_v5;
			tmem[grp_start_off + gran + 6] = rg_v6;
			tmem[grp_start_off + gran + 7] = rg_v7;
		}
		__syncthreads();

		// Step 2
		grp_start_wp_id = ((warp_id>>3)<<3);
		grp_start_off = (grp_start_wp_id<sml_wp)?grp_start_wp_id*128:sml_len+(grp_start_wp_id-sml_wp) * 256;
		lhs_len = ((grp_start_wp_id + 0 < sml_wp) ? 128 : 256)+
		          ((grp_start_wp_id + 1 < sml_wp) ? 128 : 256)+
		          ((grp_start_wp_id + 2 < sml_wp) ? 128 : 256)+
		          ((grp_start_wp_id + 3 < sml_wp) ? 128 : 256);
		rhs_len = ((grp_start_wp_id + 4 < sml_wp) ? 128 : 256)+
		          ((grp_start_wp_id + 5 < sml_wp) ? 128 : 256)+
		          ((grp_start_wp_id + 6 < sml_wp) ? 128 : 256)+
		          ((grp_start_wp_id + 7 < sml_wp) ? 128 : 256);
		// gran denotes where each thread will write its merged entries
		gran = (warp_id<sml_wp) ? (tid1<<2) : (tid1<<3);
		if ((warp_id & 7) == 0) {
			gran += 0;
		}
		if ((warp_id & 7) == 1) {
			gran += ((grp_start_wp_id + 0 < sml_wp) ? 128 : 256);
		}
		if ((warp_id & 7) == 2) {
			gran += ((grp_start_wp_id + 0 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 1 < sml_wp) ? 128 : 256);
		}
		if ((warp_id & 7) == 3) {
			gran += ((grp_start_wp_id + 0 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 1 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 2 < sml_wp) ? 128 : 256);
		}
		if ((warp_id & 7) == 4) {
			gran += ((grp_start_wp_id + 0 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 1 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 2 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 3 < sml_wp) ? 128 : 256);
		}
		if ((warp_id & 7) == 5) {
			gran += ((grp_start_wp_id + 0 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 1 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 2 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 3 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 4 < sml_wp) ? 128 : 256);
		}
		if ((warp_id & 7) == 6) {
			gran += ((grp_start_wp_id + 0 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 1 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 2 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 3 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 4 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 5 < sml_wp) ? 128 : 256);
		}
		if ((warp_id & 7) == 7) {
			gran += ((grp_start_wp_id + 0 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 1 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 2 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 3 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 4 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 5 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 6 < sml_wp) ? 128 : 256);
		}

		start = smem + grp_start_off;
		s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
		s_b = lhs_len + gran - s_a;
		if (sml_warp) {
			tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
			tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
			if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
			if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k0 = p ? tmp_k0 : tmp_k1;
			rg_v0 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k1 = p ? tmp_k0 : tmp_k1;
			rg_v1 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k2 = p ? tmp_k0 : tmp_k1;
			rg_v2 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k3 = p ? tmp_k0 : tmp_k1;
			rg_v3 = p ? tmp_v0 : tmp_v1;
		} else {
			tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
			tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
			if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
			if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k0 = p ? tmp_k0 : tmp_k1;
			rg_v0 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k1 = p ? tmp_k0 : tmp_k1;
			rg_v1 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k2 = p ? tmp_k0 : tmp_k1;
			rg_v2 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k3 = p ? tmp_k0 : tmp_k1;
			rg_v3 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k4 = p ? tmp_k0 : tmp_k1;
			rg_v4 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k5 = p ? tmp_k0 : tmp_k1;
			rg_v5 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k6 = p ? tmp_k0 : tmp_k1;
			rg_v6 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k7 = p ? tmp_k0 : tmp_k1;
			rg_v7 = p ? tmp_v0 : tmp_v1;
		}

		__syncthreads();
		// Store merged results back into shared memory
		if (sml_warp) {
			smem[grp_start_off + gran + 0] = rg_k0;
			smem[grp_start_off + gran + 1] = rg_k1;
			smem[grp_start_off + gran + 2] = rg_k2;
			smem[grp_start_off + gran + 3] = rg_k3;
			tmem[grp_start_off + gran + 0] = rg_v0;
			tmem[grp_start_off + gran + 1] = rg_v1;
			tmem[grp_start_off + gran + 2] = rg_v2;
			tmem[grp_start_off + gran + 3] = rg_v3;
		} else {
			smem[grp_start_off + gran + 0] = rg_k0;
			smem[grp_start_off + gran + 1] = rg_k1;
			smem[grp_start_off + gran + 2] = rg_k2;
			smem[grp_start_off + gran + 3] = rg_k3;
			smem[grp_start_off + gran + 4] = rg_k4;
			smem[grp_start_off + gran + 5] = rg_k5;
			smem[grp_start_off + gran + 6] = rg_k6;
			smem[grp_start_off + gran + 7] = rg_k7;
			tmem[grp_start_off + gran + 0] = rg_v0;
			tmem[grp_start_off + gran + 1] = rg_v1;
			tmem[grp_start_off + gran + 2] = rg_v2;
			tmem[grp_start_off + gran + 3] = rg_v3;
			tmem[grp_start_off + gran + 4] = rg_v4;
			tmem[grp_start_off + gran + 5] = rg_v5;
			tmem[grp_start_off + gran + 6] = rg_v6;
			tmem[grp_start_off + gran + 7] = rg_v7;
		}
		__syncthreads();

		// Step 3
		grp_start_wp_id = ((warp_id>>4)<<4);
		grp_start_off = (grp_start_wp_id<sml_wp)?grp_start_wp_id*128:sml_len+(grp_start_wp_id-sml_wp) * 256;
		lhs_len = ((grp_start_wp_id + 0 < sml_wp) ? 128 : 256)+
		          ((grp_start_wp_id + 1 < sml_wp) ? 128 : 256)+
		          ((grp_start_wp_id + 2 < sml_wp) ? 128 : 256)+
		          ((grp_start_wp_id + 3 < sml_wp) ? 128 : 256)+
		          ((grp_start_wp_id + 4 < sml_wp) ? 128 : 256)+
		          ((grp_start_wp_id + 5 < sml_wp) ? 128 : 256)+
		          ((grp_start_wp_id + 6 < sml_wp) ? 128 : 256)+
		          ((grp_start_wp_id + 7 < sml_wp) ? 128 : 256);
		rhs_len = ((grp_start_wp_id + 8 < sml_wp) ? 128 : 256)+
		          ((grp_start_wp_id + 9 < sml_wp) ? 128 : 256)+
		          ((grp_start_wp_id + 10 < sml_wp) ? 128 : 256)+
		          ((grp_start_wp_id + 11 < sml_wp) ? 128 : 256)+
		          ((grp_start_wp_id + 12 < sml_wp) ? 128 : 256)+
		          ((grp_start_wp_id + 13 < sml_wp) ? 128 : 256)+
		          ((grp_start_wp_id + 14 < sml_wp) ? 128 : 256)+
		          ((grp_start_wp_id + 15 < sml_wp) ? 128 : 256);
		// gran denotes where each thread will write its merged entries
		gran = (warp_id<sml_wp) ? (tid1<<2) : (tid1<<3);
		if ((warp_id & 15) == 0) {
			gran += 0;
		}
		if ((warp_id & 15) == 1) {
			gran += ((grp_start_wp_id + 0 < sml_wp) ? 128 : 256);
		}
		if ((warp_id & 15) == 2) {
			gran += ((grp_start_wp_id + 0 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 1 < sml_wp) ? 128 : 256);
		}
		if ((warp_id & 15) == 3) {
			gran += ((grp_start_wp_id + 0 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 1 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 2 < sml_wp) ? 128 : 256);
		}
		if ((warp_id & 15) == 4) {
			gran += ((grp_start_wp_id + 0 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 1 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 2 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 3 < sml_wp) ? 128 : 256);
		}
		if ((warp_id & 15) == 5) {
			gran += ((grp_start_wp_id + 0 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 1 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 2 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 3 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 4 < sml_wp) ? 128 : 256);
		}
		if ((warp_id & 15) == 6) {
			gran += ((grp_start_wp_id + 0 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 1 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 2 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 3 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 4 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 5 < sml_wp) ? 128 : 256);
		}
		if ((warp_id & 15) == 7) {
			gran += ((grp_start_wp_id + 0 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 1 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 2 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 3 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 4 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 5 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 6 < sml_wp) ? 128 : 256);
		}
		if ((warp_id & 15) == 8) {
			gran += ((grp_start_wp_id + 0 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 1 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 2 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 3 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 4 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 5 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 6 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 7 < sml_wp) ? 128 : 256);
		}
		if ((warp_id & 15) == 9) {
			gran += ((grp_start_wp_id + 0 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 1 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 2 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 3 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 4 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 5 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 6 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 7 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 8 < sml_wp) ? 128 : 256);
		}
		if ((warp_id & 15) == 10) {
			gran += ((grp_start_wp_id + 0 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 1 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 2 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 3 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 4 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 5 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 6 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 7 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 8 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 9 < sml_wp) ? 128 : 256);
		}
		if ((warp_id & 15) == 11) {
			gran += ((grp_start_wp_id + 0 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 1 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 2 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 3 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 4 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 5 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 6 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 7 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 8 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 9 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 10 < sml_wp) ? 128 : 256);
		}
		if ((warp_id & 15) == 12) {
			gran += ((grp_start_wp_id + 0 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 1 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 2 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 3 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 4 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 5 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 6 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 7 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 8 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 9 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 10 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 11 < sml_wp) ? 128 : 256);
		}
		if ((warp_id & 15) == 13) {
			gran += ((grp_start_wp_id + 0 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 1 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 2 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 3 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 4 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 5 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 6 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 7 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 8 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 9 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 10 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 11 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 12 < sml_wp) ? 128 : 256);
		}
		if ((warp_id & 15) == 14) {
			gran += ((grp_start_wp_id + 0 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 1 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 2 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 3 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 4 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 5 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 6 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 7 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 8 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 9 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 10 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 11 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 12 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 13 < sml_wp) ? 128 : 256);
		}
		if ((warp_id & 15) == 15) {
			gran += ((grp_start_wp_id + 0 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 1 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 2 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 3 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 4 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 5 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 6 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 7 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 8 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 9 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 10 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 11 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 12 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 13 < sml_wp) ? 128 : 256)+
			        ((grp_start_wp_id + 14 < sml_wp) ? 128 : 256);
		}

		start = smem + grp_start_off;
		s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
		s_b = lhs_len + gran - s_a;
		if (sml_warp) {
			tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
			tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
			if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
			if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k0 = p ? tmp_k0 : tmp_k1;
			rg_v0 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k1 = p ? tmp_k0 : tmp_k1;
			rg_v1 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k2 = p ? tmp_k0 : tmp_k1;
			rg_v2 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k3 = p ? tmp_k0 : tmp_k1;
			rg_v3 = p ? tmp_v0 : tmp_v1;
		} else {
			tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
			tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
			if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
			if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k0 = p ? tmp_k0 : tmp_k1;
			rg_v0 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k1 = p ? tmp_k0 : tmp_k1;
			rg_v1 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k2 = p ? tmp_k0 : tmp_k1;
			rg_v2 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k3 = p ? tmp_k0 : tmp_k1;
			rg_v3 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k4 = p ? tmp_k0 : tmp_k1;
			rg_v4 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k5 = p ? tmp_k0 : tmp_k1;
			rg_v5 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k6 = p ? tmp_k0 : tmp_k1;
			rg_v6 = p ? tmp_v0 : tmp_v1;
			if (p) {
				++s_a;
				tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
				if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
			} else {
				++s_b;
				tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
				if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
			}
			p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
			rg_k7 = p ? tmp_k0 : tmp_k1;
			rg_v7 = p ? tmp_v0 : tmp_v1;
		}

		__syncthreads();
		// Store merged results back into shared memory
		if (sml_warp) {
			smem[grp_start_off + gran + 0] = rg_k0;
			smem[grp_start_off + gran + 1] = rg_k1;
			smem[grp_start_off + gran + 2] = rg_k2;
			smem[grp_start_off + gran + 3] = rg_k3;
			tmem[grp_start_off + gran + 0] = rg_v0;
			tmem[grp_start_off + gran + 1] = rg_v1;
			tmem[grp_start_off + gran + 2] = rg_v2;
			tmem[grp_start_off + gran + 3] = rg_v3;
		} else {
			smem[grp_start_off + gran + 0] = rg_k0;
			smem[grp_start_off + gran + 1] = rg_k1;
			smem[grp_start_off + gran + 2] = rg_k2;
			smem[grp_start_off + gran + 3] = rg_k3;
			smem[grp_start_off + gran + 4] = rg_k4;
			smem[grp_start_off + gran + 5] = rg_k5;
			smem[grp_start_off + gran + 6] = rg_k6;
			smem[grp_start_off + gran + 7] = rg_k7;
			tmem[grp_start_off + gran + 0] = rg_v0;
			tmem[grp_start_off + gran + 1] = rg_v1;
			tmem[grp_start_off + gran + 2] = rg_v2;
			tmem[grp_start_off + gran + 3] = rg_v3;
			tmem[grp_start_off + gran + 4] = rg_v4;
			tmem[grp_start_off + gran + 5] = rg_v5;
			tmem[grp_start_off + gran + 6] = rg_v6;
			tmem[grp_start_off + gran + 7] = rg_v7;
		}
		__syncthreads();

		if ((warp_id << 8) + 0 + tid1 < seg_size) keyB[k + (warp_id << 8) + 0 + tid1] = smem[(warp_id << 8) + 0 + tid1];
		if ((warp_id << 8) + 32 + tid1 < seg_size) keyB[k + (warp_id << 8) + 32 + tid1] = smem[(warp_id << 8) + 32 + tid1];
		if ((warp_id << 8) + 64 + tid1 < seg_size) keyB[k + (warp_id << 8) + 64 + tid1] = smem[(warp_id << 8) + 64 + tid1];
		if ((warp_id << 8) + 96 + tid1 < seg_size) keyB[k + (warp_id << 8) + 96 + tid1] = smem[(warp_id << 8) + 96 + tid1];
		if ((warp_id << 8) + 128 + tid1 < seg_size) keyB[k + (warp_id << 8) + 128 + tid1] = smem[(warp_id << 8) + 128 + tid1];
		if ((warp_id << 8) + 160 + tid1 < seg_size) keyB[k + (warp_id << 8) + 160 + tid1] = smem[(warp_id << 8) + 160 + tid1];
		if ((warp_id << 8) + 192 + tid1 < seg_size) keyB[k + (warp_id << 8) + 192 + tid1] = smem[(warp_id << 8) + 192 + tid1];
		if ((warp_id << 8) + 224 + tid1 < seg_size) keyB[k + (warp_id << 8) + 224 + tid1] = smem[(warp_id << 8) + 224 + tid1];

		if ((warp_id << 8) + 0 + tid1 < seg_size) valB[k + (warp_id << 8) + 0 + tid1] = val[k + tmem[(warp_id << 8) + 0 + tid1]];
		if ((warp_id << 8) + 32 + tid1 < seg_size) valB[k + (warp_id << 8) + 32 + tid1] = val[k + tmem[(warp_id << 8) + 32 + tid1]];
		if ((warp_id << 8) + 64 + tid1 < seg_size) valB[k + (warp_id << 8) + 64 + tid1] = val[k + tmem[(warp_id << 8) + 64 + tid1]];
		if ((warp_id << 8) + 96 + tid1 < seg_size) valB[k + (warp_id << 8) + 96 + tid1] = val[k + tmem[(warp_id << 8) + 96 + tid1]];
		if ((warp_id << 8) + 128 + tid1 < seg_size) valB[k + (warp_id << 8) + 128 + tid1] = val[k + tmem[(warp_id << 8) + 128 + tid1]];
		if ((warp_id << 8) + 160 + tid1 < seg_size) valB[k + (warp_id << 8) + 160 + tid1] = val[k + tmem[(warp_id << 8) + 160 + tid1]];
		if ((warp_id << 8) + 192 + tid1 < seg_size) valB[k + (warp_id << 8) + 192 + tid1] = val[k + tmem[(warp_id << 8) + 192 + tid1]];
		if ((warp_id << 8) + 224 + tid1 < seg_size) valB[k + (warp_id << 8) + 224 + tid1] = val[k + tmem[(warp_id << 8) + 224 + tid1]];
	}
}

#endif
