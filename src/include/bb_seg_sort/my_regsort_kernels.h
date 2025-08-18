#ifndef _H_MY_REGSORT_KERNELS
#define _H_MY_REGSORT_KERNELS

#include <limits>

#include "bb_exch.h"


// 2 threads | 1 ppt
template<class K, class T>
__global__
void my_t2_ppt1_orig(K *key, T *val, K *keyB, T *valB, int n, int *segs, int *bin, int bin_size, int length) {
	const int gid = threadIdx.x + blockIdx.x * blockDim.x;
	const int bin_it = (gid>>1);
	const int tid = (threadIdx.x & 1);
	const int bit1 = (tid>>0)&0x1;
	K rg_k0;
	int rg_v0;
	int k;
	int seg_size;

	if(bin_it < bin_size) {
		k = segs[bin[bin_it]];
		seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
		rg_k0  = (tid+0	<seg_size)?key[k+tid+0	]:std::numeric_limits<K>::max();
		if(tid+0	<seg_size) rg_v0 = tid+0	;

		// sort 2 elements
		// exch_intxn: generate exch_intxn()
		exch_intxn(rg_k0, rg_v0, 0x1, bit1);

		if((tid<<0)+0 <seg_size) keyB[k+(tid<<0)+0] = rg_k0;

		if((tid<<0)+0 <seg_size) valB[k+(tid<<0)+0] = val[k+rg_v0];
	}
}

// 2 threads | 2 ppt
template<class K, class T>
__global__
void my_t2_ppt2_orig(K *key, T *val, K *keyB, T *valB, int n, int *segs, int *bin, int bin_size, int length) {
	const int gid = threadIdx.x + blockIdx.x * blockDim.x;
	const int bin_it = (gid>>1);
	const int tid = (threadIdx.x & 1);
	const int bit1 = (tid>>0)&0x1;
	K rg_k0;
	K rg_k1;
	int rg_v0;
	int rg_v1;
	int k;
	int seg_size;

	if(bin_it < bin_size) {
		k = segs[bin[bin_it]];
		seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
		rg_k0  = (tid+0	<seg_size)?key[k+tid+0	]:std::numeric_limits<K>::max();
		rg_k1  = (tid+2	<seg_size)?key[k+tid+2	]:std::numeric_limits<K>::max();
		if(tid+0	<seg_size) rg_v0 = tid+0	;
		if(tid+2	<seg_size) rg_v1 = tid+2	;

		// sort 4 elements
		// exch_intxn: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
		// exch_intxn: generate exch_intxn()
		exch_intxn(rg_k0, rg_k1, rg_v0, rg_v1, 0x1, bit1);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);

		if((tid<<1)+0 <seg_size) keyB[k+(tid<<1)+0] = rg_k0;
		if((tid<<1)+1 <seg_size) keyB[k+(tid<<1)+1] = rg_k1;

		if((tid<<1)+0 <seg_size) valB[k+(tid<<1)+0] = val[k+rg_v0];
		if((tid<<1)+1 <seg_size) valB[k+(tid<<1)+1] = val[k+rg_v1];
	}
}

// 2 threads | 4 ppt
template<class K, class T>
__global__
void my_t2_ppt4_orig(K *key, T *val, K *keyB, T *valB, int n, int *segs, int *bin, int bin_size, int length) {
	const int gid = threadIdx.x + blockIdx.x * blockDim.x;
	const int bin_it = (gid>>1);
	const int tid = (threadIdx.x & 1);
	const int bit1 = (tid>>0)&0x1;
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

	if(bin_it < bin_size) {
		k = segs[bin[bin_it]];
		seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
		rg_k0  = (tid+0	<seg_size)?key[k+tid+0	]:std::numeric_limits<K>::max();
		rg_k1  = (tid+2	<seg_size)?key[k+tid+2	]:std::numeric_limits<K>::max();
		rg_k2  = (tid+4	<seg_size)?key[k+tid+4	]:std::numeric_limits<K>::max();
		rg_k3  = (tid+6	<seg_size)?key[k+tid+6	]:std::numeric_limits<K>::max();
		if(tid+0	<seg_size) rg_v0 = tid+0	;
		if(tid+2	<seg_size) rg_v1 = tid+2	;
		if(tid+4	<seg_size) rg_v2 = tid+4	;
		if(tid+6	<seg_size) rg_v3 = tid+6	;

		// sort 8 elements
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

		if((tid<<2)+0 <seg_size) keyB[k+(tid<<2)+0] = rg_k0;
		if((tid<<2)+1 <seg_size) keyB[k+(tid<<2)+1] = rg_k1;
		if((tid<<2)+2 <seg_size) keyB[k+(tid<<2)+2] = rg_k2;
		if((tid<<2)+3 <seg_size) keyB[k+(tid<<2)+3] = rg_k3;

		if((tid<<2)+0 <seg_size) valB[k+(tid<<2)+0] = val[k+rg_v0];
		if((tid<<2)+1 <seg_size) valB[k+(tid<<2)+1] = val[k+rg_v1];
		if((tid<<2)+2 <seg_size) valB[k+(tid<<2)+2] = val[k+rg_v2];
		if((tid<<2)+3 <seg_size) valB[k+(tid<<2)+3] = val[k+rg_v3];
	}
}

// 4 threads | 1 ppt
template<class K, class T>
__global__
void my_t4_ppt1_orig(K *key, T *val, K *keyB, T *valB, int n, int *segs, int *bin, int bin_size, int length) {
	const int gid = threadIdx.x + blockIdx.x * blockDim.x;
	const int bin_it = (gid>>2);
	const int tid = (threadIdx.x & 3);
	const int bit1 = (tid>>0)&0x1;
	const int bit2 = (tid>>1)&0x1;
	K rg_k0;
	int rg_v0;
	int k;
	int seg_size;

	if(bin_it < bin_size) {
		k = segs[bin[bin_it]];
		seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
		rg_k0  = (tid+0	<seg_size)?key[k+tid+0	]:std::numeric_limits<K>::max();
		if(tid+0	<seg_size) rg_v0 = tid+0	;

		// sort 4 elements
		// exch_intxn: generate exch_intxn()
		exch_intxn(rg_k0, rg_v0, 0x1, bit1);
		// exch_intxn: generate exch_intxn()
		exch_intxn(rg_k0, rg_v0, 0x3, bit2);
		// exch_paral: generate exch_paral()
		exch_paral(rg_k0, rg_v0, 0x1, bit1);

		if((tid<<0)+0 <seg_size) keyB[k+(tid<<0)+0] = rg_k0;

		if((tid<<0)+0 <seg_size) valB[k+(tid<<0)+0] = val[k+rg_v0];
	}
}

// 8 threads | 1 ppt
template<class K, class T>
__global__
void my_t8_ppt1_orig(K *key, T *val, K *keyB, T *valB, int n, int *segs, int *bin, int bin_size, int length) {
	const int gid = threadIdx.x + blockIdx.x * blockDim.x;
	const int bin_it = (gid>>3);
	const int tid = (threadIdx.x & 7);
	const int bit1 = (tid>>0)&0x1;
	const int bit2 = (tid>>1)&0x1;
	const int bit3 = (tid>>2)&0x1;
	K rg_k0;
	int rg_v0;
	int k;
	int seg_size;

	if(bin_it < bin_size) {
		k = segs[bin[bin_it]];
		seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
		rg_k0  = (tid+0	<seg_size)?key[k+tid+0	]:std::numeric_limits<K>::max();
		if(tid+0	<seg_size) rg_v0 = tid+0	;

		// sort 8 elements
		// exch_intxn: generate exch_intxn()
		exch_intxn(rg_k0, rg_v0, 0x1, bit1);
		// exch_intxn: generate exch_intxn()
		exch_intxn(rg_k0, rg_v0, 0x3, bit2);
		// exch_paral: generate exch_paral()
		exch_paral(rg_k0, rg_v0, 0x1, bit1);
		// exch_intxn: generate exch_intxn()
		exch_intxn(rg_k0, rg_v0, 0x7, bit3);
		// exch_paral: generate exch_paral()
		exch_paral(rg_k0, rg_v0, 0x2, bit2);
		// exch_paral: generate exch_paral()
		exch_paral(rg_k0, rg_v0, 0x1, bit1);

		if((tid<<0)+0 <seg_size) keyB[k+(tid<<0)+0] = rg_k0;

		if((tid<<0)+0 <seg_size) valB[k+(tid<<0)+0] = val[k+rg_v0];
	}
}

// 16 threads | 1 ppt
template<class K, class T>
__global__
void my_t16_ppt1_orig(K *key, T *val, K *keyB, T *valB, int n, int *segs, int *bin, int bin_size, int length) {
	const int gid = threadIdx.x + blockIdx.x * blockDim.x;
	const int bin_it = (gid>>4);
	const int tid = (threadIdx.x & 15);
	const int bit1 = (tid>>0)&0x1;
	const int bit2 = (tid>>1)&0x1;
	const int bit3 = (tid>>2)&0x1;
	const int bit4 = (tid>>3)&0x1;
	K rg_k0;
	int rg_v0;
	int k;
	int seg_size;

	if(bin_it < bin_size) {
		k = segs[bin[bin_it]];
		seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
		rg_k0  = (tid+0	<seg_size)?key[k+tid+0	]:std::numeric_limits<K>::max();
		if(tid+0	<seg_size) rg_v0 = tid+0	;

		// sort 16 elements
		// exch_intxn: generate exch_intxn()
		exch_intxn(rg_k0, rg_v0, 0x1, bit1);
		// exch_intxn: generate exch_intxn()
		exch_intxn(rg_k0, rg_v0, 0x3, bit2);
		// exch_paral: generate exch_paral()
		exch_paral(rg_k0, rg_v0, 0x1, bit1);
		// exch_intxn: generate exch_intxn()
		exch_intxn(rg_k0, rg_v0, 0x7, bit3);
		// exch_paral: generate exch_paral()
		exch_paral(rg_k0, rg_v0, 0x2, bit2);
		// exch_paral: generate exch_paral()
		exch_paral(rg_k0, rg_v0, 0x1, bit1);
		// exch_intxn: generate exch_intxn()
		exch_intxn(rg_k0, rg_v0, 0xf, bit4);
		// exch_paral: generate exch_paral()
		exch_paral(rg_k0, rg_v0, 0x4, bit3);
		// exch_paral: generate exch_paral()
		exch_paral(rg_k0, rg_v0, 0x2, bit2);
		// exch_paral: generate exch_paral()
		exch_paral(rg_k0, rg_v0, 0x1, bit1);

		if((tid<<0)+0 <seg_size) keyB[k+(tid<<0)+0] = rg_k0;

		if((tid<<0)+0 <seg_size) valB[k+(tid<<0)+0] = val[k+rg_v0];
	}
}

// 16 threads | 2 ppt
template<class K, class T>
__global__
void my_t16_ppt2_orig(K *key, T *val, K *keyB, T *valB, int n, int *segs, int *bin, int bin_size, int length) {
	const int gid = threadIdx.x + blockIdx.x * blockDim.x;
	const int bin_it = (gid>>4);
	const int tid = (threadIdx.x & 15);
	const int bit1 = (tid>>0)&0x1;
	const int bit2 = (tid>>1)&0x1;
	const int bit3 = (tid>>2)&0x1;
	const int bit4 = (tid>>3)&0x1;
	K rg_k0;
	K rg_k1;
	int rg_v0;
	int rg_v1;
	int k;
	int seg_size;

	if(bin_it < bin_size) {
		k = segs[bin[bin_it]];
		seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
		rg_k0  = (tid+0	<seg_size)?key[k+tid+0	]:std::numeric_limits<K>::max();
		rg_k1  = (tid+16	<seg_size)?key[k+tid+16	]:std::numeric_limits<K>::max();
		if(tid+0	<seg_size) rg_v0 = tid+0	;
		if(tid+16	<seg_size) rg_v1 = tid+16	;

		// sort 32 elements
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

		if((tid<<1)+0 <seg_size) keyB[k+(tid<<1)+0] = rg_k0;
		if((tid<<1)+1 <seg_size) keyB[k+(tid<<1)+1] = rg_k1;

		if((tid<<1)+0 <seg_size) valB[k+(tid<<1)+0] = val[k+rg_v0];
		if((tid<<1)+1 <seg_size) valB[k+(tid<<1)+1] = val[k+rg_v1];
	}
}

// 4 threads | 4 ppt | *strided*
template<class K, class T>
__global__
void my_t4_ppt4_strd(K *key, T *val, K *keyB, T *valB, int n, int *segs, int *bin, int bin_size, int length) {
	const int gid = threadIdx.x + blockIdx.x * blockDim.x;
	const int bin_it = (gid>>2);
	const int tid = (threadIdx.x & 3);
	const int bit1 = (tid>>0)&0x1;
	const int bit2 = (tid>>1)&0x1;
	K rg_k0;
	K rg_k1;
	K rg_k2;
	K rg_k3;
	int rg_v0;
	int rg_v1;
	int rg_v2;
	int rg_v3;
	int normalized_bin_size = (bin_size/8)*8;
	int k;
	int seg_size;

	if(bin_it < bin_size) {
		k = segs[bin[bin_it]];
		seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
		rg_k0  = (tid+0	<seg_size)?key[k+tid+0	]:std::numeric_limits<K>::max();
		rg_k1  = (tid+4	<seg_size)?key[k+tid+4	]:std::numeric_limits<K>::max();
		rg_k2  = (tid+8	<seg_size)?key[k+tid+8	]:std::numeric_limits<K>::max();
		rg_k3  = (tid+12	<seg_size)?key[k+tid+12	]:std::numeric_limits<K>::max();
		if(tid+0	<seg_size) rg_v0 = tid+0	;
		if(tid+4	<seg_size) rg_v1 = tid+4	;
		if(tid+8	<seg_size) rg_v2 = tid+8	;
		if(tid+12	<seg_size) rg_v3 = tid+12	;

		// sort 16 elements
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
	}
	if (bin_it < normalized_bin_size) {
		// transpose the in register data of the whole warp
		int lane_id = threadIdx.x & 31;
		rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x1);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x1);
		rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x1);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x1);
		if(lane_id&0x1) SWP(K, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
		if(lane_id&0x1) SWP(K, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
		rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x1);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x1);
		rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x1);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x1);
		rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x2);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x2);
		rg_v2  = __shfl_xor_sync(0xffffffff,rg_v2 , 0x2);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x2);
		if(lane_id&0x2) SWP(K, rg_k0 , rg_k2 , int, rg_v0 , rg_v2 );
		if(lane_id&0x2) SWP(K, rg_k1 , rg_k3 , int, rg_v1 , rg_v3 );
		rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x2);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x2);
		rg_v2  = __shfl_xor_sync(0xffffffff,rg_v2 , 0x2);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x2);
		rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x4);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x4);
		rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x4);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x4);
		if(lane_id&0x4) SWP(K, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
		if(lane_id&0x4) SWP(K, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
		rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x4);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x4);
		rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x4);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x4);
		rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x8);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x8);
		rg_v2  = __shfl_xor_sync(0xffffffff,rg_v2 , 0x8);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x8);
		if(lane_id&0x8) SWP(K, rg_k0 , rg_k2 , int, rg_v0 , rg_v2 );
		if(lane_id&0x8) SWP(K, rg_k1 , rg_k3 , int, rg_v1 , rg_v3 );
		rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x8);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x8);
		rg_v2  = __shfl_xor_sync(0xffffffff,rg_v2 , 0x8);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x8);
		rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x10);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x10);
		rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x10);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x10);
		if(lane_id&0x10) SWP(K, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
		if(lane_id&0x10) SWP(K, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
		rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x10);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x10);
		rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x10);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x10);
		int kk;
		int ss;
		int base = (lane_id/16)*16;
		kk = __shfl_sync(0xffffffff, k, 0);
		ss = __shfl_sync(0xffffffff, seg_size, 0);
		if((lane_id>>4)==0&&lane_id-base<ss) keyB[kk+lane_id-base] = rg_k0;
		kk = __shfl_sync(0xffffffff, k, 4);
		ss = __shfl_sync(0xffffffff, seg_size, 4);
		if((lane_id>>4)==1&&lane_id-base<ss) keyB[kk+lane_id-base] = rg_k0;
		kk = __shfl_sync(0xffffffff, k, 8);
		ss = __shfl_sync(0xffffffff, seg_size, 8);
		if((lane_id>>4)==0&&lane_id-base<ss) keyB[kk+lane_id-base] = rg_k2;
		kk = __shfl_sync(0xffffffff, k, 12);
		ss = __shfl_sync(0xffffffff, seg_size, 12);
		if((lane_id>>4)==1&&lane_id-base<ss) keyB[kk+lane_id-base] = rg_k2;
		kk = __shfl_sync(0xffffffff, k, 16);
		ss = __shfl_sync(0xffffffff, seg_size, 16);
		if((lane_id>>4)==0&&lane_id-base<ss) keyB[kk+lane_id-base] = rg_k1;
		kk = __shfl_sync(0xffffffff, k, 20);
		ss = __shfl_sync(0xffffffff, seg_size, 20);
		if((lane_id>>4)==1&&lane_id-base<ss) keyB[kk+lane_id-base] = rg_k1;
		kk = __shfl_sync(0xffffffff, k, 24);
		ss = __shfl_sync(0xffffffff, seg_size, 24);
		if((lane_id>>4)==0&&lane_id-base<ss) keyB[kk+lane_id-base] = rg_k3;
		kk = __shfl_sync(0xffffffff, k, 28);
		ss = __shfl_sync(0xffffffff, seg_size, 28);
		if((lane_id>>4)==1&&lane_id-base<ss) keyB[kk+lane_id-base] = rg_k3;
		kk = __shfl_sync(0xffffffff, k, 0);
		ss = __shfl_sync(0xffffffff, seg_size, 0);
		if((lane_id>>4)==0&&lane_id-base<ss) valB[kk+lane_id-base] = val[kk+rg_v0];
		kk = __shfl_sync(0xffffffff, k, 4);
		ss = __shfl_sync(0xffffffff, seg_size, 4);
		if((lane_id>>4)==1&&lane_id-base<ss) valB[kk+lane_id-base] = val[kk+rg_v0];
		kk = __shfl_sync(0xffffffff, k, 8);
		ss = __shfl_sync(0xffffffff, seg_size, 8);
		if((lane_id>>4)==0&&lane_id-base<ss) valB[kk+lane_id-base] = val[kk+rg_v2];
		kk = __shfl_sync(0xffffffff, k, 12);
		ss = __shfl_sync(0xffffffff, seg_size, 12);
		if((lane_id>>4)==1&&lane_id-base<ss) valB[kk+lane_id-base] = val[kk+rg_v2];
		kk = __shfl_sync(0xffffffff, k, 16);
		ss = __shfl_sync(0xffffffff, seg_size, 16);
		if((lane_id>>4)==0&&lane_id-base<ss) valB[kk+lane_id-base] = val[kk+rg_v1];
		kk = __shfl_sync(0xffffffff, k, 20);
		ss = __shfl_sync(0xffffffff, seg_size, 20);
		if((lane_id>>4)==1&&lane_id-base<ss) valB[kk+lane_id-base] = val[kk+rg_v1];
		kk = __shfl_sync(0xffffffff, k, 24);
		ss = __shfl_sync(0xffffffff, seg_size, 24);
		if((lane_id>>4)==0&&lane_id-base<ss) valB[kk+lane_id-base] = val[kk+rg_v3];
		kk = __shfl_sync(0xffffffff, k, 28);
		ss = __shfl_sync(0xffffffff, seg_size, 28);
		if((lane_id>>4)==1&&lane_id-base<ss) valB[kk+lane_id-base] = val[kk+rg_v3];
	}
	else if(bin_it < bin_size) {
		if((tid<<2)+0 <seg_size) keyB[k+(tid<<2)+0] = rg_k0;
		if((tid<<2)+1 <seg_size) keyB[k+(tid<<2)+1] = rg_k1;
		if((tid<<2)+2 <seg_size) keyB[k+(tid<<2)+2] = rg_k2;
		if((tid<<2)+3 <seg_size) keyB[k+(tid<<2)+3] = rg_k3;

		if((tid<<2)+0 <seg_size) valB[k+(tid<<2)+0] = val[k+rg_v0];
		if((tid<<2)+1 <seg_size) valB[k+(tid<<2)+1] = val[k+rg_v1];
		if((tid<<2)+2 <seg_size) valB[k+(tid<<2)+2] = val[k+rg_v2];
		if((tid<<2)+3 <seg_size) valB[k+(tid<<2)+3] = val[k+rg_v3];
	}
}

// 8 threads | 4 ppt | *strided*
template<class K, class T>
__global__
void my_t8_ppt4_strd(K *key, T *val, K *keyB, T *valB, int n, int *segs, int *bin, int bin_size, int length) {
	const int gid = threadIdx.x + blockIdx.x * blockDim.x;
	const int bin_it = (gid>>3);
	const int tid = (threadIdx.x & 7);
	const int bit1 = (tid>>0)&0x1;
	const int bit2 = (tid>>1)&0x1;
	const int bit3 = (tid>>2)&0x1;
	K rg_k0;
	K rg_k1;
	K rg_k2;
	K rg_k3;
	int rg_v0;
	int rg_v1;
	int rg_v2;
	int rg_v3;
	int normalized_bin_size = (bin_size/4)*4;
	int k;
	int seg_size;

	if(bin_it < bin_size) {
		k = segs[bin[bin_it]];
		seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
		rg_k0  = (tid+0	<seg_size)?key[k+tid+0	]:std::numeric_limits<K>::max();
		rg_k1  = (tid+8	<seg_size)?key[k+tid+8	]:std::numeric_limits<K>::max();
		rg_k2  = (tid+16	<seg_size)?key[k+tid+16	]:std::numeric_limits<K>::max();
		rg_k3  = (tid+24	<seg_size)?key[k+tid+24	]:std::numeric_limits<K>::max();
		if(tid+0	<seg_size) rg_v0 = tid+0	;
		if(tid+8	<seg_size) rg_v1 = tid+8	;
		if(tid+16	<seg_size) rg_v2 = tid+16	;
		if(tid+24	<seg_size) rg_v3 = tid+24	;

		// sort 32 elements
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
	}
	if (bin_it < normalized_bin_size) {
		// transpose the in register data of the whole warp
		int lane_id = threadIdx.x & 31;
		rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x1);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x1);
		rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x1);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x1);
		if(lane_id&0x1) SWP(K, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
		if(lane_id&0x1) SWP(K, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
		rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x1);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x1);
		rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x1);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x1);
		rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x2);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x2);
		rg_v2  = __shfl_xor_sync(0xffffffff,rg_v2 , 0x2);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x2);
		if(lane_id&0x2) SWP(K, rg_k0 , rg_k2 , int, rg_v0 , rg_v2 );
		if(lane_id&0x2) SWP(K, rg_k1 , rg_k3 , int, rg_v1 , rg_v3 );
		rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x2);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x2);
		rg_v2  = __shfl_xor_sync(0xffffffff,rg_v2 , 0x2);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x2);
		rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x4);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x4);
		rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x4);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x4);
		if(lane_id&0x4) SWP(K, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
		if(lane_id&0x4) SWP(K, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
		rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x4);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x4);
		rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x4);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x4);
		rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x8);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x8);
		rg_v2  = __shfl_xor_sync(0xffffffff,rg_v2 , 0x8);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x8);
		if(lane_id&0x8) SWP(K, rg_k0 , rg_k2 , int, rg_v0 , rg_v2 );
		if(lane_id&0x8) SWP(K, rg_k1 , rg_k3 , int, rg_v1 , rg_v3 );
		rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x8);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x8);
		rg_v2  = __shfl_xor_sync(0xffffffff,rg_v2 , 0x8);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x8);
		rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x10);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x10);
		rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x10);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x10);
		if(lane_id&0x10) SWP(K, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
		if(lane_id&0x10) SWP(K, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
		rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x10);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x10);
		rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x10);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x10);
		int kk;
		int ss;
		kk = __shfl_sync(0xffffffff, k, 0);
		ss = __shfl_sync(0xffffffff, seg_size, 0);
		if (lane_id + 0 < ss) keyB[kk+lane_id+0] = rg_k0;
		kk = __shfl_sync(0xffffffff, k, 8);
		ss = __shfl_sync(0xffffffff, seg_size, 8);
		if (lane_id + 0 < ss) keyB[kk+lane_id+0] = rg_k2;
		kk = __shfl_sync(0xffffffff, k, 16);
		ss = __shfl_sync(0xffffffff, seg_size, 16);
		if (lane_id + 0 < ss) keyB[kk+lane_id+0] = rg_k1;
		kk = __shfl_sync(0xffffffff, k, 24);
		ss = __shfl_sync(0xffffffff, seg_size, 24);
		if (lane_id + 0 < ss) keyB[kk+lane_id+0] = rg_k3;
		kk = __shfl_sync(0xffffffff, k, 0);
		ss = __shfl_sync(0xffffffff, seg_size, 0);
		if(lane_id+0 < ss) valB[kk+lane_id+0] = val[kk+rg_v0];
		kk = __shfl_sync(0xffffffff, k, 8);
		ss = __shfl_sync(0xffffffff, seg_size, 8);
		if(lane_id+0 < ss) valB[kk+lane_id+0] = val[kk+rg_v2];
		kk = __shfl_sync(0xffffffff, k, 16);
		ss = __shfl_sync(0xffffffff, seg_size, 16);
		if(lane_id+0 < ss) valB[kk+lane_id+0] = val[kk+rg_v1];
		kk = __shfl_sync(0xffffffff, k, 24);
		ss = __shfl_sync(0xffffffff, seg_size, 24);
		if(lane_id+0 < ss) valB[kk+lane_id+0] = val[kk+rg_v3];
	}
	else if(bin_it < bin_size) {
		if((tid<<2)+0 <seg_size) keyB[k+(tid<<2)+0] = rg_k0;
		if((tid<<2)+1 <seg_size) keyB[k+(tid<<2)+1] = rg_k1;
		if((tid<<2)+2 <seg_size) keyB[k+(tid<<2)+2] = rg_k2;
		if((tid<<2)+3 <seg_size) keyB[k+(tid<<2)+3] = rg_k3;

		if((tid<<2)+0 <seg_size) valB[k+(tid<<2)+0] = val[k+rg_v0];
		if((tid<<2)+1 <seg_size) valB[k+(tid<<2)+1] = val[k+rg_v1];
		if((tid<<2)+2 <seg_size) valB[k+(tid<<2)+2] = val[k+rg_v2];
		if((tid<<2)+3 <seg_size) valB[k+(tid<<2)+3] = val[k+rg_v3];
	}
}

// 8 threads | 8 ppt | *strided*
template<class K, class T>
__global__
void my_t8_ppt8_strd(K *key, T *val, K *keyB, T *valB, int n, int *segs, int *bin, int bin_size, int length) {
	const int gid = threadIdx.x + blockIdx.x * blockDim.x;
	const int bin_it = (gid>>3);
	const int tid = (threadIdx.x & 7);
	const int bit1 = (tid>>0)&0x1;
	const int bit2 = (tid>>1)&0x1;
	const int bit3 = (tid>>2)&0x1;
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
	int normalized_bin_size = (bin_size/4)*4;
	int k;
	int seg_size;

	if(bin_it < bin_size) {
		k = segs[bin[bin_it]];
		seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
		rg_k0  = (tid+0	<seg_size)?key[k+tid+0	]:std::numeric_limits<K>::max();
		rg_k1  = (tid+8	<seg_size)?key[k+tid+8	]:std::numeric_limits<K>::max();
		rg_k2  = (tid+16	<seg_size)?key[k+tid+16	]:std::numeric_limits<K>::max();
		rg_k3  = (tid+24	<seg_size)?key[k+tid+24	]:std::numeric_limits<K>::max();
		rg_k4  = (tid+32	<seg_size)?key[k+tid+32	]:std::numeric_limits<K>::max();
		rg_k5  = (tid+40	<seg_size)?key[k+tid+40	]:std::numeric_limits<K>::max();
		rg_k6  = (tid+48	<seg_size)?key[k+tid+48	]:std::numeric_limits<K>::max();
		rg_k7  = (tid+56	<seg_size)?key[k+tid+56	]:std::numeric_limits<K>::max();
		if(tid+0	<seg_size) rg_v0 = tid+0	;
		if(tid+8	<seg_size) rg_v1 = tid+8	;
		if(tid+16	<seg_size) rg_v2 = tid+16	;
		if(tid+24	<seg_size) rg_v3 = tid+24	;
		if(tid+32	<seg_size) rg_v4 = tid+32	;
		if(tid+40	<seg_size) rg_v5 = tid+40	;
		if(tid+48	<seg_size) rg_v6 = tid+48	;
		if(tid+56	<seg_size) rg_v7 = tid+56	;

		// sort 64 elements
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
	}
	if (bin_it < normalized_bin_size) {
		// transpose the in register data of the whole warp
		int lane_id = threadIdx.x & 31;
		rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x1);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x1);
		rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x1);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x1);
		rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x1);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x1);
		rg_v5  = __shfl_xor_sync(0xffffffff,rg_v5 , 0x1);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x1);
		if(lane_id&0x1) SWP(K, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
		if(lane_id&0x1) SWP(K, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
		if(lane_id&0x1) SWP(K, rg_k4 , rg_k5 , int, rg_v4 , rg_v5 );
		if(lane_id&0x1) SWP(K, rg_k6 , rg_k7 , int, rg_v6 , rg_v7 );
		rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x1);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x1);
		rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x1);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x1);
		rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x1);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x1);
		rg_v5  = __shfl_xor_sync(0xffffffff,rg_v5 , 0x1);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x1);
		rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x2);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x2);
		rg_k6  = __shfl_xor_sync(0xffffffff,rg_k6 , 0x2);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x2);
		rg_v2  = __shfl_xor_sync(0xffffffff,rg_v2 , 0x2);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x2);
		rg_v6  = __shfl_xor_sync(0xffffffff,rg_v6 , 0x2);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x2);
		if(lane_id&0x2) SWP(K, rg_k0 , rg_k2 , int, rg_v0 , rg_v2 );
		if(lane_id&0x2) SWP(K, rg_k1 , rg_k3 , int, rg_v1 , rg_v3 );
		if(lane_id&0x2) SWP(K, rg_k4 , rg_k6 , int, rg_v4 , rg_v6 );
		if(lane_id&0x2) SWP(K, rg_k5 , rg_k7 , int, rg_v5 , rg_v7 );
		rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x2);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x2);
		rg_k6  = __shfl_xor_sync(0xffffffff,rg_k6 , 0x2);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x2);
		rg_v2  = __shfl_xor_sync(0xffffffff,rg_v2 , 0x2);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x2);
		rg_v6  = __shfl_xor_sync(0xffffffff,rg_v6 , 0x2);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x2);
		rg_k4  = __shfl_xor_sync(0xffffffff,rg_k4 , 0x4);
		rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x4);
		rg_k6  = __shfl_xor_sync(0xffffffff,rg_k6 , 0x4);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x4);
		rg_v4  = __shfl_xor_sync(0xffffffff,rg_v4 , 0x4);
		rg_v5  = __shfl_xor_sync(0xffffffff,rg_v5 , 0x4);
		rg_v6  = __shfl_xor_sync(0xffffffff,rg_v6 , 0x4);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x4);
		if(lane_id&0x4) SWP(K, rg_k0 , rg_k4 , int, rg_v0 , rg_v4 );
		if(lane_id&0x4) SWP(K, rg_k1 , rg_k5 , int, rg_v1 , rg_v5 );
		if(lane_id&0x4) SWP(K, rg_k2 , rg_k6 , int, rg_v2 , rg_v6 );
		if(lane_id&0x4) SWP(K, rg_k3 , rg_k7 , int, rg_v3 , rg_v7 );
		rg_k4  = __shfl_xor_sync(0xffffffff,rg_k4 , 0x4);
		rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x4);
		rg_k6  = __shfl_xor_sync(0xffffffff,rg_k6 , 0x4);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x4);
		rg_v4  = __shfl_xor_sync(0xffffffff,rg_v4 , 0x4);
		rg_v5  = __shfl_xor_sync(0xffffffff,rg_v5 , 0x4);
		rg_v6  = __shfl_xor_sync(0xffffffff,rg_v6 , 0x4);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x4);
		rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x8);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x8);
		rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x8);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x8);
		rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x8);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x8);
		rg_v5  = __shfl_xor_sync(0xffffffff,rg_v5 , 0x8);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x8);
		if(lane_id&0x8) SWP(K, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
		if(lane_id&0x8) SWP(K, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
		if(lane_id&0x8) SWP(K, rg_k4 , rg_k5 , int, rg_v4 , rg_v5 );
		if(lane_id&0x8) SWP(K, rg_k6 , rg_k7 , int, rg_v6 , rg_v7 );
		rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x8);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x8);
		rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x8);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x8);
		rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x8);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x8);
		rg_v5  = __shfl_xor_sync(0xffffffff,rg_v5 , 0x8);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x8);
		rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x10);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x10);
		rg_k6  = __shfl_xor_sync(0xffffffff,rg_k6 , 0x10);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x10);
		rg_v2  = __shfl_xor_sync(0xffffffff,rg_v2 , 0x10);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x10);
		rg_v6  = __shfl_xor_sync(0xffffffff,rg_v6 , 0x10);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x10);
		if(lane_id&0x10) SWP(K, rg_k0 , rg_k2 , int, rg_v0 , rg_v2 );
		if(lane_id&0x10) SWP(K, rg_k1 , rg_k3 , int, rg_v1 , rg_v3 );
		if(lane_id&0x10) SWP(K, rg_k4 , rg_k6 , int, rg_v4 , rg_v6 );
		if(lane_id&0x10) SWP(K, rg_k5 , rg_k7 , int, rg_v5 , rg_v7 );
		rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x10);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x10);
		rg_k6  = __shfl_xor_sync(0xffffffff,rg_k6 , 0x10);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x10);
		rg_v2  = __shfl_xor_sync(0xffffffff,rg_v2 , 0x10);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x10);
		rg_v6  = __shfl_xor_sync(0xffffffff,rg_v6 , 0x10);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x10);
		int kk;
		int ss;
		kk = __shfl_sync(0xffffffff, k, 0);
		ss = __shfl_sync(0xffffffff, seg_size, 0);
		if (lane_id + 0 < ss) keyB[kk+lane_id+0] = rg_k0;
		if (lane_id + 32 < ss) keyB[kk+lane_id+32] = rg_k4;
		kk = __shfl_sync(0xffffffff, k, 8);
		ss = __shfl_sync(0xffffffff, seg_size, 8);
		if (lane_id + 0 < ss) keyB[kk+lane_id+0] = rg_k1;
		if (lane_id + 32 < ss) keyB[kk+lane_id+32] = rg_k5;
		kk = __shfl_sync(0xffffffff, k, 16);
		ss = __shfl_sync(0xffffffff, seg_size, 16);
		if (lane_id + 0 < ss) keyB[kk+lane_id+0] = rg_k2;
		if (lane_id + 32 < ss) keyB[kk+lane_id+32] = rg_k6;
		kk = __shfl_sync(0xffffffff, k, 24);
		ss = __shfl_sync(0xffffffff, seg_size, 24);
		if (lane_id + 0 < ss) keyB[kk+lane_id+0] = rg_k3;
		if (lane_id + 32 < ss) keyB[kk+lane_id+32] = rg_k7;
		kk = __shfl_sync(0xffffffff, k, 0);
		ss = __shfl_sync(0xffffffff, seg_size, 0);
		if(lane_id+0 < ss) valB[kk+lane_id+0] = val[kk+rg_v0];
		if(lane_id+32 < ss) valB[kk+lane_id+32] = val[kk+rg_v4];
		kk = __shfl_sync(0xffffffff, k, 8);
		ss = __shfl_sync(0xffffffff, seg_size, 8);
		if(lane_id+0 < ss) valB[kk+lane_id+0] = val[kk+rg_v1];
		if(lane_id+32 < ss) valB[kk+lane_id+32] = val[kk+rg_v5];
		kk = __shfl_sync(0xffffffff, k, 16);
		ss = __shfl_sync(0xffffffff, seg_size, 16);
		if(lane_id+0 < ss) valB[kk+lane_id+0] = val[kk+rg_v2];
		if(lane_id+32 < ss) valB[kk+lane_id+32] = val[kk+rg_v6];
		kk = __shfl_sync(0xffffffff, k, 24);
		ss = __shfl_sync(0xffffffff, seg_size, 24);
		if(lane_id+0 < ss) valB[kk+lane_id+0] = val[kk+rg_v3];
		if(lane_id+32 < ss) valB[kk+lane_id+32] = val[kk+rg_v7];
	}
	else if(bin_it < bin_size) {
		if((tid<<3)+0 <seg_size) keyB[k+(tid<<3)+0] = rg_k0;
		if((tid<<3)+1 <seg_size) keyB[k+(tid<<3)+1] = rg_k1;
		if((tid<<3)+2 <seg_size) keyB[k+(tid<<3)+2] = rg_k2;
		if((tid<<3)+3 <seg_size) keyB[k+(tid<<3)+3] = rg_k3;
		if((tid<<3)+4 <seg_size) keyB[k+(tid<<3)+4] = rg_k4;
		if((tid<<3)+5 <seg_size) keyB[k+(tid<<3)+5] = rg_k5;
		if((tid<<3)+6 <seg_size) keyB[k+(tid<<3)+6] = rg_k6;
		if((tid<<3)+7 <seg_size) keyB[k+(tid<<3)+7] = rg_k7;

		if((tid<<3)+0 <seg_size) valB[k+(tid<<3)+0] = val[k+rg_v0];
		if((tid<<3)+1 <seg_size) valB[k+(tid<<3)+1] = val[k+rg_v1];
		if((tid<<3)+2 <seg_size) valB[k+(tid<<3)+2] = val[k+rg_v2];
		if((tid<<3)+3 <seg_size) valB[k+(tid<<3)+3] = val[k+rg_v3];
		if((tid<<3)+4 <seg_size) valB[k+(tid<<3)+4] = val[k+rg_v4];
		if((tid<<3)+5 <seg_size) valB[k+(tid<<3)+5] = val[k+rg_v5];
		if((tid<<3)+6 <seg_size) valB[k+(tid<<3)+6] = val[k+rg_v6];
		if((tid<<3)+7 <seg_size) valB[k+(tid<<3)+7] = val[k+rg_v7];
	}
}

// 8 threads | 16 ppt | *strided*
template<class K, class T>
__global__
void my_t8_ppt16_strd(K *key, T *val, K *keyB, T *valB, int n, int *segs, int *bin, int bin_size, int length) {
	const int gid = threadIdx.x + blockIdx.x * blockDim.x;
	const int bin_it = (gid>>3);
	const int tid = (threadIdx.x & 7);
	const int bit1 = (tid>>0)&0x1;
	const int bit2 = (tid>>1)&0x1;
	const int bit3 = (tid>>2)&0x1;
	K rg_k0;
	K rg_k1;
	K rg_k2;
	K rg_k3;
	K rg_k4;
	K rg_k5;
	K rg_k6;
	K rg_k7;
	K rg_k8;
	K rg_k9;
	K rg_k10;
	K rg_k11;
	K rg_k12;
	K rg_k13;
	K rg_k14;
	K rg_k15;
	int rg_v0;
	int rg_v1;
	int rg_v2;
	int rg_v3;
	int rg_v4;
	int rg_v5;
	int rg_v6;
	int rg_v7;
	int rg_v8;
	int rg_v9;
	int rg_v10;
	int rg_v11;
	int rg_v12;
	int rg_v13;
	int rg_v14;
	int rg_v15;
	int normalized_bin_size = (bin_size/4)*4;
	int k;
	int seg_size;

	if(bin_it < bin_size) {
		k = segs[bin[bin_it]];
		seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
		rg_k0  = (tid+0	<seg_size)?key[k+tid+0	]:std::numeric_limits<K>::max();
		rg_k1  = (tid+8	<seg_size)?key[k+tid+8	]:std::numeric_limits<K>::max();
		rg_k2  = (tid+16	<seg_size)?key[k+tid+16	]:std::numeric_limits<K>::max();
		rg_k3  = (tid+24	<seg_size)?key[k+tid+24	]:std::numeric_limits<K>::max();
		rg_k4  = (tid+32	<seg_size)?key[k+tid+32	]:std::numeric_limits<K>::max();
		rg_k5  = (tid+40	<seg_size)?key[k+tid+40	]:std::numeric_limits<K>::max();
		rg_k6  = (tid+48	<seg_size)?key[k+tid+48	]:std::numeric_limits<K>::max();
		rg_k7  = (tid+56	<seg_size)?key[k+tid+56	]:std::numeric_limits<K>::max();
		rg_k8  = (tid+64	<seg_size)?key[k+tid+64	]:std::numeric_limits<K>::max();
		rg_k9  = (tid+72	<seg_size)?key[k+tid+72	]:std::numeric_limits<K>::max();
		rg_k10  = (tid+80	<seg_size)?key[k+tid+80	]:std::numeric_limits<K>::max();
		rg_k11  = (tid+88	<seg_size)?key[k+tid+88	]:std::numeric_limits<K>::max();
		rg_k12  = (tid+96	<seg_size)?key[k+tid+96	]:std::numeric_limits<K>::max();
		rg_k13  = (tid+104	<seg_size)?key[k+tid+104	]:std::numeric_limits<K>::max();
		rg_k14  = (tid+112	<seg_size)?key[k+tid+112	]:std::numeric_limits<K>::max();
		rg_k15  = (tid+120	<seg_size)?key[k+tid+120	]:std::numeric_limits<K>::max();
		if(tid+0	<seg_size) rg_v0 = tid+0	;
		if(tid+8	<seg_size) rg_v1 = tid+8	;
		if(tid+16	<seg_size) rg_v2 = tid+16	;
		if(tid+24	<seg_size) rg_v3 = tid+24	;
		if(tid+32	<seg_size) rg_v4 = tid+32	;
		if(tid+40	<seg_size) rg_v5 = tid+40	;
		if(tid+48	<seg_size) rg_v6 = tid+48	;
		if(tid+56	<seg_size) rg_v7 = tid+56	;
		if(tid+64	<seg_size) rg_v8 = tid+64	;
		if(tid+72	<seg_size) rg_v9 = tid+72	;
		if(tid+80	<seg_size) rg_v10 = tid+80	;
		if(tid+88	<seg_size) rg_v11 = tid+88	;
		if(tid+96	<seg_size) rg_v12 = tid+96	;
		if(tid+104	<seg_size) rg_v13 = tid+104	;
		if(tid+112	<seg_size) rg_v14 = tid+112	;
		if(tid+120	<seg_size) rg_v15 = tid+120	;

		// sort 128 elements
		// exch_intxn: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
		CMP_SWP(K, rg_k2, rg_k3, int, rg_v2, rg_v3);
		CMP_SWP(K, rg_k4, rg_k5, int, rg_v4, rg_v5);
		CMP_SWP(K, rg_k6, rg_k7, int, rg_v6, rg_v7);
		CMP_SWP(K, rg_k8, rg_k9, int, rg_v8, rg_v9);
		CMP_SWP(K, rg_k10, rg_k11, int, rg_v10, rg_v11);
		CMP_SWP(K, rg_k12, rg_k13, int, rg_v12, rg_v13);
		CMP_SWP(K, rg_k14, rg_k15, int, rg_v14, rg_v15);
		// exch_intxn: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k3, int, rg_v0, rg_v3);
		CMP_SWP(K, rg_k1, rg_k2, int, rg_v1, rg_v2);
		CMP_SWP(K, rg_k4, rg_k7, int, rg_v4, rg_v7);
		CMP_SWP(K, rg_k5, rg_k6, int, rg_v5, rg_v6);
		CMP_SWP(K, rg_k8, rg_k11, int, rg_v8, rg_v11);
		CMP_SWP(K, rg_k9, rg_k10, int, rg_v9, rg_v10);
		CMP_SWP(K, rg_k12, rg_k15, int, rg_v12, rg_v15);
		CMP_SWP(K, rg_k13, rg_k14, int, rg_v13, rg_v14);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
		CMP_SWP(K, rg_k2, rg_k3, int, rg_v2, rg_v3);
		CMP_SWP(K, rg_k4, rg_k5, int, rg_v4, rg_v5);
		CMP_SWP(K, rg_k6, rg_k7, int, rg_v6, rg_v7);
		CMP_SWP(K, rg_k8, rg_k9, int, rg_v8, rg_v9);
		CMP_SWP(K, rg_k10, rg_k11, int, rg_v10, rg_v11);
		CMP_SWP(K, rg_k12, rg_k13, int, rg_v12, rg_v13);
		CMP_SWP(K, rg_k14, rg_k15, int, rg_v14, rg_v15);
		// exch_intxn: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k7, int, rg_v0, rg_v7);
		CMP_SWP(K, rg_k1, rg_k6, int, rg_v1, rg_v6);
		CMP_SWP(K, rg_k2, rg_k5, int, rg_v2, rg_v5);
		CMP_SWP(K, rg_k3, rg_k4, int, rg_v3, rg_v4);
		CMP_SWP(K, rg_k8, rg_k15, int, rg_v8, rg_v15);
		CMP_SWP(K, rg_k9, rg_k14, int, rg_v9, rg_v14);
		CMP_SWP(K, rg_k10, rg_k13, int, rg_v10, rg_v13);
		CMP_SWP(K, rg_k11, rg_k12, int, rg_v11, rg_v12);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k2, int, rg_v0, rg_v2);
		CMP_SWP(K, rg_k1, rg_k3, int, rg_v1, rg_v3);
		CMP_SWP(K, rg_k4, rg_k6, int, rg_v4, rg_v6);
		CMP_SWP(K, rg_k5, rg_k7, int, rg_v5, rg_v7);
		CMP_SWP(K, rg_k8, rg_k10, int, rg_v8, rg_v10);
		CMP_SWP(K, rg_k9, rg_k11, int, rg_v9, rg_v11);
		CMP_SWP(K, rg_k12, rg_k14, int, rg_v12, rg_v14);
		CMP_SWP(K, rg_k13, rg_k15, int, rg_v13, rg_v15);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
		CMP_SWP(K, rg_k2, rg_k3, int, rg_v2, rg_v3);
		CMP_SWP(K, rg_k4, rg_k5, int, rg_v4, rg_v5);
		CMP_SWP(K, rg_k6, rg_k7, int, rg_v6, rg_v7);
		CMP_SWP(K, rg_k8, rg_k9, int, rg_v8, rg_v9);
		CMP_SWP(K, rg_k10, rg_k11, int, rg_v10, rg_v11);
		CMP_SWP(K, rg_k12, rg_k13, int, rg_v12, rg_v13);
		CMP_SWP(K, rg_k14, rg_k15, int, rg_v14, rg_v15);
		// exch_intxn: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k15, int, rg_v0, rg_v15);
		CMP_SWP(K, rg_k1, rg_k14, int, rg_v1, rg_v14);
		CMP_SWP(K, rg_k2, rg_k13, int, rg_v2, rg_v13);
		CMP_SWP(K, rg_k3, rg_k12, int, rg_v3, rg_v12);
		CMP_SWP(K, rg_k4, rg_k11, int, rg_v4, rg_v11);
		CMP_SWP(K, rg_k5, rg_k10, int, rg_v5, rg_v10);
		CMP_SWP(K, rg_k6, rg_k9, int, rg_v6, rg_v9);
		CMP_SWP(K, rg_k7, rg_k8, int, rg_v7, rg_v8);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k4, int, rg_v0, rg_v4);
		CMP_SWP(K, rg_k1, rg_k5, int, rg_v1, rg_v5);
		CMP_SWP(K, rg_k2, rg_k6, int, rg_v2, rg_v6);
		CMP_SWP(K, rg_k3, rg_k7, int, rg_v3, rg_v7);
		CMP_SWP(K, rg_k8, rg_k12, int, rg_v8, rg_v12);
		CMP_SWP(K, rg_k9, rg_k13, int, rg_v9, rg_v13);
		CMP_SWP(K, rg_k10, rg_k14, int, rg_v10, rg_v14);
		CMP_SWP(K, rg_k11, rg_k15, int, rg_v11, rg_v15);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k2, int, rg_v0, rg_v2);
		CMP_SWP(K, rg_k1, rg_k3, int, rg_v1, rg_v3);
		CMP_SWP(K, rg_k4, rg_k6, int, rg_v4, rg_v6);
		CMP_SWP(K, rg_k5, rg_k7, int, rg_v5, rg_v7);
		CMP_SWP(K, rg_k8, rg_k10, int, rg_v8, rg_v10);
		CMP_SWP(K, rg_k9, rg_k11, int, rg_v9, rg_v11);
		CMP_SWP(K, rg_k12, rg_k14, int, rg_v12, rg_v14);
		CMP_SWP(K, rg_k13, rg_k15, int, rg_v13, rg_v15);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
		CMP_SWP(K, rg_k2, rg_k3, int, rg_v2, rg_v3);
		CMP_SWP(K, rg_k4, rg_k5, int, rg_v4, rg_v5);
		CMP_SWP(K, rg_k6, rg_k7, int, rg_v6, rg_v7);
		CMP_SWP(K, rg_k8, rg_k9, int, rg_v8, rg_v9);
		CMP_SWP(K, rg_k10, rg_k11, int, rg_v10, rg_v11);
		CMP_SWP(K, rg_k12, rg_k13, int, rg_v12, rg_v13);
		CMP_SWP(K, rg_k14, rg_k15, int, rg_v14, rg_v15);
		// exch_intxn: generate exch_intxn()
		exch_intxn(rg_k0, rg_k1, rg_k2, rg_k3, rg_k4, rg_k5, rg_k6, rg_k7, rg_k8, rg_k9, rg_k10, rg_k11, rg_k12, rg_k13, rg_k14, rg_k15, rg_v0, rg_v1, rg_v2, rg_v3, rg_v4, rg_v5, rg_v6, rg_v7, rg_v8, rg_v9, rg_v10, rg_v11, rg_v12, rg_v13, rg_v14, rg_v15, 0x1, bit1);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k8, int, rg_v0, rg_v8);
		CMP_SWP(K, rg_k1, rg_k9, int, rg_v1, rg_v9);
		CMP_SWP(K, rg_k2, rg_k10, int, rg_v2, rg_v10);
		CMP_SWP(K, rg_k3, rg_k11, int, rg_v3, rg_v11);
		CMP_SWP(K, rg_k4, rg_k12, int, rg_v4, rg_v12);
		CMP_SWP(K, rg_k5, rg_k13, int, rg_v5, rg_v13);
		CMP_SWP(K, rg_k6, rg_k14, int, rg_v6, rg_v14);
		CMP_SWP(K, rg_k7, rg_k15, int, rg_v7, rg_v15);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k4, int, rg_v0, rg_v4);
		CMP_SWP(K, rg_k1, rg_k5, int, rg_v1, rg_v5);
		CMP_SWP(K, rg_k2, rg_k6, int, rg_v2, rg_v6);
		CMP_SWP(K, rg_k3, rg_k7, int, rg_v3, rg_v7);
		CMP_SWP(K, rg_k8, rg_k12, int, rg_v8, rg_v12);
		CMP_SWP(K, rg_k9, rg_k13, int, rg_v9, rg_v13);
		CMP_SWP(K, rg_k10, rg_k14, int, rg_v10, rg_v14);
		CMP_SWP(K, rg_k11, rg_k15, int, rg_v11, rg_v15);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k2, int, rg_v0, rg_v2);
		CMP_SWP(K, rg_k1, rg_k3, int, rg_v1, rg_v3);
		CMP_SWP(K, rg_k4, rg_k6, int, rg_v4, rg_v6);
		CMP_SWP(K, rg_k5, rg_k7, int, rg_v5, rg_v7);
		CMP_SWP(K, rg_k8, rg_k10, int, rg_v8, rg_v10);
		CMP_SWP(K, rg_k9, rg_k11, int, rg_v9, rg_v11);
		CMP_SWP(K, rg_k12, rg_k14, int, rg_v12, rg_v14);
		CMP_SWP(K, rg_k13, rg_k15, int, rg_v13, rg_v15);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
		CMP_SWP(K, rg_k2, rg_k3, int, rg_v2, rg_v3);
		CMP_SWP(K, rg_k4, rg_k5, int, rg_v4, rg_v5);
		CMP_SWP(K, rg_k6, rg_k7, int, rg_v6, rg_v7);
		CMP_SWP(K, rg_k8, rg_k9, int, rg_v8, rg_v9);
		CMP_SWP(K, rg_k10, rg_k11, int, rg_v10, rg_v11);
		CMP_SWP(K, rg_k12, rg_k13, int, rg_v12, rg_v13);
		CMP_SWP(K, rg_k14, rg_k15, int, rg_v14, rg_v15);
		// exch_intxn: generate exch_intxn()
		exch_intxn(rg_k0, rg_k1, rg_k2, rg_k3, rg_k4, rg_k5, rg_k6, rg_k7, rg_k8, rg_k9, rg_k10, rg_k11, rg_k12, rg_k13, rg_k14, rg_k15, rg_v0, rg_v1, rg_v2, rg_v3, rg_v4, rg_v5, rg_v6, rg_v7, rg_v8, rg_v9, rg_v10, rg_v11, rg_v12, rg_v13, rg_v14, rg_v15, 0x3, bit2);
		// exch_paral: generate exch_paral()
		exch_paral(rg_k0, rg_k1, rg_k2, rg_k3, rg_k4, rg_k5, rg_k6, rg_k7, rg_k8, rg_k9, rg_k10, rg_k11, rg_k12, rg_k13, rg_k14, rg_k15, rg_v0, rg_v1, rg_v2, rg_v3, rg_v4, rg_v5, rg_v6, rg_v7, rg_v8, rg_v9, rg_v10, rg_v11, rg_v12, rg_v13, rg_v14, rg_v15, 0x1, bit1);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k8, int, rg_v0, rg_v8);
		CMP_SWP(K, rg_k1, rg_k9, int, rg_v1, rg_v9);
		CMP_SWP(K, rg_k2, rg_k10, int, rg_v2, rg_v10);
		CMP_SWP(K, rg_k3, rg_k11, int, rg_v3, rg_v11);
		CMP_SWP(K, rg_k4, rg_k12, int, rg_v4, rg_v12);
		CMP_SWP(K, rg_k5, rg_k13, int, rg_v5, rg_v13);
		CMP_SWP(K, rg_k6, rg_k14, int, rg_v6, rg_v14);
		CMP_SWP(K, rg_k7, rg_k15, int, rg_v7, rg_v15);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k4, int, rg_v0, rg_v4);
		CMP_SWP(K, rg_k1, rg_k5, int, rg_v1, rg_v5);
		CMP_SWP(K, rg_k2, rg_k6, int, rg_v2, rg_v6);
		CMP_SWP(K, rg_k3, rg_k7, int, rg_v3, rg_v7);
		CMP_SWP(K, rg_k8, rg_k12, int, rg_v8, rg_v12);
		CMP_SWP(K, rg_k9, rg_k13, int, rg_v9, rg_v13);
		CMP_SWP(K, rg_k10, rg_k14, int, rg_v10, rg_v14);
		CMP_SWP(K, rg_k11, rg_k15, int, rg_v11, rg_v15);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k2, int, rg_v0, rg_v2);
		CMP_SWP(K, rg_k1, rg_k3, int, rg_v1, rg_v3);
		CMP_SWP(K, rg_k4, rg_k6, int, rg_v4, rg_v6);
		CMP_SWP(K, rg_k5, rg_k7, int, rg_v5, rg_v7);
		CMP_SWP(K, rg_k8, rg_k10, int, rg_v8, rg_v10);
		CMP_SWP(K, rg_k9, rg_k11, int, rg_v9, rg_v11);
		CMP_SWP(K, rg_k12, rg_k14, int, rg_v12, rg_v14);
		CMP_SWP(K, rg_k13, rg_k15, int, rg_v13, rg_v15);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
		CMP_SWP(K, rg_k2, rg_k3, int, rg_v2, rg_v3);
		CMP_SWP(K, rg_k4, rg_k5, int, rg_v4, rg_v5);
		CMP_SWP(K, rg_k6, rg_k7, int, rg_v6, rg_v7);
		CMP_SWP(K, rg_k8, rg_k9, int, rg_v8, rg_v9);
		CMP_SWP(K, rg_k10, rg_k11, int, rg_v10, rg_v11);
		CMP_SWP(K, rg_k12, rg_k13, int, rg_v12, rg_v13);
		CMP_SWP(K, rg_k14, rg_k15, int, rg_v14, rg_v15);
		// exch_intxn: generate exch_intxn()
		exch_intxn(rg_k0, rg_k1, rg_k2, rg_k3, rg_k4, rg_k5, rg_k6, rg_k7, rg_k8, rg_k9, rg_k10, rg_k11, rg_k12, rg_k13, rg_k14, rg_k15, rg_v0, rg_v1, rg_v2, rg_v3, rg_v4, rg_v5, rg_v6, rg_v7, rg_v8, rg_v9, rg_v10, rg_v11, rg_v12, rg_v13, rg_v14, rg_v15, 0x7, bit3);
		// exch_paral: generate exch_paral()
		exch_paral(rg_k0, rg_k1, rg_k2, rg_k3, rg_k4, rg_k5, rg_k6, rg_k7, rg_k8, rg_k9, rg_k10, rg_k11, rg_k12, rg_k13, rg_k14, rg_k15, rg_v0, rg_v1, rg_v2, rg_v3, rg_v4, rg_v5, rg_v6, rg_v7, rg_v8, rg_v9, rg_v10, rg_v11, rg_v12, rg_v13, rg_v14, rg_v15, 0x2, bit2);
		// exch_paral: generate exch_paral()
		exch_paral(rg_k0, rg_k1, rg_k2, rg_k3, rg_k4, rg_k5, rg_k6, rg_k7, rg_k8, rg_k9, rg_k10, rg_k11, rg_k12, rg_k13, rg_k14, rg_k15, rg_v0, rg_v1, rg_v2, rg_v3, rg_v4, rg_v5, rg_v6, rg_v7, rg_v8, rg_v9, rg_v10, rg_v11, rg_v12, rg_v13, rg_v14, rg_v15, 0x1, bit1);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k8, int, rg_v0, rg_v8);
		CMP_SWP(K, rg_k1, rg_k9, int, rg_v1, rg_v9);
		CMP_SWP(K, rg_k2, rg_k10, int, rg_v2, rg_v10);
		CMP_SWP(K, rg_k3, rg_k11, int, rg_v3, rg_v11);
		CMP_SWP(K, rg_k4, rg_k12, int, rg_v4, rg_v12);
		CMP_SWP(K, rg_k5, rg_k13, int, rg_v5, rg_v13);
		CMP_SWP(K, rg_k6, rg_k14, int, rg_v6, rg_v14);
		CMP_SWP(K, rg_k7, rg_k15, int, rg_v7, rg_v15);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k4, int, rg_v0, rg_v4);
		CMP_SWP(K, rg_k1, rg_k5, int, rg_v1, rg_v5);
		CMP_SWP(K, rg_k2, rg_k6, int, rg_v2, rg_v6);
		CMP_SWP(K, rg_k3, rg_k7, int, rg_v3, rg_v7);
		CMP_SWP(K, rg_k8, rg_k12, int, rg_v8, rg_v12);
		CMP_SWP(K, rg_k9, rg_k13, int, rg_v9, rg_v13);
		CMP_SWP(K, rg_k10, rg_k14, int, rg_v10, rg_v14);
		CMP_SWP(K, rg_k11, rg_k15, int, rg_v11, rg_v15);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k2, int, rg_v0, rg_v2);
		CMP_SWP(K, rg_k1, rg_k3, int, rg_v1, rg_v3);
		CMP_SWP(K, rg_k4, rg_k6, int, rg_v4, rg_v6);
		CMP_SWP(K, rg_k5, rg_k7, int, rg_v5, rg_v7);
		CMP_SWP(K, rg_k8, rg_k10, int, rg_v8, rg_v10);
		CMP_SWP(K, rg_k9, rg_k11, int, rg_v9, rg_v11);
		CMP_SWP(K, rg_k12, rg_k14, int, rg_v12, rg_v14);
		CMP_SWP(K, rg_k13, rg_k15, int, rg_v13, rg_v15);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
		CMP_SWP(K, rg_k2, rg_k3, int, rg_v2, rg_v3);
		CMP_SWP(K, rg_k4, rg_k5, int, rg_v4, rg_v5);
		CMP_SWP(K, rg_k6, rg_k7, int, rg_v6, rg_v7);
		CMP_SWP(K, rg_k8, rg_k9, int, rg_v8, rg_v9);
		CMP_SWP(K, rg_k10, rg_k11, int, rg_v10, rg_v11);
		CMP_SWP(K, rg_k12, rg_k13, int, rg_v12, rg_v13);
		CMP_SWP(K, rg_k14, rg_k15, int, rg_v14, rg_v15);
	}
	if (bin_it < normalized_bin_size) {
		// transpose the in register data of the whole warp
		int lane_id = threadIdx.x & 31;
		rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x1);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x1);
		rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x1);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x1);
		rg_k9  = __shfl_xor_sync(0xffffffff,rg_k9 , 0x1);
		rg_k11  = __shfl_xor_sync(0xffffffff,rg_k11 , 0x1);
		rg_k13  = __shfl_xor_sync(0xffffffff,rg_k13 , 0x1);
		rg_k15  = __shfl_xor_sync(0xffffffff,rg_k15 , 0x1);
		rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x1);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x1);
		rg_v5  = __shfl_xor_sync(0xffffffff,rg_v5 , 0x1);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x1);
		rg_v9  = __shfl_xor_sync(0xffffffff,rg_v9 , 0x1);
		rg_v11  = __shfl_xor_sync(0xffffffff,rg_v11 , 0x1);
		rg_v13  = __shfl_xor_sync(0xffffffff,rg_v13 , 0x1);
		rg_v15  = __shfl_xor_sync(0xffffffff,rg_v15 , 0x1);
		if(lane_id&0x1) SWP(K, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
		if(lane_id&0x1) SWP(K, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
		if(lane_id&0x1) SWP(K, rg_k4 , rg_k5 , int, rg_v4 , rg_v5 );
		if(lane_id&0x1) SWP(K, rg_k6 , rg_k7 , int, rg_v6 , rg_v7 );
		if(lane_id&0x1) SWP(K, rg_k8 , rg_k9 , int, rg_v8 , rg_v9 );
		if(lane_id&0x1) SWP(K, rg_k10 , rg_k11 , int, rg_v10 , rg_v11 );
		if(lane_id&0x1) SWP(K, rg_k12 , rg_k13 , int, rg_v12 , rg_v13 );
		if(lane_id&0x1) SWP(K, rg_k14 , rg_k15 , int, rg_v14 , rg_v15 );
		rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x1);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x1);
		rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x1);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x1);
		rg_k9  = __shfl_xor_sync(0xffffffff,rg_k9 , 0x1);
		rg_k11  = __shfl_xor_sync(0xffffffff,rg_k11 , 0x1);
		rg_k13  = __shfl_xor_sync(0xffffffff,rg_k13 , 0x1);
		rg_k15  = __shfl_xor_sync(0xffffffff,rg_k15 , 0x1);
		rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x1);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x1);
		rg_v5  = __shfl_xor_sync(0xffffffff,rg_v5 , 0x1);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x1);
		rg_v9  = __shfl_xor_sync(0xffffffff,rg_v9 , 0x1);
		rg_v11  = __shfl_xor_sync(0xffffffff,rg_v11 , 0x1);
		rg_v13  = __shfl_xor_sync(0xffffffff,rg_v13 , 0x1);
		rg_v15  = __shfl_xor_sync(0xffffffff,rg_v15 , 0x1);
		rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x2);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x2);
		rg_k6  = __shfl_xor_sync(0xffffffff,rg_k6 , 0x2);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x2);
		rg_k10  = __shfl_xor_sync(0xffffffff,rg_k10 , 0x2);
		rg_k11  = __shfl_xor_sync(0xffffffff,rg_k11 , 0x2);
		rg_k14  = __shfl_xor_sync(0xffffffff,rg_k14 , 0x2);
		rg_k15  = __shfl_xor_sync(0xffffffff,rg_k15 , 0x2);
		rg_v2  = __shfl_xor_sync(0xffffffff,rg_v2 , 0x2);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x2);
		rg_v6  = __shfl_xor_sync(0xffffffff,rg_v6 , 0x2);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x2);
		rg_v10  = __shfl_xor_sync(0xffffffff,rg_v10 , 0x2);
		rg_v11  = __shfl_xor_sync(0xffffffff,rg_v11 , 0x2);
		rg_v14  = __shfl_xor_sync(0xffffffff,rg_v14 , 0x2);
		rg_v15  = __shfl_xor_sync(0xffffffff,rg_v15 , 0x2);
		if(lane_id&0x2) SWP(K, rg_k0 , rg_k2 , int, rg_v0 , rg_v2 );
		if(lane_id&0x2) SWP(K, rg_k1 , rg_k3 , int, rg_v1 , rg_v3 );
		if(lane_id&0x2) SWP(K, rg_k4 , rg_k6 , int, rg_v4 , rg_v6 );
		if(lane_id&0x2) SWP(K, rg_k5 , rg_k7 , int, rg_v5 , rg_v7 );
		if(lane_id&0x2) SWP(K, rg_k8 , rg_k10 , int, rg_v8 , rg_v10 );
		if(lane_id&0x2) SWP(K, rg_k9 , rg_k11 , int, rg_v9 , rg_v11 );
		if(lane_id&0x2) SWP(K, rg_k12 , rg_k14 , int, rg_v12 , rg_v14 );
		if(lane_id&0x2) SWP(K, rg_k13 , rg_k15 , int, rg_v13 , rg_v15 );
		rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x2);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x2);
		rg_k6  = __shfl_xor_sync(0xffffffff,rg_k6 , 0x2);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x2);
		rg_k10  = __shfl_xor_sync(0xffffffff,rg_k10 , 0x2);
		rg_k11  = __shfl_xor_sync(0xffffffff,rg_k11 , 0x2);
		rg_k14  = __shfl_xor_sync(0xffffffff,rg_k14 , 0x2);
		rg_k15  = __shfl_xor_sync(0xffffffff,rg_k15 , 0x2);
		rg_v2  = __shfl_xor_sync(0xffffffff,rg_v2 , 0x2);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x2);
		rg_v6  = __shfl_xor_sync(0xffffffff,rg_v6 , 0x2);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x2);
		rg_v10  = __shfl_xor_sync(0xffffffff,rg_v10 , 0x2);
		rg_v11  = __shfl_xor_sync(0xffffffff,rg_v11 , 0x2);
		rg_v14  = __shfl_xor_sync(0xffffffff,rg_v14 , 0x2);
		rg_v15  = __shfl_xor_sync(0xffffffff,rg_v15 , 0x2);
		rg_k4  = __shfl_xor_sync(0xffffffff,rg_k4 , 0x4);
		rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x4);
		rg_k6  = __shfl_xor_sync(0xffffffff,rg_k6 , 0x4);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x4);
		rg_k12  = __shfl_xor_sync(0xffffffff,rg_k12 , 0x4);
		rg_k13  = __shfl_xor_sync(0xffffffff,rg_k13 , 0x4);
		rg_k14  = __shfl_xor_sync(0xffffffff,rg_k14 , 0x4);
		rg_k15  = __shfl_xor_sync(0xffffffff,rg_k15 , 0x4);
		rg_v4  = __shfl_xor_sync(0xffffffff,rg_v4 , 0x4);
		rg_v5  = __shfl_xor_sync(0xffffffff,rg_v5 , 0x4);
		rg_v6  = __shfl_xor_sync(0xffffffff,rg_v6 , 0x4);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x4);
		rg_v12  = __shfl_xor_sync(0xffffffff,rg_v12 , 0x4);
		rg_v13  = __shfl_xor_sync(0xffffffff,rg_v13 , 0x4);
		rg_v14  = __shfl_xor_sync(0xffffffff,rg_v14 , 0x4);
		rg_v15  = __shfl_xor_sync(0xffffffff,rg_v15 , 0x4);
		if(lane_id&0x4) SWP(K, rg_k0 , rg_k4 , int, rg_v0 , rg_v4 );
		if(lane_id&0x4) SWP(K, rg_k1 , rg_k5 , int, rg_v1 , rg_v5 );
		if(lane_id&0x4) SWP(K, rg_k2 , rg_k6 , int, rg_v2 , rg_v6 );
		if(lane_id&0x4) SWP(K, rg_k3 , rg_k7 , int, rg_v3 , rg_v7 );
		if(lane_id&0x4) SWP(K, rg_k8 , rg_k12 , int, rg_v8 , rg_v12 );
		if(lane_id&0x4) SWP(K, rg_k9 , rg_k13 , int, rg_v9 , rg_v13 );
		if(lane_id&0x4) SWP(K, rg_k10 , rg_k14 , int, rg_v10 , rg_v14 );
		if(lane_id&0x4) SWP(K, rg_k11 , rg_k15 , int, rg_v11 , rg_v15 );
		rg_k4  = __shfl_xor_sync(0xffffffff,rg_k4 , 0x4);
		rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x4);
		rg_k6  = __shfl_xor_sync(0xffffffff,rg_k6 , 0x4);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x4);
		rg_k12  = __shfl_xor_sync(0xffffffff,rg_k12 , 0x4);
		rg_k13  = __shfl_xor_sync(0xffffffff,rg_k13 , 0x4);
		rg_k14  = __shfl_xor_sync(0xffffffff,rg_k14 , 0x4);
		rg_k15  = __shfl_xor_sync(0xffffffff,rg_k15 , 0x4);
		rg_v4  = __shfl_xor_sync(0xffffffff,rg_v4 , 0x4);
		rg_v5  = __shfl_xor_sync(0xffffffff,rg_v5 , 0x4);
		rg_v6  = __shfl_xor_sync(0xffffffff,rg_v6 , 0x4);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x4);
		rg_v12  = __shfl_xor_sync(0xffffffff,rg_v12 , 0x4);
		rg_v13  = __shfl_xor_sync(0xffffffff,rg_v13 , 0x4);
		rg_v14  = __shfl_xor_sync(0xffffffff,rg_v14 , 0x4);
		rg_v15  = __shfl_xor_sync(0xffffffff,rg_v15 , 0x4);
		rg_k8  = __shfl_xor_sync(0xffffffff,rg_k8 , 0x8);
		rg_k9  = __shfl_xor_sync(0xffffffff,rg_k9 , 0x8);
		rg_k10  = __shfl_xor_sync(0xffffffff,rg_k10 , 0x8);
		rg_k11  = __shfl_xor_sync(0xffffffff,rg_k11 , 0x8);
		rg_k12  = __shfl_xor_sync(0xffffffff,rg_k12 , 0x8);
		rg_k13  = __shfl_xor_sync(0xffffffff,rg_k13 , 0x8);
		rg_k14  = __shfl_xor_sync(0xffffffff,rg_k14 , 0x8);
		rg_k15  = __shfl_xor_sync(0xffffffff,rg_k15 , 0x8);
		rg_v8  = __shfl_xor_sync(0xffffffff,rg_v8 , 0x8);
		rg_v9  = __shfl_xor_sync(0xffffffff,rg_v9 , 0x8);
		rg_v10  = __shfl_xor_sync(0xffffffff,rg_v10 , 0x8);
		rg_v11  = __shfl_xor_sync(0xffffffff,rg_v11 , 0x8);
		rg_v12  = __shfl_xor_sync(0xffffffff,rg_v12 , 0x8);
		rg_v13  = __shfl_xor_sync(0xffffffff,rg_v13 , 0x8);
		rg_v14  = __shfl_xor_sync(0xffffffff,rg_v14 , 0x8);
		rg_v15  = __shfl_xor_sync(0xffffffff,rg_v15 , 0x8);
		if(lane_id&0x8) SWP(K, rg_k0 , rg_k8 , int, rg_v0 , rg_v8 );
		if(lane_id&0x8) SWP(K, rg_k1 , rg_k9 , int, rg_v1 , rg_v9 );
		if(lane_id&0x8) SWP(K, rg_k2 , rg_k10 , int, rg_v2 , rg_v10 );
		if(lane_id&0x8) SWP(K, rg_k3 , rg_k11 , int, rg_v3 , rg_v11 );
		if(lane_id&0x8) SWP(K, rg_k4 , rg_k12 , int, rg_v4 , rg_v12 );
		if(lane_id&0x8) SWP(K, rg_k5 , rg_k13 , int, rg_v5 , rg_v13 );
		if(lane_id&0x8) SWP(K, rg_k6 , rg_k14 , int, rg_v6 , rg_v14 );
		if(lane_id&0x8) SWP(K, rg_k7 , rg_k15 , int, rg_v7 , rg_v15 );
		rg_k8  = __shfl_xor_sync(0xffffffff,rg_k8 , 0x8);
		rg_k9  = __shfl_xor_sync(0xffffffff,rg_k9 , 0x8);
		rg_k10  = __shfl_xor_sync(0xffffffff,rg_k10 , 0x8);
		rg_k11  = __shfl_xor_sync(0xffffffff,rg_k11 , 0x8);
		rg_k12  = __shfl_xor_sync(0xffffffff,rg_k12 , 0x8);
		rg_k13  = __shfl_xor_sync(0xffffffff,rg_k13 , 0x8);
		rg_k14  = __shfl_xor_sync(0xffffffff,rg_k14 , 0x8);
		rg_k15  = __shfl_xor_sync(0xffffffff,rg_k15 , 0x8);
		rg_v8  = __shfl_xor_sync(0xffffffff,rg_v8 , 0x8);
		rg_v9  = __shfl_xor_sync(0xffffffff,rg_v9 , 0x8);
		rg_v10  = __shfl_xor_sync(0xffffffff,rg_v10 , 0x8);
		rg_v11  = __shfl_xor_sync(0xffffffff,rg_v11 , 0x8);
		rg_v12  = __shfl_xor_sync(0xffffffff,rg_v12 , 0x8);
		rg_v13  = __shfl_xor_sync(0xffffffff,rg_v13 , 0x8);
		rg_v14  = __shfl_xor_sync(0xffffffff,rg_v14 , 0x8);
		rg_v15  = __shfl_xor_sync(0xffffffff,rg_v15 , 0x8);
		rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x10);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x10);
		rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x10);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x10);
		rg_k9  = __shfl_xor_sync(0xffffffff,rg_k9 , 0x10);
		rg_k11  = __shfl_xor_sync(0xffffffff,rg_k11 , 0x10);
		rg_k13  = __shfl_xor_sync(0xffffffff,rg_k13 , 0x10);
		rg_k15  = __shfl_xor_sync(0xffffffff,rg_k15 , 0x10);
		rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x10);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x10);
		rg_v5  = __shfl_xor_sync(0xffffffff,rg_v5 , 0x10);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x10);
		rg_v9  = __shfl_xor_sync(0xffffffff,rg_v9 , 0x10);
		rg_v11  = __shfl_xor_sync(0xffffffff,rg_v11 , 0x10);
		rg_v13  = __shfl_xor_sync(0xffffffff,rg_v13 , 0x10);
		rg_v15  = __shfl_xor_sync(0xffffffff,rg_v15 , 0x10);
		if(lane_id&0x10) SWP(K, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
		if(lane_id&0x10) SWP(K, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
		if(lane_id&0x10) SWP(K, rg_k4 , rg_k5 , int, rg_v4 , rg_v5 );
		if(lane_id&0x10) SWP(K, rg_k6 , rg_k7 , int, rg_v6 , rg_v7 );
		if(lane_id&0x10) SWP(K, rg_k8 , rg_k9 , int, rg_v8 , rg_v9 );
		if(lane_id&0x10) SWP(K, rg_k10 , rg_k11 , int, rg_v10 , rg_v11 );
		if(lane_id&0x10) SWP(K, rg_k12 , rg_k13 , int, rg_v12 , rg_v13 );
		if(lane_id&0x10) SWP(K, rg_k14 , rg_k15 , int, rg_v14 , rg_v15 );
		rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x10);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x10);
		rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x10);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x10);
		rg_k9  = __shfl_xor_sync(0xffffffff,rg_k9 , 0x10);
		rg_k11  = __shfl_xor_sync(0xffffffff,rg_k11 , 0x10);
		rg_k13  = __shfl_xor_sync(0xffffffff,rg_k13 , 0x10);
		rg_k15  = __shfl_xor_sync(0xffffffff,rg_k15 , 0x10);
		rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x10);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x10);
		rg_v5  = __shfl_xor_sync(0xffffffff,rg_v5 , 0x10);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x10);
		rg_v9  = __shfl_xor_sync(0xffffffff,rg_v9 , 0x10);
		rg_v11  = __shfl_xor_sync(0xffffffff,rg_v11 , 0x10);
		rg_v13  = __shfl_xor_sync(0xffffffff,rg_v13 , 0x10);
		rg_v15  = __shfl_xor_sync(0xffffffff,rg_v15 , 0x10);
		int kk;
		int ss;
		kk = __shfl_sync(0xffffffff, k, 0);
		ss = __shfl_sync(0xffffffff, seg_size, 0);
		if (lane_id + 0 < ss) keyB[kk+lane_id+0] = rg_k0;
		if (lane_id + 32 < ss) keyB[kk+lane_id+32] = rg_k2;
		if (lane_id + 64 < ss) keyB[kk+lane_id+64] = rg_k4;
		if (lane_id + 96 < ss) keyB[kk+lane_id+96] = rg_k6;
		kk = __shfl_sync(0xffffffff, k, 8);
		ss = __shfl_sync(0xffffffff, seg_size, 8);
		if (lane_id + 0 < ss) keyB[kk+lane_id+0] = rg_k8;
		if (lane_id + 32 < ss) keyB[kk+lane_id+32] = rg_k10;
		if (lane_id + 64 < ss) keyB[kk+lane_id+64] = rg_k12;
		if (lane_id + 96 < ss) keyB[kk+lane_id+96] = rg_k14;
		kk = __shfl_sync(0xffffffff, k, 16);
		ss = __shfl_sync(0xffffffff, seg_size, 16);
		if (lane_id + 0 < ss) keyB[kk+lane_id+0] = rg_k1;
		if (lane_id + 32 < ss) keyB[kk+lane_id+32] = rg_k3;
		if (lane_id + 64 < ss) keyB[kk+lane_id+64] = rg_k5;
		if (lane_id + 96 < ss) keyB[kk+lane_id+96] = rg_k7;
		kk = __shfl_sync(0xffffffff, k, 24);
		ss = __shfl_sync(0xffffffff, seg_size, 24);
		if (lane_id + 0 < ss) keyB[kk+lane_id+0] = rg_k9;
		if (lane_id + 32 < ss) keyB[kk+lane_id+32] = rg_k11;
		if (lane_id + 64 < ss) keyB[kk+lane_id+64] = rg_k13;
		if (lane_id + 96 < ss) keyB[kk+lane_id+96] = rg_k15;
		kk = __shfl_sync(0xffffffff, k, 0);
		ss = __shfl_sync(0xffffffff, seg_size, 0);
		if(lane_id+0 < ss) valB[kk+lane_id+0] = val[kk+rg_v0];
		if(lane_id+32 < ss) valB[kk+lane_id+32] = val[kk+rg_v2];
		if(lane_id+64 < ss) valB[kk+lane_id+64] = val[kk+rg_v4];
		if(lane_id+96 < ss) valB[kk+lane_id+96] = val[kk+rg_v6];
		kk = __shfl_sync(0xffffffff, k, 8);
		ss = __shfl_sync(0xffffffff, seg_size, 8);
		if(lane_id+0 < ss) valB[kk+lane_id+0] = val[kk+rg_v8];
		if(lane_id+32 < ss) valB[kk+lane_id+32] = val[kk+rg_v10];
		if(lane_id+64 < ss) valB[kk+lane_id+64] = val[kk+rg_v12];
		if(lane_id+96 < ss) valB[kk+lane_id+96] = val[kk+rg_v14];
		kk = __shfl_sync(0xffffffff, k, 16);
		ss = __shfl_sync(0xffffffff, seg_size, 16);
		if(lane_id+0 < ss) valB[kk+lane_id+0] = val[kk+rg_v1];
		if(lane_id+32 < ss) valB[kk+lane_id+32] = val[kk+rg_v3];
		if(lane_id+64 < ss) valB[kk+lane_id+64] = val[kk+rg_v5];
		if(lane_id+96 < ss) valB[kk+lane_id+96] = val[kk+rg_v7];
		kk = __shfl_sync(0xffffffff, k, 24);
		ss = __shfl_sync(0xffffffff, seg_size, 24);
		if(lane_id+0 < ss) valB[kk+lane_id+0] = val[kk+rg_v9];
		if(lane_id+32 < ss) valB[kk+lane_id+32] = val[kk+rg_v11];
		if(lane_id+64 < ss) valB[kk+lane_id+64] = val[kk+rg_v13];
		if(lane_id+96 < ss) valB[kk+lane_id+96] = val[kk+rg_v15];
	}
	else if(bin_it < bin_size) {
		if((tid<<4)+0 <seg_size) keyB[k+(tid<<4)+0] = rg_k0;
		if((tid<<4)+1 <seg_size) keyB[k+(tid<<4)+1] = rg_k1;
		if((tid<<4)+2 <seg_size) keyB[k+(tid<<4)+2] = rg_k2;
		if((tid<<4)+3 <seg_size) keyB[k+(tid<<4)+3] = rg_k3;
		if((tid<<4)+4 <seg_size) keyB[k+(tid<<4)+4] = rg_k4;
		if((tid<<4)+5 <seg_size) keyB[k+(tid<<4)+5] = rg_k5;
		if((tid<<4)+6 <seg_size) keyB[k+(tid<<4)+6] = rg_k6;
		if((tid<<4)+7 <seg_size) keyB[k+(tid<<4)+7] = rg_k7;
		if((tid<<4)+8 <seg_size) keyB[k+(tid<<4)+8] = rg_k8;
		if((tid<<4)+9 <seg_size) keyB[k+(tid<<4)+9] = rg_k9;
		if((tid<<4)+10 <seg_size) keyB[k+(tid<<4)+10] = rg_k10;
		if((tid<<4)+11 <seg_size) keyB[k+(tid<<4)+11] = rg_k11;
		if((tid<<4)+12 <seg_size) keyB[k+(tid<<4)+12] = rg_k12;
		if((tid<<4)+13 <seg_size) keyB[k+(tid<<4)+13] = rg_k13;
		if((tid<<4)+14 <seg_size) keyB[k+(tid<<4)+14] = rg_k14;
		if((tid<<4)+15 <seg_size) keyB[k+(tid<<4)+15] = rg_k15;

		if((tid<<4)+0 <seg_size) valB[k+(tid<<4)+0] = val[k+rg_v0];
		if((tid<<4)+1 <seg_size) valB[k+(tid<<4)+1] = val[k+rg_v1];
		if((tid<<4)+2 <seg_size) valB[k+(tid<<4)+2] = val[k+rg_v2];
		if((tid<<4)+3 <seg_size) valB[k+(tid<<4)+3] = val[k+rg_v3];
		if((tid<<4)+4 <seg_size) valB[k+(tid<<4)+4] = val[k+rg_v4];
		if((tid<<4)+5 <seg_size) valB[k+(tid<<4)+5] = val[k+rg_v5];
		if((tid<<4)+6 <seg_size) valB[k+(tid<<4)+6] = val[k+rg_v6];
		if((tid<<4)+7 <seg_size) valB[k+(tid<<4)+7] = val[k+rg_v7];
		if((tid<<4)+8 <seg_size) valB[k+(tid<<4)+8] = val[k+rg_v8];
		if((tid<<4)+9 <seg_size) valB[k+(tid<<4)+9] = val[k+rg_v9];
		if((tid<<4)+10 <seg_size) valB[k+(tid<<4)+10] = val[k+rg_v10];
		if((tid<<4)+11 <seg_size) valB[k+(tid<<4)+11] = val[k+rg_v11];
		if((tid<<4)+12 <seg_size) valB[k+(tid<<4)+12] = val[k+rg_v12];
		if((tid<<4)+13 <seg_size) valB[k+(tid<<4)+13] = val[k+rg_v13];
		if((tid<<4)+14 <seg_size) valB[k+(tid<<4)+14] = val[k+rg_v14];
		if((tid<<4)+15 <seg_size) valB[k+(tid<<4)+15] = val[k+rg_v15];
	}
}

// 16 threads | 4 ppt | *strided*
template<class K, class T>
__global__
void my_t16_ppt4_strd(K *key, T *val, K *keyB, T *valB, int n, int *segs, int *bin, int bin_size, int length) {
	const int gid = threadIdx.x + blockIdx.x * blockDim.x;
	const int bin_it = (gid>>4);
	const int tid = (threadIdx.x & 15);
	const int bit1 = (tid>>0)&0x1;
	const int bit2 = (tid>>1)&0x1;
	const int bit3 = (tid>>2)&0x1;
	const int bit4 = (tid>>3)&0x1;
	K rg_k0;
	K rg_k1;
	K rg_k2;
	K rg_k3;
	int rg_v0;
	int rg_v1;
	int rg_v2;
	int rg_v3;
	int normalized_bin_size = (bin_size/2)*2;
	int k;
	int seg_size;

	if(bin_it < bin_size) {
		k = segs[bin[bin_it]];
		seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
		rg_k0  = (tid+0	<seg_size)?key[k+tid+0	]:std::numeric_limits<K>::max();
		rg_k1  = (tid+16	<seg_size)?key[k+tid+16	]:std::numeric_limits<K>::max();
		rg_k2  = (tid+32	<seg_size)?key[k+tid+32	]:std::numeric_limits<K>::max();
		rg_k3  = (tid+48	<seg_size)?key[k+tid+48	]:std::numeric_limits<K>::max();
		if(tid+0	<seg_size) rg_v0 = tid+0	;
		if(tid+16	<seg_size) rg_v1 = tid+16	;
		if(tid+32	<seg_size) rg_v2 = tid+32	;
		if(tid+48	<seg_size) rg_v3 = tid+48	;

		// sort 64 elements
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
	}
	if (bin_it < normalized_bin_size) {
		// transpose the in register data of the whole warp
		int lane_id = threadIdx.x & 31;
		rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x1);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x1);
		rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x1);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x1);
		if(lane_id&0x1) SWP(K, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
		if(lane_id&0x1) SWP(K, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
		rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x1);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x1);
		rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x1);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x1);
		rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x2);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x2);
		rg_v2  = __shfl_xor_sync(0xffffffff,rg_v2 , 0x2);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x2);
		if(lane_id&0x2) SWP(K, rg_k0 , rg_k2 , int, rg_v0 , rg_v2 );
		if(lane_id&0x2) SWP(K, rg_k1 , rg_k3 , int, rg_v1 , rg_v3 );
		rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x2);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x2);
		rg_v2  = __shfl_xor_sync(0xffffffff,rg_v2 , 0x2);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x2);
		rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x4);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x4);
		rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x4);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x4);
		if(lane_id&0x4) SWP(K, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
		if(lane_id&0x4) SWP(K, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
		rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x4);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x4);
		rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x4);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x4);
		rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x8);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x8);
		rg_v2  = __shfl_xor_sync(0xffffffff,rg_v2 , 0x8);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x8);
		if(lane_id&0x8) SWP(K, rg_k0 , rg_k2 , int, rg_v0 , rg_v2 );
		if(lane_id&0x8) SWP(K, rg_k1 , rg_k3 , int, rg_v1 , rg_v3 );
		rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x8);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x8);
		rg_v2  = __shfl_xor_sync(0xffffffff,rg_v2 , 0x8);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x8);
		rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x10);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x10);
		rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x10);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x10);
		if(lane_id&0x10) SWP(K, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
		if(lane_id&0x10) SWP(K, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
		rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x10);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x10);
		rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x10);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x10);
		int kk;
		int ss;
		kk = __shfl_sync(0xffffffff, k, 0);
		ss = __shfl_sync(0xffffffff, seg_size, 0);
		if (lane_id + 0 < ss) keyB[kk+lane_id+0] = rg_k0;
		if (lane_id + 32 < ss) keyB[kk+lane_id+32] = rg_k2;
		kk = __shfl_sync(0xffffffff, k, 16);
		ss = __shfl_sync(0xffffffff, seg_size, 16);
		if (lane_id + 0 < ss) keyB[kk+lane_id+0] = rg_k1;
		if (lane_id + 32 < ss) keyB[kk+lane_id+32] = rg_k3;
		kk = __shfl_sync(0xffffffff, k, 0);
		ss = __shfl_sync(0xffffffff, seg_size, 0);
		if(lane_id+0 < ss) valB[kk+lane_id+0] = val[kk+rg_v0];
		if(lane_id+32 < ss) valB[kk+lane_id+32] = val[kk+rg_v2];
		kk = __shfl_sync(0xffffffff, k, 16);
		ss = __shfl_sync(0xffffffff, seg_size, 16);
		if(lane_id+0 < ss) valB[kk+lane_id+0] = val[kk+rg_v1];
		if(lane_id+32 < ss) valB[kk+lane_id+32] = val[kk+rg_v3];
	}
	else if(bin_it < bin_size) {
		if((tid<<2)+0 <seg_size) keyB[k+(tid<<2)+0] = rg_k0;
		if((tid<<2)+1 <seg_size) keyB[k+(tid<<2)+1] = rg_k1;
		if((tid<<2)+2 <seg_size) keyB[k+(tid<<2)+2] = rg_k2;
		if((tid<<2)+3 <seg_size) keyB[k+(tid<<2)+3] = rg_k3;

		if((tid<<2)+0 <seg_size) valB[k+(tid<<2)+0] = val[k+rg_v0];
		if((tid<<2)+1 <seg_size) valB[k+(tid<<2)+1] = val[k+rg_v1];
		if((tid<<2)+2 <seg_size) valB[k+(tid<<2)+2] = val[k+rg_v2];
		if((tid<<2)+3 <seg_size) valB[k+(tid<<2)+3] = val[k+rg_v3];
	}
}

// 16 threads | 8 ppt | *strided*
template<class K, class T>
__global__
void my_t16_ppt8_strd(K *key, T *val, K *keyB, T *valB, int n, int *segs, int *bin, int bin_size, int length) {
	const int gid = threadIdx.x + blockIdx.x * blockDim.x;
	const int bin_it = (gid>>4);
	const int tid = (threadIdx.x & 15);
	const int bit1 = (tid>>0)&0x1;
	const int bit2 = (tid>>1)&0x1;
	const int bit3 = (tid>>2)&0x1;
	const int bit4 = (tid>>3)&0x1;
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
	int normalized_bin_size = (bin_size/2)*2;
	int k;
	int seg_size;

	if(bin_it < bin_size) {
		k = segs[bin[bin_it]];
		seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
		rg_k0  = (tid+0	<seg_size)?key[k+tid+0	]:std::numeric_limits<K>::max();
		rg_k1  = (tid+16	<seg_size)?key[k+tid+16	]:std::numeric_limits<K>::max();
		rg_k2  = (tid+32	<seg_size)?key[k+tid+32	]:std::numeric_limits<K>::max();
		rg_k3  = (tid+48	<seg_size)?key[k+tid+48	]:std::numeric_limits<K>::max();
		rg_k4  = (tid+64	<seg_size)?key[k+tid+64	]:std::numeric_limits<K>::max();
		rg_k5  = (tid+80	<seg_size)?key[k+tid+80	]:std::numeric_limits<K>::max();
		rg_k6  = (tid+96	<seg_size)?key[k+tid+96	]:std::numeric_limits<K>::max();
		rg_k7  = (tid+112	<seg_size)?key[k+tid+112	]:std::numeric_limits<K>::max();
		if(tid+0	<seg_size) rg_v0 = tid+0	;
		if(tid+16	<seg_size) rg_v1 = tid+16	;
		if(tid+32	<seg_size) rg_v2 = tid+32	;
		if(tid+48	<seg_size) rg_v3 = tid+48	;
		if(tid+64	<seg_size) rg_v4 = tid+64	;
		if(tid+80	<seg_size) rg_v5 = tid+80	;
		if(tid+96	<seg_size) rg_v6 = tid+96	;
		if(tid+112	<seg_size) rg_v7 = tid+112	;

		// sort 128 elements
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
	}
	if (bin_it < normalized_bin_size) {
		// transpose the in register data of the whole warp
		int lane_id = threadIdx.x & 31;
		rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x1);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x1);
		rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x1);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x1);
		rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x1);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x1);
		rg_v5  = __shfl_xor_sync(0xffffffff,rg_v5 , 0x1);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x1);
		if(lane_id&0x1) SWP(K, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
		if(lane_id&0x1) SWP(K, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
		if(lane_id&0x1) SWP(K, rg_k4 , rg_k5 , int, rg_v4 , rg_v5 );
		if(lane_id&0x1) SWP(K, rg_k6 , rg_k7 , int, rg_v6 , rg_v7 );
		rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x1);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x1);
		rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x1);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x1);
		rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x1);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x1);
		rg_v5  = __shfl_xor_sync(0xffffffff,rg_v5 , 0x1);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x1);
		rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x2);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x2);
		rg_k6  = __shfl_xor_sync(0xffffffff,rg_k6 , 0x2);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x2);
		rg_v2  = __shfl_xor_sync(0xffffffff,rg_v2 , 0x2);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x2);
		rg_v6  = __shfl_xor_sync(0xffffffff,rg_v6 , 0x2);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x2);
		if(lane_id&0x2) SWP(K, rg_k0 , rg_k2 , int, rg_v0 , rg_v2 );
		if(lane_id&0x2) SWP(K, rg_k1 , rg_k3 , int, rg_v1 , rg_v3 );
		if(lane_id&0x2) SWP(K, rg_k4 , rg_k6 , int, rg_v4 , rg_v6 );
		if(lane_id&0x2) SWP(K, rg_k5 , rg_k7 , int, rg_v5 , rg_v7 );
		rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x2);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x2);
		rg_k6  = __shfl_xor_sync(0xffffffff,rg_k6 , 0x2);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x2);
		rg_v2  = __shfl_xor_sync(0xffffffff,rg_v2 , 0x2);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x2);
		rg_v6  = __shfl_xor_sync(0xffffffff,rg_v6 , 0x2);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x2);
		rg_k4  = __shfl_xor_sync(0xffffffff,rg_k4 , 0x4);
		rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x4);
		rg_k6  = __shfl_xor_sync(0xffffffff,rg_k6 , 0x4);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x4);
		rg_v4  = __shfl_xor_sync(0xffffffff,rg_v4 , 0x4);
		rg_v5  = __shfl_xor_sync(0xffffffff,rg_v5 , 0x4);
		rg_v6  = __shfl_xor_sync(0xffffffff,rg_v6 , 0x4);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x4);
		if(lane_id&0x4) SWP(K, rg_k0 , rg_k4 , int, rg_v0 , rg_v4 );
		if(lane_id&0x4) SWP(K, rg_k1 , rg_k5 , int, rg_v1 , rg_v5 );
		if(lane_id&0x4) SWP(K, rg_k2 , rg_k6 , int, rg_v2 , rg_v6 );
		if(lane_id&0x4) SWP(K, rg_k3 , rg_k7 , int, rg_v3 , rg_v7 );
		rg_k4  = __shfl_xor_sync(0xffffffff,rg_k4 , 0x4);
		rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x4);
		rg_k6  = __shfl_xor_sync(0xffffffff,rg_k6 , 0x4);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x4);
		rg_v4  = __shfl_xor_sync(0xffffffff,rg_v4 , 0x4);
		rg_v5  = __shfl_xor_sync(0xffffffff,rg_v5 , 0x4);
		rg_v6  = __shfl_xor_sync(0xffffffff,rg_v6 , 0x4);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x4);
		rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x8);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x8);
		rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x8);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x8);
		rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x8);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x8);
		rg_v5  = __shfl_xor_sync(0xffffffff,rg_v5 , 0x8);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x8);
		if(lane_id&0x8) SWP(K, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
		if(lane_id&0x8) SWP(K, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
		if(lane_id&0x8) SWP(K, rg_k4 , rg_k5 , int, rg_v4 , rg_v5 );
		if(lane_id&0x8) SWP(K, rg_k6 , rg_k7 , int, rg_v6 , rg_v7 );
		rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x8);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x8);
		rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x8);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x8);
		rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x8);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x8);
		rg_v5  = __shfl_xor_sync(0xffffffff,rg_v5 , 0x8);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x8);
		rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x10);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x10);
		rg_k6  = __shfl_xor_sync(0xffffffff,rg_k6 , 0x10);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x10);
		rg_v2  = __shfl_xor_sync(0xffffffff,rg_v2 , 0x10);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x10);
		rg_v6  = __shfl_xor_sync(0xffffffff,rg_v6 , 0x10);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x10);
		if(lane_id&0x10) SWP(K, rg_k0 , rg_k2 , int, rg_v0 , rg_v2 );
		if(lane_id&0x10) SWP(K, rg_k1 , rg_k3 , int, rg_v1 , rg_v3 );
		if(lane_id&0x10) SWP(K, rg_k4 , rg_k6 , int, rg_v4 , rg_v6 );
		if(lane_id&0x10) SWP(K, rg_k5 , rg_k7 , int, rg_v5 , rg_v7 );
		rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x10);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x10);
		rg_k6  = __shfl_xor_sync(0xffffffff,rg_k6 , 0x10);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x10);
		rg_v2  = __shfl_xor_sync(0xffffffff,rg_v2 , 0x10);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x10);
		rg_v6  = __shfl_xor_sync(0xffffffff,rg_v6 , 0x10);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x10);
		int kk;
		int ss;
		kk = __shfl_sync(0xffffffff, k, 0);
		ss = __shfl_sync(0xffffffff, seg_size, 0);
		if (lane_id + 0 < ss) keyB[kk+lane_id+0] = rg_k0;
		if (lane_id + 32 < ss) keyB[kk+lane_id+32] = rg_k4;
		if (lane_id + 64 < ss) keyB[kk+lane_id+64] = rg_k1;
		if (lane_id + 96 < ss) keyB[kk+lane_id+96] = rg_k5;
		kk = __shfl_sync(0xffffffff, k, 16);
		ss = __shfl_sync(0xffffffff, seg_size, 16);
		if (lane_id + 0 < ss) keyB[kk+lane_id+0] = rg_k2;
		if (lane_id + 32 < ss) keyB[kk+lane_id+32] = rg_k6;
		if (lane_id + 64 < ss) keyB[kk+lane_id+64] = rg_k3;
		if (lane_id + 96 < ss) keyB[kk+lane_id+96] = rg_k7;
		kk = __shfl_sync(0xffffffff, k, 0);
		ss = __shfl_sync(0xffffffff, seg_size, 0);
		if(lane_id+0 < ss) valB[kk+lane_id+0] = val[kk+rg_v0];
		if(lane_id+32 < ss) valB[kk+lane_id+32] = val[kk+rg_v4];
		if(lane_id+64 < ss) valB[kk+lane_id+64] = val[kk+rg_v1];
		if(lane_id+96 < ss) valB[kk+lane_id+96] = val[kk+rg_v5];
		kk = __shfl_sync(0xffffffff, k, 16);
		ss = __shfl_sync(0xffffffff, seg_size, 16);
		if(lane_id+0 < ss) valB[kk+lane_id+0] = val[kk+rg_v2];
		if(lane_id+32 < ss) valB[kk+lane_id+32] = val[kk+rg_v6];
		if(lane_id+64 < ss) valB[kk+lane_id+64] = val[kk+rg_v3];
		if(lane_id+96 < ss) valB[kk+lane_id+96] = val[kk+rg_v7];
	}
	else if(bin_it < bin_size) {
		if((tid<<3)+0 <seg_size) keyB[k+(tid<<3)+0] = rg_k0;
		if((tid<<3)+1 <seg_size) keyB[k+(tid<<3)+1] = rg_k1;
		if((tid<<3)+2 <seg_size) keyB[k+(tid<<3)+2] = rg_k2;
		if((tid<<3)+3 <seg_size) keyB[k+(tid<<3)+3] = rg_k3;
		if((tid<<3)+4 <seg_size) keyB[k+(tid<<3)+4] = rg_k4;
		if((tid<<3)+5 <seg_size) keyB[k+(tid<<3)+5] = rg_k5;
		if((tid<<3)+6 <seg_size) keyB[k+(tid<<3)+6] = rg_k6;
		if((tid<<3)+7 <seg_size) keyB[k+(tid<<3)+7] = rg_k7;

		if((tid<<3)+0 <seg_size) valB[k+(tid<<3)+0] = val[k+rg_v0];
		if((tid<<3)+1 <seg_size) valB[k+(tid<<3)+1] = val[k+rg_v1];
		if((tid<<3)+2 <seg_size) valB[k+(tid<<3)+2] = val[k+rg_v2];
		if((tid<<3)+3 <seg_size) valB[k+(tid<<3)+3] = val[k+rg_v3];
		if((tid<<3)+4 <seg_size) valB[k+(tid<<3)+4] = val[k+rg_v4];
		if((tid<<3)+5 <seg_size) valB[k+(tid<<3)+5] = val[k+rg_v5];
		if((tid<<3)+6 <seg_size) valB[k+(tid<<3)+6] = val[k+rg_v6];
		if((tid<<3)+7 <seg_size) valB[k+(tid<<3)+7] = val[k+rg_v7];
	}
}

// 16 threads | 16 ppt | *strided*
template<class K, class T>
__global__
void my_t16_ppt16_strd(K *key, T *val, K *keyB, T *valB, int n, int *segs, int *bin, int bin_size, int length) {
	const int gid = threadIdx.x + blockIdx.x * blockDim.x;
	const int bin_it = (gid>>4);
	const int tid = (threadIdx.x & 15);
	const int bit1 = (tid>>0)&0x1;
	const int bit2 = (tid>>1)&0x1;
	const int bit3 = (tid>>2)&0x1;
	const int bit4 = (tid>>3)&0x1;
	K rg_k0;
	K rg_k1;
	K rg_k2;
	K rg_k3;
	K rg_k4;
	K rg_k5;
	K rg_k6;
	K rg_k7;
	K rg_k8;
	K rg_k9;
	K rg_k10;
	K rg_k11;
	K rg_k12;
	K rg_k13;
	K rg_k14;
	K rg_k15;
	int rg_v0;
	int rg_v1;
	int rg_v2;
	int rg_v3;
	int rg_v4;
	int rg_v5;
	int rg_v6;
	int rg_v7;
	int rg_v8;
	int rg_v9;
	int rg_v10;
	int rg_v11;
	int rg_v12;
	int rg_v13;
	int rg_v14;
	int rg_v15;
	int normalized_bin_size = (bin_size/2)*2;
	int k;
	int seg_size;

	if(bin_it < bin_size) {
		k = segs[bin[bin_it]];
		seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
		rg_k0  = (tid+0	<seg_size)?key[k+tid+0	]:std::numeric_limits<K>::max();
		rg_k1  = (tid+16	<seg_size)?key[k+tid+16	]:std::numeric_limits<K>::max();
		rg_k2  = (tid+32	<seg_size)?key[k+tid+32	]:std::numeric_limits<K>::max();
		rg_k3  = (tid+48	<seg_size)?key[k+tid+48	]:std::numeric_limits<K>::max();
		rg_k4  = (tid+64	<seg_size)?key[k+tid+64	]:std::numeric_limits<K>::max();
		rg_k5  = (tid+80	<seg_size)?key[k+tid+80	]:std::numeric_limits<K>::max();
		rg_k6  = (tid+96	<seg_size)?key[k+tid+96	]:std::numeric_limits<K>::max();
		rg_k7  = (tid+112	<seg_size)?key[k+tid+112	]:std::numeric_limits<K>::max();
		rg_k8  = (tid+128	<seg_size)?key[k+tid+128	]:std::numeric_limits<K>::max();
		rg_k9  = (tid+144	<seg_size)?key[k+tid+144	]:std::numeric_limits<K>::max();
		rg_k10  = (tid+160	<seg_size)?key[k+tid+160	]:std::numeric_limits<K>::max();
		rg_k11  = (tid+176	<seg_size)?key[k+tid+176	]:std::numeric_limits<K>::max();
		rg_k12  = (tid+192	<seg_size)?key[k+tid+192	]:std::numeric_limits<K>::max();
		rg_k13  = (tid+208	<seg_size)?key[k+tid+208	]:std::numeric_limits<K>::max();
		rg_k14  = (tid+224	<seg_size)?key[k+tid+224	]:std::numeric_limits<K>::max();
		rg_k15  = (tid+240	<seg_size)?key[k+tid+240	]:std::numeric_limits<K>::max();
		if(tid+0	<seg_size) rg_v0 = tid+0	;
		if(tid+16	<seg_size) rg_v1 = tid+16	;
		if(tid+32	<seg_size) rg_v2 = tid+32	;
		if(tid+48	<seg_size) rg_v3 = tid+48	;
		if(tid+64	<seg_size) rg_v4 = tid+64	;
		if(tid+80	<seg_size) rg_v5 = tid+80	;
		if(tid+96	<seg_size) rg_v6 = tid+96	;
		if(tid+112	<seg_size) rg_v7 = tid+112	;
		if(tid+128	<seg_size) rg_v8 = tid+128	;
		if(tid+144	<seg_size) rg_v9 = tid+144	;
		if(tid+160	<seg_size) rg_v10 = tid+160	;
		if(tid+176	<seg_size) rg_v11 = tid+176	;
		if(tid+192	<seg_size) rg_v12 = tid+192	;
		if(tid+208	<seg_size) rg_v13 = tid+208	;
		if(tid+224	<seg_size) rg_v14 = tid+224	;
		if(tid+240	<seg_size) rg_v15 = tid+240	;

		// sort 256 elements
		// exch_intxn: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
		CMP_SWP(K, rg_k2, rg_k3, int, rg_v2, rg_v3);
		CMP_SWP(K, rg_k4, rg_k5, int, rg_v4, rg_v5);
		CMP_SWP(K, rg_k6, rg_k7, int, rg_v6, rg_v7);
		CMP_SWP(K, rg_k8, rg_k9, int, rg_v8, rg_v9);
		CMP_SWP(K, rg_k10, rg_k11, int, rg_v10, rg_v11);
		CMP_SWP(K, rg_k12, rg_k13, int, rg_v12, rg_v13);
		CMP_SWP(K, rg_k14, rg_k15, int, rg_v14, rg_v15);
		// exch_intxn: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k3, int, rg_v0, rg_v3);
		CMP_SWP(K, rg_k1, rg_k2, int, rg_v1, rg_v2);
		CMP_SWP(K, rg_k4, rg_k7, int, rg_v4, rg_v7);
		CMP_SWP(K, rg_k5, rg_k6, int, rg_v5, rg_v6);
		CMP_SWP(K, rg_k8, rg_k11, int, rg_v8, rg_v11);
		CMP_SWP(K, rg_k9, rg_k10, int, rg_v9, rg_v10);
		CMP_SWP(K, rg_k12, rg_k15, int, rg_v12, rg_v15);
		CMP_SWP(K, rg_k13, rg_k14, int, rg_v13, rg_v14);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
		CMP_SWP(K, rg_k2, rg_k3, int, rg_v2, rg_v3);
		CMP_SWP(K, rg_k4, rg_k5, int, rg_v4, rg_v5);
		CMP_SWP(K, rg_k6, rg_k7, int, rg_v6, rg_v7);
		CMP_SWP(K, rg_k8, rg_k9, int, rg_v8, rg_v9);
		CMP_SWP(K, rg_k10, rg_k11, int, rg_v10, rg_v11);
		CMP_SWP(K, rg_k12, rg_k13, int, rg_v12, rg_v13);
		CMP_SWP(K, rg_k14, rg_k15, int, rg_v14, rg_v15);
		// exch_intxn: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k7, int, rg_v0, rg_v7);
		CMP_SWP(K, rg_k1, rg_k6, int, rg_v1, rg_v6);
		CMP_SWP(K, rg_k2, rg_k5, int, rg_v2, rg_v5);
		CMP_SWP(K, rg_k3, rg_k4, int, rg_v3, rg_v4);
		CMP_SWP(K, rg_k8, rg_k15, int, rg_v8, rg_v15);
		CMP_SWP(K, rg_k9, rg_k14, int, rg_v9, rg_v14);
		CMP_SWP(K, rg_k10, rg_k13, int, rg_v10, rg_v13);
		CMP_SWP(K, rg_k11, rg_k12, int, rg_v11, rg_v12);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k2, int, rg_v0, rg_v2);
		CMP_SWP(K, rg_k1, rg_k3, int, rg_v1, rg_v3);
		CMP_SWP(K, rg_k4, rg_k6, int, rg_v4, rg_v6);
		CMP_SWP(K, rg_k5, rg_k7, int, rg_v5, rg_v7);
		CMP_SWP(K, rg_k8, rg_k10, int, rg_v8, rg_v10);
		CMP_SWP(K, rg_k9, rg_k11, int, rg_v9, rg_v11);
		CMP_SWP(K, rg_k12, rg_k14, int, rg_v12, rg_v14);
		CMP_SWP(K, rg_k13, rg_k15, int, rg_v13, rg_v15);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
		CMP_SWP(K, rg_k2, rg_k3, int, rg_v2, rg_v3);
		CMP_SWP(K, rg_k4, rg_k5, int, rg_v4, rg_v5);
		CMP_SWP(K, rg_k6, rg_k7, int, rg_v6, rg_v7);
		CMP_SWP(K, rg_k8, rg_k9, int, rg_v8, rg_v9);
		CMP_SWP(K, rg_k10, rg_k11, int, rg_v10, rg_v11);
		CMP_SWP(K, rg_k12, rg_k13, int, rg_v12, rg_v13);
		CMP_SWP(K, rg_k14, rg_k15, int, rg_v14, rg_v15);
		// exch_intxn: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k15, int, rg_v0, rg_v15);
		CMP_SWP(K, rg_k1, rg_k14, int, rg_v1, rg_v14);
		CMP_SWP(K, rg_k2, rg_k13, int, rg_v2, rg_v13);
		CMP_SWP(K, rg_k3, rg_k12, int, rg_v3, rg_v12);
		CMP_SWP(K, rg_k4, rg_k11, int, rg_v4, rg_v11);
		CMP_SWP(K, rg_k5, rg_k10, int, rg_v5, rg_v10);
		CMP_SWP(K, rg_k6, rg_k9, int, rg_v6, rg_v9);
		CMP_SWP(K, rg_k7, rg_k8, int, rg_v7, rg_v8);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k4, int, rg_v0, rg_v4);
		CMP_SWP(K, rg_k1, rg_k5, int, rg_v1, rg_v5);
		CMP_SWP(K, rg_k2, rg_k6, int, rg_v2, rg_v6);
		CMP_SWP(K, rg_k3, rg_k7, int, rg_v3, rg_v7);
		CMP_SWP(K, rg_k8, rg_k12, int, rg_v8, rg_v12);
		CMP_SWP(K, rg_k9, rg_k13, int, rg_v9, rg_v13);
		CMP_SWP(K, rg_k10, rg_k14, int, rg_v10, rg_v14);
		CMP_SWP(K, rg_k11, rg_k15, int, rg_v11, rg_v15);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k2, int, rg_v0, rg_v2);
		CMP_SWP(K, rg_k1, rg_k3, int, rg_v1, rg_v3);
		CMP_SWP(K, rg_k4, rg_k6, int, rg_v4, rg_v6);
		CMP_SWP(K, rg_k5, rg_k7, int, rg_v5, rg_v7);
		CMP_SWP(K, rg_k8, rg_k10, int, rg_v8, rg_v10);
		CMP_SWP(K, rg_k9, rg_k11, int, rg_v9, rg_v11);
		CMP_SWP(K, rg_k12, rg_k14, int, rg_v12, rg_v14);
		CMP_SWP(K, rg_k13, rg_k15, int, rg_v13, rg_v15);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
		CMP_SWP(K, rg_k2, rg_k3, int, rg_v2, rg_v3);
		CMP_SWP(K, rg_k4, rg_k5, int, rg_v4, rg_v5);
		CMP_SWP(K, rg_k6, rg_k7, int, rg_v6, rg_v7);
		CMP_SWP(K, rg_k8, rg_k9, int, rg_v8, rg_v9);
		CMP_SWP(K, rg_k10, rg_k11, int, rg_v10, rg_v11);
		CMP_SWP(K, rg_k12, rg_k13, int, rg_v12, rg_v13);
		CMP_SWP(K, rg_k14, rg_k15, int, rg_v14, rg_v15);
		// exch_intxn: generate exch_intxn()
		exch_intxn(rg_k0, rg_k1, rg_k2, rg_k3, rg_k4, rg_k5, rg_k6, rg_k7, rg_k8, rg_k9, rg_k10, rg_k11, rg_k12, rg_k13, rg_k14, rg_k15, rg_v0, rg_v1, rg_v2, rg_v3, rg_v4, rg_v5, rg_v6, rg_v7, rg_v8, rg_v9, rg_v10, rg_v11, rg_v12, rg_v13, rg_v14, rg_v15, 0x1, bit1);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k8, int, rg_v0, rg_v8);
		CMP_SWP(K, rg_k1, rg_k9, int, rg_v1, rg_v9);
		CMP_SWP(K, rg_k2, rg_k10, int, rg_v2, rg_v10);
		CMP_SWP(K, rg_k3, rg_k11, int, rg_v3, rg_v11);
		CMP_SWP(K, rg_k4, rg_k12, int, rg_v4, rg_v12);
		CMP_SWP(K, rg_k5, rg_k13, int, rg_v5, rg_v13);
		CMP_SWP(K, rg_k6, rg_k14, int, rg_v6, rg_v14);
		CMP_SWP(K, rg_k7, rg_k15, int, rg_v7, rg_v15);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k4, int, rg_v0, rg_v4);
		CMP_SWP(K, rg_k1, rg_k5, int, rg_v1, rg_v5);
		CMP_SWP(K, rg_k2, rg_k6, int, rg_v2, rg_v6);
		CMP_SWP(K, rg_k3, rg_k7, int, rg_v3, rg_v7);
		CMP_SWP(K, rg_k8, rg_k12, int, rg_v8, rg_v12);
		CMP_SWP(K, rg_k9, rg_k13, int, rg_v9, rg_v13);
		CMP_SWP(K, rg_k10, rg_k14, int, rg_v10, rg_v14);
		CMP_SWP(K, rg_k11, rg_k15, int, rg_v11, rg_v15);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k2, int, rg_v0, rg_v2);
		CMP_SWP(K, rg_k1, rg_k3, int, rg_v1, rg_v3);
		CMP_SWP(K, rg_k4, rg_k6, int, rg_v4, rg_v6);
		CMP_SWP(K, rg_k5, rg_k7, int, rg_v5, rg_v7);
		CMP_SWP(K, rg_k8, rg_k10, int, rg_v8, rg_v10);
		CMP_SWP(K, rg_k9, rg_k11, int, rg_v9, rg_v11);
		CMP_SWP(K, rg_k12, rg_k14, int, rg_v12, rg_v14);
		CMP_SWP(K, rg_k13, rg_k15, int, rg_v13, rg_v15);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
		CMP_SWP(K, rg_k2, rg_k3, int, rg_v2, rg_v3);
		CMP_SWP(K, rg_k4, rg_k5, int, rg_v4, rg_v5);
		CMP_SWP(K, rg_k6, rg_k7, int, rg_v6, rg_v7);
		CMP_SWP(K, rg_k8, rg_k9, int, rg_v8, rg_v9);
		CMP_SWP(K, rg_k10, rg_k11, int, rg_v10, rg_v11);
		CMP_SWP(K, rg_k12, rg_k13, int, rg_v12, rg_v13);
		CMP_SWP(K, rg_k14, rg_k15, int, rg_v14, rg_v15);
		// exch_intxn: generate exch_intxn()
		exch_intxn(rg_k0, rg_k1, rg_k2, rg_k3, rg_k4, rg_k5, rg_k6, rg_k7, rg_k8, rg_k9, rg_k10, rg_k11, rg_k12, rg_k13, rg_k14, rg_k15, rg_v0, rg_v1, rg_v2, rg_v3, rg_v4, rg_v5, rg_v6, rg_v7, rg_v8, rg_v9, rg_v10, rg_v11, rg_v12, rg_v13, rg_v14, rg_v15, 0x3, bit2);
		// exch_paral: generate exch_paral()
		exch_paral(rg_k0, rg_k1, rg_k2, rg_k3, rg_k4, rg_k5, rg_k6, rg_k7, rg_k8, rg_k9, rg_k10, rg_k11, rg_k12, rg_k13, rg_k14, rg_k15, rg_v0, rg_v1, rg_v2, rg_v3, rg_v4, rg_v5, rg_v6, rg_v7, rg_v8, rg_v9, rg_v10, rg_v11, rg_v12, rg_v13, rg_v14, rg_v15, 0x1, bit1);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k8, int, rg_v0, rg_v8);
		CMP_SWP(K, rg_k1, rg_k9, int, rg_v1, rg_v9);
		CMP_SWP(K, rg_k2, rg_k10, int, rg_v2, rg_v10);
		CMP_SWP(K, rg_k3, rg_k11, int, rg_v3, rg_v11);
		CMP_SWP(K, rg_k4, rg_k12, int, rg_v4, rg_v12);
		CMP_SWP(K, rg_k5, rg_k13, int, rg_v5, rg_v13);
		CMP_SWP(K, rg_k6, rg_k14, int, rg_v6, rg_v14);
		CMP_SWP(K, rg_k7, rg_k15, int, rg_v7, rg_v15);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k4, int, rg_v0, rg_v4);
		CMP_SWP(K, rg_k1, rg_k5, int, rg_v1, rg_v5);
		CMP_SWP(K, rg_k2, rg_k6, int, rg_v2, rg_v6);
		CMP_SWP(K, rg_k3, rg_k7, int, rg_v3, rg_v7);
		CMP_SWP(K, rg_k8, rg_k12, int, rg_v8, rg_v12);
		CMP_SWP(K, rg_k9, rg_k13, int, rg_v9, rg_v13);
		CMP_SWP(K, rg_k10, rg_k14, int, rg_v10, rg_v14);
		CMP_SWP(K, rg_k11, rg_k15, int, rg_v11, rg_v15);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k2, int, rg_v0, rg_v2);
		CMP_SWP(K, rg_k1, rg_k3, int, rg_v1, rg_v3);
		CMP_SWP(K, rg_k4, rg_k6, int, rg_v4, rg_v6);
		CMP_SWP(K, rg_k5, rg_k7, int, rg_v5, rg_v7);
		CMP_SWP(K, rg_k8, rg_k10, int, rg_v8, rg_v10);
		CMP_SWP(K, rg_k9, rg_k11, int, rg_v9, rg_v11);
		CMP_SWP(K, rg_k12, rg_k14, int, rg_v12, rg_v14);
		CMP_SWP(K, rg_k13, rg_k15, int, rg_v13, rg_v15);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
		CMP_SWP(K, rg_k2, rg_k3, int, rg_v2, rg_v3);
		CMP_SWP(K, rg_k4, rg_k5, int, rg_v4, rg_v5);
		CMP_SWP(K, rg_k6, rg_k7, int, rg_v6, rg_v7);
		CMP_SWP(K, rg_k8, rg_k9, int, rg_v8, rg_v9);
		CMP_SWP(K, rg_k10, rg_k11, int, rg_v10, rg_v11);
		CMP_SWP(K, rg_k12, rg_k13, int, rg_v12, rg_v13);
		CMP_SWP(K, rg_k14, rg_k15, int, rg_v14, rg_v15);
		// exch_intxn: generate exch_intxn()
		exch_intxn(rg_k0, rg_k1, rg_k2, rg_k3, rg_k4, rg_k5, rg_k6, rg_k7, rg_k8, rg_k9, rg_k10, rg_k11, rg_k12, rg_k13, rg_k14, rg_k15, rg_v0, rg_v1, rg_v2, rg_v3, rg_v4, rg_v5, rg_v6, rg_v7, rg_v8, rg_v9, rg_v10, rg_v11, rg_v12, rg_v13, rg_v14, rg_v15, 0x7, bit3);
		// exch_paral: generate exch_paral()
		exch_paral(rg_k0, rg_k1, rg_k2, rg_k3, rg_k4, rg_k5, rg_k6, rg_k7, rg_k8, rg_k9, rg_k10, rg_k11, rg_k12, rg_k13, rg_k14, rg_k15, rg_v0, rg_v1, rg_v2, rg_v3, rg_v4, rg_v5, rg_v6, rg_v7, rg_v8, rg_v9, rg_v10, rg_v11, rg_v12, rg_v13, rg_v14, rg_v15, 0x2, bit2);
		// exch_paral: generate exch_paral()
		exch_paral(rg_k0, rg_k1, rg_k2, rg_k3, rg_k4, rg_k5, rg_k6, rg_k7, rg_k8, rg_k9, rg_k10, rg_k11, rg_k12, rg_k13, rg_k14, rg_k15, rg_v0, rg_v1, rg_v2, rg_v3, rg_v4, rg_v5, rg_v6, rg_v7, rg_v8, rg_v9, rg_v10, rg_v11, rg_v12, rg_v13, rg_v14, rg_v15, 0x1, bit1);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k8, int, rg_v0, rg_v8);
		CMP_SWP(K, rg_k1, rg_k9, int, rg_v1, rg_v9);
		CMP_SWP(K, rg_k2, rg_k10, int, rg_v2, rg_v10);
		CMP_SWP(K, rg_k3, rg_k11, int, rg_v3, rg_v11);
		CMP_SWP(K, rg_k4, rg_k12, int, rg_v4, rg_v12);
		CMP_SWP(K, rg_k5, rg_k13, int, rg_v5, rg_v13);
		CMP_SWP(K, rg_k6, rg_k14, int, rg_v6, rg_v14);
		CMP_SWP(K, rg_k7, rg_k15, int, rg_v7, rg_v15);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k4, int, rg_v0, rg_v4);
		CMP_SWP(K, rg_k1, rg_k5, int, rg_v1, rg_v5);
		CMP_SWP(K, rg_k2, rg_k6, int, rg_v2, rg_v6);
		CMP_SWP(K, rg_k3, rg_k7, int, rg_v3, rg_v7);
		CMP_SWP(K, rg_k8, rg_k12, int, rg_v8, rg_v12);
		CMP_SWP(K, rg_k9, rg_k13, int, rg_v9, rg_v13);
		CMP_SWP(K, rg_k10, rg_k14, int, rg_v10, rg_v14);
		CMP_SWP(K, rg_k11, rg_k15, int, rg_v11, rg_v15);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k2, int, rg_v0, rg_v2);
		CMP_SWP(K, rg_k1, rg_k3, int, rg_v1, rg_v3);
		CMP_SWP(K, rg_k4, rg_k6, int, rg_v4, rg_v6);
		CMP_SWP(K, rg_k5, rg_k7, int, rg_v5, rg_v7);
		CMP_SWP(K, rg_k8, rg_k10, int, rg_v8, rg_v10);
		CMP_SWP(K, rg_k9, rg_k11, int, rg_v9, rg_v11);
		CMP_SWP(K, rg_k12, rg_k14, int, rg_v12, rg_v14);
		CMP_SWP(K, rg_k13, rg_k15, int, rg_v13, rg_v15);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
		CMP_SWP(K, rg_k2, rg_k3, int, rg_v2, rg_v3);
		CMP_SWP(K, rg_k4, rg_k5, int, rg_v4, rg_v5);
		CMP_SWP(K, rg_k6, rg_k7, int, rg_v6, rg_v7);
		CMP_SWP(K, rg_k8, rg_k9, int, rg_v8, rg_v9);
		CMP_SWP(K, rg_k10, rg_k11, int, rg_v10, rg_v11);
		CMP_SWP(K, rg_k12, rg_k13, int, rg_v12, rg_v13);
		CMP_SWP(K, rg_k14, rg_k15, int, rg_v14, rg_v15);
		// exch_intxn: generate exch_intxn()
		exch_intxn(rg_k0, rg_k1, rg_k2, rg_k3, rg_k4, rg_k5, rg_k6, rg_k7, rg_k8, rg_k9, rg_k10, rg_k11, rg_k12, rg_k13, rg_k14, rg_k15, rg_v0, rg_v1, rg_v2, rg_v3, rg_v4, rg_v5, rg_v6, rg_v7, rg_v8, rg_v9, rg_v10, rg_v11, rg_v12, rg_v13, rg_v14, rg_v15, 0xf, bit4);
		// exch_paral: generate exch_paral()
		exch_paral(rg_k0, rg_k1, rg_k2, rg_k3, rg_k4, rg_k5, rg_k6, rg_k7, rg_k8, rg_k9, rg_k10, rg_k11, rg_k12, rg_k13, rg_k14, rg_k15, rg_v0, rg_v1, rg_v2, rg_v3, rg_v4, rg_v5, rg_v6, rg_v7, rg_v8, rg_v9, rg_v10, rg_v11, rg_v12, rg_v13, rg_v14, rg_v15, 0x4, bit3);
		// exch_paral: generate exch_paral()
		exch_paral(rg_k0, rg_k1, rg_k2, rg_k3, rg_k4, rg_k5, rg_k6, rg_k7, rg_k8, rg_k9, rg_k10, rg_k11, rg_k12, rg_k13, rg_k14, rg_k15, rg_v0, rg_v1, rg_v2, rg_v3, rg_v4, rg_v5, rg_v6, rg_v7, rg_v8, rg_v9, rg_v10, rg_v11, rg_v12, rg_v13, rg_v14, rg_v15, 0x2, bit2);
		// exch_paral: generate exch_paral()
		exch_paral(rg_k0, rg_k1, rg_k2, rg_k3, rg_k4, rg_k5, rg_k6, rg_k7, rg_k8, rg_k9, rg_k10, rg_k11, rg_k12, rg_k13, rg_k14, rg_k15, rg_v0, rg_v1, rg_v2, rg_v3, rg_v4, rg_v5, rg_v6, rg_v7, rg_v8, rg_v9, rg_v10, rg_v11, rg_v12, rg_v13, rg_v14, rg_v15, 0x1, bit1);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k8, int, rg_v0, rg_v8);
		CMP_SWP(K, rg_k1, rg_k9, int, rg_v1, rg_v9);
		CMP_SWP(K, rg_k2, rg_k10, int, rg_v2, rg_v10);
		CMP_SWP(K, rg_k3, rg_k11, int, rg_v3, rg_v11);
		CMP_SWP(K, rg_k4, rg_k12, int, rg_v4, rg_v12);
		CMP_SWP(K, rg_k5, rg_k13, int, rg_v5, rg_v13);
		CMP_SWP(K, rg_k6, rg_k14, int, rg_v6, rg_v14);
		CMP_SWP(K, rg_k7, rg_k15, int, rg_v7, rg_v15);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k4, int, rg_v0, rg_v4);
		CMP_SWP(K, rg_k1, rg_k5, int, rg_v1, rg_v5);
		CMP_SWP(K, rg_k2, rg_k6, int, rg_v2, rg_v6);
		CMP_SWP(K, rg_k3, rg_k7, int, rg_v3, rg_v7);
		CMP_SWP(K, rg_k8, rg_k12, int, rg_v8, rg_v12);
		CMP_SWP(K, rg_k9, rg_k13, int, rg_v9, rg_v13);
		CMP_SWP(K, rg_k10, rg_k14, int, rg_v10, rg_v14);
		CMP_SWP(K, rg_k11, rg_k15, int, rg_v11, rg_v15);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k2, int, rg_v0, rg_v2);
		CMP_SWP(K, rg_k1, rg_k3, int, rg_v1, rg_v3);
		CMP_SWP(K, rg_k4, rg_k6, int, rg_v4, rg_v6);
		CMP_SWP(K, rg_k5, rg_k7, int, rg_v5, rg_v7);
		CMP_SWP(K, rg_k8, rg_k10, int, rg_v8, rg_v10);
		CMP_SWP(K, rg_k9, rg_k11, int, rg_v9, rg_v11);
		CMP_SWP(K, rg_k12, rg_k14, int, rg_v12, rg_v14);
		CMP_SWP(K, rg_k13, rg_k15, int, rg_v13, rg_v15);
		// exch_paral: switch to exch_local()
		CMP_SWP(K, rg_k0, rg_k1, int, rg_v0, rg_v1);
		CMP_SWP(K, rg_k2, rg_k3, int, rg_v2, rg_v3);
		CMP_SWP(K, rg_k4, rg_k5, int, rg_v4, rg_v5);
		CMP_SWP(K, rg_k6, rg_k7, int, rg_v6, rg_v7);
		CMP_SWP(K, rg_k8, rg_k9, int, rg_v8, rg_v9);
		CMP_SWP(K, rg_k10, rg_k11, int, rg_v10, rg_v11);
		CMP_SWP(K, rg_k12, rg_k13, int, rg_v12, rg_v13);
		CMP_SWP(K, rg_k14, rg_k15, int, rg_v14, rg_v15);
	}
	if (bin_it < normalized_bin_size) {
		// transpose the in register data of the whole warp
		int lane_id = threadIdx.x & 31;
		rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x1);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x1);
		rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x1);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x1);
		rg_k9  = __shfl_xor_sync(0xffffffff,rg_k9 , 0x1);
		rg_k11  = __shfl_xor_sync(0xffffffff,rg_k11 , 0x1);
		rg_k13  = __shfl_xor_sync(0xffffffff,rg_k13 , 0x1);
		rg_k15  = __shfl_xor_sync(0xffffffff,rg_k15 , 0x1);
		rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x1);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x1);
		rg_v5  = __shfl_xor_sync(0xffffffff,rg_v5 , 0x1);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x1);
		rg_v9  = __shfl_xor_sync(0xffffffff,rg_v9 , 0x1);
		rg_v11  = __shfl_xor_sync(0xffffffff,rg_v11 , 0x1);
		rg_v13  = __shfl_xor_sync(0xffffffff,rg_v13 , 0x1);
		rg_v15  = __shfl_xor_sync(0xffffffff,rg_v15 , 0x1);
		if(lane_id&0x1) SWP(K, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
		if(lane_id&0x1) SWP(K, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
		if(lane_id&0x1) SWP(K, rg_k4 , rg_k5 , int, rg_v4 , rg_v5 );
		if(lane_id&0x1) SWP(K, rg_k6 , rg_k7 , int, rg_v6 , rg_v7 );
		if(lane_id&0x1) SWP(K, rg_k8 , rg_k9 , int, rg_v8 , rg_v9 );
		if(lane_id&0x1) SWP(K, rg_k10 , rg_k11 , int, rg_v10 , rg_v11 );
		if(lane_id&0x1) SWP(K, rg_k12 , rg_k13 , int, rg_v12 , rg_v13 );
		if(lane_id&0x1) SWP(K, rg_k14 , rg_k15 , int, rg_v14 , rg_v15 );
		rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x1);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x1);
		rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x1);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x1);
		rg_k9  = __shfl_xor_sync(0xffffffff,rg_k9 , 0x1);
		rg_k11  = __shfl_xor_sync(0xffffffff,rg_k11 , 0x1);
		rg_k13  = __shfl_xor_sync(0xffffffff,rg_k13 , 0x1);
		rg_k15  = __shfl_xor_sync(0xffffffff,rg_k15 , 0x1);
		rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x1);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x1);
		rg_v5  = __shfl_xor_sync(0xffffffff,rg_v5 , 0x1);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x1);
		rg_v9  = __shfl_xor_sync(0xffffffff,rg_v9 , 0x1);
		rg_v11  = __shfl_xor_sync(0xffffffff,rg_v11 , 0x1);
		rg_v13  = __shfl_xor_sync(0xffffffff,rg_v13 , 0x1);
		rg_v15  = __shfl_xor_sync(0xffffffff,rg_v15 , 0x1);
		rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x2);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x2);
		rg_k6  = __shfl_xor_sync(0xffffffff,rg_k6 , 0x2);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x2);
		rg_k10  = __shfl_xor_sync(0xffffffff,rg_k10 , 0x2);
		rg_k11  = __shfl_xor_sync(0xffffffff,rg_k11 , 0x2);
		rg_k14  = __shfl_xor_sync(0xffffffff,rg_k14 , 0x2);
		rg_k15  = __shfl_xor_sync(0xffffffff,rg_k15 , 0x2);
		rg_v2  = __shfl_xor_sync(0xffffffff,rg_v2 , 0x2);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x2);
		rg_v6  = __shfl_xor_sync(0xffffffff,rg_v6 , 0x2);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x2);
		rg_v10  = __shfl_xor_sync(0xffffffff,rg_v10 , 0x2);
		rg_v11  = __shfl_xor_sync(0xffffffff,rg_v11 , 0x2);
		rg_v14  = __shfl_xor_sync(0xffffffff,rg_v14 , 0x2);
		rg_v15  = __shfl_xor_sync(0xffffffff,rg_v15 , 0x2);
		if(lane_id&0x2) SWP(K, rg_k0 , rg_k2 , int, rg_v0 , rg_v2 );
		if(lane_id&0x2) SWP(K, rg_k1 , rg_k3 , int, rg_v1 , rg_v3 );
		if(lane_id&0x2) SWP(K, rg_k4 , rg_k6 , int, rg_v4 , rg_v6 );
		if(lane_id&0x2) SWP(K, rg_k5 , rg_k7 , int, rg_v5 , rg_v7 );
		if(lane_id&0x2) SWP(K, rg_k8 , rg_k10 , int, rg_v8 , rg_v10 );
		if(lane_id&0x2) SWP(K, rg_k9 , rg_k11 , int, rg_v9 , rg_v11 );
		if(lane_id&0x2) SWP(K, rg_k12 , rg_k14 , int, rg_v12 , rg_v14 );
		if(lane_id&0x2) SWP(K, rg_k13 , rg_k15 , int, rg_v13 , rg_v15 );
		rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x2);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x2);
		rg_k6  = __shfl_xor_sync(0xffffffff,rg_k6 , 0x2);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x2);
		rg_k10  = __shfl_xor_sync(0xffffffff,rg_k10 , 0x2);
		rg_k11  = __shfl_xor_sync(0xffffffff,rg_k11 , 0x2);
		rg_k14  = __shfl_xor_sync(0xffffffff,rg_k14 , 0x2);
		rg_k15  = __shfl_xor_sync(0xffffffff,rg_k15 , 0x2);
		rg_v2  = __shfl_xor_sync(0xffffffff,rg_v2 , 0x2);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x2);
		rg_v6  = __shfl_xor_sync(0xffffffff,rg_v6 , 0x2);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x2);
		rg_v10  = __shfl_xor_sync(0xffffffff,rg_v10 , 0x2);
		rg_v11  = __shfl_xor_sync(0xffffffff,rg_v11 , 0x2);
		rg_v14  = __shfl_xor_sync(0xffffffff,rg_v14 , 0x2);
		rg_v15  = __shfl_xor_sync(0xffffffff,rg_v15 , 0x2);
		rg_k4  = __shfl_xor_sync(0xffffffff,rg_k4 , 0x4);
		rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x4);
		rg_k6  = __shfl_xor_sync(0xffffffff,rg_k6 , 0x4);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x4);
		rg_k12  = __shfl_xor_sync(0xffffffff,rg_k12 , 0x4);
		rg_k13  = __shfl_xor_sync(0xffffffff,rg_k13 , 0x4);
		rg_k14  = __shfl_xor_sync(0xffffffff,rg_k14 , 0x4);
		rg_k15  = __shfl_xor_sync(0xffffffff,rg_k15 , 0x4);
		rg_v4  = __shfl_xor_sync(0xffffffff,rg_v4 , 0x4);
		rg_v5  = __shfl_xor_sync(0xffffffff,rg_v5 , 0x4);
		rg_v6  = __shfl_xor_sync(0xffffffff,rg_v6 , 0x4);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x4);
		rg_v12  = __shfl_xor_sync(0xffffffff,rg_v12 , 0x4);
		rg_v13  = __shfl_xor_sync(0xffffffff,rg_v13 , 0x4);
		rg_v14  = __shfl_xor_sync(0xffffffff,rg_v14 , 0x4);
		rg_v15  = __shfl_xor_sync(0xffffffff,rg_v15 , 0x4);
		if(lane_id&0x4) SWP(K, rg_k0 , rg_k4 , int, rg_v0 , rg_v4 );
		if(lane_id&0x4) SWP(K, rg_k1 , rg_k5 , int, rg_v1 , rg_v5 );
		if(lane_id&0x4) SWP(K, rg_k2 , rg_k6 , int, rg_v2 , rg_v6 );
		if(lane_id&0x4) SWP(K, rg_k3 , rg_k7 , int, rg_v3 , rg_v7 );
		if(lane_id&0x4) SWP(K, rg_k8 , rg_k12 , int, rg_v8 , rg_v12 );
		if(lane_id&0x4) SWP(K, rg_k9 , rg_k13 , int, rg_v9 , rg_v13 );
		if(lane_id&0x4) SWP(K, rg_k10 , rg_k14 , int, rg_v10 , rg_v14 );
		if(lane_id&0x4) SWP(K, rg_k11 , rg_k15 , int, rg_v11 , rg_v15 );
		rg_k4  = __shfl_xor_sync(0xffffffff,rg_k4 , 0x4);
		rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x4);
		rg_k6  = __shfl_xor_sync(0xffffffff,rg_k6 , 0x4);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x4);
		rg_k12  = __shfl_xor_sync(0xffffffff,rg_k12 , 0x4);
		rg_k13  = __shfl_xor_sync(0xffffffff,rg_k13 , 0x4);
		rg_k14  = __shfl_xor_sync(0xffffffff,rg_k14 , 0x4);
		rg_k15  = __shfl_xor_sync(0xffffffff,rg_k15 , 0x4);
		rg_v4  = __shfl_xor_sync(0xffffffff,rg_v4 , 0x4);
		rg_v5  = __shfl_xor_sync(0xffffffff,rg_v5 , 0x4);
		rg_v6  = __shfl_xor_sync(0xffffffff,rg_v6 , 0x4);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x4);
		rg_v12  = __shfl_xor_sync(0xffffffff,rg_v12 , 0x4);
		rg_v13  = __shfl_xor_sync(0xffffffff,rg_v13 , 0x4);
		rg_v14  = __shfl_xor_sync(0xffffffff,rg_v14 , 0x4);
		rg_v15  = __shfl_xor_sync(0xffffffff,rg_v15 , 0x4);
		rg_k8  = __shfl_xor_sync(0xffffffff,rg_k8 , 0x8);
		rg_k9  = __shfl_xor_sync(0xffffffff,rg_k9 , 0x8);
		rg_k10  = __shfl_xor_sync(0xffffffff,rg_k10 , 0x8);
		rg_k11  = __shfl_xor_sync(0xffffffff,rg_k11 , 0x8);
		rg_k12  = __shfl_xor_sync(0xffffffff,rg_k12 , 0x8);
		rg_k13  = __shfl_xor_sync(0xffffffff,rg_k13 , 0x8);
		rg_k14  = __shfl_xor_sync(0xffffffff,rg_k14 , 0x8);
		rg_k15  = __shfl_xor_sync(0xffffffff,rg_k15 , 0x8);
		rg_v8  = __shfl_xor_sync(0xffffffff,rg_v8 , 0x8);
		rg_v9  = __shfl_xor_sync(0xffffffff,rg_v9 , 0x8);
		rg_v10  = __shfl_xor_sync(0xffffffff,rg_v10 , 0x8);
		rg_v11  = __shfl_xor_sync(0xffffffff,rg_v11 , 0x8);
		rg_v12  = __shfl_xor_sync(0xffffffff,rg_v12 , 0x8);
		rg_v13  = __shfl_xor_sync(0xffffffff,rg_v13 , 0x8);
		rg_v14  = __shfl_xor_sync(0xffffffff,rg_v14 , 0x8);
		rg_v15  = __shfl_xor_sync(0xffffffff,rg_v15 , 0x8);
		if(lane_id&0x8) SWP(K, rg_k0 , rg_k8 , int, rg_v0 , rg_v8 );
		if(lane_id&0x8) SWP(K, rg_k1 , rg_k9 , int, rg_v1 , rg_v9 );
		if(lane_id&0x8) SWP(K, rg_k2 , rg_k10 , int, rg_v2 , rg_v10 );
		if(lane_id&0x8) SWP(K, rg_k3 , rg_k11 , int, rg_v3 , rg_v11 );
		if(lane_id&0x8) SWP(K, rg_k4 , rg_k12 , int, rg_v4 , rg_v12 );
		if(lane_id&0x8) SWP(K, rg_k5 , rg_k13 , int, rg_v5 , rg_v13 );
		if(lane_id&0x8) SWP(K, rg_k6 , rg_k14 , int, rg_v6 , rg_v14 );
		if(lane_id&0x8) SWP(K, rg_k7 , rg_k15 , int, rg_v7 , rg_v15 );
		rg_k8  = __shfl_xor_sync(0xffffffff,rg_k8 , 0x8);
		rg_k9  = __shfl_xor_sync(0xffffffff,rg_k9 , 0x8);
		rg_k10  = __shfl_xor_sync(0xffffffff,rg_k10 , 0x8);
		rg_k11  = __shfl_xor_sync(0xffffffff,rg_k11 , 0x8);
		rg_k12  = __shfl_xor_sync(0xffffffff,rg_k12 , 0x8);
		rg_k13  = __shfl_xor_sync(0xffffffff,rg_k13 , 0x8);
		rg_k14  = __shfl_xor_sync(0xffffffff,rg_k14 , 0x8);
		rg_k15  = __shfl_xor_sync(0xffffffff,rg_k15 , 0x8);
		rg_v8  = __shfl_xor_sync(0xffffffff,rg_v8 , 0x8);
		rg_v9  = __shfl_xor_sync(0xffffffff,rg_v9 , 0x8);
		rg_v10  = __shfl_xor_sync(0xffffffff,rg_v10 , 0x8);
		rg_v11  = __shfl_xor_sync(0xffffffff,rg_v11 , 0x8);
		rg_v12  = __shfl_xor_sync(0xffffffff,rg_v12 , 0x8);
		rg_v13  = __shfl_xor_sync(0xffffffff,rg_v13 , 0x8);
		rg_v14  = __shfl_xor_sync(0xffffffff,rg_v14 , 0x8);
		rg_v15  = __shfl_xor_sync(0xffffffff,rg_v15 , 0x8);
		rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x10);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x10);
		rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x10);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x10);
		rg_k9  = __shfl_xor_sync(0xffffffff,rg_k9 , 0x10);
		rg_k11  = __shfl_xor_sync(0xffffffff,rg_k11 , 0x10);
		rg_k13  = __shfl_xor_sync(0xffffffff,rg_k13 , 0x10);
		rg_k15  = __shfl_xor_sync(0xffffffff,rg_k15 , 0x10);
		rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x10);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x10);
		rg_v5  = __shfl_xor_sync(0xffffffff,rg_v5 , 0x10);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x10);
		rg_v9  = __shfl_xor_sync(0xffffffff,rg_v9 , 0x10);
		rg_v11  = __shfl_xor_sync(0xffffffff,rg_v11 , 0x10);
		rg_v13  = __shfl_xor_sync(0xffffffff,rg_v13 , 0x10);
		rg_v15  = __shfl_xor_sync(0xffffffff,rg_v15 , 0x10);
		if(lane_id&0x10) SWP(K, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
		if(lane_id&0x10) SWP(K, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
		if(lane_id&0x10) SWP(K, rg_k4 , rg_k5 , int, rg_v4 , rg_v5 );
		if(lane_id&0x10) SWP(K, rg_k6 , rg_k7 , int, rg_v6 , rg_v7 );
		if(lane_id&0x10) SWP(K, rg_k8 , rg_k9 , int, rg_v8 , rg_v9 );
		if(lane_id&0x10) SWP(K, rg_k10 , rg_k11 , int, rg_v10 , rg_v11 );
		if(lane_id&0x10) SWP(K, rg_k12 , rg_k13 , int, rg_v12 , rg_v13 );
		if(lane_id&0x10) SWP(K, rg_k14 , rg_k15 , int, rg_v14 , rg_v15 );
		rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x10);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x10);
		rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x10);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x10);
		rg_k9  = __shfl_xor_sync(0xffffffff,rg_k9 , 0x10);
		rg_k11  = __shfl_xor_sync(0xffffffff,rg_k11 , 0x10);
		rg_k13  = __shfl_xor_sync(0xffffffff,rg_k13 , 0x10);
		rg_k15  = __shfl_xor_sync(0xffffffff,rg_k15 , 0x10);
		rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x10);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x10);
		rg_v5  = __shfl_xor_sync(0xffffffff,rg_v5 , 0x10);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x10);
		rg_v9  = __shfl_xor_sync(0xffffffff,rg_v9 , 0x10);
		rg_v11  = __shfl_xor_sync(0xffffffff,rg_v11 , 0x10);
		rg_v13  = __shfl_xor_sync(0xffffffff,rg_v13 , 0x10);
		rg_v15  = __shfl_xor_sync(0xffffffff,rg_v15 , 0x10);
		int kk;
		int ss;
		kk = __shfl_sync(0xffffffff, k, 0);
		ss = __shfl_sync(0xffffffff, seg_size, 0);
		if (lane_id + 0 < ss) keyB[kk+lane_id+0] = rg_k0;
		if (lane_id + 32 < ss) keyB[kk+lane_id+32] = rg_k2;
		if (lane_id + 64 < ss) keyB[kk+lane_id+64] = rg_k4;
		if (lane_id + 96 < ss) keyB[kk+lane_id+96] = rg_k6;
		if (lane_id + 128 < ss) keyB[kk+lane_id+128] = rg_k8;
		if (lane_id + 160 < ss) keyB[kk+lane_id+160] = rg_k10;
		if (lane_id + 192 < ss) keyB[kk+lane_id+192] = rg_k12;
		if (lane_id + 224 < ss) keyB[kk+lane_id+224] = rg_k14;
		kk = __shfl_sync(0xffffffff, k, 16);
		ss = __shfl_sync(0xffffffff, seg_size, 16);
		if (lane_id + 0 < ss) keyB[kk+lane_id+0] = rg_k1;
		if (lane_id + 32 < ss) keyB[kk+lane_id+32] = rg_k3;
		if (lane_id + 64 < ss) keyB[kk+lane_id+64] = rg_k5;
		if (lane_id + 96 < ss) keyB[kk+lane_id+96] = rg_k7;
		if (lane_id + 128 < ss) keyB[kk+lane_id+128] = rg_k9;
		if (lane_id + 160 < ss) keyB[kk+lane_id+160] = rg_k11;
		if (lane_id + 192 < ss) keyB[kk+lane_id+192] = rg_k13;
		if (lane_id + 224 < ss) keyB[kk+lane_id+224] = rg_k15;
		kk = __shfl_sync(0xffffffff, k, 0);
		ss = __shfl_sync(0xffffffff, seg_size, 0);
		if(lane_id+0 < ss) valB[kk+lane_id+0] = val[kk+rg_v0];
		if(lane_id+32 < ss) valB[kk+lane_id+32] = val[kk+rg_v2];
		if(lane_id+64 < ss) valB[kk+lane_id+64] = val[kk+rg_v4];
		if(lane_id+96 < ss) valB[kk+lane_id+96] = val[kk+rg_v6];
		if(lane_id+128 < ss) valB[kk+lane_id+128] = val[kk+rg_v8];
		if(lane_id+160 < ss) valB[kk+lane_id+160] = val[kk+rg_v10];
		if(lane_id+192 < ss) valB[kk+lane_id+192] = val[kk+rg_v12];
		if(lane_id+224 < ss) valB[kk+lane_id+224] = val[kk+rg_v14];
		kk = __shfl_sync(0xffffffff, k, 16);
		ss = __shfl_sync(0xffffffff, seg_size, 16);
		if(lane_id+0 < ss) valB[kk+lane_id+0] = val[kk+rg_v1];
		if(lane_id+32 < ss) valB[kk+lane_id+32] = val[kk+rg_v3];
		if(lane_id+64 < ss) valB[kk+lane_id+64] = val[kk+rg_v5];
		if(lane_id+96 < ss) valB[kk+lane_id+96] = val[kk+rg_v7];
		if(lane_id+128 < ss) valB[kk+lane_id+128] = val[kk+rg_v9];
		if(lane_id+160 < ss) valB[kk+lane_id+160] = val[kk+rg_v11];
		if(lane_id+192 < ss) valB[kk+lane_id+192] = val[kk+rg_v13];
		if(lane_id+224 < ss) valB[kk+lane_id+224] = val[kk+rg_v15];
	}
	else if(bin_it < bin_size) {
		if((tid<<4)+0 <seg_size) keyB[k+(tid<<4)+0] = rg_k0;
		if((tid<<4)+1 <seg_size) keyB[k+(tid<<4)+1] = rg_k1;
		if((tid<<4)+2 <seg_size) keyB[k+(tid<<4)+2] = rg_k2;
		if((tid<<4)+3 <seg_size) keyB[k+(tid<<4)+3] = rg_k3;
		if((tid<<4)+4 <seg_size) keyB[k+(tid<<4)+4] = rg_k4;
		if((tid<<4)+5 <seg_size) keyB[k+(tid<<4)+5] = rg_k5;
		if((tid<<4)+6 <seg_size) keyB[k+(tid<<4)+6] = rg_k6;
		if((tid<<4)+7 <seg_size) keyB[k+(tid<<4)+7] = rg_k7;
		if((tid<<4)+8 <seg_size) keyB[k+(tid<<4)+8] = rg_k8;
		if((tid<<4)+9 <seg_size) keyB[k+(tid<<4)+9] = rg_k9;
		if((tid<<4)+10 <seg_size) keyB[k+(tid<<4)+10] = rg_k10;
		if((tid<<4)+11 <seg_size) keyB[k+(tid<<4)+11] = rg_k11;
		if((tid<<4)+12 <seg_size) keyB[k+(tid<<4)+12] = rg_k12;
		if((tid<<4)+13 <seg_size) keyB[k+(tid<<4)+13] = rg_k13;
		if((tid<<4)+14 <seg_size) keyB[k+(tid<<4)+14] = rg_k14;
		if((tid<<4)+15 <seg_size) keyB[k+(tid<<4)+15] = rg_k15;

		if((tid<<4)+0 <seg_size) valB[k+(tid<<4)+0] = val[k+rg_v0];
		if((tid<<4)+1 <seg_size) valB[k+(tid<<4)+1] = val[k+rg_v1];
		if((tid<<4)+2 <seg_size) valB[k+(tid<<4)+2] = val[k+rg_v2];
		if((tid<<4)+3 <seg_size) valB[k+(tid<<4)+3] = val[k+rg_v3];
		if((tid<<4)+4 <seg_size) valB[k+(tid<<4)+4] = val[k+rg_v4];
		if((tid<<4)+5 <seg_size) valB[k+(tid<<4)+5] = val[k+rg_v5];
		if((tid<<4)+6 <seg_size) valB[k+(tid<<4)+6] = val[k+rg_v6];
		if((tid<<4)+7 <seg_size) valB[k+(tid<<4)+7] = val[k+rg_v7];
		if((tid<<4)+8 <seg_size) valB[k+(tid<<4)+8] = val[k+rg_v8];
		if((tid<<4)+9 <seg_size) valB[k+(tid<<4)+9] = val[k+rg_v9];
		if((tid<<4)+10 <seg_size) valB[k+(tid<<4)+10] = val[k+rg_v10];
		if((tid<<4)+11 <seg_size) valB[k+(tid<<4)+11] = val[k+rg_v11];
		if((tid<<4)+12 <seg_size) valB[k+(tid<<4)+12] = val[k+rg_v12];
		if((tid<<4)+13 <seg_size) valB[k+(tid<<4)+13] = val[k+rg_v13];
		if((tid<<4)+14 <seg_size) valB[k+(tid<<4)+14] = val[k+rg_v14];
		if((tid<<4)+15 <seg_size) valB[k+(tid<<4)+15] = val[k+rg_v15];
	}
}

// 32 threads | 8 ppt | *strided*
template<class K, class T>
__global__
void my_t32_ppt8_strd(K *key, T *val, K *keyB, T *valB, int n, int *segs, int *bin, int bin_size, int length) {
	const int gid = threadIdx.x + blockIdx.x * blockDim.x;
	const int bin_it = (gid>>5);
	const int tid = (threadIdx.x & 31);
	const int bit1 = (tid>>0)&0x1;
	const int bit2 = (tid>>1)&0x1;
	const int bit3 = (tid>>2)&0x1;
	const int bit4 = (tid>>3)&0x1;
	const int bit5 = (tid>>4)&0x1;
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
	int normalized_bin_size = (bin_size/1)*1;
	int k;
	int seg_size;

	if(bin_it < bin_size) {
		k = segs[bin[bin_it]];
		seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
		rg_k0  = (tid+0	<seg_size)?key[k+tid+0	]:std::numeric_limits<K>::max();
		rg_k1  = (tid+32	<seg_size)?key[k+tid+32	]:std::numeric_limits<K>::max();
		rg_k2  = (tid+64	<seg_size)?key[k+tid+64	]:std::numeric_limits<K>::max();
		rg_k3  = (tid+96	<seg_size)?key[k+tid+96	]:std::numeric_limits<K>::max();
		rg_k4  = (tid+128	<seg_size)?key[k+tid+128	]:std::numeric_limits<K>::max();
		rg_k5  = (tid+160	<seg_size)?key[k+tid+160	]:std::numeric_limits<K>::max();
		rg_k6  = (tid+192	<seg_size)?key[k+tid+192	]:std::numeric_limits<K>::max();
		rg_k7  = (tid+224	<seg_size)?key[k+tid+224	]:std::numeric_limits<K>::max();
		if(tid+0	<seg_size) rg_v0 = tid+0	;
		if(tid+32	<seg_size) rg_v1 = tid+32	;
		if(tid+64	<seg_size) rg_v2 = tid+64	;
		if(tid+96	<seg_size) rg_v3 = tid+96	;
		if(tid+128	<seg_size) rg_v4 = tid+128	;
		if(tid+160	<seg_size) rg_v5 = tid+160	;
		if(tid+192	<seg_size) rg_v6 = tid+192	;
		if(tid+224	<seg_size) rg_v7 = tid+224	;

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
	if (bin_it < normalized_bin_size) {
		// transpose the in register data of the whole warp
		int lane_id = threadIdx.x & 31;
		rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x1);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x1);
		rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x1);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x1);
		rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x1);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x1);
		rg_v5  = __shfl_xor_sync(0xffffffff,rg_v5 , 0x1);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x1);
		if(lane_id&0x1) SWP(K, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
		if(lane_id&0x1) SWP(K, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
		if(lane_id&0x1) SWP(K, rg_k4 , rg_k5 , int, rg_v4 , rg_v5 );
		if(lane_id&0x1) SWP(K, rg_k6 , rg_k7 , int, rg_v6 , rg_v7 );
		rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x1);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x1);
		rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x1);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x1);
		rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x1);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x1);
		rg_v5  = __shfl_xor_sync(0xffffffff,rg_v5 , 0x1);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x1);
		rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x2);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x2);
		rg_k6  = __shfl_xor_sync(0xffffffff,rg_k6 , 0x2);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x2);
		rg_v2  = __shfl_xor_sync(0xffffffff,rg_v2 , 0x2);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x2);
		rg_v6  = __shfl_xor_sync(0xffffffff,rg_v6 , 0x2);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x2);
		if(lane_id&0x2) SWP(K, rg_k0 , rg_k2 , int, rg_v0 , rg_v2 );
		if(lane_id&0x2) SWP(K, rg_k1 , rg_k3 , int, rg_v1 , rg_v3 );
		if(lane_id&0x2) SWP(K, rg_k4 , rg_k6 , int, rg_v4 , rg_v6 );
		if(lane_id&0x2) SWP(K, rg_k5 , rg_k7 , int, rg_v5 , rg_v7 );
		rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x2);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x2);
		rg_k6  = __shfl_xor_sync(0xffffffff,rg_k6 , 0x2);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x2);
		rg_v2  = __shfl_xor_sync(0xffffffff,rg_v2 , 0x2);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x2);
		rg_v6  = __shfl_xor_sync(0xffffffff,rg_v6 , 0x2);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x2);
		rg_k4  = __shfl_xor_sync(0xffffffff,rg_k4 , 0x4);
		rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x4);
		rg_k6  = __shfl_xor_sync(0xffffffff,rg_k6 , 0x4);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x4);
		rg_v4  = __shfl_xor_sync(0xffffffff,rg_v4 , 0x4);
		rg_v5  = __shfl_xor_sync(0xffffffff,rg_v5 , 0x4);
		rg_v6  = __shfl_xor_sync(0xffffffff,rg_v6 , 0x4);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x4);
		if(lane_id&0x4) SWP(K, rg_k0 , rg_k4 , int, rg_v0 , rg_v4 );
		if(lane_id&0x4) SWP(K, rg_k1 , rg_k5 , int, rg_v1 , rg_v5 );
		if(lane_id&0x4) SWP(K, rg_k2 , rg_k6 , int, rg_v2 , rg_v6 );
		if(lane_id&0x4) SWP(K, rg_k3 , rg_k7 , int, rg_v3 , rg_v7 );
		rg_k4  = __shfl_xor_sync(0xffffffff,rg_k4 , 0x4);
		rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x4);
		rg_k6  = __shfl_xor_sync(0xffffffff,rg_k6 , 0x4);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x4);
		rg_v4  = __shfl_xor_sync(0xffffffff,rg_v4 , 0x4);
		rg_v5  = __shfl_xor_sync(0xffffffff,rg_v5 , 0x4);
		rg_v6  = __shfl_xor_sync(0xffffffff,rg_v6 , 0x4);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x4);
		rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x8);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x8);
		rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x8);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x8);
		rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x8);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x8);
		rg_v5  = __shfl_xor_sync(0xffffffff,rg_v5 , 0x8);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x8);
		if(lane_id&0x8) SWP(K, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
		if(lane_id&0x8) SWP(K, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
		if(lane_id&0x8) SWP(K, rg_k4 , rg_k5 , int, rg_v4 , rg_v5 );
		if(lane_id&0x8) SWP(K, rg_k6 , rg_k7 , int, rg_v6 , rg_v7 );
		rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x8);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x8);
		rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x8);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x8);
		rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x8);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x8);
		rg_v5  = __shfl_xor_sync(0xffffffff,rg_v5 , 0x8);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x8);
		rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x10);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x10);
		rg_k6  = __shfl_xor_sync(0xffffffff,rg_k6 , 0x10);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x10);
		rg_v2  = __shfl_xor_sync(0xffffffff,rg_v2 , 0x10);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x10);
		rg_v6  = __shfl_xor_sync(0xffffffff,rg_v6 , 0x10);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x10);
		if(lane_id&0x10) SWP(K, rg_k0 , rg_k2 , int, rg_v0 , rg_v2 );
		if(lane_id&0x10) SWP(K, rg_k1 , rg_k3 , int, rg_v1 , rg_v3 );
		if(lane_id&0x10) SWP(K, rg_k4 , rg_k6 , int, rg_v4 , rg_v6 );
		if(lane_id&0x10) SWP(K, rg_k5 , rg_k7 , int, rg_v5 , rg_v7 );
		rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x10);
		rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x10);
		rg_k6  = __shfl_xor_sync(0xffffffff,rg_k6 , 0x10);
		rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x10);
		rg_v2  = __shfl_xor_sync(0xffffffff,rg_v2 , 0x10);
		rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x10);
		rg_v6  = __shfl_xor_sync(0xffffffff,rg_v6 , 0x10);
		rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x10);
		int kk;
		int ss;
		kk = __shfl_sync(0xffffffff, k, 0);
		ss = __shfl_sync(0xffffffff, seg_size, 0);
		if (lane_id + 0 < ss) keyB[kk+lane_id+0] = rg_k0;
		if (lane_id + 32 < ss) keyB[kk+lane_id+32] = rg_k4;
		if (lane_id + 64 < ss) keyB[kk+lane_id+64] = rg_k1;
		if (lane_id + 96 < ss) keyB[kk+lane_id+96] = rg_k5;
		if (lane_id + 128 < ss) keyB[kk+lane_id+128] = rg_k2;
		if (lane_id + 160 < ss) keyB[kk+lane_id+160] = rg_k6;
		if (lane_id + 192 < ss) keyB[kk+lane_id+192] = rg_k3;
		if (lane_id + 224 < ss) keyB[kk+lane_id+224] = rg_k7;
		kk = __shfl_sync(0xffffffff, k, 0);
		ss = __shfl_sync(0xffffffff, seg_size, 0);
		if(lane_id+0 < ss) valB[kk+lane_id+0] = val[kk+rg_v0];
		if(lane_id+32 < ss) valB[kk+lane_id+32] = val[kk+rg_v4];
		if(lane_id+64 < ss) valB[kk+lane_id+64] = val[kk+rg_v1];
		if(lane_id+96 < ss) valB[kk+lane_id+96] = val[kk+rg_v5];
		if(lane_id+128 < ss) valB[kk+lane_id+128] = val[kk+rg_v2];
		if(lane_id+160 < ss) valB[kk+lane_id+160] = val[kk+rg_v6];
		if(lane_id+192 < ss) valB[kk+lane_id+192] = val[kk+rg_v3];
		if(lane_id+224 < ss) valB[kk+lane_id+224] = val[kk+rg_v7];
	}
	else if(bin_it < bin_size) {
		if((tid<<3)+0 <seg_size) keyB[k+(tid<<3)+0] = rg_k0;
		if((tid<<3)+1 <seg_size) keyB[k+(tid<<3)+1] = rg_k1;
		if((tid<<3)+2 <seg_size) keyB[k+(tid<<3)+2] = rg_k2;
		if((tid<<3)+3 <seg_size) keyB[k+(tid<<3)+3] = rg_k3;
		if((tid<<3)+4 <seg_size) keyB[k+(tid<<3)+4] = rg_k4;
		if((tid<<3)+5 <seg_size) keyB[k+(tid<<3)+5] = rg_k5;
		if((tid<<3)+6 <seg_size) keyB[k+(tid<<3)+6] = rg_k6;
		if((tid<<3)+7 <seg_size) keyB[k+(tid<<3)+7] = rg_k7;

		if((tid<<3)+0 <seg_size) valB[k+(tid<<3)+0] = val[k+rg_v0];
		if((tid<<3)+1 <seg_size) valB[k+(tid<<3)+1] = val[k+rg_v1];
		if((tid<<3)+2 <seg_size) valB[k+(tid<<3)+2] = val[k+rg_v2];
		if((tid<<3)+3 <seg_size) valB[k+(tid<<3)+3] = val[k+rg_v3];
		if((tid<<3)+4 <seg_size) valB[k+(tid<<3)+4] = val[k+rg_v4];
		if((tid<<3)+5 <seg_size) valB[k+(tid<<3)+5] = val[k+rg_v5];
		if((tid<<3)+6 <seg_size) valB[k+(tid<<3)+6] = val[k+rg_v6];
		if((tid<<3)+7 <seg_size) valB[k+(tid<<3)+7] = val[k+rg_v7];
	}
}

#endif