
#include <miner.h>
extern "C" {
#include <stdint.h>
#include <memory.h>
}
#define HARAKAS_RATE 32

#include <cuda_helper.h>

#define NPT 2
#define NBN 2


__global__ void verus_gpu_hash(uint32_t threads, uint32_t startNonce, uint32_t *resNonce);

__device__ void haraka512_full(unsigned char *out, const unsigned char *in);
__device__ void haraka512_perm(unsigned char *out, const unsigned char *in);
	

static uint32_t *d_nonces[MAX_GPUS];

__constant__ uint8_t blockhash_half[128];
__constant__ uint32_t ptarget[8];

__host__
void verus_init(int thr_id)
{
	
	CUDA_SAFE_CALL(cudaMalloc(&d_nonces[thr_id], 2*sizeof(uint32_t)));
   
};


void verus_setBlock(void *blockf,const void *pTargetIn) 
{
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(ptarget, pTargetIn, 8*sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
 	CUDA_SAFE_CALL(cudaMemcpyToSymbol(blockhash_half, blockf, 64*sizeof(uint8_t), 0, cudaMemcpyHostToDevice));
};

__host__ 
void verus_hash(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *resNonces)
{
	cudaMemset(d_nonces[thr_id], 0xff, 2 * sizeof(uint32_t));
	const uint32_t threadsperblock = 256;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	verus_gpu_hash<<<grid, block>>>(threads, startNonce, d_nonces[thr_id]);
	cudaThreadSynchronize();
	cudaMemcpy(resNonces, d_nonces[thr_id], 2*sizeof(uint32_t), cudaMemcpyDeviceToHost);
 
};

__global__ 
void verus_gpu_hash(uint32_t threads, uint32_t startNonce, uint32_t *resNonce)
{
	uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;
	if (thread < threads)
	{
			uint32_t nounce = startNonce + thread;

			uint8_t hash_buf[64];
			uint8_t blockhash[64];
    
			memcpy(hash_buf,blockhash_half,128);
			memset(hash_buf + 32, 0x0,32);
			//memcpy(hash_buf + 32, (unsigned char *)&full_data + 1486 - 14, 15);
			((uint32_t *)&hash_buf)[8] = nounce;
  
			haraka512_full((unsigned char*)blockhash, (unsigned char*)hash_buf); // ( out, in)

			if (((uint64_t*)&blockhash)[3] < ((uint64_t*)&ptarget)[3]) { resNonce[0] = nounce;}   
    }
};

__device__ void memcpy_decker(unsigned char *dst, unsigned char *src, int len) {
    int i;
    for (i=0; i<len; i++) { dst[i] = src[i]; }
}

//__constant__ static const
__device__  unsigned char sbox[256] =
{ 0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe,
  0xd7, 0xab, 0x76, 0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4,
  0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0, 0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7,
  0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15, 0x04, 0xc7, 0x23, 0xc3,
  0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75, 0x09,
  0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3,
  0x2f, 0x84, 0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe,
  0x39, 0x4a, 0x4c, 0x58, 0xcf, 0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85,
  0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8, 0x51, 0xa3, 0x40, 0x8f, 0x92,
  0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2, 0xcd, 0x0c,
  0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19,
  0x73, 0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14,
  0xde, 0x5e, 0x0b, 0xdb, 0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2,
  0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79, 0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5,
  0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08, 0xba, 0x78, 0x25,
  0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
  0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86,
  0xc1, 0x1d, 0x9e, 0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e,
  0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf, 0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42,
  0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16 };

__device__  unsigned char smod[256] =
{ 0x00, 0x03, 0x06, 0x05, 0x0C, 0x0F, 0x0A, 0x09, 0x18, 0x1B, 0x1E, 0x1D, 0x14,
0x17, 0x12, 0x11, 0x30, 0x33, 0x36, 0x35, 0x3C, 0x3F, 0x3A, 0x39, 0x28, 0x2B,
0x2E, 0x2D, 0x24, 0x27, 0x22, 0x21, 0x60, 0x63, 0x66, 0x65, 0x6C, 0x6F, 0x6A,
0x69, 0x78, 0x7B, 0x7E, 0x7D, 0x74, 0x77, 0x72, 0x71, 0x50, 0x53, 0x56, 0x55,
0x5C, 0x5F, 0x5A, 0x59, 0x48, 0x4B, 0x4E, 0x4D, 0x44, 0x47, 0x42, 0x41, 0xC0,
0xC3, 0xC6, 0xC5, 0xCC, 0xCF, 0xCA, 0xC9, 0xD8, 0xDB, 0xDE, 0xDD, 0xD4, 0xD7,
0xD2, 0xD1, 0xF0, 0xF3, 0xF6, 0xF5, 0xFC, 0xFF, 0xFA, 0xF9, 0xE8, 0xEB, 0xEE,
0xED, 0xE4, 0xE7, 0xE2, 0xE1, 0xA0, 0xA3, 0xA6, 0xA5, 0xAC, 0xAF, 0xAA, 0xA9,
0xB8, 0xBB, 0xBE, 0xBD, 0xB4, 0xB7, 0xB2, 0xB1, 0x90, 0x93, 0x96, 0x95, 0x9C,
0x9F, 0x9A, 0x99, 0x88, 0x8B, 0x8E, 0x8D, 0x84, 0x87, 0x82, 0x81, 0x9B, 0x98,
0x9D, 0x9E, 0x97, 0x94, 0x91, 0x92, 0x83, 0x80, 0x85, 0x86, 0x8F, 0x8C, 0x89,
0x8A, 0xAB, 0xA8, 0xAD, 0xAE, 0xA7, 0xA4, 0xA1, 0xA2, 0xB3, 0xB0, 0xB5, 0xB6,
0xBF, 0xBC, 0xB9, 0xBA, 0xFB, 0xF8, 0xFD, 0xFE, 0xF7, 0xF4, 0xF1, 0xF2, 0xE3,
0xE0, 0xE5, 0xE6, 0xEF, 0xEC, 0xE9, 0xEA, 0xCB, 0xC8, 0xCD, 0xCE, 0xC7, 0xC4,
0xC1, 0xC2, 0xD3, 0xD0, 0xD5, 0xD6, 0xDF, 0xDC, 0xD9, 0xDA, 0x5B, 0x58, 0x5D,
0x5E, 0x57, 0x54, 0x51, 0x52, 0x43, 0x40, 0x45, 0x46, 0x4F, 0x4C, 0x49, 0x4A,
0x6B, 0x68, 0x6D, 0x6E, 0x67, 0x64, 0x61, 0x62, 0x73, 0x70, 0x75, 0x76, 0x7F,
0x7C, 0x79, 0x7A, 0x3B, 0x38, 0x3D, 0x3E, 0x37, 0x34, 0x31, 0x32, 0x23, 0x20,
0x25, 0x26, 0x2F, 0x2C, 0x29, 0x2A, 0x0B, 0x08, 0x0D, 0x0E, 0x07, 0x04, 0x01,
0x02, 0x13, 0x10, 0x15, 0x16, 0x1F, 0x1C, 0x19, 0x1A };


#define XT(x) (((x) << 1) ^ ((((x) >> 7) & 1) * 0x1b))

// Simulate _mm_aesenc_si128 instructions from AESNI
__device__  void aesenc(unsigned char *s,const unsigned char sharedMemory1[256])
{
    unsigned char i, t, u, v[4][4];
    for (i = 0; i < 16; ++i) {
        v[((i / 4) + 4 - (i%4) ) % 4][i % 4] = sharedMemory1[s[i]];
    }
    for (i = 0; i < 4; ++i) {
        t = v[i][0];
        u = v[i][0] ^ v[i][1] ^ v[i][2] ^ v[i][3];
        v[i][0] ^= u ^ XT(v[i][0] ^ v[i][1]);
        v[i][1] ^= u ^ XT(v[i][1] ^ v[i][2]);
        v[i][2] ^= u ^ XT(v[i][2] ^ v[i][3]);
        v[i][3] ^= u ^ XT(v[i][3] ^ t);
    }
    for (i = 0; i < 16; ++i) {
        s[i] = v[i / 4][i % 4]; // VerusHash have 0 rc vector
    }
}

__device__ void aesenc_double(unsigned char *s, const unsigned char sharedMemory1[256], const unsigned char sharedMemory2[256])
{
	unsigned char v0, v1, v2, v3, v4, v5, v6, v7;
	unsigned char t0, t1, t2;

	v0 = sharedMemory1[s[0]];
	v1 = sharedMemory1[s[5]];
	v2 = sharedMemory1[s[10]];
	v3 = sharedMemory1[s[15]];
	v4 = sharedMemory1[s[3]];
	v5 = sharedMemory1[s[2]];
	v6 = sharedMemory1[s[1]];

	t0 = v0 ^ v1;
	t1 = v1 ^ v2;
	t2 = v2 ^ v3;

	s[0] = sharedMemory2[t0] ^ v0 ^ t2;
	s[1] = sharedMemory2[t1] ^ t0 ^ v3;
	s[2] = sharedMemory2[t2] ^ v2 ^ t0;
	s[3] = sharedMemory2[v3 ^ v0] ^ v3 ^ t1;

	v0 = sharedMemory1[s[4]];
	v1 = sharedMemory1[s[9]];
	v2 = sharedMemory1[s[14]];
	v3 = v4;
	v4 = sharedMemory1[s[7]];
	v7 = sharedMemory1[s[6]];

	t0 = v0 ^ v1;
	t1 = v1 ^ v2;
	t2 = v2 ^ v3;

	s[4] = sharedMemory2[t0] ^ v0 ^ t2;
	s[5] = sharedMemory2[t1] ^ t0 ^ v3;
	s[6] = sharedMemory2[t2] ^ v2 ^ t0;
	s[7] = sharedMemory2[v3 ^ v0] ^ v3 ^ t1;

	v0 = sharedMemory1[s[8]];
	v1 = sharedMemory1[s[13]];
	v2 = v5;
	v3 = v4;
	v5 = sharedMemory1[s[11]];

	t0 = v0 ^ v1;
	t1 = v1 ^ v2;
	t2 = v2 ^ v3;

	s[8] = sharedMemory2[t0] ^ v0 ^ t2;
	s[9] = sharedMemory2[t1] ^ t0 ^ v3;

	s[10] = sharedMemory2[t2] ^ v2 ^ t0;
	s[11] = sharedMemory2[v3 ^ v0] ^ v3 ^ t1;

	v0 = sharedMemory1[s[12]];
	v1 = v6;
	v2 = v7;
	v3 = v5;

	t0 = v0 ^ v1;
	t1 = v1 ^ v2;
	t2 = v2 ^ v3;
	s[12] = sharedMemory2[t0] ^ v0 ^ t2;
	s[13] = sharedMemory2[t1] ^ t0 ^ v3;
	s[14] = sharedMemory2[t2] ^ v2 ^ t0;
	s[15] = sharedMemory2[v3 ^ v0] ^ v3 ^ t1;

	v0 = sharedMemory1[s[0]];
	v1 = sharedMemory1[s[5]];
	v2 = sharedMemory1[s[10]];
	v3 = sharedMemory1[s[15]];
	v4 = sharedMemory1[s[3]];
	v5 = sharedMemory1[s[2]];
	v6 = sharedMemory1[s[1]];

	t0 = v0 ^ v1;
	t1 = v1 ^ v2;
	t2 = v2 ^ v3;
	s[0] = sharedMemory2[t0] ^ v0 ^ t2;
	s[1] = sharedMemory2[t1] ^ t0 ^ v3;
	s[2] = sharedMemory2[t2] ^ v2 ^ t0;
	s[3] = sharedMemory2[v3 ^ v0] ^ v3 ^ t1;

	v0 = sharedMemory1[s[4]];
	v1 = sharedMemory1[s[9]];
	v2 = sharedMemory1[s[14]];
	v3 = v4;
	v4 = sharedMemory1[s[7]];
	v7 = sharedMemory1[s[6]];

	t0 = v0 ^ v1;
	t1 = v1 ^ v2;
	t2 = v2 ^ v3;
	s[4] = sharedMemory2[t0] ^ v0 ^ t2;
	s[5] = sharedMemory2[t1] ^ t0 ^ v3;
	s[6] = sharedMemory2[t2] ^ v2 ^ t0;
	s[7] = sharedMemory2[v3 ^ v0] ^ v3 ^ t1;

	v0 = sharedMemory1[s[8]];
	v1 = sharedMemory1[s[13]];
	v2 = v5;
	v3 = v4;
	v5 = sharedMemory1[s[11]];

	t0 = v0 ^ v1;
	t1 = v1 ^ v2;
	t2 = v2 ^ v3;
	s[8] = sharedMemory2[t0] ^ v0 ^ t2;
	s[9] = sharedMemory2[t1] ^ t0 ^ v3;
	s[10] = sharedMemory2[t2] ^ v2 ^ t0;
	s[11] = sharedMemory2[v3 ^ v0] ^ v3 ^ t1;

	v0 = sharedMemory1[s[12]];
	v1 = v6;
	v2 = v7;
	v3 = v5;

	t0 = v0 ^ v1;
	t1 = v1 ^ v2;
	t2 = v2 ^ v3;
	s[12] = sharedMemory2[t0] ^ v0 ^ t2;
	s[13] = sharedMemory2[t1] ^ t0 ^ v3;
	s[14] = sharedMemory2[t2] ^ v2 ^ t0;
	s[15] = sharedMemory2[v3 ^ v0] ^ v3 ^ t1;
}

// Simulate _mm_unpacklo_epi32
__device__ __forceinline__ void unpacklo32(unsigned char *t, unsigned char *a, unsigned char *b)
{
    unsigned char tmp[16];
    memcpy_decker(tmp, a, 4);
    memcpy_decker(tmp + 4, b, 4);
    memcpy_decker(tmp + 8, a + 4, 4);
    memcpy_decker(tmp + 12, b + 4, 4);
    memcpy_decker(t, tmp, 16);
}

// Simulate _mm_unpackhi_epi32
__device__ __forceinline__ void unpackhi32(unsigned char *t, unsigned char *a, unsigned char *b)
{
    unsigned char tmp[16];
    memcpy_decker(tmp, a + 8, 4);
    memcpy_decker(tmp + 4, b + 8, 4);
    memcpy_decker(tmp + 8, a + 12, 4);
    memcpy_decker(tmp + 12, b + 12, 4);
    memcpy_decker(t, tmp, 16);
}



__device__ void haraka512_perm(unsigned char *out, const unsigned char *in) 
{

    int i, j;
	__shared__ unsigned char sharedMemory1[256];
	
	if (threadIdx.x < 256) 
		sharedMemory1[threadIdx.x] = sbox[threadIdx.x];

    unsigned char s[64], tmp[16];
    memcpy_decker(s, (unsigned char *)in, 64);
#pragma unroll
    for (i = 0; i < 5; ++i) {
        // aes round(s)
		
			for (j = 0; j < 2; ++j) {

				aesenc(s, sharedMemory1);
				aesenc(s + 16, sharedMemory1);
				aesenc(s + 32, sharedMemory1);
				aesenc(s + 48, sharedMemory1);
			}
		
		unpacklo32(tmp, s, s + 16);
		
		unpackhi32(s, s, s + 16);
        unpacklo32(s + 16, s + 32, s + 48);
        unpackhi32(s + 32, s + 32, s + 48);
        unpacklo32(s + 48, s, s + 32);
        unpackhi32(s, s, s + 32);
        unpackhi32(s + 32, s + 16, tmp);
        unpacklo32(s + 16, s + 16, tmp);
    }

    memcpy_decker(out, s, 64);
}

/*__device__ void haraka512_full(unsigned char *out, const unsigned char *in)
{
    int i;

    unsigned char buf[64];
    haraka512_perm(buf, in);
    for (i = 0; i < 64; i++) {
        buf[i] = buf[i] ^ in[i];
    }

    memcpy_decker(out,      buf + 8, 8);
    memcpy_decker(out + 8,  buf + 24, 8);
    memcpy_decker(out + 16, buf + 32, 8);
    memcpy_decker(out + 24, buf + 48, 8);
}*/

__device__ void haraka512_full(unsigned char *out, const unsigned char *in) {

	__shared__ unsigned char sharedMemory1[256];
	__shared__ unsigned char sharedMemory2[256];

	if (threadIdx.x < 256)
		sharedMemory1[threadIdx.x] = sbox[threadIdx.x];
	if (threadIdx.x < 256)
		sharedMemory2[threadIdx.x] = smod[threadIdx.x];

	unsigned char s[64];
	uint32_t *sd = (uint32_t*)(&s[0]);
	memcpy_decker(s, (unsigned char *)in, 64);
	
	#pragma unroll
	for (int i = 0; i < 4; ++i) {

		aesenc_double(s, sharedMemory1, sharedMemory2);
		aesenc_double(s + 16, sharedMemory1, sharedMemory2);
		aesenc_double(s + 32, sharedMemory1, sharedMemory2);
		aesenc_double(s + 48, sharedMemory1, sharedMemory2);

		// mixing
		uint32_t t;

		t = sd[0];
		sd[0] = sd[3];
		sd[3] = sd[15];
		sd[15] = sd[14];
		sd[14] = sd[6];
		sd[6] = sd[12];
		sd[12] = sd[2];
		sd[2] = sd[7];
		sd[7] = sd[4];
		sd[4] = sd[8];
		sd[8] = sd[9];
		sd[9] = sd[1];
		sd[1] = sd[11];
		sd[11] = sd[5];
		sd[5] = t;

		t = sd[13];
		sd[13] = sd[10];
		sd[10] = t;
	}

	aesenc_double(s, sharedMemory1, sharedMemory2);
	aesenc_double(s + 16, sharedMemory1, sharedMemory2);
	aesenc_double(s + 32, sharedMemory1, sharedMemory2);
	aesenc_double(s + 48, sharedMemory1, sharedMemory2);

	uint32_t *outd = ((uint32_t *)out);
	uint32_t *ind = ((uint32_t *)in);

	//__syncthreads();
	*outd++ = sd[7] ^ ind[2];
	*outd++ = sd[15] ^ ind[3];
	*outd++ = sd[12] ^ ind[6];
	*outd++ = sd[4] ^ ind[7];
	*outd++ = sd[9] ^ ind[8];
	*outd++ = sd[1] ^ ind[9];
	*outd++ = sd[2] ^ ind[12];
	*outd = sd[10] ^ ind[13];
	//__syncthreads();


	/*

	// original variant

	int i, j;
	unsigned char buf[64];

	unsigned char s[64], tmp[16];
	memcpy_decker(s, (unsigned char *)in, 64);
	#pragma unroll
	for (i = 0; i < 5; ++i) {
		// aes round(s)

		for (j = 0; j < 2; ++j) {

			aesenc(s, sharedMemory1);
			aesenc(s + 16, sharedMemory1);
			aesenc(s + 32, sharedMemory1);
			aesenc(s + 48, sharedMemory1);
		}

		unpacklo32(tmp, s, s + 16);

		unpackhi32(s, s, s + 16);
		unpacklo32(s + 16, s + 32, s + 48);
		unpackhi32(s + 32, s + 32, s + 48);
		unpacklo32(s + 48, s, s + 32);
		unpackhi32(s, s, s + 32);
		unpackhi32(s + 32, s + 16, tmp);
		unpacklo32(s + 16, s + 16, tmp);
	}

	memcpy_decker(buf, s, 64);
	

	for (i = 0; i < 64; i++) {
		buf[i] = buf[i] ^ in[i];
	}

	memcpy_decker(out, buf + 8, 8);
	memcpy_decker(out + 8, buf + 24, 8);
	memcpy_decker(out + 16, buf + 32, 8);
	memcpy_decker(out + 24, buf + 48, 8);
	*/

}
