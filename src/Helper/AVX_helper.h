#ifndef AVX_HELPER_H
#define AVX_HELPER_H

#include <stdint.h>

#ifdef __AVX512VL__
#define __mAVXi __m512i
#define AVX_setzero() _mm512_setzero_si512()
#define AVX_and(left, right)                 _mm512_and_epi32(left, right)
#define AVX_andnot(left, right)              _mm512_andnot_epi32(left, right)
#define AVX_or(left, right)                  _mm512_or_epi32(left, right)
#define AVX_xor(left, right)                 _mm512_xor_epi32(left, right)
#define AVX_bytes 64
static const uint32_t AVX_bits = 512;
static const uint32_t AVX_bitpointer_bits = 9;
static const uint32_t AVX_bytepointer_bits = 6;
static const uint32_t AVX_bytepointer_mask = 0x3f;
#else
#ifdef __AVX2__
#define __mAVXi __m256i
#define AVX_setzero() _mm256_setzero_si512()
#define AVX_and(left, right)                 _mm256_and_epi32(left, right)
#define AVX_andnot(left, right)              _mm256_andnot_epi32(left, right)
#define AVX_or(left, right)                  _mm256_or_epi32(left, right)
#define AVX_xor(left, right)                 _mm256_xor_epi32(left, right)
#define AVX_bytes 32
static const uint32_t AVX_bits = 256;
static const uint32_t AVX_bitpointer_bits = 8;
static const uint32_t AVX_bytepointer_bits = 5;
static const uint32_t AVX_bytepointer_mask = 0x1f;
#else
#define __mAVXi uint32_t
#define AVX_setzero() 0
#define AVX_and(left, right)      left & right
#define AVX_andnot(left, right)   left & ~right
#define AVX_or(left, right)       left | right
#define AVX_xor(left, right)      left ^ right
#define AVX_bytes 4
static const uint32_t AVX_bits = 32;
static const uint32_t AVX_bitpointer_bits = 5;
static const uint32_t AVX_bytepointer_bits = 2;
static const uint32_t AVX_bytepointer_mask = 0x3;
#endif
#endif

#endif // AVX_HELPER_H