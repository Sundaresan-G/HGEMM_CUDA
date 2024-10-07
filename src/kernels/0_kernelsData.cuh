#pragma once

#include <cassert>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
const int WARPSIZE = 32; // warpSize is not constexpr

// Enum to choose one of 16x16x16, 32x8x16, 8x32x16
enum WMMA_MNK {
    MNK_16x16x16,
    MNK_32x8x16,
    MNK_8x32x16
}; 