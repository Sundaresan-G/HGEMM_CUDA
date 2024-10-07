#include "kernels.cuh"
#include "runner.cuh"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <curand_kernel.h>

float get_sec() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return (1e6 * time.tv_sec + time.tv_usec);
}

float cpu_elapsed_time(float &beg, float &end) { return 1.0e-6 * (end - beg); }

void cudaCheck(cudaError_t error, const char *file, int line) {
  // if not defined NDEBUG
  # ifndef NDEBUG
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
  # endif
}

void CudaDeviceInfo() {
  int deviceId;

  cudaGetDevice(&deviceId);

  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props, deviceId);

  printf("Device ID: %d\n\
    Name: %s\n\
    Compute Capability: %d.%d\n\
    memoryBusWidth: %d\n\
    maxThreadsPerBlock: %d\n\
    maxThreadsPerMultiProcessor: %d\n\
    maxRegsPerBlock: %d\n\
    maxRegsPerMultiProcessor: %d\n\
    totalGlobalMem: %zuMB\n\
    sharedMemPerBlock: %zuKB\n\
    sharedMemPerMultiprocessor: %zuKB\n\
    totalConstMem: %zuKB\n\
    multiProcessorCount: %d\n\
    Warp Size: %d\n",
         deviceId, props.name, props.major, props.minor, props.memoryBusWidth,
         props.maxThreadsPerBlock, props.maxThreadsPerMultiProcessor,
         props.regsPerBlock, props.regsPerMultiprocessor,
         props.totalGlobalMem / 1024 / 1024, props.sharedMemPerBlock / 1024,
         props.sharedMemPerMultiprocessor / 1024, props.totalConstMem / 1024,
         props.multiProcessorCount, props.warpSize);
};

__global__ void init_random_matrix_kernel(half *mat, int num_elements) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  curandState localState;
  curand_init(1234, i, 0, &localState);
  if (i < num_elements) {
    mat[i] = __float2half(curand_uniform(&localState));
  }
}

void init_random_matrix(half *mat, int num_elements){
  init_random_matrix_kernel<<<CEIL_DIV(num_elements, 256), 256>>>(mat, num_elements);
}

__global__ void verifyKernel(half *matRef, half *matOut, int N){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    assert(fabs(__half2float(matRef[i] - matOut[i])) <= 1E-3);
  }
}

bool verify_matrix(half *matRef, half *matOut, int N) {
  verifyKernel<<<CEIL_DIV(N, 256), 256>>>(matRef, matOut, N);
  return true;
}

int div_ceil(int numerator, int denominator) {
  std::div_t res = std::div(numerator, denominator);
  return res.rem ? (res.quot + 1) : res.quot;
}

void runCublasFP16(cublasHandle_t& handle, int M, int N, int K, half alpha,
                   half *A, half *B, half beta, half *C) {
  // cuBLAS uses column-major order. So we change the order of our row-major A &
  // B, since (B^T*A^T)^T = (A*B)
  // This runs cuBLAS in full fp32 mode
  if(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N,
              A, K, &beta, C, N) != CUBLAS_STATUS_SUCCESS){
    printf("cuBLAS Hgemm failed\n");
    exit(EXIT_FAILURE);
  }
}

void run_hgemm_hierarchialTiling(int M, int N, int K, half alpha, half *A, half *B,
                     half beta, half *C) {
    const uint NUM_THREADS = 128;
    const uint BN = 128;
    const uint BM = 128;
    const uint BK = 16;

    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#wmma-type-sizes
    // For accumulator precision of fp16, m-n-k supported are 16x16x16, 32x8x16, 8x32x16
    const uint WN = 64;
    const uint WM = 64;
    const uint WK = 16;

    static_assert(BM % WM == 0 and BN % WN == 0 and BK % WK == 0,
                  "BM, BN, BK must be a multiple of WM, WN, WK respectively");

    const WMMA_MNK wmma_mnk = MNK_16x16x16;

    static_assert(WK % 16 == 0, "WK must be a multiple of 16");
    
    dim3 blockDim(NUM_THREADS);

    constexpr uint NUM_WARPS = NUM_THREADS / 32;

    // ensure NUM_THREADS is a multiple of BK and BN
    static_assert(NUM_THREADS % BK == 0, "NUM_THREADS must be a multiple of BK");
    static_assert(NUM_THREADS % BN == 0, "NUM_THREADS must be a multiple of BN");

    // warptile in threadblocktile
    static_assert((BN / WN) * (BM / WM) == NUM_WARPS);

    // static_assert((NUM_THREADS * 4) % BK == 0,
    //               "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
    //               "issues during GMEM->SMEM tiling (loading only parts of the "
    //               "final row of Bs during each iteraion)");
    // static_assert((NUM_THREADS * 4) % BN == 0,
    //               "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
    //               "issues during GMEM->SMEM tiling (loading only parts of the "
    //               "final row of As during each iteration)");
    // static_assert(BN % (16 * TN) == 0,
    //               "BN must be a multiple of 16*TN to avoid quantization effects");
    // static_assert(BM % (16 * TM) == 0,
    //               "BM must be a multiple of 16*TM to avoid quantization effects");
    // static_assert((BM * BK) % (4 * NUM_THREADS) == 0,
    //               "BM*BK must be a multiple of 4*256 to vectorize loads");
    // static_assert((BN * BK) % (4 * NUM_THREADS) == 0,
    //               "BN*BK must be a multiple of 4*256 to vectorize loads");

    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

    switch (wmma_mnk) {
        case MNK_16x16x16:
            static_assert(WN % 16 == 0, "WN must be a multiple of 16");
            static_assert(WM % 16 == 0, "WM must be a multiple of 16");
            hgemmHierarchialTiling<BM, BN, BK, WM, WN, WK, NUM_THREADS, 16, 16, 16>
              <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
            break;
        case MNK_32x8x16:
            static_assert(WN % 32 == 0, "WN must be a multiple of 32");
            static_assert(WM % 8 == 0, "WM must be a multiple of 8");
            hgemmHierarchialTiling<BM, BN, BK, WM, WN, WK, NUM_THREADS, 8, 32, 16>
              <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
            break;
        case MNK_8x32x16:
            static_assert(WN % 8 == 0, "WN must be a multiple of 8");
            static_assert(WM % 32 == 0, "WM must be a multiple of 32");
            hgemmHierarchialTiling<BM, BN, BK, WM, WN, WK, NUM_THREADS, 32, 8, 16>
              <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
            break;
    }
}

// void run_hgemm_coalesce(int M, int N, int K, float alpha, float *A, float *B,
//                         float beta, float *C) {
//   dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
//   dim3 blockDim(32 * 32);
//   hgemm_global_mem_coalesce<32>
//       <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
// }

// void run_hgemm_shared_mem_block(int M, int N, int K, float alpha, float *A,
//                                 float *B, float beta, float *C) {
//   dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
//   dim3 blockDim(32 * 32);
//   // L1 cache becomes useless, since we access GMEM only via SMEM, so we carve
//   // out all of L1 to SMEM. This doesn't currently make a difference, since
//   // occupancy is limited by reg and thread count, but it's good to do anyway.
//   cudaFuncSetAttribute(hgemm_shared_mem_block<32>,
//                        cudaFuncAttributePreferredSharedMemoryCarveout,
//                        cudaSharedmemCarveoutMaxShared);
//   hgemm_shared_mem_block<32>
//       <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
// }

// void runSgemm1DBlocktiling(int M, int N, int K, float alpha, float *A, float *B,
//                            float beta, float *C) {
//   const uint BM = 64;
//   const uint BN = 64;
//   const uint BK = 8;
//   const uint TM = 8;
//   dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
//   dim3 blockDim((BM * BN) / TM);
//   hgemm1DBlocktiling<BM, BN, BK, TM>
//       <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
// }

// void runSgemm2DBlocktiling(int M, int N, int K, float alpha, float *A, float *B,
//                            float beta, float *C) {
//   const uint BK = 8;
//   const uint TM = 8;
//   const uint TN = 8;
//   if (M >= 128 and N >= 128) {
//     const uint BM = 128;
//     const uint BN = 128;
//     dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
//     dim3 blockDim((BM * BN) / (TM * TN));
//     hgemm2DBlocktiling<BM, BN, BK, TM, TN>
//         <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
//   } else {
//     // this is a hacky solution to the underlying problem
//     // of not having proper bounds checking in the kernel
//     const uint BM = 64;
//     const uint BN = 64;
//     dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
//     dim3 blockDim((BM * BN) / (TM * TN));
//     hgemm2DBlocktiling<BM, BN, BK, TM, TN>
//         <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
//   }
// }

// void runSgemmVectorize(int M, int N, int K, float alpha, float *A, float *B,
//                        float beta, float *C) {
//   const uint BK = 8;
//   const uint TM = 8;
//   const uint TN = 8;
//   if (M >= 128 and N >= 128) {
//     const uint BM = 128;
//     const uint BN = 128;
//     dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
//     dim3 blockDim((BM * BN) / (TM * TN));
//     hgemmVectorize<BM, BN, BK, TM, TN>
//         <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
//   } else {
//     // this is a hacky solution to the underlying problem
//     // of not having proper bounds checking in the kernel
//     const uint BM = 64;
//     const uint BN = 64;
//     dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
//     dim3 blockDim((BM * BN) / (TM * TN));
//     hgemmVectorize<BM, BN, BK, TM, TN>
//         <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
//   }
// }

// void runSgemmResolveBankConflicts(int M, int N, int K, float alpha, float *A,
//                                   float *B, float beta, float *C) {
//   const uint BK = 8;
//   const uint TM = 8;
//   const uint TN = 8;
//   if (M >= 128 and N >= 128) {
//     const uint BM = 128;
//     const uint BN = 128;
//     dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
//     dim3 blockDim((BM * BN) / (TM * TN));
//     hgemmResolveBankConflicts<BM, BN, BK, TM, TN>
//         <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
//   } else {
//     // this is a hacky solution to the underlying problem
//     // of not having proper bounds checking in the kernel
//     const uint BM = 64;
//     const uint BN = 64;
//     dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
//     dim3 blockDim((BM * BN) / (TM * TN));
//     hgemmResolveBankConflicts<BM, BN, BK, TM, TN>
//         <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
//   }
// }

// void runSgemmResolveBankExtraCol(int M, int N, int K, float alpha, float *A,
//                                  float *B, float beta, float *C) {
//   const uint BK = 8;
//   const uint TM = 8;
//   const uint TN = 8;
//   if (M >= 128 and N >= 128) {
//     const uint BM = 128;
//     const uint BN = 128;
//     dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
//     dim3 blockDim((BM * BN) / (TM * TN));
//     hgemmResolveBankExtraCol<BM, BN, BK, TM, TN>
//         <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
//   } else {
//     // this is a hacky solution to the underlying problem
//     // of not having proper bounds checking in the kernel
//     const uint BM = 64;
//     const uint BN = 64;
//     dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
//     dim3 blockDim((BM * BN) / (TM * TN));
//     hgemmResolveBankExtraCol<BM, BN, BK, TM, TN>
//         <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
//   }
// }

// void runSgemmAutotuned(int M, int N, int K, float alpha, float *A, float *B,
//                        float beta, float *C) {
//   // A100
//   // const uint K9_BK = 16;
//   // const uint K9_TM = 4;
//   // const uint K9_TN = 4;
//   // const uint K9_BM = 64;
//   // const uint K9_BN = 64;
//   // A6000
//   const uint K9_BK = 16;
//   const uint K9_TM = 8;
//   const uint K9_TN = 8;
//   const uint K9_BM = 128;
//   const uint K9_BN = 128;
//   dim3 blockDim(K9_NUM_THREADS);

//   static_assert(
//       (K9_NUM_THREADS * 4) % K9_BK == 0,
//       "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization issues "
//       "during GMEM->SMEM tiling (loading only parts of the final row of Bs "
//       "during each iteraion)");
//   static_assert(
//       (K9_NUM_THREADS * 4) % K9_BN == 0,
//       "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization issues "
//       "during GMEM->SMEM tiling (loading only parts of the final row of As "
//       "during each iteration)");
//   static_assert(
//       K9_BN % (16 * K9_TN) == 0,
//       "K9_BN must be a multiple of 16*K9_TN to avoid quantization effects");
//   static_assert(
//       K9_BM % (16 * K9_TM) == 0,
//       "K9_BM must be a multiple of 16*K9_TM to avoid quantization effects");
//   static_assert((K9_BM * K9_BK) % (4 * K9_NUM_THREADS) == 0,
//                 "K9_BM*K9_BK must be a multiple of 4*256 to vectorize loads");
//   static_assert((K9_BN * K9_BK) % (4 * K9_NUM_THREADS) == 0,
//                 "K9_BN*K9_BK must be a multiple of 4*256 to vectorize loads");

//   dim3 gridDim(CEIL_DIV(N, K9_BN), CEIL_DIV(M, K9_BM));
//   hgemmAutotuned<K9_BM, K9_BN, K9_BK, K9_TM, K9_TN>
//       <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
// }

// void runSgemmWarptiling(int M, int N, int K, float alpha, float *A, float *B,
//                         float beta, float *C) {
//   // Settings for A100
//   // const uint K10_NUM_THREADS = 128;
//   // const uint K10_BN = 128;
//   // const uint K10_BM = 64;
//   // const uint K10_BK = 16;
//   // const uint K10_WN = 64;
//   // const uint K10_WM = 32;
//   // const uint K10_WNITER = 1;
//   // const uint K10_TN = 4;
//   // const uint K10_TM = 4;
//   // Settings for A6000
//   const uint K10_NUM_THREADS = 128;
//   const uint K10_BN = 128;
//   const uint K10_BM = 128;
//   const uint K10_BK = 16;
//   const uint K10_WN = 64;
//   const uint K10_WM = 64;
//   const uint K10_WNITER = 4;
//   const uint K10_TN = 4;
//   const uint K10_TM = 8;
//   dim3 blockDim(K10_NUM_THREADS);

//   constexpr uint NUM_WARPS = K10_NUM_THREADS / 32;

//   // warptile in threadblocktile
//   static_assert((K10_BN % K10_WN == 0) and (K10_BM % K10_WM == 0));
//   static_assert((K10_BN / K10_WN) * (K10_BM / K10_WM) == NUM_WARPS);

//   // threads in warpsubtile
//   static_assert((K10_WM * K10_WN) % (WARPSIZE * K10_TM * K10_TN * K10_WNITER) ==
//                 0);
//   constexpr uint K10_WMITER =
//       (K10_WM * K10_WN) / (32 * K10_TM * K10_TN * K10_WNITER);
//   // warpsubtile in warptile
//   static_assert((K10_WM % K10_WMITER == 0) and (K10_WN % K10_WNITER == 0));

//   static_assert((K10_NUM_THREADS * 4) % K10_BK == 0,
//                 "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
//                 "issues during GMEM->SMEM tiling (loading only parts of the "
//                 "final row of Bs during each iteraion)");
//   static_assert((K10_NUM_THREADS * 4) % K10_BN == 0,
//                 "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
//                 "issues during GMEM->SMEM tiling (loading only parts of the "
//                 "final row of As during each iteration)");
//   static_assert(K10_BN % (16 * K10_TN) == 0,
//                 "BN must be a multiple of 16*TN to avoid quantization effects");
//   static_assert(K10_BM % (16 * K10_TM) == 0,
//                 "BM must be a multiple of 16*TM to avoid quantization effects");
//   static_assert((K10_BM * K10_BK) % (4 * K10_NUM_THREADS) == 0,
//                 "BM*BK must be a multiple of 4*256 to vectorize loads");
//   static_assert((K10_BN * K10_BK) % (4 * K10_NUM_THREADS) == 0,
//                 "BN*BK must be a multiple of 4*256 to vectorize loads");

//   dim3 gridDim(CEIL_DIV(N, K10_BN), CEIL_DIV(M, K10_BM));
//   hgemmWarptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM,
//                   K10_TN, K10_NUM_THREADS>
//       <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
// }

// void runSgemmDoubleBuffering(int M, int N, int K, float alpha, float *A,
//                              float *B, float beta, float *C) {
//   // Settings for A100
//   // const uint K11_NUM_THREADS = 256;
//   // const uint K11_BN = 128;
//   // const uint K11_BM = 64;
//   // const uint K11_BK = 16;
//   // const uint K11_WN = 32;
//   // const uint K11_WM = 32;
//   // const uint K11_WNITER = 2;
//   // const uint K11_TN = 4;
//   // const uint K11_TM = 4;
//   // Settings for A6000
//   const uint K11_NUM_THREADS = 256;
//   const uint K11_BN = 256;
//   const uint K11_BM = 128;
//   const uint K11_BK = 16;
//   const uint K11_WN = 32;
//   const uint K11_WM = 128;
//   const uint K11_WNITER = 1;
//   const uint K11_TN = 8;
//   const uint K11_TM = 8;
//   dim3 blockDim(K11_NUM_THREADS);

//   constexpr uint NUM_WARPS = K11_NUM_THREADS / 32;

//   // warptile in threadblocktile
//   static_assert((K11_BN % K11_WN == 0) and (K11_BM % K11_WM == 0));
//   static_assert((K11_BN / K11_WN) * (K11_BM / K11_WM) == NUM_WARPS);

//   // threads in warpsubtile
//   static_assert((K11_WM * K11_WN) % (WARPSIZE * K11_TM * K11_TN * K11_WNITER) ==
//                 0);
//   constexpr uint K11_WMITER =
//       (K11_WM * K11_WN) / (32 * K11_TM * K11_TN * K11_WNITER);
//   // warpsubtile in warptile
//   static_assert((K11_WM % K11_WMITER == 0) and (K11_WN % K11_WNITER == 0));

//   static_assert((K11_NUM_THREADS / 2 * 4) % K11_BK == 0,
//                 "NUM_THREADS*4 must be multiple of BK to avoid quantization "
//                 "issues during GMEM->SMEM tiling (loading only parts of the "
//                 "final row of Bs during each iteraion)");
//   static_assert((K11_NUM_THREADS / 2 * 4) % K11_BN == 0,
//                 "NUM_THREADS*4 must be multiple of BN to avoid quantization "
//                 "issues during GMEM->SMEM tiling (loading only parts of the "
//                 "final row of As during each iteration)");
//   static_assert(K11_BN % (16 * K11_TN) == 0,
//                 "BN must be a multiple of 16*TN to avoid quantization effects");
//   static_assert(K11_BM % (16 * K11_TM) == 0,
//                 "BM must be a multiple of 16*TM to avoid quantization effects");
//   static_assert((K11_BM * K11_BK) % (4 * K11_NUM_THREADS / 2) == 0,
//                 "BM*BK must be a multiple of 4*256 to vectorize loads");
//   static_assert((K11_BN * K11_BK) % (4 * K11_NUM_THREADS / 2) == 0,
//                 "BN*BK must be a multiple of 4*256 to vectorize loads");

//   dim3 gridDim(CEIL_DIV(N, K11_BN), CEIL_DIV(M, K11_BM));
//   hgemmDoubleBuffering<K11_BM, K11_BN, K11_BK, K11_WM, K11_WN, K11_WNITER,
//                        K11_TM, K11_TN, K11_NUM_THREADS>
//       <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
// }

// void runSgemmDoubleBuffering2(int M, int N, int K, float alpha, float *A,
//                               float *B, float beta, float *C) {
//   // Settings for A6000
//   const uint K12_NUM_THREADS = 128;
//   const uint K12_BN = 128;
//   const uint K12_BM = 128;
//   const uint K12_BK = 16;
//   const uint K12_WN = 64;
//   const uint K12_WM = 64;
//   const uint K12_WNITER = 4;
//   const uint K12_TN = 4;
//   const uint K12_TM = 8;
//   dim3 blockDim(K12_NUM_THREADS);

//   constexpr uint NUM_WARPS = K12_NUM_THREADS / 32;

//   // warptile in threadblocktile
//   static_assert((K12_BN % K12_WN == 0) and (K12_BM % K12_WM == 0));
//   static_assert((K12_BN / K12_WN) * (K12_BM / K12_WM) == NUM_WARPS);

//   // threads in warpsubtile
//   static_assert((K12_WM * K12_WN) % (WARPSIZE * K12_TM * K12_TN * K12_WNITER) ==
//                 0);
//   constexpr uint K12_WMITER =
//       (K12_WM * K12_WN) / (32 * K12_TM * K12_TN * K12_WNITER);
//   // warpsubtile in warptile
//   static_assert((K12_WM % K12_WMITER == 0) and (K12_WN % K12_WNITER == 0));

//   static_assert((K12_NUM_THREADS * 4) % K12_BK == 0,
//                 "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
//                 "issues during GMEM->SMEM tiling (loading only parts of the "
//                 "final row of Bs during each iteraion)");
//   static_assert((K12_NUM_THREADS * 4) % K12_BN == 0,
//                 "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
//                 "issues during GMEM->SMEM tiling (loading only parts of the "
//                 "final row of As during each iteration)");
//   static_assert(K12_BN % (16 * K12_TN) == 0,
//                 "BN must be a multiple of 16*TN to avoid quantization effects");
//   static_assert(K12_BM % (16 * K12_TM) == 0,
//                 "BM must be a multiple of 16*TM to avoid quantization effects");
//   static_assert((K12_BM * K12_BK) % (4 * K12_NUM_THREADS) == 0,
//                 "BM*BK must be a multiple of 4*256 to vectorize loads");
//   static_assert((K12_BN * K12_BK) % (4 * K12_NUM_THREADS) == 0,
//                 "BN*BK must be a multiple of 4*256 to vectorize loads");

//   dim3 gridDim(CEIL_DIV(N, K12_BN), CEIL_DIV(M, K12_BM));
//   runSgemmDoubleBuffering2<K12_BM, K12_BN, K12_BK, K12_WM, K12_WN, K12_WNITER,
//                            K12_TM, K12_TN, K12_NUM_THREADS>
//       <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
// }

void run_kernel(int kernel_num, int M, int N, int K, half alpha, half *A,
                half *B, half beta, half *C, cublasHandle_t& handle) {
  switch (kernel_num) {
  case 0:
    runCublasFP16(handle, M, N, K, alpha, A, B, beta, C);
    break;
  case 1:
    run_hgemm_hierarchialTiling(M, N, K, alpha, A, B, beta, C);
    break;
  // case 2:
  //   run_hgemm_coalesce(M, N, K, alpha, A, B, beta, C);
  //   break;
  // case 3:
  //   run_hgemm_shared_mem_block(M, N, K, alpha, A, B, beta, C);
  //   break;
  // case 4:
  //   runSgemm1DBlocktiling(M, N, K, alpha, A, B, beta, C);
  //   break;
  // case 5:
  //   runSgemm2DBlocktiling(M, N, K, alpha, A, B, beta, C);
  //   break;
  // case 6:
  //   runSgemmVectorize(M, N, K, alpha, A, B, beta, C);
  //   break;
  // case 7:
  //   runSgemmResolveBankConflicts(M, N, K, alpha, A, B, beta, C);
  //   break;
  // case 8:
  //   runSgemmResolveBankExtraCol(M, N, K, alpha, A, B, beta, C);
  //   break;
  // case 9:
  //   runSgemmAutotuned(M, N, K, alpha, A, B, beta, C);
  //   break;
  // case 10:
  //   runSgemmWarptiling(M, N, K, alpha, A, B, beta, C);
  //   break;
  // case 11:
  //   runSgemmDoubleBuffering(M, N, K, alpha, A, B, beta, C);
  //   break;
  // case 12:
  //   runSgemmDoubleBuffering2(M, N, K, alpha, A, B, beta, C);
  //   break;
  default:
    throw std::invalid_argument("Unknown kernel number");
  }
}