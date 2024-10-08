#include "0_kernelsData.cuh"

/*
 * @tparam BM The threadblock size for M dimension SMEM caching.
 * @tparam BN The threadblock size for N dimension SMEM caching.
 * @tparam BK The threadblock size for K dimension SMEM caching.
 * @tparam WM M dim of continuous tile computed by each warp
 * @tparam WN N dim of continuous tile computed by each warp
 * @tparam WMITER The number of subwarp tiling steps in M dimension.
 * @tparam WNITER The number of subwarp tiling steps in N dimension.
 * @tparam TM The per-thread tile size for M dimension.
 * @tparam TN The per-thread tile size for N dimension.
 * @tparam TK 
 */
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WK, const int NUM_THREADS, const int mma_m, const int mma_n, const int mma_k>
__global__ void __launch_bounds__(NUM_THREADS)
    hgemmHierarchialTiling(int M, int N, int K, half alpha, half *A, half *B,
                    half beta, half *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;

  // Placement of the warp in the threadblock tile
  const uint warpIdx = threadIdx.x / WARPSIZE; // the warp this thread is in
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);

  // number of iterations in M and N dimensions for the warp
  constexpr uint WMITER = WM / mma_m;
  constexpr uint WNITER = WN / mma_n;

  // stride of the warp in M and N dimensions
  // constexpr uint WSUBM = WM / WMITER; // 64/2=32
  // constexpr uint WSUBN = WN / WNITER; // 32/2=16

  // Placement of the thread in the warp subtile
  // const uint threadIdxInWarp = threadIdx.x % WARPSIZE;         // [0, 31]
  // const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN); // i%(16/4)
  // const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN); // i/4

  // allocate space for the current blocktile in SMEM
  __shared__ half As[BM * BK];
  __shared__ half Bs[BK * BN];

  // Move C_ptr to warp's output tile
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  // calculating the indices that this thread will load into SMEM
  // we'll load 128bit / 32bit = 4 elements per thread at each step
  // const uint innerRowA = threadIdx.x / (BK / 4);
  // const uint innerColA = threadIdx.x % (BK / 4);
  // constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
  // const uint innerRowB = threadIdx.x / (BN / 4);
  // const uint innerColB = threadIdx.x % (BN / 4);
  // constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

  const uint innerRowA = threadIdx.x / (BK);
  const uint innerColA = threadIdx.x % (BK);
  constexpr uint rowStrideA = (NUM_THREADS) / BK;
  const uint innerRowB = threadIdx.x / (BN);
  const uint innerColB = threadIdx.x % (BN);
  constexpr uint rowStrideB = NUM_THREADS / (BN);

  wmma::fragment<wmma::matrix_a, mma_m, mma_n, mma_k, half, wmma::row_major> a_frag[WMITER];
  wmma::fragment<wmma::matrix_b, mma_m, mma_n, mma_k, half, wmma::row_major> b_frag[WNITER];
  wmma::fragment<wmma::accumulator, mma_m, mma_n, mma_k, half> c_acc[WMITER][WNITER];
  wmma::fragment<wmma::accumulator, mma_m, mma_n, mma_k, half> c_frag;

  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {

          // Initialize the output to zero
          wmma::fill_fragment(c_acc[wSubRowIdx][wSubColIdx], 0.0f);
        }
  }

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {

    // load blocktile into SMEM
    for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
      As[(innerRowA + offset) * BK + innerColA] = 
        A[(innerRowA + offset) * K + innerColA];
    }

    for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
      Bs[(innerRowB + offset) * BN + innerColB] =
          B[(innerRowB + offset) * N + innerColB];
    }

    __syncthreads();
    auto AsWarpLocalInBlock = &As[warpRow * WM * BK];
    auto BsWarpLocalInBlock = &Bs[warpCol * WN];
    for (uint warpKidx = 0; warpKidx < BK; warpKidx += WK) {
      for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        wmma::load_matrix_sync(a_frag[wSubRowIdx], AsWarpLocalInBlock, BK);
        AsWarpLocalInBlock += mma_m * BK;
      }
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        wmma::load_matrix_sync(b_frag[wSubColIdx], BsWarpLocalInBlock, BN);
        BsWarpLocalInBlock += mma_n;
      }

      for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
          // Perform the matrix multiplication
          wmma::mma_sync(c_acc[wSubRowIdx][wSubColIdx], a_frag[wSubRowIdx], b_frag[wSubColIdx], c_acc[wSubRowIdx][wSubColIdx]);
        }
      }
    }
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down
    __syncthreads();
  }

  // write out the results
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      // Load the current sub-matrix from C
      wmma::load_matrix_sync(c_frag, C, N, wmma::mem_row_major);

      for(int i=0; i < c_frag.num_elements; i++) {
        c_frag.x[i] = alpha * c_acc[wSubRowIdx][wSubColIdx].x[i] + beta * c_frag.x[i];
      }

      wmma::store_matrix_sync(C, c_frag, N, wmma::mem_row_major);  

      // Move C column to the right
      C += mma_n; 
    }
    // Move C row down
    C += mma_m * N;
  }
}