#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/types.h>

#define CEIL_DIV(m, n) (((m) + (n) - 1) / (n))

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_shared_memory_block_1d_tiling(int M, int N, int K,
                                                    float alpha, const float *A,
                                                    const float *B, float beta,
                                                    float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // allocating buffers for the fast shared memory
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // think of this in terms of C
  // these are row and col indices for C's block
  const uint threadCol = threadIdx.x % BN;
  const uint threadRow = threadIdx.x / BN;

  // advance pointers to starting positions
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  const int innerColA = threadIdx.x % BK;
  const int innerRowA = threadIdx.x / BK;
  const int innerColB = threadIdx.x % BN;
  const int innerRowB = threadIdx.x / BN;
  float threadResults[TM] = {0.0};

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];

    __syncthreads();
    A += BK;
    B += BK * N;

    for (uint dotIdx = 0; dotIdx < BK; dotIdx++) {
      float Btmp = Bs[dotIdx * BN + threadCol];
      for (uint resIdx = 0; resIdx < TM; resIdx++) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * Btmp;
      }
    }
    __syncthreads();
  }
  // write out the results
  for (uint resIdx = 0; resIdx < TM; resIdx++) {
    C[(threadRow * TM + resIdx) * N + threadCol] =
        alpha * threadResults[resIdx] +
        beta * C[(threadRow * TM + resIdx) * N + threadCol];
  }
}
void run_sgemm_shared_memory_1d_tiling(int M, int N, int K, float alpha,
                                       float *A, float *B, float beta,
                                       float *C) {
  const uint BM = 64;
  const uint BN = 64;
  const uint BK = 8;
  const uint TM = 8;
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  // dim3 blockDim(32, 32, 1);
  dim3 blockDim((BN * BM) / TM);
  sgemm_shared_memory_block_1d_tiling<BM, BN, BK, TM>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
