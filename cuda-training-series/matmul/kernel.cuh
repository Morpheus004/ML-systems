#pragma once
#include <cuda_runtime.h>
#include <stdio.h>

#define CEIL_DIV(m, n) (((m) + (n) - 1) / (n))

template <const int BLOCK_SIZE>
__global__ void sgemm_coalescing(int M, int N, int K, float alpha,
                                 const float *A, const float *B, float beta,
                                 float *C) {
  // const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  // const uint y = blockIdx.y * blockDim.y + threadIdx.y;
  const uint x = blockIdx.x * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE);
  const uint y = blockIdx.y * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE);
  if (x < M && y < N) {
    float tmp = 0.0f;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    // C = α*(A@B)+β*C
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}

void run_sgemm_coalescing(int M, int N, int K, float alpha, float *A, float *B,
                          float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
  // dim3 blockDim(32, 32, 1);
  dim3 blockDim(32 * 32);
  sgemm_coalescing<32><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
