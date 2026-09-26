#pragma once
#include <cuda_runtime.h>
#include <stdio.h>

#define CEIL_DIV(m, n) (((m) + (n) - 1) / (n))

template <const int BLOCK_SIZE>
__global__ void sgemm_shared_memory_block(int M, int N, int K, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C) {
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  // allocating buffers for the fast shared memory
  __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
  __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

  const uint threadCol = threadIdx.x % BLOCK_SIZE;
  const uint threadRow = threadIdx.x / BLOCK_SIZE;

  // advance pointers to starting positions
  A += cRow * BLOCK_SIZE * K;
  B += cCol * BLOCK_SIZE;
  C += cRow * BLOCK_SIZE * N + cCol * BLOCK_SIZE;

  float tmp = 0.0;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCK_SIZE) {
    As[threadRow * BLOCK_SIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCK_SIZE + threadCol] = B[threadRow * N + threadCol];

    // we can do this or add offsets to pointer below.
    // In my bechmarks I found that approach to be faster
    //
    // As[threadRow * BLOCK_SIZE + threadCol] =
    //     A[threadRow * K + (threadCol + bkIdx)];
    // Bs[threadRow * BLOCK_SIZE + threadCol] =
    //     B[(threadRow + bkIdx) * N + threadCol];

    // block threads in this block until cache is fully populated
    __syncthreads();

    // execute dot product on the currenlty fetched block
    for (int dotIdx = 0; dotIdx < BLOCK_SIZE; dotIdx++) {
      tmp += (As[threadRow * BLOCK_SIZE + dotIdx] *
              Bs[dotIdx * BLOCK_SIZE + threadCol]);
    }
    // Sync here again as faster threads may start loading
    // next block while slower threads are still executing
    __syncthreads();
    // advance pointers to next chunk
    A += BLOCK_SIZE;
    B += BLOCK_SIZE * N;
  }
  C[threadRow * N + threadCol] =
      alpha * tmp + beta * C[threadRow * N + threadCol];
}

void run_sgemm_shared_memory(int M, int N, int K, float alpha, float *A,
                             float *B, float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
  // dim3 blockDim(32, 32, 1);
  dim3 blockDim(32 * 32);
  // prefer maximum memory for shared memory
  cudaFuncSetAttribute(sgemm_shared_memory_block<32>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  sgemm_shared_memory_block<32>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
