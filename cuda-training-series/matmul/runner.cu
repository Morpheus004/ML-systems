#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "kernels/1_naive.cuh"

#define CUDA_CHECK(err)                                                        \
  if (err != cudaSuccess) {                                                    \
    printf("CUDA Error: %s (at %s:%d)\n", cudaGetErrorString(err), __FILE__,   \
           __LINE__);                                                          \
    exit(1);                                                                   \
  }

void randomize_matrix(float *mat, int size) {
  for (int i = 0; i < size; ++i) {
    mat[i] = (float)rand() / (float)RAND_MAX;
  }
}

bool verify_matrix(const float *ref, const float *out, int size,
                   float tol = 1e-2f) {
  float max_diff = 0.0f;
  for (int i = 0; i < size; ++i) {
    float diff = fabs(ref[i] - out[i]);
    if (diff > max_diff)
      max_diff = diff;
    if (diff > tol) {
      printf("Verification FAILED at index %d: ref=%f, out=%f, diff=%f\n", i,
             ref[i], out[i], diff);
      return false;
    }
  }
  printf("Verification PASSED (max diff: %e)\n", max_diff);
  return true;
}

int main(int argc, char **argv) {
  int M = 4096;
  int N = 4096;
  int K = 4096;
  float alpha = 1.0f;
  float beta = 0.0f;
  int repeats = 10;

  if (argc > 1)
    M = N = K = atoi(argv[1]);

  printf("Matrix Size: M=%d, N=%d, K=%d\n", M, N, K);

  size_t bytes_A = M * K * sizeof(float);
  size_t bytes_B = K * N * sizeof(float);
  size_t bytes_C = M * N * sizeof(float);

  float *h_A = (float *)malloc(bytes_A);
  float *h_B = (float *)malloc(bytes_B);
  float *h_C = (float *)malloc(bytes_C);
  float *h_C_ref = (float *)malloc(bytes_C);

  srand(42);
  randomize_matrix(h_A, M * K);
  randomize_matrix(h_B, K * N);

  float *d_A, *d_B, *d_C, *d_C_ref;
  CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
  CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
  CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
  CUDA_CHECK(cudaMalloc(&d_C_ref, bytes_C));

  CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  double total_flops = 2.0 * (double)M * (double)N * (double)K;

  // ----------------------------------------------------
  // 0. cuBLAS (Ground Truth Reference for Verification)
  // ----------------------------------------------------
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Single pass to generate reference output matrix C_ref
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K,
              &beta, d_C_ref, N);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_C_ref, d_C_ref, bytes_C, cudaMemcpyDeviceToHost));

  // Baseline GFLOPS logged in README.md
  const double cublas_gflops = 6943.64;
  printf("[cuBLAS]      Baseline Reference: %8.2f GFLOPS\n", cublas_gflops);

  // ----------------------------------------------------
  // 1. Kernel 1: Naive
  // ----------------------------------------------------
  // Warmup
  run_sgemm_naive(M, N, K, alpha, d_A, d_B, beta, d_C);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < repeats; ++i) {
    run_sgemm_naive(M, N, K, alpha, d_A, d_B, beta, d_C);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float naive_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&naive_ms, start, stop));
  naive_ms /= repeats;
  double naive_gflops = (total_flops * 1e-9) / (naive_ms * 1e-3);

  CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));
  printf("[1. Naive]    Time: %7.3f ms | Perf: %8.2f GFLOPS | %5.2f%% of "
         "cuBLAS | ",
         naive_ms, naive_gflops, (naive_gflops / cublas_gflops) * 100.0);
  verify_matrix(h_C_ref, h_C, M * N);

  // Cleanup
  cublasDestroy(handle);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_C_ref);
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_ref);

  return 0;
}
