#include <cstdio>
#include <stdio.h>

__global__ void hello() {

  printf("Hello from block: %d, thread: %d\n", (int)blockIdx.x,
         (int)threadIdx.x);
}

int main() {

  printf("Host Print\n");
  hello<<<2, 2>>>();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA device synchronize error: %s\n", cudaGetErrorString(err));
  }
  return 0;
}
