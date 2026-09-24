# SGEMM CUDA Optimization Benchmark

Following Simon Boehm's [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM).

## Environment
- **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU
- **Compute Capability:** 8.9 (Ada Lovelace, `sm_89`)
- **CUDA:** 13.1
- **Matrix Size:** $M = N = K = 4096$ (FP32)

---

## Benchmark Results

| # | Kernel | Time (ms) | GFLOPS | % of cuBLAS | Description |
|---|---|---:|---:|---:|---|
| 0 | **cuBLAS** | 19.79 ms | 6,943.6 GFLOPS | 100.0% | NVIDIA cuBLAS reference baseline |
| 1 | **Naive** | 1142.33 ms | 120.3 GFLOPS | 1.73% | One thread per element $C[x, y]$, uncoalesced global memory access |
| 2 | **Coalescing** | - | - | - | Memory coalescing via contiguous row/col indexing |
| 3 | **Shared Memory Cache** | - | - | - | 1D/2D tile staging in `__shared__` memory |
| 4 | **1D Block Tiling** | - | - | - | Each thread computes multiple elements along a column |
| 5 | **2D Block Tiling** | - | - | - | Each thread computes a $TM \times TN$ tile in registers |
| 6 | **Vectorized Access** | - | - | - | `float4` vectorized loads (`LDG.E.128`) |
| 7 | **Resolve Bank Conflicts**| - | - | - | Padding / layout tweaks to eliminate bank conflicts |
| 8 | **Double Buffering** | - | - | - | Overlapping compute and memory transfers |
| 9 | **Autotuning** | - | - | - | Grid/block/tile parameter sweep |
| 10 | **Tensor Cores** | - | - | - | Hardware WMMA / MMA instructions |

---

## Build & Run

```bash
nvcc -O3 -arch=sm_89 runner.cu -lcublas -o runner
./runner
```
To test custom matrix sizes:
```bash
./runner 4096
```
