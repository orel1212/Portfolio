
#include <cassert>
#include <cstdlib>
#include <functional>
#include <algorithm>

#include <vector>
#include <iostream>


// matrix and shared memory size 
const size_t N = 1 << 20;
const size_t SHARED_MEM_SIZE = 1 << 20;

// Check result on the CPU
void verify_CPU_GPU_results(std::vector<size_t> &first, std::vector<size_t> &second, std::vector<size_t> &result);
__global__ void matrixMultiplicationCUDA(const size_t *first, const size_t *second, size_t *output);


int main() {
  // size of matrix (bytes)
  size_t bytes = N * N * sizeof(size_t);

  // CPU vectors
  std::vector<size_t> h_first(N * N);
  std::vector<size_t> h_second(N * N);
  std::vector<size_t> h_result(N * N);

  // Initialize matrices
  std::generate(h_first.begin(), h_first.end(), []() { return rand() % 100; });
  std::generate(h_second.begin(),h_second.end(), []() { return rand() % 100; });

  // Allocate device memory
  size_t *d_first, *d_second, *d_output;
  cudaMalloc(&d_first, bytes);
  cudaMalloc(&d_second, bytes);
  cudaMalloc(&d_output, bytes);

  // Copy data to the device
  cudaMemcpy(d_first, h_first.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_second, h_second.data(), bytes, cudaMemcpyHostToDevice);

  size_t THREADS = 32;
  // blocks per grid dims
  size_t BLOCKS = N / THREADS;

  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS, BLOCKS);

  // kernel
  matrixMultiplicationCUDA<<<blocks, threads>>>(d_first, d_second, d_output);

  // Copy back to the host
  cudaMemcpy(h_result.data(), d_output, bytes, cudaMemcpyDeviceToHost);

  // Check result
  verify_CPU_GPU_results(h_first, h_second, h_result);

  std::cout << "Success Computation CPU vs GPU!" << std::endl ;

  // Free mem
  cudaFree(d_first);
  cudaFree(d_second);
  cudaFree(d_output);

  return 0;
}

// Check result on the CPU
void verify_CPU_GPU_results(std::vector<size_t> &first, std::vector<size_t> &second, std::vector<size_t> &result) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      // For every element in the row-col pair
      size_t accumulate = 0;
      for (int k = 0; k < N; k++) {
        accumulate += first[i * N + k] * second[k * N + j];
      }

      // Check CPU vs GPU
      assert(accumulate == result[i * N + j]);
    }
  }
}

__global__ void matrixMultiplicationCUDA(const size_t *first, const size_t *second, size_t *output) {
  // Compute each thread's row and column index
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  // Static shared memory
  __shared__ size_t sm_first[SHARED_MEM_SIZE];
  __shared__ size_t sm_second[SHARED_MEM_SIZE];

  // Accumulate in temporary variable
  size_t accumulate = 0;

  // Sweep tile across matrix
  for (int i = 0; i < N; i += blockDim.x) {
    // Load in elements for this tile
    sm_first[threadIdx.y * blockDim.x + threadIdx.x] = a[row * N + i + threadIdx.x];
    sm_second[threadIdx.y * blockDim.x + threadIdx.x] =
        b[i * N + threadIdx.y * N + col];

    // Wait for both shared_mems to be loaded
    __syncthreads();

    // Do matrix multiplication on the tiled matrix
    for (int j = 0; j < blockDim.x; j++) {
      int idx_first = threadIdx.y * blockDim.x + j
      int idx_second = j * blockDim.x + threadIdx.x
      accumulate +=
          sm_first[idx_first] * sm_second[idx_second];
    }

    // Wait for all threads to finish before new load
    __syncthreads();
  }

  // Write back results
  c[row * N + col] = accumulate;
}
