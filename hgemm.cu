#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <runner.cuh>
#include <vector>
#include <cuda_fp16.hpp>

const std::string errLogFile = "matrixValidationFailure.txt";

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Please select a kernel (range 0 - 12, 0 for NVIDIA cuBLAS)"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // get kernel number
  int kernel_num = std::stoi(argv[1]);
  if (kernel_num < 0 || kernel_num > 12) {
    std::cerr << "Please enter a valid kernel number (0-12)" << std::endl;
    exit(EXIT_FAILURE);
  }

  // get environment variable for device
  int deviceIdx = 0;
  if (getenv("DEVICE") != NULL) {
    deviceIdx = atoi(getenv("DEVICE"));
  }
  cudaCheck(cudaSetDevice(deviceIdx));

  // print some device info
  CudaDeviceInfo();

  printf("\nRunning kernel %d on device %d.\n", kernel_num, deviceIdx);

  // Declare the handle, create the handle, cublasCreate will return a value of
  // type cublasStatus_t to determine whether the handle was created
  // successfully (the value is 0)
  cublasHandle_t handle;
  if (cublasCreate(&handle)) {
    std::cerr << "Create cublas handle error." << std::endl;
    exit(EXIT_FAILURE);
  };

  // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
  // publishing event tasks in the target stream
  float elapsed_time;
  cudaEvent_t beg, end;
  cudaCheck(cudaEventCreate(&beg));
  cudaCheck(cudaEventCreate(&end));

  // cuBLAS FLOPs ceiling is reached at 8192
  std::vector<int> SIZE = {16384};

  long m, n, k, max_size;
  max_size = SIZE[SIZE.size() - 1];
  std::cout << "Max size: " << max_size << std::endl;

  half alpha = 1.0, beta = 0.0; // GEMM input parameters, C=α*AB+β*C

  half *dA = nullptr, *dB = nullptr, *dC = nullptr,
        *dC_ref = nullptr; // device matrices

  cudaCheck(cudaMalloc((void **)&dA, sizeof(half) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dB, sizeof(half) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC, sizeof(half) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC_ref, sizeof(half) * max_size * max_size));

  // Initialize matrices with random values
  init_random_matrix(dA, max_size);
  init_random_matrix(dB, max_size);
  init_random_matrix(dC, max_size);
  
  // Copy dC to dC_ref
  cudaCheck(cudaMemcpy(dC_ref, dC, sizeof(half) * max_size * max_size,
                       cudaMemcpyDeviceToDevice));

  // Create a managed error flag
  int *errorFlagPtr = nullptr;
  cudaCheck(cudaMallocManaged((void **)&errorFlagPtr, sizeof(int)));

  int repeat_times = 1;
  for (int size : SIZE) {
    m = n = k = size;

    std::cout << "dimensions(m=n=k) " << m << ", alpha: " << __half2float(alpha)
              << ", beta: " << __half2float(beta) << std::endl;
    // Verify the correctness of the calculation, and execute it once before the
    // kernel function timing to avoid cold start errors
    if (kernel_num != 0) {
      run_kernel(0, m, n, k, alpha, dA, dB, beta, dC_ref,
                 handle); // cuBLAS
      run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC,
                 handle); // Executes the kernel, modifies the result matrix
      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaGetLastError()); // Check for async errors during kernel run

      if (!verify_matrix(dC_ref, dC, m * n, errorFlagPtr)) {
        std::cout
            << "Failed to pass the correctness verification against NVIDIA "
               "cuBLAS."
            << std::endl;
        exit(EXIT_FAILURE);
      }
    }

    if (kernel_num != 0) {
      cudaEventRecord(beg);
      for (int j = 0; j < repeat_times; j++) {
        // We don't reset dC between runs to save time
        run_kernel(0, m, n, k, alpha, dA, dB, beta, dC_ref, handle);
      }
      cudaEventRecord(end);
      cudaEventSynchronize(beg);
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&elapsed_time, beg, end);
      // elapsed_time /= 1000.; // Convert to seconds

      long flops = 2 * m * n * k;
      printf(
          "\ncuBLAS: Average elapsed time: (%7.6f) ms, performance: (%7.6f) TFLOPS. size: "
          "(%ld).\n",
          elapsed_time / repeat_times,
          (repeat_times * flops * 1e-9) / elapsed_time, m);
      fflush(stdout);
    }

    cudaEventRecord(beg);
    for (int j = 0; j < repeat_times; j++) {
      // We don't reset dC between runs to save time
      run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);
    // elapsed_time /= 1000.; // Convert to seconds

    long flops = 2 * m * n * k;
    printf(
        "\tAverage elapsed time: (%7.6f) ms, performance: (%7.6f) TFLOPS. size: "
        "(%ld).\n",
        elapsed_time / repeat_times,
        (repeat_times * flops * 1e-9) / elapsed_time, m);
    fflush(stdout);
    // make dC and dC_ref equal again (we modified dC while calling our kernel
    // for benchmarking)
    cudaCheck(cudaMemcpy(dC, dC_ref, sizeof(half) * m * n,
                         cudaMemcpyDeviceToDevice));
  }

  // Free up GPU space
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(dC_ref);
  cudaFree(errorFlagPtr);
  cublasDestroy(handle);

  return 0;
};