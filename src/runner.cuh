#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

void cudaCheck(cudaError_t error, const char *file,
               int line); // CUDA error check
void CudaDeviceInfo();    // print CUDA information

bool verify_matrix(half *mat1, half *mat2, int N);

float cpu_elapsed_time(float &beg, float &end); // Calculate time difference

void run_kernel(int kernel_num, int m, int n, int k, half alpha, half *A,
                half *B, half beta, half *C, cublasHandle_t& handle);

void init_random_matrix(half *mat, int num_elements);