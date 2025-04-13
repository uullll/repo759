#include "matmul.cuh"
#include<cstdio>

using namespace std;

__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block){
    dim3 threads(threads_per_block, threads_per_block);
    dim3 blocksPerGrid((n + threads_per_block-1) / threads_per_block, (n + threads_per_block-1) / threads_per_block);

    matmul_kernel<<<blocksPerGrid, threads>>>(A, B, C, n);
    cudaDeviceSynchronize();
}
