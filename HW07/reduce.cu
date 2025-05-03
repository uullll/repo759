#include "reduce.cuh"
#include <cuda_runtime.h>
#include <iostream>
using namespace std;

__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n){
    extern __shared__ float shared[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    float sum = 0;
    if (idx<n) {
        sum= g_idata[idx];
        if(idx + blockDim.x < n) {
            sum += g_idata[idx + blockDim.x];
        }       
    }
    shared[tid] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    if (tid == 0)
        g_odata[blockIdx.x] = shared[0];
    
}
__host__ void reduce(float **input, float **output, unsigned int N,
    unsigned int threads_per_block){
    unsigned int blocks = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);
    unsigned int shared_mem_size = threads_per_block * sizeof(float);

    while (blocks > 1) {
        // Launch the kernel
        reduce_kernel<<<blocks, threads_per_block, shared_mem_size>>>(*input, *output, N);
        cudaDeviceSynchronize();

        // Swap input and output for the next iteration
        float *temp = *input;
        *input = *output;
        *output = temp;

        // Update N and calculate new number of blocks
        N = blocks;
        blocks = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);
    }

    // Final reduction to a single value
    reduce_kernel<<<1, threads_per_block, shared_mem_size>>>(*input, *output, N);
    cudaDeviceSynchronize();
}