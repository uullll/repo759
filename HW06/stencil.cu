#include "stencil.cuh"
#include <cstdio>

__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R){
    extern __shared__ float shared_data[];
    // Allocate shared memory for mask and image
    float* shared_image = shared_data;
    float* shared_mask = &shared_data[blockDim.x + 2*R];
    float* shared_output = &shared_data[blockDim.x + 4*R + 1];
    // Load mask into shared memory
    if (threadIdx.x < 2 * R + 1) {
        shared_mask[threadIdx.x] = mask[threadIdx.x];
    }
    __syncthreads();

    // Load image data into shared memory
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    int shared_index = threadIdx.x + R ;
    
    if (threadIdx.x < R) {
        int left_idx = global_index - R;
        int right_idx = global_index + R;

        //0 til n-1
        if (left_idx >= 0){
            shared_image[threadIdx.x] = image[left_idx];
        }
        else{
            shared_image[threadIdx.x] = 1.0;
        }
        if (right_idx < n){
            shared_image[shared_index + blockDim.x] = image[right_idx];
        }
        else{
            shared_image[shared_index + blockDim.x] = 1.0;
        }
    }
    if(global_index < n){
        shared_image[shared_index] = image[global_index];
    }
    __syncthreads();

    // Compute the convolution
    if (global_index < n) {
        float result = 0.0f;
        
        for (int i = -(int)R; i <= (int)R; ++i) {
            result+=shared_image[shared_index+i]*shared_mask[i+R];
        }
           
        
        
        shared_output[threadIdx.x] = result;
    }
    __syncthreads();
    if(global_index < n){
        output[global_index] = shared_output[threadIdx.x];
    }
}

__host__ void stencil(const float* image,
                      const float* mask,
                      float* output,
                      unsigned int n,
                      unsigned int R,
                      unsigned int threads_per_block){
    size_t shared_memory_size = (2 * R + 1 + threads_per_block + 2 * R+threads_per_block) * sizeof(float);
    stencil_kernel<<<(n + threads_per_block - 1) / threads_per_block, threads_per_block, shared_memory_size>>>(image, mask, output, n, R);
    cudaDeviceSynchronize();
}