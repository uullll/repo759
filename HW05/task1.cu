#include<cstdio>
// CUDA kernel function to compute factorial
__global__ void factorialKernel(){
    int a = threadIdx.x + 1; // a from 1 to 8
    int b = 1;

    for (int i = 1; i <= a; ++i) {
        b *= i;
    }
    if(a%2==0)
        printf("%d!=%d\n", a, b);
}
int main(){
    factorialKernel<<<1,8>>>();
    cudaDeviceSynchronize();

    return 0;
}