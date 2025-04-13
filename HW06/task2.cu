#include "stencil.cuh"
#include <cstdio>
#include <iostream>
#include <random>
#include <cstdlib>
#include <ctime>
#include <cuda.h>
using namespace std;
float* Ramdon_Value(int Min,int Max,int n){
    float *dA=new float[n];
    random_device entropy_source;
    mt19937_64 generator(entropy_source());
    uniform_real_distribution<float> dist(Min, Max);

    for (int i = 0; i < n; ++i) {
        dA[i] = dist(generator);  
    }
    return dA;
}
int main(int argc, char *argv[]){
    if(argc!=4){
        cerr << "Usage: " << argv[0] << " <number_of_size><radiu><threads per block>" << endl;
        return 1;
    }
    int n=atoi(argv[1]);
    int R=atoi(argv[2]);
    int threads_per_block=atoi(argv[3]);
    size_t size = n *sizeof(float);
    float *h_image=Ramdon_Value(-1.0,1.0,n);
    float *h_mask=Ramdon_Value(-1.0,1.0,2*R+1);
    float *h_output=new float[n];

    
    float *d_image, *d_mask, *d_output;
    cudaMalloc(&d_image,n *sizeof(float));
    cudaMalloc(&d_mask,(2 * R + 1) * sizeof(float));
    cudaMalloc(&d_output,n *sizeof(float));

    cudaMemcpy(d_image,h_image,n *sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask,h_mask,(2 * R + 1) * sizeof(float),cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    

    stencil(d_image,d_mask,d_output,n,R,threads_per_block);
    

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaMemcpy(h_output,d_output,size,cudaMemcpyDeviceToHost);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cout<<h_output[n-1]<<endl;
    
    cout <<milliseconds <<endl;

    delete[] h_image;
    delete[] h_mask;
    delete[] h_output;
    cudaFree(d_image);
    cudaFree(d_mask);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;

}