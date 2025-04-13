#include<cstdio>
#include "matmul.cuh"
#include <iostream>
#include <random>
#include <cstdlib>
#include <ctime>
#include <cuda.h>
using namespace std;

__global__ void task1kernel(float * array, int n){
    int idx=threadIdx.x+blockIdx.x*blockDim.x;
    if (idx<n){
        array[idx]=idx;
    }
}
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
    if(argc!=3){
        cerr << "Usage: " << argv[0] << " <number_of_size><threads per block>" << endl;
        return 1;
    }
    int n=stoi(argv[1]);
    unsigned int threads_per_block=stoi(argv[2]);
    size_t size = n *n *sizeof(float);
    float *ha=Ramdon_Value(-1.0,1.0,n*n);
    float *hb=Ramdon_Value(-1.0,1.0,n*n);
    float *hc=new float[n*n];

    float *da, *db, *dc;
    cudaMalloc(&da,size);
    cudaMalloc(&db,size);
    cudaMalloc(&dc,size);
    cudaMemcpy(da,ha,size,cudaMemcpyHostToDevice);
    cudaMemcpy(db,hb,size,cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    matmul(da,db,dc,n,threads_per_block);
    cudaMemcpy(hc,dc,size,cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout<<hc[n*n-1]<<endl;
    cout<< milliseconds <<endl;
    
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    delete[] ha;
    delete[] hb;
    delete[] hc;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}