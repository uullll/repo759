#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda.h>
#include <iostream>
#include <random>
#include "vscale.cuh"
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
int main(int argc, char** argv){

    if (argc < 2) {
        cerr << "Usage: ./task3 <n>" << endl;
        return 1;
    }

    int N = stoi(argv[1]);
    int thread_per_clock = (argc > 2) ? stoi(argv[2]) : 512;
    float *a=Ramdon_Value(-10.0,10.0,N);
    float *b=Ramdon_Value(0.0,1.0,N);
    

    float *ga,*gb;
    cudaMalloc(&ga,N*sizeof(float));
    cudaMalloc(&gb,N*sizeof(float));
    

    cudaMemcpy(a,ga,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(b,gb,N*sizeof(float),cudaMemcpyHostToDevice);

    

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    
    int num_blocks=(N+thread_per_clock-1)/thread_per_clock;

    cudaEventRecord(start);


    vscale<<<num_blocks,thread_per_clock>>>(ga,gb,N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds=0;
    cudaEventElapsedTime(&milliseconds,start,stop);
    cudaMemcpy(b,gb,N*sizeof(float),cudaMemcpyDeviceToHost);
    cout << "Kernel execution time: " << milliseconds << " ms" << endl;
    cout << "First element of result: " << b[0] << endl;
    cout << "Last element of result: " << b[N - 1] << endl;

    delete[] a;
    delete[] b;
    
    cudaFree(ga);
    cudaFree(gb);
    

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}