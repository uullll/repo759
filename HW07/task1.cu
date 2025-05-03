#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#include "matmul.cuh"
#include <random>
#include <cstdlib>
#include <ctime>
#include <cuda.h>
#include <chrono>
using namespace std;
template <typename T>

__global__ void task1kernel(float * array, int n){
    int idx=threadIdx.x+blockIdx.x*blockDim.x;
    if (idx<n){
        array[idx]=idx;
    }
}
template <typename T>
T *CreateMatrix(int n){
    T *matrix = new T[n];
    random_device entropy_source;
    mt19937_64 generator(entropy_source());
    uniform_real_distribution<T> dist(-10, 10);

    for (int i = 0; i < n; ++i) {
        matrix[i] = dist(generator);  
    }
    return matrix;
}
int *CreateMatrix_1(int n){
    int *matrix = new int[n];
    random_device entropy_source;
    mt19937_64 generator(entropy_source());
    uniform_int_distribution<int> dist(-10, 10);

    for (int i = 0; i < n; ++i) {
        matrix[i] = dist(generator);  
    }
    return matrix;
}
template <typename T>
T *ConvertMatrix(int *matrix, int n){
    T *new_matrix = new T[n];
    for (int i = 0; i < n; ++i) {
        new_matrix[i] = matrix[i];  
    }
    return new_matrix;
}
int main(int argc, char *argv[]){
    if(argc!=3){
        cerr << "Usage: " << argv[0] << " <number_of_size><block_dim>" << endl;
        return 1;
    }
    unsigned int n=atoi(argv[1]);
    unsigned int block_dim=atoi(argv[2]);
    size_t size = n *n *sizeof(float);
    
    int *ha_1=CreateMatrix_1(n*n);
    int *hb_1=CreateMatrix_1(n*n);
    int *hc_1=new int[n*n];

    float *ha_2=CreateMatrix<float>(n*n);
    float *hb_2=CreateMatrix<float>(n*n);
    float *hc_2=new float[n*n];

    double *ha_3=CreateMatrix<double>(n*n);
    double *hb_3=CreateMatrix<double>(n*n);
    double *hc_3=new double[n*n];
    
    int *da_1, *db_1, *dc_1;
    float *da_2, *db_2, *dc_2;
    double *da_3, *db_3, *dc_3;
    cudaMalloc(&da_1,size);
    cudaMalloc(&db_1,size);
    cudaMalloc(&dc_1,size);
    cudaMalloc(&da_2,size);
    cudaMalloc(&db_2,size);
    cudaMalloc(&dc_2,size);
    cudaMalloc(&da_3,size);
    cudaMalloc(&db_3,size);
    cudaMalloc(&dc_3,size);
    cudaMemcpy(da_1,ha_1,size,cudaMemcpyHostToDevice);
    cudaMemcpy(db_1,hb_1,size,cudaMemcpyHostToDevice);
    cudaMemcpy(da_2,ha_2,size,cudaMemcpyHostToDevice);
    cudaMemcpy(db_2,hb_2,size,cudaMemcpyHostToDevice);
    cudaMemcpy(da_3,ha_3,size,cudaMemcpyHostToDevice);
    cudaMemcpy(db_3,hb_3,size,cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, 0);
    matmul_1(da_1,db_1,dc_1,n,block_dim);
    cudaMemcpy(hc_1,dc_1,size,cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cout<<hc_1[0]<<endl<<hc_1[n*n-1]<<endl;
    cout<< milliseconds <<endl;


    
    cudaEventRecord(start, 0);
    matmul_2(da_2,db_2,dc_2,n,block_dim);
    cudaMemcpy(hc_2,dc_2,size,cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout<<hc_2[0]<<endl<<hc_2[n*n-1]<<endl;
    cout<< milliseconds <<endl;

    
    cudaEventRecord(start, 0);
    matmul_3(da_3,db_3,dc_3,n,block_dim);
    cudaMemcpy(hc_3,dc_3,size,cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout<<hc_3[0]<<endl<<hc_3[n*n-1]<<endl;
    cout<< milliseconds <<endl;


    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(da_1);
    cudaFree(db_1);
    cudaFree(dc_1);
    cudaFree(da_2);
    cudaFree(db_2);
    cudaFree(dc_2);
    cudaFree(da_3);
    cudaFree(db_3);
    cudaFree(dc_3);
    delete[] ha_1;
    delete[] hb_1;
    delete[] hc_1;
    delete[] ha_2;
    delete[] hb_2;
    delete[] hc_2;
    delete[] ha_3;
    delete[] hb_3;
    delete[] hc_3;
    return 0;      
}