#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include "matmul.h"
#include <cstring>  
#include <omp.h> 
using namespace std;
using namespace chrono;
#define N 1024

void generateRandomMatrix(float *matrix, int n) {
    for (int i = 0; i < n * n; i++) {
        matrix[i] = (float)rand() / RAND_MAX * 2.0 - 1.0;
    }
}

int main(int argc, char *argv[]){
    if (argc != 3) {
        cerr << "Usage: ./task2 <n> <t>" << endl;
        return 1;
    }

    int n = atoi(argv[1]);  
    int t = atoi(argv[2]);
    if(n<=0||t<1||t>20){
        cerr<<"Error:n should be positive and t should be in range [1,20]"<<endl;
        return 1;
    }

    srand(time(nullptr));
    float* A = new float[n*n];
    float* B = new float[n*n];
    float* C = new float[n*n]();
    auto start = high_resolution_clock::now();
    generateRandomMatrix(A,n);
    generateRandomMatrix(B,n);

    omp_set_num_threads(t);
    mmul(A,B,C,n);
    auto end = high_resolution_clock::now();
    double elapsedTime = duration<double, milli>(end - start).count();
    cout<<C[0]<<endl;
    
    cout<<C[n*n-1]<<endl;
    cout<<elapsedTime<<endl;
    delete[]A;
    delete[]B;
    delete[]C;
    return 0;
}