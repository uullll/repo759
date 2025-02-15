#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include "matmul.h"
#include <cstring>  


using namespace std;
using namespace chrono;

#define N 1024 // matrix size 

void generateRandomMatrix_vec(vector<double> &matrix, int n) {
    for (int i = 0; i < n * n; i++) {
        matrix[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
    }
}

void generateRandomMatrix(double *matrix, int n) {
    for (int i = 0; i < n * n; i++) {
        matrix[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
    }
}

int main() {
    int n = 1024;  // Set the matrix size

    double *A     = (double*)malloc(N*N*sizeof(double)); // input_A
	double *B     = (double*)malloc(N*N*sizeof(double)); // input_B
	double *C     = (double*)malloc(N*N*sizeof(double)); // output
	
    vector<double> A_vec(n * n), B_vec(n * n);

    generateRandomMatrix_vec(A_vec, n);
    generateRandomMatrix_vec(B_vec, n);

    generateRandomMatrix(A,N);
    generateRandomMatrix(B,N);
    // Test the four matrix multiplication implementations
    for (int method = 1; method <= 4; method++) {
        memset(C, 0, sizeof(C));  // Reset C to zero before each test
        auto start = high_resolution_clock::now();

        if (method == 1) mmul1(A, B, C, n);
        else if (method == 2) mmul2(A, B, C, n);
        else if (method == 3) mmul3(A, B, C, n);
        else mmul4(A_vec, B_vec, C, n);

        auto end = high_resolution_clock::now();
        double elapsedTime = duration<double, milli>(end - start).count();

        // Output execution time and the last element of matrix C
        cout << elapsedTime << endl;
        cout << C[n * n - 1] << endl;
    }


    free (A);
	free (B);
	free (C);
	
	A_vec.clear();
	B_vec.clear();
    return 0;
}


