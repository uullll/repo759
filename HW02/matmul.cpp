#include "matmul.h"

#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <stdlib.h>

//i,j,k
void mmul1(const double *A, const double *B, double *C, const unsigned int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];  // 计算 C[i, j] = A[i, :] · B[:, j]
            }
            C[i * n + j] = sum;
        }
    }
}

//i,k,j
void mmul2(const double *A, const double *B, double *C, const unsigned int n) {
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            double a_ik = A[i * n + k];  // 预取 A[i, k]
            for (int j = 0; j < n; j++) {
                C[i * n + j] += a_ik * B[k * n + j];
            }
        }
    }
}
//j,k,i
void mmul3(const double *A, const double *B, double *C, const unsigned int n) {
    for (int j = 0; j < n; j++) {
        for (int k = 0; k < n; k++) {
            double b_kj = B[k * n + j];  // 预取 B[k, j]
            for (int i = 0; i < n; i++) {
                C[i * n + j] += A[i * n + k] * b_kj;
            }
        }
    }
}
//vector
void mmul4(const std::vector<double> &A, const std::vector<double> &B, double* C, const unsigned int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}