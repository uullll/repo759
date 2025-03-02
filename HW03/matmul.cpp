#include <cstddef>
#include <omp.h>
#include "matmul.h"

#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <stdlib.h>
void mmul(const float* A, const float* B, float* C, const std::size_t n){
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        for (int k = 0; k < n; k++)
        {
            double a_ik = A[i*n+k];
            for(int j=0;j<n;j++){
                C[i*n+j]+=a_ik*B[k*n+j];
            }
        }
        
    }
    

}