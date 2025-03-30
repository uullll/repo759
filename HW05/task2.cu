#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda.h>

int Ramdom_value(){
    const int RANGE = 10;
    int a = rand() % (RANGE + 1);
    return a;
}

__global__ void Addlinear(int *dA,int a){
    int x=threadIdx.x;
    int y=blockIdx.y;
    
    

    int Idx=y*blockDim.x+x;
    dA[Idx]=a*x+y;
}
int main(){
    int *dA;
    cudaMalloc(&dA,16*sizeof(int));
    int a=Ramdom_value();
    Addlinear<<<2,8>>>(dA,a);
    int hA[16];
    cudaMemcpy(hA,dA,16*sizeof(int),cudaMemcpyDeviceToHost);

    for(int i=0;i<16;++i){
        printf("%d ",hA[i]);
        printf("\n");
    }
    cudaFree(dA);
    return 0;
}