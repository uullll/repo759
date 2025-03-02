#include <cstddef>
#include <omp.h>
#include<algorithm>
#include<iostream>
#include<vector>
#include"msort.h"
using namespace std;
void merge(int* arr, int left, int mid, int right);
void parallelMergeSort(int* arr,int left,int right,std::size_t threshold);
void msort(int* arr, const std::size_t n, const std::size_t threshold){
    #pragma omp parallel
    {
        #pragma omp single
        parallelMergeSort(arr,0,n-1,threshold);
    }

}
void merge(int* arr, int left, int mid, int right){
    int n1 = mid - left + 1;
    int n2 = right - mid;

    vector<int> L(n1);
    vector<int> R(n2);

    for (int i = 0; i < n1; i++) L[i] = arr[left + i];
    for (int i = 0; i < n2; i++) R[i] = arr[mid + 1 + i];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    }
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
}
void parallelMergeSort(int* arr,int left,int right,std::size_t threshold){
    if(right-left+1<=threshold){
        sort(arr+left,arr+right+1);
        return;
    }
    int mid=left+(right-left)/2;
    #pragma omp task shared(arr)
    parallelMergeSort(arr,left,mid,threshold);

    #pragma omp task shared(arr)
    parallelMergeSort(arr,mid+1,right,threshold);

    #pragma omp taskwait
    merge(arr,left,mid,right);
}