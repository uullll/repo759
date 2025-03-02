#include<iostream>
#include <omp.h> 
#include"msort.h"
#include <chrono>
#include <cstdlib>
#include <ctime>
using namespace std;
using namespace chrono;
#define MIN -1000
#define MAX 1000
void generateRandomarray(int *arr,int size){
    for (int i = 0; i < size; i++)
    {
        arr[i]=MIN + rand() % (MAX - MIN + 1);
    }
    
}

int main(int argc, char *argv[]){
    if (argc != 4) {
        cerr << "Usage: ./task2 <n> <m>" << endl;
        return 1;
    }

    int n = atoi(argv[1]);  
    int t = atoi(argv[2]); 
    int ts= atoi(argv[3]); 

    srand(time(nullptr));
    int* arr = new int[n];
    generateRandomarray(arr,n);

    omp_set_num_threads(t);

    auto start = high_resolution_clock::now();
    msort(arr,n,ts);
    auto end = high_resolution_clock::now();
    double elapsedTime = duration<double, milli>(end - start).count();

    cout<<arr[0]<<endl;
    cout<<arr[n-1]<<endl;
    cout<<elapsedTime<<endl;

    return 0;
}