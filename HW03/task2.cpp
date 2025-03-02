#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include "convolution.h"

using namespace std;
using namespace chrono;

void generateRandomMatrix(float *matrix, int size, float min_val, float max_val) {
    for (int i = 0; i < size; ++i) {
        matrix[i] = min_val + static_cast<float>(rand()) / RAND_MAX * (max_val - min_val);
    }
    
}

int main(int argc, char *argv[]){

    if (argc != 3) {
        cerr << "Usage: ./task2 <n> <t>" << endl;
        return 1;
    }

    int n = atoi(argv[1]);  
    int t = atoi(argv[2]);
    if (n < 3) {
        cerr << "Error: n must be positive integers." << endl;
        return 1;
    }
    srand(time(nullptr)); 
    float* image = new float[n*n];
    float* mask  = new float[3*3];
    float* output= new float[(n-2)*(n-2)]();
    generateRandomMatrix(image, n * n, -10.0f, 10.0f);
    generateRandomMatrix(mask, 3 * 3, -1.0f, 1.0f);

    omp_set_num_threads(t);
    auto start = high_resolution_clock::now();
    convolve(image, output,n ,mask , 3);
    auto end = high_resolution_clock::now();
    double elapsedTime = duration<double, milli>(end - start).count();


    cout<<output[0]<<endl;
    cout<<output[(n-2)*(n-2)-1]<<endl;
    cout<<elapsedTime<<endl;

    //delete[] image; image= nullptr;
	//delete[] mask;  mask= nullptr;
	//delete[] output; output= nullptr;
    return 0;
}