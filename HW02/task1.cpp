#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include "scan.h"
using namespace std;
using namespace chrono;
void generateRandomArray(vector<float> &arr, int n) {
    for (int i = 0; i < n; ++i) {
        arr[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  
    }
}
int main(int argc,char* argv[]){
    if (argc != 2) {
        cerr << "Usage: ./task1 <n>" << endl;
        return 1;
    }
    int n = atoi(argv[1]);
    if (n <= 0) {
        cerr << "Error: n must be a positive integer." << endl;
        return 1;
    }
    srand(time(nullptr));  
    vector<float> inputArray(n);
    vector<float> outputArray(n);

    generateRandomArray(inputArray, n);  // 生成随机数组

    
    auto start = high_resolution_clock::now();

    
    scan(inputArray.data(), outputArray.data(), n);

    
    auto end = high_resolution_clock::now();
    double elapsedTime = duration<double, milli>(end - start).count();

    
    cout << elapsedTime << endl;
    
    cout << outputArray[0] << endl;
    cout << outputArray[n - 1] << endl;

    return 0;
}