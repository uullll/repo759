/** i)      Creates an n×n image matrix (stored in 1D in row-major order) of random float numbers
            between -10.0 and 10.0. The value of n should be read as the first command line argument.
    ii)     Creates an m×m mask matrix (stored in 1D in row-major order) of random float numbers
            between -1.0 and 1.0. The value of m should be read as the second command line argument.
    iii)    Applies the mask to image using your convolve function.
    iv)     Prints out the time taken by your convolve function in milliseconds.
    v)      Prints the first element of the resulting convolved array.
    vi)     Prints the last element of the resulting convolved array.
    vii)    Deallocates memory when necessary via the delete function.
 */

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

int main(int argc, char *argv[]) {
    if (argc != 3) {
        cerr << "Usage: ./task2 <n> <m>" << endl;
        return 1;
    }

    int n = atoi(argv[1]);  // 读取图像大小
    int m = atoi(argv[2]);  // 读取卷积核大小
    if (n <= 0 || m <= 0 || m % 2 == 0) {
        cerr << "Error: n and m must be positive integers, and m must be odd." << endl;
        return 1;
    }

    srand(time(nullptr));  // 随机种子
    float* image  = new float[n*n];
	float* mask   = new float[m*m];
	float* output = new float[n*n];

    generateRandomMatrix(image, n * n, -10.0f, 10.0f);
    generateRandomMatrix(mask, m * m, -1.0f, 1.0f);

    // 计时
    auto start = high_resolution_clock::now();
    convolve(image, output,n ,mask , m);
    auto end = high_resolution_clock::now();
    double elapsedTime = duration<double, milli>(end - start).count();

    // 输出时间、卷积结果的第一个和最后一个值
    cout << elapsedTime << endl;
    cout << output[0] << endl;
    cout << output[n * n - 1] << endl;
    delete[] image;
	delete[] mask;
	delete[] output;
    return 0;
}