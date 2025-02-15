#include <stdio.h>
#include <stdlib.h>
#include "convolution.h"
float computeConvolution(const float *image, const float *kernel, int x, int y, int n, int m);
void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m){
    for (int x = 0; x < n; x++) {
        for (int y = 0; y < n; y++) {
            output[x * n + y] = computeConvolution(image, mask, x, y, n, m);
        }
    }
}
float computeConvolution(const float *image, const float *kernel, int x, int y, int n, int m) {
    int offset = m / 2;
    float sum = 0.0f;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            int img_x = x + i - offset;
            int img_y = y + j - offset;

            
            float pixel_value;
                    if (img_x < 0 || img_x >= n || img_y < 0 || img_y >= n) {
                        
                        if (img_x < 0 || img_x >= n || img_y < 0 || img_y >= n) {
                            if (img_x < 0 || img_x >= n || img_y < 0 || img_y >= n)
                                pixel_value = 0;  
                            else
                                pixel_value = 1;  
                        }
                    } else {
                        pixel_value = image[img_x * n + img_y];  
                    }

            sum += pixel_value * kernel[i * m + j];
        }
    }

    return sum;
}