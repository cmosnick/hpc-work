#include "stdlib.h"
// #include "stdio.h"
#include <iostream>
#include <cuda_runtime.h>


const char *imageFilename = "lena.ppm";
const char *refFilename   = "lena_blurred.ppm";


// Texture reference for 2D float texture
texture<float, 2, cudaReadModeElementType> tex;


// Kernel function to perform blurring
__global__ void blurKernel(float *outputData, int width, int height, int filterSize){
    // calculate normalized texture coordinates
    // unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    // unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    // float u = (float)x - (float)width/2; 
    // float v = (float)y - (float)height/2; 
    // float tu = u*cosf(theta) - v*sinf(theta); 
    // float tv = v*cosf(theta) + u*sinf(theta); 

    // tu /= (float)width; 
    // tv /= (float)height; 

    // // read from texture and write to global memory
    // outputData[y*width + x] = tex2D(tex, tu+0.5f, tv+0.5f);
}


int main(int argc, char **argv){
    // Check args
    if(argc < 3){
        std::cout << "\n\nIncorrect number of args.  Should be \n(1)blur filter size {3, 7, 11, or 15}\n(2)input file\n(3)output file" << std::endl;
    }

    else{
        std::cout << "Good job!" << std::endl;
    }

}