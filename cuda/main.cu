#include "stdlib.h"
// #include "stdio.h"
#include <iostream>
#include <cuda_runtime.h>
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda.h>         // helper functions for CUDA error check


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
        return 0;
    }
    // else{
        // std::cout << "Good job!" << std::endl;
    // }

    // Get filter buffer size
    int filterSize = atoi(argv[1]);
    if( !(filterSize==3 || filterSize==7 || filterSize==11 || filterSize==15) ){
        std::cout << "Incorrect input for filter size.\nMust be {3, 7, 11, or 15}" << std::endl;
        return 0;
    }

    char *inputfile = argv[2];
    if(!inputfile){
        return 0;
    }
    char *outputfile = argv[3];
    if(!outputfile){
        return 0;
    }


    // Load PGM onto device
    int devID = findCudaDevice(argc, (const char **) argv);

    // load image from disk
    float *hData = NULL;
    unsigned int width, height;
    char *imagePath = sdkFindFilePath(imageFilename, inputfile);

    if (imagePath == NULL)
    {
        std::cout << "Unable to source image file:"<< imageFilename << " %s\n" << std::endl;
        exit(EXIT_FAILURE);
    }

    sdkLoadPGM(imagePath, &hData, &width, &height);

    unsigned int size = width * height * sizeof(float);
    printf("Loaded '%s', %d x %d pixels\n with size %d", imageFilename, width, height, size);

    return 1;
}












