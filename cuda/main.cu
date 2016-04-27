#include "stdlib.h"
// #include "stdio.h"
#include <iostream>
#include <cuda_runtime.h>
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda.h>         // helper functions for CUDA error check

#define DEBUG_MESSAGES_ON 1

#define GRID_SIZE   
#define BLOCK_SIZE  32


// const char *imageFilename = "lena.ppm";
// const char *refFilename   = "lena_blurred.ppm";


// Texture reference for 2D float texture
texture<unsigned char, 2, cudaReadModeElementType> tex;


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
        #if DEBUG_MESSAGES_ON
        std::cout << "\n\nIncorrect number of args.  Should be \n(1)blur filter size {3, 7, 11, or 15}\n(2)input file\n(3)output file" << std::endl;
        #endif
        return 0;
    }
    // else{
        // std::cout << "Good job!" << std::endl;
    // }

    // Get filter buffer size
    int filterSize = atoi(argv[1]);
    if( !(filterSize==3 || filterSize==7 || filterSize==11 || filterSize==15) ){
        #if DEBUG_MESSAGES_ON
        std::cout << "Incorrect input for filter size.\nMust be {3, 7, 11, or 15}" << std::endl;
        #endif
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
    float *origData = NULL;
    unsigned int width, height;
    char *imagePath = sdkFindFilePath(inputfile, argv[0]);

    if (imagePath == NULL)
    {
        #if DEBUG_MESSAGES_ON
        std::cout << "Unable to source image file:"<< inputfile << " %s\n" << std::endl;
        #endif
        exit(EXIT_FAILURE);
    }

    sdkLoadPGM(imagePath, &origData, &width, &height);

    unsigned char size = width * height * sizeof(unsigned char);
    #if DEBUG_MESSAGES_ON
    std::cout << "Loaded " << inputfile << ", " << width << " x "<< height << " pixels with size " << size << std::endl;
    #endif

    // //Load reference image from image (output)
    // float *outData = (float *) malloc(size);
    // char *outPath = sdkFindFilePath(outputfile, argv[0]);

    // if (outPath == NULL)
    // {
    //     std::cout << "Unable to find reference image file: " << outputfile << "\n" << std::endl;
    //     exit(EXIT_FAILURE);
    // }

    // sdkLoadPGM(outPath, &outData, &width, &height);


    // Allocate device memory for result
    float *outData = NULL;
    checkCudaErrors(cudaMalloc((void **) &outData, size));

    // Allocate array and copy image data
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    cudaArray *inArray;
    checkCudaErrors(cudaMallocArray(&inArray,
                                    &channelDesc,
                                    width,
                                    height));
    checkCudaErrors(cudaMemcpyToArray(inArray,
                                      0,
                                      0,
                                      origData,
                                      size,
                                      cudaMemcpyHostToDevice));

    #if DEBUG_MESSAGES_ON
    std::cout << "\n\nLoaded " << inputfile << " onto device." << std::endl;
    #endif

    // Set texture parameters
    tex.addressMode[0] = cudaAddressModeBorder;
    tex.addressMode[1] = cudaAddressModeBorder;
    tex.filterMode = cudaFilterModePoint;
    tex.normalized = false;    // access with normalized texture coordinates

    // Bind the array to the texture
    checkCudaErrors(cudaBindTextureToArray(tex, inArray, channelDesc));

    uint pixelRadius = filterSize >> 1;
    #if DEBUG_MESSAGES_ON
    std::cout << "Picxel radius is " << pixelRadius << std::endl;
    #endif
    // Set up grid
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

    #if DEBUG_MESSAGES_ON
    std::cout << "\n\nBlocjks and grid set up.\nBlock is " << dimBlock.x << " x " << dimBlock.y << \
        "\nGrid is " << dimGrid.x << " x " << dimGrid.y << std::endl;
    #endif

    blurKernel<<<dimGrid, dimBlock, 0>>>(outData, width, height, filterSize);



    return 1;
}












