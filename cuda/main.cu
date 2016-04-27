#include "stdlib.h"
// #include "stdio.h"
#include <iostream>
#include <cuda_runtime.h>
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda.h>         // helper functions for CUDA error check

// #define DEBUG_MESSAGES_ON 1

#define GRID_SIZE   
#define BLOCK_SIZE  4


// Texture reference for 2D float texture
texture<float, 2, cudaReadModeElementType> tex;


// Kernel function to perform blurring
__global__ void blurKernel(float *outputData, int width, int height, int filterSize){
    extern __shared__ float window[];

    // calculate texture coordinates
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int tid=threadIdx.y*blockDim.y+ threadIdx.x;


    uint pixelRadius = filterSize >> 1;

    if(x <= pixelRadius || y <= pixelRadius || x >= (width-pixelRadius) || y >= (height-pixelRadius)){
        // Do nothing
    }
    else{
        uint x_start    = x - pixelRadius;
        uint x_end      = x + pixelRadius;
        uint y_start    = y - pixelRadius;
        uint y_end      = y + pixelRadius;

        int arraySize = filterSize * filterSize;
        // Create 2D to hold values in.  Each row is 

        // Fill array with values
        for(int i = x_start, ii=0 ; i <= x_end; i++, ii++){
            for(int j =  y_start, jj=0 ; j <= y_end ; j++, jj++){
                window[(tid * arraySize) + (ii * filterSize) + jj] = tex2D(tex, i, j);
            }
        }
        syncthreads();

        // Partial bubble sort from https://anisrahman.wordpress.com/2010/02/02/2d-median-filtering-using-cuda/
        int halfArraySize = (arraySize/2) + 1;
        for(int i = 0 ; i < halfArraySize ; i++){
            int min = i;
            for(int j = i + 1 ; j < arraySize ; j++){
                if(window[(tid * arraySize) + j] < window[(tid * arraySize) + min]){
                    min = j;
                }
            }
            // swap min into place
            float  temp = window[(tid * arraySize) + i];
            window[(tid * arraySize) + i]     = window[(tid * arraySize) + min];
            window[(tid * arraySize) + min]   = temp;
            syncthreads();

        }
        // get median
        outputData[(y * width) + x] = window[(tid * arraySize) + halfArraySize];

        // outputData[(y*width) + x] = tex2D(tex, x, y);
    }
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

    #if TEST_MODE
    std::cout << "In test mode!" << std::endl;
    #endif

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
    // char *outimagePath = sdkFindFilePath(outputfile, argv[0]);
    // if (outimagePath == NULL)
    // {
    //     #if DEBUG_MESSAGES_ON
    //     std::cout << "Unable to source image file:"<< outputfile << "\n" << std::endl;
    //     #endif
    //     exit(EXIT_FAILURE);
    // }
    // Sanity check to make sure file is loaded in correctly
    // TODO: either file is not loaded correctly, or it is not saving correctly
    // sdkSavePGM(outimagePath, origData, width, height);

    int size = width * height * sizeof(float);
    #if DEBUG_MESSAGES_ON
    std::cout << "Loaded " << inputfile << ", " << width << " x "<< height << " pixels with size " << (uint)size << std::endl;
    #endif

    // Allocate device memory for result
    float *outData = NULL;
    checkCudaErrors(cudaMalloc((void **) &outData, size));

    // Allocate array and copy image data
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
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

    #if DEBUG_MESSAGES_ON
    uint pixelRadius = filterSize >> 1;
    std::cout << "Pixel radius is " << pixelRadius << std::endl;
    #endif
    // Set up grid
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

    #if DEBUG_MESSAGES_ON
    std::cout << "\n\nBlocks and grid set up.\nBlock is " << dimBlock.x << " x " << dimBlock.y << \
        "\nGrid is " << dimGrid.x << " x " << dimGrid.y << std::endl;
    #endif

    int window_size = BLOCK_SIZE * BLOCK_SIZE * filterSize * filterSize * sizeof(float);
    getLastCudaError("Before Kernel execution");

    #if DEBUG_MESSAGES_ON
    std::cout << "Window size is: " << window_size << std::endl;
    #endif

    blurKernel<<<dimGrid, dimBlock, window_size>>>(outData, width, height, filterSize);
    getLastCudaError("Kernel execution failed");


    // Allocate mem for the result on host side
    float *hOutputData = (float *) malloc(size);
    // copy result from device to host
    checkCudaErrors(cudaMemcpy(hOutputData,
                               (const void*)outData,
                               size,
                               cudaMemcpyDeviceToHost));
    // Write to file
    char *outimagePath = sdkFindFilePath(outputfile, argv[0]);
    if (outimagePath == NULL)
    {
        #if DEBUG_MESSAGES_ON
        std::cout << "Unable to source image file:"<< outputfile << "\n" << std::endl;
        #endif
        exit(EXIT_FAILURE);
    }
    sdkSavePGM(outimagePath, hOutputData, width, height);
    #if DEBUG_MESSAGES_ON
    std::cout << "Wrote to " << outputfile << "." << std::endl;
    #endif



    return 0;
}












