#include "stdlib.h"
#include <iostream>
#include <cuda_runtime.h>
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda.h>         // helper functions for CUDA error check


#define GRID_SIZE   
#define BLOCK_SIZE  4

typedef unsigned int uint;
void createGoldenStandard( float *origData, float *standData, unsigned int width, unsigned int height, uint filterSize);
float compareToStandard( float *standData, float *testData, uint width, uint height);

const char *statsFileName = "out_stats.txt";

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
        outputData[(y * width) + x] = tex2D(tex, x, y);
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
        for(int i = 0 ; i <= halfArraySize ; i++){
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
    #if TEST_MODE
    std::cout << "Testing " << filterSize << " pixel filter with " << BLOCK_SIZE << " x " << BLOCK_SIZE << " blocks" << std::endl;
    #endif
    // Load PGM onto device
    int devID = findCudaDevice(argc, (const char **) argv);

    #if TEST_MODE
    // Open statistics file
    // char *statsPath = sdkFindFilePath(statsFileName, argv[0]);
    // if(statsPath == NULL){
    //     std::cout<< "Could not find stats file" << std::endl;
    // }

    FILE *statsFile = fopen(statsFileName, "a");
    if(!statsFile){
        std::cout << "Couldn't open stats file" << std::endl;
    }
    #endif

    /***************
    LOAD INPUT FILE
    ****************/
    float *origData = NULL;
    unsigned int width, height;
    char *imagePath = sdkFindFilePath(inputfile, argv[0]);

    if (imagePath == NULL){
        #if DEBUG_MESSAGES_ON
        std::cout << "Unable to source image file:"<< inputfile << " %s\n" << std::endl;
        #endif
        exit(EXIT_FAILURE);
    }
    sdkLoadPGM(imagePath, &origData, &width, &height);
    
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


    /************
    SET UP TEXTURE AND GRID INFORMATION
    *************/
    // Set texture parameters
    tex.addressMode[0] = cudaAddressModeBorder;
    tex.addressMode[1] = cudaAddressModeBorder;
    tex.filterMode = cudaFilterModePoint;
    tex.normalized = false;    // access with normalized texture coordinates

    // Bind the array to the texture
    checkCudaErrors(cudaBindTextureToArray(tex, inArray, channelDesc));

    // Set up grid
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);
    int window_size = BLOCK_SIZE * BLOCK_SIZE * filterSize * filterSize * sizeof(float);


    #if DEBUG_MESSAGES_ON
    std::cout << "\n\nBlocks and grid set up.\nBlock is " << dimBlock.x << " x " << dimBlock.y << \
        "\nGrid is " << dimGrid.x << " x " << dimGrid.y << std::endl;
    std::cout << "Pixel radius is " << (filterSize >> 1) << std::endl;
    std::cout << "Window size is: " << window_size << std::endl;
    #endif


    /***************
    CALL KERNEL TO PROCESS FILE
    ****************/
    blurKernel<<<dimGrid, dimBlock, window_size>>>(outData, width, height, filterSize);
    getLastCudaError("Kernel execution failed");


    /***************
    SAVE OUTPUT TO FILE
    ****************/
    // Allocate mem for the result on host side
    float *hOutputData = (float *) malloc(size);
    // copy result from device to host
    checkCudaErrors(cudaMemcpy(hOutputData,
                               (const void*)outData,
                               size,
                               cudaMemcpyDeviceToHost));
    // Write to file
    char *outimagePath = sdkFindFilePath(outputfile, argv[0]);
    if (outimagePath == NULL){
        #if DEBUG_MESSAGES_ON
        std::cout << "Unable to source image file:"<< outputfile << "\n" << std::endl;
        #endif
        exit(EXIT_FAILURE);
    }
    sdkSavePGM(outimagePath, hOutputData, width, height);
    #if DEBUG_MESSAGES_ON
    std::cout << "Wrote to " << outputfile << "." << std::endl;
    #endif



    /*************
    CREATE STANDARD FILE TO TEST CUDA SOLUTION ON HOST
    **************/
    #if TEST_MODE
    // Allocate mem for the standard
    float *standData = (float *) malloc(size);
    createGoldenStandard(origData, standData, width, height, filterSize);

    // Compare output to standard, get percentage correct back
    float percentage = compareToStandard(standData, hOutputData, width, height);
    std::cout << "percentage correct: " << percentage << "%" << std::endl;

    fprintf(statsFile, "AccuracyStats: %d %f\n", filterSize, percentage);

    // Print to file if specified
    if(argc >= 4){
        char *stdOutImagePath = sdkFindFilePath(argv[4], argv[0]);
        if (stdOutImagePath == NULL){
            #if DEBUG_MESSAGES_ON
            std::cout << "Unable to source image file:"<< argv[4] << "\n" << std::endl;
            #endif
            exit(EXIT_FAILURE);
        }
        sdkSavePGM(stdOutImagePath, standData, width, height);
    }
    #endif

    return 0;
}


void createGoldenStandard( float *origData, float *standData, unsigned int width, unsigned int height, uint filterSize){
    if(origData == NULL || standData == NULL){
        #if DEBUG_MESSAGES_ON
        std::cout << "Data is null" << std::endl;
        #endif
        return;
    }

    uint pixelRadius = filterSize >> 1,
        // x_start = pixelRadius,
        // x_end   = width-pixelRadius,
        // y_start = pixelRadius,
        // y_end   = height-pixelRadius,
        arraySize = filterSize * filterSize,
        halfArraySize = arraySize/2 + 1;


    for(int y = 0 ; y <= height ; y++){
        for(int x = 0 ; x <= width ; x++){
            if(x <= pixelRadius || x >= width-pixelRadius || y <= pixelRadius || y >= height-pixelRadius){
                standData[(y*width) + x] = origData[(y*width) + x];
            }
            else{
                // At 1 pixel currently.  Iterate through its neighbors and find median.
                float neighbors[arraySize];
                uint p_x_start = x - pixelRadius,
                     p_x_end   = x + pixelRadius,
                     p_y_start = y - pixelRadius,
                     p_y_end   = y + pixelRadius;

                // Add neighbors to neighbors array
                for(int i = 0, yy = p_y_start ; yy <= p_y_end ; i++, yy++){
                    for(int j = 0, xx = p_x_start ; xx <= p_x_end ; j++, xx++){
                        neighbors[(i * filterSize) + j] = origData[(yy * width) + xx];
                    }
                }

                // Get median, assign to new array
                for(int i = 0 ; i <= halfArraySize ; i++){
                    int min = i;
                    for(int j = i + 1 ; j < arraySize ; j++){
                        if(neighbors[j] < neighbors[min]){
                            min = j;
                        }
                    }
                    float temp = neighbors[i];
                    neighbors[i] = neighbors[min];
                    neighbors[min] = temp;
                }

                standData[(y*width) + x] = neighbors[halfArraySize];
            }
        }
    }
    return;
}

float compareToStandard(float *standData, float *testData, uint width, uint height){
    if(standData == NULL || testData == NULL){
        return 0;
    }

    uint count = 0, numCorrect = 0;
    for(int y = 0 ; y < height ; y++){
        for(int x = 0 ; x < width ; x++){
            if(standData[(y*width) + x] == testData[(y*width) + x]){
                numCorrect++;
            }
            else{
                // std::cout << standData[(y*width) + x] << " vs " << testData[(y*width) + x] << std::endl;
            }
            count++;
        }
    }

    return (float)100.0*(float)((float)numCorrect/(float)count);
}









