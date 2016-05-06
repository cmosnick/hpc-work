#include "stdlib.h"
#include <iostream>
#include <cuda_runtime.h>
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <chrono>


    
#define BLOCK_SIZE  4
#define FILTER_WINDOW_SIZE 9
#define WINDOW_SIZE (BLOCK_SIZE*BLOCK_SIZE*FILTER_WINDOW_SIZE*sizeof(float))

enum XORY{
    X_SOBEL = 0,
    Y_SOBEL = 1
};

typedef unsigned int uint;
void createGoldenStandard( float *origData, float *standData, unsigned int width, unsigned int height, uint filterSize);
float compareToStandard( float *standData, float *testData, uint width, uint height);

const char *statsFileName = "out_stats.txt";

// Texture reference for 2D float texture
texture<float, 2, cudaReadModeElementType> tex;


// Kernel function to perform blurring
__global__ void sobelKernel(float *outputData, int width, int height, int xory){
    extern __shared__ float window[];
    const int sobel_x[3][3] = {{-1, 0, 1}, {-2, 0 , 2}, {-1, 0, 1}};
    const int sobel_y[3][3] = {{-1, -2, 1}, {0, 0, 0}, {1, 2, 1}};

    // calculate texture coordinates
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int pixelRadius = 1;

    if(x < pixelRadius || y < pixelRadius || x >= (width-pixelRadius) || y >= (height-pixelRadius)){
        // Do nothing
        outputData[(y * width) + x] = tex2D(tex, x, y);
    }
    else{
        // Calculate bounds for processing
        uint x_start    = x - pixelRadius;
        uint x_end      = x + pixelRadius;
        uint y_start    = y - pixelRadius;
        uint y_end      = y + pixelRadius;

        float sum = 0;

        // Fill array with values
        for(int i = x_start, ii=0 ; i <= x_end; i++, ii++){
            for(int j =  y_start, jj=0 ; j <= y_end ; j++, jj++){
                if(xory == X_SOBEL){
                    sum += tex2D(tex, i, j) * sobel_x[ii][jj];                    
                }
                else if(xory == Y_SOBEL){
                    sum += tex2D(tex, i, j) * sobel_y[ii][jj];                    
                }
            }
        }
        outputData[(y * width) + x] = sum;
    }
}


int main(int argc, char **argv){
    // Check args
    if(argc < 3){
        #if DEBUG_MESSAGES_ON
        std::cout << "\n\nIncorrect number of args.  Should be \n(1)input file\n(2)X output file\n(3)Y output file" << std::endl;
        #endif
        return 0;
    }
    // Get input and output files
    char *inputfile = argv[1];
    if(!inputfile){
        return 0;
    }
    char *xoutputfile = argv[2];
    if(!xoutputfile){
        return 0;
    }
    char *youtputfile = argv[3];
    if(!youtputfile){
        return 0;
    }
    #if TEST_MODE
    std::cout << "\n\n\nTesting " << filterSize << " pixel filter with " << BLOCK_SIZE << " x " << BLOCK_SIZE << " blocks" << std::endl;
    #endif
    // Load PGM onto device
    int devID = findCudaDevice(argc, (const char **) argv);

    #if TEST_MODE
    FILE *statsFile = fopen(statsFileName, "a");
    if(!statsFile){
        std::cout << "Couldn't open stats file" << std::endl;
    }
    #endif

    /***************
    LOAD INPUT FILE
    ****************/
    // Start timing for load time
    std::chrono::time_point <std::chrono::system_clock> start, end;

    float *origData = NULL;
    unsigned int width, height;
    char *imagePath = sdkFindFilePath(inputfile, argv[0]);

    if (imagePath == NULL){
        #if DEBUG_MESSAGES_ON
        std::cout << "Unable to source image file:"<< inputfile << " %s\n" << std::endl;
        #endif
        exit(EXIT_FAILURE);
    }
    start = std::chrono::system_clock::now();
    sdkLoadPGM(imagePath, &origData, &width, &height);
    end = std::chrono::system_clock::now();

    // Print to stats file
    #if TEST_MODE
    if(statsFile){
        std::chrono::duration<double> timeElapsed = end-start;
        fprintf(statsFile, "LoadTimeStats: %f\n", timeElapsed);
    }
    #endif

    int size = width * height * sizeof(float);
    #if DEBUG_MESSAGES_ON
    std::cout << "Loaded " << inputfile << ", " << width << " x "<< height << " pixels with size " << (uint)size << std::endl;
    #endif


    // Allocate array and copy image data
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray *inArray;
    checkCudaErrors(cudaMallocArray(&inArray,
                                    &channelDesc,
                                    width,
                                    height));
    // Start timer
    start = std::chrono::system_clock::now();
    checkCudaErrors(cudaMemcpyToArray(inArray,
                                      0,
                                      0,
                                      origData,
                                      size,
                                      cudaMemcpyHostToDevice));

    #if DEBUG_MESSAGES_ON
    std::cout << "\nLoaded " << inputfile << " onto device." << std::endl;
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
    // int window_size = BLOCK_SIZE * BLOCK_SIZE * filterSize * filterSize * sizeof(float);


    #if DEBUG_MESSAGES_ON
    std::cout << "\nBlocks and grid set up.\nBlock is " << dimBlock.x << " x " << dimBlock.y << \
        "\nGrid is " << dimGrid.x << " x " << dimGrid.y << std::endl;
    // std::cout << "Pixel radius is " << (filterSize >> 1) << std::endl;
    // std::cout << "Window size is: " << WINDOW_SIZE << std::endl;
    #endif


    /***************
    CALL KERNEL TO PROCESS FILE
    ****************/
    // Allocate device memory for result
    float *outData = NULL;
    checkCudaErrors(cudaMalloc((void **) &outData, size));
    sobelKernel<<<dimGrid, dimBlock>>>(outData, width, height, X_SOBEL);
    getLastCudaError("Kernel execution failed");


    /***************
    SAVE X OUTPUT TO FILE
    ****************/
    // Allocate mem for the result on host side
    float *hOutputData = (float *) malloc(size);
    // copy result from device to host
    checkCudaErrors(cudaMemcpy(hOutputData,
                               (const void*)outData,
                               size,
                               cudaMemcpyDeviceToHost));
    end = std::chrono::system_clock::now();
    // Write to stats file
    #if TEST_MODE
    if(statsFile){
        std::chrono::duration<double> timeElapsed = end-start;
        fprintf(statsFile, "ComputeTimeStats: %f\n", timeElapsed);
    }
    #endif


    // Write to file
    char *outimagePath = sdkFindFilePath(xoutputfile, argv[0]);
    if (outimagePath == NULL){
        #if DEBUG_MESSAGES_ON
        std::cout << "Unable to source image file:"<< xoutputfile << "\n" << std::endl;
        #endif
        exit(EXIT_FAILURE);
    }
    sdkSavePGM(outimagePath, hOutputData, width, height);
    #if DEBUG_MESSAGES_ON
    std::cout << "Wrote to " << xoutputfile << "." << std::endl;
    #endif


    /****************
    Do Y sobel filter now
    *****************/
    // Allocate device memory for result
    float *yOutData = NULL;
    checkCudaErrors(cudaMalloc((void **) &yOutData, size));
    sobelKernel<<<dimGrid, dimBlock>>>(yOutData, width, height, Y_SOBEL);
    getLastCudaError("Kernel execution failed");


    /***************
    SAVE Y OUTPUT TO FILE
    ****************/
    // Allocate mem for the result on host side
    float *hyOutputData = (float *) malloc(size);
    // copy result from device to host
    checkCudaErrors(cudaMemcpy(hyOutputData,
                               (const void*)yOutData,
                               size,
                               cudaMemcpyDeviceToHost));
    end = std::chrono::system_clock::now();
    // Write to stats file
    #if TEST_MODE
    if(statsFile){
        std::chrono::duration<double> timeElapsed = end-start;
        fprintf(statsFile, "ComputeTimeStats: %f\n", timeElapsed);
    }
    #endif


    // Write to file
    char *youtimagePath = sdkFindFilePath(youtputfile, argv[0]);
    if (youtimagePath == NULL){
        #if DEBUG_MESSAGES_ON
        std::cout << "Unable to source image file:"<< xoutputfile << "\n" << std::endl;
        #endif
        exit(EXIT_FAILURE);
    }
    sdkSavePGM(youtimagePath, hyOutputData, width, height);
    #if DEBUG_MESSAGES_ON
    std::cout << "Wrote to " << youtputfile << "." << std::endl;
    #endif

    /*************
    CREATE STANDARD FILE TO TEST CUDA SOLUTION ON HOST
    **************/
    // #if TEST_MODE
    // start = std::chrono::system_clock::now();
    // // Allocate mem for the standard
    // float *standData = (float *) malloc(size);
    // // createGoldenStandard(origData, standData, width, height, filterSize);
    // end = std::chrono::system_clock::now();
    // std::chrono::duration<double> timeElapsed = end-start;
    // // Print timing to file
    // // if(statsFile){
    // //     fprintf(statsFile, "GSTimingStats: %d %f\n", filterSize, timeElapsed);
    // // }

    // // Compare output to standard, get percentage correct back
    // float percentage = compareToStandard(standData, hOutputData, width, height);
    // std::cout << "Percentage correct: " << percentage << "%" << std::endl;

    // fprintf(statsFile, "AccuracyStats: %d %f\n", filterSize, percentage);

    // // Print to file if specified
    // if(argc >= 4){
    //     char *stdOutImagePath = sdkFindFilePath(argv[4], argv[0]);
    //     if (stdOutImagePath == NULL){
    //         #if DEBUG_MESSAGES_ON
    //         std::cout << "Unable to source image file:"<< argv[4] << "\n" << std::endl;
    //         #endif
    //         exit(EXIT_FAILURE);
    //     }
    //     sdkSavePGM(stdOutImagePath, standData, width, height);
    // }
    // #endif

    return 0;
}


// void createGoldenStandard( float *origData, float *standData, unsigned int width, unsigned int height, uint filterSize){
//     if(origData == NULL || standData == NULL){
//         #if DEBUG_MESSAGES_ON
//         std::cout << "Data is null" << std::endl;
//         #endif
//         return;
//     }

//     uint pixelRadius = filterSize >> 1,
//         arraySize = filterSize * filterSize,
//         halfArraySize = arraySize/2 + 1;


//     for(int y = 0 ; y <= height ; y++){
//         for(int x = 0 ; x <= width ; x++){
//             if(x <= pixelRadius || x >= width-pixelRadius || y <= pixelRadius || y >= height-pixelRadius){
//                 standData[(y*width) + x] = origData[(y*width) + x];
//             }
//             else{
//                 // At 1 pixel currently.  Iterate through its neighbors and find median.
//                 float neighbors[arraySize];
//                 uint p_x_start = x - pixelRadius,
//                      p_x_end   = x + pixelRadius,
//                      p_y_start = y - pixelRadius,
//                      p_y_end   = y + pixelRadius;

//                 // Add neighbors to neighbors array
//                 for(int i = 0, yy = p_y_start ; yy <= p_y_end ; i++, yy++){
//                     for(int j = 0, xx = p_x_start ; xx <= p_x_end ; j++, xx++){
//                         neighbors[(i * filterSize) + j] = origData[(yy * width) + xx];
//                     }
//                 }

//                 // Get median, assign to new array
//                 for(int i = 0 ; i <= halfArraySize ; i++){
//                     int min = i;
//                     for(int j = i + 1 ; j < arraySize ; j++){
//                         if(neighbors[j] < neighbors[min]){
//                             min = j;
//                         }
//                     }
//                     float temp = neighbors[i];
//                     neighbors[i] = neighbors[min];
//                     neighbors[min] = temp;
//                 }
//                 standData[(y*width) + x] = neighbors[halfArraySize];
//             }
//         }
//     }
//     return;
// }

// float compareToStandard(float *standData, float *testData, uint width, uint height){
//     if(standData == NULL || testData == NULL){
//         return 0;
//     }

//     uint count = 0, numCorrect = 0;
//     for(int y = 0 ; y < height ; y++){
//         for(int x = 0 ; x < width ; x++){
//             if(standData[(y*width) + x] == testData[(y*width) + x]){
//                 numCorrect++;
//             }
//             else{
//                 // std::cout << standData[(y*width) + x] << " vs " << testData[(y*width) + x] << std::endl;
//             }
//             count++;
//         }
//     }

//     return (float)100.0*(float)((float)numCorrect/(float)count);
// }









