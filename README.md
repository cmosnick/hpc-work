#High Performance Computing#
######Various experiments on CPU and GPU performance with multiprocessing, multi threading, and other optimizations.


####Homework 0
Basic file parsing
* Read file into map of \<string, vector\<float\>\> holding 4098 floats
* find bounding min and max values for each column (after file load)
* Record times and report



####Homework 1
Matrix multiplication compared to Boost library solution, single processor
* Take two matrices, multiply them together
* Transpose right hand matrix to reduce cache misses by making columns contiguous rows
* Improve on Boost library times significantly



####Homework 2
Multiprocessing: Find [Manhattan metric](https://xlinux.nist.gov/dads//HTML/manhattanDistance.html) (L1 Norm) for each vector, compared to specified vector of floats (from homework 0 file).  Find K-nearest neighbors of comparison vector based on distance.
* Read in homework 0 file as before: into map<string, vector<float>>
* Block-partition input
* Send input blocks to processes
* Write process' k-nearest neighbors to shared memory
* After convergence, filter k-nearest neighbors from unified shared memory solution
* Return solution



####Homework 3
Multithreading: Use boost threads to accomplish homework 2



####Homework 4
Use OpenMPI to accomplish Manhattan distance problem



####Homework 5
GPU programming: use CUDA to perform median filters of varying sizes (3x3, 7x7, 11x11, 15x15) on lena.pgm
* Load .pgm into buffer on CPU
* Load buffer into GPU
* Perform filter kernel function
* Load back onto cpu
* Compare to CPU solution for accuracy
* Compare to CPU time


####Homework 6
GPU programming: use CUDA to perform Sobel Filter (edge detection) on lena.pgm


