To run:

*   put me on a machine with CUDA capabilities!

*   make

*   ./main \<filtersize = 3, 7, 11, 15\> \<inputfile.pgm\>  \<outputfile.pgm\>

*   ex: ./main 3 data/lena.pgm  data/lena_out_3.pgm


To run tests, genrate data files, & generate charts:

*   make clean

*   make tests NUMRUNS="\<number of runs\>"

*   ex: make tests NUMRUNS="20"

*   Note: you may get an error when the python script attempts to run if you're on a basic AWS instance and matplotlib is not installed.  This is OK, this just means no charts will be generated.

To show images , they should already be copied into jpegs.
*   cd data

*   ./open_script (works for OSX, uses "open" command)
