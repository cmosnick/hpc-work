To run:

*   make

*   ./main \<filtersize = 3, 7, 11, 15\> \<inputfile.pgm\>  \<outputfile.pgm\>

*   ex: ./main 3 data/lena.pgm  data/lena_out_3.pgm


To run tests, genrate data files, & generate charts:

*   make tests NUMRUNS="\<number of runs\>"
*   ex: make tests NUMRUNS="20"
