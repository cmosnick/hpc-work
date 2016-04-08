#include <mpi.h>




int main(int argc, char *argv[]){
    MPI_Init(&argc, &argv);
    // Get rank (process number in MPI) to determine which process we're in
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    // Master branch
    if(rank == 0){
        // Check args
        if(argc < 5){
            std::cout << "Please enter (1) a filename query\n \
            (2) a csv input file\n \
            (3) the number fo results to generate\n\
            (4) the number of processes." << std::endl;

            MPI_Finalize();
            exit(0);
        }

        // Check input file
        FILE *infile = fopen(argv[2], "r");
        if(!infile){
            std::cout << "File invalid" << std::endl;
            exit(0);
        }


    }
    // Worker branches
    else{

    }

    // Donezo
    MPI_Finalize();

    return 0;
}