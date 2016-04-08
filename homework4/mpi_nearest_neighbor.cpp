#include <mpi.h>
#include "directory_scanner.hpp"

typedef std::map<std::string,scottgs::path_list_type> content_type;

namespace cmoz{
    void parseFiles(const content_type directoryContents);

}

int main(int argc, char *argv[]){
    MPI_Init(&argc, &argv);
    // Get rank (process number in MPI) to determine which process we're in
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    // Master branch
    if(rank == 0){
        // Check args
        if(argc < 4){
            std::cout << "Please enter (1) a filename query\n \
            (2) a csv input file\n \
            (3) the number of results to generate"<< std::endl;

            MPI_Finalize();
            exit(0);
        }

        // Read directory
        
        try{
            content_type directoryContents = scottgs::getFiles(argv[2]);

        }
        catch(std::exception& e){
            std::cout << "Something went wrong reading the input directory." << std::endl;
            MPI_Finalize();
            exit(0);
        }

        // Check input file (in place of directory read for now)
        // FILE *infile = fopen(argv[2], "r");
        // if(!infile){
        //     std::cout << "File invalid" << std::endl;
        //     MPI_Finalize();
        //     exit(0);
        // }




    }
    // Worker branches
    else{

    }

    // Donezo
    MPI_Finalize();

    return 0;
}

void cmoz::parseFiles(const content_type directoryContents){
    // Called by master thread to delegate files to other threads
    int threadCount;
    MPI_Comm_size(MPI_COMM_WORLD, &threadCount);


 }










