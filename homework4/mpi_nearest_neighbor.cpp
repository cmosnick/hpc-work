#include <mpi.h>
#include "directory_scanner.hpp"

typedef std::map<std::string,scottgs::path_list_type> content_type;
typedef content_type::const_iterator content_type_citr;

namespace cmoz{
    void parseFiles(const content_type directoryContents);
    void printDirContents(const scottgs::path_list_type file_list);

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
        content_type directoryContents;
        scottgs::path_list_type *fileList;
        try{
            directoryContents = scottgs::getFiles(argv[2]);
            const std::string regular_file("REGULAR");
            fileList = &(directoryContents[regular_file]);
        }
        catch(std::exception& e){
            std::cout << "Something went wrong reading the input directory." << std::endl;
            MPI_Finalize();
            exit(0);
        }
        std::cout << "Sucessful reading directory" << std::endl;

        // cmoz::printDirContents(directoryContents);
        cmoz::printDirContents(*fileList);



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


void cmoz::printDirContents(const scottgs::path_list_type file_list){
    // For each type of file found in the directory, 
    // List all files of that type

     // for (content_type_citr f = directoryContents.begin(); 
     //     f!=directoryContents.end();
     //     ++f)
     // {
         // const scottgs::path_list_type file_list(f->second);
            
         // std::cout << "Showing: " << f->first << " type files (" << file_list.size() << ")" << std::endl;
         for (scottgs::path_list_type::const_iterator i = file_list.begin();
             i!=file_list.end(); ++i)
         {
             //boost::filesystem::path file_path(boost::filesystem::system_complete(*i));
             boost::filesystem::path file_path(*i);
             std::cout << "\t" << file_path.string() << std::endl;
         }
                
     // }
}










