#include <mpi.h>
#include <boost/algorithm/string.hpp>
#include "directory_scanner.hpp"
#include "./homework3_copy/hw3.hpp"

#define DEBUG_MESSAGES 1

#define FILE_PATH_SIZE 1024
#define LINE_MESSAGE_SIZE (LINE_SIZE + FILE_PATH_SIZE + 3)

#define MAX_RESULTS 1000
#define RESULT_MESSAGE_SIZE (FNAME_SIZE + FLOAT_CHARS + 3) * (MAX_RESULTS)


enum MPI_TAGS {
    TERMINATE = 0, 
    PROCESS   = 1
};

typedef struct resultMessage{
    char    name[FNAME_SIZE];
    float   dist;
} resultMessage_t;

typedef std::map<std::string,scottgs::path_list_type> content_type;
typedef content_type::const_iterator content_type_citr;

namespace cmoz{
    void parseFiles(const scottgs::path_list_type file_list, int numResults);
    void printDirContents(const scottgs::path_list_type file_list);
    void workerParseFile(FILE *search_vector_file, int numResults);
    void getSearchVector(FILE *search_vector_file, std::vector<float> &floats);
    void createMpiTypes(int numResults, MPI_Datatype *cmoz_result_type, MPI_Datatype *cmoz_multipleResults_type);
}

int main(int argc, char *argv[]){
    MPI_Init(&argc, &argv);
    // Get rank (process number in MPI) to determine which process we're in
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Check args
    if(argc < 4){
        std::cout << "Please enter (1) a filename query\n \
        (2) a csv input file\n \
        (3) the number of results to generate\n" << std::endl;

        MPI_Finalize();
        exit(0);
    }

    // number of results
    int numResults = 0;
    if(atoi(argv[3])){
        numResults = atoi(argv[3]);
    }
    else{
        std::cout << "Invalid number for argument 3" << std::endl;
        MPI_Finalize();
        exit(0);
    }



    // Master branch
    if(rank == 0){

        // Read directory
        content_type directoryContents;
        scottgs::path_list_type *fileList;
        try{
            directoryContents = scottgs::getFiles(argv[2]);
            fileList = &(directoryContents[REG_FILE]);
        }
        catch(std::exception& e){
            std::cout << "Something went wrong reading the input directory." << std::endl;
            MPI_Finalize();
            exit(0);
        }
        #if DEBUG_MESSAGES
        std::cout << "Sucessful reading directory" << std::endl;
        // cmoz::printDirContents(directoryContents);
        cmoz::printDirContents(*fileList);
        #endif

        // Delegate parsing of files
        cmoz::parseFiles(*fileList, numResults);
        MPI_Barrier(MPI_COMM_WORLD); // this is linked to the above barrier
    }
    // Worker branches
    else{
        // Read input vector
        FILE *vectorInfile = fopen(argv[2], "r");
        if(!vectorInfile){
            std::cout << "File invalid" << std::endl;
            MPI_Finalize();
            exit(0);
        }

        cmoz::workerParseFile(vectorInfile, numResults);
        MPI_Barrier(MPI_COMM_WORLD); // this is linked to the above barrier
    }

    // Donezo
    MPI_Finalize();

    return 0;
}

// Called by master to delegate parsing of files
void cmoz::parseFiles(const scottgs::path_list_type file_list, int numResults){
    // Called by master thread to delegate files to other threads
    int threadCount;
    MPI_Comm_size(MPI_COMM_WORLD, &threadCount);
    
    #if DEBUG_MESSAGES
    std::cout << "Number of threads: " << threadCount << std::endl;
    #endif

    // Create MPI types in this particular thread.  Each thread must do it
    MPI_Datatype cmoz_result_type;
    MPI_Datatype cmoz_multipleResults_type;
    createMpiTypes(numResults, &cmoz_result_type, &cmoz_multipleResults_type);

    // iterate through filenames, send to threads
    scottgs::path_list_type::const_iterator file = file_list.begin();

    // Send first time to each thread
    // Send a file to each thread
    for (int rank = 1; rank < threadCount && file!=file_list.end(); ++rank, ++file){
        // Compose message consisting of "<file path to parse>"
        char *msg = (char *) malloc( FILE_PATH_SIZE * sizeof(char));
        std::string file_path(file->generic_string());
        // std::cout << file_path << std::endl;
        size_t length = file_path.length();
        file_path.copy(msg, length);
        msg[length+1] = '\0';

        MPI_Send(msg,               /* message buffer */
            FILE_PATH_SIZE,         /* buffer size */
            MPI_CHAR,               /* data item is an integer */
            rank,                   /* destination process rank */
            PROCESS,                /* user chosen message tag */
            MPI_COMM_WORLD);        /* default communicator */
        free(msg);
    }

    //Send all others in between, getting MPI_Recv's from threads
    for( ; file!=file_list.end(); ++file){
        // Receive results from a worker
        resultMessage_t results[numResults];
        MPI_Status status;

        // Receive a message from the worker
        MPI_Recv(&results,       /* message buffer */
            1,                  /* buffer size */
            cmoz_multipleResults_type,   /* data item is an array of results */
            MPI_ANY_SOURCE,     /* Recieve from thread */
            MPI_ANY_TAG,        /* tag */
            MPI_COMM_WORLD,     /* default communicator */
            &status
        );

        #if DEBUG_MESSAGES
        // Print return values
        std::cout << "\n\n";
        for(int i = 0 ; i < numResults ; i++){
            std::string name(results[i].name);
            std::cout << name << ": " << results[i].dist << std::endl;
        }
        #endif     

        // TODO: merge results with global results

        // Send new task
        const int sourceCaught = status.MPI_SOURCE;

        // Compose message consisting of "<file path to parse>"
        char *msg = (char *) malloc( FILE_PATH_SIZE * sizeof(char));
        std::string file_path(file->generic_string());
        // std::cout << file_path << std::endl;
        size_t length = file_path.length();
        file_path.copy(msg, length);
        msg[length+1] = '\0';

        MPI_Send(msg,               /* message buffer */
            FILE_PATH_SIZE,         /* buffer size */
            MPI_CHAR,               /* data item is an integer */
            sourceCaught,           /* destination process rank */
            PROCESS,                /* user chosen message tag */
            MPI_COMM_WORLD);        /* default communicator */
        free(msg);
    }


    // Collect remaining results
    for(int rank = 1 ; rank < threadCount ; rank ++){

        // Receive results from a worker
        resultMessage_t results[numResults];
        MPI_Status status;

        // Receive a message from the worker
        MPI_Recv(&results,      /* message buffer */
            1,                  /* buffer size */
            cmoz_multipleResults_type,   /* data item is an array of results */
            MPI_ANY_SOURCE,     /* Recieve from thread */
            MPI_ANY_TAG,        /* tag */
            MPI_COMM_WORLD,     /* default communicator */
            &status
        );

        #if DEBUG_MESSAGES
        // Print return values
        std::cout << "\n\n";
        for(int i = 0 ; i < numResults ; i++){
            std::string name(results[i].name);
            std::cout << name << ": " << results[i].dist << std::endl;
        }
        #endif     

        // TODO: merge results with global results
    }

    // Send each thread a terminate signal
    for (int rank = 1; rank < threadCount; ++rank){
        MPI_Send(0, 0, MPI_INT, rank, TERMINATE, MPI_COMM_WORLD);
    }

    return;
}

void cmoz::workerParseFile(FILE *search_vector_file, int numResults){
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Create MPI types in this particular thread.  Each thread must do it
    MPI_Datatype cmoz_result_type;
    MPI_Datatype cmoz_multipleResults_type;
    createMpiTypes(numResults, &cmoz_result_type, &cmoz_multipleResults_type);

    // Get line in search file
    std::vector<float> floats(NUM_FLOATS);
    getSearchVector(search_vector_file, floats);

    char msg[LINE_MESSAGE_SIZE];
    MPI_Status status;

    // Start recieving files to parse
    while(1){
        MPI_Recv(msg,           /* message buffer */
            FILE_PATH_SIZE,     /* buffer size */
            MPI_CHAR,           /* data item is an integer */
            0,                  /* Receive from master */
            MPI_ANY_TAG,        
            MPI_COMM_WORLD,     /* default communicator */
            &status
        );

        // Check for teminate tag
        if(status.MPI_TAG == TERMINATE){
            #if DEBUG_MESSAGES
            std::cout << "Thread" << rank << " received a terminate signal" << std::endl;
            #endif
            return;
        }

        // Convert message into string
        std::string messageReceived(msg);

        #if DEBUG_MESSAGES
        std::cout << messageReceived << std::endl;
        // std::cout << "here" << std::endl;
        #endif

        // TODO: read in / check file name passed in, process against search vector, and send list back


        // Send back array <<filename>, <dist>> of size results
        resultMessage_t results[numResults];
        for(int i=0 ; i < numResults ; i ++){
            std::string name("test\0");
            name.copy(&(results[i].name[0]), 5);
            results[i].dist = .5;
        }
        MPI_Send( 
            &results,                   /*Send buffer object, array of cmoz_result_types*/
            1,                          /*Number of objects in bufer*/
            cmoz_multipleResults_type,  /*Type of object in buffer*/
            0,                          /*Destination: master*/
            PROCESS,                    /*Message tag*/
            MPI_COMM_WORLD              /*Communictaion channel*/
        );
    }
    return;
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

void cmoz::getSearchVector(FILE *search_vector_file, std::vector<float> &floats){
    // File already open
    char    *searchVector = (char*)malloc(sizeof(char) * LINE_SIZE);
    size_t  lineSize = LINE_SIZE;
    getline(&searchVector, &lineSize, search_vector_file);

    if(searchVector == NULL){
        #if DEBUG_MESSAGES
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::cout << "Error reading in search vector in thread" << rank << endl;
        #endif
        MPI_Finalize();
        exit(0);
    }
    fclose(search_vector_file);

    // Parse search vector into vector of floats
    // Get first token, the filename
    char *line = (char*)malloc(sizeof(char) * LINE_SIZE);
    char *token = NULL;
    token = strtok(line, DELIMS);
    if(token){
        // Ignore first token, don't need it
        // read in floats
        uint i = 0;
        do{
            token = strtok(NULL, DELIMS);
            if(token && is_float(token)){
                float temp = atof(token);
                floats[i]=temp;
            }
            else 
                break;
            i++;
        }while( i <= NUM_FLOATS);
    }
    free(line);
}

void cmoz::createMpiTypes(int numResults, MPI_Datatype *cmoz_result_type, MPI_Datatype *cmoz_multipleResults_type){
    // Create mpi struct type for one result.
    int             numElements = 2;
    int             blockLengths[2] = {FNAME_SIZE, 1};  /*number of chars, number of floats*/
    MPI_Datatype    types[2] = {MPI_CHAR, MPI_FLOAT};   /*types in struct*/
    // MPI_Datatype    cmoz_result_type;
    MPI_Aint        offsets[2];
    offsets[0] = (MPI_Aint)offsetof(resultMessage_t, name);
    offsets[1] = (MPI_Aint)offsetof(resultMessage_t, dist);
    
    MPI_Type_create_struct(
        numElements,            /*Number of results, or elements*/
        blockLengths,           /*Number of chars, number of floats*/
        offsets,                /*Offset of name & distance in each element*/
        types,                  /*Specify they are chars, and float*/
        cmoz_result_type       /*Object to put type in*/
    );
    MPI_Type_commit(cmoz_result_type);



    // Create mpi struct of cmoz_result_type
                    numElements = 1;
    int             blockLength[1] = {numResults};
    MPI_Datatype    type[1] = {*cmoz_result_type};
    MPI_Aint        offset[1];
    offset[0] = 0;
    // MPI_Datatype    cmoz_multipleResults_type;

    MPI_Type_create_struct(
        numElements,            /*Number of results, or elements*/
        blockLength,           /*Number of chars, number of floats*/
        offset,                /*Offset of name & distance in each element*/
        type,                  /*Specify they are chars, and float*/
        cmoz_multipleResults_type       /*Object to put type in*/
    );
    MPI_Type_commit(cmoz_multipleResults_type);
}








