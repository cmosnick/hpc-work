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

typedef struct message{
    char    name[FNAME_SIZE];
    float   dist;
} message_t;

typedef std::map<std::string,scottgs::path_list_type> content_type;
typedef content_type::const_iterator content_type_citr;

namespace cmoz{
    void parseFiles(const scottgs::path_list_type file_list, FILE *search_vector_file);
    void printDirContents(const scottgs::path_list_type file_list);
    void workerParseFile();
}

int main(int argc, char *argv[]){
    MPI_Init(&argc, &argv);
    // Get rank (process number in MPI) to determine which process we're in
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Create mpi struct type for message passing
    const int       numItems = 2;
    int             blockLengths[2] = {FNAME_SIZE, 1};
    MPI_Datatype    types[2] = {MPI_CHAR, MPI_FLOAT};
    MPI_Datatype    mpi_message_type;
    MPI_Aint        offsets[2];
    
    offsets[0] = offsetof(message_t, name);
    offsets[1] = offsetof(message_t, dist);

    MPI_Type_create_struct(numItems, blockLengths, offsets, types, &mpi_message_type);
    MPI_Type_commit(&mpi_message_type);


    // Master branch
    if(rank == 0){
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

        // Read input vector
        FILE *infile = fopen(argv[2], "r");
        if(!infile){
            std::cout << "File invalid" << std::endl;
            MPI_Finalize();
            exit(0);
        }


        // Delegate parsing of files
        cmoz::parseFiles(*fileList, infile);
        MPI_Barrier(MPI_COMM_WORLD); // this is linked to the above barrier

    }
    // Worker branches
    else{
        cmoz::workerParseFile();
        MPI_Barrier(MPI_COMM_WORLD); // this is linked to the above barrier
    }

    // Donezo
    MPI_Finalize();

    return 0;
}

// Called by master to delegate parsing of files
void cmoz::parseFiles(const scottgs::path_list_type file_list, FILE *search_vector_file){
    // Called by master thread to delegate files to other threads
    int threadCount;
    MPI_Comm_size(MPI_COMM_WORLD, &threadCount);
    
    #if DEBUG_MESSAGES
    std::cout << "Number of threads: " << threadCount << std::endl;
    #endif

    // Get line in search file
    char    *searchVector = (char*)malloc(sizeof(char) * LINE_SIZE);
    size_t  lineSize = LINE_SIZE;
    getline(&searchVector, &lineSize, search_vector_file);
    if(searchVector == NULL){
        MPI_Finalize();
        exit(0);
    }
    fclose(search_vector_file);

    #if DEBUG_MESSAGES
    // std::cout << searchVector << std::endl;
    #endif

    // iterate through filenames, send to threads
    scottgs::path_list_type::const_iterator file = file_list.begin();

    // Send first time to each thread
    // Send each file to a thread
    for (int rank = 1; rank < threadCount && file!=file_list.end(); ++rank, ++file){
        // Compose message consisting of "<file path to parse>, <search vector>"
        char *msg = (char *) malloc( LINE_MESSAGE_SIZE * sizeof(char));
        std::string file_path(file->generic_string());
        // std::cout << file_path << std::endl;
        size_t length = file_path.length();
        file_path.copy(msg, length);
        strncpy(msg+length, ", ", 2);
        strncpy((msg+length+2), searchVector, LINE_SIZE);

        MPI_Send(msg,           /* message buffer */
            LINE_MESSAGE_SIZE,            /* buffer size */
            MPI_CHAR,          /* data item is an integer */
            rank,              /* destination process rank */
            PROCESS,          /* user chosen message tag */
            MPI_COMM_WORLD);   /* default communicator */
        free(msg);
    }

    //Send all others in between, getting MPI_Recv's from threads
    for( ; file!=file_list.end(); ++file){
        // Receive results from a worker
        char resultMsg[RESULT_MESSAGE_SIZE];
        MPI_Status status;

        // Receive a message from the worker
        MPI_Recv(resultMsg,             /* message buffer */
            0,   /* buffer size */
            MPI_CHAR,       /* data item is an integer */
            MPI_ANY_SOURCE,     /* Recieve from thread */
            MPI_ANY_TAG,        /* tag */
            MPI_COMM_WORLD,     /* default communicator */
            &status
        );

        // Get return message
        std::string returnLine(resultMsg);
        #if DEBUG_MESSAGES
        std::cout << returnLine << std::endl;
        #endif      

        // TODO: merge results with global results

        // Send new task
        const int sourceCaught = status.MPI_SOURCE;

        // Compose message consisting of "<file path to parse>, <search vector>"
        char *msg = (char *) malloc( LINE_MESSAGE_SIZE * sizeof(char));
        std::string file_path(file->generic_string());
        // std::cout << file_path << std::endl;
        size_t length = file_path.length();
        file_path.copy(msg, length);
        strncpy(msg+length, ", ", 2);
        strncpy((msg+length+2), searchVector, LINE_SIZE);

        MPI_Send(msg,          /* message buffer */
            LINE_MESSAGE_SIZE, /* buffer size */
            MPI_CHAR,          /* data item is an integer */
            sourceCaught,      /* destination process rank */
            PROCESS,           /* user chosen message tag */
            MPI_COMM_WORLD);   /* default communicator */
        free(msg);
    }


    // Collect remaining results
    for(int rank = 1 ; rank < threadCount ; rank ++){

        // Receive results from a worker
        char resultMsg[RESULT_MESSAGE_SIZE];
        MPI_Status status;

        // Receive a message from the worker
        MPI_Recv(resultMsg,     /* message buffer */
            0,/* buffer size */
            MPI_CHAR,           /* data item is an integer */
            rank,               /* Recieve from thread */
            MPI_ANY_TAG,        /* tag */
            MPI_COMM_WORLD,     /* default communicator */
            &status
        );

        // Get return message
        std::string returnLine(resultMsg);
        #if DEBUG_MESSAGES
        // std::cout << returnLine << std::endl;
        #endif      

        // TODO: merge results with global results

    }

    // Send each thread a terminate signal
    for (int rank = 1; rank < threadCount; ++rank){
        MPI_Send(0, 0, MPI_INT, rank, TERMINATE, MPI_COMM_WORLD);
    }

    return;
}


void cmoz::workerParseFile(){
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    char msg[LINE_MESSAGE_SIZE];
    MPI_Status status;

    while(1){
        MPI_Recv(msg,           /* message buffer */
            LINE_MESSAGE_SIZE,     /* buffer size */
            MPI_CHAR,       /* data item is an integer */
            0,              /* Receive from master */
            MPI_ANY_TAG,        
            MPI_COMM_WORLD,     /* default communicator */
            &status
        );

        // Check for teminate tag
        if(status.MPI_TAG == TERMINATE){
            #if DEBUG_MESSAGES
            std::cout << rank << " received a terminate signal" << std::endl;
            #endif
            return;
        }

        // Convert message into string
        std:string messageReceived(msg);

        #if DEBUG_MESSAGES
        // std::cout << messageReceived << std::endl;
        #endif

        // Parse message
        std::vector<std::string> messages;
        boost::split(messages, messageReceived, boost::is_any_of(", "));
        #if DEBUG_MESSAGES         
            // std::cout << "found token count = " << messages.size() << std::endl << messages[0] << " , " << messages[1] << " , " << messages[2] << std::endl;
        #endif 

        // TODO: read in file name passed in, process against search vector, and send list back

        MPI_Send( 0, 0, MPI_INT, 0, PROCESS, MPI_COMM_WORLD);
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










