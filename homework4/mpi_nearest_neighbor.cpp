#include <mpi.h>
#include <boost/algorithm/string.hpp>
#include <math.h>
#include "directory_scanner.hpp"

#define NUM_FLOATS 4097
#define FNAME_SIZE 256
#define FLOAT_CHARS 47
#define LINE_SIZE FNAME_SIZE + ( NUM_FLOATS * (FLOAT_CHARS + 3) )
#define DELIMS ","

#define DEBUG_MESSAGES 1
#define DEBUG_MESSAGES_2 0
#define DEBUG_MESSAGES_3 0
#define PRINT_FINAL_RESULTS 1

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
    void getResults(int numResults, std::string filename, std::vector<float> &searchVector, std::vector< std::pair<std::string, float> > &results);
    int  readFile(std::string filename, std::vector<std::pair<std::string, std::vector<float> > > &lines);
    float computeL1Norm(const std::vector<float> *v1, const std::vector<float> *v2);
    void printLines(std::vector<std::pair<std::string, std::vector<float> > > &lines);
    void printResults(std::vector<std::pair <std::string, float> > &results);
    void sortAndCut(int numResults, std::vector<std::pair <std::string, float> > &results);
    bool comp(const std::pair<std::string, float> &el1, const std::pair<std::string, float> &el2);
    bool is_float(const char* token);
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
        #if DEBUG_MESSAGES_2
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
    
    // Create glocal results object
    std::vector<std::pair <std::string, float> > globalResults;


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
        msg[length] = '\0';

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
        MPI_Recv(results,       /* message buffer */
            1,                  /* buffer size */
            cmoz_multipleResults_type,   /* data item is an array of results */
            MPI_ANY_SOURCE,     /* Recieve from thread */
            MPI_ANY_TAG,        /* tag */
            MPI_COMM_WORLD,     /* default communicator */
            &status
        );

        // Merge results with global results
        #if DEBUG_MESSAGES
        std::cout << "\n\n";
        #endif
        for(int i = 0 ; i < numResults ; i++){
            std::string name(results[i].name);
            float temp = results[i].dist;
            #if DEBUG_MESSAGES
            std::cout << name << ": " << temp << std::endl;
            #endif
            globalResults.push_back(std::pair<std::string, float> (name, temp));
        }

        sortAndCut(numResults, globalResults);


        // Send new task
        const int sourceCaught = status.MPI_SOURCE;

        // Compose message consisting of "<file path to parse>"
        char *msg = (char *) malloc( FILE_PATH_SIZE * sizeof(char));
        std::string file_path(file->generic_string());
        // std::cout << file_path << std::endl;
        size_t length = file_path.length();
        file_path.copy(msg, length);
        msg[length] = '\0';

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
        MPI_Recv(results,      /* message buffer */
            1,                  /* buffer size */
            cmoz_multipleResults_type,   /* data item is an array of results */
            MPI_ANY_SOURCE,     /* Recieve from thread */
            MPI_ANY_TAG,        /* tag */
            MPI_COMM_WORLD,     /* default communicator */
            &status
        );

        // Merge results with global results
        #if DEBUG_MESSAGES
        std::cout << "\n\n";
        #endif
        for(int i = 0 ; i < numResults ; i++){
            std::string name(results[i].name);
            float temp = results[i].dist;
            #if DEBUG_MESSAGES
            std::cout << name << ": " << temp << std::endl;
            #endif
            globalResults.push_back(std::pair<std::string, float> (name, temp));
        }

        sortAndCut(numResults, globalResults);
    }

    // Send each thread a terminate signal
    for (int rank = 1; rank < threadCount; ++rank){
        MPI_Send(0, 0, MPI_INT, rank, TERMINATE, MPI_COMM_WORLD);
    }


    // Print final results:
    #if PRINT_FINAL_RESULTS
    printResults(globalResults);
    #endif
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
    std::vector<float> searchVector(NUM_FLOATS);
    getSearchVector(search_vector_file, searchVector);

    char msg[FILE_PATH_SIZE];
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
            #if DEBUG_MESSAGES_2
            std::cout << "Thread" << rank << " received a terminate signal" << std::endl;
            #endif
            return;
        }

        // Convert message into string, filename
        std::string messageReceived(msg);
        #if DEBUG_MESSAGES
        std::cout <<"Processing " << messageReceived << "..." << std::endl;
        #endif






        // read in / check file name passed in, process against search vector, and send list back
        std::vector< std::pair < std::string, float> > results;
        #if DEBUG_MESSAGES
        std::cout << "About to get results from " << messageReceived << std::endl;
        #endif

        getResults(numResults, messageReceived, searchVector, results);

        #if DEBUG_MESSAGES
        std::cout << "Got results for " << messageReceived << std::endl;
        #endif




        // Send back array <<filename>, <dist>> of size results
        resultMessage_t sendResults[numResults];
        for(int i=0 ; i < numResults ; i ++){
            results[i].first.copy(&(sendResults[i].name[0]), results[i].first.length()+1);
            sendResults[i].dist = results[i].second;
            // std::cout << sendResults[i].name[0] << ": " << sendResults[i].dist << std::endl;
        }
        MPI_Send( 
            &sendResults,               /*Send buffer object, array of cmoz_result_types*/
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
        std::cout << "Error reading in search vector in thread" << rank << std::endl;
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
            if(token && cmoz::is_float(token)){
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

void cmoz::getResults(int numResults, std::string filename, std::vector<float> &searchVector, std::vector<std::pair <std::string, float> > &results){
    results.clear();
    if(numResults < 1){
        #if DEBUG_MESSAGES
        std::cout << "No point in processing if no results desired" << std::endl;
        #endif
        MPI_Finalize();
        return;
    }

    filename[filename.length()] = '\0';
    std::vector<std::pair<std::string, std::vector<float> > > lines;
    
    // Parse file
    #if DEBUG_MESSAGES
    std::cout << "About to read file " << filename << std::endl;
    #endif

    readFile(filename, lines);

    #if DEBUG_MESSAGES
    std::cout << "Read in file " << filename << std::endl;
    #endif
    #if DEBUG_MESSAGES_3
    printLines(lines);
    #endif

    // Get distances from lines, add to result vector
    std::vector<std::pair<std::string, std::vector<float> > >::iterator itr = lines.begin();

    for( ; itr != lines.end() ; itr++){
        float temp = computeL1Norm(&(itr->second), &searchVector);
        results.push_back(std::pair<std::string, float> (itr->first, temp));
    }

    #if DEBUG_MESSAGES
    std::cout << "Got distances for " << filename << std::endl;
    #endif
    
    // partial sort and cut results
    sortAndCut(numResults, results);

    #if DEBUG_MESSAGES
    std::cout << "Sorted and cut results from " << filename << std::endl;
    #endif

    return;
}

int cmoz::readFile(std::string filename, std::vector<std::pair<std::string, std::vector<float> > > &lines){
    FILE *infile = fopen(filename.c_str(), "r");
    if(!infile){
        std::cout << "File "<< filename.c_str() <<" invalid." << std::endl;
        std::cout << "File "<< filename <<" invalid." << std::endl;
        MPI_Finalize();
        exit(0);
    }

    char *line = (char*)malloc(sizeof(char) * LINE_SIZE);
    char *token = NULL;

    // Get line
    size_t lineSize = LINE_SIZE;
    int numLines = 0;
    while(getline(&line, &lineSize, infile)){
        // Get first token, the filename
        token = strtok(line, DELIMS);
        if(token){
            // Put token into string
            std::string fname (token);
            #if DEBUG_MESSAGES
            std::cout << fname << std::endl;
            #endif

            // read in floats
            std::vector<float> floats(NUM_FLOATS);
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

            // Add fname and vector to pair
            std::pair <std::string, std::vector<float> > temp(fname, floats);
            lines.push_back(temp);
        }
        else{
            break;
        }
        numLines ++;
    }
    free(line);
    fclose(infile);

    return numLines;
}

float cmoz::computeL1Norm(const std::vector<float> *v1, const std::vector<float> *v2){
    if(v1 && v2){
        int s1 = (*v1).size(),
            s2 = (*v2).size(),
            i=0;
        // Take smallest size
        int size = (s1 < s2) ? s1:s2;
        if(size != 4097){
            std::cout << "\nSize = " << size << std::endl;
        }
        float sum = 0;
        for( ; i < size ; i++){
            sum += fabs( ((*v1)[i] - (*v2)[i]) );
        }
        return (float)sum/size;
    }
    return -1;
}

void cmoz::printLines(std::vector<std::pair<std::string, std::vector<float> > > &lines){
    std::vector<std::pair<std::string, std::vector<float> > >::iterator itr = lines.begin();
    for( ; itr != lines.end() ; itr++){
        std::cout << itr->first << ": " << itr->second[0] << std::endl;   
    }
}

void cmoz::printResults(std::vector<std::pair <std::string, float> > &results){
    std::vector<std::pair<std::string, float> >::iterator itr = results.begin();
    std::cout << "\n\nFINAL RESULTS\n\n_____________" << std::endl;
    for( ; itr != results.end() ; itr++){
        std::cout << itr->first << ": " << itr->second << std::endl;
    }
}

void cmoz::sortAndCut(int numResults, std::vector<std::pair <std::string, float> > &results){
    std::vector<std::pair<std::string, float> >::iterator middle = results.begin() + numResults;
    partial_sort(results.begin(), middle, results.end(), &cmoz::comp);
    results.resize(numResults);
}

bool cmoz::comp(const std::pair<std::string, float> &el1, const std::pair<std::string, float> &el2){
    return (el1.second < el2.second);
}

// Iterates through token to check if it is a float
bool cmoz::is_float(const char* token){
    if(token){
        uint i = 0;
        while(token[i] != ',' && token[i]!='\n' && token[i]!='\0'){
            // Check if char is . or 0-9.  If not, return false
            if(! ((token[i] == '.') || (token[i]>= '0' && token[i]<= '9')) ){
                return false;
            }
            i++;
        }
        return true;
    }
    return false;
}









