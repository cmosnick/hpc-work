#ifndef header

#include <stdlib.h>
#include <iostream>
#include <map>
#include <vector>
#include <math.h>
#include <sys/shm.h>

using namespace std;

#define NUM_FLOATS 4097
#define FNAME_SIZE 256
#define FLOAT_CHARS 47
#define LINE_SIZE FNAME_SIZE + ( NUM_FLOATS * (FLOAT_CHARS + 3) )
#define DELIMS ","

#endif

typedef struct fnameToLineNum{
	char fname[FNAME_SIZE];
	uint lineNum;
}fnameToLineNum_t;

typedef struct lineNumToValues{
	uint lineNum;
	float values[NUM_FLOATS];
}lineNumToValues_t;

typedef struct lineDistance{
	uint lineNum;
	float distance;
}lineDistance_t;

typedef struct childInfo{
	lineDistance_t *start;
	lineDistance_t *end;
}childInfo_t;

int read_in_file(map< string, vector<float> > *files, FILE *infile, map<string, uint> &fnames, vector< pair< uint, vector<float> > > &lines);
bool is_float(const char* token);
void print_vector(std::vector<float> *vector);
bool process_query(map< string, vector<float> > *files, string queryFilename, int numResults, int numProcesses);
float compute_L1_norm(const vector<float> *v1, const vector<float> *v2);
bool do_work(int processNumber, childInfo_t *childInfo);
bool shm_setup(size_t size, lineDistance_t *shm);
bool copy_vector_into_array( std::vector<float> *v, float *array );
void print_lines(vector<pair<uint, vector<float>>> &lines);
void print_filenames(map<string, uint> &fnames);




int main(int argc, char const *argv[])
{
	// Check args
	if(argc < 5){
		cout << "Please enter (1) a filename query\n \
		(2) a csv input file\n \
		(3) the number fo results to generate\n\
		(4) the number of processes." <<endl;
		exit(0);
	}
	// Check input file
	FILE *infile = fopen(argv[2], "r");
	if(!infile){
		cout << "File invalid" << endl;
		exit(0);
	}
	/***************************
	Part zero: Get args
	****************************/
	string queryFilename(argv[1]);
	// number of results
	int numResults = 0;
	if(is_float(argv[3])){
		numResults = atoi(argv[3]);
	}
	else{
		cout << "Invalid number for argument 3" << endl;
		exit(0);
	}
	// number of processes
	int numProcesses = 0;
	if(is_float(argv[4])){
		numProcesses = atoi(argv[4]);
	}
	else{
		cout << "Invalid number for argument 4" << endl;
		exit(0);
	}

	/***************************
	Part one: Read in file, store in map
	****************************/
	// Make map of files
	map< string, vector<float> > files;
	map< string, uint > fnames;
	vector< pair< uint, vector<float> > > lines;

	// Start clock
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

	int numLines = read_in_file(&files, infile, fnames, lines);
	
	end = std::chrono::system_clock::now();
    std::chrono::duration<double> timeElapsed = end-start;

	if(numLines <= 0){
		cout << "\n\nFile read was unsucessful."<< endl;
		exit(0);
	}

	print_filenames(fnames);
	print_lines(lines);
	// Print database map
	// map< string, vector<float> >::iterator itr=files.begin(); 
	// for(; itr != files.end() ; itr++){
	// 	print_vector(&(itr->second));
	// }

	// Print number of lines parsed
	cout << "\n\nNumber of lines parsed: " << numLines << endl;
	cout << "Time to process file: " << timeElapsed.count() << "s" << endl;

	

	/***************************
	Part two: Perform query
	****************************/
	start = std::chrono::system_clock::now();

	bool success = process_query(&files, queryFilename, numResults, numProcesses);

    end = std::chrono::system_clock::now();
    timeElapsed = end-start;

    if(!success){
    	cout << "\n\nUnsucessful processing of query" << endl;
    	exit(0);
    }
    cout << "\n\nTime to process query: " << timeElapsed.count() << "s" << endl;


	return 0;
}


// Read csv into map of vectors
int read_in_file(map< string, vector<float> > *files, FILE *infile, map<string, uint> &fnames, vector< pair< uint, vector<float> > > &lines){
	if(files && infile){
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
				fnames[fname] = numLines;

				// read in floats
				vector<float> floats(NUM_FLOATS);
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

				// Add fname and vector to map
				(*files)[fname] = floats;
				// print_vector(&((*files)[fname]));
				pair <uint, vector<float>> temp(numLines, floats);
				lines.push_back(temp);
			}
			else{
				break;
			}
			numLines ++;
		}
		free(line);

		return numLines;
	}
	return 0;
}

// Iterates through token to check if it is a float
bool is_float(const char* token){
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

// Print vector of floats, could be anything really though
void print_vector(vector<float> *vector){
	size_t size = (*vector).size();
	cout << "\n";
	for(int i = 0 ; i< size ; i++){
		cout<<(*vector)[i] << ", ";
	}
}

bool process_query(map< string, vector<float> > *files, string queryFilename, int numResults, int numProcesses){
	if(files && numResults > 0 && numProcesses > 0){
		// Check that requested file to query is in csv file
		if((*files).count(queryFilename) == 0){
			cout << "\n\nError: Queried filename is not in the database.  Query terminated." << endl;
			return false;
		}

		// Test L1 norm function
		// float difference = compute_L1_norm( &((*files)[queryFilename]), &((*files)["agricultural/agricultural05.tif"]) );
		// cout << "\n\nDifference: " << difference << endl;
		int i=0;
		pid_t pids[numProcesses];

		// Reorganize map into contiguous data structure
		int size = (*files).size();
		int numLines = size;
		// fnameToLineNum_t fileNames[size];
		// lineNumToValues_t lines[size];

		// map< string, vector<float> >::iterator itr=(*files).begin(); 
		// for(; itr != (*files).end() ; itr++, i++){
		// 	// copy filename into filenames array
		// 	int len = itr->first.length();
		// 	itr->first.copy( (fileNames[i].fname), len, 0);
		// 	fileNames[i].lineNum = i;
		// 	// copy line numbers and values into line numbers array
		// 	lines[i].lineNum = i;
		// 	if(! copy_vector_into_array( &(itr->second), (lines[i].values)) ){
		// 		cout << "\n\nError: error copying vector into array" << endl;
		// 	}
		// }

		// // Print new structures out be sure they work
		// print_filename_array(fileNames, size);
		// print_lines_array(lines, size);


		// Set up shared memory
		lineDistance_t *shm = NULL;
		size_t shmsize = sizeof(lineDistance_t)*numResults*numProcesses;
		int shmId = shm_setup(shmsize, shm);
		cout << "\n\nShared memory id: " << shmId << endl;

		// Setup step variable to divide shared memory
		lineDistance_t *shmEnd = shm + numLines;
		childInfo_t childInfo[numProcesses];
		int step = (int)(numLines / numProcesses);
		int stepSize = step * sizeof(lineDistance_t);
		for(i = 0 ; i < numProcesses ; i++){
			childInfo[i].start = shm + (stepSize * i);
			// Last process takes extra 
			if( (i+1) == numProcesses ){
				childInfo[i].end = shmEnd;
			}
			else{
				childInfo[i].end = shm + (stepSize * (i+1));
			}
		}

		// Fork process into worker processes
		for(i = 0 ; i < numProcesses ; i++){
			pid_t pid = fork();
			if(pid >= 0){
				if(pid == 0){	// Child process
					// compute work on partition
					// call helper on partition
					// lineDistance_t *start = shm;

					do_work(i+1, &(childInfo[i]));
					exit(0);
				}
				else{			// Parent process
					pids[i] = pid;
					// increment shared memory

				}
			}
			else{
				cout << "\n\nError: fork failed!" << endl;
				// return false;
				break;
			}
		}

		// Wait for processes to finish
		int status;
		for( i = 0 ; i < numProcesses ; i++){
			/*pid_t pid = */wait(&status);
			// cout << "Child with PID " << (long)pid << " exited with status ." << status << "\n" << endl;
		}
		cout << "\n\nProcesses are finished!" << endl;

		// Gather data in shared memory and get final solution


		// Destroy shared memory
		shmctl(shmId, IPC_RMID, NULL);
		shmdt(shm);
		return true;
	}
	return false;
}

// Computes the L1 norm between two vectors of floats
// Returns 0 on error or if vectors are exactly alike
float compute_L1_norm(const vector<float> *v1, const vector<float> *v2){
	if(v1 && v2){
		int s1 = (*v1).size(),
			s2 = (*v2).size(),
			i=0;
		// Take smallest size
		int size = (s1 < s2) ? s1:s2;
		float sum = 0;
		for( ; i < size ; i++){
			sum += fabs( ((*v1)[i] - (*v2)[i]) );
		}
		return sum/size;
	}
	return 0;
}

bool do_work(int processNumber, childInfo_t *childInfo){
	cout << "\nProcess "<< processNumber << endl;
	cout << "Child info: start: " << childInfo->start << " end: " << childInfo->end << endl;
	
	
	return true;
}

// Shm will be an array of [linenumber, dist] structs
bool shm_setup(size_t size, lineDistance_t *shm){
	if(size > 0){
		key_t shmKey = 123456;
		int shmId;
		int shmFlag = IPC_CREAT | 0666; // Flag to create with rw permissions
		if ((shmId = shmget(shmKey, size, shmFlag)) < 0){
			std::cerr << "Init: Failed to initialize shared memory (" << shmId << ")" << std::endl; 
			return 0;
		}

		if ((shm = (lineDistance_t *)shmat(shmId, NULL, 0)) == (lineDistance_t *) -1){
			std::cerr << "Init: Failed to attach shared memory (" << shmId << ")" << std::endl; 
			return 0;
		}
		return shmId;
	}
	return 0;
}

bool copy_vector_into_array( std::vector<float> *v, float *array ){
	int size = v->size();
	for(int i = 0 ; i <size ; i++){
		array[i] = (*v)[i];
	}
	return true;
}

void print_filenames(map<string, uint> &fnames){
	map<string, uint>::iterator itr = fnames.begin();
	for( ; itr != fnames.end() ; itr++){
		cout << itr->first << ": " << itr->second << endl;
	}
}

void print_lines(vector<pair<uint, vector<float>>> &lines){
	int size = lines.size();
	for(int i=0 ; i<size ; i++){
		cout << lines[i].first << ": " << endl;
	}
}




/*Notes******************

For processing: instead of map which is not contiguous,
	use indexing data structure of fn=>line number
	then use 2D array of floats

Try interleaving after doing block partitioning if I have time
	Interleaving is better for cache issues

Lookup by value with a set (can be used for wait with pids)

Think about partial sort for sort:
	fixed-size max heap, load sorted, throw out minvalue when bigger value comes in

***********************/









