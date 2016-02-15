#ifndef header

#include <stdlib.h>
#include <iostream>
#include <map>
#include <vector>
#include <math.h>

using namespace std;

#define NUM_FLOATS 4097
#define FNAME_SIZE 256
#define FLOAT_CHARS 47
#define LINE_SIZE FNAME_SIZE + ( NUM_FLOATS * (FLOAT_CHARS + 3) )
#define DELIMS ","

#endif

int read_in_file(map< string, vector<float> > *files, FILE *infile);
bool is_float(const char* token);
void print_vector(std::vector<float> *vector);
bool process_query(map< string, vector<float> > *files, string queryFilename, int numResults, int numProcesses);
float compute_L1_norm(vector<float> *v1, vector<float> *v2);

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

	// Start clock
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

	int numLines = read_in_file(&files, infile);
	
	end = std::chrono::system_clock::now();
    std::chrono::duration<double> timeElapsed = end-start;

	if(numLines <= 0){
		cout << "\n\nFile read was unsucessful."<< endl;
		exit(0);
	}

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
int read_in_file(map< string, vector<float> > *files, FILE *infile){
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
		float difference = compute_L1_norm( &((*files)[queryFilename]), &((*files)["agricultural/agricultural05.tif"]) );

		cout << "\n\nDifference: " << difference << endl;

		return true;
	}
	return false;
}

// Computes the L1 norm between two vectors of floats
// Returns 0 on error or if vectors are exactly alike
float compute_L1_norm(vector<float> *v1, vector<float> *v2){
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

