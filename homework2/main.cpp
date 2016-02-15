#ifndef header

#include <stdlib.h>
#include <iostream>
#include <map>
#include <vector>

using namespace std;

#define NUM_FLOATS 4097
#define FNAME_SIZE 256
#define FLOAT_CHARS 47
#define LINE_SIZE FNAME_SIZE + ( NUM_FLOATS * (FLOAT_CHARS + 3) )
#define DELIMS ","

#endif

int read_in_file(map< string, vector<float> > *files, FILE *infile);
bool is_float(char* token);
void print_vector(std::vector<float> *vector);

int main(int argc, char const *argv[])
{
	if(argc < 5){
		cout << "Please enter (1) a filename query\n \
		(2) a csv input file\n \
		(3) the number fo results to generate\n\
		(4) the number of processes." <<endl;
		exit(0);
	}
	// cout << argv[1] << endl;
	FILE *infile = fopen(argv[2], "r");
	if(!infile){
		cout << "File invalid" << endl;
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
	cout << "\n\nTime to process file: " << timeElapsed.count() << "s" << endl;

	

	/***************************
	Part two: Perform query
	****************************/
	start = std::chrono::system_clock::now();
	// find_bounding_min_max(files, &minMaxVector);
    end = std::chrono::system_clock::now();
    timeElapsed = end-start;
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
bool is_float(char* token){
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


