#ifndef header

#include <stdlib.h>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <vector>
#include <cstring>
#include <chrono>
#include <ctime>

using namespace std;

#define NUM_FLOATS 4097
#define FNAME_SIZE 256
#define FILE_T_SIZE return sizeof(file_t)
#define FLOAT_CHARS 47
#define LINE_SIZE FNAME_SIZE + ( NUM_FLOATS * (FLOAT_CHARS + 3) )

#define DELIMS ","

#endif

typedef struct file_t{
	size_t size;
	char fname[256];
	float nums[4096];
}file_t;

ssize_t get_token_length(char* token);
bool is_float(char* token);
void print_vector(std::vector<float> vector);
// void print_file(file_t* file);
void destroy_vector( std::vector<file_t*> vector);
void destroy_file(file_t* file);


int main(int argc, char const *argv[])
{
	if(argc < 2){
		cout << "Please enter a csv input file." <<endl;
		exit(0);
	}

	cout << argv[1] << endl;

	FILE *infile = fopen(argv[1], "r");
	if(!infile){
		cout << "File invalid" << endl;
		exit(0);
	}

	// Start clock
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

	// Make map of files
	map< string, vector<float> > files;


	char *line = (char*)malloc(sizeof(char) * LINE_SIZE);
	char *token = NULL;
	// Get filename
	size_t file_t_size = (size_t)sizeof(file_t);

	int numLines = 0;
	while(getline(&line, &file_t_size, infile)){
		// std::cout << line << "\n\n" << std::endl;
		// Get first token, the filename
		token = strtok(line, DELIMS);
		if(token){
			// Put token into string
			// size_t size = get_token_length(token);
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
			files[fname] = floats;
			// cout<< "\n\n" << fname << ": " << endl;
			// print_vector(files[fname]);
		}
		else{
			break;
		}
		numLines ++;
	}
	// Print time to process file
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> timeElapsed = end-start;

	// Print number of lines parsed
	cout << "\n\nNumber of lines parsed: " << numLines << endl;
	cout << "\n\nTime to process file: " << timeElapsed.count() << "s" << endl;



	return 0;
}


// Find length of token.  Stops at comma(,), endline(\n), or null term(\0)
// Return -1 on error
ssize_t get_token_length(char* token){
	if(token){
		uint i = 0;
		while(token[i] != ',' && token[i] != '\n' && token[i] != '\0'){
			i++;
		}
		return i;
	}
	return -1;
}

// Iterates through token to check if if it is a float
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

void print_vector(std::vector<float> vector){
	size_t size = vector.size();
	for(int i = 0 ; i< size ; i++){
		cout<<vector[i] << ", ";
	}
}

void destroy_vector( std::vector<file_t*> vector){

}






