#ifndef header

#include <stdlib.h>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <vector>
#include <cstring>

using namespace std;

#define NUM_FLOATS 4096
#define FNAME_SIZE 256
#define FILE_T_SIZE return sizeof(file_t)
#define FLOAT_CHARS 47


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
	// Make map of files
	map< string, vector<float> > files;


	char *line = (char*)malloc(sizeof(char) * (FNAME_SIZE + (NUM_FLOATS +2)));
	char delims[] = ",\0";
	char *token = NULL;
	// Get filename
	size_t file_t_size = (size_t)sizeof(file_t);
	while(getline(&line, &file_t_size, infile)){
		// std::cout << line << "\n\n" << std::endl;
		// Get first token, the filename
		token = strtok(line, delims);
		if(token){
			//Instantiate struct
			// std::cout << token << "\n\n" << std::endl;
			size_t size = get_token_length(token);
			// char *fnameChar = (char *)malloc(sizeof(char)*(size+1));
			// strncpy(fnameChar, token, size);
			// fnameChar[size] = '\0';
			// Get file name from first token
			std::string fname (token, size);
			// free(fnameChar);
			// cout << fname << "\n\n" << endl;

			// read in floats
			vector<float> floats(NUM_FLOATS);
			// files.insert(pair<string, vector<float>>(fname, vector<float>(NUM_FLOATS)));

			uint i = 0;
			do{
				token = strtok(NULL, delims);
				if(token && is_float(token)){
					float temp = atof(token);
					// files[fnameChar].push_back(temp);
					floats[i]=temp;
					// std::cout << temp;
				}
				else 
					break;
				i++;
			}while( i <= NUM_FLOATS);

			// Add file to map
			// files.insert(pair<string, vector<float>>(fname, floats));
			files[fname] = floats;
			cout<< fname << ": ";
			print_vector(files[fname]);
			cout<< endl <<endl;
		}
		else{
			break;
		}
	}
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
		cout<<vector[i];
	}
}

void destroy_vector( std::vector<file_t*> vector){

}






