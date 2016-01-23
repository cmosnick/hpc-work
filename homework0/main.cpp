#ifndef header

#include <stdlib.h>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <vector>

#define NUM_FLOATS 4096
#define FNAME_SIZE 256
#define FILE_T_SIZE return sizeof(file_t)

#endif

typedef struct file_t{
	size_t size;
	char fname[256];
	float nums[4096];
}file_t;

ssize_t get_token_length(char* token);
bool is_float(char* token);
void print_vector(std::vector<file_t*> vector);
void print_file(file_t* file);
void destroy_vector( std::vector<file_t*> vector);
void destroy_file(file_t* file);


int main(int argc, char const *argv[])
{
	if(argc < 2){
		std::cout << "Please enter a csv input file." <<std::endl;
		exit(0);
	}

	std::cout << argv[1] << std::endl;

	FILE *infile = fopen(argv[1], "r");
	if(!infile){
		std::cout << "File invalid" << std::endl;
		exit(0);
	}
	// Make map of files
	std::map< std::string, std::vector<float> > files;


	char *line = (char*)malloc(sizeof(char) * 256);
	char delims[] = ",";
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

			std::string fname (token, size);
			std::cout << fname << "\n\n" << std::endl;

			// file_t *file = (file_t*)calloc(1, sizeof(file_t));
			
			// Get file name from first token
			// ssize_t token_size = get_token_length(token);
			// Not sure why token size would be 0.  line atrts with comma?  w/e.
			// if(token_size >= 0){
				// strncpy(file->fname, token, token_size);
			// }

			// read in floats
			std::vector<float> floats(NUM_FLOATS);
			uint i;
			do{
				token = strtok(NULL, delims);
				if(token && is_float(token)){
					float temp = atof(token);
					// if(temp != 0){
						// file->nums[i] = temp;
					// }
					// This is literally pointless.  It will be 0.0 value if error anyways.  
					// Insert error handling below if desired.
					// else{
						// file->nums[i] = 0.0;
					// }
					// std::cout << temp;
					floats.push_back(temp);
				}
				else 
					break;
			}while( i < NUM_FLOATS);
			// file->size = i;

			// Add file to map
			// files.push_back(file);
			files[fname] = floats;
		}
		else{
			break;
		}
	}
	// print_vector(files);
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

void print_vector(std::vector<file_t*> vector){
	size_t size = vector.size();
	for(int i = 0 ; i< size ; i++){
		print_file(vector[i]);
	}
}

void print_file(file_t* file){
	if(file){
		std::cout <<file->fname <<std::endl;
		for(int i =0 ; i< file->size ; i++){
			std::cout << file->nums[i] << ", ";
		}
		std::cout << std::endl;
	}
}

void destroy_vector( std::vector<file_t*> vector){

}

void destroy_file(file_t* file){

}






