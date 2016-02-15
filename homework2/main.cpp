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
#include <math.h>

using namespace std;

#define NUM_FLOATS 4097
#define FNAME_SIZE 256
// #define FILE_T_SIZE return sizeof(file_t)
#define FLOAT_CHARS 47
#define LINE_SIZE FNAME_SIZE + ( NUM_FLOATS * (FLOAT_CHARS + 3) )

#define DELIMS ","

#endif

ssize_t get_token_length(char* token);
bool is_float(char* token);
void print_vector(std::vector<float> vector);
void find_bounding_min_max(map<string, vector<float>> fileMap, vector<vector<float>> *minMaxVector);


int main(int argc, char const *argv[])
{
	if(argc < 2){
		cout << "Please enter a csv input file." <<endl;
		exit(0);
	}
	// cout << argv[1] << endl;
	FILE *infile = fopen(argv[1], "r");
	if(!infile){
		cout << "File invalid" << endl;
		exit(0);
	}

	/***************************
	Part one: Read in file, store in map
	****************************/
	// Start clock
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

	// Make map of files
	map< string, vector<float> > files;

	char *line = (char*)malloc(sizeof(char) * LINE_SIZE);
	char *token = NULL;

	// Get line
	size_t lineSize = LINE_SIZE;
	int numLines = 0;
	while(getline(&line, &lineSize, infile)){
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
	free(line);



	/***************************
	Part two: find bounding min and max of each column
	****************************/
	vector<vector<float>> minMaxVector({vector<float>(NUM_FLOATS, INFINITY), vector<float>(NUM_FLOATS, -INFINITY)});
	// print_vector(minMaxVector[0]);
	// print_vector(minMaxVector[1]);
	
	start = std::chrono::system_clock::now();
	find_bounding_min_max(files, &minMaxVector);
    end = std::chrono::system_clock::now();
    timeElapsed = end-start;
    cout << "\n\nTime to process Columns: " << timeElapsed.count() << "s" << endl;

	// cout << "\n\nColumn minimums:\n";
	// print_vector(minMaxVector[0]);
	// cout << "\n\nColumn maximums:\n";
	// print_vector(minMaxVector[1]);


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

void print_vector(vector<float> vector){
	size_t size = vector.size();
	for(int i = 0 ; i< size ; i++){
		cout<<vector[i] << ", ";
	}
}

void find_bounding_min_max(map<string, vector<float>> fileMap, vector<vector<float>> *minMaxVector){
	map<string,vector<float>>::iterator it = fileMap.begin();
	// Iterate through each row
	for (; it != fileMap.end(); ++it){
		// cout << it->first << endl;
		// Iterate through each fo the row's floats, compare to min max vector
		size_t size = it->second.size();
		int i = 0;
		for(; i < size  && i< NUM_FLOATS; i++){
			// Compare to min
			if(it->second[i] < (*minMaxVector)[0][i]){
				(*minMaxVector)[0][i] = it->second[i];
			}
			// Compare to max
			else if(it->second[i] > (*minMaxVector)[1][i]){
				(*minMaxVector)[1][i] = it->second[i];
			}
		}
		// cout << i <<endl;
	}
}





