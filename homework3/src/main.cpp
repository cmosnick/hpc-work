#include "hw3.hpp"

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
	map< string, uint > fnames;
	vector< pair< uint, vector<float> > > lines;

	// Start clock
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

	int numLines = read_in_file(infile, fnames, lines);
	
	end = std::chrono::system_clock::now();
    std::chrono::duration<double> timeElapsed1 = end-start;

	if(numLines <= 0){
		cout << "\n\nFile read was unsucessful."<< endl;
		exit(0);
	}

	


	/***************************
	Part two: Perform query
	****************************/
	start = std::chrono::system_clock::now();

	bool success = process_query(fnames, lines, queryFilename, numResults, numProcesses);

    end = std::chrono::system_clock::now();
    std::chrono::duration<double> timeElapsed2 = end-start;

    if(!success){
    	cout << "\n\nUnsucessful processing of query" << endl;
    	exit(0);
    }
    cout << "\n\nNumber of lines parsed: " << numLines << endl;
    cout << "Time to process file: " << timeElapsed1.count() << "s" << endl;
    cout << "Time to process query: " << timeElapsed2.count() << "s" << endl;


	return 0;
}