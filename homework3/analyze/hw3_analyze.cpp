#include "../src/hw3.hpp"
#include <stdio.h>
#define ITR 100


void compareNumThreads(const char* fileName, uint numThreads, uint numResults);
void compareFileSize(const char* fileNames[3], uint numThreads, uint numResults);

int main(int argc, char * argv[])
{
	
	// ---------------------------------------------
	// BEGIN: Timing Analysis
	// ---------------------------------------------
	std::cout << "Running Timing Analysis" << std::endl
		  << "-----------------------" << std::endl;
	// const unsigned int PROCS = 12;
	const unsigned int RESULTS = 100;
	
	char fname1[] = "../../4200_HPC.csv\0";
	char fname2[] = "../../6300_HPC.csv\0";
	char fname3[] = "../../8400_HPC.csv\0";

	char* fnames[] = { (char *)&fname1, (char*)&fname2, (char *)&fname3};

	// Run tests for vary numebr of threads
	// compareNumThreads("../../8400_HPC.csv", PROCS, RESULTS);

	// Run test for varying file sizes
	compareFileSize((const char **)fnames, 4, RESULTS);
	// Run test for varying ???



}


void compareNumThreads(const char* fileName, uint numThreads, uint numResults){
	
	// ---------------------------------------------
	// Read in file to test on
	// ---------------------------------------------
	map< string, uint > fnames;
	vector< pair< uint, vector<float> > > lines;

	FILE *infile = fopen(fileName, "r");

	int numLines = read_in_file(infile, fnames, lines);
	std::cout << numLines << std::endl;
	string queryFilename("agricultural/agricultural00_rot_000.tif");

	for (int i = 1 ; i <= numThreads ; i++){
		// Run Timing Experiment
        std::chrono::high_resolution_clock c;
        std::chrono::high_resolution_clock::time_point start = c.now();
		for (unsigned int j = 0; j < ITR; ++j){
			// This is an assignemnt of a call to a object functor;
			process_query(fnames, lines, queryFilename, numResults, i, true);
		}
        std::chrono::high_resolution_clock::time_point stop = c.now();
		double avgMs = (double) std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / (1000000 * ITR);

		// Compute Ops and Elements
		const unsigned long elements = numLines;
		// Log timing statistics
		std::cout << "------------------------------------------------------------------" << std::endl
			  << ITR << " iterations of file querying run using " << fileName << std::endl
			  << "Average Time (s): " << avgMs<< std::endl
			  << "Data  Point:interleavethreads: " << avgMs << ":" << i << ":" << elements << std::endl;
		std::cout << "Data Point:threadTimethreads: " << i << ":" << avgMs << std::endl;
	}
	std::cout << "Timing Analysis Completed for varying thread counts" << std::endl
		  << "=========================" << std::endl;
}

void compareFileSize(const char* fileNames[3], uint numThreads, uint numResults){
	// ---------------------------------------------
	// Read in file to test on
	// ---------------------------------------------
	for(int i = 0 ; i <3 ; i++){
		map< string, uint > fnames;
		vector< pair< uint, vector<float> > > lines;

		FILE *infile = fopen(fileNames[i], "r");

		int numLines = read_in_file(infile, fnames, lines);
		std::cout << numLines << std::endl;
		string queryFilename("agricultural/agricultural00_rot_000.tif");
			
		// Run Timing Experiment
	    std::chrono::high_resolution_clock c;
	    std::chrono::high_resolution_clock::time_point start = c.now();
		for (unsigned int j = 0; j < ITR; ++j){
			process_query(fnames, lines, queryFilename, numResults, numThreads, true);
		}
	    std::chrono::high_resolution_clock::time_point stop = c.now();
		double avgMs = (double) std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / (1000000 * ITR);

		// Compute Ops and Elements
		const unsigned long elements = numLines;
		// Log timing statistics
		std::cout << "------------------------------------------------------------------" << std::endl
			  << ITR << " iterations of file querying run using " << fileNames[i] << std::endl
			  << "Average Time (s): " << avgMs<< std::endl
			  << "Data  Point:interleavefiles: " << avgMs << ":" << elements << ":" << i << std::endl;
		std::cout << "Data Point:fileTimefiles: " << i << ":" << avgMs << std::endl;
		fnames.clear();
		lines.clear();
		fclose(infile);
	}
	std::cout << "Timing Analysis Completed for varying file sizes" << std::endl
		  << "=========================" << std::endl;
}
