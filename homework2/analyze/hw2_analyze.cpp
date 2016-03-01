#include "../src/main.hpp"

int main(int argc, char * argv[])
{
	
	// ---------------------------------------------
	// BEGIN: Timing Analysis
	// ---------------------------------------------
	std::cout << "Running Timing Analysis" << std::endl
		  << "-----------------------" << std::endl;
	const unsigned int ITR= 1000;
	const unsigned int PROCS = 12;
	const unsigned int RESULTS = 100;
	
	// ---------------------------------------------
	// Read in file to test on
	// ---------------------------------------------
	map< string, vector<float> > files;
	map< string, uint > fnames;
	vector< pair< uint, vector<float> > > lines;


// Change this to 1 to enable larger file, should modify to name of larger file.  I haven't tried it out yet.
#if 0
	const char fname[16] = "../8400_HPC.csv";
#else
	const char fname[16] = "../8400_HPC.csv";
#endif

	FILE *infile = fopen(fname, "r");
	if(!infile){
		cout << "File invalid" << endl;
		exit(0);
	}
	int numLines = read_in_file(&files, infile, fnames, lines);
	std::cout << numLines << std::endl;
	string queryFilename("agricultural/agricultural00.tif");
	// ***********************************
	// Test functor : Your implementation
	// ***********************************

	// Iterate through  1-12 processes
	for (int i = 1 ; i <= PROCS ; i++)
	{
		// Run Timing Experiment
                std::chrono::high_resolution_clock c;
                std::chrono::high_resolution_clock::time_point start = c.now();
		for (unsigned int i = 0; i < ITR; ++i)
		{
			// This is an assignemnt of a call to a object functor;
			process_query(fnames, lines, queryFilename, RESULTS, i, false);
		}
                std::chrono::high_resolution_clock::time_point stop = c.now();
		double avgMs = (double) std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / (1000000 * ITR);

		// Compute Ops and Elements
		const unsigned long elements = numLines;
		// Log timing statistics
		std::cout << "------------------------------------------------------------------" << std::endl
			  << ITR << " iterations of file querying ran using " << fname << std::endl
			  << "Average Time (s): " << avgMs<< std::endl
			  << "Data  Point:f:" << avgMs << ":" << i << ":" << elements << std::endl;
	}
	std::cout << "Timing Analysis Completed" << std::endl
		  << "=========================" << std::endl;
	// ---------------------------------------------
	// END: Timing Analysis
	// ---------------------------------------------
}

