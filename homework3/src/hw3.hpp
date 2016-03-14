#include <stdlib.h>
#include <iostream>
#include <map>
#include <vector>
#include <math.h>
#include <sys/shm.h>
// #include <boost/thread.hpp>
#include "MosnickThread.hpp"


using namespace std;

#define NUM_FLOATS 4097
#define FNAME_SIZE 256
#define FLOAT_CHARS 47
#define LINE_SIZE FNAME_SIZE + ( NUM_FLOATS * (FLOAT_CHARS + 3) )
#define DELIMS ","


int read_in_file(FILE *infile, map<string, uint> &fnames, vector< pair< uint, vector<float> > > &lines);
bool is_float(const char* token);
bool process_query(map<string, uint> &fnames, vector< pair< uint, vector<float> > > &lines, string queryFilename, int numResults, int numProcesses);
bool process_query(map<string, uint> &fnames, vector< pair< uint, vector<float> > > &lines, string queryFilename, int numResults, int numProcesses, bool isInterleaved);
string find_fname_by_linenum(map<string, uint> &fnames, uint lineNum);







