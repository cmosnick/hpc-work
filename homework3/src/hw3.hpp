// #define header
#include <stdlib.h>
#include <iostream>
#include <map>
#include <vector>
#include <math.h>
#include <sys/shm.h>
#include <boost/thread.hpp>
#include "MosnickThread.hpp"


using namespace std;

#define NUM_FLOATS 4097
#define FNAME_SIZE 256
#define FLOAT_CHARS 47
#define LINE_SIZE FNAME_SIZE + ( NUM_FLOATS * (FLOAT_CHARS + 3) )
#define DELIMS ","

// class MosnickThread;

typedef struct fnameToLineNum{
	char fname[FNAME_SIZE];
	uint lineNum;
}fnameToLineNum_t;

typedef struct lineNumToValues{
	uint lineNum;
	float values[NUM_FLOATS];
}lineNumToValues_t;

// typedef struct lineDistance{
// 	uint lineNum;
// 	float distance;
// }lineDistance_t;

typedef struct childInfo{
	lineDistance_t *start;
	lineDistance_t *end;
	uint startLine;
	uint endLine;
	uint shmStartLine;
	uint shmEndline;
	// uint queryLine;	// Line of queried filename, so the result doent show up as 0
}childInfo_t;


int read_in_file(FILE *infile, map<string, uint> &fnames, vector< pair< uint, vector<float> > > &lines);
bool is_float(const char* token);
void print_vector(std::vector<float> *vector);
bool process_query(map<string, uint> &fnames, vector< pair< uint, vector<float> > > &lines, string queryFilename, int numResults, int numProcesses);
// float compute_L1_norm(const vector<float> *v1, const vector<float> *v2);
// bool do_work(int processNumber, childInfo_t childInfo, const vector<float> *targetVector, const vector<pair<uint, vector<float> > > &lines, int numResults);
void print_lines(vector<pair<uint, vector<float> > > &lines);
void print_filenames(map<string, uint> &fnames);
bool isLine(const pair<uint, vector<float> > pair, uint lineNum);
void print_line_distances(vector<pair<uint, float> > &lineDistances);
// bool comp(pair<uint, float> &el1, pair<uint, float> &el2);
string find_fname_by_linenum(map<string, uint> &fnames, uint lineNum);







