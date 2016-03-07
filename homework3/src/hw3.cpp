#ifndef header
#include "hw3.hpp"
// #include <boost/thread.hpp>

#endif

// Read csv into map of vectors
int read_in_file(FILE *infile, map<string, uint> &fnames, vector< pair< uint, vector<float> > > &lines){
	if(infile){
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
				fnames[fname] = numLines;

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
				// (*files)[fname] = floats;
				pair <uint, vector<float> > temp(numLines, floats);
				lines.push_back(temp);
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
bool is_float(const char* token){
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

// Print vector of floats, could be anything really though
void print_vector(vector<float> *vector){
	size_t size = (*vector).size();
	cout << "\n";
	for(int i = 0 ; i< size ; i++){
		cout<<(*vector)[i] << ", ";
	}
}

bool process_query(map<string, uint> &fnames, vector< pair< uint, vector<float> > > &lines, string queryFilename, int numResults, int numProcesses){
	if(!fnames.empty() && !lines.empty() && numResults > 0 && numProcesses > 0){
		// Check that requested file to query is in csv file
		if(fnames.count(queryFilename) == 0){
			cout << "\n\nError: Queried filename is not in the database.  Query terminated." << endl;
			return false;
		}

		// Get target vector of file name
		uint queryLineNum = fnames[queryFilename];
		vector<pair<uint, vector<float> > >::iterator itr = lines.begin();
		while(itr != lines.end()){
			if(itr->first == queryLineNum){
				break;
			}
			itr++;
		}
		const vector<float> *queryFloats = &(itr->second);
		
		unsigned int totalLines = fnames.size();
		// Create vector of thread objects
		std::vector<mosnick::MosnickThread *> mosnickThreads(numProcesses);
		for(int i = 0 ; i < numProcesses ; i++){
			mosnickThreads[i] = new mosnick::MosnickThread(numResults, numProcesses, totalLines, queryFloats);
		}
		// Create thread group
		boost::thread_group tg;


		// Create and call threads
		for(int i = 0 ; i < numProcesses ; i++){
			mosnick::MosnickThread *threadObj = (mosnick::MosnickThread *)mosnickThreads[i];
			boost::thread *thread = new boost::thread(boost::bind(&mosnick::MosnickThread::doWorkInterleave, threadObj, i, lines));
			tg.add_thread( thread );
		}

		// Gather threads
		tg.join_all();

		// Gather data in threads and get final solution
		vector<pair<uint, float> > lineDistances;
		// Copy contents of each thread object's results into congragated result.
		for(int i = 0 ; i < numProcesses ; i++){
			for(int j = 0 ; j < numResults ; j++){
				lineDistances.push_back(mosnickThreads[i]->results[j]);
			}
		}

		// Sort aggregated results and cut off
		vector<pair<uint, float> >::iterator middle = lineDistances.begin() + numResults;
		partial_sort(lineDistances.begin(), middle, lineDistances.end(), &mosnick::MosnickThread::comp);
		lineDistances.resize(numResults);
		
		
		// Match with filenames, print
		vector<pair<uint, float> >::iterator lDItr = lineDistances.begin();
		while(lDItr < lineDistances.end()){
			cout << find_fname_by_linenum(fnames, lDItr->first);
			cout << ":  " << lDItr->second << endl;
			lDItr++;
		}

		// Free thread objects
		for(int i = 0 ; i < numProcesses ; i++){
			delete mosnickThreads[i];
		}

		return true;
	}
	return false;
}

// Computes the L1 norm between two vectors of floats
// Returns 0 on error or if vectors are exactly alike
// float compute_L1_norm(const vector<float> *v1, const vector<float> *v2){
// 	if(v1 && v2){
// 		int s1 = (*v1).size(),
// 			s2 = (*v2).size(),
// 			i=0;
// 		// Take smallest size
// 		int size = (s1 < s2) ? s1:s2;
// 		if(size != NUM_FLOATS){
// 			cout << "\nSize = " << size << endl;
// 		}
// 		float sum = 0;
// 		for( ; i < size ; i++){
// 			sum += fabs( ((*v1)[i] - (*v2)[i]) );
// 		}
// 		return (float)sum/size;
// 	}
// 	return -1;
// }

// bool do_work(int processNumber, childInfo_t childInfo, const vector<float> *targetVector, const vector<pair<uint, vector<float> > > &lines, int numResults){	// process files
// 	uint startLine = childInfo.startLine;
// 	uint endLine = childInfo.endLine;

// 	// Create vecator of pair <uint, float> to heapify later
// 	vector<pair<uint, float> > lineDistances;

// 	// iterate though lines to get distances
// 	while(startLine < endLine){
// 		float temp = mosnick::MosnickThread::compute_L1_norm(targetVector, &(lines[startLine].second));
// 		pair<uint, float> tempPair;
// 		tempPair.first = startLine;
// 		tempPair.second = temp;
// 		lineDistances.push_back(tempPair);
// 		startLine++;
// 	}

// 	// Sort to get top results (shortest distance)
// 	vector<pair<uint, float> >::iterator middle = lineDistances.begin() + numResults;
// 	partial_sort(lineDistances.begin(), middle, lineDistances.end(), &mosnick::MosnickThread::comp);
// 	lineDistances.resize(numResults);


// 	// Store in shared memory
// 	lineDistance_t *shm = childInfo.start;
// 	for(int i = 0, j = 0 ; i < numResults ; i++, j++){
// 		shm[i].lineNum = lineDistances[j].first;
// 		shm[i].distance = lineDistances[j].second;
// 	}
// 	return true;
// }


void print_filenames(map<string, uint> &fnames){
	map<string, uint>::iterator itr = fnames.begin();
	for( ; itr != fnames.end() ; itr++){
		cout << itr->first << ": " << itr->second << endl;
	}
}

void print_lines(vector<pair<uint, vector<float> > > &lines){
	int size = lines.size();
	for(int i=0 ; i<size ; i++){
		cout << lines[i].first << ": " << endl;
	}
}

void print_line_distances(vector<pair<uint, float> > &lineDistances){
	vector<pair<uint, float> >::iterator itr = lineDistances.begin();
	while(itr < lineDistances.end()){
		cout << itr->first << ":  " << itr->second << endl;
		itr++;
	}
}

string find_fname_by_linenum(map<string, uint> &fnames, uint lineNum){
	map<string, uint>::iterator itr = fnames.begin();
	for( ; itr != fnames.end() ; itr++){
		if(itr->second == lineNum){
			return itr->first;
		}
	}
	return NULL;
}









