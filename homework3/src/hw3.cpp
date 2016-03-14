#include "hw3.hpp"

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

// Will by default use row interleave
bool process_query(map<string, uint> &fnames, vector< pair< uint, vector<float> > > &lines, string queryFilename, int numResults, int numProcesses){
	return process_query(fnames, lines, queryFilename, numResults, numProcesses, true);
}

bool process_query(map<string, uint> &fnames, vector< pair< uint, vector<float> > > &lines, string queryFilename, int numResults, int numProcesses, bool isInterleaved){
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
		cout<< "joined!" << endl;

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
		// for(int i = 0 ; i < numProcesses ; i++){
		// 	delete mosnickThreads[i];
		// }

		return true;
	}
	return false;
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









