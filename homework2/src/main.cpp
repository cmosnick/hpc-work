#ifndef header
#include "main.hpp"
#endif

// Read csv into map of vectors
int read_in_file(map< string, vector<float> > *files, FILE *infile, map<string, uint> &fnames, vector< pair< uint, vector<float> > > &lines){
	if(files && infile){
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
				(*files)[fname] = floats;
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

		// uint queryLine = fnames[queryFilename];

		// Set up shared memory
		int numLines = fnames.size();
		lineDistance_t *shm = NULL;
		int aggResults = numResults * numProcesses;
		size_t shmsize = sizeof(lineDistance_t)*aggResults;
		key_t shmKey = 123456;
		int shmId;
		int shmFlag = IPC_CREAT | 0666; // Flag to create with rw permissions
		
		if ((shmId = shmget(shmKey, shmsize, shmFlag)) < 0){
			std::cerr << "Init: Failed to initialize shared memory (" << shmId << ")" << std::endl; 
			return 0;
		}

		if (( shm = (lineDistance_t *)shmat(shmId, NULL, 0)) == (lineDistance_t *) -1){
			std::cerr << "Init: Failed to attach shared memory (" << shmId << ")" << std::endl; 
			return 0;
		}


		// Setup steps to divide shared memory
		int i=0;
		lineDistance_t *shmEnd = &(shm[aggResults]);
		childInfo_t childInfo[numProcesses];
		int step = (int)(numLines / numProcesses);
		// int stepSize = step * sizeof(lineDistance_t);
		for(i = 0 ; i < numProcesses ; i++){
			childInfo[i].start = &shm[i*numResults];
			childInfo[i].startLine = step * i;
			childInfo[i].shmStartLine = i * numResults;
			// Last process takes extra 
			if( (i+1) == numProcesses ){
				childInfo[i].end = shmEnd;
				childInfo[i].endLine = numLines;
			}
			else{
				childInfo[i].end = &(shm[(i+1)*numResults]);
				childInfo[i].endLine = (step *(i+1));
			}
			childInfo[i].shmEndline = (i+1)*numResults;
			// childInfo[i].queryLine = queryLine;
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

		// Fork process into worker processes
		pid_t pids[numProcesses];
		for(i = 0 ; i < numProcesses ; i++){
			pid_t pid = fork();
			if(pid >= 0){
				if(pid == 0){	// Child process
					// call helper on partition
					do_work(i+1, childInfo[i], queryFloats, lines, numResults);
					exit(0);
				}
				else{			// Parent process
					pids[i] = pid;
				}
			}
			else{
				cout << "\n\nError: fork failed!" << endl;
				// return false;
				break;
			}
		}

		// Wait for processes to finish
		int status;
		for( i = 0 ; i < numProcesses ; i++){
			/*pid_t pid = */wait(&status);
			// cout << "Child with PID " << (long)pid << " exited with status ." << status << "\n" << endl;
		}
		cout << "\n\nProcesses are finished!" << endl;

		// Gather data in shared memory and get final solution
		vector<pair<uint, float> > lineDistances;
		lineDistance_t *shmStart = shm;
		while(shmStart < shmEnd){
			uint tempLine = shmStart[0].lineNum;
			float tempDist = shmStart[0].distance;
			pair<uint, float> tempPair;
			tempPair.first = tempLine;
			tempPair.second = tempDist;
			lineDistances.push_back(tempPair);
			shmStart++;
		}
		// Sort aggregated results and cut off
		sort(lineDistances.begin(), lineDistances.end(), &comp);
		lineDistances.resize(numResults);
		
		// cout << "\n\nFinal line distances: "<< endl;
		// print_line_distances(lineDistances);
		
		// Match with filenames, print
		vector<pair<uint, float> >::iterator lDItr = lineDistances.begin();
		while(lDItr < lineDistances.end()){
			cout << find_fname_by_linenum(fnames, lDItr->first);
			cout << ":  " << lDItr->second << endl;
			lDItr++;
		}

		// Destroy shared memory
		shmctl(shmId, IPC_RMID, NULL);
		shmdt(shm);
		return true;
	}
	return false;
}

// Computes the L1 norm between two vectors of floats
// Returns 0 on error or if vectors are exactly alike
float compute_L1_norm(const vector<float> *v1, const vector<float> *v2){
	if(v1 && v2){
		int s1 = (*v1).size(),
			s2 = (*v2).size(),
			i=0;
		// Take smallest size
		int size = (s1 < s2) ? s1:s2;
		if(size != NUM_FLOATS){
			cout << "\nSize = " << size << endl;
		}
		float sum = 0;
		for( ; i < size ; i++){
			sum += fabs( ((*v1)[i] - (*v2)[i]) );
		}
		return (float)sum/size;
	}
	return -1;
}

bool do_work(int processNumber, const childInfo_t childInfo, const vector<float> *targetVector, const vector<pair<uint, vector<float> > > &lines, int numResults){
	// process files
	uint startLine = childInfo.startLine;
	uint endLine = childInfo.endLine;
	// uint queryLine = childInfo.queryLine;

	// Create vecator of pair <uint, float> to heapify later
	vector<pair<uint, float> > lineDistances;

	// iterate though lines to get distances
	while(startLine < endLine){
		// if(startLine != queryLine){
			float temp = compute_L1_norm(targetVector, &(lines[startLine].second));
			pair<uint, float> tempPair;
			tempPair.first = startLine;
			tempPair.second = temp;
			lineDistances.push_back(tempPair);
		// }
		// else{
		// 	// cout<<"\n\nrefused to process query line"<< queryLine << endl;
		// }
		startLine++;
	}

	// Sort to get top results (shortest distance)
	sort(lineDistances.begin(), lineDistances.end(), &comp);
	lineDistances.resize(numResults);
	// cout << "\n\nTemp line distances: "<< endl;
	// print_line_distances(lineDistances);

	// Store in shared memory
	lineDistance_t *shm = childInfo.start;
	// lineDistance_t *shmEnd = childInfo.end;
	// uint shmStartLine = childInfo.shmStartLine;
	// uint shmEndline = childInfo.shmEndline;

	for(int i = 0, j = 0 ; i < numResults ; i++, j++){
		shm[i].lineNum = lineDistances[j].first;
		shm[i].distance = lineDistances[j].second;
		// cout << "\nProcess " << processNumber << " printing: " << lineDistances[j].first << ", " << lineDistances[j].second << " at " << &shm[i] << endl;
	}
	// print_shm(shm, shmEnd);
	return true;
}

// Shm will be an array of [linenumber, dist] structs
bool shm_setup(size_t size, lineDistance_t *shm){
	if(size > 0){
		key_t shmKey = 123456;
		int shmId;
		int shmFlag = IPC_CREAT | 0666; // Flag to create with rw permissions
		if ((shmId = shmget(shmKey, size, shmFlag)) < 0){
			std::cerr << "Init: Failed to initialize shared memory (" << shmId << ")" << std::endl; 
			return 0;
		}

		if (( shm = (lineDistance_t *)shmat(shmId, NULL, 0)) == (lineDistance_t *) -1){
			std::cerr << "Init: Failed to attach shared memory (" << shmId << ")" << std::endl; 
			return 0;
		}
		cout << "\n\nShm in setup shm: "<< shm << endl;
		return shmId;
	}
	return 0;
}

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

void print_shm(lineDistance_t *start, const lineDistance_t *end){
	if(end != NULL){
		cout << "Shm info: start: " << start << " end: " << end << endl;
		while(start < end){
			cout << (start)[0].lineNum << ": " << (start)[0].distance <<endl;
			start++;
		}
	}
	return;
}

bool comp(pair<uint, float> &el1, pair<uint, float> &el2){
	return (el1.second < el2.second);
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



/*Notes******************

For processing: instead of map which is not contiguous,
	use indexing data structure of fn=>line number
	then use 2D array of floats

Try interleaving after doing block partitioning if I have time
	Interleaving is better for cache issues

Lookup by value with a set (can be used for wait with pids)

Think about partial sort for sort:
	fixed-size max heap, load sorted, throw out minvalue when bigger value comes in

***********************/









