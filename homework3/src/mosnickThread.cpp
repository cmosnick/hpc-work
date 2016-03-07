#include "MosnickThread.hpp"
#include <iostream>


mosnick::MosnickThread::MosnickThread (unsigned int numResults, unsigned int step, unsigned int totalLines, const std::vector<float> *queryFloats) : 
_numResults(numResults), _step(step), _totalLines(totalLines), _queryFloats(queryFloats){
	std::cout << "Constructing MosnickThread with N = (" << numResults << ")" << std::endl; 
	if(_numResults < 1){
		std::cout << "numResults must be positive non-zero number" << std::endl;
	}
	if(_step < 1){
		std::cout << "step must be 1 or greater" << std::endl;
	}
	std::cout << "Step size: " << _step << std::endl;


	results.resize(numResults);

	// more error checking

}

// Destructor
mosnick::MosnickThread::~MosnickThread(){
	std::cout << "Destructing MosnickThread" << std::endl;
	// Do extra destruction
}

// Block partition method
void mosnick::MosnickThread::doWorkBlock(unsigned int startingIndex, unsigned int numberToProcess){

	// return 0;
}

// Row interleave method
void mosnick::MosnickThread::doWorkInterleave(unsigned int startingIndex, const std::vector<std::pair<uint, std::vector<float> > > &lines){
	if(startingIndex > _numResults){
		return;
	}
	unsigned int currentLine = startingIndex;
	std::vector<std::pair<uint, float>> lineDistances;

	while(currentLine < _totalLines){
		float temp = compute_L1_norm(_queryFloats, &(lines[currentLine].second));
		std::pair<uint, float> tempPair;
		tempPair.first = currentLine;
		tempPair.second = temp;
		lineDistances.push_back(tempPair);
		currentLine += _step;
	}

	// Sort to get top results (shortest distance)
	std::vector<std::pair<uint, float> >::iterator middle = lineDistances.begin() + _numResults;
	partial_sort(lineDistances.begin(), middle, lineDistances.end(), &MosnickThread::comp);
	lineDistances.resize(_numResults);

	results.clear();
	for(int i=0 ; i < _numResults ; i++){
		results.push_back(lineDistances[i]);
	}
	return;
}

// Computes the L1 norm between two vectors of floats
// Returns 0 on error or if vectors are exactly alike
float mosnick::MosnickThread::compute_L1_norm(const std::vector<float> *v1, const std::vector<float> *v2){
	if(v1 && v2){
		int s1 = (*v1).size(),
			s2 = (*v2).size(),
			i=0;
		// Take smallest size
		int size = (s1 < s2) ? s1:s2;
		if(size != 4097){
			std::cout << "\nSize = " << size << std::endl;
		}
		float sum = 0;
		for( ; i < size ; i++){
			sum += fabs( ((*v1)[i] - (*v2)[i]) );
		}
		return (float)sum/size;
	}
	return -1;
}

bool mosnick::MosnickThread::comp(const std::pair<uint, float> &el1, const std::pair<uint, float> &el2){
	return (el1.second < el2.second);
}


