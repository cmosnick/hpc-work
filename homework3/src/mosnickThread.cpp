#include "MosnickThread.hpp"
#include <iostream>


mosnick::MosnickThread::MosnickThread (unsigned int N, unsigned int step) : _N(N), _step(step){
	std::cout << "Constructing MosnickThread with N = (" << N << ")" << std::endl; 
	if(_N < 1){
		std::cout << "N must be positive non-zero number" << std::endl;
	}
	if(_step < 1){
		std::cout << "step must be 1 or greater" << std::endl;
	}
	std::cout << "Step size: " << _step << std::endl;

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
void mosnick::MosnickThread::doWorkInterleave(unsigned int startingIndex){

	// return 0;
}

// callable object to start thread on
boost::thread* mosnick::MosnickThread::do_work( int startingIndex){
	// _thread = boost::thread ();
	_index = startingIndex;
	std::cout << "Starting index: " << startingIndex << std::endl;

	// int threadIndex
	return &_thread;
}


