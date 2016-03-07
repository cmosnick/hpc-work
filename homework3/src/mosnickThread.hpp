#ifndef _MOSNICKTHREAD_HPP
#define _MOSNICKTHREAD_HPP
#include <boost/thread/thread.hpp>

class MosnickThread;

namespace mosnick{
	class MosnickThread{
	public:
		MosnickThread(unsigned int N, unsigned int step);
		~MosnickThread();
		void doWorkBlock(unsigned int startingIndex, unsigned numberToProcess);
		void doWorkInterleave(unsigned int startingIndex);
		boost::thread* do_work( int startingIndex );
	private:
		
		// Size to compute
		unsigned int _N;

		// index we are currently working on
		int _index;

		// Step size for block partitioning
		int _step;

		boost::thread _thread;
	};
} // End namespace mosnick
#endif