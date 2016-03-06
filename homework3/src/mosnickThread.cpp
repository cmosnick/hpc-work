#ifndef _MOSNICKTHREAD_HPP
#define _MOSNICKTHREAD_HPP

namespace mosnick{
	class MosnickThread{
	public:
		MosnickThread(unsigned int N);
		~MosnickThread();
		unsigned long doWorkBlock() const;
		unsigned long doWorkInterleave() const;
		void operator()();
	private:
		
		// Size to compute
		unsigned int _N;
		
		// index we are currently working on
		int _index;
		
		// step size for block partitoning 
		int _step;
	};
} // End namespace mosnick
#endif