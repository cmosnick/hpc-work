#ifndef _MOSNICKTHREAD_HPP
#define _MOSNICKTHREAD_HPP
#include <boost/thread/thread.hpp>

typedef struct lineDistance{
	uint lineNum;
	float distance;
}lineDistance_t;

namespace mosnick{
	class MosnickThread{
	public:
		MosnickThread(unsigned int numResults, unsigned int step, unsigned int totalLines, const std::vector<float> *queryFloats);
		~MosnickThread();
		void doWorkBlock(unsigned int startingIndex, unsigned numberToProcess);
		void doWorkInterleave(unsigned int startingIndex, const std::vector<std::pair<uint, std::vector<float> > > &lines);
		static float compute_L1_norm(const std::vector<float> *v1, const std::vector<float> *v2);
		static bool comp(const std::pair<uint, float> &el1, const std::pair<uint, float> &el2);
		
		// Results
		std::vector<std::pair<uint, float>> results;

	private:
		
		// Size to compute
		unsigned int _numResults;

		// Step size for block partitioning
		unsigned int _step;

		// Total numebr of line
		unsigned int _totalLines;

		const std::vector<float> *_queryFloats;

	};
} // End namespace mosnick
#endif