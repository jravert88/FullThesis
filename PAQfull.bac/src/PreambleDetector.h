#ifndef PREAMBLEDETECTOR_H_
#define PREAMBLEDETECTOR_H_

#include <vector>
#include "Samples.h"
using namespace PAQ_SOQPSK;
using namespace std;

namespace PAQ_SOQPSK {

	class PreambleDetector
	{
	public:
		PreambleDetector();
		virtual ~PreambleDetector();

		vector<int>& findPreambleCudaAndrew(Samples& inputSamples, Samples& preambleSamples);
		vector<int>& ncpdi2(float* iSamples, float* qSamples, int numSamples);
		vector<int>& findPreambleNCPDI2(Samples& inputSamples);

		vector<int>& findPreambleWithSavedBlocks(Samples& inputSamples, int start, int num_samples, int loc, vector<string>& output_files);
		vector<int>& findPreambleWithSavedBlocksFileSave(Samples& inputSamples, int start, int num_samples, ofstream& LFile);


	private:

	};

}
#endif
