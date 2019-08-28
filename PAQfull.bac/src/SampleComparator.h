/*
 * SampleComparator.h
 *
 *  Created on: May 7, 2013
 *      Author: adm85
 */

#ifndef SAMPLECOMPARATOR_H_
#define SAMPLECOMPARATOR_H_

#include <vector>
#include <string>
#include "Samples.h"

namespace PAQ_SOQPSK {

using namespace std;

class SampleComparator {
	public:
		SampleComparator();
		virtual ~SampleComparator();

		void compareSamplesToMatlab(Samples* demodSamples, Samples* matlabSamples, float displayThreshold);
		void compareErrorToMatlab(vector<float>* demodError, string matlabErrorFileName, float displayThreshold);
};

} /* namespace SOQPSK_Demod */
#endif /* SAMPLECOMPARATOR_H_ */
