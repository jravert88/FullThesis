/*
 * SampleComparator.cu
 *
 *  Created on: May 7, 2013
 *      Author: adm85
 */

#include <string>
#include <vector>
#include <iostream>
#include "SampleComparator.h"
#include "FileReader.h"
#include "Samples.h"

using namespace std;

namespace PAQ_SOQPSK {

	SampleComparator::SampleComparator() {}

	SampleComparator::~SampleComparator() {}

	/**
	 * Prints out the comparison of our samples to a known set of MATLAB samples
	 */
	void SampleComparator::compareSamplesToMatlab(Samples* demodSamples, Samples* matlabSamples, float displayThreshold = .000001) {
		//Vectors to store the differences on the I and Q lines, and the indices of these differences
		vector<float> iDiffArray, qDiffArray;
		vector<int> failureArray;

		//Variables to store where the maximum difference occurred.
		float maxDifference = 0;
		int maxDifferenceSample = 0;


		//Perform the comparison
		for(int i=0; i < demodSamples->getSize(); i++) {
			float iDiff = demodSamples->getI().at(i) - matlabSamples->getI().at(i);
			float qDiff = demodSamples->getQ().at(i) - matlabSamples->getQ().at(i);


			if(fabs(iDiff) > maxDifference) {
				maxDifference = fabs(iDiff);
				maxDifferenceSample = i;
			}
			if(fabs(qDiff) > maxDifference) {
				maxDifference = fabs(qDiff);
				maxDifferenceSample = i;
			}

			if((iDiff != 0) || (qDiff != 0)) {
				//Save the index
				failureArray.push_back(i);
				//Save the difference
				iDiffArray.push_back(iDiff);
				qDiffArray.push_back(qDiff);
			}

		}

		cout << "Total number of samples different: " << failureArray.size() << endl;
		cout << "Max difference: " << maxDifference << " at sample: " << maxDifferenceSample << endl << endl;
		cout << "Only displaying samples with difference larger than " << displayThreshold << endl;
		for(int i=0; i < failureArray.size(); i++) {
			if((fabs(iDiffArray.at(i)) > displayThreshold) || (fabs(qDiffArray.at(i)) > displayThreshold)) {
				int sampleIndex = failureArray.at(i);
				cout << "Error at sample: " << sampleIndex << endl <<
						"\t  Diff_I: " << iDiffArray.at(i) << "  \tMine: " << demodSamples->getI().at(sampleIndex) << "  \tMatlab: " << matlabSamples->getI().at(sampleIndex) << endl <<
						"\t  Diff_Q: " << qDiffArray.at(i) << "  \tMine: " << demodSamples->getQ().at(sampleIndex) << "  \tMatlab: " << matlabSamples->getQ().at(sampleIndex) << endl;
			}
		}
	}

	/**
	 * Compares the input error measurements against the known good measurements from MATLAB. Assumes that
	 * both arrays have the same size.
	 */
	void SampleComparator::compareErrorToMatlab(vector<float>* demodError, string matlabErrorFileName, float displayThreshold = .000001) {
		cout << "Demod Samples loaded: " << demodError->size() << endl;
		//Load the MATLAB samples
		FileReader fileReader;
		vector<float> matlabErrorSamples = fileReader.loadErrorSamplesFile(matlabErrorFileName.c_str());
		cout << "Matlab samples loaded: " << matlabErrorSamples.size() << endl;
		//Set up the variables we'll need
		vector<float> diffArray;
		float maxDifference = 0;
		vector<int> failureIndexArray;
		int maxDifferenceSample = 0;
		float tempDiff;

		//Compare what we're getting to what MATLAB gets
		for(int i=0; i < demodError->size(); i++) {
			//Compare
			tempDiff = demodError->at(i) - matlabErrorSamples.at(i);

			//Check maximum
			if(fabs(tempDiff) > maxDifference) {
				maxDifference = fabs(tempDiff);
				maxDifferenceSample = i;
			}

			//Add to array if necessary
			if(tempDiff != 0) {
				//Save the index
				failureIndexArray.push_back(i);
				//Save the difference
				diffArray.push_back(tempDiff);
			}
		}

		//Now display the information
		cout << "Total number of samples different: " << failureIndexArray.size() << endl;
		cout << "Max difference: " << maxDifference << " at sample: " << maxDifferenceSample << endl << endl;
		cout << "Only displaying samples with a difference greater than: " << displayThreshold << endl;
		for(int i=0; i < failureIndexArray.size(); i++) {
			if((fabs(diffArray.at(i)) > displayThreshold)) {
				int sampleIndex = failureIndexArray.at(i);
				cout << "Error at sample: " << sampleIndex << endl <<
						"\t  Difference: " << diffArray.at(i) << "  \tMine: " << demodError->at(sampleIndex) <<
						"  \tMatlab: " << matlabErrorSamples.at(sampleIndex) << endl;

			}
		}
	}
} /* namespace SOQPSK_Demod */
