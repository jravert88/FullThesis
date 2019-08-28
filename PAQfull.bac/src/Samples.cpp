/* 
 * File:   Samples.cpp
 * Author: Andrew McMurdie
 * 
 * Created on April 1, 2013, 9:28 PM
 */

#include "Samples.h"
#include <iostream>

namespace PAQ_SOQPSK {

	Samples::Samples() {
	}

	Samples::~Samples() {

	}

	void Samples::setI(vector<float>& newI) {
		I = newI;
	}

	vector<float>& Samples::getI() {
		return I;
	}

	void Samples::setQ(vector<float>& newQ) {
		Q = newQ;
	}

	vector<float>& Samples::getQ() {
		return Q;
	}

	void Samples::printFirst100() {
		cout << "---------------------------------" << endl;
		cout << "Samples:" << endl;
		for(int i=0; i<100; i++) {
			cout << "I[" << i << "]: " << I.at(i) << "  Q[" << i << "]: " << Q.at(i) << endl;
		}
	}

	/**
	 * Lets the user choose to print the samples from the list.
	 * @param numSamplesToPrint
	 */
	void Samples::displaySamples(int numSamplesToPrint) {
		int displayLength;
		if(numSamplesToPrint >= I.size()) {
			displayLength = I.size();
		} else {
			displayLength = numSamplesToPrint;
		}
		cout << "Samples:" << endl;
		for(int i=0; i<displayLength; i++) {
			cout << "I[" << i << "]: " << I.at(i) << "  Q[" << i << "]: " << Q.at(i) << endl;
		}
	}

	/**
	 * Gets the size of I and Q in the number of samples. Because we specify tha
	 * both are always to be the same size, we can simply grab the number of samples
	 * in I and return this.
	 * @return The number of samples stored in I and Q
	 */
	int Samples::getSize() {
		return I.size();
	}

	/**
	 * Creates a hard copy of the inputSamples.
	 */
	void Samples::copySamples(Samples& inputSamples) {
		for(int i=0; i < inputSamples.getSize(); i++) {
			I.push_back(inputSamples.getI().at(i));
			Q.push_back(inputSamples.getQ().at(i));
		}
	}

}
