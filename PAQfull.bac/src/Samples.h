/* 
 * File:   Samples.h
 * Author: Andrew McMurdie
 *
 * Created on April 1, 2013, 9:28 PM
 */

#ifndef SAMPLES_H
#define	SAMPLES_H

#include <vector>
using namespace std;


namespace PAQ_SOQPSK {

	class Samples {
	public:
		Samples();
		virtual ~Samples();
		
		void copySamples(Samples& inputSamples);
		void setI(vector<float>& newI);
		void setQ(vector<float>& newQ);
		
		vector<float>& getI();
		vector<float>& getQ();
		
		void printFirst100();
		void displaySamples(int numSamplesToPrint);
		
		int getSize();
	private:
		vector<float> I;
		vector<float> Q;
	};

}
#endif	/* SAMPLES_H */

