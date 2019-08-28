/* 
 * File:   Filter.h
 * Author: Andrew McMurdie
 *
 * Created on April 1, 2013, 10:38 PM
 */

#ifndef FILTER_H
#define	FILTER_H

#include <vector>
#include "Samples.h"
using namespace std;

namespace PAQ_SOQPSK {
	class Filter {
	public:
		Filter();
		virtual ~Filter();
		
		void setFilter(vector<float>& new_h);
		Samples& runFilter(Samples& x);
		Samples& runComplexFilter(Samples& x, Samples& g);
		Samples& runComplexFilterCuda(Samples& x, Samples& g);
		Samples& runFilterNewBounds(Samples& x);
		Samples& runCudaFilter(Samples& x);
	private:
		vector<float> h;
		vector<float>& timeReverseH(vector<float> h);
		bool checkRange(int index);
	};

}
#endif	/* FILTER_H */

