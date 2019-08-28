/*
 * Downsampler.h
 *
 *  Created on: Apr 30, 2013
 *      Author: adm85
 */

#ifndef DOWNSAMPLER_H_
#define DOWNSAMPLER_H_

#include "Samples.h"
using namespace std;
using namespace PAQ_SOQPSK;

namespace PAQ_SOQPSK {

	class Downsampler {
	public:
		Downsampler();
		virtual ~Downsampler();

		Samples& downsample(unsigned int factor, Samples& x);
		Samples& downsample_cuda(unsigned int factor, Samples& x);
	};

}
#endif /* DOWNSAMPLER_H_ */
