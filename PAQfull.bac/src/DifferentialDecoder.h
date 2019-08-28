/*
 * DifferentialDecoder.h
 *
 *  Created on: May 15, 2013
 *      Author: adm85
 */

#ifndef DIFFERENTIALDECODER_H_
#define DIFFERENTIALDECODER_H_

#include <vector>
using namespace std;

namespace PAQ_SOQPSK {

class DifferentialDecoder {

	public:
		DifferentialDecoder();
		virtual ~DifferentialDecoder();

		vector<unsigned short>& decode(vector<int>& bitDecisionArray);
		vector<unsigned short>& decodeCuda(vector<int>& bitDecisionArray);
	private:
		vector<unsigned short>& decodeBitstream(vector<unsigned short> encodedBitstream);
		vector<unsigned short>& decodeBitstreamCuda(vector<unsigned short> encodedBitstream);
		vector<unsigned short>& convertDecisionsToBits(vector<int> bitDecisionArray);
		vector<unsigned short>& convertDecisionsToBitsCuda(vector<int> bitDecisionArray);
		unsigned short initialDelta;
	};

} /* namespace SOQPSK_Demod */
#endif /* DIFFERENTIALDECODER_H_ */
