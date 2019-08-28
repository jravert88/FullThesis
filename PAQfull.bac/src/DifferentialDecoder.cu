/*
 * DifferentialDecoder.cpp
 *
 *  Created on: May 15, 2013
 *      Author: adm85
 */

#include <vector>
#include <iostream>
#include <fstream>
#include "DifferentialDecoder.h"
#include "Kernels.h"
using namespace std;

namespace PAQ_SOQPSK {

	DifferentialDecoder::DifferentialDecoder() {
		//We set the initial delta_minus_one value to a zero.
		initialDelta = 0;
	}

	DifferentialDecoder::~DifferentialDecoder() {}

	/**
	 * Turns a stream of bit indices into the appropriate bits
	 */
	vector<unsigned short>& DifferentialDecoder::convertDecisionsToBits(vector<int> bitDecisionArray){
		vector<unsigned short>* bitstreamArray = new vector<unsigned short>;

		//Push the appropriate binary version of the index onto the return array
		for(int i=0; i < bitDecisionArray.size(); i++) {
			switch(bitDecisionArray.at(i)) {
			case 0:
				bitstreamArray->push_back(0);
				bitstreamArray->push_back(0);
				break;
			case 1:
				bitstreamArray->push_back(0);
				bitstreamArray->push_back(1);
				break;
			case 2:
				bitstreamArray->push_back(1);
				bitstreamArray->push_back(0);
				break;
			case 3:
				bitstreamArray->push_back(1);
				bitstreamArray->push_back(1);
				break;
			default:
				cout << "Error -- invalid bit decision in bitDecisionArray." << endl;
				throw exception();
			}
		}

		return *bitstreamArray;
	}

	vector<unsigned short>& DifferentialDecoder::convertDecisionsToBitsCuda(vector<int> bitDecisionArray)
	{
		// Initialization
		int* bitdec;
		unsigned short* bitstream;
		int size = bitDecisionArray.size();
		int shortsize = size * 2;
		int intsize = size * sizeof(int);
		int shortbyte = shortsize * sizeof(unsigned short);
		unsigned short* out_stream = new unsigned short[shortsize];

		int num_threads = 192;
		int num_blocks = size / num_threads;
		if(size % num_threads)
			num_blocks++;

		// Allocate memory on GPU
		cudaMalloc(&bitstream, shortbyte);
		cudaMalloc(&bitdec, intsize);

		// Copy data to GPU
		cudaMemcpy(bitdec, bitDecisionArray.data(), intsize, cudaMemcpyHostToDevice);

		// Run on GPU
		cudaConvertToBits<<<num_blocks, num_threads>>>(bitdec, bitstream, size);

		// Retrieve data from GPU
		cudaMemcpy(out_stream, bitstream, shortbyte, cudaMemcpyDeviceToHost);

		// Free memory on GPU
		cudaFree(bitstream);
		cudaFree(bitdec);

		vector<unsigned short>* ret_vector = new vector<unsigned short>(out_stream, out_stream + shortsize);

		return *ret_vector;
	}

	/**
	 * Uses the OQPSK decoding algorithm to decode the bitstream
	 */
	vector<unsigned short>& DifferentialDecoder::decodeBitstream(vector<unsigned short> encodedBitstream) {
		//Check that the input array is th e right size. It must be a multiple of two.
		if((encodedBitstream.size() % 2) != 0) {
			cout << "Error -- encodedBitstream has odd size." << endl;
			throw exception();
		}

		//Variables
		vector<unsigned short>* decodedBits = new vector<unsigned short>;
		unsigned short b2k, b2k_plus_1;


		//For the first decision, we use the initialDelta chosen earlier
		b2k = encodedBitstream.at(0) ^ initialDelta;
		b2k_plus_1 = encodedBitstream.at(0) ^ encodedBitstream.at(1);
		decodedBits->push_back(b2k);
		decodedBits->push_back(b2k_plus_1);

		//Now we iterate through the rest of the bitstream, following the correct formula
		for(int i=2; i < encodedBitstream.size(); i+=2) {
			b2k = !encodedBitstream.at(i-1) ^ encodedBitstream.at(i);
			b2k_plus_1 = encodedBitstream.at(i) ^ encodedBitstream.at(i+1);
			decodedBits->push_back(b2k);
			decodedBits->push_back(b2k_plus_1);
		}

		return *decodedBits;
	}

	/**
	 * Runs the decoder on the GPU
	 */
	vector<unsigned short>& DifferentialDecoder::decodeBitstreamCuda(vector<unsigned short> encodedBitstream)
	{
		if(encodedBitstream.size() % 2) {
			cout << "Error -- encodedBitstream has odd size." << endl;
			throw exception();
		}
		unsigned short b2k, b2k_plus_1;

		b2k = encodedBitstream[0] ^ initialDelta;
		b2k_plus_1 = encodedBitstream[0] ^ encodedBitstream[1];

		// Initialization
		int size_bits = encodedBitstream.size();
		int size = size_bits * sizeof(unsigned short);
		unsigned short* encoded_bits, *dec_bits;
		unsigned short* decoded_bits = new unsigned short[size_bits];
		decoded_bits[0] = b2k;
		decoded_bits[1] = b2k_plus_1;

		int num_threads = 192;
		int num_blocks = size / num_threads;
		if(size % num_threads)
			num_blocks++;

		// Allocate memory on GPU
		cudaMalloc(&encoded_bits, size);
		cudaMalloc(&dec_bits, size);

		// Copy data to GPU
		cudaMemcpy(encoded_bits, encodedBitstream.data(), size, cudaMemcpyHostToDevice);
		cudaMemcpy(dec_bits, decoded_bits, 2 * sizeof(unsigned short), cudaMemcpyHostToDevice);

		// Run on GPU
		cudaDecodeBitstream<<<num_blocks, num_threads>>>(encoded_bits, dec_bits, size_bits);

		// Copy data from GPU
		cudaMemcpy(decoded_bits, dec_bits, size, cudaMemcpyDeviceToHost);

		// Free GPU memory
		cudaFree(encoded_bits);
		cudaFree(dec_bits);

		vector<unsigned short>* ret_vector = new vector<unsigned short>(decoded_bits, decoded_bits + size_bits);

		return *ret_vector;
	}

	/**
	 * Wrapper function to decode bits.
	 */
	vector<unsigned short>& DifferentialDecoder::decode(vector<int>& bitDecisionArray) {
		//Convert decisions to bits
		vector<unsigned short> encodedBitstream = convertDecisionsToBits(bitDecisionArray);

		//Decode bits
		return decodeBitstream(encodedBitstream);
	}

	vector<unsigned short>& DifferentialDecoder::decodeCuda(vector<int>& bitDecisionArray)
	{
		vector<unsigned short> encoded = convertDecisionsToBitsCuda(bitDecisionArray);
		// Decode CUDA
		return decodeBitstreamCuda(encoded);
	}

} /* namespace SOQPSK_Demod */
