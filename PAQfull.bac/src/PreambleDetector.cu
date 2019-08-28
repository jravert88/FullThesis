/*
 * PreambleDetector.cu
 *
 *  Created on: July 8, 2013
 *      Author: eswindle
 */

#include <fstream>
#include <iostream>
#include <sstream>
#include <time.h>
#include <string.h>
#include "PreambleDetector.h"
#include "Kernels.h"

using namespace std;

namespace PAQ_SOQPSK {

	//const int PREAMBLE_LENGTH = 256;
	//const int FULL_PACKET_LENGTH = (192+6144)*2;

	PreambleDetector::PreambleDetector() {}
	PreambleDetector::~PreambleDetector() {}

	timespec diff(timespec start, timespec end)
	{
		timespec temp;
		if ((end.tv_nsec-start.tv_nsec)<0) {
			temp.tv_sec = end.tv_sec-start.tv_sec-1;
			temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
		} else {
			temp.tv_sec = end.tv_sec-start.tv_sec;
			temp.tv_nsec = end.tv_nsec-start.tv_nsec;
		}
		return temp;
	}




	/**
	 * Runs NCPDI-2 detector on given input samples
	 */
	vector<int>& PreambleDetector::findPreambleWithSavedBlocks(Samples& inputSamples, int start, int num_samples, int loc, vector<string>& output_files) {
		vector<int>* maximumLocations = new vector<int>;

		//Setup variables
		float* r_r = inputSamples.getI().data();
		float* r_i = inputSamples.getQ().data();

		int samplesSize = inputSamples.getSize() * sizeof(float);
		int uLength = inputSamples.getSize() - 256;
		int uSize = uLength * sizeof(float);
		int sumsLength = inputSamples.getSize() - 32;
		int sumsSize = sumsLength * sizeof(float) * 2; //Multiplied by two because we have to store the real and imaginary part
		//int lSize = samplesSize*8;

		float* devRR;
		float* devRI;
		float* devL;
		float* blockSums;

		//Put data down on GPU
		cudaMalloc(&devRR, samplesSize);
		cudaMalloc(&devRI, samplesSize);
		cudaMalloc(&devL,  uSize);
		cudaMalloc(&blockSums, sumsSize);

		cudaMemcpy(devRR, r_r, samplesSize, cudaMemcpyHostToDevice);
		cudaMemcpy(devRI, r_i, samplesSize, cudaMemcpyHostToDevice);

		//Call Kernel
		int threadsPerBlock = 256;
		int numBlocks = sumsLength / threadsPerBlock; //We're calling a thread for each of the blocks
		if(numBlocks % threadsPerBlock) {
			numBlocks++;
		}
		calculateInnerSumBlocksNew<<<numBlocks, threadsPerBlock>>>(devRR, devRI, blockSums, uLength, sumsLength);
		threadsPerBlock = 128;
		numBlocks = uLength / threadsPerBlock;
		if(numBlocks % threadsPerBlock) {
			numBlocks++;
		}
		calculateOuterSums<<<numBlocks, threadsPerBlock>>>(blockSums, devL, uLength);

		//Pull data from GPU
		//float* L = new float[uLength];
		float L[uLength];
		cudaMemcpy(&L, devL, uSize, cudaMemcpyDeviceToHost);


		//Find Maximums
		int idx, maxIdx;
		float maxVal;
		int numPackets = inputSamples.getSize() / 12672;
		for(int i=0; i < numPackets; i++) {
			maxVal = 0;
			maxIdx = 0;
			for(int j=0; j<12672; j++) {
				idx = (12672*i)+j;
				if(L[idx] >= maxVal) {
					maxVal = L[idx];
					maxIdx = idx;
				}
			}
			maximumLocations->push_back(maxIdx);
		}

		//Free GPU memory
		cudaFree(devRR);
		cudaFree(devRI);
		cudaFree(devL);
		cudaFree(blockSums);

		return *maximumLocations;
	}


	/**
	 * A complete NCPDI-1 function detector. Does not run fast enough to use for PAQ real-time project, but
	 * can be used to test other results.
	 */
	vector<int>& PreambleDetector::findPreambleCudaAndrew(Samples& inputSamples, Samples& preambleSamples) {
		vector<int>* maximumLocations = new vector<int>;

		int Lq = 32;
		int N = inputSamples.getSize() - 256;
		int samplesSize = inputSamples.getSize() * sizeof(float);
		int preambleSize = preambleSamples.getSize() * sizeof(float);
		int LSize = N * sizeof(float);

		float *d_xI, *d_xQ;
		float *d_sR, *d_sI;
		float *d_L;

		cudaMalloc(&d_xI, samplesSize);
		cudaMalloc(&d_xQ, samplesSize);
		cudaMalloc(&d_sR, preambleSize);
		cudaMalloc(&d_sI, preambleSize);
		cudaMalloc(&d_L,  LSize);

		cudaMemcpy(d_xI, inputSamples.getI().data(), samplesSize, cudaMemcpyHostToDevice);
		cudaMemcpy(d_xQ, inputSamples.getQ().data(), samplesSize, cudaMemcpyHostToDevice);
		cudaMemcpy(d_sR, preambleSamples.getI().data(), preambleSize, cudaMemcpyHostToDevice);
		cudaMemcpy(d_sI, preambleSamples.getQ().data(), preambleSize, cudaMemcpyHostToDevice);

		int threadsPerBlock = 512;
		int numBlocks = N / threadsPerBlock;
		if((N % threadsPerBlock) > 0) {
			numBlocks++;
		}
		cudaBYUSimplified<<<numBlocks, threadsPerBlock>>>(d_xI, d_xQ, d_sR, d_sI, N, Lq, d_L);

		//Copy back from memory
		float *L = new float[N];
		cudaMemcpy(L, d_L, LSize, cudaMemcpyDeviceToHost);

		//Find Maximums
		int idx, maxIdx;
		float maxVal;
		int numPackets = inputSamples.getSize() / 960;
		for(int i=0; i < numPackets; i++) {
			maxVal = 0;
			maxIdx = 0;
			for(int j=0; j<960; j++) {
				idx = (960*i)+j;
				if(L[idx] >= maxVal) {
					maxVal = L[idx];
					maxIdx = idx;
				}
			}
			maximumLocations->push_back(maxIdx);
		}

		delete [] L;
		cudaFree(d_xI);
		cudaFree(d_xQ);
		cudaFree(d_sR);
		cudaFree(d_sI);
		cudaFree(d_L);

		return *maximumLocations;
	}

	vector<int>& PreambleDetector::findPreambleNCPDI2(Samples& inputSamples) {
		vector<int>* maximumLocations = new vector<int>;

		int Lq = 32;
		int N = inputSamples.getSize() - 256;
		int samplesSize = inputSamples.getSize() * sizeof(float);
		int LSize = N * sizeof(float);

		float *d_xI, *d_xQ;
		float *d_L;

		cudaMalloc(&d_xI, samplesSize);
		cudaMalloc(&d_xQ, samplesSize);
		cudaMalloc(&d_L,  LSize);

		cudaMemcpy(d_xI, inputSamples.getI().data(), samplesSize, cudaMemcpyHostToDevice);
		cudaMemcpy(d_xQ, inputSamples.getQ().data(), samplesSize, cudaMemcpyHostToDevice);

		int threadsPerBlock = 512;
		int numBlocks = N / threadsPerBlock;
		if((N % threadsPerBlock) > 0) {
			numBlocks++;
		}
		cudaNCPDI2<<<numBlocks, threadsPerBlock>>>(d_xI, d_xQ, N, Lq, d_L);

		//Copy back from memory
		float *L = new float[N];
		cudaMemcpy(L, d_L, LSize, cudaMemcpyDeviceToHost);

		//Find Maximums
		int idx, maxIdx;
		float maxVal;
		int numPackets = inputSamples.getSize() / 960;
		for(int i=0; i < numPackets; i++) {
			maxVal = 0;
			maxIdx = 0;
			for(int j=0; j<960; j++) {
				idx = (960*i)+j;
				if(L[idx] >= maxVal) {
					maxVal = L[idx];
					maxIdx = idx;
				}
			}
			maximumLocations->push_back(maxIdx);
		}

		delete [] L;
		cudaFree(d_xI);
		cudaFree(d_xQ);
		cudaFree(d_L);

		return *maximumLocations;
	}
}

