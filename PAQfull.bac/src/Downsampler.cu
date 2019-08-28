/*
 * Downsampler.cpp
 *
 *  Created on: Apr 30, 2013
 *      Author: adm85
 */

#include "Downsampler.h"
#include "Samples.h"
#include "Kernels.h"
namespace PAQ_SOQPSK {

	Downsampler::Downsampler() {}

	Downsampler::~Downsampler() {}

	/**
	 * Downsamples the input samples by the chosen input factor. Returns a blank list if x has no samples.
	 * @param int factor -- downsample factor.
	 *        Samples& x -- The sample set to downsample
	 * @return Downsampled set.
	 */
	Samples& Downsampler::downsample(unsigned int factor, Samples& x) {
		Samples* downsampled = new Samples();

		//If the input sample size is zero, return a blank list.
		if(x.getSize() == 0) return x;

		//Iterate through the list, grabbing sample 0, and then multiples of the factor
		for(int i=0; i < x.getSize(); i+=factor) {
			downsampled->getI().push_back(x.getI().at(i));
			downsampled->getQ().push_back(x.getQ().at(i));
		}

		return *downsampled;
	}

	/**
	 * Downsamples using the GPU
	 */
	Samples& Downsampler::downsample_cuda(unsigned int factor, Samples& x) {
		Samples* downsampled = new Samples();

		//Calculate how many samples we will be left with
		int numDownsampledSamples = x.getSize() / factor;
		if((x.getSize() % factor) != 0) {
			numDownsampledSamples++;
		}

		//Allocate memory
		float* d_I, *d_Q;
		float* d_downsampled_I, *d_downsampled_Q;
		int samplesSize = x.getSize() * sizeof(float);
		int downsampledSize = numDownsampledSamples * sizeof(float);

		cudaMalloc(&d_I, samplesSize);
		cudaMalloc(&d_Q, samplesSize);
		cudaMalloc(&d_downsampled_I, downsampledSize);
		cudaMalloc(&d_downsampled_Q, downsampledSize);

		//Copy to memory
		cudaMemcpy(d_I, x.getI().data(), samplesSize, cudaMemcpyHostToDevice);
		cudaMemcpy(d_Q, x.getQ().data(), samplesSize, cudaMemcpyHostToDevice);

		int numThreads = 256;
		int numBlocks = numDownsampledSamples / numThreads;
		if((numDownsampledSamples % numThreads) > 0) numBlocks++; //We have to add another block in case we didn't divide cleanly

		//Run the program on the GPU
		downsampleCuda<<<numBlocks, numThreads>>>(d_I, d_Q, numDownsampledSamples, d_downsampled_I, d_downsampled_Q, factor);

		//Copy memory out
		float* h_downsampled_I = new float[numDownsampledSamples];
		float* h_downsampled_Q = new float[numDownsampledSamples];

		cudaMemcpy(h_downsampled_I, d_downsampled_I, downsampledSize, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_downsampled_Q, d_downsampled_Q, downsampledSize, cudaMemcpyDeviceToHost);

		//Populate our return object
		downsampled->getI().insert(downsampled->getI().begin(), h_downsampled_I, h_downsampled_I + numDownsampledSamples);
		downsampled->getQ().insert(downsampled->getQ().begin(), h_downsampled_Q, h_downsampled_Q + numDownsampledSamples);

		//Free memory
		cudaFree(d_I);
		cudaFree(d_Q);
		cudaFree(d_downsampled_I);
		cudaFree(d_downsampled_Q);
		delete(h_downsampled_I);
		delete(h_downsampled_Q);

		return *downsampled;
	}
}
