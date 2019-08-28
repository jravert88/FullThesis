/* 
 * File:   Filter.cpp
 * Author: Andrew McMurdie
 * 
 * Created on April 1, 2013, 10:38 PM
 */

#include "Filter.h"
#include "Samples.h"
#include "Kernels.h"
#include <iostream>
#include <unistd.h>
#include <stdlib.h>
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
//#include <thrust/copy.h>
//#include <cudart>


using namespace std;

namespace PAQ_SOQPSK {

	Filter::Filter() {
	}

	Filter::~Filter() {
	}

	void Filter::setFilter(vector<float>& new_h) {
		h = new_h;
	}

	/**
	 * Performs a direct convolution with the samples
	 * @param x
	 * @return 
	 */
	Samples& Filter::runFilter(Samples& x) {
		Samples* y = new Samples();
	  
		//Direct convolution
		int N = x.getI().size() + h.size() - 1; //Length is L+M-1
		unsigned int x_size = x.getI().size();
		
		int index;
		float sumR, sumI; //Real and Imaginary parts
		for(int i=0; i < N; i++) {
			sumR = 0;
			sumI = 0;
			//for(int j=0; j<h.size(); j++) {
			for(int j=0; j < N; j++) {
				index = i-j;
				//If either element is out of bounds, then the product is zero
				//and we don't compute it. Otherwise, we get the sum as found below.
				if(checkRange(index) && (j < x_size)) {
					sumR += h.at(index) * x.getI().at(j);
					sumI += h.at(index) * x.getQ().at(j);
				}
			}
			
			//Put these back in the output object
			y->getI().push_back(sumR);
			y->getQ().push_back(sumI);
		}
		
		return *y;
	}

	/**
	 * Runs the convolution with complex values.
	 */
	Samples& Filter::runComplexFilter(Samples& x, Samples& g) {
		Samples* y = new Samples();

		//Direct convolution
		int N = x.getSize() + g.getSize() - 1; //Length is L+M-1
		unsigned int x_size = x.getI().size();

		int index;
		float sumR, sumI; //Real and Imaginary parts
		for(int i=0; i < N; i++) {
			sumR = 0;
			sumI = 0;

			for(int j=(i-g.getSize()+1); j <= i; j++) {
				index = i-j;
				//If either element is out of bounds, then the product is zero
				//and we don't compute it. Otherwise, we get the sum as found below.
				if((index >= 0) && (index < g.getSize()) && (j < x_size) && (j >= 0)) {
					//sumR += (ac - bd)
					//sumI += (ad + bc);
					sumR += (x.getI().at(j) * g.getI().at(index)) - (x.getQ().at(j) * g.getQ().at(index));
					sumI += (x.getI().at(j) * g.getQ().at(index)) + (x.getQ().at(j) * g.getI().at(index));
				}
			}

			//Put these back in the output object
			y->getI().push_back(sumR);
			y->getQ().push_back(sumI);
		}

		return *y;
	}


	/**
	 * Filters the samples, but is much closer to O(N) time, rather than O(N^2)
	 * @param Samples& x. Input samples
	 * @return Filtered samples
	 */
	Samples& Filter::runFilterNewBounds(Samples& x) {
		Samples* y = new Samples();

			//Direct convolution
			int N = x.getI().size() + h.size() - 1; //Length is L+M-1

			int index;
			float sumR, sumI; //Real and Imaginary parts
			for(int i=0; i < N; i++) {
				//cout << "i: " << i << endl;
				sumR = 0;
				sumI = 0;
				
				for(int j=(i-h.size()+1); j <= i; j++) {
					index = i-j;
					//If either element is out of bounds, then the product is zero
					//and we don't compute it. Otherwise, we get the sum as found below.
					if(checkRange(index) && (j < x.getI().size()) && (j >= 0)) {
						sumR += h.at(index) * x.getI().at(j);
						sumI += h.at(index) * x.getQ().at(j);
					}
				}

				//Put these back in the output object
				y->getI().push_back(sumR);
				y->getQ().push_back(sumI);
			}

			return *y;
	}

	vector<float>& Filter::timeReverseH(vector<float> h) {
		vector<float>* invX = new vector<float>;
		
		//Invert
		for(int i=h.size()-1; i>=0; i--) {
			invX->push_back(h.at(i));
		}
		
		//Return inverted filter
		return *invX;
	}

	/**
	 * Checks an access to the filter h[n] sequence.
	 * @param index
	 * @return 
	 */
	bool Filter::checkRange(int index) {
		if((index >= 0) && (index < h.size())) {
			return true;
		} else {
			return false;
		}
	}

	/**
	 * Runs the filter on the GPU
	 * @param Samples& x. The input samples
	 * @return Samples& y. The filtered samples
	 *
	 */
	Samples& Filter::runCudaFilter(Samples& x) {
		Samples* y = new Samples();
		
		int convLength = x.getI().size() + h.size() - 1; //N + M - 1
		cout << "X size: " << x.getI().size() << endl;
		cout << "H size: " << h.size() << endl;
		cout << "convLength: " << convLength << endl;
		
		//Copy the samples and the filter to the device.
		float* d_I;
		float* d_Q;
		float* d_H;
		float* d_filtered_I;
		float* d_filtered_Q;
		int samplesSize = x.getSize() * sizeof(float);
		int filterSize = h.size() * sizeof(float);
		
		cudaMalloc(&d_I, samplesSize * sizeof(float));
		cudaMalloc(&d_Q, samplesSize * sizeof(float));
		cudaMalloc(&d_H, filterSize);
		cudaMalloc(&d_filtered_I, convLength * sizeof(float));
		cudaMalloc(&d_filtered_Q, convLength * sizeof(float));
		
		cout << "    Cuda Malloc completed" << endl;
		
		cudaMemcpy(d_I, x.getI().data(), samplesSize, cudaMemcpyHostToDevice);
		cudaMemcpy(d_Q, x.getQ().data(), samplesSize, cudaMemcpyHostToDevice);
		cudaMemcpy(d_H, h.data(), 		 filterSize,  cudaMemcpyHostToDevice);
		
		cout << "    Cuda Memcpy completed" << endl;
		
		int numThreads = 256;
		int numBlocks = convLength / numThreads;
		if((convLength % numThreads) > 0) numBlocks++; //We have to add another block in case we didn't divide cleanly
		
		//Call the filter
		runFilterCuda<<<numBlocks, numThreads>>>(d_I, d_Q, x.getSize(), d_H, h.size(), d_filtered_I, d_filtered_Q, convLength);
		//cudaDeviceSynchronize();
		
		cout << "    Cuda Filter completed." << endl;

		//cudaDeviceSynchronize();
		//usleep(1000000);
		//Get the results, and store in samples object
		
		float* filtered_I = new float[convLength];
		float* filtered_Q = new float[convLength];
		
		//cout << "Random filtered_I: " << filtered_I[24033] << endl;
		cout << "    Beginning copy back to host memory..." << endl;
		
		//Copy the memory back to host
		cudaMemcpy(filtered_I, d_filtered_I, convLength * sizeof(float), cudaMemcpyDeviceToHost);
		cout << "    Copy to I completed." << endl;
		cudaMemcpy(filtered_Q, d_filtered_Q, convLength * sizeof(float), cudaMemcpyDeviceToHost);
		
		cout << "    Copy from Cuda memory completed." << endl;
		
		//Populate our return object
		y->getI().insert(y->getI().begin(), filtered_I, filtered_I + convLength);
		y->getQ().insert(y->getQ().begin(), filtered_Q, filtered_Q + convLength);
		
		//Free memory
		cudaFree(d_I);
		cudaFree(d_Q);
		cudaFree(d_H);
		delete(filtered_I);
		delete(filtered_Q);
		
		cout << "    Cuda Memory Free complete." << endl;
		
		return *y;
	}

	Samples& Filter::runComplexFilterCuda(Samples& x, Samples& g) {
		Samples* y = new Samples();

		int convLength = x.getI().size() + g.getSize() - 1; //N + M - 1
		//cout << "X size: " << x.getI().size() << endl;
		//cout << "G size: " << h.size() << endl;
		//cout << "convLength: " << convLength << endl;

		//Copy the samples and the filter to the device.
		float* d_I;
		float* d_Q;
		float* d_HR;
		float* d_HI;
		float* d_filtered_I;
		float* d_filtered_Q;
		int samplesSize = x.getSize() * sizeof(float);
		int filterSize = g.getSize() * sizeof(float);

		cudaMalloc(&d_I, samplesSize * sizeof(float));
		cudaMalloc(&d_Q, samplesSize * sizeof(float));
		cudaMalloc(&d_HR, filterSize);
		cudaMalloc(&d_HI, filterSize);
		cudaMalloc(&d_filtered_I, convLength * sizeof(float));
		cudaMalloc(&d_filtered_Q, convLength * sizeof(float));

		//cout << "    Cuda Malloc completed" << endl;

		cudaMemcpy(d_I, x.getI().data(),  samplesSize, cudaMemcpyHostToDevice);
		cudaMemcpy(d_Q, x.getQ().data(),  samplesSize, cudaMemcpyHostToDevice);
		cudaMemcpy(d_HR, g.getI().data(), filterSize,  cudaMemcpyHostToDevice);
		cudaMemcpy(d_HI, g.getQ().data(), filterSize,  cudaMemcpyHostToDevice);

		//cout << "    Cuda Memcpy completed" << endl;

		int numThreads = 256;
		int numBlocks = convLength / numThreads;
		if((convLength % numThreads) > 0) numBlocks++; //We have to add another block in case we didn't divide cleanly

		//Call the filter
		//runFilterCuda<<<numBlocks, numThreads>>>(d_I, d_Q, x.getSize(), d_H, h.size(), d_filtered_I, d_filtered_Q, convLength);
		cudaRunComplexFilter<<<numBlocks, numThreads>>>(d_I, d_Q, x.getSize(), d_HR, d_HI, g.getSize(), d_filtered_I, d_filtered_Q, convLength);
		//cudaDeviceSynchronize();

		//cout << "    Cuda Filter completed." << endl;

		//cudaDeviceSynchronize();
		//usleep(1000000);
		//Get the results, and store in samples object

		float* filtered_I = new float[convLength];
		float* filtered_Q = new float[convLength];

		//cout << "Random filtered_I: " << filtered_I[24033] << endl;
		//cout << "    Beginning copy back to host memory..." << endl;

		//Copy the memory back to host
		cudaMemcpy(filtered_I, d_filtered_I, convLength * sizeof(float), cudaMemcpyDeviceToHost);
		//cout << "    Copy to I completed." << endl;
		cudaMemcpy(filtered_Q, d_filtered_Q, convLength * sizeof(float), cudaMemcpyDeviceToHost);

		//cout << "    Copy from Cuda memory completed." << endl;

		//Populate our return object
		y->getI().insert(y->getI().begin(), filtered_I, filtered_I + convLength);
		y->getQ().insert(y->getQ().begin(), filtered_Q, filtered_Q + convLength);

		//Free memory
		cudaFree(d_I);
		cudaFree(d_Q);
		cudaFree(d_HR);
		cudaFree(d_HI);
		cudaFree(d_filtered_I);
		cudaFree(d_filtered_Q);
		delete [] filtered_I;
		delete [] filtered_Q;

		return *y;
	}

}
