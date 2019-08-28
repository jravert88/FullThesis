/*
 * PreambleDetector.cu
 *
 *  Created on: Aug 26, 2014
 *      Author: Andrew McMurdie
 */

#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <fstream>
#include <math.h>
#include <cmath>
#include <string.h>
#include <cuda.h>
#include <cublas.h>
#include <cublas_api.h>
#include <complex>
#include <sstream>
#include <string>
#include <iomanip>

#include "GPUHandler.h"
#include "Environment.h"
#include "Kernels.h"
#include "Samples.h"
#include "FileReader.h"
//#include "mat.h"

using namespace std;
using namespace PAQ_SOQPSK;

namespace PAQ_SOQPSK {
GPUHandler::GPUHandler() {
	initialize_streams();


	//Initialize preamble detector variables
	initialize_pd_variables();

	//Initialize Frequency Offset Estimator variables
	initialize_foe_variables();

	//Initialize Channel Estimator Variables
	initialize_channel_estimator_variables();

	//Noise variance estimator
	initialize_nv_estimator_variables();

	//Equalizers
	initialize_equalizers_variables();

	initialize_CMA_variables();

	initialize_freq_variables();

	initialize_apply_equalizers_and_detection_filters();

	initialize_demodulators_variables();

	//initialize_BERT_variables();

	//Initialize polyphaseFilter variables
	initialize_polyphaseFilters();
}
GPUHandler::GPUHandler(uint numInputSamples) {
	//	cudaDeviceReset();
	wcet = 0;
	bcet = 10000.0;
	Ultrawcet = 0;
	Ultrabcet = 10000.0;

	initialize_streams();

	//Initialize preamble detector variables
	initialize_pd_variables();

	//Initialize Frequency Offset Estimator variables
	initialize_foe_variables();

	//Initialize Channel Estimator Variables
	initialize_channel_estimator_variables();

	//Noise variance estimator
	initialize_nv_estimator_variables();

	//Equalizers
	initialize_equalizers_variables();

	//CMA
	initialize_CMA_variables();

	//FREQ
	initialize_freq_variables();

	initialize_apply_equalizers_and_detection_filters();

	initialize_demodulators_variables();

	//initialize_BERT_variables();

	//Initialize polyphaseFilter variables
	initialize_polyphaseFilters();

	//Check GPU Memory
	checkGPUStats();

	cudaThreadSynchronize();
	cudaError_t code = cudaGetLastError();
	if (code != cudaSuccess)
		printf ("Failed to Initialize!!!!!!!!!!!!!!! -- %s Error number: %d\n", cudaGetErrorString(code), code);
	else
		printf ("Successfully Initialized GPU Variables");

	//	int nDevices;
	//
	//	//	gpuErrchk(cudaGetDeviceCount(&nDevices));
	//	cudaGetDeviceCount(&nDevices);
	//	cout << "Number of Devices: " << nDevices << endl;
	//	for (int i = 0; i < nDevices; i++) {
	//		cudaDeviceProp prop;
	//		cudaGetDeviceProperties(&prop, i);
	//		printf("Device Number: %d\n", i);
	//		printf("  Device name: %s\n", prop.name);
	//		printf("  Processor Speed: %fMHz\n", prop.clockRate/1000);
	//		printf("  Memory Clock Rate (MHz): %d\n",
	//				prop.memoryClockRate/1000);
	//		printf("  Memory Bus Width (bits): %d\n",
	//				prop.memoryBusWidth);
	//		printf("  Peak Memory Bandwidth (GB/s): %f\n",
	//				2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
	//		printf("  Shared Memory Per Multiprocessor: %f\n\n",
	//				prop.sharedMemPerMultiprocessor);
	//	}
}
GPUHandler::~GPUHandler() {

	//	checkGPUStats();

	//Free memory
	free_streams();
	free_polyphaseFilters();
	free_pd_variables();
	free_foe_variables();
	free_channel_estimator_variables();
	free_nv_estimator_variables();
	free_equalizers_variables();
	free_CMA_variables();
	free_freq_variables();
	free_apply_equalizers_and_detection_filters();
	free_demodulators_variables();
	//free_BERT_variables();

	//	cudaDeviceReset();
}

void GPUHandler::initialize_streams(){
	cudaStreamCreate(&stream_GPU0_array[stream_0]);
	cudaStreamCreate(&stream_GPU0_array[stream_1]);
	cudaSetDevice(device_GPU1);
	cudaStreamCreate(&stream_GPU1_array[stream_0]);
	cudaStreamCreate(&stream_GPU1_array[stream_1]);
	cudaSetDevice(device_GPU2);
	cudaStreamCreate(&stream_GPU2_array[stream_0]);
	cudaStreamCreate(&stream_GPU2_array[stream_1]);
	cudaSetDevice(device_GPU0);
}
void GPUHandler::free_streams(){
	cudaStreamDestroy(stream_GPU0_array[stream_0]);
	cudaStreamDestroy(stream_GPU0_array[stream_1]);
	cudaStreamDestroy(stream_GPU1_array[stream_0]);
	cudaStreamDestroy(stream_GPU1_array[stream_1]);
	cudaStreamDestroy(stream_GPU2_array[stream_0]);
	cudaStreamDestroy(stream_GPU2_array[stream_1]);
}

void GPUHandler::initialize_polyphaseFilters(){

	CreateRunDirectorySync();

	DAQoneMb = 1024*1024;

	UltraMbGrag_little = 336;
	UltraMbGrag_big = 343;

	resampleRate = 99.0/224.0;

	// 339Mbs of data at 2 bytes/sample  downsampled by 2, resampled by 99/224
	PolyPush_little = (DAQoneMb*UltraMbGrag_little)/2/2*resampleRate;

	// 343Mbs of data at 2 bytes/sample  downsampled by 2, resampled by 99/224
	PolyPush_big = (DAQoneMb*UltraMbGrag_big)/2/2*resampleRate;

	ultraviewSamplesToHalfLength 	= (UltraMbGrag_big*1024*1024/2+34+1024);
	halfbandSamplesToPolyLength 	= (UltraMbGrag_big*1024*1024/2/2+19);
	FIFOsamplesFromPolyphaseLength 	= PolyPush_big*2;

	PolyPhaseFilterLength = 1980;
	HalfBandFilterLength = 18;

	cudaError_t cudaStat;
	cudaSetDevice(device_GPU1); cudaStat = cudaMalloc(&dev_newFilterBank, 									PolyPhaseFilterLength * sizeof(float));					if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_newFilterBank on GPU" << endl;
	cudaSetDevice(device_GPU1); cudaStat = cudaMalloc(&dev_halfBandFilter,									HalfBandFilterLength * sizeof(float));					if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_halfBandFilter on GPU" << endl;
	cudaSetDevice(device_GPU1); cudaStat = cudaMalloc(&dev_ultraviewSamplesToHalf, 							ultraviewSamplesToHalfLength * sizeof(unsigned short));	if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_ultraviewSamplesToHalf on GPU" << endl;
	cudaSetDevice(device_GPU1); cudaStat = cudaMalloc(&dev_ultraviewSamplesToHalf_last_times, 				ultraviewSamplesToHalfLength * sizeof(unsigned short));	if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_ultraviewSamplesToHalf_last_times on GPU" << endl;
	cudaSetDevice(device_GPU1); cudaStat = cudaMalloc(&dev_ultraviewLastIteration34, 						34 * sizeof(unsigned short));							if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_ultraviewLastIteration34 on GPU" << endl;

	cudaSetDevice(device_GPU1); cudaStat = cudaMalloc(&dev_halfbandSamplesToPoly, 							halfbandSamplesToPolyLength * sizeof(cuComplex));		if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_halfbandSamplesToPoly on GPU" << endl;
	cudaSetDevice(device_GPU1); cudaStat = cudaMalloc(&dev_halfbandSamplesToPolyLastIteration19,			19 * sizeof(cuComplex));								if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_halfbandSamplesToPolyLastIteration19 on GPU" << endl;

	cudaSetDevice(device_GPU1); cudaStat = cudaMalloc(&dev_FIFOsamplesFromPolyphase,						FIFOsamplesFromPolyphaseLength * sizeof(cuComplex));	if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_halfbandSamples on GPU" << endl;

	cudaSetDevice(device_GPU1); cudaStat = cudaMalloc(&dev_PAQcomplexSamplesFromPolyFIFO,					TOTAL_SAMPLES_LENGTH * sizeof(cuComplex));				if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_PAQcomplexSamplesFromPolyFIFO on GPU" << endl;
	cudaSetDevice(device_GPU1); cudaStat = cudaMalloc(&dev_PAQcomplexSamplesFromPolyFIFO_last_times,		TOTAL_SAMPLES_LENGTH * sizeof(cuComplex));				if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_PAQcomplexSamplesFromPolyFIFO on GPU" << endl;
	cudaSetDevice(device_GPU1); cudaStat = cudaMalloc(&dev_PAQcomplexSamplesFromPolyFIFO_two_times,			TOTAL_SAMPLES_LENGTH * sizeof(cuComplex));				if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_PAQcomplexSamplesFromPolyFIFO on GPU" << endl;
	cudaSetDevice(device_GPU1); cudaStat = cudaMalloc(&dev_PAQcomplexSamplesFromPolyFIFOLastIteration12671,	(SAMPLES_PER_PACKET-1) * sizeof(cuComplex));			if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_PAQcomplexSamplesFromPolyFIFOLastIteration12671 on GPU" << endl;
	cudaSetDevice(device_GPU1); cudaStat = cudaMalloc(&dev_iSamples_GPU1, 									TOTAL_SAMPLES_LENGTH * sizeof(float));					if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_iSamples_GPU1 on GPU" << endl;
	cudaSetDevice(device_GPU1); cudaStat = cudaMalloc(&dev_qSamples_GPU1, 									TOTAL_SAMPLES_LENGTH * sizeof(float));					if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_qSamples_GPU1 on GPU" << endl;

	// This is the pointer to where the Ultraview samples are copied to from the host
	// We start at the 35th sample because the Halfband filter needs to look 34 samples in the past
	dev_ultraviewSamplesToHalf_startIdx = &dev_ultraviewSamplesToHalf[34];

	// This is the pointer to where the halfband kernel puts the samples
	// We start at the 20th sample because the Polyphase filter needs to look 19 samples in the past
	dev_halfbandSamplesToPoly_startIdx = &dev_halfbandSamplesToPoly[19];

	// This is the pointer to where the samples from the Poly FIFO are saved
	// We start at the 12671th sample because there is a partial packet on the end of the prior 39,321,600 samples
	dev_PAQcomplexSamplesFromPolyFIFO_startIdx = &dev_PAQcomplexSamplesFromPolyFIFO[SAMPLES_PER_PACKET-1];


	ifstream myFile;
	float halfBandFilter[HalfBandFilterLength];
	myFile.open("/home/adm85/git/JeffPaq/PAQfull/src/HalfBandFilter.txt");	if (!myFile.is_open()) printf("\n\n\t\tCould not open the file!\n\n");	else	{		int i = 0;		float output;		while (!myFile.eof()) {			myFile >> output;
	halfBandFilter[i++] = (float)output;																													}		myFile.close();	}
	cudaMemcpy(dev_halfBandFilter, halfBandFilter, 18 * sizeof(float), cudaMemcpyHostToDevice);

	float newFilterBank[PolyPhaseFilterLength];			//host new row pointer for csr format
	myFile.open("/home/adm85/git/JeffPaq/PAQfull/src/NewFilterBank.txt");	if (!myFile.is_open()) printf("\n\n\t\tCould not open the file!\n\n");	else	{		int i = 0;		float output;		while (!myFile.eof()) {			myFile >> output;
	newFilterBank[i++] = (float)output;																													}		myFile.close();	}
	cudaMemcpy(dev_newFilterBank, newFilterBank, PolyPhaseFilterLength * sizeof(float), cudaMemcpyHostToDevice);

	cout << "\nPolyphase Filter stuff was malloced\n";
}
void GPUHandler::free_polyphaseFilters(){
	cudaFree(dev_x_short);
	cudaFree(dev_y_p_old);
	cudaFree(dev_z_p);
	cudaFree(dev_halfBandFilter);
	cudaFree(dev_newFilterBank);
	cudaFree(dev_iSamples_GPU1);
	cudaFree(dev_qSamples_GPU1);
	cudaFree(dev_old_iSamples_GPU1);
	cudaFree(dev_old_qSamples_GPU1);


	cudaFree(dev_newFilterBank);
	cudaFree(dev_halfBandFilter);
	cudaFree(dev_ultraviewSamplesToHalf);
	cudaFree(dev_ultraviewSamplesToHalf_last_times);
	cudaFree(dev_ultraviewLastIteration34);
	cudaFree(dev_halfbandSamplesToPoly);
	cudaFree(dev_halfbandSamplesToPolyLastIteration19);
	cudaFree(dev_FIFOsamplesFromPolyphase);
	cudaFree(dev_PAQcomplexSamplesFromPolyFIFO);
	cudaFree(dev_PAQcomplexSamplesFromPolyFIFO_last_times);
	cudaFree(dev_PAQcomplexSamplesFromPolyFIFO_two_times);
	cudaFree(dev_PAQcomplexSamplesFromPolyFIFOLastIteration12671);
	cudaFree(dev_iSamples_GPU1);
	cudaFree(dev_qSamples_GPU1);
}
void GPUHandler::DAQCopytoDevice(int numMb,unsigned short* ultraviewSamples_short){
	cudaError_t cudaStat;

	cudaStat = cudaGetLastError(); if(cudaStat != cudaSuccess)	cout << "ERROR Before host to GPU copy " << cudaStat << endl;

	// Copy from Host to GPU
	cudaSetDevice(device_GPU1); cudaStat = cudaMemcpy(dev_ultraviewSamplesToHalf_startIdx,   ultraviewSamples_short, numMb*1024*1024/2*sizeof(unsigned short), cudaMemcpyHostToDevice);  if(cudaStat != cudaSuccess)	cout << "ERROR - Could not copy DAQ samples to GPU " << cudaStat << endl;
}
void GPUHandler::RunHalfbandFilterWithDAQCopy(int numMb,unsigned short* ultraviewSamples_short){
	cudaError_t cudaStat;

	cudaStat = cudaGetLastError(); if(cudaStat != cudaSuccess)	cout << "ERROR Before host to GPU copy " << cudaStat << endl;

	//	// Copy from Host to GPU
	//	cudaSetDevice(device_GPU1); cudaStat = cudaMemcpy(dev_ultraviewSamplesToHalf_startIdx,   ultraviewSamples_short, numMb*1024*1024/2*sizeof(unsigned short), cudaMemcpyHostToDevice);  if(cudaStat != cudaSuccess)	cout << "ERROR - Could not copy DAQ samples to GPU " << cudaStat << endl;

	// Prepend last 34 samples from last iteration
	cudaSetDevice(device_GPU1); cudaStat = cudaMemcpy(dev_ultraviewSamplesToHalf,   dev_ultraviewLastIteration34, 34*sizeof(unsigned short), cudaMemcpyDeviceToDevice); if(cudaStat != cudaSuccess)	cout << "ERROR - Could not prepend 34 DAQ samples " << cudaStat << endl;

	// Save last 34 samples from current iteration
	unsigned short* temp = &dev_ultraviewSamplesToHalf_startIdx[numMb*1024*1024/2-34];
	cudaSetDevice(device_GPU1); cudaStat = cudaMemcpy(dev_ultraviewLastIteration34,   temp, 34*sizeof(unsigned short), cudaMemcpyDeviceToDevice); if(cudaStat != cudaSuccess)	cout << "ERROR - Could not save last 34 of DAQ samples " << cudaStat << endl;

	int maxThreads = numMb*1024*1024/2/2;
	int numThreads = 32; //~145ms
	int numBlocks = maxThreads/numThreads;
	if((maxThreads % numThreads) > 0) numBlocks++;
	cudaSetDevice(device_GPU1); runHalfBandFilter<<<numBlocks, numThreads>>>(dev_halfbandSamplesToPoly_startIdx,dev_ultraviewSamplesToHalf,dev_halfBandFilter,maxThreads);
	cudaStat = cudaGetLastError(); if(cudaStat != cudaSuccess)	cout << "ERROR - Could not run runHalfBandFilter Error " << cudaStat << endl;
}
void GPUHandler::RunPolyphaseFilters(int numMb,int PolyWriteIdx){
	cudaError_t cudaStat;

	// Prepend last 19 samples from last iteration
	cudaSetDevice(device_GPU1); cudaStat = cudaMemcpy(dev_halfbandSamplesToPoly,   dev_halfbandSamplesToPolyLastIteration19, 19*sizeof(cuComplex), cudaMemcpyDeviceToDevice); if(cudaStat != cudaSuccess)	cout << "ERROR - Could not prepend 19 of half samples " << cudaStat << endl;

	// Save last 19 samples from current iteration
	cuComplex* temp = &dev_halfbandSamplesToPoly_startIdx[numMb*1024*1024/2/2-19];
	cudaSetDevice(device_GPU1); cudaStat = cudaMemcpy(dev_halfbandSamplesToPolyLastIteration19,   temp, 19*sizeof(cuComplex), cudaMemcpyDeviceToDevice); if(cudaStat != cudaSuccess)	cout << "ERROR - Could not save last 19 of half samples " << cudaStat << endl;

	cuComplex* dev_PolyPushFIFOlocation = &dev_FIFOsamplesFromPolyphase[PolyWriteIdx];
	int maxThreads = numMb*1024*1024/2/2*99.0/224.0;
	int numThreads = 32; //~145ms
	int numBlocks = maxThreads/numThreads;
	if((maxThreads % numThreads) > 0) numBlocks++;
	cudaSetDevice(device_GPU1); runPolyPhaseFilter<<<numBlocks, numThreads>>>(dev_PolyPushFIFOlocation,dev_halfbandSamplesToPoly_startIdx,dev_newFilterBank,maxThreads);
	cudaStat = cudaGetLastError(); if(cudaStat != cudaSuccess)	cout << "ERROR - Could not run runPolyPhaseFilter Error " << cudaStat << endl;
}
void GPUHandler::PullFromPolyFIFOandConvertFromComplexToRealandImag(bool conj){
	cudaError_t cudaStat;

	//	cudaSetDevice(device_GPU1); cudaStat = cudaMemcpy(dev_PAQcomplexSamplesFromPolyFIFO_two_times,   dev_PAQcomplexSamplesFromPolyFIFO_last_times, TOTAL_SAMPLES_LENGTH*sizeof(cuComplex), cudaMemcpyDeviceToDevice); if(cudaStat != cudaSuccess)	cout << "ERROR - Could not save PAQ samples two times samples " << cudaStat << endl;
	//	cudaSetDevice(device_GPU1); cudaStat = cudaMemcpy(dev_PAQcomplexSamplesFromPolyFIFO_last_times,   dev_PAQcomplexSamplesFromPolyFIFO, TOTAL_SAMPLES_LENGTH*sizeof(cuComplex), cudaMemcpyDeviceToDevice); if(cudaStat != cudaSuccess)	cout << "ERROR - Could not save PAQ samples last times samples " << cudaStat << endl;



	// Prepend last 12671 samples from last iteration
	cudaSetDevice(device_GPU1); cudaStat = cudaMemcpy(dev_PAQcomplexSamplesFromPolyFIFO,   dev_PAQcomplexSamplesFromPolyFIFOLastIteration12671, (SAMPLES_PER_PACKET-1)*sizeof(cuComplex), cudaMemcpyDeviceToDevice); if(cudaStat != cudaSuccess)	cout << "ERROR - Could not prepend 12671 samples " << cudaStat << endl;

	// Pull 39,321,600 complex samples from the Poly FIFO
	cudaSetDevice(device_GPU1); cudaStat = cudaMemcpy(dev_PAQcomplexSamplesFromPolyFIFO_startIdx,   dev_FIFOsamplesFromPolyphase, NUM_INPUT_SAMPLES_DEFAULT*sizeof(cuComplex), cudaMemcpyDeviceToDevice); if(cudaStat != cudaSuccess)	cout << "ERROR - Could not pull 39,321,600samples " << cudaStat << endl;

	// Save last 12671 samples from current iteration
	cuComplex* temp = &dev_PAQcomplexSamplesFromPolyFIFO[TOTAL_SAMPLES_LENGTH-(SAMPLES_PER_PACKET-1)];
	cudaSetDevice(device_GPU1); cudaStat = cudaMemcpy(dev_PAQcomplexSamplesFromPolyFIFOLastIteration12671,   temp, (SAMPLES_PER_PACKET-1)*sizeof(cuComplex), cudaMemcpyDeviceToDevice); if(cudaStat != cudaSuccess)	cout << "ERROR - Could not pull save last 12671 samples " << cudaStat << endl;

	int maxThreads = TOTAL_SAMPLES_LENGTH;
	int numThreads = 32;
	int numBlocks = maxThreads/numThreads;
	if((maxThreads % numThreads) > 0) numBlocks++;
	cudaSetDevice(device_GPU1); ComplexSamplesToiORq<<<numBlocks, numThreads>>>(dev_iSamples_GPU1,dev_qSamples_GPU1,dev_PAQcomplexSamplesFromPolyFIFO,conj,maxThreads);
	cudaStat = cudaGetLastError(); if(cudaStat != cudaSuccess)	cout << "ERROR - Could not run ComplexSamplesToiORq Error " << cudaStat << endl;
}
void GPUHandler::ShiftPolyFIFO(int numSamplesInFIFObeforeShift){
	cudaError_t cudaStat;

	// The function is given how many samples are in the FIFO before shifting
	// We need to subtract off 39,321,600 to figure out how many samples are left over
	//	int numSamplesToShift = numSamplesInFIFObeforeShift-NUM_INPUT_SAMPLES_DEFAULT;
	// We are going to just shift the whole FIFO blindly...it is safer

	// Get the pointer that points to the 39,321,601st sample
	cuComplex* shiftFromPointer = &dev_FIFOsamplesFromPolyphase[NUM_INPUT_SAMPLES_DEFAULT];
	cudaSetDevice(device_GPU1); cudaStat = cudaMemcpy(dev_FIFOsamplesFromPolyphase,   shiftFromPointer, (FIFOsamplesFromPolyphaseLength-NUM_INPUT_SAMPLES_DEFAULT)*sizeof(cuComplex), cudaMemcpyDeviceToDevice); if(cudaStat != cudaSuccess)	cout << "ERROR - Could not shift the Poly FIFO on GPU " << cudaStat << endl;
}


void GPUHandler::initialize_pd_variables() {
	cudaError_t cudaStat;

	Nfft_pd = pow(2, ceil(log(39321600+12672)/log(2)));

	L_pd 			= new float[TOTAL_SAMPLES_LENGTH];
	old_iSamples_pd = new float[NUM_OLD_SAMPLES];
	old_qSamples_pd = new float[NUM_OLD_SAMPLES];
	max_locations 	= new int[MAX_PACKETS_PER_MEMORY_SECTION+1];
	for(int i=0; i < NUM_OLD_SAMPLES; i++)
		old_iSamples_pd[i] = 0;
	for(int i=0; i < NUM_OLD_SAMPLES; i++)
		old_qSamples_pd[i] = 0;

	cudaStat = cudaMalloc(&dev_iSamples_pd, 			(TOTAL_SAMPLES_LENGTH+12672) * sizeof(float));					if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_iSamples_pd on GPU" << endl;
	cudaStat = cudaMalloc(&dev_qSamples_pd, 			(TOTAL_SAMPLES_LENGTH+12672) * sizeof(float));					if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_qSamples_pd on GPU" << endl;
	cudaStat = cudaMalloc(&dev_old_iSamples_pd, 		NUM_OLD_SAMPLES * sizeof(float));						if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_old_iSamples_pd_GPU on GPU" << endl;
	cudaStat = cudaMalloc(&dev_old_qSamples_pd, 		NUM_OLD_SAMPLES * sizeof(float));						if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_old_qSamples_pd_GPU on GPU" << endl;
	cudaStat = cudaMalloc(&dev_blockSums_pd, 			((TOTAL_SAMPLES_LENGTH - 32) * 2) * sizeof(float));		if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_blockSums_pd on GPU" << endl;
	cudaStat = cudaMalloc(&dev_L_pd, 					TOTAL_SAMPLES_LENGTH * sizeof(float));					if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_L_pd on GPU" << endl;
	//	cudaStat = cudaMalloc(&dev_L_pd_last_times,			TOTAL_SAMPLES_LENGTH * sizeof(float));					if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_L_pd_last_times on GPU" << endl;
	cudaStat = cudaMalloc(&dev_max_locations, 			(MAX_PACKETS_PER_MEMORY_SECTION+3) * sizeof(int));		if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_max_locations on GPU" << endl;
	//	cudaStat = cudaMalloc(&dev_max_locations_last_times,(MAX_PACKETS_PER_MEMORY_SECTION+1) * sizeof(int));		if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_max_locations_last_times on GPU" << endl;
	cudaStat = cudaMalloc(&dev_max_locations_save, 			(MAX_PACKETS_PER_MEMORY_SECTION+1) * sizeof(int));	if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_max_locations_save on GPU" << endl;
	cudaStat = cudaMalloc(&dev_num_good_maximums, 		sizeof(int));											if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_num_good_maximums on GPU" << endl;
	cudaStat = cudaMalloc(&dev_endIndex_FirstWindow, 	sizeof(int));											if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_endIndex_FirstWindow on GPU" << endl;
	cudaStat = cudaMalloc(&dev_myFirstMaxMod, 			sizeof(int));											if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_myFirstMaxMod on GPU" << endl;
	cudaStat = cudaMalloc(&dev_myFirstMaxActual, 		sizeof(int));											if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_myFirstMaxActual on GPU" << endl;
	cudaSetDevice(device_GPU2);	cudaStat = cudaMalloc(&dev_Samples_GPU2, 			(TOTAL_SAMPLES_LENGTH+12672) * sizeof(cuComplex));				if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_qSamples_pd on GPU" << endl;
	cudaSetDevice(device_GPU1);	cudaStat = cudaMalloc(&dev_Samples_GPU1, 			TOTAL_SAMPLES_LENGTH * sizeof(cuComplex));				if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_qSamples_pd on GPU" << endl;
	cudaSetDevice(device_GPU0);	cudaStat = cudaMalloc(&dev_Samples_GPU0, 			Nfft_pd * sizeof(cuComplex));				if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_Samples_GPU0 on GPU" << endl;
	cudaSetDevice(device_GPU0);	cudaStat = cudaMalloc(&dev_Samples_GPU0_pd,			Nfft_pd * sizeof(cuComplex));				if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_Samples_GPU0_pd on GPU" << endl;
	cudaSetDevice(device_GPU0);	cudaStat = cudaMalloc(&dev_Matched_GPU0, 			Nfft_pd * sizeof(cuComplex));				if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_Matched_GPU0 on GPU" << endl;

	//Initialize Window Start Offset
	int endIndex_FirstWindow = 12672;
	cudaMemcpy(dev_endIndex_FirstWindow, &endIndex_FirstWindow, sizeof(int), cudaMemcpyHostToDevice);

	maxThreads_calculateInnerSumBlocksNew = (TOTAL_SAMPLES_LENGTH - 32);
	numThreads_calculateInnerSumBlocksNew = 128; // Tested Again
	numBlocks_calculateInnerSumBlocksNew = maxThreads_calculateInnerSumBlocksNew/numThreads_calculateInnerSumBlocksNew;
	if((maxThreads_calculateInnerSumBlocksNew % numThreads_calculateInnerSumBlocksNew) > 0) numBlocks_calculateInnerSumBlocksNew++;

	maxThreads_calculateOuterSumsNew = TOTAL_SAMPLES_LENGTH;
	numThreads_calculateOuterSumsNew = 384; // Tested Again
	numBlocks_calculateOuterSumsNew = maxThreads_calculateOuterSumsNew/numThreads_calculateOuterSumsNew;
	if((maxThreads_calculateOuterSumsNew % numThreads_calculateOuterSumsNew) > 0) numBlocks_calculateOuterSumsNew++;

	maxThreads_findPreambleMaximums = MAX_PACKETS_PER_MEMORY_SECTION;
	numThreads_findPreambleMaximums = 13; // Tested Again
	numBlocks_findPreambleMaximums = maxThreads_findPreambleMaximums/numThreads_findPreambleMaximums;
	if((maxThreads_findPreambleMaximums % numThreads_findPreambleMaximums) > 0) numBlocks_findPreambleMaximums++;

	maxThreads_cudaMaxAdjust = MAX_PACKETS_PER_MEMORY_SECTION;
	numThreads_cudaMaxAdjust = 512; // Tested Again
	numBlocks_cudaMaxAdjust = maxThreads_cudaMaxAdjust/numThreads_cudaMaxAdjust;
	if((maxThreads_cudaMaxAdjust % numThreads_cudaMaxAdjust) > 0) numBlocks_cudaMaxAdjust++;

	maxThreads_stripSignalFloatToComplex = TOTAL_SAMPLES_LENGTH;
	numThreads_stripSignalFloatToComplex = 128; // Tested Again
	numBlocks_stripSignalFloatToComplex = maxThreads_stripSignalFloatToComplex/numThreads_stripSignalFloatToComplex;
	if((maxThreads_stripSignalFloatToComplex % numThreads_stripSignalFloatToComplex) > 0) numBlocks_stripSignalFloatToComplex++;

	myMaxi = 0;
	myMaxq = 0;

	cudaSetDevice(device_GPU0);	cufftPlan1d(&plan_pd, Nfft_pd, CUFFT_C2C, 1);


	cudaMemset(dev_Matched_GPU0, 0, Nfft_pd*sizeof(cufftComplex));


	cuComplex p[384];
	float p_i[384] = {-0.838982631688505,-0.683830361742895,-0.636616680604897,-0.634446649338958,-0.639868338730554,-0.705693526767968,-0.879181483486105,-0.998831639976740,-0.696130008980401,0.0880811688082395,0.812727044689038,0.991154291045280,0.793298448051547,0.684193733947563,0.847265745876414,0.998443764720107,0.657332799118448,-0.0349944525437909,-0.534413923260552,-0.728639443786322,-0.760535771427214,-0.737376336663779,-0.703573693897091,-0.668757060429745,-0.648717615714069,-0.706729708318285,-0.886811942538315,-0.994839197791722,-0.636952740415358,0.0875849626538512,0.587371620681858,0.691505980860904,0.461760394470964,-0.0563000924037576,-0.521678381939263,-0.659228261312206,-0.465913001403911,0.00449869344116628,0.466495906243130,0.697966190424475,0.763954041983948,0.772692322985251,0.768324195341408,0.708517216641752,0.476487060784614,-0.0483255106685605,-0.717915740597009,-0.996113300634708,-0.582644617954213,0.132714623695101,0.609068381281102,0.730005492117424,0.544414694518884,-0.00353330049461398,-0.710198234341746,-0.996151283404189,-0.582848247603495,0.132714623695191,0.609068381281174,0.730005492117489,0.544414694518957,-0.00353330049452333,-0.710198234341682,-0.996151283404197,-0.582848247603569,0.132714623695101,0.608832959290986,0.729300304694098,0.530917745127470,-0.0567974266418796,-0.764025997276590,-0.996104413624349,-0.799783670070293,-0.683756782582573,-0.838952603331356,-0.999993757874326,-0.704001752794662,0.0876505594515629,0.812581023815745,0.991154291045279,0.793298448051542,0.684193733947560,0.847423358132623,0.998385723218666,0.645412190615699,-0.0871543344301889,-0.587168808680034,-0.691505980860906,-0.461760394470967,0.0563000924037468,0.521425148685835,0.658452469201042,0.451954319548684,-0.0567316827730204,-0.521638946023374,-0.658452469201129,-0.451689544293653,0.0577612423696372,0.535229014047310,0.697609376697474,0.521329986474089,-0.0572290047877856,-0.763996173625419,-0.996194821302554,-0.809290202502172,-0.721657234664347,-0.879499507925112,-0.999995393855305,-0.892124804524308,-0.752973239720736,-0.715565207269552,-0.676193701136731,-0.466766331335624,0.0487572823674530,0.717883562151343,0.996203604723379,0.595579544975826,-0.0797279229008580,-0.543698881187507,-0.728935444063191,-0.760891063208189,-0.738072535968645,-0.714858703989376,-0.707412384139441,-0.707283941278989,-0.707106781186505,-0.707316617397655,-0.707835609505449,-0.718335064535858,-0.743770019512190,-0.761384182856532,-0.708212092345386,-0.476527699263787,0.0472954392117449,0.706470559587205,0.999426482977911,0.657905567318613,-0.0781684356021010,-0.760144398986021,-0.996906282648766,-0.649338000447898,0.0796274053041744,0.770796660814217,0.991167668708968,0.573650007722556,-0.133143068735165,-0.609267096520230,-0.730005492117501,-0.544663628894475,0.00250205442054749,0.698625737647798,0.999441027953760,0.658094253751041,-0.0781684356019823,-0.760144398985944,-0.996906282648775,-0.649338000447988,0.0796274053040557,0.770796660814141,0.991167668708985,0.573650007722659,-0.133143068735040,-0.609267096520125,-0.730005492117410,-0.544663628894364,0.00250205442067367,0.698625737647885,0.999441027953756,0.658094253750952,-0.0781684356021010,-0.760144398986021,-0.996906282648766,-0.649338000447898,0.0796274053041744,0.770796660814217,0.991167668708968,0.573650007722556,-0.133143068735165,-0.609267096520230,-0.730005492117501,-0.544663628894475,0.00250205442054749,0.698625737647798,0.999441027953760,0.658094253751041,-0.0781684356019823,-0.760144398985944,-0.996906282648775,-0.649338000447988,0.0796274053040557,0.770796660814141,0.991167668708985,0.573650007722659,-0.133143068735040,-0.609267096520125,-0.730005492117410,-0.544663628894364,0.00250205442067367,0.698625737647885,0.999441027953756,0.658094253750952,-0.0781684356021010,-0.760144398986021,-0.996906282648766,-0.649338000447898,0.0796274053041744,0.770796660814217,0.991167668708968,0.573650007722556,-0.133143068735165,-0.609267096520230,-0.730005492117501,-0.544663628894475,0.00250205442054749,0.698625737647798,0.999441027953760,0.658094253751041,-0.0781684356019823,-0.760144398985944,-0.996906282648775,-0.649338000447988,0.0796274053040557,0.770796660814141,0.991167668708985,0.573650007722659,-0.133143068735040,-0.609267096520125,-0.730005492117410,-0.544663628894364,0.00250205442067367,0.698625737647885,0.999441027953756,0.658094253750952,-0.0781684356021010,-0.760144398986021,-0.996906282648766,-0.649338000447898,0.0796274053041744,0.770796660814217,0.991167668708968,0.573650007722556,-0.133143068735165,-0.609267096520230,-0.730005492117501,-0.544663628894475,0.00250205442054749,0.698625737647798,0.999441027953760,0.658094253751041,-0.0781684356019823,-0.760144398985944,-0.996906282648775,-0.649338000447988,0.0796274053040557,0.770796660814141,0.991167668708985,0.573650007722659,-0.133143068735040,-0.609267096520125,-0.730005492117410,-0.544663628894364,0.00250205442067367,0.698625737647885,0.999441027953756,0.658094253750952,-0.0781684356021010,-0.760144398986021,-0.996906282648766,-0.649338000447898,0.0796274053041744,0.770796660814217,0.991167668708968,0.573650007722556,-0.133143068735165,-0.609267096520230,-0.730005492117501,-0.544663628894475,0.00250205442054749,0.698625737647798,0.999441027953760,0.658094253751041,-0.0781684356019823,-0.760144398985944,-0.996906282648775,-0.649338000447988,0.0796274053040557,0.770796660814141,0.991167668708985,0.573650007722659,-0.133143068735040,-0.609267096520125,-0.730005492117410,-0.544663628894364,0.00250205442067367,0.698625737647885,0.999441027953756,0.658094253750952,-0.0781684356021010,-0.760144398986021,-0.996906282648766,-0.649338000447898,0.0796274053041744,0.770796660814217,0.991167668708968,0.573650007722556,-0.133143068735165,-0.609267096520230,-0.730005492117501,-0.544663628894475,0.00250205442054749,0.698625737647798,0.999441027953760,0.658094253751041,-0.0781684356019823,-0.760144398985944,-0.996906282648775,-0.649338000447988,0.0796274053040557,0.770796660814141,0.991167668708985,0.573650007722659,-0.133143068735040,-0.609267096520125,-0.730005492117410,-0.544663628894364,0.00250205442067367,0.698625737647885,0.999441027953756,0.658094253750952,-0.0781684356021010,-0.760144398986021,-0.996906282648766,-0.649338000447898,0.0796274053041744,0.770796660814217,0.991167668708968,0.573650007722556,-0.133143068735165,-0.609267096520230,-0.730005492117501,-0.544663628894475,0.00250205442054749,0.698625737647798,0.999441027953760,0.658094253751041,-0.0781684356019823,-0.760144398985944,-0.996906282648775,-0.649338000447988,0.0796274053040557,0.770796660814141,0.991167668708985,0.573650007722659,-0.133143068735040,-0.609267096520125,-0.730005492117410,-0.544663628894364,0.00250205442067367,0.698625737647885,0.999441027953756,0.658094253750952,-0.0781684356021010,-0.760144398986021,-0.996906282648766,-0.649338000447898,0.0796274053041744,0.770796660814217,0.991167668708968,0.573650007722556,-0.133143068735165,-0.609267096520225,-0.730005492117496,-0.544663628894469,0.00250205442054749,0.698625737647803,0.999441027953760,0.658094253751041,-0.0781684356019894,-0.760337210765814,-0.996824696887343,-0.637309983311412,0.131592464010629,0.809812979764285,0.996782104311381};
	float p_q[384] = {0.544158197333303,0.729641032534891,0.771180395222546,0.772966654612325,0.768484553579447,0.708517216641761,0.476487060784625,-0.0483255106685480,-0.717915740597005,-0.996113300634709,-0.582644617954223,0.132714623695089,0.608832959290976,0.729300304694090,0.531169234674303,-0.0557678105320651,-0.753600418791753,-0.999387506571480,-0.845222904697492,-0.684897482079442,-0.649296034470882,-0.675482152338835,-0.710622302813529,-0.743480997823997,-0.761029207758976,-0.707483653083484,-0.462130477864662,0.101464134240253,0.770902851517210,0.996157053037785,0.809317353834432,0.722370734757160,0.887004700155540,0.998413891928257,0.853142230707889,0.751942883128290,0.884830534691700,0.999989880827462,0.884523357214721,0.716130712247664,0.645270657737037,0.634780729070800,0.640060880583228,0.705693526767977,0.879181483486111,0.998831639976739,0.696130008980398,-0.0880811688082519,-0.812727044689045,-0.991154291045278,-0.793117713157145,-0.683441278734609,-0.838816213715442,-0.999993757874325,-0.704001752794599,0.0876505594516515,0.812581023815801,0.991154291045266,0.793117713157089,0.683441278734540,0.838816213715395,0.999993757874326,0.704001752794663,-0.0876505594515612,-0.812581023815748,-0.991154291045278,-0.793298448051539,-0.684193733947554,-0.847423358132618,-0.998385723218667,-0.645185458209895,0.0881816146262478,0.600288331628137,0.729709985043598,0.544204492230210,-0.00353330049452504,-0.710198234341683,-0.996151283404197,-0.582848247603573,0.132714623695096,0.608832959290982,0.729300304694092,0.530917745127462,-0.0567974266418885,-0.763834474349413,-0.996194821302556,-0.809464508248058,-0.722370734757158,-0.887004700155539,-0.998413891928257,-0.853297025846191,-0.752622312852237,-0.892041082597257,-0.998389461167205,-0.853166343681948,-0.752622312852161,-0.892175182111558,-0.998330425700788,-0.844706992111433,-0.716478302214214,-0.853355169435872,-0.998361077471973,-0.645220773600571,0.0871543344302048,0.587409029666716,0.692250558437195,0.475899795712802,-0.00303517185239273,-0.451789036113611,-0.658051138031429,-0.698545942759160,-0.736723882158716,-0.884380682699189,-0.998810656439017,-0.696163192931729,0.0870538795008387,0.803296337354023,0.996816662335615,0.839280362331595,0.684582440899851,0.648879642098527,0.674721373345186,0.699268927760007,0.706801046098655,0.706929576696901,0.707106781186590,0.706896882687383,0.706377200875035,0.695697301315928,0.668435605032255,0.648300953335634,0.705999739557863,0.879159457569764,0.998880944572359,0.707742430857824,-0.0338630347488611,-0.753100434531261,-0.996940166547381,-0.649754178663116,0.0785993868641453,0.760499941600475,0.996824696887334,0.637081241034182,-0.132614676812292,-0.819100524136026,-0.991096828391547,-0.792965071801910,-0.683441278734527,-0.838654595980670,-0.999996869856939,-0.715487301561719,0.0334309982311625,0.752935557122826,0.996940166547391,0.649754178663207,-0.0785993868640266,-0.760499941600398,-0.996824696887343,-0.637081241034274,0.132614676812167,0.819100524135953,0.991096828391564,0.792965071801991,0.683441278734625,0.838654595980743,0.999996869856939,0.715487301561634,-0.0334309982312886,-0.752935557122904,-0.996940166547381,-0.649754178663116,0.0785993868641453,0.760499941600475,0.996824696887334,0.637081241034182,-0.132614676812292,-0.819100524136026,-0.991096828391547,-0.792965071801910,-0.683441278734527,-0.838654595980670,-0.999996869856939,-0.715487301561719,0.0334309982311625,0.752935557122826,0.996940166547391,0.649754178663207,-0.0785993868640266,-0.760499941600398,-0.996824696887343,-0.637081241034274,0.132614676812167,0.819100524135953,0.991096828391564,0.792965071801991,0.683441278734625,0.838654595980743,0.999996869856939,0.715487301561634,-0.0334309982312886,-0.752935557122904,-0.996940166547381,-0.649754178663116,0.0785993868641453,0.760499941600475,0.996824696887334,0.637081241034182,-0.132614676812292,-0.819100524136026,-0.991096828391547,-0.792965071801910,-0.683441278734527,-0.838654595980670,-0.999996869856939,-0.715487301561719,0.0334309982311625,0.752935557122826,0.996940166547391,0.649754178663207,-0.0785993868640266,-0.760499941600398,-0.996824696887343,-0.637081241034274,0.132614676812167,0.819100524135953,0.991096828391564,0.792965071801991,0.683441278734625,0.838654595980743,0.999996869856939,0.715487301561634,-0.0334309982312886,-0.752935557122904,-0.996940166547381,-0.649754178663116,0.0785993868641453,0.760499941600475,0.996824696887334,0.637081241034182,-0.132614676812292,-0.819100524136026,-0.991096828391547,-0.792965071801910,-0.683441278734527,-0.838654595980670,-0.999996869856939,-0.715487301561719,0.0334309982311625,0.752935557122826,0.996940166547391,0.649754178663207,-0.0785993868640266,-0.760499941600398,-0.996824696887343,-0.637081241034274,0.132614676812167,0.819100524135953,0.991096828391564,0.792965071801991,0.683441278734625,0.838654595980743,0.999996869856939,0.715487301561634,-0.0334309982312886,-0.752935557122904,-0.996940166547381,-0.649754178663116,0.0785993868641453,0.760499941600475,0.996824696887334,0.637081241034182,-0.132614676812292,-0.819100524136026,-0.991096828391547,-0.792965071801910,-0.683441278734527,-0.838654595980670,-0.999996869856939,-0.715487301561719,0.0334309982311625,0.752935557122826,0.996940166547391,0.649754178663207,-0.0785993868640266,-0.760499941600398,-0.996824696887343,-0.637081241034274,0.132614676812167,0.819100524135953,0.991096828391564,0.792965071801991,0.683441278734625,0.838654595980743,0.999996869856939,0.715487301561634,-0.0334309982312886,-0.752935557122904,-0.996940166547381,-0.649754178663116,0.0785993868641453,0.760499941600475,0.996824696887334,0.637081241034182,-0.132614676812292,-0.819100524136026,-0.991096828391547,-0.792965071801910,-0.683441278734527,-0.838654595980670,-0.999996869856939,-0.715487301561719,0.0334309982311625,0.752935557122826,0.996940166547391,0.649754178663207,-0.0785993868640266,-0.760499941600398,-0.996824696887343,-0.637081241034274,0.132614676812167,0.819100524135953,0.991096828391564,0.792965071801991,0.683441278734625,0.838654595980743,0.999996869856939,0.715487301561634,-0.0334309982312886,-0.752935557122904,-0.996940166547381,-0.649754178663116,0.0785993868641453,0.760499941600475,0.996824696887334,0.637081241034182,-0.132614676812292,-0.819100524136026,-0.991096828391547,-0.792965071801910,-0.683441278734527,-0.838654595980670,-0.999996869856939,-0.715487301561719,0.0334309982311625,0.752935557122826,0.996940166547391,0.649754178663207,-0.0785993868640266,-0.760499941600398,-0.996824696887343,-0.637081241034274,0.132614676812167,0.819100524135953,0.991096828391564,0.792965071801991,0.683441278734625,0.838654595980743,0.999996869856939,0.715487301561634,-0.0334309982312886,-0.752935557122904,-0.996940166547381,-0.649754178663116,0.0785993868641453,0.760499941600475,0.996824696887334,0.637081241034182,-0.132614676812292,-0.819100524136026,-0.991096828391547,-0.792965071801915,-0.683441278734533,-0.838654595980674,-0.999996869856939,-0.715487301561714,0.0334309982311696,0.752935557122826,0.996940166547390,0.649528541270407,-0.0796274053040610,-0.770607542898204,-0.991303900635729,-0.586688109480063,0.0801588206286395};
	for(int i = 0; i < 384; i++){
		cuComplex temp;
		temp.x = p_i[i];
		temp.y = p_q[i];
		p[i] = temp;
	}

	cudaMemcpy(dev_Matched_GPU0,	 p,		384 * sizeof(cuComplex), cudaMemcpyHostToDevice);
	cufftExecC2C(plan_pd, dev_Matched_GPU0, dev_Matched_GPU0, CUFFT_FORWARD);





}
void GPUHandler::free_pd_variables() {
	delete [] L_pd;
	delete [] max_locations;
	delete [] old_iSamples_pd;
	delete [] old_qSamples_pd;

	cudaFree(dev_Samples_GPU0_pd);
	cudaFree(dev_Matched_GPU0);
	cudaFree(dev_iSamples_pd);
	cudaFree(dev_qSamples_pd);
	cudaFree(dev_iSamples_pd_last_times);
	cudaFree(dev_qSamples_pd_last_times);
	cudaFree(dev_old_iSamples_pd);
	cudaFree(dev_old_qSamples_pd);
	cudaFree(dev_Samples_GPU0);
	cudaFree(dev_Samples_GPU1);
	cudaFree(dev_Samples_GPU2);
	cudaFree(dev_blockSums_pd);
	cudaFree(dev_L_pd);
	cudaFree(dev_L_pd_last_times);
	cudaFree(dev_max_locations);
	cudaFree(dev_max_locations_last_times);
	cudaFree(dev_max_locations_save);
	cudaFree(dev_num_good_maximums);
	cudaFree(dev_endIndex_FirstWindow);
	cudaFree(dev_myFirstMaxMod);
	cudaFree(dev_myFirstMaxActual);
}
void GPUHandler::preFindPreambles(float* iSamples, float* qSamples) {
	cudaSetDevice(2);	cudaDeviceSynchronize();	cudaSetDevice(1);	cudaDeviceSynchronize();	cudaSetDevice(0);	cudaDeviceSynchronize();

	cudaError_t cudaStat;

	//	cudaSetDevice(device_GPU0); cudaMemcpy(dev_max_locations_last_times, dev_max_locations, 3105*sizeof(int), cudaMemcpyDeviceToDevice);

	cudaSetDevice(device_GPU1); cudaStat = cudaMemcpy(dev_iSamples_pd,   dev_iSamples_GPU1, TOTAL_SAMPLES_LENGTH*sizeof(float), cudaMemcpyDeviceToDevice); if(cudaStat != cudaSuccess)	cout << "ERROR - Could not copy i samples from GPU1 to GPU0 " << cudaStat << endl;
	cudaSetDevice(device_GPU1); cudaStat = cudaMemcpy(dev_qSamples_pd,   dev_qSamples_GPU1, TOTAL_SAMPLES_LENGTH*sizeof(float), cudaMemcpyDeviceToDevice); if(cudaStat != cudaSuccess)	cout << "ERROR - Could not copy q samples from GPU1 to GPU0 " << cudaStat << endl;

	cudaSetDevice(2);	cudaDeviceSynchronize();	cudaSetDevice(1);	cudaDeviceSynchronize();	cudaSetDevice(0);	cudaDeviceSynchronize();

	//	cudaSetDevice(device_GPU0); cudaStat = cudaMemcpy(iSamples,   dev_iSamples_pd, 39321600*sizeof(float), cudaMemcpyDeviceToHost); if(cudaStat != cudaSuccess)	cout << "ERROR - Could not copy iSamples from GPU0 to host " << cudaStat << endl;
	//	cudaSetDevice(device_GPU0); cudaStat = cudaMemcpy(qSamples,   dev_qSamples_pd, 39321600*sizeof(float), cudaMemcpyDeviceToHost); if(cudaStat != cudaSuccess)	cout << "ERROR - Could not copy iSamples from GPU0 to host " << cudaStat << endl;
	//	cudaSetDevice(2);	cudaDeviceSynchronize();	cudaSetDevice(1);	cudaDeviceSynchronize();	cudaSetDevice(0);	cudaDeviceSynchronize();
	////	myMaxi = 0;
	////	myMaxq = 0;
	//	int myi_i = 0;
	//	int myi_q = 0;
	//	for(int i = 1; i < TOTAL_SAMPLES_LENGTH-1; i++){
	//		float iTest =abs(abs(iSamples[i+1])-abs(iSamples[i]));
	//		float qTest =abs(abs(qSamples[i+1])-abs(qSamples[i]));
	//		if(iTest > myMaxi){
	//			myMaxi = iTest;
	//			myi_i = i;
	//		}
	//		if(qTest > myMaxq){
	//			myMaxq = qTest;
	//			myi_q = i;
	//		}
	//
	//	}
	//	cout << "Max diff on real found is " << myMaxi << " at " << myi_i << endl;
	//	cout << "Max diff on imag found is " << myMaxq << " at " << myi_q << endl;




	//	cudaSetDevice(2);	cudaDeviceSynchronize();	cudaSetDevice(1);	cudaDeviceSynchronize();	cudaSetDevice(0);	cudaDeviceSynchronize();
	//	//Copy old samples into the iSamples and qSamples buffers
	//	cudaMemcpy(qSamples, dev_old_qSamples_pd, NUM_OLD_SAMPLES * sizeof(float), cudaMemcpyDeviceToHost);
	//	cudaMemcpy(iSamples, dev_old_iSamples_pd, NUM_OLD_SAMPLES * sizeof(float), cudaMemcpyDeviceToHost);
	//
	//	//Copy Samples to GPU0
	//	cudaMemcpy(dev_qSamples_pd, qSamples, TOTAL_SAMPLES_LENGTH * sizeof(float), cudaMemcpyHostToDevice);
	//	cudaMemcpy(dev_iSamples_pd, iSamples, TOTAL_SAMPLES_LENGTH * sizeof(float), cudaMemcpyHostToDevice);
	//
	//	//Save the last 12671 samples from the current 150MB section. These will be the 'old' samples in the next run
	//	cudaMemcpy(dev_old_qSamples_pd, &qSamples[NUM_INPUT_SAMPLES_DEFAULT], NUM_OLD_SAMPLES * sizeof(float), cudaMemcpyHostToDevice);
	//	cudaMemcpy(dev_old_iSamples_pd, &iSamples[NUM_INPUT_SAMPLES_DEFAULT], NUM_OLD_SAMPLES * sizeof(float), cudaMemcpyHostToDevice);
	//
	//	cudaSetDevice(2);	cudaDeviceSynchronize();	cudaSetDevice(1);	cudaDeviceSynchronize();	cudaSetDevice(0);	cudaDeviceSynchronize();
}
void GPUHandler::preambleDetector() {
	calculateInnerSumBlocksNew	<<<numBlocks_calculateInnerSumBlocksNew, 	numThreads_calculateInnerSumBlocksNew>>>	(dev_iSamples_pd, dev_qSamples_pd, dev_blockSums_pd, 0, maxThreads_calculateInnerSumBlocksNew);
	calculateOuterSumsNew		<<<numBlocks_calculateOuterSumsNew, 		numThreads_calculateOuterSumsNew>>>			(dev_blockSums_pd, dev_L_pd, maxThreads_calculateOuterSumsNew);
	cudaFirstMaxSearch			<<<1, 										1>>>										(dev_L_pd, SAMPLES_PER_PACKET, dev_endIndex_FirstWindow,dev_myFirstMaxMod,dev_myFirstMaxActual);
	findPreambleMaximums		<<<numBlocks_findPreambleMaximums, 			numThreads_findPreambleMaximums>>>			(dev_L_pd, dev_max_locations, dev_max_locations_save, SAMPLES_PER_PACKET, dev_myFirstMaxMod, TOTAL_SAMPLES_LENGTH, maxThreads_findPreambleMaximums);
	cudaLongestChainSearch		<<<1, 										1>>>										(dev_max_locations, SAMPLES_PER_PACKET);
	//cudaMaxAdjust				<<<numBlocks_cudaMaxAdjust, 				numThreads_cudaMaxAdjust>>>					(dev_max_locations, SAMPLES_PER_PACKET, maxThreads_cudaMaxAdjust);
	stripSignalFloatToComplex	<<<numBlocks_stripSignalFloatToComplex, 	numThreads_stripSignalFloatToComplex>>>		(dev_iSamples_pd, dev_qSamples_pd, dev_Samples_GPU0, dev_max_locations, SAMPLES_PER_PACKET, maxThreads_stripSignalFloatToComplex);

	//	cudaSetDevice(device_GPU0);
	//
	//	// Set optimal number of threads per block
	//	int numTreadsPerBlock = 512;
	//
	//	// Compute number of blocks for set number of threads
	//	int numBlocks = TOTAL_SAMPLES_LENGTH/numTreadsPerBlock;
	//
	//	// If there are left over points, run an extra block
	//	if(TOTAL_SAMPLES_LENGTH % numTreadsPerBlock > 0)
	//		numBlocks++;
	//
	//	//	cudaEvent_t start_GPU_FFT, stop_GPU_FFT;
	//	//	cudaEventCreate(&start_GPU_FFT);
	//	//	cudaEventCreate(&stop_GPU_FFT);
	//	//
	//	//	cudaDeviceSynchronize();
	//	//	cudaEventRecord(start_GPU_FFT);
	//
	//	cudaMemset(dev_Samples_GPU0_pd, 0, Nfft_pd*sizeof(cuComplex));
	//	floatToSamples<<<numBlocks, numTreadsPerBlock>>>(dev_Samples_GPU0_pd, dev_iSamples_pd, dev_qSamples_pd, TOTAL_SAMPLES_LENGTH);
	//
	//	cufftExecC2C(plan_pd, dev_Samples_GPU0_pd, dev_Samples_GPU0_pd, CUFFT_FORWARD);
	//	//			cufftExecC2C(plan_pd, dev_myFilter5, dev_myFilter5, CUFFT_FORWARD);
	//
	//	// Set optimal number of threads per block
	//	numTreadsPerBlock = 512;
	//
	//	// Compute number of blocks for set number of threads
	//	numBlocks = Nfft_pd/numTreadsPerBlock;
	//
	//	// If there are left over points, run an extra block
	//	if(Nfft_pd % numTreadsPerBlock > 0)
	//		numBlocks++;
	//
	//	// Run computation on device
	//	PointToPointMultiply<<<numBlocks, numTreadsPerBlock>>>(dev_Samples_GPU0_pd, dev_Matched_GPU0, Nfft_pd);
	//	//
	//	//	cout << "Nfft_pd " << Nfft_pd << endl;
	//	//
	//	cufftExecC2C(plan_pd, dev_Samples_GPU0_pd, dev_Samples_GPU0_pd, CUFFT_INVERSE);
	//
	//	// Run computation on device
	//	myAbs<<<numBlocks, numTreadsPerBlock>>>(dev_Samples_GPU0_pd, Nfft_pd);
	//
	//	cudaFirstMaxSearch			<<<1, 										1>>>										(dev_Samples_GPU0_pd, SAMPLES_PER_PACKET, dev_endIndex_FirstWindow,dev_myFirstMaxMod,dev_myFirstMaxActual);
	//	findPreambleMaximums		<<<numBlocks_findPreambleMaximums, 			numThreads_findPreambleMaximums>>>			(dev_Samples_GPU0_pd, dev_max_locations, dev_max_locations_save, SAMPLES_PER_PACKET, dev_myFirstMaxMod, TOTAL_SAMPLES_LENGTH, maxThreads_findPreambleMaximums);
	//	stripSignalFloatToComplex	<<<numBlocks_stripSignalFloatToComplex, 	numThreads_stripSignalFloatToComplex>>>		(dev_iSamples_pd, dev_qSamples_pd, dev_Samples_GPU0, dev_max_locations, SAMPLES_PER_PACKET, maxThreads_stripSignalFloatToComplex);
	//	//cudaMaxAdjust				<<<1,1>>>					(dev_max_locations, SAMPLES_PER_PACKET, maxThreads_cudaMaxAdjust);
	//
	//	//	cudaDeviceSynchronize();
	//	//	cudaEventRecord(stop_GPU_FFT);
	//	//	cudaEventSynchronize(stop_GPU_FFT);
	//	//	float milliseconds_GPU_FFT = 0;
	//	//	cudaEventElapsedTime(&milliseconds_GPU_FFT, start_GPU_FFT, stop_GPU_FFT);
	//	//	cudaEventDestroy(start_GPU_FFT);
	//	//	cudaEventDestroy(stop_GPU_FFT);
	//	//
	//	//	cout << "\n\n FFT TIME!!! " << milliseconds_GPU_FFT << endl << endl << endl;
	//	//	//
	//	//	//	//findPreambleMaximums		<<<numBlocks_findPreambleMaximums, 			numThreads_findPreambleMaximums>>>			(dev_Samples_GPU0_pd, dev_max_locations, dev_max_locations_save, SAMPLES_PER_PACKET, dev_myFirstMaxMod, TOTAL_SAMPLES_LENGTH, maxThreads_findPreambleMaximums);
	//	//	//
	//	//			writeBatch_Max			(3103);
	//	//			writeBatch_samples0_pd	(39321600+12671+500);
	//	//			writeBatch_raw_i		(TOTAL_SAMPLES_LENGTH);
	//	//	//		writeBatch_raw_q		(TOTAL_SAMPLES_LENGTH);
	//	//	//		writeBatch_L			(39321600+12671);
	//
	//	cudaError_t code = cudaGetLastError();
	//	if(code)
	//		cout << "\n\n\nCode " << code << " in FFT stuff\n\n\n" << endl;

}

void GPUHandler::initialize_foe_variables() {
	cudaError_t cudaStat;

	cudaStat = cudaMalloc(&dev_r1, 			r1_length * (MAX_PACKETS_PER_MEMORY_SECTION+1) 	* sizeof(cuComplex));	if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_r1 on GPU" << endl;
	cudaStat = cudaMalloc(&dev_r1_conj, 	r1_length * (MAX_PACKETS_PER_MEMORY_SECTION+1) 	* sizeof(cuComplex));	if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_r1_conj on GPU" << endl;
	cudaStat = cudaMalloc(&dev_complex_w0, 	(MAX_PACKETS_PER_MEMORY_SECTION+1) 				* sizeof(cuComplex));	if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_complex_w0 on GPU" << endl;
	cudaStat = cudaMalloc(&dev_w0, 			(MAX_PACKETS_PER_MEMORY_SECTION+1) 				* sizeof(float));		if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_complex_w0 on GPU" << endl;
	cudaStat = cudaMalloc(&dev_r1_conj_list,MAX_PACKETS_PER_MEMORY_SECTION 					* sizeof(cuComplex*));	if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_r1_conj_list on GPU" << endl;
	cudaStat = cudaMalloc(&dev_r1_list, 	MAX_PACKETS_PER_MEMORY_SECTION 					* sizeof(cuComplex*));	if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_r1_list on GPU" << endl;
	cudaStat = cudaMalloc(&dev_complex_w0_list, MAX_PACKETS_PER_MEMORY_SECTION 				* sizeof(cuComplex*));	if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_complex_w0_list on GPU" << endl;

	cublasCreate_v2(&dot_handle);
	//	cublasSetStream_v2(dot_handle,stream_GPU0_array[stream_0]);
	r1_alpha.x = 1;	r1_alpha.y = 0;
	r1_beta.x = 0; r1_beta.y = 0;

	// create lists of device pointers to inputs and outputs
	cuComplex **r1_conj_list = 0, **r1_list = 0, **w0_list = 0;

	// malloc space for cuComplex POINTERS
	r1_conj_list = 	(cuComplex**)malloc(MAX_PACKETS_PER_MEMORY_SECTION * sizeof(cuComplex*));
	r1_list = 		(cuComplex**)malloc(MAX_PACKETS_PER_MEMORY_SECTION * sizeof(cuComplex*));
	w0_list = 		(cuComplex**)malloc(MAX_PACKETS_PER_MEMORY_SECTION * sizeof(cuComplex*));

	for(int i = 0; i < MAX_PACKETS_PER_MEMORY_SECTION; i++){
		r1_conj_list[i] = dev_r1_conj + dot_m*dot_k * i;
		r1_list[i] = dev_r1 + dot_k*dot_n * i;
		w0_list[i] = dev_complex_w0 + dot_m*dot_n * i;
	}

	cudaMemcpy(dev_r1_conj_list,	r1_conj_list,	MAX_PACKETS_PER_MEMORY_SECTION * sizeof(cuComplex*), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_r1_list, r1_list,		MAX_PACKETS_PER_MEMORY_SECTION * sizeof(cuComplex*), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_complex_w0_list,	w0_list,		MAX_PACKETS_PER_MEMORY_SECTION * sizeof(cuComplex*), cudaMemcpyHostToDevice);

	delete [] r1_conj_list;
	delete [] r1_list;
	delete [] w0_list;

	maxThreads_signalStripperFreq = r1_length * MAX_PACKETS_PER_MEMORY_SECTION;
	numThreads_signalStripperFreq = 512; // Tested Again
	numBlocks_signalStripperFreq = maxThreads_signalStripperFreq/numThreads_signalStripperFreq;
	if((maxThreads_signalStripperFreq % numThreads_signalStripperFreq) > 0) numBlocks_signalStripperFreq++;

	maxThreads_tanAndLq = MAX_PACKETS_PER_MEMORY_SECTION+1;
	numThreads_tanAndLq = 512; // Tested Again
	numBlocks_tanAndLq = maxThreads_tanAndLq/numThreads_tanAndLq;
	if((maxThreads_tanAndLq % numThreads_tanAndLq) > 0) numBlocks_tanAndLq++;

	maxThreads_derotateBatchSumW0 = TOTAL_SAMPLES_LENGTH;
	numThreads_derotateBatchSumW0 = 128; // Tested
	numBlocks_derotateBatchSumW0 = maxThreads_derotateBatchSumW0/numThreads_derotateBatchSumW0;
	if((maxThreads_derotateBatchSumW0 % numThreads_derotateBatchSumW0) > 0) numBlocks_derotateBatchSumW0++;
}
void GPUHandler::free_foe_variables() {
	cudaFree(dev_r1);
	cudaFree(dev_r1_conj);
	cudaFree(dev_complex_w0);
	cudaFree(dev_w0);
	cudaFree(dev_r1_conj_list);
	cudaFree(dev_r1_list);
	cudaFree(dev_complex_w0_list);
}
void GPUHandler::estimateFreqOffsetAndRotate() {
	cudaSetDevice(device_GPU0); signalStripper		<<<numBlocks_signalStripperFreq, 	numThreads_signalStripperFreq>>>		(dev_Samples_GPU0, dev_r1, 	 	SAMPLES_PER_PACKET, 	r1_start,		r1_length, r1_flag, 	 maxThreads_signalStripperFreq);
	cudaSetDevice(device_GPU0); signalStripper		<<<numBlocks_signalStripperFreq, 	numThreads_signalStripperFreq>>>		(dev_Samples_GPU0, dev_r1_conj, SAMPLES_PER_PACKET, 	r1_conj_start, 	r1_length, r1_conj_flag, maxThreads_signalStripperFreq);
	cudaSetDevice(device_GPU0); cublasCgemmBatched	(dot_handle, 						CUBLAS_OP_T, 							CUBLAS_OP_T, dot_m, dot_n, dot_k, &r1_alpha, (const cuComplex**)dev_r1_conj_list, dot_lda, (const cuComplex**)dev_r1_list, dot_ldb, &r1_beta, dev_complex_w0_list, dot_ldc, MAX_PACKETS_PER_MEMORY_SECTION);
	cudaSetDevice(device_GPU0); tanAndLq			<<<numBlocks_tanAndLq, 				numThreads_tanAndLq>>>					(dev_w0, dev_complex_w0, maxThreads_tanAndLq);
	cudaSetDevice(device_GPU0); w0Ave				<<<1, 								1>>>									(dev_w0, MAX_PACKETS_PER_MEMORY_SECTION);
	cudaSetDevice(device_GPU0); derotateBatchSumW0	<<<numBlocks_derotateBatchSumW0, 	numThreads_derotateBatchSumW0>>>		(dev_Samples_GPU0, dev_w0, SAMPLES_PER_PACKET, maxThreads_derotateBatchSumW0);

	// Copy the Derotated Samples from GPU0 to GPU1 and GPU2
	cudaSetDevice(device_GPU1); cudaMemcpy(dev_Samples_GPU1,	 dev_Samples_GPU0,		TOTAL_SAMPLES_LENGTH * sizeof(cuComplex), cudaMemcpyDeviceToDevice);
	cudaSetDevice(device_GPU2); cudaMemcpy(dev_Samples_GPU2,	 dev_Samples_GPU0,		TOTAL_SAMPLES_LENGTH * sizeof(cuComplex), cudaMemcpyDeviceToDevice);
}

void GPUHandler::initialize_channel_estimator_variables() {
	cudaError_t cudaStat;

	cudaStat = cudaMalloc(&dev_channelEst_piX, 			CHAN_EST_MATRIX_NUM_ENTRIES * (MAX_PACKETS_PER_MEMORY_SECTION+1) 	* sizeof(cuComplex));	if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_r2 on GPU" << endl;
	cudaStat = cudaMalloc(&dev_r2, 						channelEst_k*channelEst_n * (MAX_PACKETS_PER_MEMORY_SECTION+1) 		* sizeof(cuComplex));	if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_channelEst_piX on GPU" << endl;
	cudaStat = cudaMalloc(&dev_channelEst, 				channelEst_m*channelEst_n * (MAX_PACKETS_PER_MEMORY_SECTION+1) 		* sizeof(cuComplex));	if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_channelEst on GPU" << endl;
	cudaStat = cudaMalloc(&dev_channelEst_piX_list,		MAX_PACKETS_PER_MEMORY_SECTION 										* sizeof(cuComplex*));	if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_channelEst_piX_list on GPU" << endl;
	cudaStat = cudaMalloc(&dev_r2_list, 				MAX_PACKETS_PER_MEMORY_SECTION 										* sizeof(cuComplex*));	if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_r2_list on GPU" << endl;
	cudaStat = cudaMalloc(&dev_channelEst_list, 		MAX_PACKETS_PER_MEMORY_SECTION 										* sizeof(cuComplex*));	if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_channelEst_list on GPU" << endl;

	cublasCreate_v2(&channelEst_handle);
	//	cublasSetStream_v2(channelEst_handle,stream_GPU0_array[stream_0]);
	channelEst_alpha.x = 1;	channelEst_alpha.y = 0;
	channelEst_beta.x = 0; channelEst_beta.y = 0;

	// create lists of device pointers to inputs and outputs
	cuComplex **piX_list = 0, **r2_list = 0, **channelEst_list = 0;

	// malloc space for cuComplex POINTERS
	piX_list = 			(cuComplex**)malloc(MAX_PACKETS_PER_MEMORY_SECTION * sizeof(cuComplex*));
	r2_list = 			(cuComplex**)malloc(MAX_PACKETS_PER_MEMORY_SECTION * sizeof(cuComplex*));
	channelEst_list = 	(cuComplex**)malloc(MAX_PACKETS_PER_MEMORY_SECTION * sizeof(cuComplex*));

	for(int i = 0; i < MAX_PACKETS_PER_MEMORY_SECTION; i++){
		piX_list[i] = dev_channelEst_piX 	+ channelEst_m*channelEst_k * i;
		r2_list[i] = dev_r2 				+ channelEst_k*channelEst_n * i;
		channelEst_list[i] = dev_channelEst + channelEst_m*channelEst_n * i;
	}

	cudaMemcpy(dev_channelEst_piX_list,		piX_list,			MAX_PACKETS_PER_MEMORY_SECTION * sizeof(cuComplex*), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_r2_list, 				r2_list,			MAX_PACKETS_PER_MEMORY_SECTION * sizeof(cuComplex*), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_channelEst_list,			channelEst_list,	MAX_PACKETS_PER_MEMORY_SECTION * sizeof(cuComplex*), cudaMemcpyHostToDevice);

	cuComplex* piX 	= new cuComplex[CHAN_EST_MATRIX_NUM_ENTRIES * MAX_PACKETS_PER_MEMORY_SECTION];

	FileReader fileReader;
	channelEstMatrixSamples = fileReader.loadCSVFile("src/ChanEstMatrix.csv");
	chanEstMatrix = new cuComplex[CHAN_EST_MATRIX_NUM_ENTRIES];
	for(int i=0; i < CHAN_EST_MATRIX_NUM_ENTRIES; i++)
		chanEstMatrix[i].x = channelEstMatrixSamples.getI().at(i);
	for(int i=0; i < CHAN_EST_MATRIX_NUM_ENTRIES; i++)
		chanEstMatrix[i].y = channelEstMatrixSamples.getQ().at(i);
	for(int batch = 0; batch < MAX_PACKETS_PER_MEMORY_SECTION; batch++)
		for(int i=0; i < CHAN_EST_MATRIX_NUM_ENTRIES; i++)
			piX[batch*CHAN_EST_MATRIX_NUM_ENTRIES+i] = chanEstMatrix[i];
	cudaMemcpy(dev_channelEst_piX,	piX,		CHAN_EST_MATRIX_NUM_ENTRIES*MAX_PACKETS_PER_MEMORY_SECTION * sizeof(cuComplex), cudaMemcpyHostToDevice);

	delete [] piX;
	delete [] piX_list;
	delete [] r2_list;
	delete [] channelEst_list;


	// Apply Filters stuff that blows up
	cudaSetDevice(device_GPU0);	cufftPlan1d(&fftPlan_apply_GPU0, conv_length, CUFFT_C2C, MAX_PACKETS_PER_MEMORY_SECTION);
	//cudaSetDevice(device_GPU1);	cufftPlan1d(&fftPlan_apply_GPU1, conv_length, CUFFT_C2C, MAX_PACKETS_PER_MEMORY_SECTION);
	cudaSetDevice(device_GPU2);	cufftPlan1d(&fftPlan_apply_GPU2, conv_length, CUFFT_C2C, MAX_PACKETS_PER_MEMORY_SECTION);
	cudaSetDevice(device_GPU0);

	cufftPlan1d(&fftPlan_signal_GPU0, conv_length, CUFFT_C2C, MAX_PACKETS_PER_MEMORY_SECTION);
	//	cufftSetStream(fftPlan_signal_GPU0,stream_GPU0_array[stream_1]);

	fftscale.x = (float)1/conv_length;
	fftscale.y = 0;


	maxThreads_signalStripperChannel = r2_length * MAX_PACKETS_PER_MEMORY_SECTION;
	numThreads_signalStripperChannel = 128; // Tested Again
	numBlocks_signalStripperChannel = maxThreads_signalStripperChannel/numThreads_signalStripperChannel;
	if((maxThreads_signalStripperChannel % numThreads_signalStripperChannel) > 0) numBlocks_signalStripperChannel++;
}
void GPUHandler::free_channel_estimator_variables() {
	/*	// create lists of device pointers to inputs and outputs
	cuComplex **piX_list = 0, **r2_list = 0, **channelEst_list = 0;

	// malloc space for cuComplex POINTERS
	piX_list = 	(cuComplex**)malloc(MAX_PACKETS_PER_MEMORY_SECTION * sizeof(cuComplex*));
	r2_list = 		(cuComplex**)malloc(MAX_PACKETS_PER_MEMORY_SECTION * sizeof(cuComplex*));
	channelEst_list = 		(cuComplex**)malloc(MAX_PACKETS_PER_MEMORY_SECTION * sizeof(cuComplex*));

	for(int i = 0; i < MAX_PACKETS_PER_MEMORY_SECTION; i++){
		piX_list[i] = dev_channelEst_piX 	+ channelEst_m*channelEst_k * i;
		r2_list[i] = dev_r2 				+ channelEst_k*channelEst_n * i;
		channelEst_list[i] = dev_channelEst + channelEst_m*channelEst_n * i;
	}

	cudaMemcpy(dev_channelEst_piX_list,		piX_list,			MAX_PACKETS_PER_MEMORY_SECTION * sizeof(cuComplex*), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_r2_list, 				r2_list,			MAX_PACKETS_PER_MEMORY_SECTION * sizeof(cuComplex*), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_channelEst_list,			channelEst_list,	MAX_PACKETS_PER_MEMORY_SECTION * sizeof(cuComplex*), cudaMemcpyHostToDevice);

	cuComplex* piX 	= new cuComplex[CHAN_EST_MATRIX_NUM_ENTRIES * MAX_PACKETS_PER_MEMORY_SECTION];*/
	cudaFree(dev_channelEst_piX);
	cudaFree(dev_r2);
	cudaFree(dev_channelEst);
	cudaFree(dev_channelEst_piX_list);
	cudaFree(dev_r2_list);
	cudaFree(dev_channelEst_list);
	delete [] chanEstMatrix;
}
void GPUHandler::estimate_channel() {
	cudaSetDevice(device_GPU0); signalStripper		<<<numBlocks_signalStripperChannel, 	numThreads_signalStripperChannel>>>	(dev_Samples_GPU0, dev_r2, SAMPLES_PER_PACKET, r2_start, r2_length, r2_flag, maxThreads_signalStripperChannel);
	cudaSetDevice(device_GPU0); cublasCgemmBatched	(channelEst_handle, CUBLAS_OP_N, CUBLAS_OP_N,	channelEst_m, channelEst_n,	channelEst_k, &r1_alpha, (const cuComplex**)dev_channelEst_piX_list, channelEst_lda, (const cuComplex**)dev_r2_list, channelEst_ldb, &r1_beta, dev_channelEst_list, channelEst_ldc, MAX_PACKETS_PER_MEMORY_SECTION);
	cudaSetDevice(device_GPU0); cudaMemcpy(dev_h_hat_mmse_GPU0, dev_channelEst, channelEst_m*MAX_PACKETS_PER_MEMORY_SECTION*sizeof(cuComplex), cudaMemcpyDeviceToDevice);
	cudaSetDevice(device_GPU1); cudaMemcpy(dev_h_hat_zf_GPU1,   dev_channelEst, channelEst_m*MAX_PACKETS_PER_MEMORY_SECTION*sizeof(cuComplex), cudaMemcpyDeviceToDevice);
	cudaSetDevice(device_GPU2); cudaMemcpy(dev_h_hat_freq_GPU2, dev_channelEst, channelEst_m*MAX_PACKETS_PER_MEMORY_SECTION*sizeof(cuComplex), cudaMemcpyDeviceToDevice);
}

void GPUHandler::initialize_nv_estimator_variables() {
	cudaError_t cudaStat;

	cudaStat = cudaMalloc(&dev_noiseMultiply_X, 	noiseMultiply_X_length*MAX_PACKETS_PER_MEMORY_SECTION* sizeof(cuComplex));	if(cudaStat != cudaSuccess) cout << "ERROR -- Could not allocate space for dev_noiseMultiply_X on GPU" << endl;
	cudaStat = cudaMalloc(&dev_noiseMultiplyIntermediate,	r2_length * MAX_PACKETS_PER_MEMORY_SECTION 		* sizeof(cuComplex));	if(cudaStat != cudaSuccess) cout << "ERROR -- Could not allocate space for dev_noiseMultiplyIntermediate on GPU" << endl;
	cudaStat = cudaMalloc(&dev_ones, 			r2_length										* sizeof(cuComplex));	if(cudaStat != cudaSuccess) cout << "ERROR -- Could not allocate space for dev_noiseMultiplyIntermediate on GPU" << endl;
	cudaStat = cudaMalloc(&dev_ones_list, 					MAX_PACKETS_PER_MEMORY_SECTION 					* sizeof(cuComplex*));	if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_r2_list on GPU" << endl;
	cudaStat = cudaMalloc(&dev_noiseMultiply_X_list,		MAX_PACKETS_PER_MEMORY_SECTION 					* sizeof(cuComplex*));	if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_channelEst_piX_list on GPU" << endl;
	cudaStat = cudaMalloc(&dev_noiseIntermediate_list, 		MAX_PACKETS_PER_MEMORY_SECTION 					* sizeof(cuComplex*));	if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_r2_list on GPU" << endl;
	cudaStat = cudaMalloc(&dev_noiseVariance_GPU0, 			MAX_PACKETS_PER_MEMORY_SECTION 					* sizeof(float));	if(cudaStat != cudaSuccess) cout << "ERROR -- Could not allocate space for dev_noiseVariance_GPU0 (noiseMultiply variance estimator) on GPU" << endl;
	cudaStat = cudaMalloc(&dev_noiseVariance_GPU0_list, 	MAX_PACKETS_PER_MEMORY_SECTION 					* sizeof(float*));	if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_channelEst_list on GPU" << endl;
	cudaStat = cudaMalloc(&dev_diffMag2, 					r2_length * MAX_PACKETS_PER_MEMORY_SECTION 		* sizeof(float));	if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_diffMag2 on GPU" << endl;

	cublasCreate_v2(&noiseMultiplyVariance_handle);
	//	cublasSetStream_v2(noiseMultiplyVariance_handle,stream_GPU0_array[stream_0]);
	cublasCreate_v2(&noiseSumVariance_handle);
	//	cublasSetStream_v2(noiseSumVariance_handle,stream_GPU0_array[stream_0]);

	noiseMultiplyVariance_alpha.x = 1;		noiseMultiplyVariance_alpha.y = 0;
	noiseMultiplyVariance_beta.x  = 0; 		noiseMultiplyVariance_beta.y  = 0;
	noiseSumVariance_alpha.x = .0016;	noiseSumVariance_alpha.y = 0;
	noiseSumVariance_beta.x  = 0; 		noiseSumVariance_beta.y  = 0;

	// create lists of device pointers to inputs and outputs
	cuComplex **noiseMultiply_X_list = 0, **noiseMultiplyIntermediate_list = 0, **ones_list = 0;
	float **noiseVariance_list = 0;
	// malloc space for cuComplex POINTERS
	noiseMultiply_X_list = 			(cuComplex**)malloc(MAX_PACKETS_PER_MEMORY_SECTION * sizeof(cuComplex*));
	noiseMultiplyIntermediate_list = 	(cuComplex**)malloc(MAX_PACKETS_PER_MEMORY_SECTION * sizeof(cuComplex*));
	noiseVariance_list = 		(float**)malloc(MAX_PACKETS_PER_MEMORY_SECTION * sizeof(float*));
	ones_list = 		(cuComplex**)malloc(MAX_PACKETS_PER_MEMORY_SECTION * sizeof(cuComplex*));
	for(int i = 0; i < MAX_PACKETS_PER_MEMORY_SECTION; i++){
		noiseMultiply_X_list[i] = 			dev_noiseMultiply_X 			+ noiseMultiply_X_length * i;
		noiseMultiplyIntermediate_list[i] = dev_noiseMultiplyIntermediate 	+ r2_length * i;
		noiseVariance_list[i] = 			dev_noiseVariance_GPU0 			+ i;


		ones_list[i] = 						dev_ones;
	}
	cudaMemcpy(dev_noiseMultiply_X_list,	noiseMultiply_X_list, MAX_PACKETS_PER_MEMORY_SECTION * sizeof(cuComplex*), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_noiseIntermediate_list,	noiseMultiplyIntermediate_list,	MAX_PACKETS_PER_MEMORY_SECTION * sizeof(cuComplex*), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_noiseVariance_GPU0_list,	noiseVariance_list, 	MAX_PACKETS_PER_MEMORY_SECTION * sizeof(float*), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ones_list, 	ones_list, 			MAX_PACKETS_PER_MEMORY_SECTION * sizeof(cuComplex*), cudaMemcpyHostToDevice);

	//Load Ones Array
	cuComplex ones[r2_length];
	for(int i = 0; i < r2_length; i++){
		ones[i].x = 1;
		ones[i].y = 0;
	}
	cudaMemcpy(dev_ones, ones, r2_length * sizeof(cuComplex), cudaMemcpyHostToDevice);
	//Load X Matrix samples
	cuComplex* noiseMultiply_temp 	= new cuComplex[noiseMultiply_X_length * MAX_PACKETS_PER_MEMORY_SECTION];
	FileReader fileReader;
	xMatSamples = fileReader.loadCSVFile("src/X_Matrix.csv");
	for(uint idx=0; idx < xMatSamples.getSize(); idx++)
		noiseMultiply_temp[idx].x = xMatSamples.getI().at(idx);
	for(uint idx=0; idx < xMatSamples.getSize(); idx++)
		noiseMultiply_temp[idx].y = xMatSamples.getQ().at(idx);
	for(int batch = 0; batch < MAX_PACKETS_PER_MEMORY_SECTION; batch++)
		for(int i=0; i < xMatSamples.getSize(); i++)
			noiseMultiply_temp[batch*xMatSamples.getSize()+i] = noiseMultiply_temp[i];
	cudaMemcpy(dev_noiseMultiply_X, noiseMultiply_temp, noiseMultiply_X_length*MAX_PACKETS_PER_MEMORY_SECTION * sizeof(cuComplex), cudaMemcpyHostToDevice);

	maxThreads_subAndSquare = r2_length * MAX_PACKETS_PER_MEMORY_SECTION;
	numThreads_subAndSquare = 128; // Tested Again
	numBlocks_subAndSquare = maxThreads_subAndSquare/numThreads_subAndSquare;
	if((maxThreads_subAndSquare % numThreads_subAndSquare) > 0) numBlocks_subAndSquare++;

	maxThreads_sumAndScale = MAX_PACKETS_PER_MEMORY_SECTION;
	numThreads_sumAndScale = 32; // Tested Again
	numBlocks_sumAndScale = maxThreads_sumAndScale/numThreads_sumAndScale;
	if((maxThreads_sumAndScale % numThreads_sumAndScale) > 0) numBlocks_sumAndScale++;
}
void GPUHandler::free_nv_estimator_variables() {
	cudaFree(dev_noiseMultiply_X);
	cudaFree(dev_noiseMultiplyIntermediate);
	cudaFree(dev_noiseMultiply_X_list);
	cudaFree(dev_noiseIntermediate_list);
	cudaFree(dev_noiseVariance_GPU0);
	cudaFree(dev_noiseVariance_GPU0_list);
	cudaFree(dev_ones);
	cudaFree(dev_ones_list);
	cudaFree(dev_diffMag2);
}
void GPUHandler::calculate_noise_variance() {
	cudaSetDevice(device_GPU0); cublasCgemmBatched	(noiseMultiplyVariance_handle, CUBLAS_OP_N, CUBLAS_OP_N, noiseMultiply_m, noiseMultiply_n, noiseMultiply_k, &noiseMultiplyVariance_alpha, (const cuComplex**)dev_noiseMultiply_X_list, noiseMultiply_lda, (const cuComplex**)dev_channelEst_list, noiseMultiply_ldb, &noiseMultiplyVariance_beta, dev_noiseIntermediate_list, noiseMultiply_ldc, MAX_PACKETS_PER_MEMORY_SECTION);
	cudaSetDevice(device_GPU0); subAndSquare		<<<numBlocks_subAndSquare, 	numThreads_subAndSquare>>>	(dev_r2, dev_noiseMultiplyIntermediate, dev_diffMag2, maxThreads_subAndSquare);
	cudaSetDevice(device_GPU0); sumAndScale			<<<numBlocks_sumAndScale, 	numThreads_sumAndScale>>>	(dev_noiseVariance_GPU0, dev_diffMag2, maxThreads_sumAndScale);
	cudaSetDevice(device_GPU2);	cudaMemcpy(dev_shs_GPU2,dev_noiseVariance_GPU0,sizeof(float),cudaMemcpyDeviceToDevice);
}

void GPUHandler::initialize_equalizers_variables() {
	cudaError_t cudaStat;
	ifstream myFile;
	int csrRowPtrhhh[m+1];			//host new row pointer for csr format
	int csrColIdxhhh[nnzA];			//host col pointer for csr format

	//----------------------------------------------
	//	ZERO-FORCING EQUALIZER ON GPU1
	//----------------------------------------------
	int csrRowPtrhhh_switched[m+1];			//host new row pointer for csr format
	int csrColIdxhhh_switched[nnzA];			//host col pointer for csr format
	cudaSetDevice(device_GPU1);
	cudaStat = cudaMalloc(&dev_shs_GPU1, sizeof(float) * MAX_PACKETS_PER_MEMORY_SECTION); if(cudaStat != cudaSuccess) cout << "ERROR -- Could not allocate space for dev_shs_GPU1 (noise variance estimator) on GPU" << endl;
	cudaStat = cudaMalloc(&dev_h_hat_zf_GPU1, CHAN_SIZE * MAX_PACKETS_PER_MEMORY_SECTION*sizeof(cuComplex)); if(cudaStat != cudaSuccess) cout << "Error -- could not malloc space for dev_h_hat_zf in ZF Equalizer initialization" << endl;
	cudaStat = cudaMalloc(&dev_chanEstCorr_zf_GPU1, sizeof(cuComplex)*(CHAN_SIZE*2-1)*MAX_PACKETS_PER_MEMORY_SECTION); if(cudaStat != cudaSuccess) cout << "Error -- could not malloc space for dev_chanEstCorr_zf_GPU0 in ZF Equalizer initialization" << endl;
	cudaStat = cudaMalloc(&dev_hhh_csr_zf_GPU1, sizeof(cuComplex)*nnzA*MAX_PACKETS_PER_MEMORY_SECTION); if(cudaStat != cudaSuccess) cout << "Error -- could not malloc space for dev_hhh_csr_zf_GPU1 in ZF Equalizer initialization" << endl;
	cudaStat = cudaMalloc(&dev_hh_un0_zf_GPU1, sizeof(cuComplex)*EQUALIZER_LENGTH*MAX_PACKETS_PER_MEMORY_SECTION); if(cudaStat != cudaSuccess) cout << "Error -- could not malloc space for dev_hh_un0_zf_GPU1 in ZF Equalizer initialization" << endl;
	cudaStat = cudaMalloc(&dev_ZF_equalizers_GPU1, sizeof(cuComplex)*EQUALIZER_LENGTH*MAX_PACKETS_PER_MEMORY_SECTION); if(cudaStat != cudaSuccess) cout << "Error -- could not malloc space for dev_ZF_equalizers_GPU1 in ZF Equalizer initialization" << endl;
	cudaStat = cudaMalloc(&dev_csrColIdxhhh_zf_GPU1, sizeof(int) * nnzA); if(cudaStat != cudaSuccess) cout << "Error -- could not malloc space for dev_csrColIdxhhh_zf_GPU1. Status: " << cudaStat << endl;
	cudaStat = cudaMalloc(&dev_csrRowPtrhhh_zf_GPU1, sizeof(int) * (m+1)); if(cudaStat != cudaSuccess) cout << "Error -- could not malloc space for dev_csrRowPtrhhh_zf_GPU1. Status: " << cudaStat << endl;

	// ZF stuff on GPU2
	cudaSetDevice(device_GPU2);
	cudaStat = cudaMalloc(&dev_ZF_equalizers_GPU2, sizeof(cuComplex)*EQUALIZER_LENGTH*MAX_PACKETS_PER_MEMORY_SECTION); if(cudaStat != cudaSuccess) cout << "Error -- could not malloc space for dev_ZF_equalizers_GPU1 in ZF Equalizer initialization" << endl;
	cudaSetDevice(device_GPU1);

	float ZF_shs[MAX_PACKETS_PER_MEMORY_SECTION];
	for(int i = 0; i < MAX_PACKETS_PER_MEMORY_SECTION; i++)
		ZF_shs[i] = 0;
	cudaMemcpy(dev_shs_GPU1, ZF_shs, sizeof(float) * MAX_PACKETS_PER_MEMORY_SECTION, cudaMemcpyHostToDevice);

	// Read in files for Hard Coded CSR
	myFile.open("/home/adm85/git/JeffPaq/PAQfull/src/zfTestFileFromMatlabRp.txt");	if (!myFile.is_open()) printf("\n\n\t\tCould not open the file!\n\n");	else	{		int i = 0;		float output;		while (!myFile.eof()) {			myFile >> output;
	csrRowPtrhhh_switched[i++] = (int)output;																													}		myFile.close();	}
	cudaMemcpy(dev_csrRowPtrhhh_zf_GPU1, csrRowPtrhhh_switched, sizeof(int) * (m+1), cudaMemcpyHostToDevice);

	myFile.open("/home/adm85/git/JeffPaq/PAQfull/src/zfTestFileFromMatlabCi.txt");	if (!myFile.is_open()) printf("\n\n\t\tCould not open the file!\n\n");	else	{		int i = 0;		float output;		while (!myFile.eof()) {			myFile >> output;
	csrColIdxhhh_switched[i++] = (int)output;																													}		myFile.close();	}
	cudaMemcpy(dev_csrColIdxhhh_zf_GPU1, csrColIdxhhh_switched, sizeof(int) * nnzA, cudaMemcpyHostToDevice);

	// Create all handles and info that cuSolver needs for ZF and MMSE
	cusolverSpCreate(&cusolver_handle_GPU1);
	cusparseCreateMatDescr(&descrA_zf_GPU1);
	cusparseSetMatType(descrA_zf_GPU1, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrA_zf_GPU1, CUSPARSE_INDEX_BASE_ONE); // base-1
	cusolverSpCreateCsrqrInfo(&info_zf_GPU1);
	cusolverSpXcsrqrAnalysisBatched(cusolver_handle_GPU1, m, m, nnzA, descrA_zf_GPU1, dev_csrRowPtrhhh_zf_GPU1, dev_csrColIdxhhh_zf_GPU1, info_zf_GPU1);

	// Find memory needed to use cuSolver
	cusolverStatus_t stat = cusolverSpCcsrqrBufferInfoBatched(cusolver_handle_GPU1, m, m, nnzA, descrA_zf_GPU1, dev_hhh_csr_zf_GPU1, dev_csrRowPtrhhh_zf_GPU1, dev_csrColIdxhhh_zf_GPU1, batchSize, info_zf_GPU1, &size_internal_zf_GPU1, &size_qr_zf_GPU1);
	cudaStat = cudaMalloc((void**)&dev_buffer_qr_zf_GPU1, size_qr_zf_GPU1); if(cudaStat != cudaSuccess) cout << "Error -- could not malloc space for dev_buffer_qr_zf_GPU1 in MMSE Equalizer initialization" << endl;
	//	cusolverSpSetStream(cusolver_handle_GPU1, stream_GPU1_array[stream_0]);


	//----------------------------------------------
	//	MMSE EQUALIZER ON GPU0
	//----------------------------------------------
	cudaSetDevice(device_GPU0);
	//	cudaStat = cudaMalloc(&dev_rd_vec_GPU0, CHAN_EST_MATRIX_NUM_COLS * MAX_PACKETS_PER_MEMORY_SECTION*sizeof(cuComplex)); if(cudaStat	!= cudaSuccess)cout << "ERROR -- Could not allocate space for dev_rd_vec_GPU0 vector on GPU" << endl;
	cudaStat = cudaMalloc(&dev_noise_piX_GPU0, noiseMultiply_X_length * sizeof(cuComplex)); if(cudaStat != cudaSuccess) cout << "ERROR -- Could not allocate space for dev_noise_piX_GPU0 on GPU" << endl;


	cuComplex tempMat[noiseMultiply_X_length];
	for(uint idx=0; idx < xMatSamples.getSize(); idx++) {
		tempMat[idx].x = xMatSamples.getI().at(idx);
		tempMat[idx].y = xMatSamples.getQ().at(idx);
	}

	cublasStatus_t copyStatus = cublasSetMatrix(X_MATRIX_ROWS, X_MATRIX_COLS, sizeof(cuComplex), tempMat, X_MATRIX_ROWS, dev_noise_piX_GPU0, X_MATRIX_ROWS); if(copyStatus != CUBLAS_STATUS_SUCCESS) cout << "ERROR -- Could not copy dev_noise_piX_GPU0 to GPU" << endl;
	cudaStat = cudaMalloc(&dev_Xh_prod_GPU0, CHAN_EST_MATRIX_NUM_COLS * MAX_PACKETS_PER_MEMORY_SECTION*sizeof(cuComplex)); if(cudaStat != cudaSuccess)		cout << "ERROR -- Could not allocate space for dev_Xh_prod_GPU0 vector on GPU" << endl;
	cudaStat = cudaMalloc(&dev_h_hat_mmse_GPU0, CHAN_SIZE * MAX_PACKETS_PER_MEMORY_SECTION*sizeof(cuComplex)); if(cudaStat != cudaSuccess) cout << "Error -- could not malloc space for dev_h_hat_mmse_GPU0 in MMSE Equalizer initialization" << endl;
	cudaStat = cudaMalloc(&dev_gemm_alpha_GPU0, sizeof(cuComplex)); if(cudaStat != cudaSuccess)	cout << "ERROR -- Could not allocate gemm_alpha on GPU_GPU0" << endl;
	cudaStat = cudaMalloc(&dev_gemm_beta_GPU0, sizeof(cuComplex)); if(cudaStat != cudaSuccess)	cout << "ERROR -- Could not allocate gemm_beta on GPU_GPU0" << endl;

	//copy alpha and beta
	cuComplex alpha, beta;	alpha.x = 1;	alpha.y = 0;	beta.x = 0;	beta.y = 0;
	cudaMemcpy(dev_gemm_alpha_GPU0, &alpha, sizeof(cuComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_gemm_beta_GPU0,  &beta,  sizeof(cuComplex), cudaMemcpyHostToDevice);
	cudaStat = cudaMalloc(&dev_chanEstCorr_mmse_GPU0, sizeof(cuComplex)*(CHAN_SIZE*2-1)*MAX_PACKETS_PER_MEMORY_SECTION); if(cudaStat != cudaSuccess) cout << "Error -- could not malloc space for dev_chanEstCorr_mmse_GPU0 in MMSE Equalizer initialization" << endl;
	cudaStat = cudaMalloc(&dev_hhh_csr_mmse_GPU0, sizeof(cuComplex)*nnzA*MAX_PACKETS_PER_MEMORY_SECTION); if(cudaStat != cudaSuccess) cout << "Error -- could not malloc space for dev_hhh_csr_mmse_GPU0 in ZF Equalizer initialization" << endl;
	cudaStat = cudaMalloc(&dev_hh_un0_mmse_GPU0, sizeof(cuComplex)*EQUALIZER_LENGTH*MAX_PACKETS_PER_MEMORY_SECTION); if(cudaStat != cudaSuccess) cout << "Error -- could not malloc space for dev_hh_un0_mmse_GPU0 in ZF Equalizer initialization" << endl;
	cudaStat = cudaMalloc(&dev_csrColIdxhhh_mmse_GPU0, sizeof(int) * nnzA); if(cudaStat != cudaSuccess) cout << "Error -- could not malloc space for dev_csrColIdxhhh_mmse_GPU0. Status: " << cudaStat << endl;
	cudaStat = cudaMalloc(&dev_csrRowPtrhhh_mmse_GPU0, sizeof(int) * (m+1)); if(cudaStat != cudaSuccess) cout << "Error -- could not malloc space for dev_csrRowPtrhhh_mmse_GPU0. Status: " << cudaStat << endl;

	// Copy files for Hard Coded CSR
	myFile.open("/home/adm85/git/JeffPaq/PAQfull/src/zfTestFileFromMatlabRp.txt");	if (!myFile.is_open()) printf("\n\n\t\tCould not open the file!\n\n");	else	{		int i = 0;		float output;		while (!myFile.eof()) {			myFile >> output;
	csrRowPtrhhh[i++] = (int)output;																													}		myFile.close();	}
	cudaStat = cudaMemcpy(dev_csrRowPtrhhh_mmse_GPU0, csrRowPtrhhh, sizeof(int) * (m+1), cudaMemcpyHostToDevice); if(cudaStat != cudaSuccess) cout << "Error -- could not malloc space for dev_csrRowPtrhhh_mmse_GPU0 in MMSE Equalizer initialization" << endl;

	myFile.open("/home/adm85/git/JeffPaq/PAQfull/src/zfTestFileFromMatlabCi.txt");	if (!myFile.is_open()) printf("\n\n\t\tCould not open the file!\n\n");	else	{		int i = 0;		float output;		while (!myFile.eof()) {			myFile >> output;
	csrColIdxhhh[i++] = (int)output;																													}		myFile.close();	}
	cudaStat = cudaMemcpy(dev_csrColIdxhhh_mmse_GPU0, csrColIdxhhh, sizeof(int) * nnzA, cudaMemcpyHostToDevice); if(cudaStat != cudaSuccess) cout << "Error -- could not malloc space for dev_csrColIdxhhh_mmse_GPU0 in MMSE Equalizer initialization" << endl;

	cusolverStatus_t cusolverStat = cusolverSpCreate(&cusolver_handle_GPU0); if(cusolverStat != CUSOLVER_STATUS_SUCCESS) cout << "\t\t\tcusolverSpCreate cusolver_handle_GPU0 Failed" << endl;
	cusparseStatus_t cusparseStat = cusparseCreateMatDescr(&descrA_mmse_GPU0); if(cusparseStat != CUSPARSE_STATUS_SUCCESS) cout << "\t\t\tcusparseCreateMatDescr descrA_mmse_GPU0 Failed" << endl;
	cusparseStat = cusparseSetMatType(descrA_mmse_GPU0, CUSPARSE_MATRIX_TYPE_GENERAL); if(cusparseStat != CUSPARSE_STATUS_SUCCESS) cout << "\t\t\tcusparseSetMatType descrA_mmse_GPU0 Failed" << endl;
	cusparseStat = cusparseSetMatIndexBase(descrA_mmse_GPU0, CUSPARSE_INDEX_BASE_ONE); if(cusparseStat != CUSPARSE_STATUS_SUCCESS) cout << "\t\t\tcusparseSetMatIndexBase descrA_mmse_GPU0 Failed" << endl;
	cusolverStat = cusolverSpCreateCsrqrInfo(&info_mmse_GPU0); if(cusolverStat != CUSOLVER_STATUS_SUCCESS) cout << "\t\t\tcusolverSpCreateCsrqrInfo info_mmse_GPU0 Failed" << endl;
	cusolverStat = cusolverSpXcsrqrAnalysisBatched(cusolver_handle_GPU0, m, m, nnzA, descrA_mmse_GPU0, dev_csrRowPtrhhh_mmse_GPU0, dev_csrColIdxhhh_mmse_GPU0, info_mmse_GPU0); if(cusolverStat != CUSOLVER_STATUS_SUCCESS) cout << "\t\t\tcusolverSpXcsrqrAnalysisBatched cusolver_handle_GPU0 Failed" << endl;
	cusolverStat = cusolverSpCcsrqrBufferInfoBatched(cusolver_handle_GPU0, m, m, nnzA, descrA_mmse_GPU0, dev_hhh_csr_mmse_GPU0, dev_csrRowPtrhhh_mmse_GPU0, dev_csrColIdxhhh_mmse_GPU0, batchSize, info_mmse_GPU0, &size_internal_mmse_GPU0, &size_qr_mmse_GPU0); if(cusolverStat != CUSOLVER_STATUS_SUCCESS) cout << "\t\t\tcusolverSpCcsrqrBufferInfoBatched cusolver_handle_GPU0 Failed" << endl;
	cudaStat = cudaMalloc((void**)&dev_buffer_qr_mmse_GPU0, size_qr_mmse_GPU0); if(cudaStat != cudaSuccess) cout << "Error -- could not malloc space for dev_buffer_qr_mmse_GPU0 in MMSE Equalizer initialization" << endl;
	//	cusolverStat = cusolverSpSetStream(cusolver_handle_GPU0, stream_GPU0_array[stream_0]); if(cusolverStat != CUSOLVER_STATUS_SUCCESS) cout << "\t\t\tcusolverSpSetStream stream_GPU0_array[stream_0] Failed" << endl;
	cudaStat = cudaMalloc(&dev_MMSE_CMA_equalizers_GPU0, sizeof(cuComplex)*EQUALIZER_LENGTH*MAX_PACKETS_PER_MEMORY_SECTION); if(cudaStat != cudaSuccess) cout << "Error -- could not malloc space for dev_MMSE_CMA_equalizers_GPU0 in MMSE Equalizer initialization" << endl;


	numBlocks_autocorr = MAX_PACKETS_PER_MEMORY_SECTION;
	numThreads_autocorr = CHAN_SIZE;

	numBlocks_fill_hhh_csr_matrices = MAX_PACKETS_PER_MEMORY_SECTION;
	numThreads_fill_hhh_csr_matrices = EQUALIZER_LENGTH;

	maxThreads_build_hh_un0_vector_reworked = MAX_PACKETS_PER_MEMORY_SECTION;
	numThreads_build_hh_un0_vector_reworked = 32;
	numBlocks_build_hh_un0_vector_reworked = maxThreads_build_hh_un0_vector_reworked / numThreads_build_hh_un0_vector_reworked;
	if((maxThreads_build_hh_un0_vector_reworked % numThreads_build_hh_un0_vector_reworked) > 0) numBlocks_build_hh_un0_vector_reworked++;
}
void GPUHandler::free_equalizers_variables() {

	//----------------------------------------------
	//	ZERO-FORCING EQUALIZER ON GPU1
	//----------------------------------------------
	cudaFree(dev_h_hat_zf_GPU1);
	cudaFree(dev_hhh_csr_zf_GPU1);
	cudaFree(dev_hh_un0_zf_GPU1);
	cudaFree(dev_chanEstCorr_zf_GPU1);
	cudaFree(dev_ZF_equalizers_GPU1);
	cudaFree(dev_csrColIdxhhh_zf_GPU1);
	cudaFree(dev_csrRowPtrhhh_zf_GPU1);
	cudaFree(dev_buffer_qr_zf_GPU1);
	//	cusolverSpDestroy(cusolver_handle_GPU1);
	//	cusparseDestroyMatDescr(descrA_zf_GPU1);
	//	cusolverSpDestroyCsrqrInfo(info_zf_GPU1);

	//----------------------------------------------
	//	ZERO-FORCING EQUALIZER ON GPU2
	//----------------------------------------------
	cudaFree(dev_ZF_equalizers_GPU2);


	cudaSetDevice(device_GPU0);

	//----------------------------------------------
	//	MMSE EQUALIZER ON GPU0
	//----------------------------------------------
	cudaSetDevice(device_GPU0);
	//	cudaFree(dev_rd_vec_GPU0);
	cudaFree(dev_noise_piX_GPU0);
	cudaFree(dev_Xh_prod_GPU0);
	cudaFree(dev_h_hat_mmse_GPU0);
	cudaFree(dev_gemm_alpha_GPU0);
	cudaFree(dev_gemm_beta_GPU0);
	cudaFree(dev_chanEstCorr_mmse_GPU0);
	cudaFree(dev_hhh_csr_mmse_GPU0);
	cudaFree(dev_hh_un0_mmse_GPU0);
	cudaFree(dev_buffer_qr_mmse_GPU0);
}
void GPUHandler::calculate_equalizers() {
	cudaSetDevice(device_GPU1);	autocorr						<<<numBlocks_autocorr, 						numThreads_autocorr>>>						(dev_h_hat_zf_GPU1,   CHAN_SIZE, dev_chanEstCorr_zf_GPU1,   corrLength, corrLength/2, dev_shs_GPU1,           0);
	cudaSetDevice(device_GPU0);	autocorr						<<<numBlocks_autocorr, 						numThreads_autocorr>>>						(dev_h_hat_mmse_GPU0, CHAN_SIZE, dev_chanEstCorr_mmse_GPU0, corrLength, corrLength/2, dev_noiseVariance_GPU0, 1);
	cudaSetDevice(device_GPU1);	fill_hhh_csr_matrices			<<<numBlocks_fill_hhh_csr_matrices, 		numThreads_fill_hhh_csr_matrices>>>			(dev_chanEstCorr_zf_GPU1,   dev_hhh_csr_zf_GPU1,   N1, N2, nnzA);
	cudaSetDevice(device_GPU0);	fill_hhh_csr_matrices			<<<numBlocks_fill_hhh_csr_matrices, 		numThreads_fill_hhh_csr_matrices>>>			(dev_chanEstCorr_mmse_GPU0, dev_hhh_csr_mmse_GPU0, N1, N2, nnzA);
	cudaSetDevice(device_GPU1);	cudaMemset						(dev_hh_un0_zf_GPU1, 	0, EQUALIZER_LENGTH*MAX_PACKETS_PER_MEMORY_SECTION*sizeof(cuComplex));
	cudaSetDevice(device_GPU0);	cudaMemset						(dev_hh_un0_mmse_GPU0, 	0, EQUALIZER_LENGTH*MAX_PACKETS_PER_MEMORY_SECTION*sizeof(cuComplex));
	cudaSetDevice(device_GPU1);	build_hh_un0_vector_reworked	<<<numBlocks_build_hh_un0_vector_reworked, 	numThreads_build_hh_un0_vector_reworked>>>	(dev_h_hat_zf_GPU1,   dev_hh_un0_zf_GPU1,     N1, N2, L1, L2, maxThreads_build_hh_un0_vector_reworked);
	cudaSetDevice(device_GPU0);	build_hh_un0_vector_reworked	<<<numBlocks_build_hh_un0_vector_reworked, 	numThreads_build_hh_un0_vector_reworked>>>	(dev_h_hat_mmse_GPU0, dev_hh_un0_mmse_GPU0, N1, N2, L1, L2, maxThreads_build_hh_un0_vector_reworked);

	cudaSetDevice(device_GPU1);	cusolverSpCcsrqrsvBatched		(cusolver_handle_GPU1, m, m, nnzA, descrA_zf_GPU1, 	dev_hhh_csr_zf_GPU1, 	dev_csrRowPtrhhh_zf_GPU1, 	dev_csrColIdxhhh_zf_GPU1, 	dev_hh_un0_zf_GPU1, 	dev_ZF_equalizers_GPU1, 			batchSize, 	info_zf_GPU1, 	dev_buffer_qr_zf_GPU1);
	cudaSetDevice(device_GPU0);	cusolverSpCcsrqrsvBatched		(cusolver_handle_GPU0, m, m, nnzA, descrA_mmse_GPU0, 	dev_hhh_csr_mmse_GPU0, 	dev_csrRowPtrhhh_mmse_GPU0, dev_csrColIdxhhh_mmse_GPU0, dev_hh_un0_mmse_GPU0, 	dev_MMSE_CMA_equalizers_GPU0, 	batchSize, 	info_mmse_GPU0, dev_buffer_qr_mmse_GPU0);

	// Move ICMA equalizer to GPU2 for MMSE
	cudaSetDevice(device_GPU2);	cudaMemcpy(dev_MMSEequalizers_GPU2,	dev_MMSE_CMA_equalizers_GPU0,	sizeof(cuComplex)*EQUALIZER_LENGTH*MAX_PACKETS_PER_MEMORY_SECTION, cudaMemcpyDeviceToDevice);
}

void GPUHandler::initialize_CMA_variables(){
	cudaError_t cudaStat;
	CMAmu = 0.02;
	cudaSetDevice(device_GPU0);	cudaStat = cudaMalloc(&dev_z_GPU0, 					SAMPLES_PER_PACKET * MAX_PACKETS_PER_MEMORY_SECTION * sizeof(cuComplex));	if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_z_GPU0" << endl;
	cudaSetDevice(device_GPU0);	cudaStat = cudaMalloc(&dev_delJ_GPU0, 				SAMPLES_PER_PACKET * MAX_PACKETS_PER_MEMORY_SECTION * sizeof(cuComplex));	if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_delJ_GPU0" << endl;
	cudaSetDevice(device_GPU0);	cudaStat = cudaMalloc(&dev_z_flipped_GPU0, 			SAMPLES_PER_PACKET * MAX_PACKETS_PER_MEMORY_SECTION * sizeof(cuComplex));	if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_z_flipped_GPU0" << endl;
	cudaSetDevice(device_GPU0);	cudaStat = cudaMalloc(&dev_x_padded_GPU0, 			CMA_FFT_LENGTH * MAX_PACKETS_PER_MEMORY_SECTION * sizeof(cuComplex));		if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_x_padded_GPU0" << endl;
	cudaSetDevice(device_GPU0);	cudaStat = cudaMalloc(&dev_z_flipped_padded_GPU0, 	CMA_FFT_LENGTH * MAX_PACKETS_PER_MEMORY_SECTION * sizeof(cuComplex));		if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_z_flipped_padded_GPU0" << endl;
	cudaSetDevice(device_GPU0);	cudaStat = cudaMalloc(&dev_delJ_fft_GPU0, 			EQUALIZER_LENGTH * MAX_PACKETS_PER_MEMORY_SECTION * sizeof(cuComplex));		if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_delJ_fft_GPU0" << endl;
	cudaSetDevice(device_GPU0); cudaStat = cudaMalloc(&dev_CMA_bits_GPU0, 			BITS_PER_DATA_PACKET*MAX_PACKETS_PER_MEMORY_SECTION*sizeof(unsigned char));	if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_CMA_bits_GPU0" << endl;
	cudaSetDevice(device_GPU0);	cudaStat = cudaMalloc(&dev_y_GPU0, 					TOTAL_SAMPLES_LENGTH * sizeof(cuComplex));									if(cudaStat != cudaSuccess)	cout << "ERROR - Could not malloc dev_y_GPU0" << endl;
	cudaSetDevice(device_GPU0); cufftPlan1d(&fftPlan_CMA_GPU0, CMA_FFT_LENGTH, CUFFT_C2C, MAX_PACKETS_PER_MEMORY_SECTION);

	maxThreads_zeroPad_CMA_equalizers = EQUALIZER_LENGTH*MAX_PACKETS_PER_MEMORY_SECTION;
	numThreads_zeroPad_CMA_equalizers = 512;
	numBlocks_zeroPad_CMA_equalizers = maxThreads_zeroPad_CMA_equalizers / numThreads_zeroPad_CMA_equalizers;
	if((maxThreads_zeroPad_CMA_equalizers % numThreads_zeroPad_CMA_equalizers) > 0) numBlocks_zeroPad_CMA_equalizers++;

	maxThreads_zeroPad_CMA_samples = SAMPLES_PER_PACKET*MAX_PACKETS_PER_MEMORY_SECTION;
	numThreads_zeroPad_CMA_samples = 144;
	numBlocks_zeroPad_CMA_samples = maxThreads_zeroPad_CMA_samples/numThreads_zeroPad_CMA_samples;
	if((maxThreads_zeroPad_CMA_samples % numThreads_zeroPad_CMA_samples) > 0) numBlocks_zeroPad_CMA_samples++;


	maxThreads_pointMultiply_CMA = conv_length*MAX_PACKETS_PER_MEMORY_SECTION;
	numThreads_pointMultiply_CMA = 112;
	numBlocks_pointMultiply_CMA = maxThreads_pointMultiply_CMA/numThreads_pointMultiply_CMA;
	if((maxThreads_pointMultiply_CMA % numThreads_pointMultiply_CMA) > 0) numBlocks_pointMultiply_CMA++;

	maxThreads_pointMultiply_CMA = conv_length*MAX_PACKETS_PER_MEMORY_SECTION;
	numThreads_pointMultiply_CMA = 112;
	numBlocks_pointMultiply_CMA = maxThreads_pointMultiply_CMA/numThreads_pointMultiply_CMA;
	if((maxThreads_pointMultiply_CMA % numThreads_pointMultiply_CMA) > 0) numBlocks_pointMultiply_CMA++;

	maxThreads_scaleAndPruneFFT_CMA = SAMPLES_PER_PACKET*MAX_PACKETS_PER_MEMORY_SECTION;
	numThreads_scaleAndPruneFFT_CMA = 128;
	numBlocks_scaleAndPruneFFT_CMA = maxThreads_scaleAndPruneFFT_CMA/numThreads_scaleAndPruneFFT_CMA;
	if((maxThreads_scaleAndPruneFFT_CMA % numThreads_scaleAndPruneFFT_CMA) > 0) numBlocks_scaleAndPruneFFT_CMA++;

	maxThreads_cudaCMAz_CMA = SAMPLES_PER_PACKET*MAX_PACKETS_PER_MEMORY_SECTION;
	numThreads_cudaCMAz_CMA = 512;
	numBlocks_cudaCMAz_CMA = maxThreads_cudaCMAz_CMA/numThreads_cudaCMAz_CMA;
	if((maxThreads_cudaCMAz_CMA % numThreads_cudaCMAz_CMA) > 0) numBlocks_cudaCMAz_CMA++;

	maxThreads_cudaCMAdelJ_CMA = EQUALIZER_LENGTH*MAX_PACKETS_PER_MEMORY_SECTION;
	numThreads_cudaCMAdelJ_CMA = 37;
	numBlocks_cudaCMAdelJ_CMA = maxThreads_cudaCMAdelJ_CMA/numThreads_cudaCMAdelJ_CMA;
	if((maxThreads_cudaCMAdelJ_CMA % numThreads_cudaCMAdelJ_CMA) > 0) numBlocks_cudaCMAdelJ_CMA++;

	maxThreads_cudaCMAflipLR_CMA = SAMPLES_PER_PACKET*MAX_PACKETS_PER_MEMORY_SECTION;
	numThreads_cudaCMAflipLR_CMA = 272;
	numBlocks_cudaCMAflipLR_CMA = maxThreads_cudaCMAflipLR_CMA/numThreads_cudaCMAflipLR_CMA;
	if((maxThreads_cudaCMAflipLR_CMA % numThreads_cudaCMAflipLR_CMA) > 0) numBlocks_cudaCMAflipLR_CMA++;

	maxThreads_zeroPad_CMA_z_flipped = SAMPLES_PER_PACKET*MAX_PACKETS_PER_MEMORY_SECTION;
	numThreads_zeroPad_CMA_z_flipped = 112;
	numBlocks_zeroPad_CMA_z_flipped = maxThreads_zeroPad_CMA_z_flipped/numThreads_zeroPad_CMA_z_flipped;
	if((maxThreads_zeroPad_CMA_z_flipped % numThreads_zeroPad_CMA_z_flipped) > 0) numBlocks_zeroPad_CMA_z_flipped++;

	maxThreads_zeroPadConj_CMA = SAMPLES_PER_PACKET*MAX_PACKETS_PER_MEMORY_SECTION;
	numThreads_zeroPadConj_CMA = 224;
	numBlocks_zeroPadConj_CMA = maxThreads_zeroPadConj_CMA/numThreads_zeroPadConj_CMA;
	if((maxThreads_zeroPadConj_CMA % numThreads_zeroPadConj_CMA) > 0) numBlocks_zeroPadConj_CMA++;

	maxThreads_pointMultiply_CMA_z_flipped = CMA_FFT_LENGTH*MAX_PACKETS_PER_MEMORY_SECTION;
	numThreads_pointMultiply_CMA_z_flipped = 112;
	numBlocks_pointMultiply_CMA_z_flipped = maxThreads_pointMultiply_CMA_z_flipped/numThreads_pointMultiply_CMA_z_flipped;
	if((maxThreads_pointMultiply_CMA_z_flipped % numThreads_pointMultiply_CMA_z_flipped) > 0) numBlocks_pointMultiply_CMA_z_flipped++;

	maxThreads_stripAndScale_CMA = EQUALIZER_LENGTH*MAX_PACKETS_PER_MEMORY_SECTION;
	numThreads_stripAndScale_CMA = 127;
	numBlocks_stripAndScale_CMA = maxThreads_stripAndScale_CMA/numThreads_stripAndScale_CMA;
	if((maxThreads_stripAndScale_CMA % numThreads_stripAndScale_CMA) > 0) numBlocks_stripAndScale_CMA++;

	maxThreads_cudaUpdateCoefficients_CMA = EQUALIZER_LENGTH*MAX_PACKETS_PER_MEMORY_SECTION;
	numThreads_cudaUpdateCoefficients_CMA = 96;
	numBlocks_cudaUpdateCoefficients_CMA = maxThreads_cudaUpdateCoefficients_CMA/numThreads_cudaUpdateCoefficients_CMA;
	if((maxThreads_cudaUpdateCoefficients_CMA % numThreads_cudaUpdateCoefficients_CMA) > 0) numBlocks_cudaUpdateCoefficients_CMA++;
}
void GPUHandler::free_CMA_variables(){
	cudaFree(dev_z_GPU0);
	cudaFree(dev_y_GPU0);
	cudaFree(dev_delJ_GPU0);
	cudaFree(dev_z_flipped_GPU0);
	cudaFree(dev_x_padded_GPU0);
	cudaFree(dev_z_flipped_padded_GPU0);
	cudaFree(dev_delJ_fft_GPU0);
	cudaFree(dev_CMA_bits_GPU0);
	cufftDestroy(fftPlan_CMA_GPU0);
}
void GPUHandler::changeCMAmu(float in) {
	CMAmu = in;
}
void GPUHandler::CMA() {
	cudaSetDevice(device_GPU0);	cudaMemset(dev_equalizers_padded_GPU0,0,sizeof(cufftComplex)*conv_length*MAX_PACKETS_PER_MEMORY_SECTION);
	cudaSetDevice(device_GPU0);	cudaMemset(dev_Samples_padded_GPU0,   0,sizeof(cufftComplex)*conv_length*MAX_PACKETS_PER_MEMORY_SECTION);
	cudaSetDevice(device_GPU0);	zeroPad<<<numBlocks_zeroPad_CMA_equalizers,numThreads_zeroPad_CMA_equalizers>>>(dev_equalizers_padded_GPU0, dev_MMSE_CMA_equalizers_GPU0, 	MAX_PACKETS_PER_MEMORY_SECTION, EQUALIZER_LENGTH, conv_length);
	cudaSetDevice(device_GPU0);	zeroPad<<<numBlocks_zeroPad_CMA_samples,numThreads_zeroPad_CMA_samples>>>(dev_Samples_padded_GPU0,    dev_Samples_GPU0, 	MAX_PACKETS_PER_MEMORY_SECTION, SAMPLES_PER_PACKET, conv_length);
	cudaSetDevice(device_GPU0);	cufftExecC2C(fftPlan_apply_GPU0, dev_equalizers_padded_GPU0, dev_equalizers_padded_GPU0,	CUFFT_FORWARD);
	cudaSetDevice(device_GPU0);	cufftExecC2C(fftPlan_apply_GPU0, dev_Samples_padded_GPU0, 	 dev_Samples_padded_GPU0, 		CUFFT_FORWARD);
	cudaSetDevice(device_GPU0);	pointMultiply<<<numBlocks_pointMultiply_CMA, numThreads_pointMultiply_CMA>>>(dev_Samples_padded_GPU0, dev_equalizers_padded_GPU0, maxThreads_pointMultiply_CMA); // dest,src,length
	cudaSetDevice(device_GPU0);	cufftExecC2C(fftPlan_apply_GPU0, dev_Samples_padded_GPU0, dev_Samples_padded_GPU0, CUFFT_INVERSE);
	cudaSetDevice(device_GPU0);	scaleAndPruneFFT<<<numBlocks_scaleAndPruneFFT_CMA, numThreads_scaleAndPruneFFT_CMA>>>(dev_y_GPU0, dev_Samples_padded_GPU0,(float)conv_length, SAMPLES_PER_PACKET, conv_length, L1,	maxThreads_scaleAndPruneFFT_CMA); // dest,src,length
	cudaSetDevice(device_GPU0);	cudaCMAz<<<numBlocks_cudaCMAz_CMA, numThreads_cudaCMAz_CMA>>>(dev_y_GPU0, dev_z_GPU0, maxThreads_cudaCMAz_CMA); // 4 ms
	//	//	//Direct Computation
	//	cudaSetDevice(device_GPU0);	cudaCMAdelJ<<<numBlocks_cudaCMAdelJ_CMA,numThreads_cudaCMAdelJ_CMA>>>(dev_delJ_GPU0, dev_z_GPU0, dev_y_GPU0, SAMPLES_PER_PACKET, L1, CMAmu, maxThreads_cudaCMAdelJ_CMA); // 555.59 ms
	cudaSetDevice(device_GPU0);	cudaCMAflipLR<<<numBlocks_cudaCMAflipLR_CMA,numThreads_cudaCMAflipLR_CMA>>>(dev_z_flipped_GPU0, dev_z_GPU0, SAMPLES_PER_PACKET, maxThreads_cudaCMAflipLR_CMA);
	cudaSetDevice(device_GPU0);	cudaMemset(dev_z_flipped_padded_GPU0,	0,CMA_FFT_LENGTH * MAX_PACKETS_PER_MEMORY_SECTION * sizeof(cuComplex));
	cudaSetDevice(device_GPU0);	cudaMemset(dev_x_padded_GPU0,			0,CMA_FFT_LENGTH * MAX_PACKETS_PER_MEMORY_SECTION * sizeof(cuComplex));
	cudaSetDevice(device_GPU0);	zeroPad<<<numBlocks_zeroPad_CMA_z_flipped,numThreads_zeroPad_CMA_z_flipped>>>(dev_z_flipped_padded_GPU0, 	dev_z_flipped_GPU0,	MAX_PACKETS_PER_MEMORY_SECTION, SAMPLES_PER_PACKET, CMA_FFT_LENGTH);
	cudaSetDevice(device_GPU0);	zeroPadConj<<<numBlocks_zeroPadConj_CMA,numThreads_zeroPadConj_CMA>>>(dev_x_padded_GPU0, 		dev_Samples_GPU0, 		MAX_PACKETS_PER_MEMORY_SECTION, SAMPLES_PER_PACKET, CMA_FFT_LENGTH);
	cudaSetDevice(device_GPU0); cufftExecC2C(fftPlan_CMA_GPU0, dev_z_flipped_padded_GPU0, 	dev_z_flipped_padded_GPU0, 	CUFFT_FORWARD);
	cudaSetDevice(device_GPU0); cufftExecC2C(fftPlan_CMA_GPU0, dev_x_padded_GPU0, 			dev_x_padded_GPU0, 			CUFFT_FORWARD);
	cudaSetDevice(device_GPU0);	pointMultiply<<<numBlocks_pointMultiply_CMA_z_flipped,numThreads_pointMultiply_CMA_z_flipped>>>(dev_z_flipped_padded_GPU0, dev_x_padded_GPU0, 	maxThreads_pointMultiply_CMA_z_flipped);
	cudaSetDevice(device_GPU0); cufftExecC2C(fftPlan_CMA_GPU0, dev_z_flipped_padded_GPU0, 	dev_z_flipped_padded_GPU0, 	CUFFT_INVERSE);
	cudaSetDevice(device_GPU0);	stripAndScale<<<numBlocks_stripAndScale_CMA,numThreads_stripAndScale_CMA>>>(dev_delJ_fft_GPU0, dev_z_flipped_padded_GPU0, CMAmu, SAMPLES_PER_PACKET-L2-1, EQUALIZER_LENGTH, CMA_FFT_LENGTH, maxThreads_stripAndScale_CMA);
	cudaSetDevice(device_GPU0);	cudaUpdateCoefficients<<<numBlocks_cudaUpdateCoefficients_CMA,numThreads_cudaUpdateCoefficients_CMA>>>(dev_MMSE_CMA_equalizers_GPU0, dev_delJ_fft_GPU0, maxThreads_cudaUpdateCoefficients_CMA);
}

void GPUHandler::initialize_freq_variables(){
	cudaError_t cudaStat;

	shiftFDE = PREAMBLE_LENGTH_IN_SAMPLES/2;

	// Pointers used for Both FDE1 and FDE2
	cudaSetDevice(device_GPU2);	cudaStat = cudaMalloc((void**)&dev_h_hat_freq_GPU2, 					sizeof(cufftComplex)*CHAN_SIZE * MAX_PACKETS_PER_MEMORY_SECTION); 		if(cudaStat != cudaSuccess) cout << "Error -- could not malloc space for dev_h_hat_zf in ZF Equalizer initialization" << endl;
	cudaSetDevice(device_GPU2);	cudaStat = cudaMalloc((void**)&dev_FDE_Y_padded_GPU2, 					sizeof(cufftComplex)*conv_length*MAX_PACKETS_PER_MEMORY_SECTION); 		if(cudaStat != cudaSuccess) cout << "ERROR -- Could not allocate space for dev_FDE_Y_padded_GPU2" << endl;
	//cudaSetDevice(device_GPU2);	cudaStat = cudaMalloc((void**)&dev_FDE_Y_padded_GPU2_blindShift,		sizeof(cufftComplex)*conv_length*MAX_PACKETS_PER_MEMORY_SECTION); 		if(cudaStat != cudaSuccess) cout << "ERROR -- Could not allocate space for dev_FDE_Y_padded_GPU2_blindShift" << endl;
	cudaSetDevice(device_GPU2);	cudaStat = cudaMalloc((void**)&dev_FDE_PSI_GPU2, 						sizeof(cufftComplex)*conv_length*MAX_PACKETS_PER_MEMORY_SECTION); 		if(cudaStat != cudaSuccess) cout << "ERROR -- Could not allocate space for dev_FDE_PSI_GPU2" << endl;
	cudaSetDevice(device_GPU2);	cudaStat = cudaMalloc((void**)&dev_FDE_H_padded_GPU2, 					sizeof(cufftComplex)*conv_length*MAX_PACKETS_PER_MEMORY_SECTION); 		if(cudaStat != cudaSuccess) cout << "ERROR -- Could not allocate space for dev_FDE_H_padded_GPU2" << endl;
	cudaSetDevice(device_GPU2);	cudaStat = cudaMalloc((void**)&dev_shs_GPU2, 							sizeof(float)); 														if(cudaStat != cudaSuccess) cout << "ERROR -- Could not allocate space for dev_shs_GPU2" << endl;
	cudaSetDevice(device_GPU2);	cudaStat = cudaMalloc((void**)&dev_FDE_detected_downsampled_GPU2, 		sizeof(cufftComplex)*BITS_PER_PACKET*MAX_PACKETS_PER_MEMORY_SECTION); 	if(cudaStat != cudaSuccess) cout << "ERROR -- Could not allocate space for dev_FDE_detected_downsampled_GPU2" << endl;
	// Pointers used for FDE1
	cudaSetDevice(device_GPU2);	cudaStat = cudaMalloc((void**)&dev_FDE1_detected_downsampled_rotated_GPU2, 		sizeof(cufftComplex)*BITS_PER_PACKET*MAX_PACKETS_PER_MEMORY_SECTION); 	if(cudaStat != cudaSuccess) cout << "ERROR -- Could not allocate space for dev_FDE1_detected_downsampled_rotated_GPU2" << endl;
	cudaSetDevice(device_GPU2);	cudaStat = cudaMalloc((void**)&dev_FDE1_ahat_GPU2, sizeof(float)*BITS_PER_PACKET*MAX_PACKETS_PER_MEMORY_SECTION);								if(cudaStat != cudaSuccess) cout << "ERROR -- Could not allocate space for dev_FDE1_ahat_GPU2" << endl;
	cudaSetDevice(device_GPU2);	cudaStat = cudaMalloc((void**)&dev_FDE1_bits_GPU2, sizeof(unsigned char)*BITS_PER_DATA_PACKET*MAX_PACKETS_PER_MEMORY_SECTION);					if(cudaStat != cudaSuccess) cout << "ERROR -- Could not allocate space for dev_FDE1_bits_GPU2" << endl;
	cudaSetDevice(device_GPU0);	cudaStat = cudaMalloc((void**)&dev_FDE1_bits_GPU0, sizeof(unsigned char)*BITS_PER_DATA_PACKET*MAX_PACKETS_PER_MEMORY_SECTION);					if(cudaStat != cudaSuccess) cout << "ERROR -- Could not allocate space for dev_FDE1_bits_GPU0" << endl;
	// Pointers used for FDE1
	cudaSetDevice(device_GPU2);	cudaStat = cudaMalloc((void**)&dev_FDE2_detected_downsampled_rotated_GPU2, 		sizeof(cufftComplex)*BITS_PER_PACKET*MAX_PACKETS_PER_MEMORY_SECTION); 	if(cudaStat != cudaSuccess) cout << "ERROR -- Could not allocate space for dev_FDE1_detected_downsampled_rotated_GPU2" << endl;
	cudaSetDevice(device_GPU2);	cudaStat = cudaMalloc((void**)&dev_FDE2_ahat_GPU2, sizeof(float)*BITS_PER_PACKET*MAX_PACKETS_PER_MEMORY_SECTION);								if(cudaStat != cudaSuccess) cout << "ERROR -- Could not allocate space for dev_FDE2_ahat_GPU2" << endl;
	cudaSetDevice(device_GPU2);	cudaStat = cudaMalloc((void**)&dev_FDE2_bits_GPU2, sizeof(unsigned char)*BITS_PER_DATA_PACKET*MAX_PACKETS_PER_MEMORY_SECTION);					if(cudaStat != cudaSuccess) cout << "ERROR -- Could not allocate space for dev_FDE2_bits_GPU2" << endl;
	cudaSetDevice(device_GPU0);	cudaStat = cudaMalloc((void**)&dev_FDE2_bits_GPU0, sizeof(unsigned char)*BITS_PER_DATA_PACKET*MAX_PACKETS_PER_MEMORY_SECTION);					if(cudaStat != cudaSuccess) cout << "ERROR -- Could not allocate space for dev_FDE2_bits_GPU0" << endl;

	float PSI_r[conv_length*10];

	ifstream myFile;	myFile.open("/home/adm85/git/JeffPaq/PAQfull/src/FreqTestFileFromMatlabPSI_r.txt");	if (!myFile.is_open())		printf("\n\n\t\tCould not open the file!\n\n");	else	{		int i = 0;		float output;		while (!myFile.eof()) {			myFile >> output;			PSI_r[i] = (float)output;			i++;		}		myFile.close();	}

	// For some reason the first index gets read on wrong...So I hard coded it
	PSI_r[0] = 0.7748754129126396161808543183724395930767059326171875000000000000000000000000000000000000000000000000;

	cudaSetDevice(device_GPU2); cudaMemcpy(dev_FDE_PSI_GPU2, PSI_r, conv_length * sizeof(float), cudaMemcpyHostToDevice); int maxThreads = conv_length*MAX_PACKETS_PER_MEMORY_SECTION; int numThreads = 128; int numBlocks = maxThreads/numThreads; if((maxThreads % numThreads) > 0) numBlocks++;
	cudaSetDevice(device_GPU2);	PSIfill<<<numBlocks,numThreads>>>(dev_FDE_PSI_GPU2, conv_length, maxThreads);
	cudaSetDevice(device_GPU2); cudaMemcpy(PSI_r ,dev_FDE_PSI_GPU2, conv_length*10 * sizeof(float), cudaMemcpyDeviceToHost);

	cudaSetDevice(device_GPU0);
}
void GPUHandler::free_freq_variables(){
	cudaFree(dev_h_hat_freq_GPU2);
	cudaFree(dev_FDE_Y_padded_GPU2);
	cudaFree(dev_FDE_Y_padded_GPU2_blindShift);
	cudaFree(dev_FDE_PSI_GPU2);
	cudaFree(dev_FDE_H_padded_GPU2);
	cudaFree(dev_FDE_detected_downsampled_GPU2);
	cudaFree(dev_FDE1_detected_downsampled_rotated_GPU2);
	cudaFree(dev_FDE2_detected_downsampled_rotated_GPU2);
	cudaFree(dev_FDE1_ahat_GPU2);
	cudaFree(dev_FDE2_ahat_GPU2);
	cudaFree(dev_FDE1_bits_GPU2);
	cudaFree(dev_FDE2_bits_GPU2);
	cudaFree(dev_FDE1_bits_GPU0);
	cudaFree(dev_FDE2_bits_GPU0);
}

void GPUHandler::initialize_apply_equalizers_and_detection_filters(){
	filter_fft = new cufftComplex[conv_length];
	cudaError_t cudaStat;
	cudaStat = cudaMalloc((void**)&dev_Samples_padded_GPU0, 		sizeof(cufftComplex)*conv_length*MAX_PACKETS_PER_MEMORY_SECTION); if(cudaStat != cudaSuccess) cout << "ERROR -- Could not allocate space for dev_Samples_padded_GPU0" << endl;
	cudaStat = cudaMalloc((void**)&dev_equalizers_padded_GPU0, 	sizeof(cufftComplex)*conv_length*MAX_PACKETS_PER_MEMORY_SECTION); if(cudaStat != cudaSuccess) cout << "ERROR -- Could not allocate space for dev_equalizers_padded_GPU0" << endl;

	cudaSetDevice(device_GPU1);
	//cudaStat = cudaMalloc((void**)&dev_Samples_padded_GPU1, sizeof(cufftComplex)*conv_length*MAX_PACKETS_PER_MEMORY_SECTION); if(cudaStat != cudaSuccess) cout << "ERROR -- Could not allocate space for dev_Samples_padded_GPU1" << endl;
	//cudaStat = cudaMalloc((void**)&dev_equalizers_padded_GPU1, sizeof(cufftComplex)*conv_length*MAX_PACKETS_PER_MEMORY_SECTION); if(cudaStat != cudaSuccess) cout << "ERROR -- Could not allocate space for dev_equalizers_padded_GPU1" << endl;
	cudaSetDevice(device_GPU0);

	cublasStatus_t stat;
	stat = cublasCreate_v2(&cublas_handle_apply_GPU0); if (stat != CUBLAS_STATUS_SUCCESS) printf ("CUBLAS initialization failed\n");
	//	stat = cublasSetStream_v2(cublas_handle_apply_GPU0,stream_GPU0_array[stream_0]); if (stat != CUBLAS_STATUS_SUCCESS) printf ("CUBLAS Stream failed\n");

	cudaSetDevice(device_GPU1);
	stat = cublasCreate_v2(&cublas_handle_apply_GPU1); if (stat != CUBLAS_STATUS_SUCCESS) printf ("CUBLAS initialization failed\n");
	//	stat = cublasSetStream_v2(cublas_handle_apply_GPU1,stream_GPU1_array[stream_0]); if (stat != CUBLAS_STATUS_SUCCESS) printf ("CUBLAS Stream failed\n");
	cudaSetDevice(device_GPU0);

	//	// Demod Stuff that blows up if it is in initialize_detection_filters_variables
	cudaSetDevice(device_GPU0);
	cudaStat = cudaMalloc((void**)&dev_signal_preDetection_GPU0, sizeof(cufftComplex)*conv_length*MAX_PACKETS_PER_MEMORY_SECTION); if(cudaStat != cudaSuccess) cout << "ERROR -- Could not allocate space for dev_signal_preDetection_GPU0" << endl;
	cudaStat = cudaMalloc((void**)&dev_filter_preDetection_GPU0, sizeof(cufftComplex)*conv_length*MAX_PACKETS_PER_MEMORY_SECTION); if(cudaStat != cudaSuccess) cout << "ERROR -- Could not allocate space for dev_filter_preDetection_GPU0" << endl;
	cudaStat = cudaMalloc((void**)&dev_detected_downsampled_GPU0, sizeof(cufftComplex)*BITS_PER_PACKET*MAX_PACKETS_PER_MEMORY_SECTION); if(cudaStat != cudaSuccess) cout << "ERROR -- Could not allocate space for dev_detected_downsampled_GPU0" << endl;
	// Init Filter Values
	float filter[demod_filter_length] = {0.003360611078123, 0.018859014445772, 0.020557325489618, -0.005891608820645, -0.056510808600904, -0.111672986765101, -0.088910244763529, 0.018850571483028, 0.356640902388316, 0.721842825302313, 0.808703717170084, 0.721842825302312, 0.356640902388316, 0.018850571483028, -0.088910244763529, -0.111672986765101, -0.056510808600904, -0.005891608820645, 0.020557325489618, 0.018859014445772, 0.003360611078123};
	cufftComplex* filterComplex = new cufftComplex[conv_length*MAX_PACKETS_PER_MEMORY_SECTION];
	for(int packet = 0; packet < MAX_PACKETS_PER_MEMORY_SECTION; packet++)		for(int i = 0; i < conv_length; i++){			int packetJump = packet*conv_length;			if(i < demod_filter_length)				filterComplex[packetJump+i].x = filter[i];			else				filterComplex[packetJump+i].x = 0;			filterComplex[packetJump+i].y = 0;		}

	cudaSetDevice(device_GPU0); cudaMemcpy(dev_filter_preDetection_GPU0, filterComplex, sizeof(cufftComplex)*conv_length*MAX_PACKETS_PER_MEMORY_SECTION,cudaMemcpyHostToDevice);
	cudaSetDevice(device_GPU0); cufftPlan1d(&fftPlan_detection_GPU0, conv_length, CUFFT_C2C, MAX_PACKETS_PER_MEMORY_SECTION);
	//	cudaSetDevice(device_GPU0); cufftSetStream(fftPlan_detection_GPU0,stream_GPU0_array[stream_0]);
	cudaSetDevice(device_GPU0); cufftExecC2C(fftPlan_detection_GPU0, dev_filter_preDetection_GPU0, dev_filter_preDetection_GPU0, CUFFT_FORWARD);

	cudaSetDevice(device_GPU1);
	//	cudaStat = cudaMalloc((void**)&dev_signal_preDetection_GPU1, sizeof(cufftComplex)*conv_length*MAX_PACKETS_PER_MEMORY_SECTION); if(cudaStat != cudaSuccess) cout << "ERROR -- Could not allocate space for dev_Samples_padded_GPU1" << endl;
	//	cudaStat = cudaMalloc((void**)&dev_filter_preDetection_GPU1, sizeof(cufftComplex)*conv_length*MAX_PACKETS_PER_MEMORY_SECTION); if(cudaStat != cudaSuccess) cout << "ERROR -- Could not allocate space for dev_filter_preDetection_GPU1" << endl;
	cudaStat = cudaMalloc((void**)&dev_detected_downsampled_GPU1, sizeof(cufftComplex)*BITS_PER_PACKET*MAX_PACKETS_PER_MEMORY_SECTION); if(cudaStat != cudaSuccess) cout << "ERROR -- Could not allocate space for dev_detected_downsampled_GPU1" << endl;
	cudaSetDevice(device_GPU0);
	// Init Filter Values
	float filter_GPU1[demod_filter_length] = {0.003360611078123, 0.018859014445772, 0.020557325489618, -0.005891608820645, -0.056510808600904, -0.111672986765101, -0.088910244763529, 0.018850571483028, 0.356640902388316, 0.721842825302313, 0.808703717170084, 0.721842825302312, 0.356640902388316, 0.018850571483028, -0.088910244763529, -0.111672986765101, -0.056510808600904, -0.005891608820645, 0.020557325489618, 0.018859014445772, 0.003360611078123};
	cufftComplex* filter_GPU1Complex = new cufftComplex[conv_length*MAX_PACKETS_PER_MEMORY_SECTION];
	for(int packet = 0; packet < MAX_PACKETS_PER_MEMORY_SECTION; packet++)
		for(int i = 0; i < conv_length; i++){
			int packetJump = packet*conv_length;
			if(i < demod_filter_length)
				filter_GPU1Complex[packetJump+i].x = filter_GPU1[i];
			else
				filter_GPU1Complex[packetJump+i].x = 0;
			filter_GPU1Complex[packetJump+i].y = 0;
		}
	//cudaSetDevice(device_GPU1); cudaMemcpy(dev_filter_preDetection_GPU1, filter_GPU1Complex, sizeof(cufftComplex)*conv_length*MAX_PACKETS_PER_MEMORY_SECTION,cudaMemcpyHostToDevice);
	cudaSetDevice(device_GPU1); cufftPlan1d(&fftPlan_detection_GPU1, conv_length, CUFFT_C2C, MAX_PACKETS_PER_MEMORY_SECTION);
	//	cudaSetDevice(device_GPU1); cufftSetStream(fftPlan_detection_GPU1,stream_GPU1_array[stream_0]);
	//cudaSetDevice(device_GPU1); cufftExecC2C(fftPlan_detection_GPU1, dev_filter_preDetection_GPU1, dev_filter_preDetection_GPU1, CUFFT_FORWARD);
	cudaSetDevice(device_GPU0);

	//MMSE STUFF
	cudaSetDevice(device_GPU2); cudaStat = cudaMalloc(&dev_MMSEequalizers_GPU2, sizeof(cuComplex)*EQUALIZER_LENGTH*MAX_PACKETS_PER_MEMORY_SECTION); if(cudaStat != cudaSuccess) cout << "Error -- could not malloc space for dev_MMSEequalizers_GPU2 in MMSE Equalizer initialization" << endl;
	cudaSetDevice(device_GPU2); cudaStat = cudaMalloc((void**)&dev_filter_preDetection_GPU2, sizeof(cufftComplex)*conv_length*MAX_PACKETS_PER_MEMORY_SECTION); if(cudaStat != cudaSuccess) cout << "ERROR -- Could not allocate space for dev_filter_preDetection_GPU2" << endl;
	float filter_GPU2[demod_filter_length] = {0.003360611078123, 0.018859014445772, 0.020557325489618, -0.005891608820645, -0.056510808600904, -0.111672986765101, -0.088910244763529, 0.018850571483028, 0.356640902388316, 0.721842825302313, 0.808703717170084, 0.721842825302312, 0.356640902388316, 0.018850571483028, -0.088910244763529, -0.111672986765101, -0.056510808600904, -0.005891608820645, 0.020557325489618, 0.018859014445772, 0.003360611078123};
	cufftComplex* filter_GPU2Complex = new cufftComplex[conv_length*MAX_PACKETS_PER_MEMORY_SECTION];
	for(int packet = 0; packet < MAX_PACKETS_PER_MEMORY_SECTION; packet++)		for(int i = 0; i < conv_length; i++){			int packetJump = packet*conv_length;			if(i < demod_filter_length)				filter_GPU2Complex[packetJump+i].x = filter_GPU2[i];			else				filter_GPU2Complex[packetJump+i].x = 0;			filter_GPU2Complex[packetJump+i].y = 0;		}
	cudaSetDevice(device_GPU2); cudaMemcpy(dev_filter_preDetection_GPU2, filter_GPU2Complex, sizeof(cufftComplex)*conv_length*MAX_PACKETS_PER_MEMORY_SECTION,cudaMemcpyHostToDevice);
	cudaSetDevice(device_GPU2); cudaStat = cudaMalloc((void**)&dev_equalizers_padded_GPU2, sizeof(cufftComplex)*conv_length*MAX_PACKETS_PER_MEMORY_SECTION); if(cudaStat != cudaSuccess) cout << "ERROR -- Could not allocate space for dev_equalizers_padded_GPU2" << endl;
	cudaSetDevice(device_GPU2); cufftPlan1d(&fftPlan_detection_GPU2, conv_length, CUFFT_C2C, MAX_PACKETS_PER_MEMORY_SECTION);
	cudaSetDevice(device_GPU2); cufftExecC2C(fftPlan_detection_GPU2, dev_filter_preDetection_GPU2, dev_filter_preDetection_GPU2, CUFFT_FORWARD);
	cudaSetDevice(device_GPU2); cudaStat = cudaMalloc((void**)&dev_Samples_padded_GPU2, sizeof(cufftComplex)*conv_length*MAX_PACKETS_PER_MEMORY_SECTION); if(cudaStat != cudaSuccess) cout << "ERROR -- Could not allocate space for dev_Samples_padded_GPU2" << endl;
	cudaSetDevice(device_GPU2); cudaStat = cudaMalloc((void**)&dev_detected_downsampled_GPU2, sizeof(cufftComplex)*BITS_PER_PACKET*MAX_PACKETS_PER_MEMORY_SECTION); if(cudaStat != cudaSuccess) cout << "ERROR -- Could not allocate space for dev_detected_downsampled_GPU2" << endl;
	cudaSetDevice(device_GPU2); cudaMalloc((void**)&dev_MMSE_ahat_GPU2, sizeof(float)*BITS_PER_PACKET*MAX_PACKETS_PER_MEMORY_SECTION);
	cudaSetDevice(device_GPU2); cudaMalloc((void**)&dev_MMSE_bits_GPU2, sizeof(unsigned char)*BITS_PER_DATA_PACKET*MAX_PACKETS_PER_MEMORY_SECTION);
	cudaSetDevice(device_GPU0); cudaMalloc((void**)&dev_MMSE_bits_GPU0, sizeof(unsigned char)*BITS_PER_DATA_PACKET*MAX_PACKETS_PER_MEMORY_SECTION);
	cudaSetDevice(device_GPU0);




	maxThreads_zeroPadEQUALIZER = EQUALIZER_LENGTH*MAX_PACKETS_PER_MEMORY_SECTION;
	numThreads_zeroPadEQUALIZER = 128; // Tested K40
	numBlocks_zeroPadEQUALIZER = maxThreads_zeroPadEQUALIZER/numThreads_zeroPadEQUALIZER;
	if((maxThreads_zeroPadEQUALIZER % numThreads_zeroPadEQUALIZER) > 0) numBlocks_zeroPadEQUALIZER++;

	maxThreads_zeroPadPACKET = SAMPLES_PER_PACKET*MAX_PACKETS_PER_MEMORY_SECTION;
	numThreads_zeroPadPACKET = 128; // Tested K40
	numBlocks_zeroPadPACKET = maxThreads_zeroPadPACKET/numThreads_zeroPadPACKET;
	if((maxThreads_zeroPadPACKET % numThreads_zeroPadPACKET) > 0) numBlocks_zeroPadPACKET++;

	maxThreads_pointMultiplyTriple = conv_length*MAX_PACKETS_PER_MEMORY_SECTION;
	numThreads_pointMultiplyTriple = 160; // Tested K40
	numBlocks_pointMultiplyTriple = maxThreads_pointMultiplyTriple/numThreads_pointMultiplyTriple;
	if((maxThreads_pointMultiplyTriple % numThreads_pointMultiplyTriple) > 0) numBlocks_pointMultiplyTriple++;

	maxThreads_pointMultiplyQuad = conv_length*MAX_PACKETS_PER_MEMORY_SECTION;
	numThreads_pointMultiplyQuad = 128; // Tested K40
	numBlocks_pointMultiplyQuad = maxThreads_pointMultiplyQuad/numThreads_pointMultiplyQuad;
	if((maxThreads_pointMultiplyQuad % numThreads_pointMultiplyQuad) > 0) numBlocks_pointMultiplyQuad++;

	maxThreads_dmodPostPruneScaledDownsample = BITS_PER_PACKET*MAX_PACKETS_PER_MEMORY_SECTION;
	numThreads_dmodPostPruneScaledDownsample = 128;
	numBlocks_dmodPostPruneScaledDownsample = maxThreads_dmodPostPruneScaledDownsample/numThreads_dmodPostPruneScaledDownsample;
	if((maxThreads_dmodPostPruneScaledDownsample % numThreads_dmodPostPruneScaledDownsample) > 0) numBlocks_dmodPostPruneScaledDownsample++;
}
void GPUHandler::free_apply_equalizers_and_detection_filters(){
	delete [] filter_fft;
	cufftDestroy(fftPlan_apply_GPU0);
	cudaFree(dev_Samples_padded_GPU0);
	cudaFree(dev_equalizers_padded_GPU0);

	//cufftDestroy(fftPlan_apply_GPU1);
	//cudaFree(dev_Samples_padded_GPU1);
	//cudaFree(dev_equalizers_padded_GPU1);

	cudaFree(dev_signal_preDetection_GPU0);
	cudaFree(dev_filter_preDetection_GPU0);
	cudaFree(dev_detected_downsampled_GPU0);
	cufftDestroy(fftPlan_detection_GPU0);

	//cudaFree(dev_signal_preDetection_GPU1);
	//cudaFree(dev_filter_preDetection_GPU1);
	cudaFree(dev_detected_downsampled_GPU1);
	cufftDestroy(fftPlan_detection_GPU1);

	//MMSE
	cudaFree(dev_Samples_padded_GPU2);
	cudaFree(dev_detected_downsampled_GPU2);
	cudaFree(dev_MMSE_ahat_GPU2);
	cudaFree(dev_MMSE_bits_GPU2);
	cudaFree(dev_MMSE_bits_GPU0);
	cufftDestroy(fftPlan_detection_GPU2);
}
void GPUHandler::apply_equalizers_and_detection_filters() {
	cudaSetDevice(device_GPU0);	cudaMemset						(dev_equalizers_padded_GPU0,				0,											sizeof(cufftComplex)*conv_length*MAX_PACKETS_PER_MEMORY_SECTION);
	cudaSetDevice(device_GPU2);	cudaMemset						(dev_FDE_H_padded_GPU2,						0,											sizeof(cufftComplex)*conv_length*MAX_PACKETS_PER_MEMORY_SECTION);

	cudaSetDevice(device_GPU0);	zeroPad							<<<numBlocks_zeroPadEQUALIZER,				numThreads_zeroPadEQUALIZER>>>				(dev_equalizers_padded_GPU0, dev_MMSE_CMA_equalizers_GPU0, 	MAX_PACKETS_PER_MEMORY_SECTION, EQUALIZER_LENGTH, conv_length);
	cudaSetDevice(device_GPU2);	zeroPad							<<<numBlocks_zeroPadEQUALIZER, 				numThreads_zeroPadEQUALIZER>>>				(dev_FDE_H_padded_GPU2, 	 dev_h_hat_freq_GPU2, 			MAX_PACKETS_PER_MEMORY_SECTION, CHAN_SIZE, 		  conv_length);

	cudaSetDevice(device_GPU0);	cudaMemset				 		(dev_Samples_padded_GPU0,   				0,											sizeof(cufftComplex)*conv_length*MAX_PACKETS_PER_MEMORY_SECTION);
	cudaSetDevice(device_GPU2);	cudaMemset						(dev_FDE_Y_padded_GPU2,						0,											sizeof(cufftComplex)*conv_length*MAX_PACKETS_PER_MEMORY_SECTION);

	cudaSetDevice(device_GPU2);	copyPreambleShiftOnEnd			<<<1, 										1>>>										(dev_Samples_GPU2, shiftFDE);
	cudaSetDevice(device_GPU0);	zeroPad							<<<numBlocks_zeroPadPACKET,					numThreads_zeroPadPACKET>>>					(dev_Samples_padded_GPU0,    dev_Samples_GPU0, 	MAX_PACKETS_PER_MEMORY_SECTION, SAMPLES_PER_PACKET, conv_length);
	cudaSetDevice(device_GPU2);	zeroPadShiftFDE					<<<numBlocks_zeroPadPACKET, 				numThreads_zeroPadPACKET>>>					(dev_FDE_Y_padded_GPU2, 	 dev_Samples_GPU2, 	MAX_PACKETS_PER_MEMORY_SECTION, SAMPLES_PER_PACKET, conv_length, shiftFDE);

	cudaSetDevice(device_GPU0);	cufftExecC2C					(fftPlan_apply_GPU0, 						dev_equalizers_padded_GPU0, 				dev_equalizers_padded_GPU0,	CUFFT_FORWARD);
	cudaSetDevice(device_GPU2);	cufftExecC2C					(fftPlan_apply_GPU2, 						dev_FDE_H_padded_GPU2,						dev_FDE_H_padded_GPU2,		CUFFT_FORWARD);

	cudaSetDevice(device_GPU0);	cufftExecC2C					(fftPlan_apply_GPU0, 						dev_Samples_padded_GPU0, 					dev_Samples_padded_GPU0, 	CUFFT_FORWARD);
	cudaSetDevice(device_GPU2);	cufftExecC2C					(fftPlan_apply_GPU2, 						dev_FDE_Y_padded_GPU2,						dev_FDE_Y_padded_GPU2, 		CUFFT_FORWARD);

	cudaSetDevice(device_GPU0);	pointMultiplyTriple				<<<numBlocks_pointMultiplyTriple, 			numThreads_pointMultiplyTriple>>>			(dev_Samples_padded_GPU0, dev_equalizers_padded_GPU0, dev_filter_preDetection_GPU0, maxThreads_pointMultiplyTriple); // dest,src,length
	cudaSetDevice(device_GPU2);	pointMultiplyQuadFDE1			<<<numBlocks_pointMultiplyTriple, 			numThreads_pointMultiplyTriple>>>			(dev_FDE_Y_padded_GPU2,   dev_FDE_H_padded_GPU2, dev_shs_GPU2, dev_filter_preDetection_GPU2, maxThreads_pointMultiplyTriple);

	cudaSetDevice(device_GPU0); cufftExecC2C					(fftPlan_detection_GPU0, 					dev_Samples_padded_GPU0,  					dev_Samples_padded_GPU0, CUFFT_INVERSE);
	cudaSetDevice(device_GPU2);	cufftExecC2C					(fftPlan_detection_GPU2, 					dev_FDE_Y_padded_GPU2, 						dev_FDE_Y_padded_GPU2,   CUFFT_INVERSE);

	cudaSetDevice(device_GPU0);	dmodPostPruneScaledDownsample		<<<numBlocks_dmodPostPruneScaledDownsample,	numThreads_dmodPostPruneScaledDownsample>>>	(dev_Samples_padded_GPU0, 	dev_detected_downsampled_GPU0, 		demod_filter_length/2+L1, SAMPLES_PER_PACKET, conv_length, downsampled_by, (float)1/conv_length, 			0,			maxThreads_dmodPostPruneScaledDownsample);
	cudaSetDevice(device_GPU2);	dmodPostPruneScaledDownsample		<<<numBlocks_dmodPostPruneScaledDownsample,	numThreads_dmodPostPruneScaledDownsample>>>	(dev_FDE_Y_padded_GPU2,  	dev_FDE1_detected_downsampled_rotated_GPU2,	demod_filter_length/2,    SAMPLES_PER_PACKET, conv_length, downsampled_by, (float)1/conv_length, 	shiftFDE+N1,	maxThreads_dmodPostPruneScaledDownsample);

	// Run ZF with the same memory as MMSE on GPU2
	// Copy c_ZF from GPU1->GPU2
	cudaSetDevice(device_GPU2);	cudaMemcpy(dev_ZF_equalizers_GPU2,	dev_ZF_equalizers_GPU1,	sizeof(cuComplex)*EQUALIZER_LENGTH*MAX_PACKETS_PER_MEMORY_SECTION, cudaMemcpyDeviceToDevice);
	cudaSetDevice(device_GPU2);	cudaMemset						(dev_equalizers_padded_GPU2,				0,											sizeof(cufftComplex)*conv_length*MAX_PACKETS_PER_MEMORY_SECTION);
	cudaSetDevice(device_GPU2);	zeroPad							<<<numBlocks_zeroPadEQUALIZER,				numThreads_zeroPadEQUALIZER>>>				(dev_equalizers_padded_GPU2, dev_ZF_equalizers_GPU2, 		MAX_PACKETS_PER_MEMORY_SECTION, EQUALIZER_LENGTH, conv_length);
	cudaSetDevice(device_GPU2);	cudaMemset						(dev_Samples_padded_GPU2,   				0,											sizeof(cufftComplex)*conv_length*MAX_PACKETS_PER_MEMORY_SECTION);
	cudaSetDevice(device_GPU2);	zeroPad							<<<numBlocks_zeroPadPACKET,					numThreads_zeroPadPACKET>>>					(dev_Samples_padded_GPU2,    dev_Samples_GPU2, 	MAX_PACKETS_PER_MEMORY_SECTION, SAMPLES_PER_PACKET, conv_length);
	cudaSetDevice(device_GPU2);	cufftExecC2C					(fftPlan_apply_GPU2, 						dev_equalizers_padded_GPU2, 				dev_equalizers_padded_GPU2,	CUFFT_FORWARD);
	cudaSetDevice(device_GPU2);	cufftExecC2C					(fftPlan_apply_GPU2, 						dev_Samples_padded_GPU2, 	  				dev_Samples_padded_GPU2, 	CUFFT_FORWARD);
	cudaSetDevice(device_GPU2);	pointMultiplyTriple				<<<numBlocks_pointMultiplyTriple, 			numThreads_pointMultiplyTriple>>>			(dev_Samples_padded_GPU2, dev_equalizers_padded_GPU2, dev_filter_preDetection_GPU2, maxThreads_pointMultiplyTriple); // dest,src,length
	cudaSetDevice(device_GPU2); cufftExecC2C					(fftPlan_detection_GPU2, 					dev_Samples_padded_GPU2, 					dev_Samples_padded_GPU2, CUFFT_INVERSE);
	cudaSetDevice(device_GPU2);	dmodPostPruneScaledDownsample	<<<numBlocks_dmodPostPruneScaledDownsample,	numThreads_dmodPostPruneScaledDownsample>>>	(dev_Samples_padded_GPU2,	dev_detected_downsampled_GPU2, 		demod_filter_length/2+L1, SAMPLES_PER_PACKET, conv_length, downsampled_by, (float)1/conv_length, 			0,			maxThreads_dmodPostPruneScaledDownsample);
	// Copy ZF equalized samples from GPU2->GPU1
	cudaSetDevice(device_GPU2);	cudaMemcpy(dev_detected_downsampled_GPU1,	dev_detected_downsampled_GPU2,	sizeof(cuComplex)*BITS_PER_PACKET*MAX_PACKETS_PER_MEMORY_SECTION, cudaMemcpyDeviceToDevice);

	// Run MMSE on GPU2
	cudaSetDevice(device_GPU2);	cudaMemset						(dev_equalizers_padded_GPU2,				0,											sizeof(cufftComplex)*conv_length*MAX_PACKETS_PER_MEMORY_SECTION);
	cudaSetDevice(device_GPU2);	zeroPad							<<<numBlocks_zeroPadEQUALIZER,				numThreads_zeroPadEQUALIZER>>>				(dev_equalizers_padded_GPU2, dev_MMSEequalizers_GPU2, 		MAX_PACKETS_PER_MEMORY_SECTION, EQUALIZER_LENGTH, conv_length);
	cudaSetDevice(device_GPU2);	cudaMemset						(dev_Samples_padded_GPU2,   				0,											sizeof(cufftComplex)*conv_length*MAX_PACKETS_PER_MEMORY_SECTION);
	cudaSetDevice(device_GPU2);	zeroPad							<<<numBlocks_zeroPadPACKET,					numThreads_zeroPadPACKET>>>					(dev_Samples_padded_GPU2,    dev_Samples_GPU2, 	MAX_PACKETS_PER_MEMORY_SECTION, SAMPLES_PER_PACKET, conv_length);
	cudaSetDevice(device_GPU2);	cufftExecC2C					(fftPlan_apply_GPU2, 						dev_equalizers_padded_GPU2, 				dev_equalizers_padded_GPU2,	CUFFT_FORWARD);
	cudaSetDevice(device_GPU2);	cufftExecC2C					(fftPlan_apply_GPU2, 						dev_Samples_padded_GPU2, 	  				dev_Samples_padded_GPU2, 	CUFFT_FORWARD);
	cudaSetDevice(device_GPU2);	pointMultiplyTriple				<<<numBlocks_pointMultiplyTriple, 			numThreads_pointMultiplyTriple>>>			(dev_Samples_padded_GPU2, dev_equalizers_padded_GPU2, dev_filter_preDetection_GPU2, maxThreads_pointMultiplyTriple); // dest,src,length
	cudaSetDevice(device_GPU2); cufftExecC2C					(fftPlan_detection_GPU2, 					dev_Samples_padded_GPU2, 					dev_Samples_padded_GPU2, CUFFT_INVERSE);
	cudaSetDevice(device_GPU2);	dmodPostPruneScaledDownsample	<<<numBlocks_dmodPostPruneScaledDownsample,	numThreads_dmodPostPruneScaledDownsample>>>	(dev_Samples_padded_GPU2,	dev_detected_downsampled_GPU2, 		demod_filter_length/2+L1, SAMPLES_PER_PACKET, conv_length, downsampled_by, (float)1/conv_length, 			0,			maxThreads_dmodPostPruneScaledDownsample);
	//	cudaSetDevice(device_GPU1);	cudaMemset						(dev_equalizers_padded_GPU1,				0,											sizeof(cufftComplex)*conv_length*MAX_PACKETS_PER_MEMORY_SECTION);
	//	cudaSetDevice(device_GPU1);	zeroPad							<<<numBlocks_zeroPadEQUALIZER,				numThreads_zeroPadEQUALIZER>>>				(dev_equalizers_padded_GPU1, dev_ZF_equalizers_GPU1, 		MAX_PACKETS_PER_MEMORY_SECTION, EQUALIZER_LENGTH, conv_length);
	//	cudaSetDevice(device_GPU1);	cudaMemset						(dev_Samples_padded_GPU1,   				0,											sizeof(cufftComplex)*conv_length*MAX_PACKETS_PER_MEMORY_SECTION);
	//	cudaSetDevice(device_GPU1);	zeroPad							<<<numBlocks_zeroPadPACKET,					numThreads_zeroPadPACKET>>>					(dev_Samples_padded_GPU1,    dev_Samples_GPU1, 	MAX_PACKETS_PER_MEMORY_SECTION, SAMPLES_PER_PACKET, conv_length);
	//	cudaSetDevice(device_GPU1);	cufftExecC2C					(fftPlan_apply_GPU1, 						dev_equalizers_padded_GPU1, 				dev_equalizers_padded_GPU1,	CUFFT_FORWARD);
	//	cudaSetDevice(device_GPU1);	cufftExecC2C					(fftPlan_apply_GPU1, 						dev_Samples_padded_GPU1, 	  				dev_Samples_padded_GPU1, 	CUFFT_FORWARD);
	//	cudaSetDevice(device_GPU1);	pointMultiplyTriple				<<<numBlocks_pointMultiplyTriple, 			numThreads_pointMultiplyTriple>>>			(dev_Samples_padded_GPU1, dev_equalizers_padded_GPU1, dev_filter_preDetection_GPU1, maxThreads_pointMultiplyTriple); // dest,src,length
	//	cudaSetDevice(device_GPU1); cufftExecC2C					(fftPlan_detection_GPU1, 					dev_Samples_padded_GPU1, 					dev_Samples_padded_GPU1, CUFFT_INVERSE);
	//	cudaSetDevice(device_GPU1);	dmodPostPruneScaledDownsample		<<<numBlocks_dmodPostPruneScaledDownsample,	numThreads_dmodPostPruneScaledDownsample>>>	(dev_Samples_padded_GPU1, 	dev_detected_downsampled_GPU1, 		demod_filter_length/2+L1, SAMPLES_PER_PACKET, conv_length, downsampled_by, (float)1/conv_length, 			0,			maxThreads_dmodPostPruneScaledDownsample);

	// Run FDE2 with the same memory as FDE1
	cudaSetDevice(device_GPU2);	cudaMemset						(dev_FDE_H_padded_GPU2,					0,													sizeof(cufftComplex)*conv_length*MAX_PACKETS_PER_MEMORY_SECTION);
	cudaSetDevice(device_GPU2);	zeroPad							<<<numBlocks_zeroPadEQUALIZER,				numThreads_zeroPadEQUALIZER>>>					(dev_FDE_H_padded_GPU2,  	 dev_h_hat_freq_GPU2, 			MAX_PACKETS_PER_MEMORY_SECTION, CHAN_SIZE, 		  conv_length);
	cudaSetDevice(device_GPU2);	cudaMemset						(dev_FDE_Y_padded_GPU2,					0,													sizeof(cufftComplex)*conv_length*MAX_PACKETS_PER_MEMORY_SECTION);
	cudaSetDevice(device_GPU2);	zeroPadShiftFDE					<<<numBlocks_zeroPadPACKET, 				numThreads_zeroPadPACKET>>>					(dev_FDE_Y_padded_GPU2, 	 dev_Samples_GPU2, 	MAX_PACKETS_PER_MEMORY_SECTION, SAMPLES_PER_PACKET, conv_length, shiftFDE);
	cudaSetDevice(device_GPU2);	cufftExecC2C					(fftPlan_apply_GPU2, 						dev_FDE_H_padded_GPU2,							dev_FDE_H_padded_GPU2,		CUFFT_FORWARD);
	cudaSetDevice(device_GPU2);	cufftExecC2C					(fftPlan_apply_GPU2, 						dev_FDE_Y_padded_GPU2,							dev_FDE_Y_padded_GPU2, 	CUFFT_FORWARD);
	cudaSetDevice(device_GPU2);	pointMultiplyQuadFDE2			<<<numBlocks_pointMultiplyQuad,				numThreads_pointMultiplyQuad>>>					(dev_FDE_Y_padded_GPU2, dev_FDE_H_padded_GPU2, dev_shs_GPU2, dev_filter_preDetection_GPU2, dev_FDE_PSI_GPU2, maxThreads_pointMultiplyQuad);
	cudaSetDevice(device_GPU2);	cufftExecC2C					(fftPlan_detection_GPU2, 					dev_FDE_Y_padded_GPU2, 							dev_FDE_Y_padded_GPU2,  CUFFT_INVERSE);
	cudaSetDevice(device_GPU2);	dmodPostPruneScaledDownsample		<<<numBlocks_dmodPostPruneScaledDownsample,	numThreads_dmodPostPruneScaledDownsample>>>	(dev_FDE_Y_padded_GPU2,  dev_FDE2_detected_downsampled_rotated_GPU2,	demod_filter_length/2,  SAMPLES_PER_PACKET, conv_length, downsampled_by, (float)1/conv_length, shiftFDE+N1,	maxThreads_dmodPostPruneScaledDownsample);
	cudaSetDevice(device_GPU0);

}

void GPUHandler::initialize_demodulators_variables(){
	cudaSetDevice(device_GPU0);	cudaMalloc((void**)&dev_CMA_ahat_GPU0, sizeof(float)*BITS_PER_PACKET*MAX_PACKETS_PER_MEMORY_SECTION);
	cudaSetDevice(device_GPU0);	cudaMalloc((void**)&dev_MMSE_bits_GPU0, sizeof(unsigned char)*BITS_PER_DATA_PACKET*MAX_PACKETS_PER_MEMORY_SECTION);

	cudaSetDevice(device_GPU1);	cudaMalloc((void**)&dev_ZF_ahat_GPU1, sizeof(float)*BITS_PER_PACKET*MAX_PACKETS_PER_MEMORY_SECTION);
	cudaSetDevice(device_GPU1);	cudaMalloc((void**)&dev_ZF_bits_GPU1, sizeof(unsigned char)*BITS_PER_DATA_PACKET*MAX_PACKETS_PER_MEMORY_SECTION);
	cudaSetDevice(device_GPU0);	cudaMalloc((void**)&dev_ZF_bits_GPU0, sizeof(unsigned char)*BITS_PER_DATA_PACKET*MAX_PACKETS_PER_MEMORY_SECTION);

	cudaSetDevice(device_GPU0);	cudaMalloc((void**)&dev_all8Channels_bits, sizeof(unsigned char)*BITS_PER_DATA_PACKET*MAX_PACKETS_PER_MEMORY_SECTION);
	cudaSetDevice(device_GPU0);


	maxThreads_cudaDemodulator = MAX_PACKETS_PER_MEMORY_SECTION;
	numThreads_cudaDemodulator = 15; // K40: 13  K20: 15
	numBlocks_cudaDemodulator = maxThreads_cudaDemodulator/numThreads_cudaDemodulator;
	if((maxThreads_cudaDemodulator % numThreads_cudaDemodulator) > 0) numBlocks_cudaDemodulator++;

	maxThreads_bitPruneDemod = BITS_PER_DATA_PACKET*MAX_PACKETS_PER_MEMORY_SECTION;
	numThreads_bitPruneDemod = 128; // Tested
	numBlocks_bitPruneDemod = maxThreads_bitPruneDemod/numThreads_bitPruneDemod;
	if((maxThreads_bitPruneDemod % numThreads_bitPruneDemod) > 0) numBlocks_bitPruneDemod++;

	maxThreads_bit8Channels = BITS_PER_DATA_PACKET*MAX_PACKETS_PER_MEMORY_SECTION/8;
	numThreads_bit8Channels = 128; // Tested
	numBlocks_bit8Channels = maxThreads_bit8Channels/numThreads_bit8Channels;
	if((maxThreads_bit8Channels % numThreads_bit8Channels) > 0) numBlocks_bit8Channels++;
}
void GPUHandler::free_demodulators_variables(){
	cudaFree(dev_CMA_ahat_GPU0);
	cudaFree(dev_MMSE_bits_GPU0);

	cudaFree(dev_ZF_ahat_GPU1);
	cudaFree(dev_ZF_bits_GPU1);
	cudaFree(dev_ZF_bits_GPU0);

	cudaFree(dev_all8Channels_bits);
}
void GPUHandler::apply_demodulators(){
	cudaSetDevice(device_GPU0);	cudaDemodulator		<<<numBlocks_cudaDemodulator,	numThreads_cudaDemodulator>>>	(dev_detected_downsampled_GPU0, 				dev_CMA_ahat_GPU0, 	BITS_PER_PACKET, 0, 	maxThreads_cudaDemodulator); // src, dest
	cudaSetDevice(device_GPU1);	cudaDemodulator		<<<numBlocks_cudaDemodulator,	numThreads_cudaDemodulator>>>	(dev_detected_downsampled_GPU1, 				dev_ZF_ahat_GPU1,   BITS_PER_PACKET, 0, 	maxThreads_cudaDemodulator);
	cudaSetDevice(device_GPU2);	cudaDemodulator		<<<numBlocks_cudaDemodulator,	numThreads_cudaDemodulator>>>	(dev_detected_downsampled_GPU2, 				dev_MMSE_ahat_GPU2, BITS_PER_PACKET, 0, 	maxThreads_cudaDemodulator);
	cudaSetDevice(device_GPU2);	cudaDemodulator		<<<numBlocks_cudaDemodulator,	numThreads_cudaDemodulator>>>	(dev_FDE1_detected_downsampled_rotated_GPU2, 	dev_FDE1_ahat_GPU2,	BITS_PER_PACKET, shiftFDE+N1, 	maxThreads_cudaDemodulator);
	cudaSetDevice(device_GPU2);	cudaDemodulator		<<<numBlocks_cudaDemodulator,	numThreads_cudaDemodulator>>>	(dev_FDE2_detected_downsampled_rotated_GPU2, 	dev_FDE2_ahat_GPU2,	BITS_PER_PACKET, shiftFDE+N1, 	maxThreads_cudaDemodulator);

	cudaSetDevice(device_GPU0);	bitPrune			<<<numBlocks_bitPruneDemod,		numThreads_bitPruneDemod>>>		(dev_CMA_bits_GPU0, 	dev_CMA_ahat_GPU0, 	BITS_PER_FRONT_PACKET, BITS_PER_DATA_PACKET, BITS_PER_PACKET, maxThreads_bitPruneDemod);
	cudaSetDevice(device_GPU1);	bitPrune			<<<numBlocks_bitPruneDemod,		numThreads_bitPruneDemod>>>		(dev_ZF_bits_GPU1,   	dev_ZF_ahat_GPU1,   BITS_PER_FRONT_PACKET, BITS_PER_DATA_PACKET, BITS_PER_PACKET, maxThreads_bitPruneDemod);
	cudaSetDevice(device_GPU2);	bitPrune			<<<numBlocks_bitPruneDemod,		numThreads_bitPruneDemod>>>		(dev_MMSE_bits_GPU2,  	dev_MMSE_ahat_GPU2,  BITS_PER_FRONT_PACKET, BITS_PER_DATA_PACKET, BITS_PER_PACKET, maxThreads_bitPruneDemod);
	cudaSetDevice(device_GPU2);	bitPrune			<<<numBlocks_bitPruneDemod,		numThreads_bitPruneDemod>>>		(dev_FDE1_bits_GPU2,   	dev_FDE1_ahat_GPU2,  BITS_PER_FRONT_PACKET, BITS_PER_DATA_PACKET, BITS_PER_PACKET, maxThreads_bitPruneDemod);
	cudaSetDevice(device_GPU2);	bitPrune			<<<numBlocks_bitPruneDemod,		numThreads_bitPruneDemod>>>		(dev_FDE2_bits_GPU2,   	dev_FDE2_ahat_GPU2,  BITS_PER_FRONT_PACKET, BITS_PER_DATA_PACKET, BITS_PER_PACKET, maxThreads_bitPruneDemod);

	cudaSetDevice(device_GPU1);	cudaMemcpy			(dev_ZF_bits_GPU0,				dev_ZF_bits_GPU1,				BITS_PER_DATA_PACKET*MAX_PACKETS_PER_MEMORY_SECTION*sizeof(char), cudaMemcpyDeviceToDevice);
	cudaSetDevice(device_GPU2);	cudaMemcpy			(dev_MMSE_bits_GPU0,			dev_MMSE_bits_GPU2,				BITS_PER_DATA_PACKET*MAX_PACKETS_PER_MEMORY_SECTION*sizeof(char), cudaMemcpyDeviceToDevice);
	cudaSetDevice(device_GPU2);	cudaMemcpy			(dev_FDE1_bits_GPU0,			dev_FDE1_bits_GPU2,				BITS_PER_DATA_PACKET*MAX_PACKETS_PER_MEMORY_SECTION*sizeof(char), cudaMemcpyDeviceToDevice);
	cudaSetDevice(device_GPU2);	cudaMemcpy			(dev_FDE2_bits_GPU0,			dev_FDE2_bits_GPU2,				BITS_PER_DATA_PACKET*MAX_PACKETS_PER_MEMORY_SECTION*sizeof(char), cudaMemcpyDeviceToDevice);
	//	writeFreq_bits2();

	cudaSetDevice(device_GPU0);	cudaMemset			(dev_all8Channels_bits,			0,								sizeof(unsigned char)*BITS_PER_DATA_PACKET*MAX_PACKETS_PER_MEMORY_SECTION);
	cudaSetDevice(device_GPU0);	bit8Channels		<<<numBlocks_bit8Channels,		numThreads_bit8Channels>>>		(dev_all8Channels_bits, dev_MMSE_bits_GPU0,			1, maxThreads_bit8Channels);
	cudaSetDevice(device_GPU0);	bit8Channels		<<<numBlocks_bit8Channels,		numThreads_bit8Channels>>>		(dev_all8Channels_bits, dev_MMSE_bits_GPU0,			2, maxThreads_bit8Channels);
	cudaSetDevice(device_GPU0);	bit8Channels		<<<numBlocks_bit8Channels,		numThreads_bit8Channels>>>		(dev_all8Channels_bits, dev_CMA_bits_GPU0,	 		3, maxThreads_bit8Channels);
	cudaSetDevice(device_GPU0);	bit8Channels		<<<numBlocks_bit8Channels,		numThreads_bit8Channels>>>		(dev_all8Channels_bits, dev_FDE1_bits_GPU0,	 		4, maxThreads_bit8Channels);
	cudaSetDevice(device_GPU0);	bit8Channels		<<<numBlocks_bit8Channels,		numThreads_bit8Channels>>>		(dev_all8Channels_bits, dev_FDE2_bits_GPU0, 		5, maxThreads_bit8Channels);
}

void GPUHandler::initialize_BERT_variables(){
	ErroredBatches = -3;
	MissedTimingBatches = -1;
	CreateRunDirectory();

	// Create Array to hold all the errored Index
	bitErrorIdx = new int[MAX_NUM_PN11];
	numBitErrorAtIdx = new int[MAX_NUM_PN11];

	// Setup the fft plan
	cufftPlan1d(&fftPlan_BERT_GPU0, XCORR_NFFT, CUFFT_C2C, 1);

	// Setup pn11a
	cuComplex h_PN11A[pn11Length];
	ifstream myFile;	myFile.open("/home/adm85/git/JeffPaq/PAQfull/src/BERTTestFileFromMatlab_pn11a.txt");	if (!myFile.is_open())		printf("\n\n\t\tCould not open the file!\n\n");	else	{		int i = 0;		float output;		while (!myFile.eof())		{			myFile >> output;			h_PN11A[i].x = (int)output;			h_PN11A[i].y = 0;	i++;		}		myFile.close();	}
	cudaSetDevice(device_GPU0);	cudaMalloc((void**)&dev_PN11A_GPU0,  sizeof(cuComplex)*XCORR_NFFT);
	cudaSetDevice(device_GPU0);	cudaMemset(dev_PN11A_GPU0, 0, sizeof(cuComplex)*XCORR_NFFT);
	cudaMemcpy(dev_PN11A_GPU0, h_PN11A, pn11Length*sizeof(cuComplex), cudaMemcpyHostToDevice);
	cufftExecC2C(fftPlan_BERT_GPU0, dev_PN11A_GPU0, dev_PN11A_GPU0, CUFFT_FORWARD);	// Pre-compute the FFT of pn11a

	cudaSetDevice(device_GPU0);	cudaMalloc((void**)&dev_BERT_bits_GPU0,  			sizeof(cuComplex)*XCORR_NFFT);
	cudaSetDevice(device_GPU0);	cudaMalloc((void**)&dev_BERT_xCorrelatedBits_GPU0,  sizeof(int)*BITS_PER_DATA_PACKET*MAX_PACKETS_PER_MEMORY_SECTION);
	cudaSetDevice(device_GPU0);	cudaMalloc((void**)&dev_BERT_peaks_GPU0,  			sizeof(int)*MAX_NUM_PN11);
	cudaSetDevice(device_GPU0);	cudaMalloc((void**)&dev_BERT_peaksIdx_GPU0,  		sizeof(int)*MAX_NUM_PN11);

}
void GPUHandler::free_BERT_variables(){
	delete[] bitErrorIdx;
	delete[] numBitErrorAtIdx;

	cudaFree(dev_PN11A_GPU0);
	cufftDestroy(fftPlan_BERT_GPU0);

	cudaFree(dev_BERT_bits_GPU0);
	cudaFree(dev_BERT_xCorrelatedBits_GPU0);
	cudaFree(dev_BERT_peaks_GPU0);
	cudaFree(dev_BERT_peaksIdx_GPU0);
}
int  GPUHandler::BERT(int processedBits){
	cudaSetDevice(device_GPU0);	cudaDeviceSynchronize();
	cudaSetDevice(device_GPU1);	cudaDeviceSynchronize();
	cudaSetDevice(device_GPU2);	cudaDeviceSynchronize();

	cout << endl;
	for(int equalizer = 0; equalizer < 1; equalizer++){

		// Make it dynamic on which Equalizer the BERT is checking.
		switch (equalizer) {
		case 0:
			cout << "Running BERT on ZF" << endl;
			dev_BERT_bits_pull_pointer_GPU0 = dev_ZF_bits_GPU0;
			break;
		case 1:
			cout << "Running BERT on MMSE" << endl;
			dev_BERT_bits_pull_pointer_GPU0 = dev_MMSE_bits_GPU0;
			break;
		case 2:
			cout << "Running BERT on CMA" << endl;
			dev_BERT_bits_pull_pointer_GPU0 = dev_CMA_bits_GPU0;
			break;
		case 3:
			cout << "Running BERT on FDE1" << endl;
			dev_BERT_bits_pull_pointer_GPU0 = dev_FDE1_bits_GPU0;
			break;
		case 4:
			cout << "Running BERT on FDE2" << endl;
			dev_BERT_bits_pull_pointer_GPU0 = dev_FDE2_bits_GPU0;
			break;
		}



		// Write out what bits are being tested
		//	writeBERT_bits(processedBits);


		// Pull bits into -1 +1
		int maxThreads = processedBits;
		int numThreads = 128;
		int numBlocks = maxThreads/numThreads;
		if((maxThreads % numThreads) > 0) numBlocks++;
		cudaSetDevice(device_GPU0);	cudaMemset(dev_BERT_bits_GPU0, 0, sizeof(cuComplex)*XCORR_NFFT);
		cudaSetDevice(device_GPU0);	pullBitsXcorr<<<numBlocks, numThreads>>>(dev_BERT_bits_GPU0, dev_BERT_bits_pull_pointer_GPU0, maxThreads);
		//	writeBERT_signed_bits(XCORR_NFFT);


		cudaSetDevice(device_GPU0);	cufftExecC2C(fftPlan_BERT_GPU0, dev_BERT_bits_GPU0,  dev_BERT_bits_GPU0,  CUFFT_FORWARD);
		//	writeBERT_BITS(XCORR_NFFT);


		maxThreads = XCORR_NFFT;
		numThreads = 128;
		numBlocks = maxThreads/numThreads;
		if((maxThreads % numThreads) > 0) numBlocks++;
		cudaSetDevice(device_GPU0);	pointToPointConj<<<numBlocks, numThreads>>>(dev_BERT_bits_GPU0, dev_PN11A_GPU0, maxThreads);
		//	writeBERT_conj_MULTI(XCORR_NFFT);

		cudaSetDevice(device_GPU0);	cufftExecC2C(fftPlan_BERT_GPU0, dev_BERT_bits_GPU0,  dev_BERT_bits_GPU0,  CUFFT_INVERSE);
		//	writeBERT_multi(XCORR_NFFT);

		maxThreads = processedBits-2047;
		numThreads = 128;
		numBlocks = maxThreads/numThreads;
		if((maxThreads % numThreads) > 0) numBlocks++;
		cudaSetDevice(device_GPU0);	pullxCorrBits<<<numBlocks, numThreads>>>(dev_BERT_xCorrelatedBits_GPU0, dev_BERT_bits_GPU0, XCORR_NFFT, processedBits, maxThreads);
		//	writeBERT_correlated(processedBits-2047);

		maxThreads = MAX_NUM_PN11;
		numThreads = 128;
		numBlocks = maxThreads/numThreads;
		if((maxThreads % numThreads) > 0) numBlocks++;
		cudaSetDevice(device_GPU0);	peakSearchXcorr<<<numBlocks, numThreads>>>(dev_BERT_peaks_GPU0, dev_BERT_peaksIdx_GPU0, dev_BERT_xCorrelatedBits_GPU0, processedBits-2047, maxThreads);


		// Find the number of peaks
		int peaks[MAX_NUM_PN11];
		cudaMemcpy(peaks, dev_BERT_peaks_GPU0, MAX_NUM_PN11*sizeof(int), cudaMemcpyDeviceToHost);
		numPeaks = 0;
		for(int i = MAX_NUM_PN11-10; i < MAX_NUM_PN11; i++)
			if(peaks[i] == 2047)
				numPeaks = i+1;

		int firstPeak = 0;
		cudaMemcpy(&firstPeak, dev_BERT_peaksIdx_GPU0, sizeof(int), cudaMemcpyDeviceToHost);

		// Count the number of Errors
		bitErrorCount = 0;
		int erroredIdx = 0;
		for(int i = 0; i < numPeaks; i++){
			int test = (2047 - peaks[i])/2;
			if(test != 0){
				bitErrorCount += test;
				cout << "\tNum PN11 Packet:" << i << "\tNum Data Packet: " << (i+firstPeak)*2047/6144 << "\tNumber of Bit Errors: " << test << endl;
				numBitErrorAtIdx[erroredIdx] = test;
				bitErrorIdx[erroredIdx++] = i;
			}
			else{
				numBitErrorAtIdx[erroredIdx] = 0;
				bitErrorIdx[erroredIdx++] = 0;
			}
		}

		cout << "\tBit Errors:\t" << bitErrorCount;

		if(bitErrorCount>0){
			//writeBatchFiles();
		}

	} // End of BERT loop

	if(ErroredBatches>0)
		cout << "\tErroredBatches: " << ErroredBatches << endl;
	cout << endl;
	return bitErrorCount;
}

int GPUHandler::postCPUrun(unsigned char* bits_8channels_host,bool lastRun,int PolyWriteIdx){
	cudaSetDevice(device_GPU0);	cudaDeviceSynchronize();
	cudaSetDevice(device_GPU1);	cudaDeviceSynchronize();
	cudaSetDevice(device_GPU2);	cudaDeviceSynchronize();
	cudaSetDevice(device_GPU0);	cudaMemcpy(bits_8channels_host, dev_all8Channels_bits, BITS_PER_DATA_PACKET*MAX_PACKETS_PER_MEMORY_SECTION*sizeof(char), cudaMemcpyDeviceToHost);

	int endIndex_FirstWindow;
	cudaMemcpy(&endIndex_FirstWindow, dev_endIndex_FirstWindow, sizeof(int), cudaMemcpyDeviceToHost);
	lastMax = myFirstMaxMod;
	cudaMemcpy(&myFirstMaxMod,		dev_myFirstMaxMod,			sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&myFirstMaxActual,	dev_myFirstMaxActual,	sizeof(int), cudaMemcpyDeviceToHost);
	shiftingMarch = lastMax-myFirstMaxMod;


	int max_locations[(MAX_PACKETS_PER_MEMORY_SECTION+1+2)];

	// Copy the array to the host
	cudaMemcpy(max_locations, dev_max_locations, (MAX_PACKETS_PER_MEMORY_SECTION+1) * sizeof(int), cudaMemcpyDeviceToHost);
	int myMin = 39321600+12672;
	int myMax = 0;
	for(int i = 0; i < (MAX_PACKETS_PER_MEMORY_SECTION+1); i++){
		if(max_locations[i]<myMin && max_locations[i]>=0 && i < 10)
			myMin = max_locations[i];
		if(max_locations[i]>myMax && max_locations[i]<=39321600+12672)
			myMax = max_locations[i];
	}
	cout << "First Preamble Index: " << myMin << "\tLast Preamble Index: " << myMax << "\t Shift From previous Batch: " << shiftingMarch << endl;


	float shs_host = 0;
	cudaSetDevice(device_GPU2);	cudaMemcpy(&shs_host,dev_noiseVariance_GPU0,sizeof(float),cudaMemcpyDeviceToHost);
	cout << "\nSHS: " << shs_host << endl << endl;

	bool errors = false;
	int maxSpace = 0;
	int minSpace = 123456789;
	int numOff = 0;
	for(int i = 10; i < MAX_PACKETS_PER_MEMORY_SECTION-10; i++){
		int space = max_locations[i]-max_locations[i-1];
		if(space-12672>0){
			if(space>maxSpace)
				maxSpace = space;
			numOff++;
		}
		if(space-12672<0){
			if(space<minSpace)
				minSpace = space;
			numOff++;
		}

		if(abs(space-12672)>10 && !errors){
			cout << "Preamble Jump detected." << endl;
			cout << "Bad Preamble Spacing at Packet " << i << "\t" << max_locations[i-1] << "\t" << max_locations[i] << "\t" << abs(max_locations[i]-max_locations[i-1]-12672) << endl;
			errors = true;
		}
	}

	//writeHhat();
	//writeBatch_Max(3103);


	if(minSpace == 123456789)
		minSpace = 12672;
	cout << "maxSpace: " << maxSpace << "\t" << "minSpace: " << minSpace << "\t" << "numOff: " << numOff << endl;
	//	if(shs_host>0.1)
	errors = false;
	if(errors){
		cout << "\n\nWriting Files!!!!!\n\n";
		CreateBatchDirectory();
		writeGeneralFile(PolyWriteIdx);
		writeBatch_Max(MAX_PACKETS_PER_MEMORY_SECTION+1+2);
		//		writeHhat();
		//		writeBatch_raw_i		(TOTAL_SAMPLES_LENGTH);
		//		writeBatch_raw_q		(TOTAL_SAMPLES_LENGTH);
		//		writeBatch_samples0_pd	(TOTAL_SAMPLES_LENGTH);

		//		if(myMax<NUM_INPUT_SAMPLES_DEFAULT)
		//			writeBatch_MMSEbits(MAX_PACKETS_PER_MEMORY_SECTION*BITS_PER_DATA_PACKET);
		//		else
		//			writeBatch_MMSEbits((MAX_PACKETS_PER_MEMORY_SECTION-1)*BITS_PER_DATA_PACKET);
	}



	//	writeFreq_prune_scale_downsample();
	//	writeFreq_ahat();
	//	if(firstMax <= 383)
	if(myMax<NUM_INPUT_SAMPLES_DEFAULT)
		return MAX_PACKETS_PER_MEMORY_SECTION*BITS_PER_DATA_PACKET;
	else
		return (MAX_PACKETS_PER_MEMORY_SECTION-1)*BITS_PER_DATA_PACKET;
}


void GPUHandler::writeFDEtestingFiles()
{
	// Create the FDE testing Directory
	CreateFDEtestingDirectory();

	// Write out the general file
	writeFDEtestingGeneralFile();

	// Write out the Derotated samples
	writeFDEtesting_samples2(TOTAL_SAMPLES_LENGTH);

	// Write out the Bit Decisions
	writeFDEtesting_FDE1bits(BITS_PER_DATA_PACKET*MAX_PACKETS_PER_MEMORY_SECTION);

	TimingReset();
}
void GPUHandler::CreateFDEtestingDirectory(){
	cout << "Creating FDE Testing Directory\n";

	FDEtestingPath = SSTR("/home/adm85/Testing/FDEtesting/");

	// Create the Date directory
	mkdir(FDEtestingPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
}
void GPUHandler::writeFDEtestingGeneralFile()
{
	// Open the General file in this batches path
	ofstream myfile;
	myfile.open (SSTR(FDEtestingPath << "GeneralFDEtestingInfo.txt").c_str());

	time_t t = time(0);   // get time now
	struct tm * now = localtime( & t );

	// Record date
	myfile << "Date: " << now->tm_wday << ", " << now->tm_mon << "-" << now->tm_mday << "-" << now->tm_year+1900;
	myfile << "\t" << now->tm_hour << ":" << now->tm_min << ":" << now->tm_sec << endl << endl;

	myfile << endl;

	myfile << "Time: " << myTime << " ms" << "\t Time: " << myTime/1000 << " sec" << endl;
	myfile << "\tThis  Case Time Left over: " << (maxTime*1000-myTime) << " ms" << endl;
	myfile << "\tWorst Case Time Left over: " << (maxTime*1000-wcet) << " ms" << endl;
	myfile << "\tBest  Case Time Left over: " << (maxTime*1000-bcet) << " ms" << endl;

	myfile << endl;

	myfile << "Number of Bit Errors:\t" << bitErrorCount << endl;

	myfile << endl;

	// Print out the Bit Error Index and Number of Bit Errors
	if(bitErrorCount>0){
		myfile << "Errors at:" << endl;
		for(int i = 0; i < numPeaks; i++){
			if(numBitErrorAtIdx[i] != 0)
				myfile << bitErrorIdx[i] << ":\t" << numBitErrorAtIdx[i] << endl;
		}
	}

	myfile.close();
}
void GPUHandler::writeFDEtesting_samples2 (int length)
{
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	cuComplex* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_Samples_GPU2;
	string fileName = "samples_GPU2";
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( FDEtestingPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	cuComplex* host_array = new cuComplex[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		cuComplex* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeFDEtesting_FDE1bits (int length)
{
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	unsigned char* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_FDE1_bits_GPU0;
	string fileName = "FDE1bits";
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( FDEtestingPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	unsigned char* host_array = new unsigned char[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(unsigned char));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		unsigned char* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(unsigned char));
	}
	out1.close();

	delete[] host_array;
}

void GPUHandler::writeBatchFiles()
{
	cout << endl;

	// Create the Batch Directory
	CreateBatchDirectory();


	float myPercentage = .1;


	// Write out the raw Samples
	writeBatch_raw_i(TOTAL_SAMPLES_LENGTH*myPercentage);
	writeBatch_raw_q(TOTAL_SAMPLES_LENGTH*myPercentage);

	// Write out the L from the Preamble detector
	//	writeBatch_L(TOTAL_SAMPLES_LENGTH*myPercentage);

	//	// Write out the Maximum Locations
	//	writeBatch_Max(MAX_PACKETS_PER_MEMORY_SECTION);
	//
	//	// Write out all W0
	//	writeBatch_w0();
	//
	//	// Write out the Derotated Samples
	//	writeBatch_samples0(TOTAL_SAMPLES_LENGTH);
	//	writeBatch_samples1(TOTAL_SAMPLES_LENGTH);
	//	writeBatch_samples2(TOTAL_SAMPLES_LENGTH);
	//
	//	// Write out the Channel Estimates
	writeBatch_channelEst(channelEst_m*(MAX_PACKETS_PER_MEMORY_SECTION+1));
	//
	//	// Write out the Equalizers
	//	writeBatch_ZF(EQUALIZER_LENGTH*MAX_PACKETS_PER_MEMORY_SECTION);
	//	writeBatch_MMSE(EQUALIZER_LENGTH*MAX_PACKETS_PER_MEMORY_SECTION);
	//	writeBatch_CMA(EQUALIZER_LENGTH*MAX_PACKETS_PER_MEMORY_SECTION);
	//
	//	// Write out the Downsampled data
	//	writeBatch_downCMA	(BITS_PER_PACKET*MAX_PACKETS_PER_MEMORY_SECTION);
	//	writeBatch_downZF	(BITS_PER_PACKET*MAX_PACKETS_PER_MEMORY_SECTION);
	//	writeBatch_downMMSE	(BITS_PER_PACKET*MAX_PACKETS_PER_MEMORY_SECTION);
	//	writeBatch_downFDE1	(BITS_PER_PACKET*MAX_PACKETS_PER_MEMORY_SECTION);
	//	writeBatch_downFDE2	(BITS_PER_PACKET*MAX_PACKETS_PER_MEMORY_SECTION);
	//
	//
	//	// Write out the Bit Decisions
	//	writeBatch_ZFbits	(BITS_PER_DATA_PACKET*MAX_PACKETS_PER_MEMORY_SECTION);
	//	writeBatch_MMSEbits	(BITS_PER_DATA_PACKET*MAX_PACKETS_PER_MEMORY_SECTION);
	//	writeBatch_CMAbits	(BITS_PER_DATA_PACKET*MAX_PACKETS_PER_MEMORY_SECTION);
	//	writeBatch_FDE1bits	(BITS_PER_DATA_PACKET*MAX_PACKETS_PER_MEMORY_SECTION);
	//	writeBatch_FDE2bits	(BITS_PER_DATA_PACKET*MAX_PACKETS_PER_MEMORY_SECTION);
	//
	//	TimingReset();
	cudaEventElapsedTime( &myTime, start, stop );
	writeGeneralFile(0);
}

void GPUHandler::CreateRunDirectorySync(){
	cout << "Creating Run Directory\n";

	ErroredBatches = -1;

	// Get the time
	time_t t = time(0);   // get time now
	struct tm * now = localtime( & t );

	// Build the Date string
	string date = SSTR((now->tm_mon + 1) << '_' << now->tm_mday << '_' <<  (now->tm_year + 1900));
	dayPath = SSTR( "/home/adm85/SyncBatchData/Day_" << date);

	// Create the Date directory
	mkdir(dayPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

	// Build the Run string
	runPath = SSTR( dayPath << "/RunStartedAt_" << now->tm_hour << '_' << now->tm_min << '_' <<  now->tm_sec);

	// Create the Batch directory
	mkdir(runPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

	cout << "Created runPath " << runPath << endl;
}

void GPUHandler::CreateRunDirectory(){
	cout << "Creating Run Directory\n";

	ErroredBatches = -2;

	// Get the time
	time_t t = time(0);   // get time now
	struct tm * now = localtime( & t );

	// Build the Date string
	string date = SSTR((now->tm_mon + 1) << '_' << now->tm_mday << '_' <<  (now->tm_year + 1900));
	dayPath = SSTR( "/home/adm85/ErroredBatchData/Day_" << date);

	// Create the Date directory
	mkdir(dayPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

	// Build the Run string
	runPath = SSTR( dayPath << "/RunStartedAt_" << now->tm_hour << '_' << now->tm_min << '_' <<  now->tm_sec);

	// Create the Batch directory
	mkdir(runPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
}
void GPUHandler::CreateBatchDirectory(){
	ErroredBatches++;
	cout << "Creating Batch Directory\n";

	// Build the Errored Batch string
	batchPath = SSTR( runPath << "/Batch" << ErroredBatches << "/");

	// Create the Batch directory
	mkdir(batchPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
}
void GPUHandler::CreateMissedTimingBatchDirectory(){
	cout << "Creating Missed Timing Batch Directory\n";

	// Build the Errored Batch string
	batchPath = SSTR( runPath << "/MissedTimingBatch" << MissedTimingBatches << "/");

	// Create the Batch directory
	mkdir(batchPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
}
void GPUHandler::writeMissedTimingBatchFiles()
{
	MissedTimingBatches++;

	// Create the Missed Timing Directory
	CreateMissedTimingBatchDirectory();

	writeGeneralFile(0);

	// Write out the raw Samples
	writeBatch_raw_i(TOTAL_SAMPLES_LENGTH);
	writeBatch_raw_q(TOTAL_SAMPLES_LENGTH);

	// Write out one W0
	writeBatch_w0();

	// Write out the Derotated Samples
	writeBatch_samples0(TOTAL_SAMPLES_LENGTH);
	writeBatch_samples1(TOTAL_SAMPLES_LENGTH);
	writeBatch_samples2(TOTAL_SAMPLES_LENGTH);

	// Write out the Channel Estimates
	writeBatch_channelEst(channelEst_m*(MAX_PACKETS_PER_MEMORY_SECTION+1));

	// Write out the Equalizers
	writeBatch_ZF(EQUALIZER_LENGTH*MAX_PACKETS_PER_MEMORY_SECTION);
	writeBatch_MMSE(EQUALIZER_LENGTH*MAX_PACKETS_PER_MEMORY_SECTION);
	writeBatch_CMA(EQUALIZER_LENGTH*MAX_PACKETS_PER_MEMORY_SECTION);

	// Write out the Bit Decisions
	writeBatch_ZFbits	(BITS_PER_DATA_PACKET*MAX_PACKETS_PER_MEMORY_SECTION);
	writeBatch_MMSEbits	(BITS_PER_DATA_PACKET*MAX_PACKETS_PER_MEMORY_SECTION);
	writeBatch_CMAbits	(BITS_PER_DATA_PACKET*MAX_PACKETS_PER_MEMORY_SECTION);
	writeBatch_FDE1bits	(BITS_PER_DATA_PACKET*MAX_PACKETS_PER_MEMORY_SECTION);
	writeBatch_FDE2bits	(BITS_PER_DATA_PACKET*MAX_PACKETS_PER_MEMORY_SECTION);

	TimingReset();

}
void GPUHandler::writeGeneralFile(int PolyWriteIdx)
{
	// Open the General file in this batches path
	ofstream myfile;
	myfile.open (SSTR(batchPath << "/GeneralBatchInfo.txt").c_str());

	time_t t = time(0);   // get time now
	struct tm * now = localtime( & t );

	// Record date
	myfile << "Date: " << now->tm_wday << ", " << now->tm_mon << "-" << now->tm_mday << "-" << now->tm_year+1900 << "\t" << now->tm_hour << ":" << now->tm_min << ":" << now->tm_sec << endl;
	myfile << "\tFirst L Max:\t" << firstMax << "\tLast L Max:\t" << lastMax << "\tshift:\t" << shiftingMarch << endl;
	myfile << "\tmyFirstMaxMod:\t" << myFirstMaxMod << "\tmyFirstMaxActual:\t" << myFirstMaxActual << "\tLast L Max:\t" << lastMax << "\tshift:\t" << shiftingMarch << endl;
	myfile << "Time: " << myTime << " ms" << "\t Time: " << myTime/1000 << " sec" << endl;
	myfile << "\tThis  Case Time Left over: " << (maxTime*1000-myTime) << " ms" << endl;
	myfile << "\tWorst Case Time Left over: " << (maxTime*1000-wcet) << " ms" << endl;
	myfile << "\tBest  Case Time Left over: " << (maxTime*1000-bcet) << " ms" << endl;

	myfile << "Number of samples left over in Poly FIFO: \t" <<  PolyWriteIdx << endl;

	myfile.close();
}
void GPUHandler::writeBatch_y_p (int length,cuComplex* local_array)
{
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	cuComplex* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_y_p_new;
	string fileName = "y_p";
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	//cout << "Writing " << fileName << " to " << path << endl;
	const char* path = filePath.c_str();
	cout << "Writing " << fileName << " to ";
	for(int i = 0; i < 30; i++)
		cout << path[i];
	cout << endl;


	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	cuComplex* host_array;
	host_array = local_array;

	if(length>1000000){
		// Allocate space on the host
		cuComplex* host_array = new cuComplex[lengthOfCopy];

		// Copy the array to the host
		cudaMemcpy(host_array, dev_array, length * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	}

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		cuComplex* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	}
	out1.close();

	if(length>1000000)
		delete[] host_array;
}
void GPUHandler::writeBatch_z_p (int length,cuComplex* local_array)
{
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	cuComplex* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_z_p;
	string fileName = "z_p";
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();
	cout << "Writing " << fileName << " to ";
	for(int i = 0; i < 30; i++)
		cout << path[i];
	cout << endl;

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	cuComplex* host_array = local_array;

	if(length>1000000){
		// Allocate space on the host
		host_array = new cuComplex[lengthOfCopy];

		// Copy the array to the host
		cudaMemcpy(host_array, dev_array, length * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	}

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		cuComplex* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	}
	out1.close();

	if(length>1000000)
		delete[] host_array;
}
void GPUHandler::writeBatch_raw_i (int length)
{
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	float* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_iSamples_pd;
	string fileName = "raw_i";
	cout << "writing " << fileName << endl;
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	float* host_array = new float[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(float), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(float));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		float* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(float));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBatch_raw_i_last_times (int length)
{
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	float* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_iSamples_pd_last_times;
	string fileName = "raw_i_last_times";
	cout << "writing " << fileName << endl;
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	float* host_array = new float[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(float), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(float));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		float* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(float));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBatch_raw_q_last_times (int length)
{
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	float* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_qSamples_pd_last_times;
	string fileName = "raw_q_last_times";
	cout << "writing " << fileName << endl;
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	float* host_array = new float[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(float), cudaMemcpyDeviceToHost);


	for(int i = 0; i<10; i++)
		cout << i << ": \t" << host_array[i] << endl;



	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(float));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		float* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(float));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBatch_raw_q (int length)
{
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	float* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_qSamples_pd;
	string fileName = "raw_q";
	cout << "writing " << fileName << endl;
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	float* host_array = new float[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(float), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(float));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		float* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(float));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBatch_L (int length)
{
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	float* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_L_pd;
	string fileName = "L";
	cout << "writing " << fileName << endl;
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	float* host_array = new float[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(float), cudaMemcpyDeviceToHost);


	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(float));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		float* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(float));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBatch_L_last_times (int length)
{
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	float* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_L_pd_last_times;
	string fileName = "L_last_times";
	cout << "writing " << fileName << endl;
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	float* host_array = new float[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(float), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(float));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		float* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(float));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBatch_Max (int length)
{
	//----------------------------------------------
	//	Find and Replace the Date Type (int)
	//----------------------------------------------
	int* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_max_locations;
	string fileName = "Max";
	cout << "writing " << fileName << endl;
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	int* host_array = new int[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(int), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(int));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		int* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(int));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBatch_Max_last_times (int length)
{
	//----------------------------------------------
	//	Find and Replace the Date Type (int)
	//----------------------------------------------
	int* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_max_locations_last_times;
	string fileName = "Max_last_times";
	cout << "writing " << fileName << endl;
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	int* host_array = new int[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(int), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(int));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		int* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(int));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBatch_w0 ()
{
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	float* dev_array;
	int length = MAX_PACKETS_PER_MEMORY_SECTION;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_w0;
	string fileName = "w0";
	cout << "writing " << fileName << endl;
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();

	// Allocate space on the host
	float host_array[MAX_PACKETS_PER_MEMORY_SECTION];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, MAX_PACKETS_PER_MEMORY_SECTION*sizeof(float), cudaMemcpyDeviceToHost);

	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], length*sizeof(float));
	out.close();
}
void GPUHandler::writeBatch_samples0 (int length)
{
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	cuComplex* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_Samples_GPU0;
	string fileName = "samples_GPU0";
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	cuComplex* host_array = new cuComplex[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		cuComplex* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBatch_samples0_pd (int length)
{
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	cuComplex* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_Samples_GPU0_pd;
	string fileName = "samples_GPU0_pd";
	cout << "writing " << fileName << endl;
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	cuComplex* host_array = new cuComplex[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		cuComplex* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBatch_samples1 (int length)
{
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	cuComplex* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_Samples_GPU1;
	string fileName = "samples_GPU1";
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	cuComplex* host_array = new cuComplex[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		cuComplex* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBatch_samples2 (int length)
{
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	cuComplex* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_Samples_GPU2;
	string fileName = "samples_GPU2";
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	cuComplex* host_array = new cuComplex[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		cuComplex* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBatch_channelEst (int length)
{
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	cuComplex* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_channelEst;
	string fileName = "channelEst";
	cout << "writing " << fileName << endl;
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	cuComplex* host_array = new cuComplex[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		cuComplex* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBatch_ZF (int length)
{
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	cuComplex* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_ZF_equalizers_GPU1;
	string fileName = "ZF";
	cout << "writing " << fileName << " Equalizers" << endl;
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	cuComplex* host_array = new cuComplex[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		cuComplex* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBatch_MMSE (int length)
{
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	cuComplex* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_MMSEequalizers_GPU2;
	string fileName = "MMSE";
	cout << "writing " << fileName << " Equalizers" << endl;
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	cuComplex* host_array = new cuComplex[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		cuComplex* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBatch_CMA (int length)
{
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	cuComplex* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_MMSE_CMA_equalizers_GPU0;
	string fileName = "CMA";
	cout << "writing " << fileName << " Equalizers" << endl;
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	cuComplex* host_array = new cuComplex[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		cuComplex* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBatch_downCMA (int length)
{
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	cuComplex* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_detected_downsampled_GPU0;
	string fileName = "downCMA";
	cout << "writing " << fileName << endl;
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	cuComplex* host_array = new cuComplex[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		cuComplex* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBatch_downZF (int length)
{
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	cuComplex* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_detected_downsampled_GPU1;
	string fileName = "downZF";
	cout << "writing " << fileName << endl;
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	cuComplex* host_array = new cuComplex[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		cuComplex* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBatch_downMMSE (int length)
{
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	cuComplex* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_detected_downsampled_GPU2;
	string fileName = "downMMSE";
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	cuComplex* host_array = new cuComplex[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		cuComplex* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBatch_downFDE1 (int length)
{
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	cuComplex* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_FDE1_detected_downsampled_rotated_GPU2;
	string fileName = "downFDE1";
	cout << "writing " << fileName << endl;
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	cuComplex* host_array = new cuComplex[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		cuComplex* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBatch_downFDE2(int length)
{
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	cuComplex* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_FDE2_detected_downsampled_rotated_GPU2;
	string fileName = "downCFDE2";
	cout << "writing " << fileName << endl;
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	cuComplex* host_array = new cuComplex[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		cuComplex* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBatch_ZFbits (int length)
{
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	unsigned char* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_ZF_bits_GPU0;
	string fileName = "ZFbits";
	cout << "writing " << fileName << endl;
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	unsigned char* host_array = new unsigned char[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(unsigned char));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		unsigned char* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(unsigned char));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBatch_MMSEbits (int length)
{
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	unsigned char* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_MMSE_bits_GPU0;
	string fileName = "MMSEbits";

	cout << "writing " << fileName << endl;
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	unsigned char* host_array = new unsigned char[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(unsigned char));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		unsigned char* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(unsigned char));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBatch_CMAbits (int length)
{
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	unsigned char* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_CMA_bits_GPU0;
	string fileName = "CMAbits";
	cout << "writing " << fileName << endl;
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	unsigned char* host_array = new unsigned char[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(unsigned char));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		unsigned char* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(unsigned char));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBatch_FDE1bits (int length)
{
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	unsigned char* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_FDE1_bits_GPU0;
	string fileName = "FDE1bits";
	cout << "writing " << fileName << endl;
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	unsigned char* host_array = new unsigned char[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(unsigned char));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		unsigned char* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(unsigned char));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBatch_FDE2bits (int length)
{
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	unsigned char* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_FDE2_bits_GPU0;
	string fileName = "FDE2bits";
	cout << "writing " << fileName << endl;
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	unsigned char* host_array = new unsigned char[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(unsigned char));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		unsigned char* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(unsigned char));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBatch_DAQsamples (int length)
{
	cudaSetDevice(device_GPU1);	cudaDeviceSynchronize();
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	unsigned short* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_ultraviewSamplesToHalf;
	string fileName = "DAQshortSamples";
	cout << "writing " << fileName << endl;
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	unsigned short* host_array = new unsigned short[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(unsigned short), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(unsigned short));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		unsigned short* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(unsigned short));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBatch_DAQsamplesLast34 (int length)
{
	cudaSetDevice(device_GPU1);	cudaDeviceSynchronize();
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	unsigned short* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_ultraviewLastIteration34;
	string fileName = "DAQshortSamplesLast34";
	cout << "writing " << fileName << endl;
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	unsigned short* host_array = new unsigned short[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(unsigned short), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(unsigned short));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		unsigned short* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(unsigned short));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBatch_DAQsamples_last_times (int length)
{
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	unsigned short* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_ultraviewSamplesToHalf_last_times;
	string fileName = "DAQshortSamples_last_times";
	cout << "writing " << fileName << endl;
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	unsigned short* host_array = new unsigned short[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(unsigned short), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(unsigned short));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		unsigned short* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(unsigned short));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBatch_DAQsamples_host (unsigned short* host_array,int length)
{
	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	string fileName = "DAQshortSamples_host";
	cout << "writing " << fileName << endl;
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(unsigned short));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		unsigned short* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(unsigned short));
	}
	out1.close();
}
void GPUHandler::writeBatch_HalfSamples (int length)
{
	cudaSetDevice(device_GPU1);	cudaDeviceSynchronize();
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	cuComplex* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_halfbandSamplesToPoly;
	string fileName = "half";
	cout << "writing " << fileName << endl;
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	cuComplex* host_array = new cuComplex[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		cuComplex* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBatch_HalfSamplesLast19 (int length)
{
	cudaSetDevice(device_GPU1);	cudaDeviceSynchronize();
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	cuComplex* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_halfbandSamplesToPolyLastIteration19;
	string fileName = "halfLast19";
	cout << "writing " << fileName << endl;
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	cuComplex* host_array = new cuComplex[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		cuComplex* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBatch_PolySamples (int length)
{
	cudaSetDevice(device_GPU1);	cudaDeviceSynchronize();
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	cuComplex* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_FIFOsamplesFromPolyphase;
	string fileName = "PolyLeftovers";
	cout << "writing " << fileName << endl;
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	cuComplex* host_array = new cuComplex[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		cuComplex* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBatch_FIFOSamples (int length)
{
	cudaSetDevice(device_GPU1);	cudaDeviceSynchronize();
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	cuComplex* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_FIFOsamplesFromPolyphase;
	string fileName = "fifo";
	cout << "writing " << fileName << endl;
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	cuComplex* host_array = new cuComplex[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		cuComplex* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBatch_PAQsamples (int length)
{
	cudaSetDevice(device_GPU1);	cudaDeviceSynchronize();
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	cuComplex* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_PAQcomplexSamplesFromPolyFIFO;
	string fileName = "PAQsamples";
	cout << "writing " << fileName << endl;
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	cuComplex* host_array = new cuComplex[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		cuComplex* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBatch_PAQsamples_last_times (int length)
{
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	cuComplex* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_PAQcomplexSamplesFromPolyFIFO_last_times;
	string fileName = "PAQsamples_last_times";
	cout << "writing " << fileName << endl;
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	cuComplex* host_array = new cuComplex[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		cuComplex* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBatch_PAQsamples_two_times (int length)
{
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	cuComplex* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_PAQcomplexSamplesFromPolyFIFO_two_times;
	string fileName = "PAQsamples_two_times";
	cout << "writing " << fileName << endl;
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	cuComplex* host_array = new cuComplex[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		cuComplex* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBatch_iFromPoly (int length)
{
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	float* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_iSamples_GPU1;
	string fileName = "iFromPoly";
	cout << "writing " << fileName << endl;
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	float* host_array = new float[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(float), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(float));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		float* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(float));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBatch_qFromPoly (int length)
{
	//----------------------------------------------
	//	Find and Replace the Date Type (float)
	//----------------------------------------------
	float* dev_array;

	//----------------------------------------------
	//	Only Change Stuff below here
	//----------------------------------------------
	dev_array = dev_qSamples_GPU1;
	string fileName = "qFromPoly";
	cout << "writing " << fileName << endl;
	//----------------------------------------------
	//	and above here
	//----------------------------------------------

	string filePath = SSTR( batchPath << fileName);
	const char* path = filePath.c_str();

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	float* host_array = new float[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(float), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(float));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		float* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(float));
	}
	out1.close();

	delete[] host_array;
}

void GPUHandler::writeBERT_bits (int length)
{
	unsigned char* dev_array;
	dev_array = dev_BERT_bits_pull_pointer_GPU0;
	const char* path = "/home/adm85/Testing/BERT2/BERT2TestFileToMatlab_bits";
	cout << "BERT2TestFileToMatlab_bits\n";


	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	unsigned char* host_array = new unsigned char[lengthOfCopy];

	// Copy the array to the host
	cudaMemcpy(host_array, dev_array, length * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.write((char *) &host_array[0], WRITE_CHUNK_LENGTH * sizeof(unsigned char));
	out.close();

	// Do the rest of the writes
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 1; i < numWrites; i++){
		unsigned char* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(unsigned char));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBERT_signed_bits (int length)
{
	cuComplex* dev_array;
	dev_array = dev_BERT_bits_GPU0;
	const char* path = "/home/adm85/Testing/BERT2/BERT2TestFileToMatlab_signed_bits";
	cout << "BERT2TestFileToMatlab_signed_bits\n";

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	cuComplex* host_array = new cuComplex[lengthOfCopy];

	// Copy the array to the host
	cudaSetDevice(device_GPU0);	cudaDeviceSynchronize();
	cudaMemcpy(host_array, dev_array, length * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	// Delete the current contents of the file then write the new file length
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.close();

	// Write the file in chunks
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 0; i < numWrites; i++){
		cuComplex* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBERT_BITS (int length)
{
	cuComplex* dev_array;
	dev_array = dev_BERT_bits_GPU0;
	const char* path = "/home/adm85/Testing/BERT2/BERT2TestFileToMatlab_BITS";
	cout << "BERT2TestFileToMatlab_BITS\n";

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	cuComplex* host_array = new cuComplex[lengthOfCopy];

	// Copy the array to the host
	cudaSetDevice(device_GPU0);	cudaDeviceSynchronize();
	cudaMemcpy(host_array, dev_array, length * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	// Delete the current contents of the file then write the new file length
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.close();

	// Write the file in chunks
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 0; i < numWrites; i++){
		cuComplex* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBERT_conj_MULTI (int length)
{
	cuComplex* dev_array;
	dev_array = dev_BERT_bits_GPU0;
	const char* path = "/home/adm85/Testing/BERT2/BERT2TestFileToMatlab_conj_MULTI";
	cout << "BERT2TestFileToMatlab_conj_MULTI\n";

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	cuComplex* host_array = new cuComplex[lengthOfCopy];

	// Copy the array to the host
	cudaSetDevice(device_GPU0);	cudaDeviceSynchronize();
	cudaMemcpy(host_array, dev_array, length * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	// Delete the current contents of the file then write the new file length
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.close();

	// Write the file in chunks
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 0; i < numWrites; i++){
		cuComplex* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBERT_multi (int length)
{
	cuComplex* dev_array;
	dev_array = dev_BERT_bits_GPU0;
	const char* path = "/home/adm85/Testing/BERT2/BERT2TestFileToMatlab_multi";
	cout << "BERT2TestFileToMatlab_multi\n";

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	cuComplex* host_array = new cuComplex[lengthOfCopy];

	// Copy the array to the host
	cudaSetDevice(device_GPU0);	cudaDeviceSynchronize();
	cudaMemcpy(host_array, dev_array, length * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	// Delete the current contents of the file then write the new file length
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.close();

	// Write the file in chunks
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 0; i < numWrites; i++){
		cuComplex* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(cuComplex));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBERT_correlated (int length)
{
	int* dev_array;
	dev_array = dev_BERT_xCorrelatedBits_GPU0;
	const char* path = "/home/adm85/Testing/BERT2/BERT2TestFileToMatlab_correlated";
	cout << "BERT2TestFileToMatlab_correlated\n";

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	int* host_array = new int[lengthOfCopy];

	// Copy the array to the host
	cudaSetDevice(device_GPU0);	cudaDeviceSynchronize();
	cudaMemcpy(host_array, dev_array, length * sizeof(int), cudaMemcpyDeviceToHost);

	// Delete the current contents of the file then write the new file length
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.close();

	// Write the file in chunks
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 0; i < numWrites; i++){
		int* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(int));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBERT_peaks (int length)
{
	int* dev_array;
	dev_array = dev_BERT_peaks_GPU0;
	const char* path = "/home/adm85/Testing/BERT2/BERT2TestFileToMatlab_peaks";
	cout << "BERT2TestFileToMatlab_peaks\n";

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	int* host_array = new int[lengthOfCopy];

	// Copy the array to the host
	cudaSetDevice(device_GPU0);	cudaDeviceSynchronize();
	cudaMemcpy(host_array, dev_array, length * sizeof(int), cudaMemcpyDeviceToHost);

	// Delete the current contents of the file then write the new file length
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.close();

	// Write the file in chunks
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 0; i < numWrites; i++){
		int* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(int));
	}
	out1.close();

	delete[] host_array;
}
void GPUHandler::writeBERT_peaksIdx (int length)
{
	int* dev_array;
	dev_array = dev_BERT_peaksIdx_GPU0;
	const char* path = "/home/adm85/Testing/BERT2/BERT2TestFileToMatlab_peaksIdx";
	cout << "BERT2TestFileToMatlab_peaksIdx\n";

	// Find the number of write chunks needed to write the whole array
	int numWrites = length/WRITE_CHUNK_LENGTH+1; // Run off the end a little bit

	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

	// Allocate space on the host
	int* host_array = new int[lengthOfCopy];

	// Copy the array to the host
	cudaSetDevice(device_GPU0);	cudaDeviceSynchronize();
	cudaMemcpy(host_array, dev_array, length * sizeof(int), cudaMemcpyDeviceToHost);

	// Delete the current contents of the file then write the new file length
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &length, sizeof(int));
	out.close();

	// Write the file in chunks
	ofstream out1(path, ios::out | ios::binary | ios::app); // Append each write on the end of the file
	for(int i = 0; i < numWrites; i++){
		int* temp_pointer = &host_array[i*WRITE_CHUNK_LENGTH];
		out1.write((char *) temp_pointer, WRITE_CHUNK_LENGTH * sizeof(int));
	}
	out1.close();

	delete[] host_array;
}

// Stuff that needs to be saved when there are errors
// Raw Samples  dev_qSamples_pd.dev_iSamples_pd
// Freq Est		dev_w0
// Samples		dev_Samples_GPU0
// Channel Est	dev_channelEst
// Equalizers	dev_MMSE_CMA_equalizers_GPU0	dev_ZF_equalizers_GPU1	dev_MMSEequalizers_GPU2
// bits			dev_ZF_bits_GPU0	dev_MMSE_bits_GPU0	dev_CMA_bits_GPU0	dev_FDE1_bits_GPU0	dev_FDE2_bits_GPU0



void GPUHandler::writeFreq_padded_h ()
{
	cudaSetDevice(2);	cudaDeviceSynchronize();	cudaSetDevice(1);	cudaDeviceSynchronize();	cudaSetDevice(0);	cudaDeviceSynchronize();
	cout << "CMATestFileToMatlab_padded_h\n";
	const int size = conv_length*75;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_FDE_H_padded_GPU2, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	ofstream out("/home/adm85/Testing/Freq/FreqTestFileToMatlab_padded_h", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writeFreq_padded_y ()
{
	cudaSetDevice(2);	cudaDeviceSynchronize();	cudaSetDevice(1);	cudaDeviceSynchronize();	cudaSetDevice(0);	cudaDeviceSynchronize();
	cout << "CMATestFileToMatlab_padded_y\n";
	const int size = conv_length*75;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_FDE_Y_padded_GPU2, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	ofstream out("/home/adm85/Testing/Freq/FreqTestFileToMatlab_padded_y", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));

	//	for(int i = 0; i<10; i++)
	//		cout << i << ":\t" << h_save[i].x << "\t" << h_save[i].y << endl;
}
void GPUHandler::writeFreq_padded_H ()
{
	cudaSetDevice(2);	cudaDeviceSynchronize();	cudaSetDevice(1);	cudaDeviceSynchronize();	cudaSetDevice(0);	cudaDeviceSynchronize();
	cout << "CMATestFileToMatlab_padded_H\n";
	const int size = conv_length*75;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_FDE_H_padded_GPU2, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	ofstream out("/home/adm85/Testing/Freq/FreqTestFileToMatlab_padded_H", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writeFreq_padded_Y ()
{
	cudaSetDevice(2);	cudaDeviceSynchronize();	cudaSetDevice(1);	cudaDeviceSynchronize();	cudaSetDevice(0);	cudaDeviceSynchronize();
	cout << "CMATestFileToMatlab_padded_Y\n";
	const int size = conv_length*75;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_FDE_Y_padded_GPU2, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	ofstream out("/home/adm85/Testing/Freq/FreqTestFileToMatlab_padded_Y", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writeFreq_padded_D ()
{
	cudaSetDevice(2);	cudaDeviceSynchronize();	cudaSetDevice(1);	cudaDeviceSynchronize();	cudaSetDevice(0);	cudaDeviceSynchronize();
	cout << "FreqTestFileToMatlab_padded_D\n";
	const int size = conv_length*75;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_filter_preDetection_GPU2, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	ofstream out("/home/adm85/Testing/Freq/FreqTestFileToMatlab_padded_D", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writeFreq_Quad ()
{
	cudaSetDevice(2);	cudaDeviceSynchronize();
	cout << "CMATestFileToMatlab_Quad\n";
	const int size = conv_length*75;

	cuComplex h_save[size];
	ofstream out("/home/adm85/Testing/Freq/FreqTestFileToMatlab_Quad", ios::out | ios::binary);
	cudaMemcpy(h_save, dev_FDE_Y_padded_GPU2, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	out.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writeFreq_quad ()
{
	cudaSetDevice(2);	cudaDeviceSynchronize();	cudaSetDevice(1);	cudaDeviceSynchronize();	cudaSetDevice(0);	cudaDeviceSynchronize();
	cout << "CMATestFileToMatlab_quad\n";
	const int size = conv_length*75;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_FDE_Y_padded_GPU2, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	ofstream out("/home/adm85/Testing/Freq/FreqTestFileToMatlab_quad", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writeFreq_prune_scale_downsample ()
{
	cudaSetDevice(2);	cudaDeviceSynchronize();
	cout << "FreqTestFileToMatlab_prune_scale_downsample\n";
	const int size = BITS_PER_PACKET*75;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_FDE1_detected_downsampled_rotated_GPU2, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	ofstream out("/home/adm85/Testing/Freq/FreqTestFileToMatlab_prune_scale_downsample", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writeFreq_prune_scale_downsample2 ()
{
	cudaSetDevice(2);	cudaDeviceSynchronize();
	cout << "FreqTestFileToMatlab_prune_scale_downsample2\n";
	const int size = BITS_PER_PACKET*75;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_FDE2_detected_downsampled_rotated_GPU2, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	ofstream out("/home/adm85/Testing/Freq/FreqTestFileToMatlab_prune_scale_downsample2", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writeFreq_prune_scale()
{
	cudaSetDevice(2);	cudaDeviceSynchronize();
	cout << "FreqTestFileToMatlab_prune_scale\n";
	const int size = SAMPLES_PER_PACKET/2*75;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_FDE_detected_downsampled_GPU2, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	ofstream out("/home/adm85/Testing/Freq/FreqTestFileToMatlab_prune_scale", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writeFreq_ahat ()
{
	cudaSetDevice(device_GPU0);	cudaDeviceSynchronize();
	cout << "FreqTestFileToMatlab_ahat\n";
	const int size = BITS_PER_PACKET*75;

	float h_save[size];
	cudaMemcpy(h_save, dev_FDE1_ahat_GPU2, size * sizeof(float), cudaMemcpyDeviceToHost);
	ofstream out("/home/adm85/Testing/Freq/FreqTestFileToMatlab_ahat", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(float));

	//for(int i = 0; i < 10; i++)
	//	cout << i << ":\t" << static_cast<unsigned>(h_save[i])*2-1 << endl;
}
void GPUHandler::writeFreq_ahat2 ()
{
	cudaSetDevice(device_GPU0);	cudaDeviceSynchronize();
	cout << "FreqTestFileToMatlab_ahat2\n";
	const int size = BITS_PER_PACKET*75;

	float h_save[size];
	cudaMemcpy(h_save, dev_FDE2_ahat_GPU2, size * sizeof(float), cudaMemcpyDeviceToHost);
	ofstream out("/home/adm85/Testing/Freq/FreqTestFileToMatlab_ahat2", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(float));

	//for(int i = 0; i < 10; i++)
	//	cout << i << ":\t" << static_cast<unsigned>(h_save[i])*2-1 << endl;
}
void GPUHandler::writeFreq_bits ()
{
	cudaSetDevice(2);	cudaDeviceSynchronize();
	cout << "FreqTestFileToMatlab_bits\n";
	const int size = BITS_PER_DATA_PACKET*75;

	unsigned char h_save[size];
	cudaMemcpy(h_save, dev_FDE1_bits_GPU2, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	ofstream out("/home/adm85/Testing/Freq/FreqTestFileToMatlab_bits", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(unsigned char));
}
void GPUHandler::writeFreq_bits2 ()
{
	cudaSetDevice(0);	cudaDeviceSynchronize();
	cout << "FreqTestFileToMatlab_bits2\n";
	const int size = BITS_PER_DATA_PACKET*75;

	unsigned char h_save[size];
	cudaMemcpy(h_save, dev_FDE2_bits_GPU0, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	ofstream out("/home/adm85/Testing/Freq/FreqTestFileToMatlab_bits2", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(unsigned char));
}
void GPUHandler::writeFreq_shs ()
{
	cout << "Writing SH^2\n";
	cudaSetDevice(2);	cudaDeviceSynchronize();
	cuComplex h_save[1];
	cudaMemcpy(h_save, dev_shs_GPU2, sizeof(float), cudaMemcpyDeviceToHost);

	ofstream out("/home/adm85/Testing/Freq/FreqTestFileToMatlab_shs", ios::out | ios::binary);
	out.write((char *) &h_save, sizeof(float));
}
void GPUHandler::writeFreq_Samples ()
{
	cudaSetDevice(2);	cudaDeviceSynchronize();
	cout << "CMATestFileToMatlab_Samples\n";
	const int size = SAMPLES_PER_PACKET*75;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_Samples_GPU2, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	ofstream out("/home/adm85/Testing/Freq/FreqTestFileToMatlab_Samples", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writeFreq_channel ()
{
	cudaSetDevice(2);	cudaDeviceSynchronize();
	cout << "Writing Channel\n";
	const int size = CHAN_SIZE*75;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_h_hat_freq_GPU2, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	ofstream out("/home/adm85/Testing/Freq/freqTestToMatlab_channel", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writeCMA_equalizer ()
{
	cudaSetDevice(2);	cudaDeviceSynchronize();	cudaSetDevice(1);	cudaDeviceSynchronize();	cudaSetDevice(0);	cudaDeviceSynchronize();
	cout << "CMATestFileToMatlab_equalizer\n";
	const int size = 186*75;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_MMSE_CMA_equalizers_GPU0, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	for(int i = 0; i < 10; i++)
		cout << "\t" << h_save[i].x << " + " << h_save[i].y << "i\n";
	ofstream out("/home/adm85/Testing/CMA/CMATestFileToMatlab_equalizer", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writeCMA_delJ ()
{
	cudaSetDevice(2);	cudaDeviceSynchronize();	cudaSetDevice(1);	cudaDeviceSynchronize();	cudaSetDevice(0);	cudaDeviceSynchronize();
	cout << "CMATestFileToMatlab_delJ\n";
	const int size = 186*75;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_delJ_fft_GPU0, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	ofstream out("/home/adm85/Testing/CMA/CMATestFileToMatlab_delJ", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writeCMA_multi ()
{cudaSetDevice(2);	cudaDeviceSynchronize();	cudaSetDevice(1);	cudaDeviceSynchronize();	cudaSetDevice(0);	cudaDeviceSynchronize();
cout << "CMATestFileToMatlab_multi\n";
const int size = CMA_FFT_LENGTH*75;

cuComplex h_save[size];
cudaMemcpy(h_save, dev_z_flipped_padded_GPU0, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);
ofstream out("/home/adm85/Testing/CMA/CMATestFileToMatlab_multi", ios::out | ios::binary);
out.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writeCMA_ifft ()
{cudaSetDevice(2);	cudaDeviceSynchronize();	cudaSetDevice(1);	cudaDeviceSynchronize();	cudaSetDevice(0);	cudaDeviceSynchronize();
cout << "CMATestFileToMatlab_ifft\n";
const int size = CMA_FFT_LENGTH*75;

cuComplex h_save[size];
cudaMemcpy(h_save, dev_z_flipped_padded_GPU0, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);
ofstream out("/home/adm85/Testing/CMA/CMATestFileToMatlab_ifft", ios::out | ios::binary);
out.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writeCMA_r_fft ()
{cudaSetDevice(2);	cudaDeviceSynchronize();	cudaSetDevice(1);	cudaDeviceSynchronize();	cudaSetDevice(0);	cudaDeviceSynchronize();
cout << "CMATestFileToMatlab_r_fft\n";
const int size = CMA_FFT_LENGTH*75;

cuComplex h_save[size];
cudaMemcpy(h_save, dev_x_padded_GPU0, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);
ofstream out("/home/adm85/Testing/CMA/CMATestFileToMatlab_r_fft", ios::out | ios::binary);
out.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writeCMA_e_fft ()
{cudaSetDevice(2);	cudaDeviceSynchronize();	cudaSetDevice(1);	cudaDeviceSynchronize();	cudaSetDevice(0);	cudaDeviceSynchronize();
cout << "CMATestFileToMatlab_e_fft\n";
const int size = CMA_FFT_LENGTH*75;

cuComplex h_save[size];
cudaMemcpy(h_save, dev_z_flipped_padded_GPU0, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);
ofstream out("/home/adm85/Testing/CMA/CMATestFileToMatlab_e_fft", ios::out | ios::binary);
out.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writeCMA_e_flipped ()
{cudaSetDevice(2);	cudaDeviceSynchronize();	cudaSetDevice(1);	cudaDeviceSynchronize();	cudaSetDevice(0);	cudaDeviceSynchronize();
cout << "CMATestFileToMatlab_e_flipped\n";
const int size = SAMPLES_PER_PACKET*75;

cuComplex h_save[size];
cudaMemcpy(h_save, dev_z_flipped_GPU0, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);
ofstream out("/home/adm85/Testing/CMA/CMATestFileToMatlab_e_flipped", ios::out | ios::binary);
out.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writeCMA_Samples ()
{
	cudaSetDevice(2);	cudaDeviceSynchronize();	cudaSetDevice(1);	cudaDeviceSynchronize();	cudaSetDevice(0);	cudaDeviceSynchronize();
	cout << "CMATestFileToMatlab_Samples\n";
	const int size = SAMPLES_PER_PACKET*75;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_Samples_GPU0, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	ofstream out("/home/adm85/Testing/CMA/CMATestFileToMatlab_Samples", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writeCMA_y ()
{
	cudaSetDevice(2);	cudaDeviceSynchronize();	cudaSetDevice(1);	cudaDeviceSynchronize();	cudaSetDevice(0);	cudaDeviceSynchronize();
	cout << "CMATestFileToMatlamb_y\n";
	const int size = SAMPLES_PER_PACKET*75;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_y_GPU0, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	//	for(int i = 0; i < 10; i++)
	//		cout << "\t" << h_save[i].x << " + " << h_save[i].y << "i\n";
	ofstream out("/home/adm85/Testing/CMA/CMATestFileToMatlab_y", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writeCMA_e ()
{cudaSetDevice(2);	cudaDeviceSynchronize();	cudaSetDevice(1);	cudaDeviceSynchronize();	cudaSetDevice(0);	cudaDeviceSynchronize();
cout << "CMATestFileToMatlab_e\n";
const int size = SAMPLES_PER_PACKET * 75;

cuComplex h_save[size];
cudaMemcpy(h_save, dev_z_GPU0, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);

ofstream out("/home/adm85/Testing/CMA/CMATestToMatlab_e", ios::out | ios::binary);
out.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writeDemodulatedBits ()
{
	cout << "write Demodulated Bits\n";
	const int size = BITS_PER_PACKET*75;

	float h_save[size];
	cudaMemcpy(h_save, dev_CMA_ahat_GPU0, size * sizeof(float), cudaMemcpyDeviceToHost);
	ofstream out("/home/adm85/Testing/DmodTesting/dmodTestFileToMatlab_Ahat_mmse", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(float));

	cudaMemcpy(h_save, dev_ZF_ahat_GPU1, sizeof(float) * size, cudaMemcpyDeviceToHost);
	ofstream out1("/home/adm85/Testing/DmodTesting/dmodTestFileToMatlab_Ahat_zf", ios::out | ios::binary);
	out1.write((char *) &h_save, size * sizeof(float));
}
void GPUHandler::writeStrippedBits ()
{
	cout << "write Stripped Bits\n";
	const int size = BITS_PER_DATA_PACKET*75;

	char h_save[size];
	cudaMemcpy(h_save, dev_MMSE_bits_GPU0, size * sizeof(char), cudaMemcpyDeviceToHost);
	ofstream out("/home/adm85/Testing/DmodTesting/dmodTestFileToMatlab_StrippedBits_mmse", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(char));

	cudaMemcpy(h_save, dev_MMSE_bits_GPU0, sizeof(char) * size, cudaMemcpyDeviceToHost);
	ofstream out1("/home/adm85/Testing/DmodTesting/dmodTestFileToMatlab_StrippedBits_zf", ios::out | ios::binary);
	out1.write((char *) &h_save, size * sizeof(char));
}
void GPUHandler::writeChannelBits ()
{
	cout << "write Stripped Bits\n";
	const int size = BITS_PER_DATA_PACKET*75;

	char h_save[size];
	cudaMemcpy(h_save, dev_all8Channels_bits, size * sizeof(char), cudaMemcpyDeviceToHost);
	ofstream out("/home/adm85/Testing/DmodTesting/dmodTestFileToMatlab_ChannelBits", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(char));
}
void GPUHandler::writeDownSampled ()
{
	cout << "Writing Down Sampled\n";
	const int size = SAMPLES_PER_PACKET*75;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_detected_downsampled_GPU0, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	ofstream out("/home/adm85/Testing/DmodTesting/dmodTestFileToMatlab_downSampled_mmse", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));


	cudaMemcpy(h_save, dev_detected_downsampled_GPU1, sizeof(cuComplex) * size, cudaMemcpyDeviceToHost);

	ofstream out1("/home/adm85/Testing/DmodTesting/dmodTestFileToMatlab_downSampled_zf", ios::out | ios::binary);
	out1.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writePreScaled ()
{
	cout << "Writing Prescaled\n";
	const int size = conv_length*75;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_Samples_padded_GPU0, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	ofstream out("/home/adm85/Testing/ConvTesting/convTestFileToMatlab_PreScaled_mmse", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));

	cudaMemcpy(h_save, dev_Samples_padded_GPU1, sizeof(cuComplex) * size, cudaMemcpyDeviceToHost);
	ofstream out1("/home/adm85/Testing/ConvTesting/convTestFileToMatlab_PreScaled_zf", ios::out | ios::binary);
	out1.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writeIFFT ()
{
	cout << "Writing IFFT\n";
	const int size = conv_length*75;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_Samples_padded_GPU0, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	ofstream out("/home/adm85/Testing/ConvTesting/convTestFileToMatlab_IFFT_mmse", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));

	cudaMemcpy(h_save, dev_Samples_padded_GPU1, sizeof(cuComplex) * size, cudaMemcpyDeviceToHost);
	ofstream out1("/home/adm85/Testing/ConvTesting/convTestFileToMatlab_IFFT_zf", ios::out | ios::binary);
	out1.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writeMultiply ()
{
	cout << "Writing Multiply\n";
	const int size = conv_length*75;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_Samples_padded_GPU0, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	ofstream out("/home/adm85/Testing/ConvTesting/convTestFileToMatlab_Multiply_mmse", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));

	cudaMemcpy(h_save, dev_Samples_padded_GPU1, sizeof(cuComplex) * size, cudaMemcpyDeviceToHost);
	ofstream out1("/home/adm85/Testing/ConvTesting/convTestFileToMatlab_Multiply_zf", ios::out | ios::binary);
	out1.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writeEqualizerPadded ()
{
	cout << "Writing EqualizerPadded\n";
	const int size = conv_length*75;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_equalizers_padded_GPU0, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	ofstream out("/home/adm85/Testing/ConvTesting/convTestFileToMatlab_EqualizerPaddedT_mmse", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));

	cudaMemcpy(h_save, dev_equalizers_padded_GPU1, sizeof(cuComplex) * size, cudaMemcpyDeviceToHost);
	ofstream out1("/home/adm85/Testing/ConvTesting/convTestFileToMatlab_EqualizerPaddedT_zf", ios::out | ios::binary);
	out1.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writeSignalPadded ()
{
	cout << "Writing EqualizerFFT\n";
	const int size = conv_length*75;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_Samples_padded_GPU0, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	ofstream out("/home/adm85/Testing/ConvTesting/convTestFileToMatlab_SignalPadded_mmse", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));

	cudaMemcpy(h_save, dev_Samples_padded_GPU1, sizeof(cuComplex) * size, cudaMemcpyDeviceToHost);
	ofstream out1("/home/adm85/Testing/ConvTesting/convTestFileToMatlab_SignalPadded_zf", ios::out | ios::binary);
	out1.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writeEqualizerFFT ()
{
	cout << "Writing EqualizerFFT\n";
	const int size = conv_length*75;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_equalizers_padded_GPU0, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	ofstream out("/home/adm85/Testing/ConvTesting/convTestFileToMatlab_EqualizerFFT_mmse", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));

	cudaMemcpy(h_save, dev_equalizers_padded_GPU1, sizeof(cuComplex) * size, cudaMemcpyDeviceToHost);
	ofstream out1("/home/adm85/Testing/ConvTesting/convTestFileToMatlab_EqualizerFFT_zf", ios::out | ios::binary);
	out1.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writeSignalFFT ()
{
	cout << "Writing SignalFFT\n";
	const int size = conv_length*75;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_Samples_padded_GPU0, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	ofstream out("/home/adm85/Testing/ConvTesting/convTestFileToMatlab_SignalFFT_mmse", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));

	cudaMemcpy(h_save, dev_Samples_padded_GPU1, sizeof(cuComplex) * size, cudaMemcpyDeviceToHost);
	ofstream out1("/home/adm85/Testing/ConvTesting/convTestFileToMatlab_SignalFFT_zf", ios::out | ios::binary);
	out1.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writeEqualizedSignal ()
{
	cout << "Writing Filtered Signal\n";
	const int size = SAMPLES_PER_PACKET*75;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_Samples_GPU0, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	ofstream out("/home/adm85/Testing/ConvTesting/convTestFileToMatlab_EqualizedSignal_mmse", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));

	cudaMemcpy(h_save, dev_Samples_GPU1, sizeof(cuComplex) * size, cudaMemcpyDeviceToHost);
	ofstream out1("/home/adm85/Testing/ConvTesting/convTestFileToMatlab_EqualizedSignal_zf", ios::out | ios::binary);
	out1.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writeDmodPadded ()
{
	cout << "Writing Dmod Padded\n";
	const int size = conv_length*75;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_signal_preDetection_GPU0, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	ofstream out("/home/adm85/Testing/DmodTesting/dmodTestFileToMatlab_DmodPadded_mmse", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));

	cudaMemcpy(h_save, dev_signal_preDetection_GPU1, sizeof(cuComplex) * size, cudaMemcpyDeviceToHost);
	ofstream out1("/home/adm85/Testing/DmodTesting/dmodTestFileToMatlab_DmodPadded_zf", ios::out | ios::binary);
	out1.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writeHhat ()
{
	cout << "Writing hHat\n";
	const int size = CHAN_SIZE * 3103;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_h_hat_mmse_GPU0, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	ofstream out("/home/adm85/Testing/ZFtesting/zfTestFileToMatlab_h_hat", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writeSHS()
{
	cout << "Writing SH^2\n";
	const int size = MAX_PACKETS_PER_MEMORY_SECTION;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_noiseVariance_GPU0, size * sizeof(float), cudaMemcpyDeviceToHost);

	ofstream out("/home/adm85/Testing/ZFtesting/zfTestFileToMatlab_shs", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(float));
}
void GPUHandler::writeR1 ()
{
	cout << "Writing r1\n";
	const int size = r1_length*75;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_r1, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	ofstream out("/home/adm85/Testing/FreqencyOffsetRotate/freqTestToMatlab_R1", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));

	cudaMemcpy(h_save, dev_r1_conj, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	ofstream out1("/home/adm85/Testing/FreqencyOffsetRotate/freqTestToMatlab_R1Conj", ios::out | ios::binary);
	out1.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writeFreqDot ()
{
	cout << "Writing FreqDot\n";
	const int size = 75;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_complex_w0, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	ofstream out("/home/adm85/Testing/FreqencyOffsetRotate/freqTestToMatlab_FreqDot", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writeW0 ()
{
	cout << "Writing W0\n";
	const int size = MAX_PACKETS_PER_MEMORY_SECTION;

	float h_save[size];
	cudaMemcpy(h_save, dev_w0, size * sizeof(float), cudaMemcpyDeviceToHost);

	ofstream out("/home/adm85/Testing/FreqencyOffsetRotate/freqTestToMatlab_W0", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(float));
}
void GPUHandler::writeDerotated ()
{
	cout << "Writing Derotated\n";
	const int size = SAMPLES_PER_PACKET*75;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_Samples_GPU0, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	ofstream out("/home/adm85/Testing/FreqencyOffsetRotate/freqTestToMatlab_Derotated", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writeR2 ()
{
	cout << "Writing r2\n";
	const int size = r2_length*75;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_r2, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	ofstream out("/home/adm85/Testing/ChannelEst/channelTestToMatlab_R2", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writeChannelEst ()
{
	cout << "Writing Channel\n";
	const int size = channelEst_m*75;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_channelEst, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	ofstream out("/home/adm85/Testing/ChannelEst/channelTestToMatlab_ChannelEst", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::runningWriteChannelEst ()
{
	//	cout << "Writing Channel\n";
	const int size = channelEst_m;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_channelEst, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	ofstream out("/home/adm85/Testing/ChannelEst/channelTestToMatlab_RunningChannelEst", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writenoiseMultiplyIntermediate ()
{
	cout << "Writing Noise Multiply Intermediate\n";
	const int size = noiseMultiply_m*75;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_noiseMultiplyIntermediate, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	ofstream out("/home/adm85/Testing/NoiseVar/noiseVarTestToMatlab_noiseMultiplyIntermediate", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writeNoiseSubAndSquare ()
{
	cout << "Writing Noise Sub and Square\n";
	const int size = noiseMultiply_m*75;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_diffMag2, size * sizeof(float), cudaMemcpyDeviceToHost);

	ofstream out("/home/adm85/Testing/NoiseVar/noiseVarTestToMatlab_NoiseSubAndSquare", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(float));
}
void GPUHandler::writeNoise_myX ()
{
	cout << "Writing Noise_myX\n";
	const int size = noiseMultiply_X_length*75;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_noiseMultiply_X, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	ofstream out("/home/adm85/Testing/NoiseVar/noiseVarTestToMatlab_X", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writeNoiseVariance ()
{
	cout << "Writing Noise Variance\n";
	const int size = MAX_PACKETS_PER_MEMORY_SECTION;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_noiseVariance_GPU0, size * sizeof(float), cudaMemcpyDeviceToHost);

	ofstream out("/home/adm85/Testing/NoiseVar/noiseVarTestToMatlab_NoiseVariance", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(float));
}
void GPUHandler::writeEqualizers ()
{
	cout << "Writing Equalizers\n";
	const int size = m*batchSize;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_ZF_equalizers_GPU1, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	ofstream out("/home/adm85/Testing/CalcEqualizers/calcEqualizersTestToMatlab_ZFequalizers", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));

	cudaMemcpy(h_save, dev_MMSEequalizers_GPU2, sizeof(cuComplex) * size, cudaMemcpyDeviceToHost);
	for(int i = 0; i < 10; i++)
		cout << "\t" << h_save[i].x << " + " << h_save[i].y << "i\n";
	ofstream out1("/home/adm85/Testing/CMA/CMATestFileToMatlab_MMSEequalizers", ios::out | ios::binary);
	//	ofstream out1("/home/adm85/Testing/CalcEqualizers/calcEqualizersTestToMatlab_MMSEequalizers", ios::out | ios::binary);
	out1.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::runningWriteEqualizers ()
{
	//	cout << "Writing Running Equalizers\n";
	const int size = m;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_ZF_equalizers_GPU1, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	ofstream out("/home/adm85/Testing/CalcEqualizers/calcEqualizersTestToMatlab_RunningZFequalizers", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));

	cudaMemcpy(h_save, dev_MMSE_CMA_equalizers_GPU0, sizeof(cuComplex) * size, cudaMemcpyDeviceToHost);

	ofstream out1("/home/adm85/Testing/CalcEqualizers/calcEqualizersTestToMatlab_RunningMMSEequalizers", ios::out | ios::binary);
	out1.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writeStrippedSignal ()
{
	cout << "Writing Stripped Signal\n";
	const int size = SAMPLES_PER_PACKET*75;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_Samples_GPU0, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	ofstream out("/home/adm85/Testing/Signal/pdTestToMatlab_strippedSignal", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writeL ()
{
	cout << "Writing L\n";
	const int size = SAMPLES_PER_PACKET*75;
	float h_save[size];
	cudaMemcpy(h_save, dev_L_pd, size * sizeof(float), cudaMemcpyDeviceToHost);
	ofstream out("/home/adm85/Testing/PreambleDetector/pdTestToMatlab_L", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(float));
}
void GPUHandler::writeMaxLocations ()
{
	cout << "Writing Max Locations\n";
	const int size = (MAX_PACKETS_PER_MEMORY_SECTION+1);
	int h_save[size];
	cudaMemcpy(h_save, dev_max_locations, size * sizeof(int), cudaMemcpyDeviceToHost);
	ofstream out("/home/adm85/Testing/PreambleDetector/pdTestToMatlab_MaxLocations", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(int));
}
void GPUHandler::writeMaxLocationslazy ()
{
	cout << "Writing Max Locations Lazy\n";
	const int size = (MAX_PACKETS_PER_MEMORY_SECTION+1);
	int h_save[size];
	cudaMemcpy(h_save, max_locations, size * sizeof(int), cudaMemcpyHostToHost);
	ofstream out("/home/adm85/Testing/PreambleDetector/pdTestToMatlab_MaxLocationslazy", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(int));
}
void GPUHandler::writePoly_i_little_front (int size, int numWrite)
{
	cudaSetDevice(2);	cudaDeviceSynchronize();	cudaSetDevice(1);	cudaDeviceSynchronize();	cudaSetDevice(0);	cudaDeviceSynchronize();
	cudaSetDevice(device_GPU1);

	string fileName;          // string which will contain the result

	ostringstream convert;   // stream used for the conversion

	convert << "/home/adm85/git/JeffPaq/UltraSetup/i_front" << numWrite;      // insert the textual representation of 'Number' in the characters in the stream

	fileName = convert.str();

	const char* path = fileName.c_str();
	cout << "Writing " << fileName << endl;

	float h_save[size];
	cudaMemcpy(h_save, dev_iSamples_GPU1, size * sizeof(float), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &size, sizeof(int));
	out.write((char *) &h_save[0], size * sizeof(float));
	out.close();
}
void GPUHandler::writePoly_q_little_front (int size, int numWrite)
{
	cudaSetDevice(2);	cudaDeviceSynchronize();	cudaSetDevice(1);	cudaDeviceSynchronize();	cudaSetDevice(0);	cudaDeviceSynchronize();
	cudaSetDevice(device_GPU1);

	string fileName;          // string which will contain the result

	ostringstream convert;   // stream used for the conversion

	convert << "/home/adm85/git/JeffPaq/UltraSetup/q_front" << numWrite;      // insert the textual representation of 'Number' in the characters in the stream

	fileName = convert.str();

	const char* path = fileName.c_str();
	cout << "Writing " << fileName << endl;

	float h_save[size];
	cudaMemcpy(h_save, dev_qSamples_GPU1, size * sizeof(float), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &size, sizeof(int));
	out.write((char *) &h_save[0], size * sizeof(float));
	out.close();
}
void GPUHandler::writePoly_i_little_back (int size, int numWrite)
{
	cudaSetDevice(2);	cudaDeviceSynchronize();	cudaSetDevice(1);	cudaDeviceSynchronize();	cudaSetDevice(0);	cudaDeviceSynchronize();
	cudaSetDevice(device_GPU1);

	string fileName;          // string which will contain the result

	ostringstream convert;   // stream used for the conversion

	convert << "/home/adm85/git/JeffPaq/UltraSetup/i_back" << numWrite;      // insert the textual representation of 'Number' in the characters in the stream

	fileName = convert.str();

	const char* path = fileName.c_str();
	cout << "Writing " << fileName << endl;

	float h_save[size];
	cudaMemcpy(h_save, &dev_iSamples_GPU1[NUM_INPUT_SAMPLES_DEFAULT-size], size * sizeof(float), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &size, sizeof(int));
	out.write((char *) &h_save[0], size * sizeof(float));
	out.close();
}
void GPUHandler::writePoly_q_little_back (int size, int numWrite)
{
	cudaSetDevice(2);	cudaDeviceSynchronize();	cudaSetDevice(1);	cudaDeviceSynchronize();	cudaSetDevice(0);	cudaDeviceSynchronize();
	cudaSetDevice(device_GPU1);

	string fileName;          // string which will contain the result

	ostringstream convert;   // stream used for the conversion

	convert << "/home/adm85/git/JeffPaq/UltraSetup/q_back" << numWrite;      // insert the textual representation of 'Number' in the characters in the stream

	fileName = convert.str();

	const char* path = fileName.c_str();
	cout << "Writing " << fileName << endl;

	float h_save[size];
	cudaMemcpy(h_save, &dev_qSamples_GPU1[NUM_INPUT_SAMPLES_DEFAULT-size], size * sizeof(float), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &size, sizeof(int));
	out.write((char *) &h_save[0], size * sizeof(float));
	out.close();
}
void GPUHandler::writePoly_z ()
{
	cudaSetDevice(2);	cudaDeviceSynchronize();	cudaSetDevice(1);	cudaDeviceSynchronize();	cudaSetDevice(0);	cudaDeviceSynchronize();
	cout << "PolyTestFileToMatlab_z\n";
	const int size = L_polyphase;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_z_p, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	ofstream out("/home/adm85/Testing/PolyPhase/PolyTestFileToMatlab_z", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writePoly_z_p ()
{
	cudaSetDevice(2);	cudaDeviceSynchronize();	cudaSetDevice(1);	cudaDeviceSynchronize();	cudaSetDevice(0);	cudaDeviceSynchronize();
	cout << "PolyTestFileToMatlab_z_p\n";
	const int size = L_polyphase;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_z_p, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	ofstream out("/home/adm85/Testing/PolyPhase/PolyTestFileToMatlab_z_p", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writePoly_z_p_little_old (int size, int numWrite)
{
	cudaSetDevice(2);	cudaDeviceSynchronize();	cudaSetDevice(1);	cudaDeviceSynchronize();	cudaSetDevice(0);	cudaDeviceSynchronize();
	cudaSetDevice(device_GPU1);

	string fileName;          // string which will contain the result

	ostringstream convert;   // stream used for the conversion

	convert << "/home/adm85/git/JeffPaq/UltraSetup/z_p_old" << numWrite;      // insert the textual representation of 'Number' in the characters in the stream

	fileName = convert.str();

	const char* path = fileName.c_str();
	cout << "Writing " << fileName << endl;

	cuComplex h_save[size];
	cudaMemcpy(h_save, &dev_z_p[NUM_INPUT_SAMPLES_DEFAULT-size], size * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &size, sizeof(int));
	out.write((char *) &h_save[0], size * sizeof(cuComplex));
	out.close();
}
void GPUHandler::writePoly_z_p_little_new (int size, int numWrite)
{
	cudaSetDevice(2);	cudaDeviceSynchronize();	cudaSetDevice(1);	cudaDeviceSynchronize();	cudaSetDevice(0);	cudaDeviceSynchronize();
	cudaSetDevice(device_GPU1);

	string fileName;          // string which will contain the result

	ostringstream convert;   // stream used for the conversion

	convert << "/home/adm85/git/JeffPaq/UltraSetup/z_p_new" << numWrite;      // insert the textual representation of 'Number' in the characters in the stream

	fileName = convert.str();

	const char* path = fileName.c_str();
	cout << "Writing " << fileName << endl;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_z_p, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &size, sizeof(int));
	out.write((char *) &h_save[0], size * sizeof(cuComplex));
	out.close();
}
void GPUHandler::writePoly_z_p_little_push (int size, int numWrite)
{
	cudaSetDevice(2);	cudaDeviceSynchronize();	cudaSetDevice(1);	cudaDeviceSynchronize();	cudaSetDevice(0);	cudaDeviceSynchronize();
	cudaSetDevice(device_GPU1);

	string fileName;          // string which will contain the result

	ostringstream convert;   // stream used for the conversion

	convert << "/home/adm85/git/JeffPaq/UltraSetup/z_p_push" << numWrite;      // insert the textual representation of 'Number' in the characters in the stream

	fileName = convert.str();

	const char* path = fileName.c_str();
	cout << "Writing " << fileName << endl;

	cuComplex h_save[size];
	cudaMemcpy(h_save, &dev_z_p_push[-size/2], size * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &size, sizeof(int));
	out.write((char *) &h_save[0], size * sizeof(cuComplex));
	out.close();
}
void GPUHandler::writePoly_y ()
{
	cudaSetDevice(2);	cudaDeviceSynchronize();	cudaSetDevice(1);	cudaDeviceSynchronize();	cudaSetDevice(0);	cudaDeviceSynchronize();
	cout << "PolyTestFileToMatlab_y\n";
	const int size = L_polyphase;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_y_p_new, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	ofstream out("/home/adm85/Testing/PolyPhase/PolyTestFileToMatlab_y", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writePoly_y_p ()
{
	cudaSetDevice(2);	cudaDeviceSynchronize();	cudaSetDevice(1);	cudaDeviceSynchronize();	cudaSetDevice(0);	cudaDeviceSynchronize();
	cudaSetDevice(device_GPU1);
	cout << "PolyTestFileToMatlab_y_p\n";
	const int size = L_polyphase;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_y_p_new, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	ofstream out("/home/adm85/Testing/PolyPhase/PolyTestFileToMatlab_y_p", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(cuComplex));
}
void GPUHandler::writePoly_y_p_little (int size, int numWrite)
{
	cudaSetDevice(2);	cudaDeviceSynchronize();	cudaSetDevice(1);	cudaDeviceSynchronize();	cudaSetDevice(0);	cudaDeviceSynchronize();
	cudaSetDevice(device_GPU1);

	string fileName;          // string which will contain the result

	ostringstream convert;   // stream used for the conversion

	convert << "/home/adm85/git/JeffPaq/UltraSetup/y_p" << numWrite;      // insert the textual representation of 'Number' in the characters in the stream

	fileName = convert.str();

	const char* path = fileName.c_str();
	cout << "Writing " << fileName << endl;

	cuComplex h_save[size];
	cudaMemcpy(h_save, dev_y_p_old, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &size, sizeof(int));
	out.write((char *) &h_save[0], size * sizeof(cuComplex));
	out.close();
}
void GPUHandler::writePoly_DAQ_float_little (int size, int numWrite)
{
	cudaSetDevice(2);	cudaDeviceSynchronize();	cudaSetDevice(1);	cudaDeviceSynchronize();	cudaSetDevice(0);	cudaDeviceSynchronize();
	cudaSetDevice(device_GPU1);

	string fileName;          // string which will contain the result

	ostringstream convert;   // stream used for the conversion

	convert << "/home/adm85/git/JeffPaq/UltraSetup/DAQ_float" << numWrite;      // insert the textual representation of 'Number' in the characters in the stream

	fileName = convert.str();

	const char* path = fileName.c_str();
	cout << "Writing " << fileName << endl;

	float h_save[size];
	cudaMemcpy(h_save, dev_x_float_old, size * sizeof(float), cudaMemcpyDeviceToHost);

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &size, sizeof(int));
	out.write((char *) &h_save[0], size * sizeof(float));
	out.close();
}
void GPUHandler::writeUnFiltered ()
{
	cout << "Writing UnFiltered Signal\n";
	const int size = SAMPLES_PER_PACKET*100;

	float h_save[size];
	cudaMemcpy(h_save, dev_iSamples_pd, size * sizeof(float), cudaMemcpyDeviceToHost);

	ofstream out("/home/adm85/Testing/Signal/pdTestToMatlab_unfilteredSignal_i", ios::out | ios::binary);
	out.write((char *) &h_save, size * sizeof(float));

	cudaMemcpy(h_save, dev_qSamples_pd, size * sizeof(float), cudaMemcpyDeviceToHost);

	ofstream out1("/home/adm85/Testing/Signal/pdTestToMatlab_unfilteredSignal_q", ios::out | ios::binary);
	out1.write((char *) &h_save, size * sizeof(float));
}

void GPUHandler::TimingReset()
{
	cout << endl << "Timing Reset." << endl << endl;
	wcet = 0;
	bcet = 10000.0;
	Ultrawcet = 0;
	Ultrabcet = 10000.0;
}
void GPUHandler::StartTiming()
{
	cudaSetDevice(2);
	cudaDeviceSynchronize();
	cudaSetDevice(1);
	cudaDeviceSynchronize();
	cudaSetDevice(0);
	cudaDeviceSynchronize();
	//	cout << "Started Timing" << endl;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0 );
}
void GPUHandler::StartTimingKernel()
{
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0 );
}
void GPUHandler::StopTiming()
{
	cudaSetDevice(2);
	cudaDeviceSynchronize();
	cudaSetDevice(1);
	cudaDeviceSynchronize();
	cudaSetDevice(0);
	cudaDeviceSynchronize();
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &myTime, start, stop );
	if(myTime > wcet){
		wcet = myTime;
	}
	if(myTime < bcet){
		bcet = myTime;
	}
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	cout << "Time: " << myTime << " ms" << "\t Time: " << myTime/1000 << " sec" << endl;
	cout << "\tWorst Case: " << wcet << " ms" << "\n\tBest  Case: " << bcet << " ms" << endl;
	cout << "\tThis  Case Time Left over: " << (maxTime*1000-myTime) << " ms" << endl;
	cout << "\tWorst Case Time Left over: " << (maxTime*1000-wcet) << " ms" << endl;
	cout << "\tBest  Case Time Left over: " << (maxTime*1000-bcet) << " ms" << endl;
}
float GPUHandler::StopTimingMain(const int numRuns, long myUltraTime)
{
	cudaSetDevice(2);
	cudaDeviceSynchronize();
	cudaSetDevice(2);
	cudaDeviceSynchronize();
	cudaSetDevice(1);
	cudaDeviceSynchronize();
	cudaSetDevice(0);
	cudaDeviceSynchronize();
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &myTime, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	if(numRuns > 10){
		if(myTime > wcet)
			wcet = myTime;
		if(myTime < bcet)
			bcet = myTime;
		cout << "Time: " << myTime << " ms" << "\t Time: " << myTime/1000 << " sec" << endl;
		//		cout << "\tWorst Case: " << wcet << " ms" << "\n\tBest  Case: " << bcet << " ms" << endl;
		cout << "\tThis  Case Time Left over: " << (maxTime*1000-myTime) << " ms" << endl;
		cout << "\tWorst Case Time Left over: " << (maxTime*1000-wcet) << " ms" << endl;
		cout << "\tBest  Case Time Left over: " << (maxTime*1000-bcet) << " ms" << endl;



		if(myUltraTime > Ultrawcet)
			Ultrawcet = myUltraTime;
		if(myUltraTime < Ultrabcet)
			Ultrabcet = myUltraTime;
		cout << "\tUltraView This  Case Exectution Time: " << myUltraTime << " ms" << endl;
		cout << "\tUltraView Worst Case Exectution Time: " << Ultrawcet << " ms" << endl;
		cout << "\tUltraView Best  Case Exectution Time: " << Ultrabcet << " ms" << endl;


	}

	return (maxTime*1000-myTime);
}
void GPUHandler::StopTimingKernel(int threads)
{
	//	for(int threads = 1; threads <= 1024; threads++){
	//		StartTiming();

	//	StopTimingKernel(threads);
	//}cout << "\nFastest Time: " << bcet << "\t with Threads set to: " << bestThread << endl << endl;

	cudaSetDevice(2);
	cudaDeviceSynchronize();
	cudaSetDevice(1);
	cudaDeviceSynchronize();
	cudaSetDevice(0);
	cudaDeviceSynchronize();
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &myTime, start, stop );
	cout << "Threads: " << threads << "\tTime in ms: " << myTime <<  endl;
	if(myTime > wcet){
		wcet = myTime;
	}
	if(myTime < bcet){
		bcet = myTime;
		bestThread = threads;
		cout << "\n\t\t\t\tTime in ms: " << myTime << "\t Threads: " << bestThread << endl << endl;
	}
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
}

void GPUHandler::NaNTesting()
{
	int countError = 0;
	cudaMemcpyAsync(filter_fft, dev_equalizers_padded_GPU0, sizeof(cufftComplex)*conv_length, cudaMemcpyDeviceToHost,stream_GPU0_array[stream_0]);
	for(int test = 0; test < conv_length; test++)
		if(filter_fft[test].x != filter_fft[test].x)
			countError++;
	if(countError > 0)
		cout << "\t\t\t\t\t\t\t\t\t\tfilter_fft0 BUSTED!!!!!!!\tCount: " << countError << endl;
	cudaMemcpyAsync(filter_fft, dev_equalizers_padded_GPU1, sizeof(cufftComplex)*conv_length, cudaMemcpyDeviceToHost,stream_GPU1_array[stream_0]);
	for(int test = 0; test < conv_length; test++)		if(filter_fft[test].x != filter_fft[test].x)
		countError++;
	if(countError > 0)
		cout << "\t\t\t\t\t\t\t\t\t\tfilter_fft1 BUSTED!!!!!!!\tCount: " << countError << endl;

	//	if(countError == 0)
	//		cout << "\tfilter_fft All Good.\t\tCount: " << countError << endl;


	countError = 0;
	cudaMemcpyAsync(filter_fft, dev_signal_preDetection_GPU0, sizeof(cufftComplex)*conv_length, cudaMemcpyDeviceToHost,stream_GPU0_array[stream_0]);
	for(int test = 0; test < conv_length; test++)
		if(filter_fft[test].x != filter_fft[test].x)
			countError++;
	if(countError > 0)
		cout << "\t\t\t\t\t\t\t\t\t\tfftPlan_detection_GPU0 BUSTED!!!!!!!\tCount: " << countError << endl;
	cudaMemcpyAsync(filter_fft, dev_signal_preDetection_GPU1, sizeof(cufftComplex)*conv_length, cudaMemcpyDeviceToHost,stream_GPU1_array[stream_0]);
	for(int test = 0; test < conv_length; test++)
		if(filter_fft[test].x != filter_fft[test].x)
			countError++;
	if(countError > 0)
		cout << "\t\t\t\t\t\t\t\t\t\tfftPlan_detection_GPU1 BUSTED!!!!!!!\tCount: " << countError << endl;

	//	if(countError == 0)
	//		cout << "\tfftPlan_demod_GPU All Good.\tCount: " << countError << endl;

}


void GPUHandler::checkGPUStats() {
	cudaError_t cudaStat;

	//Get the K20 device number
	int deviceNum;
	cudaStat = cudaGetDevice(&deviceNum); if(cudaStat != cudaSuccess) cout << "Error - could not get device number in checkGPUStats()" << endl;
	//cout << "Device number: " << deviceNum << endl;

	cout << "------------------------------------------------------" << endl;
	cout << "                     GPU MEMORY                       " << endl;
	cout << "------------------------------------------------------" << endl;

	size_t free, total;
	//Read out the Free and Total available global memory (in bytes)
	cudaSetDevice(device_GPU0);
	cudaStat = cudaMemGetInfo(&free,&total); if(cudaStat != cudaSuccess) cout << "Error - could not get device memory info checkGPUStats()" << endl;
	float usedPercentage = ((float)(total-free))/((float)total);
	float freePercentage = ((float)free)/((float)total);
	float freeMB = float(free) / (float)1048576;
	float usedMB = float(total-free) / (float)1048576;
	float totalMB = float(total) / (float)1048576;
	cout << "K40 \tFree memory (in MB): " << freeMB << "\tTotal memory (in MB): " << totalMB << "\tUsed memory (in MB): " << usedMB << "\tPrecent Free: %" << freePercentage*100 << "\tPrecent Used: %" << usedPercentage*100 << endl;

	cudaSetDevice(device_GPU1);
	//Read out the Free and Total available global memory (in bytes)
	cudaStat = cudaMemGetInfo(&free,&total); if(cudaStat != cudaSuccess) cout << "Error - could not get device memory info checkGPUStats()" << endl;
	usedPercentage = ((float)(total-free))/((float)total);
	freePercentage = ((float)free)/((float)total);
	freeMB = float(free) / (float)1048576;
	usedMB = float(total-free) / (float)1048576;
	totalMB = float(total) / (float)1048576;
	cout << "K20a \tFree memory (in MB): " << freeMB << "\tTotal memory (in MB): " << totalMB << "\tUsed memory (in MB): " << usedMB << "\tPrecent Free: %" << freePercentage*100 << "\tPrecent Used: %" << usedPercentage*100 << endl;

	cudaSetDevice(device_GPU2);
	//Read out the Free and Total available global memory (in bytes)
	cudaStat = cudaMemGetInfo(&free,&total); if(cudaStat != cudaSuccess) cout << "Error - could not get device memory info checkGPUStats()" << endl;
	usedPercentage = ((float)(total-free))/((float)total);
	freePercentage = ((float)free)/((float)total);
	freeMB = float(free) / (float)1048576;
	usedMB = float(total-free) / (float)1048576;
	totalMB = float(total) / (float)1048576;
	cout << "K20b \tFree memory (in MB): " << freeMB << "\tTotal memory (in MB): " << totalMB << "\tUsed memory (in MB): " << usedMB << "\tPrecent Free: %" << freePercentage*100 << "\tPrecent Used: %" << usedPercentage*100 << endl;

	cudaSetDevice(device_GPU0);
}

}
