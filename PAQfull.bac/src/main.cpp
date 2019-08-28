/*
 * File:   main.cpp
 * Author: Andrew McMurdie
 *
 * Created on April 1, 2013, 9:24 PM
 */

//#define NUM_RUNS 10
#define NUM_RUNS 500000000
//#define CMA_RUNS 8

#include <cstdlib>
#include <iostream>
#include <sys/time.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <fstream>
#include <math.h>
#include <sys/time.h>
#include <signal.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublas_api.h>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <unistd.h>
#include <iomanip>
#include "uvdma/_AppSource/uvAPI.h"
#include "DAQHandler.h"
#include "FileReader.h"
#include "Samples.h"
#include "Filter.h"
#include "PreambleDetector.h"
#include "Environment.h"
#include "FPGAHandler.h"
#include "GPUHandler.h"
#include "DmaDriverDll.h"
#include "DmaDriverIoctl.h"
#include "stdafx.h"
#include "StdTypes.h"
using namespace std;
using namespace PAQ_SOQPSK;


bool runProgram = true;
bool regionSwap = false;
bool writeFiles = false;
bool conj = false;
long mtime, seconds, useconds;
unsigned long numOverruns = 0;
int numMbGrabbedTotal = 0;
int numToAve = 0;
//bool hardCoded[8] = {true,	true, 	true, 	true, 	true, 	false, 	false, 	false};
bool hardCoded[8] = {false,	false, 	false, 	false, 	false, 	false, 	false, 	false};
#define PN11_LENGTH 2047
int pnIdx = 0;
int next_pnIdx = 0;
int bits_FIFO_oldestSample_index = 0;
int bitBufferSize = 6144*3104*15;
int numRuns = -1;
int CMAruns = 4;
#define bitErrorFIFO_length 5
int bitErrorFIFO[bitErrorFIFO_length];
int MbRead = 0;

int getNextPNIdx() {
	pnIdx = next_pnIdx;
	next_pnIdx = (next_pnIdx + 1) % PN11_LENGTH;
	return pnIdx;
}
void loadBitDecChunk(unsigned char* pn11, unsigned char* bitDecArray, int length, bool hardCoded[]) {
	int i;
	unsigned char temp = 0;
	for(i=0; i < length; i+=8) {
		temp =	(pn11[getNextPNIdx()]     ) |
				(pn11[getNextPNIdx()] << 1) |
				(pn11[getNextPNIdx()] << 2) |
				(pn11[getNextPNIdx()] << 3) |
				(pn11[getNextPNIdx()] << 4) |
				(pn11[getNextPNIdx()] << 5) |
				(pn11[getNextPNIdx()] << 6) |
				(pn11[getNextPNIdx()] << 7);
		for(int j = 0; j < 8; j++)
			if(hardCoded[j])
				bitDecArray[i+j] = temp;
	}
}

void signalHandler(int signalNum) {
	cout << "Caught signal number: " << signalNum << endl;
	if(signalNum == SIGUSR1){
		regionSwap = true;
		cout << "Swapping..." << endl;
	}
	else{
		cout << "Killing..." << endl;
		runProgram = false;
		numRuns = NUM_RUNS+1;
	}
}

void writeRegionCalc(unsigned char* regBuffer, bool& iWriteRegion, bool& qWriteRegion) {
	//Check against bit index 2
	iWriteRegion = (regBuffer[0] & 0x20) > 0;

	//Check against bit index 5
	qWriteRegion = (regBuffer[0] & 0x04) > 0;
}

int main() {

	signal(SIGABRT, signalHandler);
	signal(SIGINT,  signalHandler);
	signal(SIGTERM, signalHandler);
	signal(SIGUSR1, signalHandler);

	fcntl (0, F_SETFL, O_NONBLOCK);
	char input[1000];
	ssize_t bytes_read;

	ULONGLONG cardAddress;
	int iStat;

	unsigned char pn11[PN11_LENGTH] = {1,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,1,0,1,0,1,0,0,1,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,1,1,1,0,0,1,0,0,0,1,1,0,1,1,1,0,1,0,1,1,1,0,1,0,1,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,1,0,0,1,0,0,0,1,0,1,0,1,1,0,1,0,1,0,0,0,0,1,1,0,0,0,0,1,0,0,1,1,1,1,0,0,1,0,1,1,1,0,0,1,1,1,0,0,1,0,1,1,1,1,0,1,1,1,0,0,1,0,0,1,0,1,0,1,1,1,0,1,1,0,0,0,0,1,0,1,0,1,1,1,0,0,1,0,0,0,0,1,0,1,1,1,0,1,0,0,1,0,0,1,0,1,0,0,1,1,0,1,1,0,0,0,1,1,1,1,0,1,1,1,0,1,1,0,0,1,0,1,0,1,0,1,1,1,1,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,1,0,1,1,1,1,1,0,0,1,0,0,1,0,0,0,1,1,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,0,0,1,1,0,0,0,1,1,1,0,1,1,1,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,0,0,0,1,1,0,0,1,1,1,0,0,1,1,1,1,1,1,0,1,1,1,1,0,0,0,0,1,0,1,0,0,1,1,0,0,1,0,0,0,1,1,1,1,1,1,0,1,0,1,1,0,0,0,0,1,0,0,0,1,1,1,0,0,1,0,1,0,1,1,0,1,1,1,0,0,0,0,1,1,0,1,0,1,1,0,0,1,1,1,0,0,0,1,1,1,1,1,0,1,1,0,1,1,0,0,0,1,0,1,1,0,1,1,1,0,1,0,0,1,1,0,1,0,1,0,0,1,1,1,1,0,0,0,0,1,1,1,0,0,1,1,0,0,1,1,0,1,1,1,1,1,1,1,1,1,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,1,1,0,1,0,0,0,1,0,0,1,1,0,0,1,0,1,0,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,1,0,0,1,0,1,0,0,1,1,1,1,1,0,0,0,1,1,1,0,0,0,1,1,0,1,1,0,1,1,0,1,1,1,0,1,1,0,1,1,0,1,0,1,0,1,1,0,1,1,0,0,0,0,0,1,1,0,1,1,1,0,0,0,1,1,1,0,1,0,1,1,0,1,1,0,1,0,0,0,1,1,0,1,1,0,0,1,0,1,1,1,0,1,1,1,1,0,0,1,0,1,0,1,0,0,1,1,1,0,0,0,0,0,1,1,1,0,1,1,0,0,0,1,1,0,1,0,1,1,1,0,1,1,1,0,0,0,1,0,1,0,1,0,1,1,0,1,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,1,1,1,1,1,0,1,0,0,1,1,0,0,0,1,0,0,1,1,1,1,1,0,1,0,1,1,1,0,0,0,1,0,0,0,1,0,1,1,0,1,0,1,0,1,0,0,1,1,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,0,0,0,1,1,0,0,1,1,1,1,0,1,1,1,1,1,1,0,0,1,0,1,0,0,0,0,1,1,1,0,0,0,1,0,0,1,1,0,1,1,0,1,0,1,1,1,1,0,1,1,0,0,0,1,0,0,1,0,1,1,1,0,1,0,1,1,0,0,1,0,1,0,0,0,1,1,1,1,0,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,1,1,1,1,1,0,0,1,1,1,0,0,0,0,1,1,1,1,0,1,1,0,0,1,1,0,0,1,0,1,1,1,1,1,1,1,1,0,0,1,0,0,0,0,0,0,1,1,1,0,1,0,0,0,0,1,1,0,1,0,0,1,0,0,1,1,1,0,0,1,1,0,1,1,1,0,1,1,1,1,1,0,1,0,1,0,1,0,0,0,1,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,1,0,0,0,1,0,1,1,0,0,0,1,0,1,0,0,1,1,1,0,1,0,0,0,1,1,1,0,1,0,0,1,0,1,1,0,1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1,1,0,0,1,1,0,0,0,1,1,1,1,1,1,1,1,0,1,1,0,0,0,0,0,0,1,0,1,1,1,0,0,0,0,1,0,0,1,0,1,1,0,0,1,0,1,1,0,0,1,1,1,1,0,0,1,1,1,1,1,0,0,1,1,1,1,0,0,0,1,1,1,1,0,0,1,1,0,1,1,0,0,1,1,1,1,1,0,1,1,1,1,1,0,0,0,1,0,1,0,0,0,1,1,0,1,0,0,0,1,0,1,1,1,0,0,1,0,1,0,0,1,0,1,1,1,0,0,0,1,1,0,0,1,0,1,1,0,1,1,1,1,1,0,0,1,1,0,1,0,0,0,1,1,1,1,1,0,0,1,0,1,1,0,0,0,1,1,1,0,0,1,1,1,0,1,1,0,1,1,1,1,0,1,0,1,1,0,1,0,0,1,0,0,0,1,1,0,0,1,1,0,1,0,1,1,1,1,1,1,1,0,0,0,1,0,0,0,0,0,1,1,0,1,0,1,0,0,0,1,1,1,0,0,0,0,1,0,1,1,0,1,1,0,0,1,0,0,1,1,0,1,1,1,1,0,1,1,1,1,0,1,0,0,1,0,1,0,0,1,0,0,1,1,0,0,0,1,1,0,1,1,1,1,1,0,1,1,1,0,1,0,0,0,1,0,1,0,1,0,0,1,0,1,0,0,0,0,0,1,1,0,0,0,1,0,0,0,1,1,1,1,0,1,0,1,0,1,1,0,0,1,0,0,0,0,0,1,1,1,1,0,1,0,0,0,1,1,0,0,1,0,0,1,0,1,1,1,1,1,0,1,1,0,0,1,0,0,0,1,0,1,1,1,1,0,1,0,1,0,0,1,0,0,1,0,0,0,0,1,1,0,1,1,0,1,0,0,1,1,1,0,1,1,0,0,1,1,1,0,1,0,1,1,1,1,1,0,1,0,0,0,1,0,0,0,1,0,0,1,0,1,0,1,0,1,0,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,0,1,1,0,0,0,0,1,1,1,0,1,1,1,0,0,1,1,0,1,0,1,0,1,1,1,1,1,0,0,0,0,0,1,0,0,0,1,1,0,0,0,1,0,1,0,1,1,1,1,0,1,0,0,0,0,1,0,0,1,0,0,1,0,0,1,0,1,1,0,1,1,0,1,1,0,0,1,1,0,1,1,0,1,1,1,1,1,1,0,1,1,0,1,0,0,0,0,1,0,1,1,0,0,1,0,0,1,0,0,1,1,1,1,0,1,1,0,1,1,1,0,0,1,0,1,1,0,1,0,1,1,1,0,0,1,1,0,0,0,1,0,1,1,1,1,1,1,0,1,0,0,1,0,0,0,0,1,0,0,1,1,0,1,0,0,1,0,1,1,1,1,0,0,1,1,0,0,1,0,0,1,1,1,1,1,1,1,0,1,1,1,0,0,0,0,0,1,0,1,0,1,1,0,0,0,1,0,0,0,0,1,1,1,0,1,0,1,0,0,1,1,0,1,0,0,0,0,1,1,1,1,0,0,1,0,0,1,1,0,0,1,1,1,0,1,1,1,1,1,1,1,0,1,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,1,0,0,1,0,1,0,1,0,0,0,1,1,0,0,0,0,0,1,0,1,1,1,1,0,0,0,1,0,0,1,0,0,1,1,0,1,0,1,1,0,1,1,1,1,0,0,0,1,1,0,1,0,0,1,1,0,1,1,1,0,0,1,1,1,1,0,1,0,1,1,1,1,0,0,1,0,0,0,1,0,0,1,1,1,0,1,0,1,0,1,1,1,0,1,0,0,0,0,0,1,0,1,0,0,1,0,0,0,1,0,0,0,1,1,0,1,0,1,0,1,0,1,1,1,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,1,0,0,1,1,1,0,0,0,1,0,1,1,1,0,1,1,0,1,0,0,1,0,1,0,1,1,0,0,1,1,0,0,0,0,1,1,1,1,1,1,1,0,0,1,1,0,0,0,0,0,1,1,1,1,1,1,0,0,0,1,1,0,0,0,0,1,1,0,1,1,1,1,0,0,1,1,1,0,1,0,0,1,1,1,1,0,1,0,0,1,1,1,0,0,1,0,0,1,1,1,0,1,1,1,0,1,1,1,0,1,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0};

	FPGAHandler fpgaHandler;
	int bitDecisionWriteArea = fpgaHandler.BIT_DEC_TOP_AREA;

	GPUHandler gpuHandler(fpgaHandler.MEMORY_BUFFER_SIZE_IN_BYTES / sizeof(float));

	char* DAQBuffer;//                                                 This is how many bytes to malloc that are page aligned
	if(posix_memalign((void **)&DAQBuffer, fpgaHandler.getPageSize(), 343*1024*1024)) {		cout << "DAQBuffer allocation FAILED." << endl;		free(DAQBuffer);		exit(1);	}

	// DAQBufferShort is the pointer that the new DAQ samples are save at
	// The 34 samples before DAQBufferShort are the last 34 samples of last batch DAQ samples
	//	cout << "Sleeping" << endl;
	//	sleep(1);
	//	cout << "1" << endl;
	//	sleep(1);
	//	cout << "2" << endl;
	//	sleep(1);
	//	cout << "3" << endl;
	//	cout << "AWAKE!" << endl;

	//	for(int i = -10; i < numMb*1024*1024/2+10; i++){
	//		if(DAQBufferShort[i] == 0)
	//			cout << i << ": \t" << DAQBufferShort[i] << endl;
	//	}


	//////////////////////////////////////////////////////////////////////////////////////
	//		SINGLE FIFO INIT
	//////////////////////////////////////////////////////////////////////////////////////
	int PolyWriteIdx = 0;
	int threshold = gpuHandler.NUM_INPUT_SAMPLES_DEFAULT - gpuHandler.PolyPush_little + 10000;
	cout << "threshold " << threshold << endl;

	//	for(int i = 0; i < 10; i++){
	//		if(PolyWriteIdx < threshold)
	//			numMb = gpuHandler.UltraMbGrag_big;
	//		else
	//			numMb = gpuHandler.UltraMbGrag_little;
	//
	//		// Grab numMb from Ultraview card
	//		daqHandler.acquire((char*)DAQBufferShort,numMb);
	//
	//		// Run halfband filter
	//		gpuHandler.RunHalfbandFilterWithDAQCopy(numMb,DAQBufferShort);
	//
	//		// Run polyphase filters
	//		gpuHandler.RunPolyphaseFilters(numMb,PolyWriteIdx);
	//
	//		// Update PolyWriteIdx (PolyWriteIdx+= blah)
	//		if(PolyWriteIdx < threshold)
	//			PolyWriteIdx += gpuHandler.PolyPush_big;
	//		else
	//			PolyWriteIdx += gpuHandler.PolyPush_little;
	//
	//		// Pull from PAQ Sample FIFO
	//		gpuHandler.PullFromPolyFIFOandConvertFromComplexToRealandImag();
	//
	//		// Shift FIFO
	//		gpuHandler.ShiftPolyFIFO(PolyWriteIdx);
	//
	//		// Update PolyWriteIdx (PolyWriteIdx-= blah)
	//		PolyWriteIdx -= gpuHandler.NUM_INPUT_SAMPLES_DEFAULT;
	//
	//		cout << PolyWriteIdx << endl;
	//
	//	}
	//
	//
	//	cout << "\n\n Program Ran \n\n";
	//	free(DAQBuffer);
	//	return 0;

	//Register buffer. 16 bytes
	unsigned char* regBuffer;
	uint registerSize = 16; //16 Bytes, 256 bits
	if(posix_memalign((void **)&regBuffer, fpgaHandler.getPageSize(), fpgaHandler.REGISTER_SIZE_IN_BYTES)) {
		cout << "Register buffer allocation FAILED." << endl;
		free(regBuffer);
		exit(1);
	}
	for(int i=0; i < 16; i++)
		if(i == 0)
			regBuffer[0] = 0x01;
		else
			regBuffer[i] = 0x00;
	//I Samples Buffer. 150 MB (157,286,400 bytes) + 50,684 bytes
	unsigned char* iBuffer;
	if(posix_memalign((void **)&iBuffer, fpgaHandler.getPageSize(), gpuHandler.FULL_SAMPLE_BUFFER_SIZE_IN_BYTES)) {		cout << "I buffer allocation FAILED." << endl;		free(iBuffer);		exit(1);	}

	//Q Samples Buffer. 150 MB (157,286,400 bytes) + 50,684 bytes
	unsigned char* qBuffer;
	if(posix_memalign((void **)&qBuffer, fpgaHandler.getPageSize(), gpuHandler.FULL_SAMPLE_BUFFER_SIZE_IN_BYTES)) {		cout << "Q buffer allocation FAILED." << endl;		free(qBuffer);		exit(1);	}

	unsigned char* bits_8channel;
	if(posix_memalign((void **)&bits_8channel, fpgaHandler.getPageSize(), bitBufferSize*sizeof(char))) {		cout << "bits_8channel allocation FAILED." << endl;		free(bits_8channel);		exit(1);	}
	unsigned char* bits_8channel_FIFO_fillPointer;
	bits_8channel_FIFO_fillPointer = &bits_8channel[bits_FIFO_oldestSample_index];
	memcpy( bits_8channel, &bits_8channel[fpgaHandler.BIT_DEC_BUFFER_SIZE_IN_BYTES], (bitBufferSize-fpgaHandler.BIT_DEC_BUFFER_SIZE_IN_BYTES)*sizeof(char) );

	bool iWriteRegion, iWriteRegionNew, qWriteRegion, qWriteRegionNew;
	bool writeRegionChanged = false;
	ULONGLONG iSamplesAddr, qSamplesAddr;
	ULONGLONG bufferOffset;
	ULONG bufferLength = fpgaHandler.DMA_TRANSFER_SIZE;
	float* iBufferFloat;
	float* qBufferFloat;


	//Set the register to contain the number 3,000,000
	for(int i=3; i < 16; i++)
		regBuffer[i] = 0x00;
	regBuffer[0] = 0xC0;
	regBuffer[1] = 0xC6;
	regBuffer[2] = 0x2D;
	//Write to register
	iStat = DoMem(fpgaHandler.boardNum, fpgaHandler.DO_MEM_WRITE, 0, regBuffer, 0, fpgaHandler.TRANSMIT_SLOW_THRESHOLD_REG, fpgaHandler.REGISTER_SIZE_IN_BYTES, &fpgaHandler.statusInfo);	if(iStat != STATUS_SUCCESSFUL)		cout << "Register write FAILED." << endl;








	iStat = fpgaHandler.regRW(fpgaHandler.DO_MEM_WRITE,regBuffer,fpgaHandler.HOST_WRITE_REGION_REG_ADDR,registerSize);
	for(int i=1; i < 16; i++)
		regBuffer[i] = 0x00;
	regBuffer[0] = 0x01;
	iStat = DoMem(fpgaHandler.boardNum, fpgaHandler.DO_MEM_WRITE, 0, regBuffer, 0, fpgaHandler.BIT_DECISION_ACTIVATE_REG, fpgaHandler.REGISTER_SIZE_IN_BYTES, &fpgaHandler.statusInfo);	if(iStat != STATUS_SUCCESSFUL) 		cout << "Register write FAILED." << endl;

	//dumb write
	if(fpgaHandler.memWrite(cardAddress,bits_8channel,fpgaHandler.BIT_DEC_BUFFER_SIZE_IN_BYTES) //44.8276 ms
			!= STATUS_SUCCESSFUL) {
		cout << "Write FAILED." << endl;
	}

	//UPDATE REGION
	if(bitDecisionWriteArea == fpgaHandler.BIT_DEC_TOP_AREA) {
		cardAddress = fpgaHandler.BIT_DEC_TOP_AREA_ADDR;
		regBuffer[0] = 0x01;
		bitDecisionWriteArea = fpgaHandler.BIT_DEC_BOTTOM_AREA;
	} else {
		cardAddress = fpgaHandler.BIT_DEC_BOTTOM_AREA_ADDR;
		regBuffer[0] = 0x02;
		bitDecisionWriteArea = fpgaHandler.BIT_DEC_TOP_AREA;
	}
	iStat = fpgaHandler.regRW(fpgaHandler.DO_MEM_WRITE,regBuffer,fpgaHandler.HOST_WRITE_REGION_REG_ADDR,registerSize);
	for(int i=1; i < 16; i++)
		regBuffer[i] = 0x00;
	regBuffer[0] = 0x01;
	iStat = DoMem(fpgaHandler.boardNum, fpgaHandler.DO_MEM_WRITE, 0, regBuffer, 0, fpgaHandler.BIT_DECISION_ACTIVATE_REG, fpgaHandler.REGISTER_SIZE_IN_BYTES, &fpgaHandler.statusInfo);	if(iStat != STATUS_SUCCESSFUL) 		cout << "Register write FAILED." << endl;


	daqHandler daqHandler;
	unsigned short* DAQBufferShort;
	DAQBufferShort = (unsigned short*)DAQBuffer;

	int numMb = 1;
	//	for(int i = 0; i < numMb*1024*1024/2+10; i++){
	//		DAQBufferShort[i] = 0;
	//	}
	// Dumb read then pause to be sure there is data available
	//	daqHandler.mySetupAcquire(numMb);
	daqHandler.acquire((char*)DAQBufferShort,numMb);
	if(PolyWriteIdx < threshold)
		numMb = gpuHandler.UltraMbGrag_big;
	else
		numMb = gpuHandler.UltraMbGrag_little;
	sleep(3);
	//-----------------------------------------------------------------------------------------------------------------------------------
	//			MAIN PROGRAM LOOP
	//-----------------------------------------------------------------------------------------------------------------------------------
	while(runProgram) {
		cout << "*********************************************************************" << endl;
		cout << "Starting new loop...Run " << numRuns++ << endl;
		cout << "numMbGrabbedTotal: " << numMbGrabbedTotal << endl;
		cout << "numMbGrabbedTotal/numToAve: " << ((float)numMbGrabbedTotal)/((float)numToAve) << endl;
		if(conj)
			cout << "Conjugating the Data..." << endl;
		else
			cout << "Not Conjugating the Data..." << endl;

		fpgaHandler.regRW(fpgaHandler.DO_MEM_READ, regBuffer, fpgaHandler.REG_MEMORY_SAMPLES_ADDR, fpgaHandler.REGISTER_SIZE_IN_BYTES);
		writeRegionCalc(regBuffer, iWriteRegion, qWriteRegion);
		writeRegionChanged = false;
		while(!writeRegionChanged) {
			fpgaHandler.regRW(fpgaHandler.DO_MEM_READ, regBuffer, fpgaHandler.REG_MEMORY_SAMPLES_ADDR, fpgaHandler.REGISTER_SIZE_IN_BYTES);
			writeRegionCalc(regBuffer, iWriteRegionNew, qWriteRegionNew);


			if((iWriteRegionNew != iWriteRegion) && (qWriteRegionNew != qWriteRegion)) {
				//A region just finished being written to. We want to begin processing the samples now.
				writeRegionChanged = true;
			}
		}
		gpuHandler.StartTiming();

		/********************************************************************************************************************************
		 * 			RUN PREAMBLE DETECTOR
		 *
		 * 	The preamble detector is the first operation run on our samples. It uses the NCPDI-2 function
		 * 	to correlate the samples with a known good preamble, letting us know where the packets start.
		 * 	This is necessary so that we can identify and use the preamble for frequency offset estimation,
		 * 	and so that we will know where the payload for each packet lies.
		 * 	Run time: 79.1204 ms 08/31/2016
		 */
		iBufferFloat = (float*)iBuffer;
		qBufferFloat = (float*)qBuffer;
		gpuHandler.DAQCopytoDevice(numMb,DAQBufferShort);
		gpuHandler.preFindPreambles(iBufferFloat, qBufferFloat);



		gpuHandler.preambleDetector();
		//cout << "\n done preambleDetector \n";

		/********************************************************************************************************************************
		 * 			FREQUENCY OFFSET ESTIMATOR AND DEROTATOR
		 * 	Run time: 132.006 ms 08/31/2016
		 */
		gpuHandler.estimateFreqOffsetAndRotate();
		//cout << "\n done estimateFreqOffsetAndRotate \n";


		/********************************************************************************************************************************
		 * 			CHANNEL ESTIMATOR
		 * 	Run time: 7.04272 ms 08/31/2016
		 */
		gpuHandler.estimate_channel();
		//cout << "\n done estimate_channel \n";


		/********************************************************************************************************************************
		 * 			NOISE VARIANCE ESTIMATOR
		 * 	Run time: 6.222 ms 08/31/2016
		 */
		gpuHandler.calculate_noise_variance();
		//cout << "\n done calculate_noise_variance \n";


		/********************************************************************************************************************************
		 * 			ZERO FORCING AND MMSE EQUALIZERS
		 * 			Minimum run time: 500.601 ms	8/13/15
		 */

		gpuHandler.calculate_equalizers();
		//cout << "\n done calculate_equalizers \n";

		// Write the Ultraview samples from the GPU with the extra 34 on the front from last iteration
		//gpuHandler.writeBatch_DAQsamples_host(DAQBufferShort,numMb*1024*1024/2);
		//gpuHandler.writeBatch_DAQsamples(numMb*1024*1024/2+34);
		//gpuHandler.writeBatch_DAQsamplesLast34(34);
		// Write the Halfband samples with the extra 20 on the front from last iteration

		// Run Halfband filter
		gpuHandler.RunHalfbandFilterWithDAQCopy(numMb,DAQBufferShort);
		//cout << "\n done halfband \n";

		// Run polyphase filters
		gpuHandler.RunPolyphaseFilters(numMb,PolyWriteIdx);
		//cout << "\n done poly \n";
		//gpuHandler.writeBatch_HalfSamples(numMb*1024*1024/2/2+19);
		//gpuHandler.writeBatch_HalfSamplesLast19(19);

		// Update PolyWriteIdx (PolyWriteIdx+= blah)
		if(numMb ==  gpuHandler.UltraMbGrag_big)
			PolyWriteIdx += gpuHandler.PolyPush_big;
		else
			PolyWriteIdx += gpuHandler.PolyPush_little;

		// Write the Polyphase output with the extra PolyWriteIdx samples on the front
		//gpuHandler.writeBatch_PolySamples(PolyWriteIdx);

		// Pull from PAQ Sample FIFO
		gpuHandler.PullFromPolyFIFOandConvertFromComplexToRealandImag(conj);
		//cout << "\n done PullFromPolyFIFOandConvertFromComplexToRealandImag \n";
		// Write the PAQ samples with the extra 12671 samples on the front
		//			gpuHandler.writeBatch_PAQsamples(39321600+12671);
		//			gpuHandler.writeBatch_iFromPoly(39321600+12671);
		//			gpuHandler.writeBatch_qFromPoly(39321600+12671);

		// Shift FIFO
		gpuHandler.ShiftPolyFIFO(PolyWriteIdx);
		//cout << "\n done ShiftPolyFIFO \n";

		// Update PolyWriteIdx (PolyWriteIdx-= blah)
		PolyWriteIdx -= gpuHandler.NUM_INPUT_SAMPLES_DEFAULT;
		// Write the FIFO samples that are left over in the FIFO after pulling PAQ samples
		//gpuHandler.writeBatch_FIFOSamples(PolyWriteIdx);
		//cout << "\n done PolyWriteIdx \n";


		/********************************************************************************************************************************
		 * 			CMA
		 * 			Maximum run time: 118.017 ms 11/25/15
		 * 			Minimum run time: 117.816 ms 11/25/15
		 */
		for(int CMArun = 0; CMArun < CMAruns; CMArun++){
			//gpuHandler.StartTiming();
			gpuHandler.CMA();
			//gpuHandler.StopTimingMain(numRuns+10);
		}
		//cout << "\n done CMA \n";

		/********************************************************************************************************************************
		 * 			APPLY EQUALIZERS
		 * 			Maximum run time: 106.756 ms	6/29/15
		 * 			Minimum run time: 70.6152 ms	8/13/15
		 */
		gpuHandler.apply_equalizers_and_detection_filters();
		//cout << "\n done apply_equalizers_and_detection_filters \n";


		/********************************************************************************************************************************
		 * 			APPLY DEMODULATORS
		 * 			Maximum run time: 70.7135 ms	8/13/15
		 * 			Minimum run time: 80.891 ms		8/13/15 (Copying Bits to Host)
		 */
		gpuHandler.apply_demodulators();
		//cout << "\n done apply_demodulators \n";




		if(numRuns > 0){
			if(hardCoded[0] || hardCoded[1] || hardCoded[2] || hardCoded[3] || hardCoded[4] || hardCoded[5] || hardCoded[6] || hardCoded[7]){
				loadBitDecChunk(pn11, bits_8channel, fpgaHandler.BIT_DEC_BUFFER_SIZE_IN_BYTES1,hardCoded); // 77 ms
				cout << "Hard coded: " << hardCoded[0] << hardCoded[1] << hardCoded[2] << hardCoded[3] << hardCoded[4] << hardCoded[5] << hardCoded[6] << hardCoded[7] << "\n";
			}

			//Write BITS TO FPGA
			if(fpgaHandler.memWrite(cardAddress,bits_8channel,fpgaHandler.BIT_DEC_BUFFER_SIZE_IN_BYTES) //44.8276 ms
					!= STATUS_SUCCESSFUL) {
				cout << "Write FAILED." << endl;
			}

			//Shift the output Bits FIFO
			memcpy( bits_8channel, &bits_8channel[fpgaHandler.BIT_DEC_BUFFER_SIZE_IN_BYTES], (bitBufferSize-fpgaHandler.BIT_DEC_BUFFER_SIZE_IN_BYTES)*sizeof(char) ); // 23ms

			// We wrote out bits, update the pointers
			bits_FIFO_oldestSample_index -= fpgaHandler.BIT_DEC_BUFFER_SIZE_IN_BYTES;
			bits_8channel_FIFO_fillPointer = &bits_8channel[bits_FIFO_oldestSample_index];

			//UPDATE REGION
			if(bitDecisionWriteArea == fpgaHandler.BIT_DEC_TOP_AREA) {
				cardAddress = fpgaHandler.BIT_DEC_TOP_AREA_ADDR;
				regBuffer[0] = 0x01;
				bitDecisionWriteArea = fpgaHandler.BIT_DEC_BOTTOM_AREA;
			} else {
				cardAddress = fpgaHandler.BIT_DEC_BOTTOM_AREA_ADDR;
				regBuffer[0] = 0x02;
				bitDecisionWriteArea = fpgaHandler.BIT_DEC_TOP_AREA;
			}
			iStat = fpgaHandler.regRW(fpgaHandler.DO_MEM_WRITE,regBuffer,fpgaHandler.HOST_WRITE_REGION_REG_ADDR,registerSize);
			for(int i=1; i < 16; i++)
				regBuffer[i] = 0x00;
			regBuffer[0] = 0x01;
			iStat = DoMem(fpgaHandler.boardNum, fpgaHandler.DO_MEM_WRITE, 0, regBuffer, 0, fpgaHandler.BIT_DECISION_ACTIVATE_REG, fpgaHandler.REGISTER_SIZE_IN_BYTES, &fpgaHandler.statusInfo);	if(iStat != STATUS_SUCCESSFUL) 		cout << "Register write FAILED." << endl;
		}

		if((iWriteRegionNew == fpgaHandler.TOP_SECTION_WRITE_FLAG)) {
			iSamplesAddr = fpgaHandler.MEMORY_I_SAMPLES_BOTTOM_ADDR;
			qSamplesAddr = fpgaHandler.MEMORY_Q_SAMPLES_BOTTOM_ADDR;
		} else {
			iSamplesAddr = fpgaHandler.MEMORY_I_SAMPLES_TOP_ADDR;
			qSamplesAddr = fpgaHandler.MEMORY_Q_SAMPLES_TOP_ADDR;
		}

		//		//Read the 150 MB in five 30 MB transfers
		//		for(int i=0; i < 5; i++) {
		//			bufferOffset = i * fpgaHandler.DMA_TRANSFER_SIZE;
		//			fpgaHandler.memRead(iSamplesAddr + bufferOffset,
		//					iBuffer + gpuHandler.OLD_SAMPLES_OFFSET_IN_BYTES + bufferOffset,
		//					&bufferLength);
		//			fpgaHandler.memRead(qSamplesAddr + bufferOffset,
		//					qBuffer + gpuHandler.OLD_SAMPLES_OFFSET_IN_BYTES + bufferOffset,
		//					&bufferLength);
		//		}


		if(!(numRuns >= NUM_RUNS-1)){
			// Run new code!!!!

			if(PolyWriteIdx < threshold)
				numMb = gpuHandler.UltraMbGrag_big;
			else
				numMb = gpuHandler.UltraMbGrag_little;
			numMbGrabbedTotal += numMb;
			++numToAve;

			struct timeval start, end;


			gettimeofday(&start, NULL);
			// Grab numMb from Ultraview card
			numOverruns += daqHandler.acquire((char*)DAQBufferShort,numMb);
			gettimeofday(&end, NULL);

			seconds  = end.tv_sec  - start.tv_sec;
			useconds = end.tv_usec - start.tv_usec;

			mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
			if(mtime>1700){
				cout << "\n\nRegions Swapped!!!\n\n";
				if(bitDecisionWriteArea == fpgaHandler.BIT_DEC_TOP_AREA) {
					//			if(!iWriteRegionNew) {
					cardAddress = fpgaHandler.BIT_DEC_TOP_AREA_ADDR;
					regBuffer[0] = 0x01;

					//Toggle for the next loop
					bitDecisionWriteArea = fpgaHandler.BIT_DEC_BOTTOM_AREA;
				} else {
					cardAddress = fpgaHandler.BIT_DEC_BOTTOM_AREA_ADDR;
					regBuffer[0] = 0x02;

					//Toggle for the next loop
					bitDecisionWriteArea = fpgaHandler.BIT_DEC_TOP_AREA;
				}
			}

			if(numOverruns)
				cout << "NumOverruns = " << numOverruns << endl;

			// if crap hits the fan, use this code!!! and comment out anything polyphase above here in the while loop
			//			// Run Halfband filter
			//			gpuHandler.RunHalfbandFilterWithDAQCopy(numMb,DAQBufferShort);
			//
			//			// Write the Ultraview samples from the GPU with the extra 34 on the front from last iteration
			//			//gpuHandler.writeBatch_DAQsamples_host(DAQBufferShort,numMb*1024*1024/2);
			//			//gpuHandler.writeBatch_DAQsamples(numMb*1024*1024/2+34);
			//			//gpuHandler.writeBatch_DAQsamplesLast34(34);
			//			// Write the Halfband samples with the extra 20 on the front from last iteration
			//
			//			// Run polyphase filters
			//			gpuHandler.RunPolyphaseFilters(numMb,PolyWriteIdx);
			//			//gpuHandler.writeBatch_HalfSamples(numMb*1024*1024/2/2+19);
			//			//gpuHandler.writeBatch_HalfSamplesLast19(19);
			//
			//			// Update PolyWriteIdx (PolyWriteIdx+= blah)
			//			if(numMb ==  gpuHandler.UltraMbGrag_big)
			//				PolyWriteIdx += gpuHandler.PolyPush_big;
			//			else
			//				PolyWriteIdx += gpuHandler.PolyPush_little;
			//
			//			// Write the Polyphase output with the extra PolyWriteIdx samples on the front
			//			//gpuHandler.writeBatch_PolySamples(PolyWriteIdx);
			//
			//			// Pull from PAQ Sample FIFO
			//			gpuHandler.PullFromPolyFIFOandConvertFromComplexToRealandImag(conj);
			//			// Write the PAQ samples with the extra 12671 samples on the front
			//			//			gpuHandler.writeBatch_PAQsamples(39321600+12671);
			//			//			gpuHandler.writeBatch_iFromPoly(39321600+12671);
			//			//			gpuHandler.writeBatch_qFromPoly(39321600+12671);
			//
			//			// Shift FIFO
			//			gpuHandler.ShiftPolyFIFO(PolyWriteIdx);
			//
			//			// Update PolyWriteIdx (PolyWriteIdx-= blah)
			//			PolyWriteIdx -= gpuHandler.NUM_INPUT_SAMPLES_DEFAULT;
			//			// Write the FIFO samples that are left over in the FIFO after pulling PAQ samples
			//			//gpuHandler.writeBatch_FIFOSamples(PolyWriteIdx);
		}










		writeFiles = writeFiles||(numRuns >= NUM_RUNS-1);
		int processedBits = gpuHandler.postCPUrun(bits_8channel_FIFO_fillPointer,writeFiles,PolyWriteIdx);
		//cout << "\n done postCPUrun \n";
		writeFiles = false;












		// We got more Bits, update the pointers
		bits_FIFO_oldestSample_index += processedBits;
		bits_8channel_FIFO_fillPointer = &bits_8channel[bits_FIFO_oldestSample_index];


		if(bits_FIFO_oldestSample_index > bitBufferSize)
			cout << "FIFO OVERFLOW!!!" << endl;

		if(bits_FIFO_oldestSample_index < 0)
			cout << "FIFO UNDERFLOW!!!" << endl;

		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		////    END OF TIMING!!!!
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		bool missedTick = false;
		if(gpuHandler.StopTimingMain(numRuns,mtime) < 0){
			missedTick = true;
			//gpuHandler.writeMissedTimingBatchFiles();
		}

		// See if user wants to switch FPGA Regions
		bytes_read = read (0, input, 20);
		if((bytes_read > 0 && input[0] == 's') || missedTick){

			if(missedTick)
				cout << "\t\tMissed Timing!!!!!!\n" << endl;

			cout << "\n\nRegions Swapped!!!\n\n";
			if(bitDecisionWriteArea == fpgaHandler.BIT_DEC_TOP_AREA) {
				//			if(!iWriteRegionNew) {
				cardAddress = fpgaHandler.BIT_DEC_TOP_AREA_ADDR;
				regBuffer[0] = 0x01;

				//Toggle for the next loop
				bitDecisionWriteArea = fpgaHandler.BIT_DEC_BOTTOM_AREA;
			} else {
				cardAddress = fpgaHandler.BIT_DEC_BOTTOM_AREA_ADDR;
				regBuffer[0] = 0x02;

				//Toggle for the next loop
				bitDecisionWriteArea = fpgaHandler.BIT_DEC_TOP_AREA;
			}
		}

		if(input[0] == 'h'){
			switch(input[1]){
			case '1':
				hardCoded[0] = !hardCoded[0];
				break;
			case '2':
				hardCoded[1] = !hardCoded[1];
				break;
			case '3':
				hardCoded[2] = !hardCoded[2];
				break;
			case '4':
				hardCoded[3] = !hardCoded[3];
				break;
			case '5':
				hardCoded[4] = !hardCoded[4];
				break;
			case '6':
				hardCoded[5] = !hardCoded[5];
				break;
			case '7':
				hardCoded[6] = !hardCoded[6];
				break;
			case '8':
				hardCoded[7] = !hardCoded[7];
				break;
			}
			if(input[1] == 'a' && input[2] == '0')
				for(int i = 0; i < 8; i++)
					hardCoded[i] = 0;
			if(input[1] == 'a' && input[2] == '1')
				for(int i = 0; i < 8; i++)
					hardCoded[i] = 1;
			input[0] = '0';
		}

		if(input[0] == 'c' && input[1] != 'o' && input[1] != 'a'){
			gpuHandler.TimingReset();
			input[0] = '0';
		}
		if(input[0] == 'w'){
			writeFiles = true;
			input[0] = '0';
		}

		if(input[0] == 'p'){
			cout << endl << "Reset FIFO." << endl << endl;
			bits_FIFO_oldestSample_index = 0;
			bits_8channel_FIFO_fillPointer = &bits_8channel[0];
			input[0] = '0';
		}

		if(input[0] == 'k'){
			cout << "killing..." << endl;
			numRuns = NUM_RUNS-1;
			input[0] = '0';
		}

		if(input[0] == 'm' && input[1] == 'u' && input[2] == ' ' && input[3] == '=' && input[4] == ' '){
			float temp = atof (&input[5]);
			cout << "Setting CMA step size to " << temp << endl;
			gpuHandler.changeCMAmu(temp);
			input[0] = '0';
		}

		if(input[0] == 'C' && input[1] == 'M' && input[2] == 'A' && input[3] == 'r' && input[4] == 'u' && input[5] == 'n' && input[6] == 's' && input[7] == ' ' && input[8] == '=' && input[9] == ' '){
			int temp = (int) atof(&input[10]);
			cout << "Setting CMAruns to " << temp << endl;
			CMAruns = temp;
			input[0] = '0';
		}

		if(input[0] == 'c' && input[1] == 'o' && input[2] == 'n' && input[3] == 'j'){
			if(conj)
				cout << "non-conj" << endl;
			else
				cout << "conj" << endl;
			conj = !conj;
			input[0] = '0';
		}

		if(input[0] == 'c' && input[1] == 'a' && input[2] == 't'){
			daqHandler.mySetupAcquire();
			gpuHandler.TimingReset();
			cout << "Catching up!!!" << endl;
			input[0] = '0';
		}


		//Stop running
		if(numRuns >= NUM_RUNS-1){
			runProgram = false;
		}

	} // End of while Loop



















	cout << "Deactivating Bit Decision Transmitter..." << endl;
	//Set the bit to turn the transmitter off
	regBuffer[0] = 0x00;

	//Write to register
	iStat = DoMem(fpgaHandler.boardNum, fpgaHandler.DO_MEM_WRITE, 0, regBuffer,	0, fpgaHandler.BIT_DECISION_ACTIVATE_REG, registerSize, &fpgaHandler.statusInfo);	if(iStat != STATUS_SUCCESSFUL) {		cout << "Register write FAILED." << endl;	}

	//Free memory
	free(DAQBuffer);
	//	free(DAQBufferFloat);
	free(regBuffer);
	free(iBuffer);
	free(qBuffer);
	//	free(bits_8channel);

	cout << endl << "Program complete" << endl;
	//Return normally
	return 0;
}
