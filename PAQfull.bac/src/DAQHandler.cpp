#include <iostream>
#include "uvdma/_AppSource/uvAPI.h"
#include "DAQHandler.h"
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <math.h>       /* pow */

using namespace std;
using namespace PAQ_SOQPSK;


namespace PAQ_SOQPSK {
daqHandler::daqHandler() {

	// Max number for unsigned int
	//unsigned int totalNumMbToGrab = pow(2,31);
	unsigned int totalNumMbToGrab = (unsigned int) (340/2*60*60*24)*2; // 24 hours worth of data

	//Default values for SetupBoard. These may change due to acquire_parser.
	BoardNum = 0;
	numBlocksToAcquire = totalNumMbToGrab;
	InternalClock = CLOCK_EXTERNAL;		// Changed default setting to 0 to match "-ic" from old version
	SingleChannelMode = 0;
	SingleChannelSelect = 1;
	ChannelSelect = 1;
	DESIQ = 0;
	DESCLKIQ = 0;
	ECLTrigger = 0;
	ECLTriggerDelay = 0;
	DualChannelMode = 0;
	DualChannelSelect = 01;
	firstchan = 0;		//default to channel 0
	secondchan = 3;	//default to channel 3
	chans = 9;			//default bitwise channel 0 and 3
	CaptureCount = 0;
	CaptureDepth = 0;
	Decimation = 1;
	PretriggerMemory = 0;
	TTLInvert = 0;
	TriggerMode = NO_TRIGGER;
	TriggerCh = 0;
	TriggerSlope = FALLING_EDGE;
	TriggerThreshold16 = 32768;
	TriggerThreshold14 = 8192;
	TriggerThreshold12 = 2048;
	TriggerHysteresis = 100;
	NumAverages = 0;
	AveragerLength = 4096;
	Fiducial = 0;
	forceCal = 0;
	Frequency = 0.0;

	user_outputfile = NULL;

	UseLargeMem = false;

	// Set up in Single Channel Mode (-scm)
	SingleChannelMode = 1;
	SingleChannelSelect = 0;

	disk_fd = INVALID_HANDLE_VALUE; // disk file handle
	output_file[0][0] = 'u';
	output_file[0][1] = 'v';
	output_file[0][2] = 'd';
	output_file[0][3] = 'm';
	output_file[0][4] = 'a';
	output_file[0][5] = '.';
	output_file[0][6] = 'd';
	output_file[0][7] = 'a';
	output_file[0][8] = 't';

	// Create a class with convienient access functions to the DLL
	uv = new uvAPI;
	sysMem = NULL;

	// Initialize settings
	if (forceCal)
		uv->setSetupDoneBit(BoardNum,0); // force full setup

	uv->setupBoard(BoardNum);

	acquire_set_session(uv);


	// read the clock freq
	unsigned int adcClock = uv->getAdcClockFreq(BoardNum);
	std::cout << " ADC clock freq ~= " << adcClock << "MHz" << std::endl;
	fflush(stdout);

	// Allocate a page aligned buffer for DMA
	error = uv->X_MemAlloc((void**)&sysMem, DIG_BLOCK_SIZE);	if (error)	{		std::cout << "failed to allocate block buffer" << std::endl;	}


	mySetupAcquire();
}
daqHandler::~daqHandler() {

	// deallocate resources
	if(sysMem){uv->X_FreeMem(sysMem);}
	delete uv;

	cout << "Stuff is freed!!!" << endl;
}

void daqHandler::acquire_set_session(uvAPI *pUV)
{

	pUV->selClock(BoardNum, InternalClock);
	int defaultchannels = pUV->GetAllChannels(BoardNum);
	//	printf("All channels = %d\n",defaultchannels);
	if (pUV->IS_ISLA216P(BoardNum))
	{
		if (pUV->HAS_microsynth(BoardNum) && Frequency!=0){
			if (Frequency<=250000000){
				pUV->MICROSYNTH_freq(BoardNum, Frequency);
			}
			else{
				printf("Frequency outside range 50MHz-250MHz, using previous frequency\n");
			}
		}
		unsigned int selectedChannels = 0;

		if (SingleChannelMode == 1)
		{
			switch (SingleChannelSelect) {
			case 0:
				selectedChannels = IN0;
				break;
			case 1:
				selectedChannels = IN3;  // due to part poulation on 2ch verion ch1 is on IN3
				break;
			case 2:
				selectedChannels = IN1;
				break;
			case 3:
				selectedChannels = IN2;
				break;
			default:
				selectedChannels = IN0;
				break;
			}
		}
		else if (DualChannelMode == 1 || defaultchannels == 2)
		{
			selectedChannels = (chans);
		}
		else if (defaultchannels == 4)
		{
			selectedChannels = (IN0 | IN1 | IN2 | IN3);
		}
		else {
			printf("channel info not found, exiting\n");
			exit(1);
		}
		//        std::cout << std::endl << std::endl << BoardNum << "AD16 setup selected channels= " <<  selectedChannels << "scs=" << SingleChannelSelect << std::endl << std::endl;
		pUV->selectAdcChannels(BoardNum, selectedChannels);

		unsigned int TriggerChannel;
		switch (TriggerCh) {
		case 0:
			TriggerChannel = IN0;
			break;
		case 1:
			TriggerChannel = IN3;  // due to part poulation on 2ch verion ch1 is on IN3
			break;
		case 2:
			TriggerChannel = IN1;
			break;
		case 3:
			TriggerChannel = IN2;
			break;
		default:
			TriggerChannel = IN0;
			break;
		}
		//		fflush(stdout);
		//		pUV->selectTrigger(BoardNum, WAVEFORM_TRIGGER,FALLING_EDGE , IN0);
		pUV->selectTrigger(BoardNum, TriggerMode, TriggerSlope, TriggerCh);			//this might need to be changed to "TriggerCh" as was changed on 12bit
		pUV->configureWaveformTrigger(BoardNum, TriggerThreshold16, TriggerHysteresis);
		pUV->configureSegmentedCapture(BoardNum, CaptureCount, CaptureDepth, 0);
		if (NumAverages > 0){
			pUV->configureAverager(BoardNum, NumAverages, AveragerLength, 0);
		}
		pUV->setFiducialMarks(BoardNum, Fiducial);
		unsigned int trigVal = pUV->isTriggerEnabled(BoardNum);
		//	std::cout << "trigVal" << trigVal << std::endl;

	}
	if (pUV->IS_adc12d2000(BoardNum))
	{
		if (pUV->HAS_microsynth(BoardNum) && Frequency!=0){
			if (Frequency>=300000000){
				pUV->MICROSYNTH_freq(BoardNum, Frequency);
			}
			else{
				printf("Frequency outside range 300MHz-2GHz, using previous frequency\n");
			}
		}
		unsigned int selectedChannels = 0;
		if (SingleChannelMode == 1)	// One channel mode
		{
			if (DESIQ == 1){
				selectedChannels = 8;
			}
			else if (DESCLKIQ == 1){
				selectedChannels = 4;
			}
			else if (SingleChannelSelect == 0){
				selectedChannels = 1;
			}
			else if (SingleChannelSelect == 1){
				selectedChannels = 2;
			}
			else{
				selectedChannels = 1;
			}
			pUV->selectAdcChannels(BoardNum, selectedChannels);
		}
		else
		{
			pUV->selectAdcChannels(BoardNum, IN0 | IN1);	// Two channel mode
			printf("two channel mode\n");
		}

		unsigned int TriggerChannel;
		//printf("triggerCh:%d\n",TriggerCh);
		//printf("triggerCh:%d\n",TriggerCh);
		switch (TriggerCh) {
		case 0:
			TriggerChannel = IN0;
			break;
		case 1:
			TriggerChannel = IN3;  // due to part poulation on 2ch verion ch1 is on IN3
			break;
		case 2:
			TriggerChannel = IN1;
			break;
		case 3:
			TriggerChannel = IN2;
			break;
		default:
			TriggerChannel = IN0;
			break;
		}

		// Set ECL Trigger Delay
		pUV->SetECLTriggerDelay(BoardNum, ECLTriggerDelay);
		// Set Decimation
		pUV->setAdcDecimation(BoardNum, Decimation);
		// Set ECL Trigger
		//       std::cout << "ECLTrigger=" << ECLTrigger << std::endl;
		pUV->SetECLTriggerEnable(BoardNum, ECLTrigger);

		//printf("boardnum=%d triggermode=%d triggerslope=%d triggerch=%d triggerthreshold12=%d triggerhysteresis=%d\n",BoardNum,TriggerMode,TriggerSlope,TriggerCh,TriggerThreshold12,TriggerHysteresis);
		//printf("capturecount=%d capturedepth=%d numaverages=%d averagerlength=%d pretrigger=%d\n",CaptureCount,CaptureDepth,NumAverages,AveragerLength,PretriggerMemory);
		pUV->selectTrigger(BoardNum, TriggerMode, TriggerSlope, TriggerCh); //not using "TriggerChannel" and instead using "TriggerCh".
		pUV->configureWaveformTrigger(BoardNum, TriggerThreshold12, TriggerHysteresis);
		pUV->configureSegmentedCapture(BoardNum, CaptureCount, CaptureDepth, 0);
		if (NumAverages > 0){
			if (NumAverages > 64){
				NumAverages = 64;
				printf("!!CAUTION!!: Averages reduced to maximum for AD12 (64)\n");
			}
			pUV->configureAverager(BoardNum, NumAverages, AveragerLength, 0);
		}
		pUV->setFiducialMarks(BoardNum, Fiducial);
		//printf("finished AD12 section\n");
		//Sleep(10000);
	}

	if (pUV->IS_AD5474(BoardNum)){
		printf("AD14 found. ");
		unsigned int selectedChannels = 0;

		if (SingleChannelMode == 1)	// One channel mode
		{
			printf("Setting board to 1 channel mode. ");

			if (SingleChannelSelect == 0)
			{
				printf("Acquire IN0.");
				selectedChannels = IN0;
			}
			else if (SingleChannelSelect == 1)
			{
				printf("Acquire IN1.");
				selectedChannels = IN1;
			}
			else
			{
				printf("Invalid channel. Defaulting to acquire IN0.");
				selectedChannels = IN0;
			}

			pUV->selectAdcChannels(BoardNum, selectedChannels);
		}
		else
		{
			selectedChannels = 3;		//1 || 2 = 3
			printf("Setting board to 2 channel mode\n");
			pUV->selectAdcChannels(BoardNum, selectedChannels);
		}

		// Configure Trigger
		pUV->selectTrigger(BoardNum, TriggerMode, TriggerSlope, TriggerCh);

		//Configure Waveform Trigger
		pUV->configureWaveformTrigger(BoardNum, TriggerThreshold14, TriggerHysteresis);

		// Configure Segmented Capture
		pUV->configureSegmentedCapture(BoardNum, CaptureCount, CaptureDepth, 0);

		// Configure Averager
		pUV->configureAverager(BoardNum, NumAverages, AveragerLength, 0);

		// Set Decimation
		pUV->setAdcDecimation(BoardNum, Decimation);

		// Set Fiducial Marks
		pUV->setFiducialMarks(BoardNum, Fiducial);

	}

	if (!pUV->IS_adc12d2000(BoardNum) && !pUV->IS_ISLA216P(BoardNum) && !pUV->IS_AD5474(BoardNum)){
		printf("AD8 found\n");


		unsigned int selectedChannels = 0;

		if (SingleChannelMode == 1){	// One channel mode
			if (SingleChannelSelect == 1){
				selectedChannels = IN1;
			}
			else{
				selectedChannels = IN0;
			}
		}
		else{
			selectedChannels = 3;
		}
		//pUV->selectAdcChannels(BoardNum, selectedChannels);
		pUV->selectAdcChannels(BoardNum, selectedChannels);
		// Set ECL Trigger Delay
		pUV->SetECLTriggerDelay(BoardNum, ECLTriggerDelay);
		// Set Decimation
		pUV->setAdcDecimation(BoardNum, Decimation);
		// Set ECL Trigger
		pUV->SetECLTriggerEnable(BoardNum, ECLTrigger);
		// Configure Segmented Capture
		pUV->configureSegmentedCapture(BoardNum, CaptureCount, CaptureDepth, 0);
	}

	//       pUV->SetAdcDecimation(BoardNum, Decimation);
	pUV->SetTTLInvert(BoardNum, TTLInvert);
	pUV->setPreTriggerMemory(BoardNum, PretriggerMemory);
}

void daqHandler::setNumBlocksToAcquire(int mySet){
	numBlocksToAcquire = mySet;
}

unsigned long daqHandler::acquire(char* host_array, int numBlocks){

	int numOverRuns = 0;
	// For each block of data requested
	for	(unsigned int i=0; i<numBlocks; i++ )
	{
		char* temp_pointer = &host_array[i*DIG_BLOCK_SIZE];
		// read a block from the board
		uv->X_Read(BoardNum,sysMem,DIG_BLOCK_SIZE);
		memcpy(temp_pointer, sysMem, DIG_BLOCK_SIZE);
	}
	numOverRuns += uv->getOverruns(BoardNum);
	//cout << "\nReading " << numBlocks << "Mb of DAQ samples" << endl;//"\tNumOverruns = " << numOverRuns << endl;

	return numOverRuns;
}

void daqHandler::mySetupAcquire(){
	uv->SetupAcquire(BoardNum,numBlocksToAcquire);
	cout << "Setting up Ultraview numToAcquire " << numBlocksToAcquire << endl;
	cout << "Ultraview can run for " << numBlocksToAcquire/170/60/60 << " Hours" << endl;
	cout << "Ultraview can run for " << numBlocksToAcquire/170/60/60/24 << " Days" << endl;
}

void daqHandler::writeDAQfile(unsigned short host_array[], int numBlocks){
	//daqHandler.writeDAQfile(&DAQBuffer[50*2],numMb);

	string fileName = "/home/adm85/git/JeffPaq/UltraSetup/uvdma";
	const char* path = fileName.c_str();
	cout << "\nWriting " << fileName << " to ";
	for(int i = 0; i < 60; i++)
		cout << path[i];
	cout << endl;

	// Find the number of write chunks needed to write the whole array
	int WRITE_CHUNK_LENGTH = 70;
	int numWrites = numBlocks*DIG_BLOCKSIZE/2/(WRITE_CHUNK_LENGTH); // Run off the end a little bit

	int length = DIG_BLOCK_SIZE*numBlocks;
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

void daqHandler::writeLittleDAQfile(unsigned short host_array[], int numIdx, int numWrite){
	//daqHandler.writeDAQfile(&DAQBuffer[50*2],numMb);

	string fileName;          // string which will contain the result

	ostringstream convert;   // stream used for the conversion

	convert << "/home/adm85/git/JeffPaq/UltraSetup/uvdma" << numWrite;      // insert the textual representation of 'Number' in the characters in the stream

	fileName = convert.str();

	const char* path = fileName.c_str();
	cout << "Writing " << fileName << endl;

	// Do the first chuck so the file is deleted then rewritten
	ofstream out(path, ios::out | ios::binary | ios::trunc);
	out.write((char *) &numIdx, sizeof(int));
	out.write((char *) &host_array[0], numIdx * sizeof(unsigned short));
	out.close();
}

void daqHandler::writeDAQfileFloat(float host_array[], int numBlocks){
	//daqHandler.writeDAQfile(&DAQBuffer[50*2],numMb);

	string fileName = "/home/adm85/git/JeffPaq/UltraSetup/uvdmaFloat";
	const char* path = fileName.c_str();
	cout << "\nWriting " << fileName << " to ";
	for(int i = 0; i < 60; i++)
		cout << path[i];
	cout << endl;

	// Find the number of write chunks needed to write the whole array
	int WRITE_CHUNK_LENGTH = 70;
	int numWrites = numBlocks*DIG_BLOCKSIZE/2/(WRITE_CHUNK_LENGTH);

	int length = DIG_BLOCK_SIZE*numBlocks;
	// Find how long the copy is actually going to be
	int lengthOfCopy = numWrites*WRITE_CHUNK_LENGTH;

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
}

}
