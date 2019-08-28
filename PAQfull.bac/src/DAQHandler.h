/*
 * daqHandler.h
 *
 *  Created on: Dec 14, 2016
 *      Author: adm85
 */

#ifndef daqHandler_H_
#define daqHandler_H_
#include "uvdma/_AppSource/uvAPI.h"

using namespace std;

namespace PAQ_SOQPSK {

class daqHandler {
public:
	//Constants
	daqHandler();
	//daqHandler(int argc, char ** argv);
	virtual ~daqHandler();


	//Super important!!!
	uvAPI *uv;
	unsigned long overruns;
	int error;
	HANDLE disk_fd; // disk file handle
	char output_file[MAX_DEVICES][128];

	// Create a class with convenient access functions to the DLL

	unsigned char * sysMem;
	// system memory buffer for large acquisitions
	unsigned char * sysMemBig;
	unsigned char * host1;
	unsigned char * host2;

	// used for accessing memory above 2GB
	size_t large_alloc_size;

	unsigned int sysmem_size;


	/*
	Reads the command sent in by the user and parses what board settings the user wishes to set.
	 */
	void acquire_parser(int argc, char ** argv);

	/*
	If "acquire" was sent with no arguments, a list of printf statements will display to guide the user.
	 */
	void acquire_parser_printf();

	/*
	Takes the arguments read by acquire_parser and sets the board to run with the desired settings.
	 */
	void acquire_set_session(uvAPI * pUV);

	/*
	runs the typical acquire program that is the default to Ultraview
	 */
	unsigned long acquire(char* host_array, int numBlocks);
	void mySetupAcquire();

	/*
	 * Sets the number of blocks (1Mb chunks) from the host
	 */
	void setNumBlocksToAcquire(int mySet);

	/*
		Writes the DAQ file in binary to /home/adm85/git/JeffPaq/UltraSetup/uvdma
	 */
	void writeDAQfile(unsigned short host_array[], int numBlocks);
	void writeLittleDAQfile(unsigned short host_array[], int numIdx, int numWrite);
	void writeDAQfileFloat(float host_array[], int numBlocks);

	/*
	Default values for SetupBoard. These may change due to acquire_parser.
	 */
	unsigned short BoardNum;
	unsigned int numBlocksToAcquire;
	unsigned int InternalClock;		// Changed default setting to 0 to match "-ic" from old version
	unsigned int SingleChannelMode;
	unsigned int SingleChannelSelect;
	unsigned int ChannelSelect;
	unsigned int DESIQ;
	unsigned int DESCLKIQ;
	unsigned int ECLTrigger;
	unsigned int ECLTriggerDelay;
	unsigned int DualChannelMode;
	unsigned int DualChannelSelect;
	unsigned int firstchan;
	unsigned int secondchan;
	unsigned int chans;
	unsigned int CaptureCount;
	unsigned int CaptureDepth;
	unsigned int Decimation;
	unsigned int PretriggerMemory;
	unsigned int TTLInvert;
	unsigned int TriggerMode;
	unsigned int TriggerCh;
	unsigned int TriggerSlope;
	unsigned int TriggerThreshold16;
	unsigned int TriggerThreshold14;
	unsigned int TriggerThreshold12;
	unsigned int TriggerHysteresis;
	unsigned int NumAverages;
	unsigned int AveragerLength;
	unsigned int Fiducial;
	unsigned int forceCal;
	double Frequency;
	char * user_outputfile;
	int early_exit;
	bool UseLargeMem;



private:
};

} /* namespace PAQ_SOQPSK */

#endif /* daqHandler_H_ */
