/**
@file uv_acquire1_main.cpp
* This is a simple example program which repeatedly acquires 1 block (1MByte) of data into onboard memory then reads it into system memory, and demonstrates use of the acquired data
*/

#include <iostream>
#include <iomanip>  
#include "uvAPI.h"
#include "calcStats.h"

#define MAX_CHANNELS 4


int main()
{
	std::cout << "uv_acquire v1.0" << std::endl << std::endl;

	// Create a class with convienient access functions to the DLL
	uvAPI *uv = new uvAPI;

	// Allocate a page aligned buffer for DMA 
	unsigned char * sysMem = NULL;
    uv->X_MemAlloc((void**)&sysMem, DIG_BLOCK_SIZE);
	if (!sysMem)
	{
		std::cout << "failed to allocate block buffer" << std::endl;
		return  -1;
	}

	// Setup the first (usually only) board in the system
    unsigned short BoardNum = 1;
    bool success = uv->setupBoard(BoardNum);
    uv->selClock(BoardNum,CLOCK_INTERNAL);
    uv->ADC12D2000_Channel_Mode(BoardNum,2,0,1);	// 2-channel mode calibrate
	unsigned int numBlocksToAcquire=10;
	std::cout << "\n\n";
	while (true)
	{
        uv->SetupAcquire(BoardNum,numBlocksToAcquire);
		Sleep(50);
		double mean[MAX_CHANNELS];
		double std_dev[MAX_CHANNELS];
		int min[MAX_CHANNELS];
		int max[MAX_CHANNELS];
        for (unsigned int chanNum=0; chanNum<uv->GetNumChannels(BoardNum) ;chanNum++)
		{
			min[chanNum] = 1000000;
			max[chanNum] = -1000000;
			mean[chanNum] = 0.0;
			std_dev[chanNum] = 0.0;
		}
		for	(unsigned int i=0; i<numBlocksToAcquire; i++ )
		{
			// read a block from the board
            uv->X_Read(BoardNum,sysMem,DIG_BLOCK_SIZE);
		
			// testForGlitches is used to check for sharp discontinuities in the acquired data.
            calcStats *stats = new calcStats(uv->GetNumChannels(BoardNum));
			stats->stats(uv, sysMem ,DIG_BLOCK_SIZE);
            for (unsigned int chanNum=0; chanNum<uv->GetNumChannels(BoardNum) ;chanNum++)
			{
				if (stats->getMin(chanNum) < min[chanNum])
					min[chanNum] = stats->getMin(chanNum);
				if (stats->getMax(chanNum) > max[chanNum])
					max[chanNum] = stats->getMax(chanNum);
				mean[chanNum] += stats->getMean(chanNum);
				std_dev[chanNum] += (stats->getStdDev(chanNum)) * (stats->getStdDev(chanNum));
			}
			delete stats;	
		}
        for (unsigned int chanNum=0; chanNum<uv->GetNumChannels(BoardNum) ;chanNum++)
		{			
			mean[chanNum] /= numBlocksToAcquire;
			std_dev[chanNum] /= numBlocksToAcquire;
			std_dev[chanNum] = sqrt(std_dev[chanNum]);
		}

        for (unsigned int chanNum=0; chanNum<uv->GetNumChannels(BoardNum) ;chanNum++)
		{
			std::cout << " mean ch" << chanNum << "=";
			std::cout.setf( std::ios::fixed, std::ios::floatfield );
			std::cout.width (7);
			std::cout.precision (2);
			std::cout << std::left  << mean[chanNum];
			std::cout << " stddev ch" << chanNum << "=";
			std::cout.width (7);
			std::cout << std::left  << std_dev[chanNum];

	
			std::cout.precision (0);
			std::cout << " min ch" << chanNum << "=" << std::setw(7)  << min[chanNum];
			std::cout << " max ch" << chanNum << "=" << std::setw(7)  << max[chanNum];
		}
		std::cout << "\r";
	//	std::cout << std::endl <<  "mins[0]=" << stats->mins[0] << std::endl;

	}

    if(sysMem){uv->X_FreeMem(sysMem);}
	delete uv;
	return 0;
}

