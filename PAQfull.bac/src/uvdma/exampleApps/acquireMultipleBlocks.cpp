// acquireMultipleBlocks.cpp : Defines the entry point for the console application.
//

#include <iostream>
#include "uvAPI.h"


int main(int argc, char ** argv)
{
	unsigned short BoardNum = 0;
	unsigned int numBlocksToAcquire = 200;
	int error;

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

	// Initialize settings
	uv->setupBoard(BoardNum);

	// uncomment next line for 2 channel mode  (setup board already puts the board into 2ch mode)
	// uvHandle.ADC12D2000_Channel_Mode_Select(boardNum,2,0,1);

	// uncomment next line for 1 channel mode ain0 input
	// uvHandle.ADC12D2000_Channel_Mode_Select(boardNum,1,1,1);

	// uncomment next line for 1 channel mode ain1 input
//	uv->ADC12D2000_Channel_Mode_Select(BoardNum,1,0,1);

	//prepare board to acquire *numBlocksToAcquire* blocks.  Note: the board stops acquiring after the number of blocks selected. 
	uv->SetupAcquire(BoardNum,numBlocksToAcquire);

/* the board is ready to begin acquisition now.  The first call to
uvHandle.getBlock(*boardNum*) begins acquisition, which continues
uninterrupted until *numBlocksToAcquire* have been acquired.
As blocks become available, they can be read by uvHandle.getBlock(*boardNum*)
It should be noted that uvHandle.getBlock(*boardNum*) is "BLOCKING", and
will not return unitl the block of data is actually on the board.
in some scenarios (such as ECL trigger mode) it may appear to lock
*/
	// read the clock freq
	unsigned int adcClock = uv->getAdcClockFreq(BoardNum); 
	std::cout << " ADC clock freq ~= " << adcClock << "MHz" << std::endl;

	Sleep(1);

	for (int i = 0; i < numBlocksToAcquire; i++)
	{
		error = uv->X_Read(BoardNum,sysMem,DIG_BLOCK_SIZE);

		std::cout << "Reading Block: " << i << "\r";

	}


	return 0;
}

