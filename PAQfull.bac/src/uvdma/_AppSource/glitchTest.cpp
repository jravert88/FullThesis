/**
 * @file   glitchTest.cpp
 * @date   August, 2013
 *
 * @brief  Class for testing for sharp discontinuities in data
 *
 */

#include "glitchTest.h"
#include <iostream>



glitchTest::glitchTest(void)
{
	std::cout << "Loaded glitchTest.\n";
	numCalls = 0;
	adcErrValue = 64;
	numErrors = 0;
	errMin = DIG_BLOCK_SIZE / 4;
	errMax = 0;

}

glitchTest::~glitchTest(void)
{
	std::cout << "un-Loaded glitchTest.\n";
}

void glitchTest::testForGlitches(unsigned short BoardNum, uvAPI *uv, unsigned char *buf, size_t buflen)
{
    unsigned int numChannels = uv->GetNumChannels(BoardNum);

	if (numCalls == 0)
	{
		std::cout << "testing for glitches " << numChannels << " channel mode" << std::endl;
        if (uv->IS_ISLA216P(BoardNum))
        {
            lastVal[0] = uv->getSample16(numChannels, buf, 0, 0);
            lastVal[1] = uv->getSample16(numChannels, buf, 0, 1);
            adcErrValue = 350;
        }
        if (uv->IS_adc12d2000(BoardNum))
        {
            lastVal[0] = uv->getSample12(numChannels, buf, 0, 0);
            lastVal[1] = uv->getSample12(numChannels, buf, 0, 1);
            adcErrValue = 64;
        }
	}
	size_t numSamplesPerChannel = buflen;

	numSamplesPerChannel /= 2;  // 2 bytes per word for all boards EXCEPT 8bit models.   Fix this later.


	numSamplesPerChannel /= numChannels;

	for (int i=0; i<numSamplesPerChannel;i++)
	{
		int val[4];
		int delta[4];
		for (int chanNum=0;chanNum<numChannels;chanNum++)
		{
            if (uv->IS_ISLA216P(BoardNum))
            {
                val[chanNum] = uv->getSample16(numChannels, buf, i, chanNum);
            }
            if (uv->IS_adc12d2000(BoardNum))
            {
                val[chanNum] = uv->getSample12(numChannels, buf, i, chanNum);
            }
			delta[chanNum] = val[chanNum] - lastVal[chanNum];
			if (delta[chanNum] >= 0)
			{
				if (delta[chanNum] > adcErrValue)
				{
					if (  (i>=64) || ((numCalls > 0) && (i==0)) ) // catch block boundry  but not first 64
					{
						if (errMin > i)
							errMin = i;
						if (errMax <i)
							errMax = i;
						numErrors++;
					}
					if (numErrors < 20)
						std::cout << "Block " << numCalls << " sample " << i << " ch " << chanNum << " " << delta[chanNum] << std::endl; 
				}
			}
			else
			{
				if (delta[chanNum] < adcErrValue)
				{
					if (  (i>=64) || ((numCalls > 0) && (i==0)) ) // catch block boundry  but not first 64
					{
						if (errMin > i)
							errMin = i;
						if (errMax <i)
							errMax = i;
						numErrors++;
					}
					if (numErrors < 20)
						std::cout << "Block " << numCalls << " sample " << i << " ch " << chanNum << " " << delta[chanNum] << std::endl; 
				}
			}
			lastVal[chanNum] = val[chanNum];
		}
	}

	numCalls++;
}
