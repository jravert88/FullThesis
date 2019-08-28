#include "calcStats.h"
#include <iostream>

calcStats::calcStats(unsigned int numCh)
{

	numChannels = numCh;
	means = new double[numChannels]();
	stddevs = new double[numChannels]();
	mins = new int[numChannels]();
	maxs = new int[numChannels]();

	for (int i=0;i<numChannels;i++)
	{
		mins[i] = 1000000;  
		maxs[i] = -1000000;
	}
}


calcStats::~calcStats(void)
{
	delete[] means;
	delete[] stddevs;
	delete[] mins;
	delete[] maxs;

}


void calcStats::stats( uvAPI *uv, unsigned char *buf, size_t buflen )
{
	size_t numSamplesPerChannel = buflen;
	numSamplesPerChannel /= 2;  // 2 bytes per word for all boards EXCEPT 8bit models.   Fix this later.
	numSamplesPerChannel /= numChannels;

	for (int i=1; i<numSamplesPerChannel;i++)
	{
		for (int chanNum=0;chanNum<numChannels;chanNum++)
		{
			int currentSample = (int)uv->getSample16(numChannels, buf, i, chanNum);
			means[chanNum] += currentSample;
			if (mins[chanNum] > currentSample)
				mins[chanNum] = currentSample;
			if (maxs[chanNum] < currentSample)
				maxs[chanNum] = currentSample;
		}
	}
	for (int chanNum=0;chanNum<numChannels;chanNum++)
	{
		means[chanNum] /= numSamplesPerChannel;
	}
	for (int i=1; i<numSamplesPerChannel;i++)
	{
		for (int chanNum=0;chanNum<numChannels;chanNum++)
		{
			double currentSample = (double)uv->getSample16(numChannels, buf, i, chanNum);
			currentSample -= means[chanNum];
			currentSample *= currentSample;
			stddevs[chanNum] += currentSample;
		}
	}
	for (int chanNum=0;chanNum<numChannels;chanNum++)
	{
		stddevs[chanNum] /= numSamplesPerChannel;
		stddevs[chanNum] = sqrt(stddevs[chanNum]);
	}
}

double calcStats::getMean(unsigned int channel)
{
	return means[channel];
}

double calcStats::getStdDev(unsigned int channel)
{
	return stddevs[channel];
}

double calcStats::getMin(unsigned int channel)
{
	return mins[channel];
}

double calcStats::getMax(unsigned int channel)
{
	return maxs[channel];
}