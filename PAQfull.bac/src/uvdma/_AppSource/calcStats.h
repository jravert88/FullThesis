#pragma once
#include "uvAPI.h"

class calcStats
{
public:
	calcStats( unsigned int numCh);
	~calcStats(void);

	void stats( uvAPI *uv, unsigned char *buf, size_t buflen );
	double getMean(unsigned int channel);
	double getStdDev(unsigned int channel);
	double getMin(unsigned int channel);
	double getMax(unsigned int channel);
	
private:
	unsigned int numChannels;
	double *means;
	double *stddevs;
	int *mins;
	int *maxs;
};

