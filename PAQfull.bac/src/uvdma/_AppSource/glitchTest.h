#ifndef GLITCHTEST_H
#define GLITCHTEST_H

#pragma once
#define GLICH_RECORD_LEN 1024
#include "uvAPI.h"

class glitchTest
{
public:
	glitchTest(void);
	~glitchTest(void);

    void testForGlitches(unsigned short BoardNum, uvAPI *uv, unsigned char *buf, size_t buflen);
	unsigned int numErrors;
	unsigned int errMin;
	unsigned int errMax;

private:
	int lastVal[4];
	int glitchList[4][GLICH_RECORD_LEN];
	unsigned int numCalls;
	unsigned int adcErrValue;
};

#endif
