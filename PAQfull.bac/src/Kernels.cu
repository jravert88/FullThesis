/*
 * File:   Kernels.h
 * Author: Andrew McMurdie
 * 		   Eric Swindlehurst
 *
 * Created on April 3, 2013, 8:44 AM
 */

#include "Kernels.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cuda_device_runtime_api.h"
#include "cudaProfiler.h"
//#include <cublas.h>
//#include <cublas_api.h>
#include <cublas_v2.h>
#include <time.h>
#include "cuComplex.h"
//#include <stdio.h>

const int PREAMBLE_LENGTH = 256;
const int FULL_PACKET_LENGTH = 12672;
const int LQ = 32;

__global__ void runFilterCuda(float* I, float* Q, int samplesLength, float* filter, int filterLength, float* filtered_I, float* filtered_Q, int convLength) {

	//This thread will process the result at the calculated sample
	int sampleIndex = (blockIdx.x * blockDim.x) + threadIdx.x;

	//Don't run if the index is off the end of our output
	if(sampleIndex >= convLength) return;

	int index;
	float sumI, sumQ;
	sumI = 0;
	sumQ = 0;

	//Calculate the sum at this point. We only iterate over the samples where the filter and the input samples overlap.
	//All the other points have a product of zero (because either the filter or the samples are zero), so we don't bother
	//iterating over them or calculating the zeros to sum in.
	for(int j=sampleIndex-filterLength+1; j <= sampleIndex; j++) {
		index = sampleIndex-j;
		if(/*(index >= 0) && */(j < samplesLength) && (j >= 0)) {
			sumI += filter[index] * I[j];
			sumQ += filter[index] * Q[j];
		}
	}

	//Save data
	filtered_I[sampleIndex] = sumI;
	filtered_Q[sampleIndex] = sumQ;
}

__global__ void cudaRunComplexFilter(float* I, float* Q, int samplesLength, float* hr, float* hi, int filterLength, float* filtered_I, float* filtered_Q, int convLength) {

	//This thread will process the result at the calculated sample
	int sampleIndex = (blockIdx.x * blockDim.x) + threadIdx.x;

	//Don't run if the index is off the end of our output
	if(sampleIndex >= convLength) return;

	int index;
	float sumI, sumQ;
	sumI = 0;
	sumQ = 0;

	//Calculate the sum at this point. We only iterate over the samples where the filter and the input samples overlap.
	//All the other points have a product of zero (because either the filter or the samples are zero), so we don't bother
	//iterating over them or calculating the zeros to sum in.
	for(int j=sampleIndex-filterLength+1; j <= sampleIndex; j++) {
		index = sampleIndex-j;
		if(/*(index >= 0) && */(j < samplesLength) && (j >= 0)) {
			//sumI += filter[index] * I[j];
			//sumQ += filter[index] * Q[j];
			sumI += (I[j] * hr[index]) - (Q[j] * hi[index]);
			sumQ += (I[j] * hi[index]) + (Q[j] * hr[index]);
		}
	}

	//Save data
	filtered_I[sampleIndex] = sumI;
	filtered_Q[sampleIndex] = sumQ;
}



__global__ void downsampleCuda(float* I, float* Q, unsigned int numDownsampledSamples, float* downsampled_I, float* downsampled_Q, unsigned int factor) {
	//Get the index this thread represents
	int sampleIndex = (blockIdx.x * blockDim.x) + threadIdx.x;

	//Don't process beyond the end of our input samples
	if(sampleIndex >= numDownsampledSamples) return;

	//Find out what sample in the I and Q lists this thread represents
	unsigned int absoluteIndex = sampleIndex * factor;

	//Assign this to the downsampled list.
	downsampled_I[sampleIndex] = I[absoluteIndex];
	downsampled_Q[sampleIndex] = Q[absoluteIndex];
}

//__device__ float atomicAdd(float* address, float val)
//{
//    unsigned long long int* address_as_ull = (unsigned long long int*)address;
//    unsigned long long int old = *address_as_ull, assumed;
//    do {
//        assumed = old;
//		old = atomicCAS(address_as_ull, assumed, __float_as_longlong(val + __longlong_as_float(assumed)));
//    } while (assumed != old);
//    return __longlong_as_float(old);
//}

__device__ int sign(float input)
{
	if(input == 0)
		return 0;

	int ret_value = input > 0 ? 1 : -1;
	return ret_value;
}

__device__ int cudaDecodeBits(float i, float q)
{
	int xBit, yBit;
	xBit = (i < 0) ? 1 : 0;
	yBit = (q < 0) ? 1 : 0;

	int bitIndex = (xBit << 1) | yBit;

	return bitIndex;
}

__device__ float cudaCalculateErrorPED(float ped1, float ped2, float xInterpolant, float xDelayedTwo, float xDelayedThree, float yInterpolant, float yDelayedTwo, float yDelayedThree, int state)
{
	float signX, y;

	switch(state)
	{
	case 0:
	case 1:
	case 2:
	case 3:
	case 5:
	case 7:
		signX = sign(ped1);
		y = ped2;
		break;
	case 4:
		signX = sign(xDelayedThree);
		y = yDelayedThree;
		break;
	case 6:
		signX = sign(xDelayedTwo);
		y = yDelayedTwo;
		break;
	default:
		return -9999; //Return garbage number
	}

	float productOne = signX * y;

	float signY = sign(yInterpolant);
	float productTwo = xInterpolant * signY;

	float output = productOne - productTwo;

	switch(state) {
	case 0:
	case 1:
	case 2:
	case 3:
	case 7:
		return 0;
	case 4:
	case 5:
	case 6:
		return output;
	default:
		return -9999;
	}
}

__device__ float cudaCalcTED2(float ted1, float xDelayedTwo, float xDelayedThree, int state)
{
	float firstSwitch = 0;
	switch(state)
	{
	case 0:
	case 1:
	case 2:
	case 3:
	case 5:
	case 7:
		firstSwitch = ted1;
		break;
	case 4:
		firstSwitch = xDelayedThree;
		break;
	case 6:
		firstSwitch = xDelayedTwo;
		break;
	}
	return firstSwitch;
}

__device__ float cudaCalcTimingError(float ted1, float ted2, float ted3, float ted4, float ted5, float xInterpolant, float xDelayedTwo, float xDelayedThree, float yInterpolant, int state)
{
	float firstSwitch = cudaCalcTED2(ted1, xDelayedTwo, xDelayedThree, state);
	float summandOne = sign(ted3);
	float summandTwo = sign(firstSwitch);
	float productOne = ted2 * (summandOne - summandTwo);

	summandOne = sign(ted5);
	summandTwo = sign(yInterpolant);

	float productTwo = ted4 * (summandOne - summandTwo);
	float switchInput = productOne + productTwo;

	switch(state)
	{
	case 0:
	case 1:
	case 2:
	case 3:
	case 4:
	case 6:
	case 7:
		return 0;
	case 5:
		return switchInput;
	default:
		return -5555;
	}
}

__global__ void cudaDemodLoop(float* i_samples, float* q_samples, int sample_size, int* bitdecisions, float* constants)
{
	float dds = 0;
	float ibuf[4] = {0};
	float qbuf[4] = {0};
	float mu = 0;
	float M = 0;
	float FI1 = 0, FI2 = 0, FI3 = 0;
	float FQ1 = 0, FQ2 = 0, FQ3 = 0;
	unsigned short strobe = 0;
	float B1 = 0;
	int bit_index = 0;
	int state = 0;
	float ped1 = 0, ped2 = 0;
	float VIp = 0, VIt = 0;
	float ted1 = 0, ted2 = 0, ted3 = 0, ted4 = 0, ted5 = 0;
	float NCO = 0, OLD_NCO = 0;
	float s1 = 0, s2 = 0;
	float K1p = constants[0];
	float K2p = constants[1];
	float K1t = constants[2];
	float K2t = constants[3];

	for(int i = 0; i < sample_size; i++)
	{
		// Derotate and add to queues
		float xr = (cos(dds) * i_samples[i]) + (sin(dds) * q_samples[i]);
		float yr = (cos(dds) * q_samples[i]) - (sin(dds) * i_samples[i]);
		for(int i = 3; i > 0; i--)
		{
			ibuf[i] = ibuf[i-1];
			qbuf[i] = qbuf[i-1];
		}
		ibuf[0] = xr;
		qbuf[0] = yr;

		mu = M;

		// Interpolate
		float h0 = 0.5 * mu * (mu-1);
		float h1 = -0.5 * mu * (mu-3);
		float h2 = (-0.5 * mu * (mu+1))+1;
		float h3 = h0;

		float i_prime = (h0*xr) + (h1*FI1) + (h2*FI2) + (h3*FI3);
		float q_prime = (h0*yr) + (h1*FQ1) + (h2*FQ2) + (h3*FQ3);

		// Bit decision
		float q_dec;
		float i_dec;
		if(strobe)
		{
			q_dec = q_prime;
			switch(state)
			{
			case 0: // Shouldn't happen
				break;
			case 1:
			case 2:
			case 3:
			case 5:
				i_dec = B1;
				break;
			case 4:
				i_dec = ibuf[3];
				break;
			case 6:
				i_dec = ibuf[2];
				break;
			case 7:
				// This shouldn't happen. Return garbage value
				return;
			}
			// Bit decision
			bitdecisions[bit_index] = cudaDecodeBits(i_dec, q_dec);
			bit_index++;
		}

		// Detect phase error
		float ep = cudaCalculateErrorPED(ped1, ped2,
				i_prime, ibuf[2], ibuf[3],
				q_prime, qbuf[2], qbuf[3],
				state);
		float vp = (K1p*ep) + (K2p*ep) + VIp;

		// Detect timing error
		float et = cudaCalcTimingError(ted1, ted2, ted3, ted4, ted5,
				i_prime, ibuf[2], ibuf[3],
				q_prime, state);
		float vt = (K1t*et) + (K2t*et) + VIt;


		// State updates
		FI3 = FI2; FI2 = FI1; FI1 = xr;
		FQ3 = FQ2; FQ2 = FQ1; FQ1 = yr;

		B1 = i_prime;

		ped1 = i_prime;
		ped2 = q_prime;
		VIp = VIp + K2p * ep;
		dds = dds + vp;

		ted3 = ted2;
		ted2 = cudaCalcTED2(ted1, ibuf[2], ibuf[3], state);
		ted1 = i_prime;
		ted5 = ted4;
		ted4 = q_prime;

		VIt = VIt + K2t*et;
		OLD_NCO = NCO;

		NCO = fmod(NCO, 1) - vt - 0.5;
		if(NCO < 0)
		{
			strobe = 1;
			M = 2*fmod(OLD_NCO, 1);
			NCO += 1;
		}
		else
			strobe = 0;

		state = (4*strobe) + 2*s1 + s2;
		s2 = s1;
		s1 = strobe;
	}
}

__global__ void cudaConvertToBits(int* bit_decisions, unsigned short* bit_stream, int dec_size)
{
	int dec_index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int bit_index = dec_index * 2;

	if(dec_index >= dec_size)
		return;
	int curr_decision = bit_decisions[dec_index];

	bit_stream[bit_index] = ((curr_decision & 2) >> 1);
	bit_stream[bit_index+1] = (curr_decision & 1);
}

__global__ void cudaDecodeBitstream(unsigned short* encoded, unsigned short* decoded, int size)
{
	int bit_index = (((blockIdx.x * blockDim.x) + threadIdx.x) * 2) + 2;

	if(bit_index >= size)
		return;
	unsigned short curr_bit = encoded[bit_index];

	decoded[bit_index] = !encoded[bit_index-1] ^ curr_bit;
	decoded[bit_index+1] = curr_bit ^ encoded[bit_index+1];
}

// Miscellaneous functions for the preamble detector
__global__ void cudaFindMax(int* results, float* u_sums, int size)
{
	// Get beginning and ending locations
	int start = blockIdx.x * FULL_PACKET_LENGTH;
	int finish = start + FULL_PACKET_LENGTH;
	if(finish > size)
		finish = size;

	float max = 0;
	int location = 0;

	// Go through symbols start-finish and find max
	for(int i = start; i < finish; i++)
	{
		float current = u_sums[i];
		if(max < current)
		{
			max = current;
			location = i;
		}
	}
	// Store result
	results[blockIdx.x] = location;
}

__global__ void cudaFindMaxOptimized(float* u_sums, int size, float* maxes, float* locs)
{
	int section_size = (FULL_PACKET_LENGTH)/gridDim.x;
	int start = blockIdx.y*FULL_PACKET_LENGTH + blockIdx.x*section_size;
	int finish = start + section_size;
	if(finish > size)
		finish = size;

	int max_index = blockIdx.y*gridDim.x + blockIdx.x;
	int location = 0;
	float max = 0;

	// Go through symbols start-finish and find max
	for(int i = start; i < finish; i++)
	{
		float current = u_sums[i];
		if(max < current)
		{
			max = current;
			location = i;
		}
	}
	maxes[max_index] = max;
	locs[max_index] = location;
}

__global__ void cudaFindMaxOptimizedPart2(int* results, float* maxes, float* locations, int num_sections)
{
	float max = 0;
	float location = 0;
	int start = blockIdx.x*num_sections;
	int finish = start + num_sections;
	for(int j = start; j < finish; j++)
	{
		float current = maxes[j];
		if(max < current)
		{
			max = current;
			location = locations[j];
		}
	}
	results[blockIdx.x] = location;
}


/**
 * Calculates the inner sums for every available block and stores them.
 */
__global__ void calculateInnerSumBlocks(float* i, float* q, float* innerSums, int uLength, int innerSumsLength) {

	int u = blockDim.x * blockIdx.x + threadIdx.x;
	if(u >= innerSumsLength) return; //Don't process off the end of the array.

	//	float real = i[0] - i[4] + i[8] + i[16] - i[20] + i[24]
	//
	//				   + q[2] + q[10] + q[14] + q[22] - q[6] - q[18] - q[26] - q[30]
	//
	//				   + .7071 * (i[1] - i[3] - i[5] + i[7] + i[9] - i[11] - i[12] - i[13] + i[15] + i[17] - i[19] - i[21] + i[23] + i[25] - i[27] - i[28] - i[29] + i[31]
	//
	//					   + q[1] + q[3] - q[5] - q[7] + q[9] + q[11] + q[12] + q[13] + q[15] - q[17] - q[19] + q[21] + q[23] - q[25] - q[27] - q[28] - q[29] - q[31]
	//
	//				   );
	//
	//	float imag = q[0] - q[4] + q[8] + q[16] - q[20] + q[24]
	//
	//				   - i[2] + i[6] - i[10] - i[14] + i[18] - i[22] + i[26] + i[30]
	//
	//				   + .7071 * (q[1] - q[3] - q[5] + q[7] + q[9] - q[11] - q[12] - q[13] + q[15] + q[17] - q[19] - q[21] + q[23] + q[25] - q[27] - q[28] - q[29] + q[31]
	//
	//					   - i[1] - i[3] + i[5] + i[7] - i[9] - i[11] - i[12] - i[13] - i[15] + i[17] + i[19] - i[21] - i[23] + i[25] + i[27] + i[28] + i[29] + i[31]
	//
	//				   );

	//Rewritten so that accesses to q and i are sequential
	//------Real------
	float real, real7;
	real   = i[u + 0];
	real7  = i[u + 1];
	//i[u + 2] is multiplied by zero
	real7 -= i[u + 3];
	real  -= i[u + 4];
	real7 -= i[u + 5];
	//i[u + 6] is multiplied by zero
	real7 += i[u + 7];
	real  += i[u + 8];
	real7 += i[u + 9];
	//i[u + 10] is multiplied by zero
	real7 -= i[u + 11];
	real7 -= i[u + 12];
	real7 -= i[u + 13];
	//i[u + 14] is multiplied by zero
	real7 += i[u + 15];
	real  += i[u + 16];
	real7 += i[u + 17];
	//i[u + 18] is multiplied by zero
	real7 -= i[u + 19];
	real  -= i[u + 20];
	real7 -= i[u + 21];
	//i[u + 22] is multiplied by zero
	real7 += i[u + 23];
	real  += i[u + 24];
	real7 += i[u + 25];
	//i[u + 26] is multiplied by zero
	real7 -= i[u + 27];
	real7 -= i[u + 28];
	real7 -= i[u + 29];
	//i[u + 30] is multiplied by zero
	real7 += i[u + 31];

	//q[u + 0] is multiplied by zero
	real7 += q[u + 1];
	real  += q[u + 2];
	real7 += q[u + 3];
	//q[u + 4] is multiplied by zero
	real7 -= q[u + 5];
	real  -= q[u + 6];
	real7 -= q[u + 7];
	//q[u + 8] is multiplied by zero
	real7 += q[u + 9];
	real  += q[u + 10];
	real7 += q[u + 11];
	real7 += q[u + 12];
	real7 += q[u + 13];
	real  += q[u + 14];
	real7 += q[u + 15];
	//q[u + 16] is multiplied by zero
	real7 -= q[u + 17];
	real  -= q[u + 18];
	real7 -= q[u + 19];
	//q[u + 20] is multiplied by zero
	real7 += q[u + 21];
	real  += q[u + 22];
	real7 += q[u + 23];
	//q[u + 24] is multiplied by zero
	real7 -= q[u + 25];
	real  -= q[u + 26];
	real7 -= q[u + 27];
	real7 -= q[u + 28];
	real7 -= q[u + 29];
	real  -= q[u + 30];
	real7 -= q[u + 31];

	real = real + .7071*real7;

	//------Imaginary------
	float imag, imag7;
	imag   = q[u + 0];
	imag7  = q[u + 1];
	//q[u + 2] is multiplied by zero
	imag7 -= q[u + 3];
	imag  -= q[u + 4];
	imag7 -= q[u + 5];
	//q[u + 6] is multiplied by zero
	imag7 += q[u + 7];
	imag  += q[u + 8];
	imag7 += q[u + 9];
	//q[u + 10] is multiplied by zero
	imag7 -= q[u + 11];
	imag7 -= q[u + 12];
	imag7 -= q[u + 13];
	//q[u + 14] is multiplied by zero
	imag7 += q[u + 15];
	imag  += q[u + 16];
	imag7 += q[u + 17];
	//q[u + 18] is multiplied by zero
	imag7 -= q[u + 19];
	imag  -= q[u + 20];
	imag7 -= q[u + 21];
	//q[u + 22] is multiplied by zero
	imag7 += q[u + 23];
	imag  += q[u + 24];
	imag7 += q[u + 25];
	//q[u + 26] is multiplied by zero
	imag7 -= q[u + 27];
	imag7 -= q[u + 28];
	imag7 -= q[u + 29];
	//q[u + 30] is multiplied by zero
	imag7 += q[u + 31];

	//q[u + 0] is multiplied by zero
	imag7 -= i[u + 1];
	imag  -= i[u + 2];
	imag7 -= i[u + 3];
	//i[u + 4] is multiplied by zero
	imag7 += i[u + 5];
	imag  += i[u + 6];
	imag7 += i[u + 7];
	//i[u + 8] is multiplied by zero
	imag7 -= i[u + 9];
	imag  -= i[u + 10];
	imag7 -= i[u + 11];
	imag7 -= i[u + 12];
	imag7 -= i[u + 13];
	imag  -= i[u + 14];
	imag7 -= i[u + 15];
	//i[u + 16] is multiplied by zero
	imag7 += i[u + 17];
	imag  += i[u + 18];
	imag7 += i[u + 19];
	//i[u + 20] is multiplied by zero
	imag7 -= i[u + 21];
	imag  -= i[u + 22];
	imag7 -= i[u + 23];
	//i[u + 24] is multiplied by zero
	imag7 += i[u + 25];
	imag  += i[u + 26];
	imag7 += i[u + 27];
	imag7 += i[u + 28];
	imag7 += i[u + 29];
	imag  += i[u + 30];
	imag7 += i[u + 31];

	imag = imag + .7071*imag7;

	//Save in array
	int saveAddress = 2*u;
	innerSums[saveAddress] = real;
	innerSums[saveAddress + 1] = imag;

}

/**
 * Calculates the outer sums and enters the values into L
 */
__global__ void calculateOuterSums(float* innerSums, float* L, int uLength) {

	int u = blockDim.x*blockIdx.x + threadIdx.x;
	if(u >= uLength) return; //Don't run off the end of the array.

	float real, imag, u_sum;
	int realIdx = 2*u;
	int imagIdx = realIdx+1;

	//Block 0
	real = innerSums[realIdx];
	imag = innerSums[imagIdx];
	u_sum = (real*real) + (imag*imag);

	//Block 1
	realIdx += 64;
	imagIdx += 64;
	real = innerSums[realIdx];
	imag = innerSums[imagIdx];
	u_sum += (real*real) + (imag*imag);

	//Block 2
	realIdx += 64;
	imagIdx += 64;
	real = innerSums[realIdx];
	imag = innerSums[imagIdx];
	u_sum += (real*real) + (imag*imag);

	//Block 3
	realIdx += 64;
	imagIdx += 64;
	real = innerSums[realIdx];
	imag = innerSums[imagIdx];
	u_sum += (real*real) + (imag*imag);

	//Block 4
	realIdx += 64;
	imagIdx += 64;
	real = innerSums[realIdx];
	imag = innerSums[imagIdx];
	u_sum += (real*real) + (imag*imag);

	//Block 5
	realIdx += 64;
	imagIdx += 64;
	real = innerSums[realIdx];
	imag = innerSums[imagIdx];
	u_sum += (real*real) + (imag*imag);

	//Block 6
	realIdx += 64;
	imagIdx += 64;
	real = innerSums[realIdx];
	imag = innerSums[imagIdx];
	u_sum += (real*real) + (imag*imag);

	//Block 7
	realIdx += 64;
	imagIdx += 64;
	real = innerSums[realIdx];
	imag = innerSums[imagIdx];
	u_sum += (real*real) + (imag*imag);

	//Save result
	L[u] = u_sum;
}

/**
 * Simple Correlator
 */
__global__ void cudaSimpleCorrelator(float* xi, float* xq, float* sr, float* si, int sLength, float* L, int uLength) {
	//This thread will process the result at the calculated sample
	int u = (blockIdx.x * blockDim.x) + threadIdx.x;

	//Don't process off the end of the array.
	if(u >= uLength) return;

	float real = 0;
	float imag = 0;
	float a, b, c, d;

	for(int n = u; n < u+sLength; n++) {
		a = xi[n];
		b = xq[n];
		c = sr[n-u];
		d = si[n-u] * (-1);

		real +=	(a*c) - (b*d);
		imag += (a*d) + (b*c);
	}

	L[u] = sqrt(real*real + imag*imag);
}

/**
 * Choi Lee Correlator. Solves for the LHS sum.
 */
__global__ void cudaChoiLee(float* xi, float* xq, float* sr, float* si, int N, float* L) {
	//This thread will process the result at the calculated sample
	int u = (blockIdx.x * blockDim.x) + threadIdx.x;

	//Don't process off the end of the array;
	if(u >= N) return;

	float uSum = 0;
	float r_i, r_q, rconj_i, rconj_q;
	float s_i, s_q, sconj_i, sconj_q;

	float rsum_i, rsum_q, ssum_i, ssum_q;
	float ksum_i, ksum_q;

	for(int i=0; i < N; i++) {
		//printf("On i: %d\n", i);
		ksum_i = 0;
		ksum_q = 0;
		for (int k=0; k < N-i; k++) {
			r_i 	= xi[u+k+i];
			r_q 	= xq[u+k+i];
			rconj_i = xi[u+k];
			rconj_q = xq[u+k] * (-1);

			s_i 	= sr[k];
			s_q 	= si[k];
			sconj_i = sr[k+i];
			sconj_q = si[k+i] *(-1);

			rsum_i = (r_i * rconj_i) - (r_q * rconj_q);
			rsum_q = (r_i * rconj_q) + (r_q * rconj_i);

			ssum_i = (s_i * sconj_i) - (s_q * sconj_q);
			ssum_q = (s_i * sconj_q) + (s_q * sconj_i);

			ksum_i += (rsum_i * ssum_i) - (rsum_q * ssum_q);
			ksum_q += (rsum_i * ssum_q) + (rsum_q * ssum_i);
		}
		//Add the absolute value of the inner sum
		uSum += sqrt((ksum_i * ksum_i) + (ksum_q * ksum_q));
	}

	L[u] = uSum;
}


__global__ void cudaSumDataCorrection(float* i_samples, float* q_samples, float* r, int N) {
	//This thread will process the result at the calculated sample
	int u = (blockIdx.x * blockDim.x) + threadIdx.x;

	//Don't process off the end of the array;
	if(u >= N) return;

	float correctionTerm = 0;
	float innerTerm;
	float r_i, r_q, r2_i, r2_q;
	//float real, imag;

	for(int i=0; i < PREAMBLE_LENGTH; i++) {
		innerTerm = 0;
		for(int k=u+i; k < u+PREAMBLE_LENGTH; k++) {
			r_i = i_samples[k];
			r_q = q_samples[k];
			r2_i = i_samples[k-i];
			r2_q = q_samples[k-i];

			innerTerm += sqrt(r_i*r_i + r_q*r_q) * sqrt(r2_i*r2_i + r2_q*r2_q);
		}
		correctionTerm += innerTerm;
	}

	r[u] = correctionTerm;
}


__global__ void cudaAddCorrAndCorrection(float* L, float* r, int N) {
	//This thread will process the result at the calculated sample
	int u = (blockIdx.x * blockDim.x) + threadIdx.x;

	//Don't process off the end of the array;
	if(u >= N) return;

	//Subtract correction term from correlation term.
	L[u] -= r[u];
}


__global__ void cudaChoiLeeFull(float* xi, float* xq, float* sr, float* si, int N, float* L, float* L2, float* r) {
	//This thread will process the result at the calculated sample
	int u = (blockIdx.x * blockDim.x) + threadIdx.x;

	//Don't process off the end of the array;
	if(u >= N) return;

	float uSum = 0;
	float r_i, r_q, rconj_i, rconj_q;
	float s_i, s_q, sconj_i, sconj_q;

	float rsum_i, rsum_q, ssum_i, ssum_q;
	float ksum_i, ksum_q;

	for(int i=1; i < PREAMBLE_LENGTH; i++) {
		//printf("On i: %d\n", i);
		ksum_i = 0;
		ksum_q = 0;
		for (int k=0; k < PREAMBLE_LENGTH-i; k++) {
			//			r_i 	= xi[u+k+i];
			//			r_q 	= xq[u+k+i];
			//			rconj_i = xi[u+k];
			//			rconj_q = xq[u+k] * (-1);
			//
			//			s_i 	= sr[k];
			//			s_q 	= si[k];
			//			sconj_i = sr[k+i];
			//			sconj_q = si[k+i] *(-1);

			r_i 	= xi[u+k];
			r_q 	= xq[u+k];
			rconj_i = xi[u+k+i];
			rconj_q = xq[u+k+i] * (-1);

			s_i 	= sr[k+i];
			s_q 	= si[k+i];
			sconj_i = sr[k];
			sconj_q = si[k] *(-1);

			rsum_i = (r_i * rconj_i) - (r_q * rconj_q);
			rsum_q = (r_i * rconj_q) + (r_q * rconj_i);

			ssum_i = (s_i * sconj_i) - (s_q * sconj_q);
			ssum_q = (s_i * sconj_q) + (s_q * sconj_i);

			ksum_i += (rsum_i * ssum_i) - (rsum_q * ssum_q);
			ksum_q += (rsum_i * ssum_q) + (rsum_q * ssum_i);
		}
		//Add the absolute value of the inner sum
		uSum += sqrt((ksum_i * ksum_i) + (ksum_q * ksum_q));
	}

	L[u] = uSum;


	//Add correction term
	float correctionTerm = 0;
	float innerTerm;
	//float r_i, r_q,
	float r2_i, r2_q;
	//float real, imag;

	for(int i=1; i < PREAMBLE_LENGTH; i++) {
		innerTerm = 0;
		for(int k=u+i; k < u+PREAMBLE_LENGTH; k++) {
			r_i = xi[k];
			r_q = xq[k];
			r2_i = xi[k-i];
			r2_q = xq[k-i];

			innerTerm += sqrt(r_i*r_i + r_q*r_q) * sqrt(r2_i*r2_i + r2_q*r2_q);
		}
		correctionTerm += innerTerm;
	}

	r[u] = correctionTerm;

	//Subtract correction term from correlation term.
	//L[u] -= r[u];
	L2[u] = uSum - correctionTerm;
}

__global__ void cudaChoiLeeFullFromPaper(float* xi, float* xq, float* sr, float* si, int N, float* L, float* L2, float* r) {
	//This thread will process the result at the calculated sample
	int u = (blockIdx.x * blockDim.x) + threadIdx.x;

	//Don't process off the end of the array;
	if(u >= N) return;

	float uSum = 0;
	float r_i, r_q, rconj_i, rconj_q;
	float s_i, s_q, sconj_i, sconj_q;

	float rsum_i, rsum_q, ssum_i, ssum_q;
	float ksum_i, ksum_q;

	for(int i=1; i < PREAMBLE_LENGTH; i++) {
		//printf("On i: %d\n", i);
		ksum_i = 0;
		ksum_q = 0;
		for (int k=i; k < PREAMBLE_LENGTH; k++) {
			//			r_i 	= xi[u+k+i];
			//			r_q 	= xq[u+k+i];
			//			rconj_i = xi[u+k];
			//			rconj_q = xq[u+k] * (-1);
			//
			//			s_i 	= sr[k];
			//			s_q 	= si[k];
			//			sconj_i = sr[k+i];
			//			sconj_q = si[k+i] *(-1);

			r_i 	= xi[u+k-i];
			r_q 	= xq[u+k-i];
			rconj_i = xi[u+k];
			rconj_q = xq[u+k] * (-1);

			s_i 	= sr[k];
			s_q 	= si[k];
			sconj_i = sr[k-i];
			sconj_q = si[k-i] *(-1);

			rsum_i = (r_i * rconj_i) - (r_q * rconj_q);
			rsum_q = (r_i * rconj_q) + (r_q * rconj_i);

			ssum_i = (s_i * sconj_i) - (s_q * sconj_q);
			ssum_q = (s_i * sconj_q) + (s_q * sconj_i);

			ksum_i += (rsum_i * ssum_i) - (rsum_q * ssum_q);
			ksum_q += (rsum_i * ssum_q) + (rsum_q * ssum_i);
		}
		//Add the absolute value of the inner sum
		uSum += sqrt((ksum_i * ksum_i) + (ksum_q * ksum_q));
	}

	L[u] = uSum;


	//Add correction term
	float correctionTerm = 0;
	float innerTerm;
	//float r_i, r_q,
	float r2_i, r2_q;
	//float real, imag;

	for(int i=1; i < PREAMBLE_LENGTH; i++) {
		innerTerm = 0;
		for(int k=u+i; k <= u+PREAMBLE_LENGTH-1; k++) {
			r_i = xi[k];
			r_q = xq[k];
			r2_i = xi[k-i];
			r2_q = xq[k-i];

			innerTerm += sqrt(r_i*r_i + r_q*r_q) * sqrt(r2_i*r2_i + r2_q*r2_q);
		}
		correctionTerm += innerTerm;
	}

	r[u] = correctionTerm;

	//Subtract correction term from correlation term.
	//L[u] -= r[u];
	L2[u] = uSum - correctionTerm;
}




/**
 * BYU Simplified Detector, with multiplies
 */
__global__ void cudaBYUSimplified(float* xi, float* xq, float* sr, float* si, int N, int Lq, float *L) {

	int u = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(u >= N) return;

	float uSum = 0;
	//float innerSum = 0;
	float r_i, r_q, q_i, q_q;
	float realPart, imagPart;

	for(int k=0; k <= 7; k++) {
		realPart = 0;
		imagPart = 0;

		for(int l = 0; l < Lq; l++) {
			r_i = xi[u+k*Lq+l];
			r_q = xq[u+k*Lq+l];
			q_i = sr[l];
			q_q = si[l]*(-1);

			realPart += (r_i * q_i) - (r_q * q_q);
			imagPart += (r_i * q_q) + (r_q * q_i);
		}

		uSum += (realPart*realPart) + (imagPart*imagPart);
	}

	L[u] = uSum;
}

/**
 * L-CL-2
 */
__global__ void cudaChoiLee2And3(float* xi, float* xq, float* sr, float* si, int N, float* L, float* L2, float* r) {
	//This thread will process the result at the calculated sample
	int u = (blockIdx.x * blockDim.x) + threadIdx.x;

	//Don't process off the end of the array;
	if(u >= N) return;

	float uSum = 0;
	float r_i, r_q, rconj_i, rconj_q;
	float s_i, s_q, sconj_i, sconj_q;

	float rsum_i, rsum_q, ssum_i, ssum_q;
	float ksum_i, ksum_q;

	//	for(int i=1; i < 2; i++) {
	//printf("On i: %d\n", i);
	ksum_i = 0;
	ksum_q = 0;
	for (int k=0; k <= PREAMBLE_LENGTH-2; k++) {
		//			r_i 	= xi[u+k+i];
		//			r_q 	= xq[u+k+i];
		//			rconj_i = xi[u+k];
		//			rconj_q = xq[u+k] * (-1);
		//
		//			s_i 	= sr[k];
		//			s_q 	= si[k];
		//			sconj_i = sr[k+i];
		//			sconj_q = si[k+i] *(-1);

		r_i 	= xi[u+k];
		r_q 	= xq[u+k];
		rconj_i = xi[u+k+1];
		rconj_q = xq[u+k+1] * (-1);

		s_i 	= sr[k+1];
		s_q 	= si[k+1];
		sconj_i = sr[k];
		sconj_q = si[k] *(-1);

		rsum_i = (r_i * rconj_i) - (r_q * rconj_q);
		rsum_q = (r_i * rconj_q) + (r_q * rconj_i);

		ssum_i = (s_i * sconj_i) - (s_q * sconj_q);
		ssum_q = (s_i * sconj_q) + (s_q * sconj_i);

		ksum_i += (rsum_i * ssum_i) - (rsum_q * ssum_q);
		ksum_q += (rsum_i * ssum_q) + (rsum_q * ssum_i);
	}
	//Add the absolute value of the inner sum
	uSum = sqrt((ksum_i * ksum_i) + (ksum_q * ksum_q));
	//}

	L[u] = uSum;


	//Add correction term
	float correctionTerm = 0;
	float innerTerm;
	//float r_i, r_q,
	float r2_i, r2_q;
	//float real, imag;


	innerTerm = 0;
	for(int k=u+1; k <= u+PREAMBLE_LENGTH-1; k++) {
		r_i = xi[k];
		r_q = xq[k];
		r2_i = xi[k-1];
		r2_q = xq[k-1];

		innerTerm += sqrt(r_i*r_i + r_q*r_q) * sqrt(r2_i*r2_i + r2_q*r2_q);
	}
	correctionTerm = innerTerm;


	r[u] = correctionTerm;

	//Subtract correction term from correlation term.
	L2[u] = uSum - correctionTerm;
}

/**
 * L-CL-2
 */
__global__ void cudaChoiLee2And3FromPaper(float* xi, float* xq, float* sr, float* si, int N, float* L, float* L2, float* r) {
	//This thread will process the result at the calculated sample
	int u = (blockIdx.x * blockDim.x) + threadIdx.x;

	//Don't process off the end of the array;
	if(u >= N) return;

	float uSum = 0;
	float r_i, r_q, rconj_i, rconj_q;
	float s_i, s_q, sconj_i, sconj_q;

	float rsum_i, rsum_q, ssum_i, ssum_q;
	float ksum_i, ksum_q;


	ksum_i = 0;
	ksum_q = 0;
	for (int k=1; k <= PREAMBLE_LENGTH-1; k++) {

		r_i 	= xi[u+k-1];
		r_q 	= xq[u+k-1];
		rconj_i = xi[u+k];
		rconj_q = xq[u+k] * (-1);

		s_i 	= sr[k];
		s_q 	= si[k];
		sconj_i = sr[k-1];
		sconj_q = si[k-1] *(-1);

		rsum_i = (r_i * rconj_i) - (r_q * rconj_q);
		rsum_q = (r_i * rconj_q) + (r_q * rconj_i);

		ssum_i = (s_i * sconj_i) - (s_q * sconj_q);
		ssum_q = (s_i * sconj_q) + (s_q * sconj_i);

		ksum_i += (rsum_i * ssum_i) - (rsum_q * ssum_q);
		ksum_q += (rsum_i * ssum_q) + (rsum_q * ssum_i);
	}
	//Add the absolute value of the inner sum
	uSum = sqrt((ksum_i * ksum_i) + (ksum_q * ksum_q));
	//}

	L[u] = uSum;


	//Add correction term
	float correctionTerm = 0;
	float innerTerm;
	//float r_i, r_q,
	float r2_i, r2_q;
	//float real, imag;


	innerTerm = 0;
	for(int k=u+1; k <= u+PREAMBLE_LENGTH-1; k++) {
		r_i = xi[k];
		r_q = xq[k];
		r2_i = xi[k-1];
		r2_q = xq[k-1];

		innerTerm += sqrt(r_i*r_i + r_q*r_q) * sqrt(r2_i*r2_i + r2_q*r2_q);
	}
	correctionTerm = innerTerm;


	r[u] = correctionTerm;

	//Subtract correction term from correlation term.
	L2[u] = uSum - correctionTerm;
}

/**
 * Simplified detector. This version is meant for the ITC paper simulations,
 * and is probably not fast enough to use in the real-time PAQ project. But,
 * it can be used to confirm correct results later.
 */
__global__ void cudaNCPDI2(float* xi, float* xq, int N, int Lq, float *L) {

	//This thread will process the result at the calculated sample
	int u = (blockIdx.x * blockDim.x) + threadIdx.x;

	//Don't process off the end of the array;
	if(u >= N) return;

	float iSum = 0, qSum = 0;
	float uSum = 0;
	int offset;

	for(int k = 0; k <= 7; k++) {
		offset = u + k*Lq;

		iSum     = xi[offset + 0] + xi[offset + 8] + xi[offset + 16] + xi[offset + 24]
		                                                                  - xi[offset + 4] - xi[offset + 20]
		                                                                                        + xq[offset + 2] + xq[offset + 10] + xq[offset + 14] + xq[offset + 22]
		                                                                                                                                                  - xq[offset + 6] - xq[offset + 18] - xq[offset + 26] - xq[offset + 30]

		                                                                                                                                                                                                            + .7071*(xi[offset + 1] + xi[offset + 7] + xi[offset + 9] + xi[offset + 15] + xi[offset + 17] + xi[offset + 23] + xi[offset + 25] + xi[offset + 31]
		                                                                                                                                                                                                                                                                                                                                                   - xi[offset + 3] - xi[offset + 5] - xi[offset + 11] - xi[offset + 12] - xi[offset + 13] - xi[offset + 19] - xi[offset + 21] - xi[offset + 27] - xi[offset + 28] - xi[offset + 29]
		                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        + xq[offset + 1] + xq[offset + 3] + xq[offset + 9] + xq[offset + 11] + xq[offset + 12] + xq[offset + 13] + xq[offset + 15] + xq[offset + 21] + xq[offset + 23]
		                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          - xq[offset + 5] - xq[offset + 7] - xq[offset + 17] - xq[offset + 19] - xq[offset + 25] - xq[offset + 27] - xq[offset + 28] - xq[offset + 29] - xq[offset + 31]);

		qSum     = xq[offset + 0] + xq[offset + 8] + xq[offset + 16] + xq[offset + 24]
		                                                                  - xq[offset + 4] - xq[offset + 20]
		                                                                                        - xi[offset + 2] - xi[offset + 10] - xi[offset + 14] - xi[offset + 22]
		                                                                                                                                                  + xi[offset + 6] + xi[offset + 18] + xi[offset + 26] + xi[offset + 30]

		                                                                                                                                                                                                            + .7071*(xq[offset + 1] + xq[offset + 7] + xq[offset + 9] + xq[offset + 15] + xq[offset + 17] + xq[offset + 23] + xq[offset + 25] + xq[offset + 31]
		                                                                                                                                                                                                                                                                                                                                                   - xq[offset + 3] - xq[offset + 5] - xq[offset + 11] - xq[offset + 12] - xq[offset + 13] - xq[offset + 19] - xq[offset + 21] - xq[offset + 27] - xq[offset + 28] - xq[offset + 29]
		                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        - xi[offset + 1] - xi[offset + 3] - xi[offset + 9] - xi[offset + 11] - xi[offset + 12] - xi[offset + 13] - xi[offset + 15] - xi[offset + 21] - xi[offset + 23]
		                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          + xi[offset + 5] + xi[offset + 7] + xi[offset + 17] + xi[offset + 19] + xi[offset + 25] + xi[offset + 27] + xi[offset + 28] + xi[offset + 29] + xi[offset + 31]);

		uSum += (iSum * iSum) + (qSum * qSum);
	}

	//float uSum = (iSum * iSum) + (qSum * qSum);

	L[u] = uSum;
}


__global__ void calculateInnerSumBlocksNew(float* i, float* q, float* innerSums, int uLength, int innerSumsLength) {
	int u = blockDim.x * blockIdx.x + threadIdx.x;
	if(u >= innerSumsLength) return; //Don't process off the end of the array.

	// Scan the full i and q
	int idx = u;// + INNER_SUM_OFFSET;

	float iSum = 0;
	float qSum = 0;

	//Calculate inner sum parts
	iSum     = i[idx    ] + i[idx +  8] + i[idx + 16] + i[idx + 24]
	                                                      - i[idx + 4] - i[idx + 20]
	                                                                       + q[idx + 2] + q[idx + 10] + q[idx + 14] + q[idx + 22]
	                                                                                                                    - q[idx + 6] - q[idx + 18] - q[idx + 26] - q[idx + 30]

	                                                                                                                                                                 + .7071*(i[idx + 1] + i[idx + 7] + i[idx +  9] + i[idx + 15] + i[idx + 17] + i[idx + 23] + i[idx + 25] + i[idx + 31]
	                                                                                                                                                                                                                                                                            - i[idx + 3] - i[idx + 5] - i[idx + 11] - i[idx + 12] - i[idx + 13] - i[idx + 19] - i[idx + 21] - i[idx + 27] - i[idx + 28] - i[idx + 29]
	                                                                                                                                                                                                                                                                                                                                                                                                            + q[idx + 1] + q[idx + 3] + q[idx +  9] + q[idx + 11] + q[idx + 12] + q[idx + 13] + q[idx + 15] + q[idx + 21] + q[idx + 23]
	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              - q[idx + 5] - q[idx + 7] - q[idx + 17] - q[idx + 19] - q[idx + 25] - q[idx + 27] - q[idx + 28] - q[idx + 29] - q[idx + 31]);

	qSum     = q[idx    ] + q[idx +  8] + q[idx + 16] + q[idx + 24]
	                                                      - q[idx + 4] - q[idx + 20]
	                                                                       - i[idx + 2] - i[idx + 10] - i[idx + 14] - i[idx + 22]
	                                                                                                                    + i[idx + 6] + i[idx + 18] + i[idx + 26] + i[idx + 30]

	                                                                                                                                                                 + .7071*(q[idx + 1] + q[idx + 7] + q[idx +  9] + q[idx + 15] + q[idx + 17] + q[idx + 23] + q[idx + 25] + q[idx + 31]
	                                                                                                                                                                                                                                                                            - q[idx + 3] - q[idx + 5] - q[idx + 11] - q[idx + 12] - q[idx + 13] - q[idx + 19] - q[idx + 21] - q[idx + 27] - q[idx + 28] - q[idx + 29]
	                                                                                                                                                                                                                                                                                                                                                                                                            - i[idx + 1] - i[idx + 3] - i[idx +  9] - i[idx + 11] - i[idx + 12] - i[idx + 13] - i[idx + 15] - i[idx + 21] - i[idx + 23]
	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              + i[idx + 5] + i[idx + 7] + i[idx + 17] + i[idx + 19] + i[idx + 25] + i[idx + 27] + i[idx + 28] + i[idx + 29] + i[idx + 31]);

	//Save in array
	int saveAddress = 2*u;
	innerSums[saveAddress] = iSum;
	innerSums[saveAddress + 1] = qSum;
}

/**
 * Calculates the outer sums and enters the values into L
 */
__global__ void calculateOuterSumsNew(float* innerSums, float* L, int uLength) {

	int u = blockDim.x*blockIdx.x + threadIdx.x;
	if(u >= uLength) return; //Don't run off the end of the array.

	float real, imag, u_sum;
	int realIdx = 2*u;
	int imagIdx = realIdx+1;

	//We're going to skip block 0, because the modulation is a modulation with memory --
	//the waveform generated for the first few bits of the preamble are changed depending
	//on the last few data bits in the packet before it. The correlation seems sensitive to
	//the changes in the waveform (from expected) that this causes, and makes us lock in the
	//wrong spot.

	//	Block 0
	real = innerSums[realIdx];
	imag = innerSums[imagIdx];
	u_sum = (real*real) + (imag*imag);
	//	u_sum = 0;

	//Block 1
	realIdx += 64;
	imagIdx += 64;
	real = innerSums[realIdx];
	imag = innerSums[imagIdx];
	u_sum += (real*real) + (imag*imag);

	//Block 2
	realIdx += 64;
	imagIdx += 64;
	real = innerSums[realIdx];
	imag = innerSums[imagIdx];
	u_sum += (real*real) + (imag*imag);

	//Block 3
	realIdx += 64;
	imagIdx += 64;
	real = innerSums[realIdx];
	imag = innerSums[imagIdx];
	u_sum += (real*real) + (imag*imag);

	//Block 4
	realIdx += 64;
	imagIdx += 64;
	real = innerSums[realIdx];
	imag = innerSums[imagIdx];
	u_sum += (real*real) + (imag*imag);

	//Block 5
	realIdx += 64;
	imagIdx += 64;
	real = innerSums[realIdx];
	imag = innerSums[imagIdx];
	u_sum += (real*real) + (imag*imag);

	//Block 6
	realIdx += 64;
	imagIdx += 64;
	real = innerSums[realIdx];
	imag = innerSums[imagIdx];
	u_sum += (real*real) + (imag*imag);

	//Block 7
	realIdx += 64;
	imagIdx += 64;
	real = innerSums[realIdx];
	imag = innerSums[imagIdx];
	u_sum += (real*real) + (imag*imag);

	//Save result
	L[u] = u_sum;
}



/**
 * Finds the maximum over a packet's worth of data (12,672 samples
 * if the timing loop is active on the Quasonix reciever).
 */
__global__ void findPreambleMaximums(float* L, int* max_locations, int* max_locations_save, uint samplesPerPacket, int* firstMaxMod, int max_sample_idx, int numSections) {
	int threadNum = blockDim.x*blockIdx.x + threadIdx.x;

	if(threadNum >= numSections)
		return;

	max_locations[threadNum] = -1;
	/*
	//Let search relative to the first max
	// Search firstMaxMod-12672/2 to firstMaxMod+12672/2

	int frontOfWindow = firstMaxMod[0]-samplesPerPacket/2 + threadNum*samplesPerPacket;
	int  backOfWindow = firstMaxMod[0]+samplesPerPacket/2 + threadNum*samplesPerPacket;

	if(frontOfWindow<0)
		frontOfWindow = 0;

	if(backOfWindow>max_sample_idx)
		backOfWindow = max_sample_idx;

	float maxVal = 0;
	int maxIdx = 0;
	for(int u=frontOfWindow; u < backOfWindow; u++) {
		if(L[u] > maxVal) {
			maxVal = L[u];
			maxIdx = u;
		}
	}
	max_locations[threadNum] = maxIdx;
	max_locations_save[threadNum] = maxIdx;
	 */

	int endIndex = threadNum*samplesPerPacket + firstMaxMod[0]+12672/2;
	if(endIndex >= max_sample_idx)
		endIndex = max_sample_idx;

	int startIndex = endIndex - samplesPerPacket;

	//Find the peak over a packet's worth of data
	float maxVal = 0;
	int maxIdx = 0;
	if(startIndex<0){
		startIndex = 0;
	}

	for(int u=startIndex; u < endIndex; u++) {
		if(L[u] > maxVal) {
			maxVal = L[u];
			maxIdx = u;
		}
	}
	max_locations[threadNum] = maxIdx;
	max_locations_save[threadNum] = maxIdx;

	//max_locations[threadNum] = findMaxSI(L, startIndex, endIndex);
	//max_locations_save[threadNum] = findMaxSI(L, startIndex*((int)(threadNum!=0)), endIndex);

}

__global__ void findPreambleMaximums(cuComplex* L, int* max_locations, int* max_locations_save, uint samplesPerPacket, int* firstMaxMod, int max_sample_idx, int numSections) {
	int threadNum = blockDim.x*blockIdx.x + threadIdx.x;

	if(threadNum >= numSections)
		return;

	max_locations[threadNum] = -1;
	/*
	//Let search relative to the first max
	// Search firstMaxMod-12672/2 to firstMaxMod+12672/2

	int frontOfWindow = firstMaxMod[0]-samplesPerPacket/2 + threadNum*samplesPerPacket;
	int  backOfWindow = firstMaxMod[0]+samplesPerPacket/2 + threadNum*samplesPerPacket;

	if(frontOfWindow<0)
		frontOfWindow = 0;

	if(backOfWindow>max_sample_idx)
		backOfWindow = max_sample_idx;

	float maxVal = 0;
	int maxIdx = 0;
	for(int u=frontOfWindow; u < backOfWindow; u++) {
		if(L[u] > maxVal) {
			maxVal = L[u];
			maxIdx = u;
		}
	}
	max_locations[threadNum] = maxIdx;
	max_locations_save[threadNum] = maxIdx;
	 */

	int endIndex = threadNum*samplesPerPacket + firstMaxMod[0]+12672/2;
	if(endIndex >= max_sample_idx)
		endIndex = max_sample_idx;

	int startIndex = endIndex - samplesPerPacket;

	//Find the peak over a packet's worth of data
	float maxVal = 0;
	int maxIdx = 0;
	if(startIndex<0){
		startIndex = 0;
	}

	for(int u=startIndex; u < endIndex; u++) {
		if(L[u].x > maxVal) {
			maxVal = L[u].x;
			maxIdx = u;
		}
	}
	max_locations[threadNum] = maxIdx;
	max_locations_save[threadNum] = maxIdx;

	//max_locations[threadNum] = findMaxSI(L, startIndex, endIndex);
	//max_locations_save[threadNum] = findMaxSI(L, startIndex*((int)(threadNum!=0)), endIndex);

}

__device__ int findMaxSI(float*L, int startIndex, int endIndex) {
	float maxVal = 0;
	int maxIdx = 0;

	for(int u=startIndex; u < endIndex; u++) {
		if(L[u] > maxVal) {
			maxVal = L[u];
			maxIdx = u;
		}
	}

	return maxIdx;
}

/**
 *	Uses the majority rules method to find where the correct peak detections are. Then, it sets all peaks to be
 *	in line with these peaks. Calculates what the startOffset should be for the next findPreambleMax function,
 *	so that the peaks should always be in the center of the packet.
 *
 *	Note that this kernel should only be used if the timing loop is running on the Quasonix receiver. If not,
 *	this function will give incorrect decisions for the preambles, and the rest of the system will output garbage.
 */
__global__ void fixMaximumsLazy(int* max_locations, int* num_good_maximums, int* startOffset, int numInputSamples) {
	int maxLength = 0;
	int length = 0;
	int maxIdx = 0;
	int idx = 0;

	//Find longest chain of correct values
	bool inChain = false;
	for(int i=0; i < 3103; i++) {
		if(abs((max_locations[i+1] - max_locations[i]) - 12672) <= 3) {
			if(inChain) {
				length++;
			} else {
				length = 1;
				idx = i;
				inChain = true;
			}
		} else {
			if(inChain) {
				inChain = false;
				if(length > maxLength) {
					maxLength = length;
					maxIdx = idx;
				}

			}
		}
	}

	//From this chain, find what max_locations[0] SHOULD be
	int correctFirstPosition = max_locations[maxIdx] % 12672;

	//Adjust all maximums to be in line
	*num_good_maximums = 0;
	int nextMax;
	for(int i=0; i < 3104; i++) {
		nextMax = correctFirstPosition + 12672*i;
		if(nextMax < numInputSamples) {
			max_locations[i] = nextMax;
			(*num_good_maximums)++;
		} else {
			max_locations[i] = -1;
		}
	}
	int lastMaxDistanceFromEnd = numInputSamples - max_locations[*num_good_maximums-1];
	int frontLengthofEndPacket = 12672 - lastMaxDistanceFromEnd;


	*startOffset =  frontLengthofEndPacket + 6336;
}



__global__ void asmCorr(float* A, float* i, float* q, float* asmI, float* asmQ, int* max_locations, int numCorrs) {
	int u = blockDim.x*blockIdx.x + threadIdx.x;
	if (u >= numCorrs) return;

	float corrReal = 0;
	float corrImag = 0;

	//For each peak that we're testing, we check peak+193 to peak+320.
	//This gives us 128 points around the start of the ASM
	//We need to know which block we're working in, and which
	//index we are in this block.
	int block = blockIdx.x;
	int offset = threadIdx.x;
	int sampIdx = max_locations[block] + 193 + offset;

	//a = asmI
	//b = asmQ
	//c = i
	//d = q

	for(int j=0; j < 128; j++) {
		corrReal += (asmI[j] * i[sampIdx]) + (asmQ[j] * q[sampIdx]);
		corrImag += (asmI[j] * q[sampIdx]) - (asmQ[j] * i[sampIdx]);
		sampIdx++;
	}

	A[u] = sqrt((corrReal*corrReal)+(corrImag*corrImag));
}

/**
 * L-CL-3
 */
__global__ void cudaCL3(float* xi, float* xq, float* sr, float* si, int N, float* L) {
	//This thread will process the result at the calculated sample
	int u = (blockIdx.x * blockDim.x) + threadIdx.x;

	//Don't process off the end of the array;
	if(u >= N) return;

	float uSum = 0;
	float r_i, r_q, rconj_i, rconj_q;
	float s_i, s_q, sconj_i, sconj_q;

	float rsum_i, rsum_q, ssum_i, ssum_q;
	float ksum_i, ksum_q;


	ksum_i = 0;
	ksum_q = 0;
	for (int k=1; k <= SAMPLES_IN_ASM-1; k++) {

		r_i 	= xi[u+k-1];
		r_q 	= xq[u+k-1];
		rconj_i = xi[u+k];
		rconj_q = xq[u+k] * (-1);

		s_i 	= sr[k];
		s_q 	= si[k];
		sconj_i = sr[k-1];
		sconj_q = si[k-1] *(-1);

		rsum_i = (r_i * rconj_i) - (r_q * rconj_q);
		rsum_q = (r_i * rconj_q) + (r_q * rconj_i);

		ssum_i = (s_i * sconj_i) - (s_q * sconj_q);
		ssum_q = (s_i * sconj_q) + (s_q * sconj_i);

		ksum_i += (rsum_i * ssum_i) - (rsum_q * ssum_q);
		ksum_q += (rsum_i * ssum_q) + (rsum_q * ssum_i);
	}
	//Add the absolute value of the inner sum
	uSum = sqrt((ksum_i * ksum_i) + (ksum_q * ksum_q));
	//}

	L[u] = uSum;
}


__global__ void cl1a1b(float* xi, float* xq, float* sr, float* si, int N, float* L, float* L2, float* r) {
	//This thread will process the result at the calculated sample
	int u = (blockIdx.x * blockDim.x) + threadIdx.x;

	//Don't process off the end of the array;
	if(u >= N) return;

	float uSum = 0;
	float r_i, r_q, rconj_i, rconj_q;
	float s_i, s_q, sconj_i, sconj_q;

	float rsum_i, rsum_q, ssum_i, ssum_q;
	float ksum_i, ksum_q;

	for(int i=1; i < SAMPLES_IN_ASM; i++) {
		//printf("On i: %d\n", i);
		ksum_i = 0;
		ksum_q = 0;
		for (int k=i; k < SAMPLES_IN_ASM; k++) {
			//			r_i 	= xi[u+k+i];
			//			r_q 	= xq[u+k+i];
			//			rconj_i = xi[u+k];
			//			rconj_q = xq[u+k] * (-1);
			//
			//			s_i 	= sr[k];
			//			s_q 	= si[k];
			//			sconj_i = sr[k+i];
			//			sconj_q = si[k+i] *(-1);

			r_i 	= xi[u+k-i];
			r_q 	= xq[u+k-i];
			rconj_i = xi[u+k];
			rconj_q = xq[u+k] * (-1);

			s_i 	= sr[k];
			s_q 	= si[k];
			sconj_i = sr[k-i];
			sconj_q = si[k-i] *(-1);

			rsum_i = (r_i * rconj_i) - (r_q * rconj_q);
			rsum_q = (r_i * rconj_q) + (r_q * rconj_i);

			ssum_i = (s_i * sconj_i) - (s_q * sconj_q);
			ssum_q = (s_i * sconj_q) + (s_q * sconj_i);

			ksum_i += (rsum_i * ssum_i) - (rsum_q * ssum_q);
			ksum_q += (rsum_i * ssum_q) + (rsum_q * ssum_i);
		}
		//Add the absolute value of the inner sum
		uSum += sqrt((ksum_i * ksum_i) + (ksum_q * ksum_q));
	}

	L[u] = uSum;


	//Add correction term
	float correctionTerm = 0;
	float innerTerm;
	//float r_i, r_q,
	float r2_i, r2_q;
	//float real, imag;

	for(int i=1; i < SAMPLES_IN_ASM; i++) {
		innerTerm = 0;
		for(int k=u+i; k <= u+SAMPLES_IN_ASM-1; k++) {
			r_i = xi[k];
			r_q = xq[k];
			r2_i = xi[k-i];
			r2_q = xq[k-i];

			innerTerm += sqrt(r_i*r_i + r_q*r_q) * sqrt(r2_i*r2_i + r2_q*r2_q);
		}
		correctionTerm += innerTerm;
	}

	r[u] = correctionTerm;

	//Subtract correction term from correlation term.
	//L[u] -= r[u];
	L2[u] = uSum - correctionTerm;
}

__global__ void cl1a1bFull(float* xi, float* xq, float* sr, float* si, int N, float* L, float* L2, float* r, int Lp) {
	//This thread will process the result at the calculated sample
	int u = (blockIdx.x * blockDim.x) + threadIdx.x;

	//Don't process off the end of the array;
	if(u >= N) return;

	float uSum = 0;
	float r_i, r_q, rconj_i, rconj_q;
	float s_i, s_q, sconj_i, sconj_q;

	float rsum_i, rsum_q, ssum_i, ssum_q;
	float ksum_i, ksum_q;

	for(int i=1; i < Lp; i++) {
		//printf("On i: %d\n", i);
		ksum_i = 0;
		ksum_q = 0;
		for (int k=i; k < Lp; k++) {
			//			r_i 	= xi[u+k+i];
			//			r_q 	= xq[u+k+i];
			//			rconj_i = xi[u+k];
			//			rconj_q = xq[u+k] * (-1);
			//
			//			s_i 	= sr[k];
			//			s_q 	= si[k];
			//			sconj_i = sr[k+i];
			//			sconj_q = si[k+i] *(-1);

			r_i 	= xi[u+k-i];
			r_q 	= xq[u+k-i];
			rconj_i = xi[u+k];
			rconj_q = xq[u+k] * (-1);

			s_i 	= sr[k];
			s_q 	= si[k];
			sconj_i = sr[k-i];
			sconj_q = si[k-i] *(-1);

			rsum_i = (r_i * rconj_i) - (r_q * rconj_q);
			rsum_q = (r_i * rconj_q) + (r_q * rconj_i);

			ssum_i = (s_i * sconj_i) - (s_q * sconj_q);
			ssum_q = (s_i * sconj_q) + (s_q * sconj_i);

			ksum_i += (rsum_i * ssum_i) - (rsum_q * ssum_q);
			ksum_q += (rsum_i * ssum_q) + (rsum_q * ssum_i);
		}
		//Add the absolute value of the inner sum
		uSum += sqrt((ksum_i * ksum_i) + (ksum_q * ksum_q));
	}

	L[u] = uSum;


	//Add correction term
	float correctionTerm = 0;
	float innerTerm;
	//float r_i, r_q,
	float r2_i, r2_q;
	//float real, imag;

	for(int i=1; i < Lp; i++) {
		innerTerm = 0;
		for(int k=u+i; k <= u+Lp-1; k++) {
			r_i = xi[k];
			r_q = xq[k];
			r2_i = xi[k-i];
			r2_q = xq[k-i];

			innerTerm += sqrt(r_i*r_i + r_q*r_q) * sqrt(r2_i*r2_i + r2_q*r2_q);
		}
		correctionTerm += innerTerm;
	}

	r[u] = correctionTerm;

	//Subtract correction term from correlation term.
	//L[u] -= r[u];
	L2[u] = uSum - correctionTerm;
}

__global__ void cudaCL3Full(float* xi, float* xq, float* sr, float* si, int N, float* L, int Lp) {
	//This thread will process the result at the calculated sample
	int u = (blockIdx.x * blockDim.x) + threadIdx.x;

	//Don't process off the end of the array;
	if(u >= N) return;

	float uSum = 0;
	float r_i, r_q, rconj_i, rconj_q;
	float s_i, s_q, sconj_i, sconj_q;

	float rsum_i, rsum_q, ssum_i, ssum_q;
	float ksum_i, ksum_q;


	ksum_i = 0;
	ksum_q = 0;
	for (int k=1; k <= Lp-1; k++) {

		r_i 	= xi[u+k-1];
		r_q 	= xq[u+k-1];
		rconj_i = xi[u+k];
		rconj_q = xq[u+k] * (-1);

		s_i 	= sr[k];
		s_q 	= si[k];
		sconj_i = sr[k-1];
		sconj_q = si[k-1] *(-1);

		rsum_i = (r_i * rconj_i) - (r_q * rconj_q);
		rsum_q = (r_i * rconj_q) + (r_q * rconj_i);

		ssum_i = (s_i * sconj_i) - (s_q * sconj_q);
		ssum_q = (s_i * sconj_q) + (s_q * sconj_i);

		ksum_i += (rsum_i * ssum_i) - (rsum_q * ssum_q);
		ksum_q += (rsum_i * ssum_q) + (rsum_q * ssum_i);
	}
	//Add the absolute value of the inner sum
	uSum = sqrt((ksum_i * ksum_i) + (ksum_q * ksum_q));
	//}

	L[u] = uSum;
}

/**
 * The NCPDI-2 algorithm uses 32-sample coherent correlation chunks. We're going to
 * treat those as if they were a simple correlation in and of themselves.
 */
__global__ void smallCorrelation(float* L, float* innerSums, int innerSumsLength) {
	//This thread will process the result at the calculated sample
	int u = (blockIdx.x * blockDim.x) + threadIdx.x;

	//Don't process off the end of the array;
	if(u >= innerSumsLength) return;

	int realIdx = 2*u;
	int imagIdx = realIdx + 1;

	L[u] = (innerSums[realIdx] * innerSums[realIdx]) + (innerSums[imagIdx] * innerSums[imagIdx]);
}

/**
 *
 */
__global__ void qCorrelator(float* i, float* q, float* L, int uLength) {
	int u = blockDim.x * blockIdx.x + threadIdx.x;
	if(u >= uLength) return; //Don't process off the end of the array.

	float iSum, qSum;

	//Calculate inner sum parts
	iSum     = i[u    ] + i[u + 8] + i[u + 16] + i[u + 24]
	                                               - i[u + 4] - i[u + 20]
	                                                              + q[u + 2] + q[u + 10] + q[u + 14] + q[u + 22]
	                                                                                                     - q[u + 6] - q[u + 18] - q[u + 26] - q[u + 30]

	                                                                                                                                            + .7071*(i[u + 1] + i[u + 7] + i[u + 9] + i[u + 15] + i[u + 17] + i[u + 23] + i[u + 25] + i[u + 31]
	                                                                                                                                                                                                                                        - i[u + 3] - i[u + 5] - i[u + 11] - i[u + 12] - i[u + 13] - i[u + 19] - i[u + 21] - i[u + 27] - i[u + 28] - i[u + 29]
	                                                                                                                                                                                                                                                                                                                                                      + q[u + 1] + q[u + 3] + q[u + 9] + q[u + 11] + q[u + 12] + q[u + 13] + q[u + 15] + q[u + 21] + q[u + 23]
	                                                                                                                                                                                                                                                                                                                                                                                                                                                       - q[u + 5] - q[u + 7] - q[u + 17] - q[u + 19] - q[u + 25] - q[u + 27] - q[u + 28] - q[u + 29] - q[u + 31]);

	qSum     = q[u    ] + q[u + 8] + q[u + 16] + q[u + 24]
	                                               - q[u + 4] - q[u + 20]
	                                                              - i[u + 2] - i[u + 10] - i[u + 14] - i[u + 22]
	                                                                                                     + i[u + 6] + i[u + 18] + i[u + 26] + i[u + 30]

	                                                                                                                                            + .7071*(q[u + 1] + q[u + 7] + q[u + 9] + q[u + 15] + q[u + 17] + q[u + 23] + q[u + 25] + q[u + 31]
	                                                                                                                                                                                                                                        - q[u + 3] - q[u + 5] - q[u + 11] - q[u + 12] - q[u + 13] - q[u + 19] - q[u + 21] - q[u + 27] - q[u + 28] - q[u + 29]
	                                                                                                                                                                                                                                                                                                                                                      - i[u + 1] - i[u + 3] - i[u + 9] - i[u + 11] - i[u + 12] - i[u + 13] - i[u + 15] - i[u + 21] - i[u + 23]
	                                                                                                                                                                                                                                                                                                                                                                                                                                                       + i[u + 5] + i[u + 7] + i[u + 17] + i[u + 19] + i[u + 25] + i[u + 27] + i[u + 28] + i[u + 29] + i[u + 31]);

	//Save in array
	L[u] = (iSum*iSum) + (qSum*qSum);

}

/**
 * Estimates the frequency offset on a per-packet basis, then rotates by -w0
 */
__global__ void gpu_estimateFreqOffsetAndRotate(float* xi, float* xq, int *max_locations, int numPreamblePeaks, float* w0) {
	int packetNum = blockDim.x * blockIdx.x + threadIdx.x;
	if(packetNum >= numPreamblePeaks) return;

	//int i = max_locations[packetNum] + INNER_SUM_OFFSET;
	int i = max_locations[packetNum];
	int upperBound = i+223;

	//Sum from n=i+2*L_q  to n=i+7*L_q-1. With L_q = 32, this becomes n=i+64 to n=i+223
	float rReal = 0, rImag = 0;
	float a, b, c, d;
	int rConjIdx;
	for(int n = i+64; n <= upperBound; n++) {
		//(a+jb)(c+jd)* = (a+jb)(c-jd) = (ac + bd) + j(bc - ad)
		rConjIdx = n-32;
		a = xi[n];
		b = xq[n];
		c = xi[rConjIdx];
		d = xq[rConjIdx];

		rReal += (a*c)+(b*d);
		rImag += (b*c)-(a*d);
	}

	//Scale by 1/(5*L_q) = .0063
	//rReal *= .0063;
	//rImag *= .0063;

	//Get angle
	float angle = atan2f(rImag, rReal);

	//Scale by (1/L_q) = .0312
	w0[packetNum] = .0312 * angle;

	float rotReal = cosf(-1*w0[packetNum]);
	float rotImag = sinf(-1*w0[packetNum]);

	for(int j=i; j < i + 12672; j++) {
		xi[i] = (xi[i]*rotReal) - (xq[i]*rotImag);
		xq[i] = (xi[i]*rotImag) + (xq[i]*rotReal);
	}
}

/**
 * Rotates a packet's worth of samples samples according to the frequency offset given.
 */
__global__ void rotateSamplesPacket(float* xi, float* xq, float* w0, int *max_locations, int numPacketsToProcess) {
	int packetNum = blockDim.x * blockIdx.x + threadIdx.x;
	if(packetNum >= numPacketsToProcess) return;

	int packetStart = max_locations[packetNum];
	int packetEnd = max_locations[packetNum+1];

	float rotReal = cosf(w0[packetNum]);
	float rotImag = sinf(w0[packetNum]);

	//(a+jb)(c+jd) = (ac-bd) + j(ad + bc)
	//a = xi[i]
	//b = xq[i]
	//c = rotReal
	//d = rotImag
	for(int i=packetStart; i < packetEnd; i++) {
		xi[i] = (xi[i]*rotReal) - (xq[i]*rotImag);
		xq[i] = (xi[i]*rotImag) + (xq[i]*rotReal);
	}
}

/**
 * Performs the matrix-vector multiply that estimates the channel on a per-packet basis.
 */
__global__ void estimateChannel(int *max_locations, int numPacketsToProcess, float* xi, float* xq, cuComplex* alpha, cuComplex* beta, cuComplex *chanEstMatrix, cuComplex *r_d, cuComplex *h_hat) {
	int packetNum = blockDim.x * blockIdx.x + threadIdx.x;
	if(packetNum >= numPacketsToProcess) return;

	//Copy received sample I and Q data into the cuComplex r_d vector
	int receivedIdx = max_locations[packetNum] + N2_CAUSAL_SAMPLES;
	int r_d_idx = packetNum * R_D_LENGTH;
	int r_d_start_idx = r_d_idx;
	int h_hat_start_idx = packetNum * CHAN_EST_LENGTH;

	for(int i=0; i < CHAN_EST_MATRIX_NUM_COLS; i++) {
		r_d[r_d_idx].x = xi[receivedIdx];
		r_d[r_d_idx].y = xq[receivedIdx];
		r_d_idx++;
		receivedIdx++;
	}

	cublasHandle_t localHandle;
	cublasCreate(&localHandle);
	//Now multiply chanEstMatrix and r_d. Answer is stored in h_hat vector
	cublasCgemm(
			localHandle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			CHAN_EST_MATRIX_NUM_ROWS,
			1,
			CHAN_EST_MATRIX_NUM_COLS,
			alpha,
			chanEstMatrix,
			CHAN_EST_MATRIX_NUM_ROWS,
			&r_d[r_d_start_idx],
			CHAN_EST_MATRIX_NUM_COLS,
			beta,
			&h_hat[h_hat_start_idx],
			CHAN_EST_LENGTH
	);
	cublasDestroy(localHandle);

}

/**
 * Estimates the noise variance
 */
__global__ void estimateNoiseVariance(cuComplex* r_d,
		cuComplex* x_mat,
		cuComplex* Xh_prod,
		cuComplex* h_hat,
		cuComplex* sigmaSquared,
		cuComplex* alpha,
		cuComplex* beta,
		int numPacketsToProcess) {
	int packetNum = blockDim.x * blockIdx.x + threadIdx.x;
	if(packetNum >= numPacketsToProcess) return;

	int r_d_start_idx = packetNum * R_D_LENGTH;
	int h_hat_start_idx = packetNum * CHAN_EST_LENGTH;

	cublasHandle_t localHandle;
	cublasCreate(&localHandle);
	//Multiply the X vector and h_hat
	cublasCgemm(localHandle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			X_MAT_NUM_ROWS,
			1,
			X_MAT_NUM_COLS,
			alpha,
			x_mat,
			X_MAT_NUM_ROWS,
			&h_hat[h_hat_start_idx],
			CHAN_EST_LENGTH,
			beta,
			&Xh_prod[r_d_start_idx],
			X_MAT_NUM_ROWS);
	cublasDestroy(localHandle);

	//Solve r_d - Xh_prod, then solve the square
	cuComplex squareVal;
	int tempIdx;
	squareVal.x = (float)0;
	squareVal.y = (float)0;
	for(int i=0; i < R_D_LENGTH; i++) {
		tempIdx = r_d_start_idx + i;
		r_d[tempIdx].x -= Xh_prod[tempIdx].x;
		r_d[tempIdx].y -= Xh_prod[tempIdx].y;

		squareVal.x += (r_d[tempIdx].x * r_d[tempIdx].x);
		squareVal.y += (r_d[tempIdx].y * r_d[tempIdx].y);
	}

	//Solve for sigmaHatSquared
	//1/(2*rho) = .0016 + j*0
	cuComplex shs;
	shs.x = .0016 * squareVal.x;
	shs.y = .0016 * squareVal.y;
	sigmaSquared[packetNum] = shs;
}

/**
 * Estimates the noise variance
 */
__global__ void resolveNoiseVariance(cublasHandle_t handle,
		cuComplex* r_d,
		cuComplex* x_mat,
		cuComplex* Xh_prod,
		cuComplex* h_hat,
		cuComplex* sigmaSquared,
		cuComplex* alpha,
		cuComplex* beta,
		int numPacketsToProcess) {
	int packetNum = blockDim.x * blockIdx.x + threadIdx.x;
	if(packetNum >= numPacketsToProcess) return;

	int r_d_start_idx = packetNum * R_D_LENGTH;

	//Solve r_d - Xh_prod, then solve the square
	cuComplex squareVal;
	int tempIdx;
	squareVal.x = 0;
	squareVal.y = 0;
	for(int i=0; i < R_D_LENGTH; i++) {
		tempIdx = r_d_start_idx + i;
		r_d[tempIdx].x -= Xh_prod[tempIdx].x;
		r_d[tempIdx].y -= Xh_prod[tempIdx].y;

		squareVal.x += (r_d[tempIdx].x * r_d[tempIdx].x);
		squareVal.y += (r_d[tempIdx].y * r_d[tempIdx].y);
	}

	//Solve for sigmaHatSquared
	//1/(2*rho) = .0016 + j*0
	cuComplex shs;
	shs.x = .0016 * squareVal.x;
	shs.y = .0016 * squareVal.y;
	sigmaSquared[packetNum] = shs;

}

//Calculates the autocorrelation sequence
__global__ void autocorr(cuComplex* h, int hLength, cuComplex* result, int corrLength, int saveOffset, float* shs, int mmseFlag) {
	//int u = (blockDim.x * blockIdx.x) + threadIdx.x;
	//if(u >= maxIdx) return;

	int lag = threadIdx.x;
	int lastIdx = hLength - lag;
	int hIdx = hLength*blockIdx.x;

	cuComplex corrSum;
	corrSum.x = 0;
	corrSum.y = 0;

	//f*[m]f[m+n]
	//(a+jb)*(c+jd) = (a-jb)(c+jd) = ac+bd + j(ad-bc)
	//a = h[hIdx+i].x
	//b = h[hIdx+i].y
	//c = h[hIdx+i+lag].x
	//d = h[hIdx+i+lag].y
	int idx1, idx2;
	for(int i=0; i < lastIdx; i++) {
		idx1 = hIdx+i;
		idx2 = idx1+lag;
		//corrSum.x += (h[hIdx+i].x * h[hIdx+i+lag].x) + (h[hIdx+i].y * h[hIdx+i+lag].y);
		//corrSum.y += (h[hIdx+i].y * h[hIdx+i+lag].x) - (h[hIdx+i].x * h[hIdx+i+lag].y);
		corrSum.x += (h[idx1].x * h[idx2].x) + (h[idx1].y * h[idx2].y);
		corrSum.y += (h[idx1].x * h[idx2].y) - (h[idx1].y * h[idx2].x);
	}

	//int corrLength = hLength*2-1;
	//int saveOffset = corrLength/2;
	int saveIdx = blockIdx.x * corrLength + saveOffset;

	result[saveIdx+lag] = corrSum;

	if(lag == 0 && mmseFlag == 1){
		cuComplex tempVal;
		tempVal.x = corrSum.x + shs[blockIdx.x];
		//		tempVal.x = corrSum.x + .0001;
		tempVal.y = corrSum.y + 0;
		result[saveIdx] = tempVal;
	}

	if(lag == 0 && mmseFlag == 0){
		cuComplex tempVal;
		tempVal.x = corrSum.x;
		tempVal.y = corrSum.y;
		result[saveIdx] = tempVal;
	}


	if(lag != 0) {
		cuComplex tempVal;
		tempVal.x = corrSum.x;
		tempVal.y = -corrSum.y;
		result[saveIdx-lag] = tempVal;


	}

}

/**
 * Fills the H*H matrices with the h_corr values
 */
__global__ void fill_hhh_csr_matrices(cuComplex* h_corr, cuComplex* hhh_csr, int N1, int N2, int nnzA) { // N=Equ Length or L1+L2+1
	int matNum = blockIdx.x;																  // corrLength =
	int y = threadIdx.x;																  // hhh is NxN

	int matrixIndex = matNum*nnzA;

	int L1 = 5*N1;
	int L2 = 5*N2;
	int EQUALIZER_LENGTH = L1 + L2 + 1;

	//Load the correlation values into shared memory.
	__shared__ cuComplex shared_h_corr[LD_N];
	shared_h_corr[y] = h_corr[(2*(N1+N2)+1)*matNum+y];
	__syncthreads();

	//Load the correlation values into the H*H matrix in csr format
	int L, T, LB, TB, addr, leadingRow, h_corr_index;
	for(int x = 0; x < EQUALIZER_LENGTH; x++){

		LB = N1+N2-2;
		leadingRow = y-LB;
		L = ((leadingRow-2)*(leadingRow-1)/2) * (y > LB);

		TB = L1+L2+1-(N1+N2+1);
		T = (TB*y - y*(y-1)/2);

		h_corr_index = N1+N2-x+y;

		addr = EQUALIZER_LENGTH*y + x - L - T - (y > TB)*((y-TB)*(y-TB-1)/2);

		if(h_corr_index >= 0 && h_corr_index <= 2*(N1+N2)){
			hhh_csr[matrixIndex+addr] = shared_h_corr[h_corr_index];
		}
	}

}

/**
 * Fills the H*H matrices with the h_corr values
 */
__global__ void fill_hhh_matrices(cuComplex* h_corr, int corrLength, cuComplex* hhh, int N) { // N=Equ Length or L1+L2+1
	int matNum = blockIdx.x;																  // corrLength =
	int colNum = threadIdx.x;																  // hhh is NxN

	//Load the correlation values into shared memory. We load them in
	//backwards so that the memory writes in the section below coalesce
	//better (for much better GPU performance)
	__shared__ cuComplex matValues[CHAN_EST_AUTOCORR_LENGTH];
	matValues[corrLength-1-colNum] = h_corr[matNum*corrLength+colNum];
	__syncthreads();

	//Load the correlation values into the H*H matrix
	int matrixIndex = blockIdx.x*N*N;
	int centerTap = corrLength/2;
	int colOffset = colNum - centerTap;
	int colIdx;
	for(int i=0; i < N; i++) {
		colIdx = colOffset+i;
		if((colIdx >= 0) && (colIdx < N)) {
			hhh[matrixIndex+(i*N)+colIdx] = matValues[colNum];
		}
	}

}

///**
// * Builds the (H*)(u_n0) vector needed to solve for the
// * zero forcing equalizer coefficients
// */
//__global__ void build_hh_un0_vector(
//		cuComplex* h_hat,
//		int h_hat_length,
//		cuComplex* hh_un0,
//		int un0_length,
//		int N1,
//		int maxThreads) {
//	int u = (blockIdx.x * blockDim.x) + threadIdx.x;
//	if(u >= maxThreads) return;
//
//	int readBlockIdx = h_hat_length*u;
//	int writeBlockIdx = un0_length*u;
//	cuComplex tempValue;
//
////	int outOfBounds = 0;
//	for(int i=N1, j=0; i >= 0; i--, j++) {
//		//Read value and perform complex conjugate
//		tempValue = h_hat[readBlockIdx+j];
//		tempValue.y = -tempValue.y;
//
//		//Write value to vector
//		hh_un0[writeBlockIdx+i] = tempValue;
//	}
//}

/**
 * Builds the (H*)(u_n0) vector needed to solve for the
 * zero forcing equalizer coefficients
 * assuming hh_un0 zero initialized
 */
__global__ void build_hh_un0_vector_reworked(
		cuComplex* h_hat,
		cuComplex* h_un0,
		int N1,
		int N2,
		int L1,
		int L2,
		int maxThreads) {

	int matrixNum = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(matrixNum >= maxThreads) return;

	int h_hat_length = N1 + N2 + 1; //Dont forget the +1  :)
	int h_un0_length = L1 + L2 + 1;
	int CHAN_SIZE = h_hat_length;
	int EQ_LENGTH = h_un0_length;
	int n0 = N1 + L1; // Changed from n0 = N1 + L1 +1;

	int h_un0_MatrixJumpIndex = h_un0_length*matrixNum;
	int h_hat_MatrixJumpIndex = h_hat_length*matrixNum;

	int upperBound = n0;
	int lowerBound = n0 - CHAN_SIZE;
	int loopLength = EQ_LENGTH;

	cuComplex tempValue;
	int writeIndex;
	int readIndex;
	for(int i = 0; i<loopLength; i++){

		writeIndex = h_un0_MatrixJumpIndex+i;
		readIndex = h_hat_MatrixJumpIndex + n0-i;

		if((i>lowerBound) && (i<=upperBound)){

			// Read that value and perform the complex conj
			tempValue = h_hat[readIndex];
			tempValue.y = -tempValue.y;

			// Write value to vector
			h_un0[writeIndex] = tempValue;//tempValue;
			//			h_un0[writeIndex+matrixNum] = tempValue;//tempValue;
		}
	}
}

//Solves c=a*b
__device__ cuComplex cuMult(cuComplex& a, cuComplex& b) {
	//(a+jb)(c+jd) = (ac-bd)+j(ad+bc)
	cuComplex c;
	c.x = (a.x*b.x) - (a.y*b.y);
	c.y = (a.x*b.y) + (a.y*b.x);
	return c;
}

//Solves c=a/b
__device__ cuComplex cuDiv(cuComplex& a, cuComplex& b) {
	float norm = b.x*b.x+b.y*b.y;
	cuComplex c;
	c.x = ((a.x*b.x)+(a.y*b.y))/norm;
	c.y = ((a.x*b.y)-(a.y*b.x))/norm;
	return c;
}

/**
 * Adds sigma to H*H matrices diagonal
 * 186 threads called for each matrix
 */
__global__ void add_sigma_to_hhh_csr_matrices(cuComplex* hhh_csr, int N, int numPackets, cuComplex* sigma, int N1, int N2, int nnzA) {
	int threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
	if(threadIndex >= N*numPackets)
		return;

	int matrixNum = threadIndex/N;

	int y = threadIndex%N;
	int x = y;

	int matrixIndex = matrixNum*nnzA;

	int L1 = 5*N1;
	int L2 = 5*N2;
	int EQUALIZER_LENGTH = L1 + L2 + 1;

	//Load the correlation values into the H*H matrix in csr format
	int L, T, LB, TB, addr, leadingRow;

	LB = N1+N2-2;
	leadingRow = y-LB;
	L = ((leadingRow-2)*(leadingRow-1)/2) * (y > LB);

	TB = L1+L2+1-(N1+N2+1);
	T = (TB*y - y*(y-1)/2);

	addr = EQUALIZER_LENGTH*y + x - L - T - (y > TB)*((y-TB)*(y-TB-1)/2);

	cuComplex temp;

	//	float sig = 0;
	//	temp.x = sig;
	//	temp.y = sig;
	//	temp.x = hhh_csr[matrixIndex+addr].x + temp.x;//sigma[matrixNum].x;
	//	temp.y = hhh_csr[matrixIndex+addr].y + temp.y;//sigma[matrixNum].y;

	temp.x = hhh_csr[matrixIndex+addr].x + sigma[matrixNum].x;
	temp.y = hhh_csr[matrixIndex+addr].y + sigma[matrixNum].y;

	hhh_csr[matrixIndex+addr] = temp;
}

__global__ void zeroPad(cuComplex *out, const cuComplex *in, int batches, int old_length, int new_length)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= batches*old_length)
		return;
	int batch = i/old_length;
	int new_i = i%old_length;

	out[batch*new_length+new_i] = in[i];
}

__global__ void zeroPadShiftFDE(cuComplex *out, const cuComplex *in, int batches, int old_length, int new_length, int shift)
{	int i = blockIdx.x * blockDim.x + threadIdx.x;
if(i >= batches*old_length)
	return;
int batch = i/old_length;
int new_i = i%old_length;

out[batch*new_length+new_i] = in[batch*old_length+new_i+shift];
}

__global__ void pointMultiply(cuComplex *a_in, cuComplex *b_in, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= size)
		return;
	cuComplex a = a_in[i];
	cuComplex b = b_in[i];
	cuComplex c;
	c.x = a.x * b.x - a.y * b.y;
	c.y = a.x * b.y + a.y * b.x;

	a_in[i] = c;
}

__global__ void pointMultiplyTriple(cuComplex *a_in, const cuComplex *b_in, const cuComplex *c_in, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= size)
		return;
	cuComplex a = a_in[i];
	cuComplex b = b_in[i];
	cuComplex c = c_in[i];
	cuComplex r;
	r.x = a.x*b.x*c.x - a.y*b.y*c.x - a.x*b.y*c.y - a.y*b.x*c.y;
	r.y = a.x*b.y*c.x + a.y*b.x*c.x + a.x*b.x*c.y - a.y*b.y*c.y;

	a_in[i] = r;
}

__global__ void scaleAndPruneFFT(cuComplex *out, const cuComplex *in, float scale, int out_length, int in_length, int frontJump, int maxThreads)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= maxThreads)
		return;

	int batch = i/out_length;
	int indexInBatch = i%out_length;

	cuComplex inTemp = in[frontJump + batch*in_length + indexInBatch];
	inTemp.x = inTemp.x/scale;
	inTemp.y = inTemp.y/scale;

	out[i] = inTemp;
}

__global__ void fillUnfilteredSignal(float *real, float *imag, cuComplex *complex, int startIndex, int endIndex, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= size)
		return;

	complex[i].x = real[i+startIndex];
	complex[i].y = imag[i+startIndex];
}

__global__ void fillPaddedSignal(float *in_i, float *in_q, cuComplex *out, int startIndex, int endIndex, int old_length, int new_length){ // Dont forget to set max_threads to num_packets_to_process*size
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i > endIndex-startIndex)
		return;

	// Find what batch
	int batch 		= i/old_length;

	// Find what index
	int i_in_batch 	= i%old_length;

	// Find how far to jump IN plus the front end
	int batchJump_in = batch*old_length+startIndex;

	// Find how far to jump OUT
	int batchJump_out = batch*new_length;

	// Pull real and imag parts into local
	cuComplex tempComplex;
	tempComplex.x = in_i[batchJump_in + i_in_batch];
	tempComplex.y = in_q[batchJump_in + i_in_batch];

	// Save local into new padded location
	out[batchJump_out + i_in_batch] = tempComplex;
}

__global__ void dmodZeroPad(cuComplex *out, cuComplex *in, int batchLength, int conv_length, int maxThreads)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= maxThreads)
		return;

	int batch = i / batchLength;
	int indexInBatch = i % batchLength;

	int batchJump = batch*conv_length;

	out[batchJump+indexInBatch] = in[i];

	//	out[i].x = batchJump;//in[batchJump + frontJump + indexInBatch];
	//	out[i].y = batchJump;//in[batchJump + frontJump + indexInBatch];
}

__global__ void dmodPostPruneScaledDownsample(cuComplex *in, cuComplex *out, int front, int packet_width_pre_downsample, int unPruned, int downBy, float scale, int shift, int maxThreads)
{
	int threadIndex = blockIdx.x*blockDim.x+threadIdx.x;
	if(threadIndex >= maxThreads)
		return;

	int multiIndex = threadIndex*downBy;

	int batch = multiIndex / packet_width_pre_downsample;
	int indexInBatch = multiIndex % packet_width_pre_downsample;

	int batchJump = batch*unPruned;
	int frontJump = front;

	cuComplex a;
	a = in[batchJump + frontJump + indexInBatch - shift];

	a.x = a.x*scale;
	a.y = a.y*scale;

	if(indexInBatch < shift){
		a.x = 0;
		a.y = 0;
	}


	out[threadIndex] = a;
}

__global__ void cudaDemodulator(const cuComplex *dfout1, float *ahat, int batchlength, int startpoint, int maxThreads)
{
	int batch = blockIdx.x * blockDim.x + threadIdx.x;
	if(batch >= maxThreads)
		return;

	int batchJump = batch*batchlength;

	const int preambleLength = 128;
	const int asmLength = 64;
	const int dataLength = 6144;

	float BnT = .002;
	float K1 = 4*BnT/(1+2*BnT);
	K1 = K1/4;

	float PHASE = 0;
	int IBIT = 1;
	float e,v;

	float a_known[preambleLength+asmLength] = {1.000000,1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,-1.000000,-1.000000,-1.000000,-1.000000,-1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,1.000000,-1.000000,-1.000000,-1.000000,1.000000,1.000000,1.000000,-1.000000,1.000000,1.000000,1.000000,-1.000000,1.000000,1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,-1.000000,1.000000,1.000000,1.000000,-1.000000,-1.000000,1.000000,-1.000000,-1.000000,1.000000,1.000000,1.000000,-1.000000,-1.000000,1.000000,-1.000000,1.000000,-1.000000,-1.000000,-1.000000,1.000000,-1.000000,-1.000000,1.000000,-1.000000,1.000000,-1.000000,1.000000,1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,-1.000000,-1.000000,};

	cuComplex dfout1_current;
	cuComplex dfoutr_current;
	cuComplex dfoutr_past;
	float ahat_current;
	float ahat_past;

	// inside preamble or asm
	for(int idx = startpoint; idx < preambleLength+asmLength; idx++){
		dfoutr_past = dfoutr_current;
		ahat_past = ahat_current;
		dfout1_current = dfout1[batchJump+idx];
		// dfoutr_current =  dfout1[idx]*exp(-1i*PHASE)
		dfoutr_current.x =  dfout1_current.x     *cos(PHASE) 	- dfout1_current.y*(-1)*sin(PHASE);
		dfoutr_current.y =  dfout1_current.x*(-1)*sin(PHASE) 	+ dfout1_current.y     *cos(PHASE);
		if(IBIT){
			ahat_current = a_known[idx];
			e = 0;}
		else{
			ahat_current = a_known[idx];
			e = ahat_past*dfoutr_past.y-ahat_current*dfoutr_current.x;}
		v = K1*e;
		IBIT = !IBIT;
		PHASE = PHASE + v;
		ahat[batchJump+idx] = ahat_current;
	}

	// inside data
	for(int idx = preambleLength+asmLength; idx < (preambleLength+asmLength+dataLength); idx++){
		dfoutr_past = dfoutr_current;
		ahat_past = ahat_current;
		dfout1_current = dfout1[batchJump+idx];
		// dfoutr_current =  dfout1[idx]*exp(-1i*PHASE)
		dfoutr_current.x =  dfout1_current.x     *cos(PHASE) 	- dfout1_current.y*(-1)*sin(PHASE);
		dfoutr_current.y =  dfout1_current.x*(-1)*sin(PHASE) 	+ dfout1_current.y     *cos(PHASE);
		if(IBIT){
			ahat_current = 2*(dfoutr_current.x>0)-1;
			e = 0;}
		else{
			ahat_current = 2*(dfoutr_current.y>0)-1;
			e = ahat_past*dfoutr_past.y-ahat_current*dfoutr_current.x;}
		v = K1*e;
		IBIT = !IBIT;
		PHASE = PHASE + v;
		ahat[batchJump+idx] = ahat_current;
	}
}

__global__ void cudaDemodulatorFDE(const cuComplex *dfout1, float *ahat, int batchlength, int startpoint, int maxThreads)
{
	int batch = blockIdx.x * blockDim.x + threadIdx.x;
	if(batch >= maxThreads)
		return;

	int batchJump = batch*batchlength;

	const int preambleLength = 128;
	const int asmLength = 64;
	const int dataLength = 6144;

	float BnT = .002;
	float K1 = 4*BnT/(1+2*BnT);
	K1 = K1/4;

	float PHASE = 0;
	int IBIT = 1;
	float e,v;

	float a_known[preambleLength+asmLength] = {1.000000,1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,-1.000000,-1.000000,-1.000000,-1.000000,-1.000000,-1.000000,-1.000000,1.000000,1.000000,-1.000000,1.000000,-1.000000,-1.000000,-1.000000,1.000000,1.000000,1.000000,-1.000000,1.000000,1.000000,1.000000,-1.000000,1.000000,1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,-1.000000,1.000000,1.000000,1.000000,-1.000000,-1.000000,1.000000,-1.000000,-1.000000,1.000000,1.000000,1.000000,-1.000000,-1.000000,1.000000,-1.000000,1.000000,-1.000000,-1.000000,-1.000000,1.000000,-1.000000,-1.000000,1.000000,-1.000000,1.000000,-1.000000,1.000000,1.000000,-1.000000,1.000000,1.000000,-1.000000,-1.000000,-1.000000,-1.000000,};

	cuComplex dfout1_current;
	cuComplex dfoutr_current;
	cuComplex dfoutr_past;
	float ahat_current;
	float ahat_past;

	// inside preamble or asm
	for(int idx = 0; idx < preambleLength+asmLength-preambleLength/2; idx++){
		dfoutr_past = dfoutr_current;
		ahat_past = ahat_current;
		dfout1_current = dfout1[batchJump+idx];
		// dfoutr_current =  dfout1[idx]*exp(-1i*PHASE)
		dfoutr_current.x =  dfout1_current.x     *cos(PHASE) 	- dfout1_current.y*(-1)*sin(PHASE);
		dfoutr_current.y =  dfout1_current.x*(-1)*sin(PHASE) 	+ dfout1_current.y     *cos(PHASE);
		if(IBIT){
			ahat_current = a_known[idx];
			e = 0;}
		else{
			ahat_current = a_known[idx];
			e = ahat_past*dfoutr_past.y-ahat_current*dfoutr_current.x;}
		v = K1*e;
		IBIT = !IBIT;
		PHASE = PHASE + v;
		ahat[batchJump+idx+preambleLength/2] = ahat_current;
	}

	// inside data
	for(int idx = preambleLength+asmLength-preambleLength/2; idx < (preambleLength+asmLength+dataLength)-preambleLength/2; idx++){
		dfoutr_past = dfoutr_current;
		ahat_past = ahat_current;
		dfout1_current = dfout1[batchJump+idx];
		// dfoutr_current =  dfout1[idx]*exp(-1i*PHASE)
		dfoutr_current.x =  dfout1_current.x     *cos(PHASE) 	- dfout1_current.y*(-1)*sin(PHASE);
		dfoutr_current.y =  dfout1_current.x*(-1)*sin(PHASE) 	+ dfout1_current.y     *cos(PHASE);
		if(IBIT){
			ahat_current = 2*(dfoutr_current.x>0)-1;
			e = 0;}
		else{
			ahat_current = 2*(dfoutr_current.y>0)-1;
			e = ahat_past*dfoutr_past.y-ahat_current*dfoutr_current.x;}
		v = K1*e;
		IBIT = !IBIT;
		PHASE = PHASE + v;
		ahat[batchJump+idx+preambleLength/2] = ahat_current;
	}
}

__global__ void bitPrune(unsigned char *out, float *in, int frontPrune, int outputlength, int inputLength, int maxThreads)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= maxThreads)
		return;

	int batch = i / outputlength;
	int indexInBatch = i % outputlength;
	int batchInJump = batch*inputLength;

	int indexOutBatch = i % outputlength;
	int batchOutJump = batch*outputlength;
	int frontJump = frontPrune;

	out[batchOutJump+indexOutBatch] = (char)(in[batchInJump + frontJump + indexInBatch]>0);
}


__global__ void stripSignalFloatToComplex(float *in_i, float *in_q, cuComplex *out, int *maxLocations, int packetLength, int maxThreads){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= maxThreads)
		return;

	int packet = i/packetLength;
	int i_packet = i%packetLength;

	if(packet<3104){
		cuComplex temp;
		temp.x = in_i[maxLocations[packet]+i_packet];
		temp.y = in_q[maxLocations[packet]+i_packet];
		//		temp.x = in_i[i];
		//		temp.y = in_q[i];

		out[i] = temp;
	}
}

__global__ void stripSignalComplexToComplex(cuComplex *in, cuComplex *out, int *maxLocations, int packetLength, int maxThreads){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= maxThreads)
		return;

	int packet = i/12672;
	int i_packet = i%12672;

	out[i] = in[maxLocations[packet]+i_packet];
}

__global__ void signalStripper(cuComplex *in, cuComplex *out, int packetLength, int r1Start, int r1Length, int r1Conj, int maxThreads){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= maxThreads)
		return;

	// Find what thread batch we are in
	int threadBatch = i / r1Length;
	int threadIndex = i % r1Length;

	// Find how far we need to jump on the input
	int inputBatchJump = threadBatch*packetLength;

	// Find how far we need to jump into input batch
	int frontJump = r1Start;

	// Find how far we need to jump on the output
	int outputBatchJump = threadBatch*r1Length;

	out[outputBatchJump+threadIndex].x = in[inputBatchJump + frontJump + threadIndex].x;
	if(r1Conj)
		out[outputBatchJump+threadIndex].y = -in[inputBatchJump + frontJump + threadIndex].y;
	else
		out[outputBatchJump+threadIndex].y = in[inputBatchJump + frontJump + threadIndex].y;
}


__global__ void cudaConj(cuComplex *array, int maxThreads){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= maxThreads)
		return;

	array[i].y = -array[i].y;
}

__global__ void tanAndLq(float* out, cuComplex *in, int maxThreads){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= maxThreads)
		return;

	out[i] = atan2(in[i].y,in[i].x)/LQ;
}

__global__ void w0Ave(float* w0, int MAX_PACKETS_PER_MEMORY_SECTION){
	float w0_sum = 0;
	for(int i = 0; i < MAX_PACKETS_PER_MEMORY_SECTION-1; i++)
		w0_sum += w0[i];
	w0_sum = w0_sum/(MAX_PACKETS_PER_MEMORY_SECTION-1);

	w0[0] = w0_sum;
}

__global__ void derotate(cuComplex* data, float* W0, int batchLength, int maxThreads){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= maxThreads)
		return;

	// Find what thread batch we are in
	int threadBatch = i / batchLength;
	int threadIndex = i % batchLength;

	float PHASE = W0[threadBatch]*threadIndex;

	cuComplex myData = data[i];
	cuComplex temp;

	temp.x = myData.x	*cos(PHASE)	+ myData.y	*sin(PHASE);
	temp.y = -myData.x	*sin(PHASE)	+ myData.y	*cos(PHASE);

	//	Backwards
	//	temp.x = myData.x	*cos(PHASE)	- myData.y	*sin(PHASE);
	//	temp.y = myData.x	*sin(PHASE)	+ myData.y	*cos(PHASE);

	data[i] = temp;
}

__global__ void derotateBatchSumW0(cuComplex* data, float* w0, int batchLength, int maxThreads){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= maxThreads)
		return;

	// Find what thread batch we are in
	//	int threadBatch = i / batchLength;
	//	int threadIndex = i % batchLength;

	float PHASE = w0[0];

	cuComplex myData = data[i];
	cuComplex temp;

	temp.x = myData.x	*cos(PHASE)	+ myData.y	*sin(PHASE);
	temp.y = -myData.x	*sin(PHASE)	+ myData.y	*cos(PHASE);

	//	Backwards
	//	temp.x = myData.x	*cos(PHASE)	- myData.y	*sin(PHASE);
	//	temp.y = myData.x	*sin(PHASE)	+ myData.y	*cos(PHASE);

	data[i] = temp;
}

__global__ void subAndSquare(cuComplex *r2, cuComplex *intermediate, float *diffMag2, int maxThreads){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= maxThreads)
		return;

	float real = r2[i].x - intermediate[i].x;
	float imag = r2[i].y - intermediate[i].y;

	diffMag2[i] = real*real + imag*imag;
}

__global__ void sumAndScale(float *noiseVariance, float *diffMag2, int maxThreads){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= maxThreads)
		return;

	int batchJump = i*347;

	float temp;
	temp = 0;
	for(int sumIndex = 0; sumIndex < 347; sumIndex++)
		temp += diffMag2[batchJump + sumIndex];

	temp = .00161812 * temp;

	noiseVariance[i] = temp;
}


__global__ void bit8Channels(unsigned char *out, unsigned char *in, int channel, int maxThreads)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= maxThreads)
		return;

	int firstIndexToGrab = i*8;


	unsigned char bit0 = (in[firstIndexToGrab + 0] & 0x01) << 0;
	unsigned char bit1 = (in[firstIndexToGrab + 1] & 0x01) << 1;
	unsigned char bit2 = (in[firstIndexToGrab + 2] & 0x01) << 2;
	unsigned char bit3 = (in[firstIndexToGrab + 3] & 0x01) << 3;
	unsigned char bit4 = (in[firstIndexToGrab + 4] & 0x01) << 4;
	unsigned char bit5 = (in[firstIndexToGrab + 5] & 0x01) << 5;
	unsigned char bit6 = (in[firstIndexToGrab + 6] & 0x01) << 6;
	unsigned char bit7 = (in[firstIndexToGrab + 7] & 0x01) << 7;

	unsigned char output = bit7 | bit6 | bit5 | bit4 | bit3 | bit2 | bit1 | bit0;

	int outputIndex = i*8 + channel-1;
	out[outputIndex] = output;
}

__global__ void cudaCMAz(cuComplex *y1, cuComplex *e, int maxThreads){
	int thread = blockIdx.x * blockDim.x + threadIdx.x;
	if(thread >= maxThreads)
		return;

	float real = y1[thread].x;
	float imag = y1[thread].y;
	e[thread].x = 4*(real*real*real + real*imag*imag - real);
	e[thread].y = 4*(imag*imag*imag + imag*real*real - imag);
}

__global__ void cudaCMAdelJ(cuComplex *delJ_in, cuComplex *e_in, cuComplex *r_in, const int batchLength, const int L1, const float mu, int maxThreads){
	int thread = blockIdx.x * blockDim.x + threadIdx.x;
	if(thread >= maxThreads)
		return;

	// Find what thread batch we are in
	int threadBatch = thread / batchLength;
	int threadIndex = thread % batchLength;

	cuComplex* delJ = &delJ_in[threadBatch*batchLength];
	cuComplex* e 	= &e_in[threadBatch*batchLength];
	cuComplex* r 	= &r_in[threadBatch*batchLength];

	delJ[threadIndex].x = 0;
	delJ[threadIndex].y = 0;
	for(int idx = 0; idx < batchLength; idx++){
		if(L1+idx-threadIndex >= 0 && L1+idx-threadIndex+1 <= batchLength){
			//delJ(threadIndex) = delJ(threadIndex) + e(idx)*conj(r(L1+idx-threadIndex+1));
			float C = e[idx].x;
			float D = e[idx].y;
			float E = r[L1+idx-threadIndex].x;
			float F = r[L1+idx-threadIndex].y;
			delJ[threadIndex].x = delJ[threadIndex].x + C*E + D*F;
			delJ[threadIndex].y = delJ[threadIndex].y - C*F + E*D;
		}
	}
	delJ[threadIndex].x = mu*delJ[threadIndex].x/batchLength;
	delJ[threadIndex].y = mu*delJ[threadIndex].y/batchLength;
}

__global__ void cudaCMAflipLR(cuComplex *out_in, cuComplex *in_in, int batchLength, int maxThreads){
	int thread = blockIdx.x * blockDim.x + threadIdx.x;
	if(thread >= maxThreads)
		return;

	// Find what thread batch we are in
	int threadBatch = thread / batchLength;
	int threadIndex = thread % batchLength;

	cuComplex* in 	= &in_in[threadBatch*batchLength];
	cuComplex* out	= &out_in[threadBatch*batchLength];

	out[batchLength-1-threadIndex] = in[threadIndex];
}

__global__ void zeroPadConj(cuComplex *out, const cuComplex *in, int batches, int old_length, int new_length){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= batches*old_length)
		return;
	int batch = i/old_length;
	int new_i = i%old_length;

	cuComplex temp = in[i];
	temp.y = -temp.y;
	out[batch*new_length+new_i] = temp;
}

__global__ void stripAndScale(cuComplex *delJ_out, cuComplex *efft_in, float mu, int forwardJump, int outputLength, int inputLength, int maxThreads){
	int thread = blockIdx.x * blockDim.x + threadIdx.x;
	if(thread >= maxThreads)
		return;

	int threadBatch = thread / outputLength;
	int threadIndex = thread % outputLength;

	cuComplex* input	= &efft_in[threadBatch*inputLength];
	cuComplex* output 	= &delJ_out[threadBatch*outputLength];

	float constant = mu / (12672*inputLength);
	output[outputLength-1-threadIndex].x = input[threadIndex+12546].x*constant;
	output[outputLength-1-threadIndex].y = input[threadIndex+12546].y*constant;
}

__global__ void cudaUpdateCoefficients(cuComplex *c, cuComplex *delJ, int maxThreads){
	int thread = blockIdx.x * blockDim.x + threadIdx.x;
	if(thread >= maxThreads)
		return;

	c[thread].x = c[thread].x - delJ[thread].x;
	c[thread].y = c[thread].y - delJ[thread].y;
}

__global__ void cudaConvGlobal(float *filtered, float *signal, float *filter, int maxThreads){
	int thread = blockIdx.x * blockDim.x + threadIdx.x;
	if(thread >= maxThreads)
		return;

	int filterLength = 21;
	int SAMPLES_PER_PACKET = 21;

	int realThread = thread*2+filterLength/2;
	float sum = 0;
	for(int inner = 0; inner<filterLength; inner++){
		if(realThread-inner>=0 && realThread-inner<SAMPLES_PER_PACKET){
			sum += signal[realThread-inner]*filter[inner];
		}
	}
	filtered[thread] = sum;
}

//dev_L_pd
__global__ void cudaFirstMaxSearch(float *quickMaxSearch, int SAMPLES_PER_PACKET, int *endIndex_FirstWindow, int *myFirstMax, int *myFirstMaxActual){
	//Search the first 12672 for a max to set the search window half a packet before each expected preamble start

	float maxVal = 0;
	int firstMax = 0;
	for(int u=SAMPLES_PER_PACKET*0; u < SAMPLES_PER_PACKET*10; u++)
		if(quickMaxSearch[u] > maxVal){
			maxVal = quickMaxSearch[u];
			firstMax = u;
			if(u<SAMPLES_PER_PACKET)
				myFirstMaxActual[0] = u;
		}
	myFirstMax[0] = firstMax%SAMPLES_PER_PACKET;

	//Add Half a packet to the first max to center windows on Guesstimated Preambles
	endIndex_FirstWindow[0] = firstMax + SAMPLES_PER_PACKET/2;
}
__global__ void cudaFirstMaxSearch(cuComplex *quickMaxSearch, int SAMPLES_PER_PACKET, int *endIndex_FirstWindow, int *myFirstMax, int *myFirstMaxActual){
	//Search the first 12672 for a max to set the search window half a packet before each expected preamble start

	float maxVal = 0;
	int firstMax = 0;
	for(int u=SAMPLES_PER_PACKET*0; u < SAMPLES_PER_PACKET*10; u++)
		if(quickMaxSearch[u].x > maxVal){
			maxVal = quickMaxSearch[u].x;
			firstMax = u;
			if(u<SAMPLES_PER_PACKET)
				myFirstMaxActual[0] = u;
		}
	myFirstMax[0] = firstMax%SAMPLES_PER_PACKET;

	//Add Half a packet to the first max to center windows on Guesstimated Preambles
	endIndex_FirstWindow[0] = firstMax + SAMPLES_PER_PACKET/2;
}

__global__ void cudaLongestChainSearch(int *max_locations_in, int MAX_PACKETS_PER_MEMORY_SECTION){
	//	int maxLength = 0;
	//	int length = 0;
	//	int maxIdx = 0;
	//	int idx = 0;
	//
	//	//Find longest chain of correct values
	//	bool inChain = false;
	//	for(int i=0; i < MAX_PACKETS_PER_MEMORY_SECTION; i++)
	//		if(abs((max_locations[i+1] - max_locations[i]) - 12672) <= 3)
	//			if(inChain)
	//				length++;
	//			else {
	//				length = 1;
	//				idx = i;
	//				inChain = true;}
	//		else
	//			if(inChain) {
	//				inChain = false;
	//				if(length > maxLength) {
	//					maxLength = length;
	//					maxIdx = idx;}}
	//
	//	max_locations[0] = max_locations[maxIdx] % 12672;

	__shared__ int max_locations[3104];
	for(int i = 0; i < 3104; i++)
		max_locations[i] = max_locations_in[i];


	int chainFront = 0;
	int chainLength = 0;

	int longestChainFront = 0;
	int longestChainBack = 0;
	int longestChainLength = 0;

	int Accordian_leading_pos32_count = 0;
	int Accordian_leading_neg32_count = 0;

	for(int i = 1; i<3103-1; i++){
		int myDiffPast = max_locations[i] - max_locations[i-1] - 12672;
		int myDiffNext = max_locations[i+1] - max_locations[i] - 12672;
		bool goodSpacing = (myDiffPast==0) || (abs(myDiffPast)==1) || (abs(myDiffPast)==2);

//		if(goodSpacing){
//			if(chainLength == 0)
//				chainFront = i;
//			chainLength++;
//			if(longestChainLength<chainLength){
//				longestChainLength = chainLength;
//				longestChainFront = chainFront;
//				longestChainBack = i;
//			}
//		}
//		else
//			chainLength = 0;

		bool past_pos32 =
				(myDiffPast==32+2) ||
				(myDiffPast==32+1) ||
				(myDiffPast==32+0) ||
				(myDiffPast==32-1) ||
				(myDiffPast==32-2);

		bool next_neg32 =
				(myDiffNext==-32+2) ||
				(myDiffNext==-32+1) ||
				(myDiffNext==-32+0) ||
				(myDiffNext==-32-1) ||
				(myDiffNext==-32-2);
		bool past_neg32 =
				(myDiffPast==-32+2) ||
				(myDiffPast==-32+1) ||
				(myDiffPast==-32+0) ||
				(myDiffPast==-32-1) ||
				(myDiffPast==-32-2);
		bool next_pos32 =
				(myDiffNext==32+2) ||
				(myDiffNext==32+1) ||
				(myDiffNext==32+0) ||
				(myDiffNext==32-1) ||
				(myDiffNext==32-2);

		bool Accordian_leading_pos32 = past_pos32 && next_neg32;
		bool Accordian_leading_neg32 = past_neg32 && next_pos32;

		if(Accordian_leading_pos32)
			max_locations[i] = max_locations[i] - 32;
		if(Accordian_leading_neg32)
			max_locations[i] = max_locations[i] + 32;
	}

    bool goodSpacingFirst2 = abs(max_locations[2]-max_locations[1]-12672)<2;
    bool first_pos32 =
        max_locations[1]-max_locations[0]-12672==32-1 ||
        max_locations[1]-max_locations[0]-12672==32+0 ||
        max_locations[1]-max_locations[0]-12672==32+1;
    bool first_neg32 =
        max_locations[1]-max_locations[0]-12672==-32-1 ||
        max_locations[1]-max_locations[0]-12672==-32+0 ||
        max_locations[1]-max_locations[0]-12672==-32+1;
    if(goodSpacingFirst2 && first_pos32)
        max_locations[0] = max_locations[0] + 32;
    else if(goodSpacingFirst2 && first_neg32)
        max_locations[0] = max_locations[0] - 32;

    bool goodSpacingLast2 = abs(max_locations[3103-1]-max_locations[3103-2]-12672)<2;
    bool last_pos32 =
        max_locations[3103]-max_locations[3103-1]-12672==32-1 ||
        max_locations[3103]-max_locations[3103-1]-12672==32+0 ||
        max_locations[3103]-max_locations[3103-1]-12672==32+1;
    bool last_neg32 =
        max_locations[3103]-max_locations[3103-1]-12672==-32-1 ||
        max_locations[3103]-max_locations[3103-1]-12672==-32+0 ||
        max_locations[3103]-max_locations[3103-1]-12672==-32+1;
    if(goodSpacingFirst2 && last_pos32)
        max_locations[3103] = max_locations[3103] - 32;
    else if(goodSpacingFirst2 && last_neg32)
        max_locations[3103] = max_locations[3103] + 32;


    goodSpacingLast2 = abs(max_locations[3102-1]-max_locations[3102-2]-12672)<2;
    last_pos32 =
        max_locations[3102]-max_locations[3102-1]-12672==32-1 ||
        max_locations[3102]-max_locations[3102-1]-12672==32+0 ||
        max_locations[3102]-max_locations[3102-1]-12672==32+1;
    last_neg32 =
        max_locations[3102]-max_locations[3102-1]-12672==-32-1 ||
        max_locations[3102]-max_locations[3102-1]-12672==-32+0 ||
        max_locations[3102]-max_locations[3102-1]-12672==-32+1;
    if(goodSpacingFirst2 && last_pos32)
        max_locations[3102] = max_locations[3102] - 32;
    else if(goodSpacingFirst2 && last_neg32)
        max_locations[3102] = max_locations[3102] + 32;


	for(int i = 0; i < 3104; i++)
		max_locations_in[i] = max_locations[i];

}

__global__ void cudaMaxAdjust(int *max_locations, int SAMPLES_PER_PACKET, int maxThreads){
	int thread = blockIdx.x * blockDim.x + threadIdx.x;
	if(thread >= maxThreads)
		return;

	//max_locations[thread] = max_locations[0] + thread*SAMPLES_PER_PACKET;

	int i;
	int gospel;

	int count = 0;
	for(i = 0; i < 3104; i++){
		if(abs(max_locations[i+1]-max_locations[i]-12672)<2)
			count++;
		else
			count = 0;
		if(count>10){
			gospel = i;
			i = 4000;
		}
	}
	for(i = gospel-1; i<3104; i++){
		if(abs(max_locations[i+1]-max_locations[i]-12672)>15)
			max_locations[i+1] = max_locations[i]+12672;
	}
}


__global__ void pointMultiplyQuadFDE1(cuComplex *Y_in, const cuComplex *H_in, const float *shs_in, const cuComplex *D_in, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= size)
		return;
	float Y_R = Y_in[i].x;
	float Y_I = Y_in[i].y;
	float H_R = H_in[i].x;
	float H_I = H_in[i].y;
	float DF_R = D_in[i].x;
	float DF_I = D_in[i].y;
	float S   = shs_in[0];

	float A, B, C, D;
	cuComplex out;

	cuComplex EQ;
	EQ.x = H_R / (H_R*H_R + H_I*H_I + S);
	EQ.y = -H_I / (H_R*H_R + H_I*H_I + S);

	cuComplex VP;
	VP.x = Y_R;
	VP.y = Y_I;

	cuComplex DF;
	DF.x = DF_R;
	DF.y = DF_I;

	//	Y * EQ
	A = VP.x;
	B = VP.y;
	C = EQ.x;
	D = EQ.y;
	out.x = A*C - B*D;
	out.y = A*D + B*C;

	//	(Y*EQ) * DF
	A = out.x;
	B = out.y;
	C = DF.x;
	D = DF.y;
	out.x = A*C - B*D;
	out.y = A*D + B*C;

	Y_in[i] = out;
}

__global__ void pointMultiplyQuadFDE2(cuComplex *Y_in, const cuComplex *H_in, const float *shs_in, const cuComplex *D_in, const float *PSI_in, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= size)
		return;
	float Y_R = Y_in[i].x;
	float Y_I = Y_in[i].y;
	float H_R = H_in[i].x;
	float H_I = H_in[i].y;
	float DF_R = D_in[i].x;
	float DF_I = D_in[i].y;
	float S   = shs_in[0] * PSI_in[i];

	float A, B, C, D;
	cuComplex out;

	cuComplex EQ;
	EQ.x = H_R / (H_R*H_R + H_I*H_I + S);
	EQ.y = -H_I / (H_R*H_R + H_I*H_I + S);

	cuComplex VP;
	VP.x = Y_R;
	VP.y = Y_I;

	cuComplex DF;
	DF.x = DF_R;
	DF.y = DF_I;

	//	Y * EQ
	A = VP.x;
	B = VP.y;
	C = EQ.x;
	D = EQ.y;
	out.x = A*C - B*D;
	out.y = A*D + B*C;

	//	(Y*EQ) * DF
	A = out.x;
	B = out.y;
	C = DF.x;
	D = DF.y;
	out.x = A*C - B*D;
	out.y = A*D + B*C;

	Y_in[i] = out;
	//	int i = blockIdx.x * blockDim.x + threadIdx.x;
	//	if(i >= size)
	//		return;
	//	float Y_R = Y_in[i].x;
	//	float Y_I = Y_in[i].y;
	//	float H_R = H_in[i].x;
	//	float H_I = H_in[i].y;
	//	float DF_R = D_in[i].x;
	//	float DF_I = D_in[i].y;
	//	float S   = shs_in[0];
	//	float P_R = PSI_in[i].x;
	//	float P_I = PSI_in[i].y;
	//
	//	float A, B, C, D, den;
	//	cuComplex out;
	//
	//	// EQ = conj(H) / (|H|^2 + shs*PSI)
	//	cuComplex EQ;
	//	A = H_R;
	//	B = -H_I;
	//	C = H_R*H_R + H_I*H_I + S*P_R;
	//	D = S*P_I;
	//	den = C*C + D*D;
	//	EQ.x = (A*C + B*D)/den;
	//	EQ.y = (B*C - A*D)/den;
	//
	//	cuComplex VP;
	//	VP.x = Y_R;
	//	VP.y = Y_I;
	//
	//	cuComplex DF;
	//	DF.x = DF_R;
	//	DF.y = DF_I;
	//
	//	//	Y * EQ
	//	A = VP.x;
	//	B = VP.y;
	//	C = EQ.x;
	//	D = EQ.y;
	//	out.x = A*C - B*D;
	//	out.y = A*D + B*C;
	//
	//	//	(Y*EQ) * DF
	//	A = out.x;
	//	B = out.y;
	//	C = DF.x;
	//	D = DF.y;
	//	out.x = A*C - B*D;
	//	out.y = A*D + B*C;
	//
	//	Y_in[i] = out;
}

__global__ void zeroPadFreq(cuComplex *out, const cuComplex *in, int batches, int old_length, int new_length, int front_jump)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= batches*old_length)
		return;
	int batch = i/old_length;
	int thread_i = i%old_length;

	int shifted_i = thread_i + front_jump;
	int rotated_i = thread_i - old_length + front_jump;

	int pull_i;
	if(thread_i < old_length-front_jump)
		pull_i = shifted_i;
	else
		pull_i = rotated_i;

	// Get the index into the current batch
	pull_i += batch*old_length;

	out[batch*new_length+thread_i] = in[pull_i];
}

__global__ void rotateHalfPreambleBack(cuComplex *out, cuComplex *in, int Np, int maxThreads)
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i >= maxThreads)
		return;

	int batch = i/(12672/2);
	int thread_i = i%(12672/2);

	int shifted_i = thread_i - Np/4;

	out[batch*12672/2 + thread_i] = in[batch*12672/2 + shifted_i];
}

__global__ void PSIfill(float *array, int conv_length, int maxThreads)
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i >= maxThreads)
		return;

	array[i] = array[i%conv_length];
}

__global__ void pointToPointConj(cuComplex *out, cuComplex *in, int max)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= max)
		return;

	//	PN11A * conj(BITS)
	//	PN11A = A + jB
	//	BITS  = C - jD
	float A = in[i].x;
	float B = in[i].y;
	float C = out[i].x;
	float D = out[i].y;
	out[i].x =  A*C + B*D;
	out[i].y = -A*D + B*C;
}

__global__ void rotateXcorrBits(cuComplex *out, cuComplex *in, int length_bits, int Nfft, int maxThreads)
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i >= maxThreads)
		return;

	int shifted_i = i - (length_bits-1);
	int rotated_i = i - (length_bits-1) + Nfft;

	int pull_i;
	if(i > length_bits-1)
		pull_i = shifted_i;
	else
		pull_i = rotated_i;

	out[i].x = in[pull_i].x/Nfft;
	out[i].y = in[pull_i].y/Nfft;
}

__global__ void pullBitsXcorr(cuComplex *array, unsigned char *bits, int maxThreads)
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i > maxThreads)
		return;

	cuComplex temp;
	temp.x = ((float)bits[i])*2-1;
	temp.y = 0;
	array[i] = temp;
}

__global__ void peakSearchXcorr(int *peak, int *peakIdx, int *xcorr_in, int maxIndex, int maxThreads)
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i >= maxThreads)
		return;

	int* xcorr = &xcorr_in[i*2047];

	int max = 0;
	int maxIdx;

	int test;
	for(int j = 0; j < 2047; j++){
		if(i*2047+j < maxIndex){
			test = xcorr[j];
			if(test > max){
				max = test;
				maxIdx = j;
			}
		}
	}

	peak[i] 	= max;
	peakIdx[i] 	= i*2047+maxIdx;
}

__global__ void pullxCorrBits(int *xcorr, cuComplex *array, int Nfft, int processedBits, int maxThreads)
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i > maxThreads)
		return;

	int pullIdx = Nfft - (processedBits-2047) + i;

	xcorr[i] = (int)round(array[pullIdx].x/Nfft);
}

__global__ void shiftFDEblindlyForward(cuComplex *out, const cuComplex *in, int shift, int maxThreads)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= maxThreads && i < shift)
		return;

	out[i] = in[i-shift];
}

__global__ void copyPreambleShiftOnEnd(cuComplex *array, int shift)
{
	for(int i = 0; i < shift; i++)
		array[3104*12672+i] = array[3103*12672+i];
}

/*__global__ void runHalfBandFilter(cuComplex *y_p, float *x, float *halfBandFilter, int maxThreads)
{
	int k = blockIdx.x*blockDim.x+threadIdx.x;
	if(k >= maxThreads)
		return;


	cuComplex temp;
	temp.x = 0;
	temp.y = 0;

	for(int convIdx = 1; convIdx <= 18; convIdx++)
		temp.y = temp.y + halfBandFilter[convIdx-1]*x[(k+1)*2-1-2*convIdx+3+34-1];
	temp.x = temp.x + x[(k+1)*2-1-16+34-1];
	if((k+1)%2 == 0){
		temp.x = -temp.x;
		temp.y = -temp.y;
	}

	y_p[k] = temp;

}*/

//__global__ void runHalfBandFilter(cuComplex *y_p, unsigned short *x, float *halfBandFilter, int maxThreads){
//	int k = blockIdx.x*blockDim.x+threadIdx.x;
//	if(k >= maxThreads)
//		return;
//
//	cuComplex temp;
//	temp.x = 0;
//	temp.y = 0;
//	for(int convIdx = 1; convIdx <= 18; convIdx++)
//		temp.y = temp.y + halfBandFilter[convIdx-1]*x[(k+1)*2-1-2*convIdx+3+34-1];
//	temp.x = temp.x + x[(k+1)*2-1-16+34-1];
//	if((k+1)%2 == 0){
//		temp.x = -temp.x;
//		temp.y = -temp.y;
//	}
//	y_p[k] = temp;
//}

__global__ void convertShortToFloat(float *x_out, unsigned short *x_in, int maxThreads){

	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx > maxThreads)
		return;
	float temp = (float) x_in[idx];

	x_out[idx] = temp; // cut out the DC stuff!!!
}

//__global__ void runHalfBandFilter(cuComplex *y22, float *x, float *g1, int maxThreads){
//
//	int idx1 = blockIdx.x*blockDim.x+threadIdx.x;
//	if(idx1 >= maxThreads)
//		return;
//	idx1++;
//
//	cuComplex temp;
//	temp.x = 0;
//	temp.y = 0;
//	for (int idx2 = 1; idx2 <= 18; idx2++)
//		temp.y = temp.y + g1[idx2] * x[2*idx1 - 2*(idx2-1) + 35-1 -2];//-2 for matlab->c++ indexing 2(-1)
//	temp.x = x[2*idx1 - 1 - 2*(9-1) + 35-1];
//
//	if(idx1%2 == 1){
//		temp.x = -temp.x;
//		temp.y = -temp.y;
//	}
//	y22[idx1-1] = temp;
//}

__global__ void runHalfBandFilter(cuComplex *y_p, unsigned short *x, float *halfBandFilter, int maxThreads){

	//	int myLength = 100000-1;
	//	cuComplex y_p[myLength];
	//	for(int idx1 = 1; idx1<=myLength; idx1++){

	int idx1 = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx1 >= maxThreads)
		return;
	idx1++;

	int Lh = 18*2-1;


	cuComplex temp;
	temp.x = 0;
	temp.y = 0;
	for(int idx2 = 1; idx2<=18; idx2++){
		float myUltra = (float)x[2*idx1 - 2*(idx2-1) + Lh-1 -1];
		temp.y += halfBandFilter[idx2-1]*(myUltra-8192.0);
	}
	temp.x = ((float)x[2*idx1 - 1 - 2*(9-1) + Lh-1 -1])-8192.0;
	if(!(idx1%2)){
		temp.x = -temp.x;
		temp.y = -temp.y;
	}
	//	temp.x = idx1-1;
	//	temp.y = 0;
	y_p[idx1-1] = temp;
	//	}
}



//__global__ void runPolyPhaseFilter(cuComplex *z_p, cuComplex *y_p, float *newFilterBank, int maxThreads){
//	int idx = blockIdx.x*blockDim.x+threadIdx.x;
//	if(idx >= maxThreads)
//		return;
//
//	int inQ = (idx+1)%99;
//
//	int jeff = 222;
//	if(inQ != 0)
//		jeff = inQ*2 + inQ/4 -1 + 1*(inQ>79)*(inQ < 21 || inQ > 79) -1*(inQ%4 == 0)*(inQ < 21 || inQ > 79) +1*(((inQ%4 == 2) && (inQ < 80 && inQ > 60)) || inQ%4 == 3)*(inQ<80 && inQ>40);
//
//	int polyBatch = idx/99;
//	int polyIdx = idx%99;
//	cuComplex temp;
//	temp.x = 0;
//	temp.y = 0;
//
//	for(int convIdx = 0; convIdx <= 19; convIdx++){
//		int yIdx = polyBatch*224+jeff-convIdx-1+20;
//		int fIdx = polyIdx*20+convIdx;
//		temp.x += y_p[yIdx].x*newFilterBank[fIdx];
//		temp.y += y_p[yIdx].y*newFilterBank[fIdx];
//	}
//	z_p[idx] = temp;
//}

__global__ void runPolyPhaseFilter(cuComplex *z_p, cuComplex *y_p, float *newFilterBank, int maxThreads){
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx >= maxThreads)
		return;

	int myLength = 100000-1;
	int inQ = (idx+1)%99;

	int jeff = 222;
	if(inQ != 0)
		jeff = inQ*2 + inQ/4 -1 + 1*(inQ>79)*(inQ < 21 || inQ > 79) -1*(inQ%4 == 0)*(inQ < 21 || inQ > 79) +1*(((inQ%4 == 2) && (inQ < 80 && inQ > 60)) || inQ%4 == 3)*(inQ<80 && inQ>40);

	int polyBatch = idx/99;
	int polyIdx = idx%99;
	cuComplex temp;
	temp.x = 0;
	temp.y = 0;

	for(int convIdx = 0; convIdx <= 19; convIdx++){
		int yIdx = polyBatch*224+jeff-convIdx-1;
		int fIdx = polyIdx*20+convIdx;
		temp.x += y_p[yIdx].x*newFilterBank[fIdx];
		temp.y += y_p[yIdx].y*newFilterBank[fIdx];
	}
	z_p[idx] = temp;
}

__global__ void ComplexSamplesToiORq(float *i, float *q, cuComplex *z_p, bool conj, int maxThreads){
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx >= maxThreads)
		return;

	cuComplex myGrab;
	myGrab = z_p[idx];

	i[idx] = myGrab.x/32.8997;
	if(conj)
		q[idx] = -myGrab.y/32.8997;
	else
		q[idx] = myGrab.y/32.8997;
}

__global__ void floatToSamples(cuComplex *complexOut,float *i, float *q, int maxThreads)
{
	int thread = blockIdx.x * blockDim.x + threadIdx.x;
	if(thread >= maxThreads)
		return;

	cuComplex temp;
	temp.x = i[thread]/67108864;
	temp.y = q[thread]/67108864;

	complexOut[thread] = temp;
}

__global__ void PointToPointMultiply(cuComplex* v0, cuComplex* v1, int lastThread){
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	// don't access elements out of bounds
	if(i >= lastThread)
		return;
	float A = v0[i].x;
	float B = v0[i].y;
	float C = v1[i].x;
	float D = v1[i].y;

	// (A+jB)(C+jD) = (AC-BD) + j(AD+BC)
	cuComplex result;
	result.x = A*C-B*D;
	result.y = A*D+B*C;

	v0[i] = result;
}

__global__ void myAbs(cuComplex *complexOut, int maxThreads)
{
	int thread = blockIdx.x * blockDim.x + threadIdx.x;
	if(thread >= maxThreads)
		return;

	cuComplex temp;
	temp = complexOut[thread+383];

	float A = temp.x;
	float B = temp.y;

	temp.x = sqrt(A*A + B*B);
	temp.y = 0;

	complexOut[thread] = temp;
}
