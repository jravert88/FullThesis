/* 
 * File:   Kernels.h
 * Author: Andrew McMurdie
 *
 * Created on April 3, 2013, 8:43 AM
 */

#ifndef KERNELS_H
#define	KERNELS_H

//#include <cublas.h>
//#include <cublas_api.h>
#include <cublas_v2.h>

//Function definitions
__global__ void runFilterCuda(float* I, float* Q, int samplesLength, float* filter, int filterLength, float* filtered_I, float* filtered_Q, int convLength);
__global__ void cudaRunComplexFilter(float* I, float* Q, int samplesLength, float* hr, float* hi, int filterLength, float* filtered_I, float* filtered_Q, int convLength);
__global__ void downsampleCuda(float* I, float* Q, unsigned int numDownsampledSamples, float* downsampled_I, float* downsampled_Q, unsigned int factor);
__device__ float atomicAdd(float* address, float val);
__device__ int sign(float input);
__device__ int cudaDecodeBits(float i, float q);
__global__ void cudaCalculateErrorPED(float* result, float delayedX, float delayedY, float xInterpolant, float xDelayedTwo, float xDelayedThree, float yInterpolant, float yDelayedTwo, float yDelayedThree, unsigned int state);
__device__ float cudaCalcTimingError(float ted1,float ted2, float ted3, float ted4, float ted5, float xInterpolant, float xDelayedTwo, float xDelayedThree, float yInterpolant, int state);
__device__ float cudaCalcTED2(float ted1, float xDelayedTwo, float xDelayedThree, int state);
__global__ void cudaDemodLoop(float* i_samples, float* q_samples, int sample_size, int* bitdecisions, float* constants);
__global__ void cudaConvertToBits(int* bit_decisions, unsigned short* bit_stream, int dec_size);
__global__ void cudaDecodeBitstream(unsigned short* encoded, unsigned short* decoded, int size);

// Miscellaneous functions used in the preamble detector
__global__ void cudaFindMax(int* results, float* u_sums, int size);
__global__ void cudaFindMaxOptimized(float* u_sums, int size, float* maxes, float* locations);
__global__ void cudaFindMaxOptimizedPart2(int* results, float* maxes, float* locations, int num_sections);

// Kernels using the save method
__global__ void calculateInnerSumBlocks(float* i, float* q, float* innerSums, int uLength, int innerSumsLength);
__global__ void calculateOuterSums(float* innerSums, float* L, int uLength);

// Simple Correlator
__global__ void cudaSimpleCorrelator(float* xi, float* xq, float* sr, float* si, int sLength, float* L, int uLength);

// Choi-Lee Correlator
__global__ void cudaChoiLee(float* xi, float* xq, float* sr, float* si, int N, float* L);
__global__ void cudaSumDataCorrection(float* i_samples, float* q_samples, float* r, int N);
__global__ void cudaAddCorrAndCorrection(float* L, float* r, int N);
__global__ void cudaChoiLeeFull(float* xi, float* xq, float* sr, float* si, int N, float* L, float* L2, float* r);
__global__ void cudaChoiLeeFullFromPaper(float* xi, float* xq, float* sr, float* si, int N, float* L, float* L2, float* r);
__global__ void cudaChoiLee2And3(float* xi, float* xq, float* sr, float* si, int N, float* L, float* L2, float* r);
__global__ void cudaChoiLee2And3FromPaper(float* xi, float* xq, float* sr, float* si, int N, float* L, float* L2, float* r);


// BYU Simplified For Paper
__global__ void cudaBYUSimplified(float* xi, float* xq, float* sr, float* si, int N, int Lq, float *L);
__global__ void cudaNCPDI2(float* xi, float* xq, int N, int Lq, float *L);

//NCPDI-2 Modified
__global__ void calculateInnerSumBlocksNew(float* i, float* q, float* innerSums, int uLength, int innerSumsLength);
__global__ void calculateInnerSumBlocksNewI(float* i, float* q, float* innerSums, int uLength, int innerSumsLength);
__global__ void calculateInnerSumBlocksNewQ(float* i, float* q, float* innerSums, int uLength, int innerSumsLength);
__global__ void calculateOuterSumsNew(float* innerSums, float* L, int uLength);
__global__ void myAbs(cuComplex *complexOut, int maxThreads);
__global__ void PointToPointMultiply(cuComplex* v0, cuComplex* v1, int lastThread);
__global__ void floatToSamples(cuComplex *complexOut,float *i, float *q, int maxThreads);

//Find Maximums
__global__ void findPreambleMaximums(float* L, int* max_locations, int* max_locations_save, uint samplesPerPacket, int* endIndex_FirstWindow, int max_sample_idx, int numSections);
__global__ void findPreambleMaximums(cuComplex* L, int* max_locations, int* max_locations_save, uint samplesPerPacket, int* firstMaxMod, int max_sample_idx, int numSections);
__device__ int  findMaxSI(float*L, int startIdx, int endIdx);
__global__ void fixMaximumsLazy(int* max_locations, int* num_good_maximums, int* startOffset, int numInputSamples);
__global__ void cudaFirstMaxSearch(float *quickMaxSearch, int SAMPLES_PER_PACKET, int *endIndex_FirstWindow, int *myFirstMax, int *myFirstMaxActual);
__global__ void cudaFirstMaxSearch(cuComplex *quickMaxSearch, int SAMPLES_PER_PACKET, int *endIndex_FirstWindow, int *myFirstMax, int *myFirstMaxActual);
__global__ void cudaLongestChainSearch(int *max_locations, int MAX_PACKETS_PER_MEMORY_SECTION);
__global__ void cudaMaxAdjust(int *max_locations, int SAMPLES_PER_PACKET, int maxThreads);

//Signal Stripping
__global__ void stripSignalFloatToComplex(float *in_i, float *in_q, cuComplex *out, int *firstMax, int packetLength, int maxThreads);
__global__ void stripSignalComplexToComplex(cuComplex *in, cuComplex *out, int *firstMax, int packetLength, int maxThreads);
__global__ void cudaConj(cuComplex *array, int maxThreads);


__global__ void asmCorr(float* A, float* i, float* q, float* asmI, float* asmQ, int* max_locations, int numCorrs);
__global__ void cudaCL3(float* xi, float* xq, float* sr, float* si, int N, float* L);
__global__ void cl1a1b(float* xi, float* xq, float* sr, float* si, int N, float* L, float* L2, float* r);
__global__ void cl1a1bFull(float* xi, float* xq, float* sr, float* si, int N, float* L, float* L2, float* r, int Lp);
__global__ void cudaCL3Full(float* xi, float* xq, float* sr, float* si, int N, float* L, int Lp);


//Short correlator -- uses q(l)
__global__ void qCorrelator(float* xi, float* xq, float* L, int uLength);
__global__ void smallCorrelation(float* L, float* innerSums, int innerSumsLength);

//Frequency Offset Estimator
__global__ void signalStripper(cuComplex *in, cuComplex *out, int packetLength, int r1Start, int r1Length, int r1Conj, int maxThreads);
__global__ void tanAndLq(float* out, cuComplex *in, int maxThreads);
__global__ void w0Ave(float* w0, int MAX_PACKETS_PER_MEMORY_SECTION);
__global__ void derotate(cuComplex* data, float* W0, int batchLength, int maxThreads);
__global__ void derotateBatchSumW0(cuComplex* data, float* w0, int batchLength, int maxThreads);
__global__ void gpu_estimateFreqOffsetAndRotate(float* xi, float* xq, int *max_locations, int numPreamblePeaks, float* w0);
__global__ void rotateSamplesPacket(float* xi, float* xq, float* w0, int *max_locations, int numPacketsToProcess);


//Channel estimator
__global__ void estimateChannel(int *max_locations, int numPacketsToProcess, float* xi, float* xq, cuComplex* alpha, cuComplex* beta, cuComplex *chanEstMatrix, cuComplex *r_d, cuComplex *h_hat);

//Noise Variance Estimator
__global__ void subAndSquare(cuComplex *r2, cuComplex *intermediate, float *diffMag2, int maxThreads);
__global__ void sumAndScale(float *noiseVariance, float *diffMag2, int maxThreads);
__global__ void estimateNoiseVariance(cuComplex* r_d, cuComplex* x_mat, cuComplex* Xh_prod, cuComplex* h_hat, cuComplex* sigmaSquared, cuComplex* alpha, cuComplex* beta, int numPacketsToProcess);
__global__ void resolveNoiseVariance(cublasHandle_t handle, cuComplex* r_d, cuComplex* x_mat, cuComplex* Xh_prod, cuComplex* h_hat, cuComplex* sigmaSquared, cuComplex* alpha, cuComplex* beta, int numPacketsToProcess);

//Zero Forcing Equalizer
__device__ cuComplex cuMult(cuComplex& a, cuComplex& b);
__device__ cuComplex cuDiv(cuComplex& a, cuComplex& b);
__global__ void autocorr(cuComplex* h, int hLength, cuComplex* result, int corrLength, int saveOffset, float* shs, int mmseFlag);
__global__ void fill_hhh_matrices(cuComplex* h_corr, int corrLength, cuComplex* hhh, int N);
__global__ void fill_hhh_csr_matrices(cuComplex* h_corr, cuComplex* hhh_csr, int N1, int N2, int nnvA);
//__global__ void build_hh_un0_vector(cuComplex* h_hat, int h_hat_length, cuComplex* hh_un0, int un0_length, int N1, int maxThreads);
__global__ void build_hh_un0_vector_reworked(cuComplex* h_hat,cuComplex* h_un0,int N1,int N2,int L1,int L2,int maxThreads);

//MMSE Equalizer
__global__ void add_sigma_to_hhh_csr_matrices(cuComplex* hhh, int N, int numPackets, cuComplex* sigma, int N1, int N2, int nnzA);

//Apply Equalizer
__global__ void zeroPad(cuComplex *out, const cuComplex *in, int batches, int old_length, int new_length);
__global__ void pointMultiply(cuComplex *a, cuComplex *b, int size);
__global__ void pointMultiplyTriple(cuComplex *a_in, const cuComplex *b_in, const cuComplex *c_in, int size);
__global__ void scaleAndPruneFFT(cuComplex *out, const cuComplex *in, float scale, int out_length, int in_length, int frontJump, int maxThreads);
__global__ void fillUnfilteredSignal(float *real, float *imag, cuComplex *complex, int start, int end, int size);
__global__ void fillPaddedSignal(float *in_i, float *in_q, cuComplex *out, int startIndex, int endIndex, int old_length, int new_length);

//Apply Demodulators
__global__ void dmodZeroPad(cuComplex *out, cuComplex *in, int SAMPLES_PER_PACKET, int conv_length, int maxThreads);
__global__ void dmodPostPruneScaledDownsample(cuComplex *in, cuComplex *out, int front, int packet_width_pre_downsample, int unPruned, int downBy, float scale, int shift, int maxThreads);
__global__ void cudaDemodulator(const cuComplex *dfout1, float *ahat, int batchlength, int startpoint, int maxThreads);
__global__ void bitPrune(unsigned char *out, float *in, int frontPrune, int outputlength, int inputLength, int maxThreads);
__global__ void bit8Channels(unsigned char *in, unsigned char *out, int channel, int maxThreads);

//Freq Equalizer
__global__ void pointMultiplyQuadFDE1(cuComplex *Y_in, const cuComplex *H_in, const float *shs_in, const cuComplex *D_in, int size);
__global__ void pointMultiplyQuadFDE2(cuComplex *Y_in, const cuComplex *H_in, const float *shs_in, const cuComplex *D_in, const float *PSI_in, int size);
__global__ void PSIfill(float *array, int conv_length, int maxThreads);
__global__ void zeroPadFreq(cuComplex *out, const cuComplex *in, int batches, int old_length, int new_length, int front_jump);
__global__ void zeroPadShiftFDE(cuComplex *out, const cuComplex *in, int batches, int old_length, int new_length, int shift);
__global__ void rotateHalfPreambleBack(cuComplex *out, cuComplex *in, int Np, int maxThreads);
__global__ void cudaDemodulatorFDE(const cuComplex *dfout1, float *ahat, int batchlength, int startpoint, int maxThreads);
__global__ void shiftFDEblindlyForward(cuComplex *out, const cuComplex *in, int shift, int maxThreads);
__global__ void copyPreambleShiftOnEnd(cuComplex *array, int shift);

//CMA
__global__ void cudaCMAz(cuComplex *y1, cuComplex *e, int maxThreads);
__global__ void cudaCMAdelJ(cuComplex *delJ_in, cuComplex *e_in, cuComplex *r_in, const int SAMPLES_PER_PACKET, const int L1, const float mu, int maxThreads);
__global__ void cudaCMAflipLR(cuComplex *out, cuComplex *in, int batchLength, int maxThreads);
__global__ void zeroPadConj(cuComplex *out, const cuComplex *in, int batches, int old_length, int new_length);
__global__ void stripAndScale(cuComplex *delJ_in, cuComplex *efft_in, float mu, int forwardJump, int EQUALIZER_LENGTH, int CMA_FFT_LENGTH, int maxThreads);
__global__ void cudaUpdateCoefficients(cuComplex *c, cuComplex *delJ, int maxThreads);
__global__ void cudaConvGlobal(float *filtered, float *signal, float *filter, int maxThreads);

//BERT
__global__ void pointToPointConj(cuComplex *out, cuComplex *in, int max);
__global__ void rotateXcorrBits(cuComplex *out, cuComplex *in, int length_bits, int Nfft, int maxThreads);
__global__ void pullBitsXcorr(cuComplex *array, unsigned char *bits, int maxThreads);
__global__ void peakSearchXcorr(int *peak, int *peakIdx, int *xcorr_in, int maxIndex, int maxThreads);
__global__ void pullxCorrBits(int *xcorr, cuComplex *array, int Nfft, int processedBits, int maxThreads);

//POLYPHASE
__global__ void convertShortToFloat(float *x_out, unsigned short *x_in, int maxThreads);
__global__ void runHalfBandFilter(cuComplex *y_p, unsigned short *x, float *halfBandfiler, int maxThreads);
__global__ void runPolyPhaseFilter(cuComplex *z_p, cuComplex *y_p, float *newFilterBank, int maxThreads);
__global__ void ComplexSamplesToiORq(float *i, float *q, cuComplex *z_p, bool conj, int maxThreads);

#define INNER_SUM_OFFSET 12416
#define SAMPLES_IN_ASM 128
#define MAX_NUM_DELTAS 3103
#define GOOD_DELTA 12672
#define OVERSHOT_DELTA 12704
#define UNDERSHOT_DELTA 12640
#define SKIP_DELTA 25312
#define FALSE_DELTA 32
#define ERROR_THRESHOLD 2
#define MAX_NUM_PACKETS 3104
#define N1_ACAUSAL_SAMPLES 12
#define N2_CAUSAL_SAMPLES 25
#define CHAN_EST_LENGTH 38
#define CHAN_EST_MATRIX_NUM_ROWS 38
#define CHAN_EST_MATRIX_NUM_COLS 347
#define X_MAT_NUM_ROWS 347
#define X_MAT_NUM_COLS 38
#define R_D_LENGTH 347
#define CHAN_EST_AUTOCORR_LENGTH 75
#define LD_N 186


#endif	/* KERNELS_H */
