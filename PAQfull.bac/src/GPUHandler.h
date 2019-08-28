/*
 * GPUHandler.h
 *
 *  Created on: Aug 26, 2014
 *      Author: adm85
 */

#ifndef GPUHANDLER_H_
#define GPUHANDLER_H_

#include <vector>
#include <cusolverSp.h>
#include <cufft.h>
#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <sys/stat.h>
#include <stdio.h>
#include <string.h>
#include <sstream>
#include <ctime>
#include "Samples.h"
#include "FileReader.h"

#define SSTR( x ) dynamic_cast< ostringstream & >(( ostringstream() << dec << x ) ).str()

using namespace std;

namespace PAQ_SOQPSK {

class GPUHandler {
public:
	//Constants
	static const uint Lq_IN_SAMPLES = 32;
	static const uint NUM_INPUT_SAMPLES_DEFAULT = 39321600;
	static const uint PREAMBLE_LENGTH_IN_SAMPLES = 256;
	static const uint SAMPLES_PER_PACKET = 12672;
	static const uint TOTAL_SAMPLES_LENGTH = NUM_INPUT_SAMPLES_DEFAULT+SAMPLES_PER_PACKET-1;
	static const uint BITS_PER_PACKET = SAMPLES_PER_PACKET/2;
	static const uint PROCESS_SAMPLES_OFFSET = 12416;
	static const uint SAMPLES_IN_INPUT_BUFFER = 39334271; //39,321,600 new samples + 12,671 old samples
	static const uint MAX_PACKETS_PER_MEMORY_SECTION = 3104;
	static const uint PREAMBLE_DETECTOR_THRESHOLD = 400;
	static const uint SAMPLE_BUFFER_SIZE = 157287420;
	static const uint NUM_OLD_SAMPLES = SAMPLES_PER_PACKET - 1;
	static const uint OLD_SAMPLES_OFFSET_IN_BYTES = NUM_OLD_SAMPLES*sizeof(float);
	static const uint NUM_ASM_SAMPLES = 128;
	static const uint FULL_SAMPLE_BUFFER_SIZE_IN_BYTES = (NUM_INPUT_SAMPLES_DEFAULT * 4) + OLD_SAMPLES_OFFSET_IN_BYTES;
	static const uint N1_ACAUSAL_SAMPLES = 12;  //N1
	static const uint N2_CAUSAL_SAMPLES = 25;   //N2
	static const uint CHAN_SIZE = N1_ACAUSAL_SAMPLES + N2_CAUSAL_SAMPLES + 1;
	static const uint CHAN_EST_MATRIX_NUM_ROWS = 38;
	static const uint CHAN_EST_MATRIX_NUM_COLS = 347;
	static const uint CHAN_EST_MATRIX_NUM_ENTRIES = 13186;
	static const float RHO_REAL = 309;
	static const float RHO_IMAG = 0;
	static const uint X_MATRIX_ROWS = 347;
	static const uint X_MATRIX_COLS = 38;
	static const uint EQUALIZER_LENGTH_MULTIPLIER = 5;
	static const uint EQUALIZER_ACAUSAL_TAPS = N1_ACAUSAL_SAMPLES * EQUALIZER_LENGTH_MULTIPLIER; //L1
	static const uint EQUALIZER_CAUSAL_TAPS = N2_CAUSAL_SAMPLES * EQUALIZER_LENGTH_MULTIPLIER;   //L2
	static const uint EQUALIZER_LENGTH = EQUALIZER_ACAUSAL_TAPS + EQUALIZER_CAUSAL_TAPS + 1; 	 //L1 + L2 +1
	static const uint N1 = N1_ACAUSAL_SAMPLES;
	static const uint N2 = N2_CAUSAL_SAMPLES;
	static const uint L1 = EQUALIZER_ACAUSAL_TAPS;
	static const uint L2 = EQUALIZER_CAUSAL_TAPS;
	static const uint batchSize = MAX_PACKETS_PER_MEMORY_SECTION;
	static const uint m = EQUALIZER_LENGTH;
	static const uint nnzA = 12544;
	static const float maxTime = 1.9071;
	static const uint device_GPU0 = 0;
	static const uint device_GPU1 = 1;
	static const uint device_GPU2 = 2;
	static const uint stream_0 = 0;
	static const uint stream_1 = 1;
	static const uint conv_length = 16384;
	static const uint testingBatchNum = 0; //3101;
	static const uint demod_filter_length = 21;
	static const uint downsampled_by = 2;
	static const uint BITS_PER_DATA_PACKET = 6144;
	static const uint BITS_PER_FRONT_PACKET = PREAMBLE_LENGTH_IN_SAMPLES/downsampled_by+NUM_ASM_SAMPLES/downsampled_by;
	static const uint r1_length = PREAMBLE_LENGTH_IN_SAMPLES/8*5;
	static const uint r1_start = PREAMBLE_LENGTH_IN_SAMPLES/8*2;
	static const uint r1_end = PREAMBLE_LENGTH_IN_SAMPLES/8*7-1;
	static const uint r1_flag = 0;
	static const uint r1_conj_start = PREAMBLE_LENGTH_IN_SAMPLES/8;
	static const uint r1_conj_end = PREAMBLE_LENGTH_IN_SAMPLES/8*6-1;
	static const uint r1_conj_flag = 1;
	static const int dot_m = 1;
	static const int dot_n = 1;
	static const int dot_k = 160;
	static const int dot_lda = dot_k;
	static const int dot_ldb = dot_n;
	static const int dot_ldc = dot_n;
	static const int channelEst_m = 38;
	static const int channelEst_n = 1;
	static const int channelEst_k = 347;
	static const int channelEst_lda = channelEst_m;
	static const int channelEst_ldb = channelEst_k;
	static const int channelEst_ldc = channelEst_m;
	static const int Np = channelEst_m;
	static const int Nasm = channelEst_m;
	static const uint r2_length = 347;
	static const uint r2_flag = 0;
	static const uint r2_start = N2;
	static const uint noiseMultiply_X_length = X_MATRIX_ROWS * X_MATRIX_COLS;
	static const int noiseMultiply_m = 347;
	static const int noiseMultiply_n = 1;
	static const int noiseMultiply_k = 38;
	static const int noiseMultiply_lda = noiseMultiply_m;
	static const int noiseMultiply_ldb = noiseMultiply_k;
	static const int noiseMultiply_ldc = noiseMultiply_m;
	static const int noiseSum_m = 1;
	static const int noiseSum_n = 1;
	static const int noiseSum_k = 347;
	static const int noiseSum_lda = noiseSum_k;
	static const int noiseSum_ldb = noiseSum_k;
	static const int noiseSum_ldc = noiseSum_m;
	static const int CMA_FFT_LENGTH = 12800;//32768;
	static const int corrLength = CHAN_SIZE*2-1;
	static const int XCORR_NFFT = 33554432;
	static const int pn11Length = 2047;
	static const int MAX_NUM_PN11 = MAX_PACKETS_PER_MEMORY_SECTION*BITS_PER_DATA_PACKET/pn11Length+1;
	static const int WRITE_CHUNK_LENGTH = 70;
	int HalfBandFilterBytesPerSample;
	int HalfBandFilterLength;
	int HalfBandFilterSizeInBytes;
	int HalfBytesPerSample;
	int HalfFIFOlength;
	int HalfOldSamples;
	int HalfNeverUnder;
	int HalfPop_big;
	int HalfPop_little;
	int HalfPush1MbBatches_big;
	int HalfPush1MbBatches_little;
	int HalfPush_big;
	int HalfPush_little;
	int HalfPushMultiple;
	int HalfSizeInBytes;
	int HalfThreshold;
	int PolyBatches_big;
	int PolyBatches_little;
	int PolyBytesPerSample;
	int PolyFIFOlength;
	int PolyNeverUnder;
	int PolyPhaseFilterBytesPerSample;
	int PolyPhaseFilterLength;
	int PolyPhaseFilterSizeInBytes;
	int PolyPop_always;
	//	int PolyPush_big;
	//	int PolyPush_little;
	int PolyPushMultiple;
	int PolySizeInBytes;
	int PolyThreshold;
	float resampleRateFromHalfToPoly;
	float SanityCheck;
	int TotalFIFObufferInBytes;
	int TotalSetupInBytes;
	int DAQoldSamples;
	int DAQbufferLength;
	int DAQBytesPerSample;
	int DAQinputsPerHalfOutput;
	int DAQoneMb;
	int DAQsamplesPerByte;
	int DAQsamplesPerHalfSample;
	int DAQSizeInBytes;
	int DAQPush_big;
	int DAQPush_little;

	int PolyPush_little;
	int PolyPush_big;
	int UltraMbGrag_little;
	int UltraMbGrag_big;
	float resampleRate;
	int ultraviewSamplesToHalfLength;
	int halfbandSamplesToPolyLength;
	int FIFOsamplesFromPolyphaseLength;
	float myMaxi;
	float myMaxq;

	GPUHandler();
	GPUHandler(uint numInputSamples);
	virtual ~GPUHandler();

	void CreateBatchDirectory	();
	void writeGeneralFile 		(int PolyWriteIdx);
	void polyphaseFilters(unsigned short* DAQBufferShort, float* dev_x_float);
	void DAQsamplesShortToFloat(unsigned short* DAQBufferShort);
	void PushHalfSamples(int numSamplesToPush, int halfIdx);
	void PushPolySamples(int numSamplesToPush, int polyIdx);
	void PopHalfSamples(int numSamplesToPop, int halfIdx);
	void PushiANDqSamples();
	void PopPolySamples(int polyIdx);
	void DAQCopytoDevice(int numMb,unsigned short* ultraviewSamples_short);
	void RunHalfbandFilterWithDAQCopy(int numMb,unsigned short* DAQBufferShort);
	void RunPolyphaseFilters(int numMb,int PolyWriteIdx);
	void PullFromPolyFIFOandConvertFromComplexToRealandImag(bool conj);
	void ShiftPolyFIFO(int numSamplesInFIFObeforeShift);
	void preFindPreambles(float* iSamples, float* qSamples);
	void preambleDetector();
	void runningWriteChannelEst ();
	void estimateFreqOffsetAndRotate();
	void estimate_channel();
	void calculate_noise_variance();
	void calculate_equalizers();
	void CMA();
	void changeCMAmu(float in);
	void freq();
	void apply_equalizers_and_detection_filters();
	void apply_detection_filters();
	void  apply_demodulators();
	int  BERT(int processedBits);
	int postCPUrun(unsigned char* bits_8channels_host,bool lastRun,int PolyWriteIdx);
	void DeviceTesting();
	void TimingReset();
	void StartTiming();
	void StartCPUTiming();
	void StopCPUTiming();
	void StartTimingKernel();
	void StopTiming();
	float StopTimingMain(const int numRuns, long ultraTime);
	void StopTimingKernel(int threads);
	void NaNTesting();
	void writeFile_float_signal (float i[], float q[], int length, int numRuns);
	void writeFile_front (float i[], float q[], int numRuns);
	void writeFile_end (float i[], float q[], int numRuns);

	void writeMissedTimingBatchFiles	();
	void writePoly_i_little_front (int size, int numWrite);
	void writePoly_q_little_front (int size, int numWrite);
	void writePoly_i_little_back (int size, int numWrite);
	void writePoly_q_little_back (int size, int numWrite);
	void writeBatch_y_p 		(int length,cuComplex* local_array);
	void writePoly_y_p_little (int size, int numWrite);
	void writePoly_DAQ_float_little (int size, int numWrite);
	void writeBatch_z_p 		(int length,cuComplex* local_array);
	void writePoly_z_p_little_old (int size, int numWrite);
	void writePoly_z_p_little_new (int size, int numWrite);
	void writePoly_z_p_little_push (int size, int numWrite);
	void writeBatchFiles 		();
	void writeBatch_MMSEbits	(int length);
	void writeBatch_raw_i 		(int length);
	void writeBatch_raw_q 		(int length);
	void writeBatch_raw_i_last_times 		(int length);
	void writeBatch_raw_q_last_times 		(int length);
	void writeBatch_L 			(int length);
	void writeBatch_L_last_times(int length);
	void writeBatch_Max 		(int length);
	void writeBatch_Max_last_times(int length);
	void writeBatch_w0 			();
	void writeBatch_samples0	(int length);
	void writeBatch_samples0_pd	(int length);
	void writeBatch_samples1	(int length);
	void writeBatch_samples2	(int length);
	void writeBatch_channelEst	(int length);
	void writeBatch_ZF			(int length);
	void writeBatch_MMSE		(int length);
	void writeBatch_CMA			(int length);
	void writeBatch_downCMA		(int length);
	void writeBatch_downZF		(int length);
	void writeBatch_downMMSE	(int length);
	void writeBatch_downFDE1	(int length);
	void writeBatch_downFDE2	(int length);
	void writeBatch_ZFbits		(int length);
	void writeBatch_CMAbits		(int length);
	void writeBatch_FDE1bits	(int length);
	void writeBatch_FDE2bits	(int length);
	void writeBatch_DAQsamples	(int length);
	void writeBatch_DAQsamplesLast34 (int length);
	void writeBatch_DAQsamples_last_times	(int length);
	void writeBatch_DAQsamples_host	(unsigned short* host_array,int length);
	void writeBatch_HalfSamples	(int length);
	void writeBatch_HalfSamplesLast19 (int length);
	void writeBatch_PolySamples	(int length);
	void writeBatch_FIFOSamples	(int length);
	void writeBatch_PAQsamples	(int length);
	void writeBatch_PAQsamples_last_times	(int length);
	void writeBatch_PAQsamples_two_times	(int length);
	void writeBatch_iFromPoly	(int length);
	void writeBatch_qFromPoly	(int length);



private:

	float myTime,wcet,bcet;
	float myUltraTime,Ultrawcet,Ultrabcet;
	int bestThread;
	cudaEvent_t start, stop;
	//----------------------------------------------
	//	STREAMS
	//----------------------------------------------
	cudaStream_t stream_GPU0_array[2];
	cudaStream_t stream_GPU1_array[2];
	cudaStream_t stream_GPU2_array[2];
	cudaStream_t stream[3];
	cudaStream_t stream_GPU1[3];

	void initialize_streams();
	void free_streams();

	//----------------------------------------------
	//	PREAMBLE DETECTOR VARIABLES/METHODS
	//----------------------------------------------
	void initialize_polyphaseFilters();
	void free_polyphaseFilters();
	float* dev_halfBandFilter;
	float* dev_newFilterBank;
	float* x;
	unsigned short* dev_x_short;
	unsigned short* dev_ultraviewSamplesToHalf;
	unsigned short* dev_ultraviewSamplesToHalf_last_times;
	unsigned short* dev_ultraviewSamplesToHalf_startIdx;
	unsigned short* dev_ultraviewLastIteration34;

	cuComplex* dev_halfbandSamplesToPoly;
	cuComplex* dev_halfbandSamplesToPoly_startIdx;
	cuComplex* dev_halfbandSamplesToPolyLastIteration19;

	cuComplex* dev_FIFOsamplesFromPolyphase;

	cuComplex* dev_PAQcomplexSamplesFromPolyFIFO;
	cuComplex* dev_PAQcomplexSamplesFromPolyFIFO_last_times;
	cuComplex* dev_PAQcomplexSamplesFromPolyFIFO_two_times;
	cuComplex* dev_PAQcomplexSamplesFromPolyFIFOLastIteration12671;
	cuComplex* dev_PAQcomplexSamplesFromPolyFIFO_startIdx;

	float* dev_x_float_old;
	float* dev_x_float_new;
	cuComplex* dev_y_p_old;
	cuComplex* dev_y_p_new;
	cuComplex* dev_y_p_push;
	cuComplex* dev_y_p_shiftFrom;
	cuComplex* dev_z_p;
	cuComplex* dev_z_p_push;
	cuComplex* dev_z_p_shiftFrom;
	float* dev_iSamples_GPU1;
	float* dev_qSamples_GPU1;
	float* dev_iSamples_push_GPU1;
	float* dev_qSamples_push_GPU1;
	float* dev_old_iSamples_GPU1;
	float* dev_old_qSamples_GPU1;
	double L_polyphase;


	//----------------------------------------------
	//	PREAMBLE DETECTOR VARIABLES/METHODS
	//----------------------------------------------
	int Nfft_pd;
	cuComplex* dev_Samples_GPU0_pd;
	cuComplex* dev_Matched_GPU0;
	cufftHandle plan_pd;

	float* L_pd;
	int*   max_locations;
	float* old_iSamples_pd;
	float* old_qSamples_pd;
	int    num_good_max_pd;

	float* dev_iSamples_pd;
	float* dev_qSamples_pd;
	float* dev_iSamples_pd_last_times;
	float* dev_qSamples_pd_last_times;

	float* dev_old_iSamples_pd;
	float* dev_old_qSamples_pd;

	cuComplex* dev_Samples_GPU0;
	cuComplex* dev_Samples_GPU1;
	cuComplex* dev_Samples_GPU2;
	float* dev_blockSums_pd;
	float* dev_L_pd;
	float* dev_L_pd_last_times;
	int*   dev_max_locations;
	int*   dev_max_locations_last_times;
	int*   dev_max_locations_save;
	int*   dev_num_good_maximums;
	int*   dev_endIndex_FirstWindow;
	int*   dev_myFirstMaxMod;
	int*   dev_myFirstMaxActual;

	int    firstMax;
	int	   lastMax;
	int    shiftingMarch;

	//	uint numInputSamples_pd;
	uint samplesSize_pd;
	uint uLength_pd;
	uint uSize_pd;
	uint oldSamplesLength_pd;
	uint oldSamplesSize_pd;
	uint totalSamplesLength;
	uint totalSamplesSize_pd;
	uint sumsLength_pd;
	uint sumsSize_pd;
	uint threadsPerBlock_Inner_pd;
	uint numBlocks_Inner_pd;
	uint threadsPerBlock_Outer_pd;
	uint numBlocks_Outer_pd;
	uint threadsPerBlock_max_finder_pd;
	uint numBlocks_max_finder_pd;
	uint max_locations_size;
	int  last_good_maximum_pd;
	int  myFirstMaxMod;
	int  myFirstMaxActual;

	int maxThreads_calculateInnerSumBlocksNew;
	int numThreads_calculateInnerSumBlocksNew;
	int numBlocks_calculateInnerSumBlocksNew;

	int maxThreads_calculateOuterSumsNew;
	int numThreads_calculateOuterSumsNew;
	int numBlocks_calculateOuterSumsNew;

	int maxThreads_findPreambleMaximums;
	int numThreads_findPreambleMaximums;
	int numBlocks_findPreambleMaximums;

	int maxThreads_cudaMaxAdjust;
	int numThreads_cudaMaxAdjust;
	int numBlocks_cudaMaxAdjust;

	int numBlocks_stripSignalFloatToComplex;
	int numThreads_stripSignalFloatToComplex;
	int maxThreads_stripSignalFloatToComplex;

	void initialize_pd_variables();
	void free_pd_variables();

	//----------------------------------------------
	//	FREQUENCY OFFSET ESTIMATOR VARIABLES/METHODS
	//----------------------------------------------
	int shiftFDE;
	cuComplex* dev_r1;
	cuComplex* dev_r1_conj;
	cuComplex* dev_complex_w0;
	float* dev_w0;
	uint leftover_packet_idx;
	bool have_leftover_packet;
	uint num_packets_to_process;
	//Dot
	cublasHandle_t dot_handle;
	cuComplex r1_alpha;
	cuComplex r1_beta;
	cuComplex** dev_r1_conj_list;
	cuComplex** dev_r1_list;
	cuComplex** dev_complex_w0_list;

	int numBlocks_signalStripperFreq;
	int numThreads_signalStripperFreq;
	int maxThreads_signalStripperFreq;

	int numBlocks_tanAndLq;
	int numThreads_tanAndLq;
	int maxThreads_tanAndLq;

	int numBlocks_derotateBatchSumW0;
	int numThreads_derotateBatchSumW0;
	int maxThreads_derotateBatchSumW0;



	void initialize_foe_variables();
	void free_foe_variables();

	//----------------------------------------------
	//	CHANNEL ESTIMATOR
	//----------------------------------------------
	cuComplex* dev_r2;
	cuComplex* dev_channelEst_piX;
	cuComplex* dev_channelEst;
	cublasHandle_t channelEst_handle;
	cuComplex channelEst_alpha;
	cuComplex channelEst_beta;
	cuComplex** dev_channelEst_piX_list;
	cuComplex** dev_r2_list;
	cuComplex** dev_channelEst_list;

	Samples channelEstMatrixSamples;
	cuComplex* chanEstMatrix;
	cuComplex* dev_chanEstMatrix;
	cuComplex* dev_rd_vec;
	cuComplex* dev_gemm_alpha;
	cuComplex* dev_gemm_beta;
	cuComplex* dev_h_hat_zf_GPU1;
	cufftHandle fftPlan_signal_GPU0;

	int numBlocks_signalStripperChannel;
	int numThreads_signalStripperChannel;
	int maxThreads_signalStripperChannel;

	void initialize_channel_estimator_variables();
	void free_channel_estimator_variables();

	//----------------------------------------------
	//	NOISE VARIANCE ESTIMATOR
	//----------------------------------------------
	cuComplex* dev_noiseMultiply_X;
	float* dev_noiseVariance_GPU0;
	cuComplex* dev_noiseMultiplyIntermediate;
	cuComplex** dev_noiseMultiply_X_list;
	cuComplex** dev_noiseIntermediate_list;
	float** dev_noiseVariance_GPU0_list;
	cublasHandle_t noiseMultiplyVariance_handle;
	cuComplex* dev_ones;
	cuComplex** dev_ones_list;
	cuComplex noiseMultiplyVariance_alpha;
	cuComplex noiseMultiplyVariance_beta;
	cublasHandle_t noiseSumVariance_handle;
	cuComplex noiseSumVariance_alpha;
	cuComplex noiseSumVariance_beta;
	Samples xMatSamples;
	float* dev_diffMag2;

	int numBlocks_subAndSquare;
	int numThreads_subAndSquare;
	int maxThreads_subAndSquare;

	int numBlocks_sumAndScale;
	int numThreads_sumAndScale;
	int maxThreads_sumAndScale;

	void initialize_nv_estimator_variables();
	void free_nv_estimator_variables();

	//----------------------------------------------
	//	MMSE and ZF EQUALIZERS
	//----------------------------------------------

	//	ZERO-FORCING EQUALIZER ON GPU1 SWITCHED
	cuComplex* dev_chanEstCorr_zf_GPU1; 	//Autocorrelation of the channel estimate
	float* dev_shs_GPU1;
	cuComplex* dev_hhh_csr_zf_GPU1;
	cuComplex* dev_hh_un0_zf_GPU1; 			//H*u_n0 (the right hand side of the equation)
	cuComplex* dev_ZF_equalizers_GPU1;			//Equalizer filter coefficients
	cuComplex* dev_ZF_equalizers_GPU2;			//Equalizer filter coefficients
	cusolverSpHandle_t cusolver_handle_GPU1;	//cuSolver handle
	cusparseMatDescr_t descrA_zf_GPU1;		//description of matrix for cuSolver
	csrqrInfo_t info_zf_GPU1;				//compressed row storage info
	size_t size_qr_zf_GPU1;					//QR factoring size
	size_t size_internal_zf_GPU1;			//memory cuSolver needs
	void *dev_buffer_qr_zf_GPU1; 				//working space for numerical factorization pointer
	int  *dev_csrRowPtrhhh_zf_GPU1;			//device new row pointer for csr format
	int  *dev_csrColIdxhhh_zf_GPU1;			//device col pointer for csr format

	//	CMA EQUALIZER ON GPU0 SWITCHED
	//	cuComplex* dev_rd_vec_GPU0;
	cuComplex* dev_noise_piX_GPU0;
	cuComplex* dev_Xh_prod_GPU0;
	cuComplex* dev_h_hat_mmse_GPU0;
	cuComplex* dev_gemm_alpha_GPU0;
	cuComplex* dev_gemm_beta_GPU0;
	cuComplex* dev_chanEstCorr_mmse_GPU0;
	cuComplex* dev_hhh_csr_mmse_GPU0;
	cuComplex* dev_hh_un0_mmse_GPU0;
	cusolverSpHandle_t cusolver_handle_GPU0;
	cusparseMatDescr_t descrA_mmse_GPU0;
	csrqrInfo_t info_mmse_GPU0;
	int* dev_csrColIdxhhh_mmse_GPU0;
	int* dev_csrRowPtrhhh_mmse_GPU0;
	size_t size_qr_mmse_GPU0;
	size_t size_internal_mmse_GPU0;
	void* dev_buffer_qr_mmse_GPU0;
	cuComplex* dev_MMSE_CMA_equalizers_GPU0;


	int numBlocks_autocorr;
	int numThreads_autocorr;

	int numThreads_fill_hhh_csr_matrices;
	int numBlocks_fill_hhh_csr_matrices;

	int numBlocks_build_hh_un0_vector_reworked;
	int numThreads_build_hh_un0_vector_reworked;
	int maxThreads_build_hh_un0_vector_reworked;

	void initialize_equalizers_variables();
	void free_equalizers_variables();

	//----------------------------------------------
	//	MMSE EQUALIZER on GPU2
	//----------------------------------------------
	cuComplex* dev_MMSEequalizers_GPU2;
	cufftComplex *dev_equalizers_padded_GPU2;
	cufftComplex *dev_filter_preDetection_GPU2;
	cufftHandle fftPlan_detection_GPU2;
	cufftHandle fftPlan_apply_GPU2;
	cufftComplex *dev_Samples_padded_GPU2;
	cufftComplex *dev_detected_downsampled_GPU2;
	float *dev_MMSE_ahat_GPU2;
	unsigned char *dev_MMSE_bits_GPU2;
	unsigned char *dev_MMSE_bits_GPU0;

	//----------------------------------------------
	//	CMA EQUALIZER on GPU0
	//----------------------------------------------
	cufftComplex *dev_z_GPU0;
	cufftComplex *dev_y_GPU0;
	cufftComplex *dev_delJ_GPU0;
	cufftComplex *dev_z_flipped_GPU0;
	cufftComplex *dev_x_padded_GPU0;
	cufftComplex *dev_z_flipped_padded_GPU0;
	cufftComplex *dev_delJ_fft_GPU0;
	float CMAmu;

	cufftHandle fftPlan_CMA_GPU0;
	unsigned char *dev_CMA_bits_GPU0;
	void initialize_CMA_variables();
	void free_CMA_variables();

	int maxThreads_zeroPad_CMA_equalizers;
	int numThreads_zeroPad_CMA_equalizers;
	int numBlocks_zeroPad_CMA_equalizers;

	int maxThreads_zeroPad_CMA_samples;
	int numThreads_zeroPad_CMA_samples;
	int numBlocks_zeroPad_CMA_samples;

	int maxThreads_pointMultiply_CMA;
	int numThreads_pointMultiply_CMA;
	int numBlocks_pointMultiply_CMA;

	int maxThreads_scaleAndPruneFFT_CMA;
	int numThreads_scaleAndPruneFFT_CMA;
	int numBlocks_scaleAndPruneFFT_CMA;

	int maxThreads_cudaCMAz_CMA;
	int numThreads_cudaCMAz_CMA;
	int numBlocks_cudaCMAz_CMA;

	int maxThreads_cudaCMAdelJ_CMA;
	int numThreads_cudaCMAdelJ_CMA;
	int numBlocks_cudaCMAdelJ_CMA;

	int maxThreads_cudaCMAflipLR_CMA;
	int numThreads_cudaCMAflipLR_CMA;
	int numBlocks_cudaCMAflipLR_CMA;

	int maxThreads_zeroPad_CMA_z_flipped;
	int numThreads_zeroPad_CMA_z_flipped;
	int numBlocks_zeroPad_CMA_z_flipped;

	int maxThreads_zeroPadConj_CMA;
	int numThreads_zeroPadConj_CMA;
	int numBlocks_zeroPadConj_CMA;

	int maxThreads_pointMultiply_CMA_z_flipped;
	int numThreads_pointMultiply_CMA_z_flipped;
	int numBlocks_pointMultiply_CMA_z_flipped;

	int maxThreads_stripAndScale_CMA;
	int numThreads_stripAndScale_CMA;
	int numBlocks_stripAndScale_CMA;

	int maxThreads_cudaUpdateCoefficients_CMA;
	int numThreads_cudaUpdateCoefficients_CMA;
	int numBlocks_cudaUpdateCoefficients_CMA;



	//----------------------------------------------
	//	FREQ EQUALIZER on GPU2
	//----------------------------------------------
	cufftComplex 	* dev_h_hat_freq_GPU2;
	cufftComplex 	* dev_FDE_Y_padded_GPU2;
	cufftComplex 	* dev_FDE_Y_padded_GPU2_blindShift;
	cufftComplex 	* dev_FDE2_Y_padded_GPU1;
	float 			* dev_FDE_PSI_GPU2;
	cufftComplex 	* dev_FDE_H_padded_GPU2;
	cufftComplex 	* dev_FDE2_H_padded_GPU1;
	float			* dev_shs_GPU2;
	cufftComplex 	* dev_FDE_detected_downsampled_GPU2;
	cufftComplex 	* dev_FDE2_detected_downsampled_GPU1;
	cufftComplex 	* dev_FDE2_detected_downsampled_GPU2;
	cufftComplex 	* dev_FDE1_detected_downsampled_rotated_GPU2;
	cufftComplex 	* dev_FDE2_detected_downsampled_rotated_GPU2;
	float 		 	* dev_FDE1_ahat_GPU2;
	float 		 	* dev_FDE2_ahat_GPU2;
	unsigned char 	* dev_FDE1_bits_GPU2;
	unsigned char 	* dev_FDE2_bits_GPU2;
	unsigned char 	* dev_FDE1_bits_GPU0;
	unsigned char 	* dev_FDE2_bits_GPU0;
	void initialize_freq_variables();
	void free_freq_variables();

	//----------------------------------------------
	//	APPLY FILTERS
	//----------------------------------------------
	cufftComplex* filter_fft;
	cublasHandle_t cublas_handle_apply_GPU0;
	cublasHandle_t cublas_handle_apply_GPU1;
	cufftHandle fftPlan_apply_GPU0;
	cufftHandle fftPlan_apply_GPU1;
	cufftComplex *dev_Samples_padded_GPU0;
	cufftComplex *dev_Samples_padded_GPU1;
	cufftComplex *dev_equalizers_padded_GPU0;
	cufftComplex *dev_equalizers_padded_GPU1;
	cuComplex fftscale;
	int startIndex;
	int endIndex;
	cufftComplex *dev_signal_test;

	int numBlocks_zeroPadEQUALIZER;
	int numThreads_zeroPadEQUALIZER;
	int maxThreads_zeroPadEQUALIZER;

	int numBlocks_zeroPadPACKET;
	int numThreads_zeroPadPACKET;
	int maxThreads_zeroPadPACKET;

	int numBlocks_pointMultiplyTriple;
	int numThreads_pointMultiplyTriple;
	int maxThreads_pointMultiplyTriple;

	int numBlocks_pointMultiplyQuad;
	int numThreads_pointMultiplyQuad;
	int maxThreads_pointMultiplyQuad;

	int numBlocks_dmodPostPruneScaledDownsample;
	int numThreads_dmodPostPruneScaledDownsample;
	int maxThreads_dmodPostPruneScaledDownsample;

	void initialize_apply_equalizers_and_detection_filters();
	void free_apply_equalizers_and_detection_filters();

	//----------------------------------------------
	//	APPLY DETECTION FILTERS
	//----------------------------------------------
	cufftComplex *dev_signal_equalized_GPU0;
	cufftComplex *dev_signal_preDetection_GPU0;
	cufftComplex *dev_filter_preDetection_GPU0;
	cufftHandle fftPlan_detection_GPU0;
	cufftComplex *dev_detected_downsampled_GPU0;

	cufftComplex *dev_signal_equalized_GPU1;
	cufftComplex *dev_signal_preDetection_GPU1;
	cufftComplex *dev_filter_preDetection_GPU1;
	cufftHandle fftPlan_detection_GPU1;
	cufftComplex *dev_detected_downsampled_GPU1;


	int numBlocks_cudaDemodulator;
	int numThreads_cudaDemodulator;
	int maxThreads_cudaDemodulator;

	int numBlocks_bitPruneDemod;
	int numThreads_bitPruneDemod;
	int maxThreads_bitPruneDemod;

	int numBlocks_bit8Channels;
	int numThreads_bit8Channels;
	int maxThreads_bit8Channels;

	void initialize_detection_filters_variables();
	void free_detection_filters_variables();

	//----------------------------------------------
	//	APPLY DEMODULATORS
	//----------------------------------------------
	float *dev_CMA_ahat_GPU0;

	float *dev_ZF_ahat_GPU1;
	unsigned char *dev_ZF_bits_GPU1;
	unsigned char *dev_ZF_bits_GPU0;

	unsigned char *dev_all8Channels_bits;

	void initialize_demodulators_variables();
	void free_demodulators_variables();


	//----------------------------------------------
	//	BERT
	//----------------------------------------------
	cufftHandle fftPlan_BERT_GPU0;
	cuComplex* 	dev_PN11A_GPU0;
	cuComplex* 	dev_BERT_bits_GPU0;
	int* 		dev_BERT_xCorrelatedBits_GPU0;
	int* 		dev_BERT_peaks_GPU0;
	int* 		dev_BERT_peaksIdx_GPU0;
	unsigned char *dev_BERT_bits_pull_pointer_GPU0;

	void initialize_BERT_variables();
	void free_BERT_variables();

	//----------------------------------------------
	//	GPU MONITORING
	//----------------------------------------------
	void checkGPUStats();


	//----------------------------------------------
	//	FDE TESTING WRITING FUNCTIONS
	//----------------------------------------------
	string 	FDEtestingPath;
	void writeFDEtestingFiles 		();
	void CreateFDEtestingDirectory	();
	void writeFDEtestingGeneralFile	();
	void writeFDEtesting_samples2	(int length);
	void writeFDEtesting_FDE1bits	(int length);

	//----------------------------------------------
	//	BATCH WRITING FUNCTIONS
	//----------------------------------------------
	int 	ErroredBatches;
	int 	MissedTimingBatches;
	string 	dayPath;
	string 	runPath;
	string 	batchPath;
	int 	bitErrorCount;
	int* 	bitErrorIdx;
	int* 	numBitErrorAtIdx;
	int	 	numPeaks;
	void CreateRunDirectory		();
	void CreateRunDirectorySync ();
	void CreateMissedTimingBatchDirectory	();
	//	void writeBatchFiles 		();






	//----------------------------------------------
	//	BETTER WRITING FUNCTIONS
	//----------------------------------------------
	void writeBERT_bits 		(int processedBits);
	void writeBERT_signed_bits 	(int length);
	void writeBERT_BITS 		(int length);
	void writeBERT_conj_MULTI 	(int length);
	void writeBERT_multi 		(int length);
	void writeBERT_correlated 	(int length);
	void writeBERT_peaks	 	(int length);
	void writeBERT_peaksIdx	 	(int length);


	//----------------------------------------------
	//	WRITING FUNCTIONS
	//----------------------------------------------
	void writeFreq_padded_h ();
	void writeFreq_padded_y ();
	void writeFreq_padded_H ();
	void writeFreq_padded_Y ();
	void writeFreq_padded_Y2 ();
	void writeFreq_padded_D ();
	void writeFreq_Quad ();
	void writeFreq_Quad2 ();
	void writeFreq_quad ();
	void writeFreq_quad2 ();
	void writeFreq_prune_scale_downsample ();
	void writeFreq_prune_scale_downsample2 ();
	void writeFreq_prune_scale();
	void writeFreq_prune_scale2();
	void writeFreq_ahat ();
	void writeFreq_ahat2 ();
	void writeFreq_bits ();
	void writeFreq_bits2 ();
	void writeFreq_shs ();
	void writeFreq_Samples ();
	void writeFreq_channel ();
	void writeCMA_equalizer ();
	void writeCMA_delJ ();
	void writeCMA_multi ();
	void writeCMA_ifft ();
	void writeCMA_r_fft ();
	void writeCMA_e_fft ();
	void writeCMA_e_flipped ();
	void writeCMA_Samples ();
	void writeCMA_y ();
	void writeCMA_e ();
	void writeUnFiltered ();
	void writeUnFilteredPadded ();
	void writeEqualizers ();
	void runningWriteEqualizers ();
	void writeDownSampled ();
	void writeDemodulatedBits ();
	void writeStrippedBits ();
	void writeChannelBits ();
	void writeSHS ();
	void writeHhat ();
	void writeSamples ();
	void writeZeroPaddedFilter ();
	void writeL ();
	void writeMaxLocations ();
	void writeMaxLocationslazy ();
	void writePoly_z ();
	void writePoly_z_p ();
	void writePoly_y ();
	void writePoly_y_p ();
	void writeStrippedSignal ();
	void writeR1 ();
	void writeFreqDot ();
	void writeR2 ();
	void writeChannelEst ();
	void writenoiseMultiplyIntermediate ();
	void writeNoiseSubAndSquare ();
	void writeNoise_myX ();
	void writeNoiseVariance ();
	void writeEqualizerPadded ();
	void writeSignalPadded ();
	void writeEqualizerFFT ();
	void writeSignalFFT ();
	void writeMultiply ();
	void writeIFFT ();
	void writeEqualizedSignal ();
	void writeDmodPadded ();
	void writePreScaled ();
	void writeAutoCorr ();
	void writeW0 ();
	void writeDerotated ();
	void readFile_conv ();


};

} /* namespace PAQ_SOQPSK */
#endif /* GPUHANDLER_H_ */
