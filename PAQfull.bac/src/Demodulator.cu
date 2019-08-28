/*
 * Demodulator.cu
 *
 *  Created on: May 13, 2013
 *      Author: adm85
 */

#include <vector>
#include <math.h>
#include <iostream>
#include <fstream>
#include "Demodulator.h"
#include "Samples.h"
#include "Kernels.h"

using namespace std;

namespace PAQ_SOQPSK {

	Demodulator::Demodulator() {
		initializeStatesAndSignals();
	}

	Demodulator::~Demodulator() {}

	vector<int>& Demodulator::demodulate(Samples& inputSamples){
		//----------------------------------------------------------------------
		//          Debug Code
		//----------------------------------------------------------------------
		//Load the CSV file
		//FileReader fileReader;
		//Samples ccwFarrowOutSamples = fileReader.loadCSVFile("/fslhome/adm85/compute/paq/soqpsk_demod/src/matlabFarrowOutput.csv");

		vector<float> ISamples;
		vector<float> QSamples;
		//vector<float> PEDError;
		//vector<float> LoopFilterError;

		//ofstream errorFile = new ofstream("/fslhome/adm85/compute/paq/soqpsk_demod/output/errorTrack.csv");
		ofstream errorFile;
		errorFile.open("/home/eswindle/Documents/Output/errorTrack.csv");
		if(!errorFile.is_open()) {
			cout << "Could not open error file." << endl;
			throw exception();
		}

		ofstream stateFile;
		stateFile.open("/home/eswindle/Documents/Output/hostState.csv");
		if(!stateFile.is_open()) {
			cout << "Could not open state file." << endl;
			throw exception();
		}

		ofstream bitDecisionFile;
		bitDecisionFile.open("/home/eswindle/Documents/Output/hostBitDecisions.csv");
		if(!bitDecisionFile.is_open()) {
			cout << "Could not open bit decision file." << endl;
			throw exception();
		}

		ofstream hostMuFile;
		hostMuFile.open("/home/eswindle/Documents/Output/hostMu.csv");
		if(!hostMuFile.is_open()) {
			cout << "Could not open hostMu file." << endl;
			throw exception();
		}

		ofstream hostStrobeFile;
		hostStrobeFile.open("/home/eswindle/Documents/Output/hostStrobe.csv");
		if(!hostStrobeFile.is_open()) {
			cout << "Could not open hostStrobe file." << endl;
			throw exception();
		}

		ofstream hostIQFile;
		hostIQFile.open("/home/eswindle/Documents/Output/hostIQ.csv");
		if(!hostIQFile.is_open()) {
			cout << "Could not open hostStrobe file." << endl;
			throw exception();
		}



		//Return array
		vector<int>* bitIndices = new vector<int>;

		//Sample values
		float I, Q;
		float I_Decision, Q_Decision;
		//Farrow interpolator values
		float h0, h1, h2, h3;
		//Bit decision
		int bitDecision;

		//cout << "About to enter loop..." << endl;
		for(int sampleIndex = 0; sampleIndex < inputSamples.getSize(); sampleIndex++) {
			//cout << "Sample Index: " << sampleIndex << endl;
			//----------------------------------------------------------------------
			//          Output Equations
			//----------------------------------------------------------------------
			//Get samples
			I = inputSamples.getI().at(sampleIndex);
			Q = inputSamples.getQ().at(sampleIndex);
			//cout << "Got Samples" << endl;
			//Derotate and add to queues
			xr = (cos(dds) * I) + (sin(dds) * Q);
			yr = (cos(dds) * Q) - (sin(dds) * I);
			for(int i=(SAMPLE_BUFFER_SIZE-1); i>0; i--) {
				I_Buffer.at(i) = I_Buffer.at(i-1);
				Q_Buffer.at(i) = Q_Buffer.at(i-1);
			}
			I_Buffer.at(0) = xr;
			Q_Buffer.at(0) = yr;

			//cout << "Derotated" << endl;
			//Grab mu
			mu = M;
			hostMuFile << mu << "\n";
			hostStrobeFile << strobe << "\n";
			stateFile << state << "\n";

			//Farrow Interpolator
			h0 = 0.5 * mu * (mu-1);
			h1 = -0.5 * mu * (mu-3);
			h2 = (-0.5 * mu * (mu+1))+1;
			h3 = h0;
			I_Prime = (h0*xr) + (h1*FI1) + (h2*FI2) + (h3*FI3);
			Q_Prime = (h0*yr) + (h1*FQ1) + (h2*FQ2) + (h3*FQ3);

			//Grab final I and Q values
			if(strobe) {
				Q_Decision = Q_Prime;
				switch(state) {
					case 0: //This should never happen
						cout << "Error in bit decision block. State should never be 0." << endl;
						throw exception();
					case 1:
					case 2:
					case 3:
					case 5:
						I_Decision = B1;
						break;
					case 4:
						I_Decision = I_Buffer.at(TIME_N_MINUS_THREE);
						break;
					case 6:
						I_Decision = I_Buffer.at(TIME_N_MINUS_TWO);
						break;
					case 7: //This should never happen
						cout << "Error in bit decision block. State should never be 7." << endl;
						throw exception();
				}
				//Save to file
				hostIQFile << I_Decision << "," << Q_Decision << "\n";

				//Bit decision
				bitDecision = getSampleIndex(I_Decision, Q_Decision);
				//cout << "SampleIndex: " << sampleIndex << "   Bit Decision: " << bitDecision << endl;
				bitDecisionFile << bitDecision << "\n";
				bitIndices->push_back(bitDecision);
			}
			//cout << "I and Q complete" << endl;
			//----------------------------------------------------------------------
			//          Internal Signal Update Equations
			//----------------------------------------------------------------------
			//Phase Loop
			ep = calcPhaseError(I_Prime, I_Buffer.at(TIME_N_MINUS_TWO), I_Buffer.at(TIME_N_MINUS_THREE),
								Q_Prime, Q_Buffer.at(TIME_N_MINUS_TWO), Q_Buffer.at(TIME_N_MINUS_THREE),
								state);
			vp = (K1p*ep) + (K2p*ep) + VIp;

			//Timing Loop
			et = calcTimingError(I_Prime, I_Buffer.at(TIME_N_MINUS_TWO), I_Buffer.at(TIME_N_MINUS_THREE),
								 Q_Prime, state);
			vt = (K1t*et) + (K2t*et) + VIt;
			//cout << "Internal signals update complete" << endl;

			errorFile << vp << "," << vt << "\n";
			//----------------------------------------------------------------------
			//          State Update Equations
			//----------------------------------------------------------------------
			//Farrow Interpolator states
			FI3 = FI2; FI2 = FI1; FI1 = xr;
			FQ3 = FQ2; FQ2 = FQ1; FQ1 = yr;

			//Bit Decision block
			B1 = I_Prime;

			//PED Loop
			ped1 = I_Prime;
			ped2 = Q_Prime;
			VIp = VIp + K2p*ep;
			dds = dds + vp;

			//TED Loop
			ted3 = ted2;
			ted2 = calcTED2(ted1, I_Buffer.at(TIME_N_MINUS_TWO), I_Buffer.at(TIME_N_MINUS_THREE), state);
			ted1 = I_Prime;
			ted5 = ted4;
			ted4 = Q_Prime;

			VIt = VIt + K2t*et;
			OLD_NCO = NCO;

			NCO = fmod(NCO, 1) - vt - 0.5;
			if(NCO < 0) {
				strobe = true;
				M = 2*fmod(OLD_NCO, 1);
				NCO += 1; //Equivalent to fmod(NCO, 1);
			} else {
				strobe = false;
			}

			//State machine
			state = (4*strobe) + 2*s1 + s2;
			s2 = s1;
			s1 = strobe;

			//cout << "State updates complete" << endl;
		}

		//Close the debug files
		errorFile.close();
		stateFile.close();
		bitDecisionFile.close();
		hostMuFile.close();
		hostStrobeFile.close();
		hostIQFile.close();

		return *bitIndices;
	}

	vector<int>& Demodulator::demodulateCuda(Samples& inputSamples)
	{
		// Initialize variables
		float* i_samples, *q_samples, *consts;
		int* bitdec;
		int sample_size = inputSamples.getSize();
		int* output_bits = new int[sample_size];
		int floatsize = sample_size * sizeof(float);
		int intsize = sample_size * sizeof(int);
		vector<float> pedFilterConstants = calculateLoopFilterConstants(.01, 1, 18.33, 1, 2);
		vector<float> tedFilterConstants = calculateLoopFilterConstants(.005, 1, 12.35, -1, 2);
		float constants[4];
		constants[0] = pedFilterConstants[0];
		constants[1] = pedFilterConstants[1];
		constants[2] = tedFilterConstants[0];
		constants[3] = tedFilterConstants[1];

		// Allocate memory on GPU
		cudaMalloc(&i_samples, floatsize);
		cudaMalloc(&q_samples, floatsize);
		cudaMalloc(&consts, 4 * sizeof(float));
		cudaMalloc(&bitdec, intsize);

		// Copy data to GPU
		cudaMemcpy(i_samples, inputSamples.getI().data(), floatsize, cudaMemcpyHostToDevice);
		cudaMemcpy(q_samples, inputSamples.getQ().data(), floatsize, cudaMemcpyHostToDevice);
		cudaMemcpy(consts, constants, 4 * sizeof(float), cudaMemcpyHostToDevice);

		cout << "Running on Demodulation loop on Device..." << endl;

		// Run on device
		cudaDemodLoop<<<1,1>>>(i_samples, q_samples, sample_size, bitdec, consts);

		// Get data back from GPU
		cudaMemcpy(output_bits, bitdec, intsize, cudaMemcpyDeviceToHost);

		// Free memory
		cudaFree(i_samples);
		cudaFree(q_samples);
		cudaFree(bitdec);

		vector<int>* output = new vector<int>(output_bits, output_bits + sample_size);

		return *output;
	}

	/**
	 * Initializes all state and signal values
	 */
	void Demodulator::initializeStatesAndSignals() {
		//States
		ped1=0;
		ped2=0;
		ted1=0;
		ted2=0;
		ted3=0;
		ted4=0;
		ted5=0;
		VIp=0;
		VIt=0;
		M=0;
		delayedRegister=0;
		NCO=0;
		OLD_NCO = 0;
		state=0;
		s1=0;
		s2=0;
		FI1 = 0;
		FI2 = 0;
		FI3 = 0;
		FQ1 = 0;
		FQ2 = 0;
		FQ3 = 0;
		strobe = false;

		//Internal signals
		xr=0;
		yr=0;
		I_Prime=0;
		Q_Prime=0;
		vp=0;
		ep=0;
		vt=0;
		et=0;
		mu=0;

		//Loop filter constants
		vector<float> pedFilterConstants = calculateLoopFilterConstants(.01, 1, 18.33, 1, 2);
		//cout << "About to assign ped constants" << endl;
		K1p = pedFilterConstants.at(0);
		K2p = pedFilterConstants.at(1);
		vector<float> tedFilterConstants = calculateLoopFilterConstants(.005, 1, 12.35, -1, 2);
		//cout << "About to assign ted constants" << endl;
		K1t = tedFilterConstants.at(0);
		K2t = tedFilterConstants.at(1);

		//Fill the buffers with zeros
		for(int i=0; i<SAMPLE_BUFFER_SIZE; i++) {
			I_Buffer.push_back(0);
			Q_Buffer.push_back(0);
		}
	}

	/**
	 * Calculates K1 and K2 for the chosen loop by specifying the particular constants.
	 */
	vector<float>& Demodulator::calculateLoopFilterConstants(float BN, float zeta, float Kp, float K0, float N) {
		float theta = BN/(zeta + .25/zeta);
		float d = 1 + 2*zeta*theta/N + theta/N*theta/N;
		float K1 = 4 * zeta/N * theta/d;
		float K2 = 4 * theta/N * theta/N/d;
		K1 = K1/(Kp*K0);
		K2 = K2/(Kp*K0);

		vector<float>* loopConstantsArray = new vector<float>;
		loopConstantsArray->push_back(K1);
		loopConstantsArray->push_back(K2);

		return *loopConstantsArray;
	}

	/**
	 * Gets the index of the current sample (from 0 to 3)
	 */
	int Demodulator::getSampleIndex(float I, float Q) {
		int xBit, yBit;
		if(I > 0) {
			xBit = 1;
		} else {
			xBit = 0;
		}

		if(Q > 0) {
			yBit = 1;
		} else {
			yBit = 0;
		}

		//Use the masks to generate the index. This should be an integer between 0 to 3 inclusive.
		int bitIndex = (xBit << 1) | yBit;

		//Now decode the bits
		switch(bitIndex) {
			case 0:
				return 3;
			case 1:
				return 2;
			case 2:
				return 1;
			case 3:
				return 0;
			default:
				cout << "Invalid bitIndex generated: " << bitIndex << endl;
				throw exception();
		}
	}

	/**
	 * Calculates phase error
	 */
	float Demodulator::calcPhaseError(float xInterpolant, float xDelayedTwo, float xDelayedThree, float yInterpolant, float yDelayedTwo, float yDelayedThree, int state) {
		//Calculate sign(x)*y

		float signX, y;

		switch(state) {
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
				cout << "Error -- invalid state in PED. State: " << state << endl;
				throw exception();
		}
		float productOne = signX * y;

		//Calculate x*sign(y)
		float signY = sign(yInterpolant);
		float productTwo = xInterpolant * signY;


		//Find the final output and return it
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
				cout << "Error -- invalid state in PED. State: " << state << endl;
				throw exception();
		}
	}


	/**
	 * Determines what should be fed into ted2.
	 */
	float Demodulator::calcTED2(float ted1, float xDelayedTwo, float xDelayedThree, int state) {
		float firstSwitch;
		switch(state) {
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
			default:
				cout << "Error in timing detector -- invalid state supplied. State: " << state << endl;
				throw exception();
		}

		return firstSwitch;
	}

	/**
	 * Calculates timing error
	 */
	float Demodulator::calcTimingError(float xInterpolant, float xDelayedTwo, float xDelayedThree, float yInterpolant, int state) {
		//Top Branch
		float firstSwitch = calcTED2(ted1, xDelayedTwo, xDelayedThree, state);

		float summandOne = sign(ted3);
		//delayedX3 = delayedX2;
		float summandTwo = sign(firstSwitch);

		float productOne = ted2 * (summandOne - summandTwo);
		//delayedX2 = firstSwitch;


		//Bottom branch
		summandOne = sign(ted5);
		//delayedY2 = delayedY;
		summandTwo = sign(yInterpolant);

		float productTwo = ted4 * (summandOne - summandTwo);
		//delayedY = yInterpolant;

		//Last Branch
		float switchInput = productOne + productTwo;

		//Return switch
		switch(state) {
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
				cout << "Error in timing detector -- invalid state supplied. State: " << state << endl;
				throw exception();
		}
	}

	/**
	 * Calculates the sign of the input. Returns 1 for positive, -1 for negative, 0 for 0
	 */
	float Demodulator::sign(float input) {
		if(input == 0) return 0;

		if(input > 0) {
			return 1;
		} else {
			return -1;
		}
	}
} /* namespace SOQPSK_Demod */
