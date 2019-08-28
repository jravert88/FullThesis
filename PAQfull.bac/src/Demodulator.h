/*
 * Demodulator.h
 *
 *  Created on: May 13, 2013
 *      Author: adm85
 */

#ifndef DEMODULATOR_H_
#define DEMODULATOR_H_


#include <vector>
#include "Samples.h"
using namespace std;

namespace PAQ_SOQPSK {


class Demodulator {
	public:
		Demodulator();
		virtual ~Demodulator();

		vector<int>& demodulate(Samples& inputSamples);
		vector<int>& demodulateCuda(Samples& inputSamples);
		vector<float>& calculateLoopFilterConstants(float newBN, float newZeta, float newKp, float newK0, float newN);
	private:
		void initializeStatesAndSignals();
		float interpolateFarrow(float val, float mu);
		int getSampleIndex(float I, float Q);
		float calcPhaseError(float xInterpolant, float xDelayedTwo, float xDelayedThree, float yInterpolant, float yDelayedTwo, float yDelayedThree, int state);
		float calcTED2(float ted1, float xDelayedTwo, float xDelayedThree, int state);
		float calcTimingError(float xInterpolant, float xDelayedTwo, float xDelayedThree, float yInterpolant, int state);
		float sign(float input);

		//States
		float ped1, ped2; //PED registers
		float ted1, ted2, ted3, ted4, ted5; //TED registers
		float dds; //DDS register
		float VIp, VIt; //Loop filter integrator registers
		float M; //Mu update
		float delayedRegister; //Old NCO
		float NCO, OLD_NCO;
		int state; //Program state (based on strobe)
		float s1, s2; //Program state registers
		float FI1, FI2, FI3; //I Sample Farrow Interpolator Registers
		float FQ1, FQ2, FQ3; //Q Sample Farrow Interpolator Registers
		float B1; //Bit decision delayed output
		bool strobe;


		//Internal signals
		float xr, yr; //Derotated input
		float I_Prime, Q_Prime; //Interpolated samples
		float vp, ep; //Error signals from the PED
		float vt, et; //Error signals from the TED
		float mu; //Timing fraction
		float K1p, K2p, K1t, K2t; //Loop filter constants

		//Variables for the loop
		vector<float> I_Buffer, Q_Buffer;
		static const int SAMPLE_BUFFER_SIZE = 4;
		static const int TIME_N_MINUS_TWO = 2;
		static const int TIME_N_MINUS_THREE = 3;

};

} /* namespace SOQPSK_Demod */
#endif /* DEMODULATOR_H_ */
