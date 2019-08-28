#ifdef _WINDOWS


// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the ACQSYNTH_EXPORTS
// symbol defined on the command line. this symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// ACQSYNTH_API functions as being imported from a DLL, wheras this DLL sees symbols
// defined with this macro as being exported.
#ifdef ACQSYNTH_EXPORTS
    #define ACQSYNTH_API __declspec(dllexport)
#else
    #define ACQSYNTH_API __declspec(dllimport)
#endif

// This class is exported from the AcqSynth.dll
class ACQSYNTH_API CAcqSynth 
{
public:
	CAcqSynth(void);

	// TODO: add your methods here.
};


extern ACQSYNTH_API int nAcqSynth;

ACQSYNTH_API int fnAcqSynth(void);


#define DLLEXP _declspec(dllexport)
#endif
