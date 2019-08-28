/*
DLL Main file. This file exports DLL functions, implementes the DLL entry/exit point in DllMain,
and implements the Windows ClassInstaller.

Created 4/11/05
*/


    #include "DllMain.h"
    #include "DllLibMain.h"
	#include "DllDriverApi.h"
	#include <stdio.h>

#ifdef _WINDOWS
    #include <setupapi.h>	// 063010
    #ifndef arraysize
        #define arraysize(p) (sizeof(p)/sizeof((p)[0]))
    #endif

    HANDLE DllConsole;                      // For DLL debug print statements
    int DllMsgLen;
    char DllMsg[DLL_MSG_LEN];
    DWORD DllJunkP;
    char ApplicationDirectory[250];         // Labview runs DLL from C:\Windows\system32 even though DLL is loaded from same directory as Labview project!

#else

#endif

CDllDriverApi *pAPI = NULL;
unsigned char * sysMem = NULL;


extern DEVICE_HANDLE;              // Handle to the low-level driver API



#ifdef _WINDOWS
// DLL_BUILD_WINDOWS  Windows DLL entry/exit point.  
bool APIENTRY DllMain(HANDLE hModule, DWORD ul_reason_for_call, PVOID lpReserved)
{

    switch( ul_reason_for_call ) 
    {
        case DLL_PROCESS_ATTACH:

            // Create the low-level API
            pAPI = new CDllDriverApi;

            // Initialize the API
            pAPI->ApiInitialize();


// If we are going to print from the DLL then allocate a console window (Windows Only)           
#ifdef DLL_PRINT_EN
    AllocConsole();
	DllConsole = GetStdHandle(STD_OUTPUT_HANDLE);

    sprintf(DllMsg, "DllMain() Console Allocated \n");
    DllMsgLen = strlen(DllMsg);
    WriteConsole(DllConsole, (void*)DllMsg, DllMsgLen, &DllJunkP, NULL );   // We have to write something to make the printf's work?
#endif

            break;

        case DLL_THREAD_ATTACH:
            printf(" DLL_THREAD_ATTACH \n");
            break;

        case DLL_THREAD_DETACH:
            printf(" DLL_THREAD_DETACH \n");
            break;

        case DLL_PROCESS_DETACH:

            break;
    }
    
    return TRUE;   
}



// Labview runs DLL from C:\Windows\system32 even though DLL was loaded from project directory!
extern "C"  DLLEXP void DllSetAppDirectory(char *AppDir)
{
    strcpy(ApplicationDirectory, AppDir);
    SetCurrentDirectory(ApplicationDirectory);
//    printf("DllSetAppDirectory() AppDir= %s \n", ApplicationDirectory);
}


#define BLOCK_SIZE 1024*1024
// Directly DMA into Labview buffer. Used to read data from the board.
extern "C"  DLLEXP bool DllReadDataBlockFile(const char * filename, unsigned int blockNum,  unsigned char * pBuffer)
{
#ifdef _WINDOWS
    int ret;
	__int64 seekPos = BLOCK_SIZE * (__int64)(blockNum);

	FILE *fileHandle;

	fileHandle = fopen(filename,"rb");
	ret = _fseeki64(fileHandle,seekPos,SEEK_SET);
	if (ret != 0)
	{
		return false;
	}

	ret = fread(pBuffer,1,BLOCK_SIZE,fileHandle);
	if (ret != BLOCK_SIZE)
	{
		return false;
	}

	fclose(fileHandle);
#endif

    return true;    
}

// Return the file size in BLOCK_SIZE units
extern "C"  DLLEXP unsigned long long DllReadDataFileSize(const char * filename)
{
    unsigned long long size;
	unsigned long long NumBlocks;

	FILE *fileHandle;

	fileHandle = fopen(filename,"rb");

	_fseeki64(fileHandle, 0, SEEK_END);
	size = _ftelli64(fileHandle);
	
	NumBlocks = size/(unsigned long long)(1024*1024);

	fclose(fileHandle);

    return NumBlocks;
}
 
// Directly DMA into Labview buffer. Used to read data from the board.
extern "C"  DLLEXP bool DllReadDataBlock(unsigned char * pBuffer)
{
    unsigned long ret=0;

    // Read from the board into the DMA buffer
    if(!ReadFile(pAPI->hCurrentDevice, pBuffer, DIG_BLOCKSIZE, &ret, NULL ) )
    {
        printf("DllReadDataBlock()->ReadFile() failed!\n");
        return false;
    }
    else
    {
        //(pBuffer) = pLabviewBounceBuffer;      // return a pointer to the DMA buffer
        //memcpy(pBuffer, pLabviewBounceBuffer, DIG_BLOCKSIZE);   // copy from our local buffer to the labview buffer
        //printf("DllReadDataBlock() pBuffer=%x \n", pBuffer);
    }

    return true;    
}


#else
    
// Under Linux we just linked to a static object, there isn't really a DLL
bool DllLoad()
{
    // Under Linux we just link to a static object, there isn't really a DLL. The user program must call this function, as the "DLL" is never "loaded".

    // Create the low-level API
    pAPI = new CDllDriverApi;

    // Initialize the API
    pAPI->ApiInitialize();
}

#endif










///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Exported DLL functions must be added to DllMain.cpp (DLL export), and AppDll.cpp (App import), and AppDll.h (definition)
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


extern "C"  DLLEXP void DllADC12D2000_SET_I_CHANNEL_OFFSET_ADJUST(unsigned short BoardNum, unsigned int Sign, unsigned int Offset)
{
    return ADC12D2000_SET_I_CHANNEL_OFFSET_ADJUST(BoardNum, Sign, Offset);
}

extern "C"  DLLEXP void DllADC12D2000_SET_I_CHANNEL_FULL_SCALE_RANGE_ADJUST(unsigned short BoardNum, unsigned int FSR)
{
    return ADC12D2000_SET_I_CHANNEL_FULL_SCALE_RANGE_ADJUST(BoardNum, FSR);
}

extern "C"  DLLEXP void DllADC12D2000_SET_Q_CHANNEL_OFFSET_ADJUST(unsigned short BoardNum, unsigned int Sign, unsigned int Offset)
{
    return ADC12D2000_SET_Q_CHANNEL_OFFSET_ADJUST(BoardNum, Sign, Offset);
}

extern "C"  DLLEXP void DllADC12D2000_SET_Q_CHANNEL_FULL_SCALE_RANGE_ADJUST(unsigned short BoardNum, unsigned int FSR)
{
    return ADC12D2000_SET_Q_CHANNEL_FULL_SCALE_RANGE_ADJUST(BoardNum, FSR);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" bool DllApiInitialize()
{
	return ATTACH_TO_DEVICE();
}

// DLL_API_LEVEL    This is a low-level exported DLL function
extern "C"  DLLEXP unsigned short DllApiGetNumDevices()
{
    return GET_NUMDEVICES();
/*
    // Test Labview cluster ordering:
    printf("DllApiGetNumDevices()\n");    
    ss->blocks_on_board = 2;
    ss->config_info_found = 3;
    ss->acquisition_ready = 4;
    ss->adc_calibration_ok = 5;
    ss->adc_clock_ok = 6;
    ss->adc_clock_freq = 7;
    ss->dac_clock_freq = 8;
    ss->samples_per_block = 9;
    ss->adc_chan_used = 10;

    ss->serial_number = 11;
    ss->adc_res = 12;
    ss->dac_res = 13;
    ss->adc_chan = 14;
    ss->dac_chan = 15;

    ss->dma_test_mode = 16;        
    ss->use_large_mem = 17;         
    ss->overruns = 18;        
    ss->frequency = 19;
    ss->freq_mode = 20; 
    ss->convert_mode = 21;
//*/
}    

extern "C"  DLLEXP short DllApiGetSerialNumber(unsigned short board_index)
{
	return GET_SERIALNUMBER(board_index);
}

// DLL_API_LEVEL    This is a low-level exported DLL function
extern "C"  DLLEXP HANDLE DllApiSetCurrentDevice(unsigned short BoardNum)  
{    
    return SET_CURRENTDEVICE(BoardNum);        // returns the handle to the currently selected device
}  

// DLL_API_LEVEL    This is a low-level exported DLL function
extern "C"  DLLEXP void DllApiSetPreventUnderOverRuns(unsigned short BoardNum, bool value)   
{
    SET_PREVENT_UNDEROVERRUNS(BoardNum, value);
}

// DLL_API_LEVEL    This is a low-level exported DLL function
extern "C"  DLLEXP void DllApiSetPioRegister(unsigned short BoardNum, unsigned long pio_offset, unsigned long data) 
{
    SET_PIO_REG(BoardNum, pio_offset, data);
}  

extern "C" DLLEXP void DllApiSetShadowRegister(unsigned short BoardNum, unsigned long pio_offset, unsigned long data, unsigned long spiAddress, unsigned long mask_info)
{
	SET_SHADOW_REG(BoardNum,  pio_offset, data, spiAddress, mask_info);
}
// DLL_API_LEVEL    This is a low-level exported DLL function
extern "C"  DLLEXP void DllSetArmed(unsigned short BoardNum) 
{
    SET_ARMED(BoardNum);
}  

// DLL_API_LEVEL    This is a low-level exported DLL function
extern "C"  DLLEXP  unsigned long DllApiGetPioRegister(unsigned short BoardNum, unsigned long pio_offset)  
{
    return GET_PIO_REG(BoardNum, pio_offset);
}

extern "C" DLLEXP unsigned long DllApiGetShadowRegister(unsigned short BoardNum, unsigned long pio_offset, unsigned long spiAddress, unsigned long mask_info)
{
	return GET_SHADOW_REG(BoardNum, pio_offset, spiAddress, mask_info);
}
// DLL_API_LEVEL    This is a low-level exported DLL function
extern "C"  DLLEXP  unsigned long DllApiGetOverruns(unsigned short BoardNum) 
{
    return GET_OVERRUNS(BoardNum);
} 

// DLL_API_LEVEL    This is a low-level exported DLL function
extern "C"  DLLEXP void DllSetAD83000xReg(unsigned short BoardNum, unsigned int FunctionCode, unsigned int Setting) 
{
    SET_CONTROL_REG(BoardNum, FunctionCode, Setting);
}  
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Exported DLL functions must be added to DllMain.cpp (DLL export), and AppDll.cpp (App import), and AppDll.h (definition)
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////










#ifdef _WINDOWS

// This is the constructor of a class that has been exported.
// see DllWinMain.h for the class definition
CAcqSynth::CAcqSynth()
{ 
	return; 
}





///////////////////////////////////////////////////////////////////////////////
// This function is the "class installer" for the AcqSynth class. The function
// is obviously trivial -- its only purpose is to allow us to provide a custom
// icon for the "sample" class of device. This function is specifically exported
// in the .DEF file, which is required in lieu of just using __declspec(dllexport)
// in the definition of the function to avoid the __stdcall name decoration. In addition,
// the class key for "sample" specifies this dll and function by name as the Installer32
// value.

extern "C" DWORD CALLBACK AcqSynthClassInstaller(DI_FUNCTION fcn, HDEVINFO infoset, PSP_DEVINFO_DATA did)
{


#if _DEBUG
	static char* difname[] = 
	{
		"0",
		"DIF_SELECTDEVICE",
		"DIF_INSTALLDEVICE",
		"DIF_ASSIGNRESOURCES",
		"DIF_PROPERTIES",
		"DIF_REMOVE",
		"DIF_FIRSTTIMESETUP",
		"DIF_FOUNDDEVICE",
		"DIF_SELECTCLASSDRIVERS",
		"DIF_VALIDATECLASSDRIVERS",
		"DIF_INSTALLCLASSDRIVERS",
		"DIF_CALCDISKSPACE",
		"DIF_DESTROYPRIVATEDATA",
		"DIF_VALIDATEDRIVER",
		"DIF_MOVEDEVICE",
		"DIF_DETECT",
		"DIF_INSTALLWIZARD",
		"DIF_DESTROYWIZARDDATA",
		"DIF_PROPERTYCHANGE",
		"DIF_ENABLECLASS",
		"DIF_DETECTVERIFY",
		"DIF_INSTALLDEVICEFILES",
		"DIF_UNREMOVE",
		"DIF_SELECTBESTCOMPATDRV",
		"DIF_ALLOW_INSTALL",
		"DIF_REGISTERDEVICE",
		"DIF_NEWDEVICEWIZARD_PRESELECT",
		"DIF_NEWDEVICEWIZARD_SELECT",
		"DIF_NEWDEVICEWIZARD_PREANALYZE",
		"DIF_NEWDEVICEWIZARD_POSTANALYZE",
		"DIF_NEWDEVICEWIZARD_FINISHINSTALL",
		"DIF_UNUSED1",
		"DIF_INSTALLINTERFACES",
		"DIF_DETECTCANCEL",
		"DIF_REGISTER_COINSTALLERS",
		"DIF_ADDPROPERTYPAGE_ADVANCED",
		"DIF_ADDPROPERTYPAGE_BASIC",
		"DIF_RESERVED1",
		"DIF_TROUBLESHOOTER",
		"DIF_POWERMESSAGEWAKE",
		};

	char msg[128];
	if (fcn < arraysize(difname))
		sprintf(msg, "ACQSYNTH - %s\n", difname[fcn]);
	else
		sprintf(msg, "ACQSYNTH - 0x%X\n", fcn);

	OutputDebugString(msg);
#endif // _DEBUG

	return ERROR_DI_DO_DEFAULT;	// do default action in all cases


}	// AcqSynthClassInstaller


#endif // End (#ifdef _WINDOWS) for Windows ClassInstaller







