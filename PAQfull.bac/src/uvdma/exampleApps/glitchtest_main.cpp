/**
@file uv_acquire_main.cpp
* This is a simple example program which acquires 8192 blocks (8GBytes) of data into onboard memory (continuously) then reads it into system memory, and demonstrates use of the acquired data
*/

#include <iostream>
#include <stdio.h>

#include "glitchTest.h"

#define SYSTEM_MEM_SIZE 10000

/*
Reads the command sent in by the user and parses what board settings the user wishes to set.
*/
void acquire_parser(int argc, char ** argv);

/*
If "acquire" was sent with no arguments, a list of printf statements will display to guide the user.
*/
void acquire_parser_printf();

/*
Takes the arguments read by acquire_parser and sets the board to run with the desired settings.
*/
void acquire_set_session(uvAPI * pUV);

/*
Default values for setupBoard. These may change due to acquire_parser.
*/
unsigned short BoardNum = 0;
unsigned int numBlocksToAcquire = 1;
unsigned int InternalClock = 0;		// Changed default setting to 0 to match "-ic" from old version
unsigned int SingleChannelMode = 0;
unsigned int SingleChannelSelect = 1;
unsigned int DESIQ = 0;
unsigned int DESCLKIQ = 0;
unsigned int ECLTrigger = 0;
unsigned int ECLTriggerDelay = 0;
unsigned int DualChannelMode = 1;
unsigned int DualChannelSelect = 01;
unsigned int CaptureCount = 0;
unsigned int CaptureDepth = 0;
unsigned int Decimation = 1;
unsigned int PretriggerMemory = 0;
unsigned int TTLInvert = 0;
unsigned int TriggerMode = NO_TRIGGER;
unsigned int TriggerCh = 0;
unsigned int TriggerSlope = FALLING_EDGE;
unsigned int TriggerThreshold = 32768;
unsigned int TriggerHysteresis = 100;
unsigned int NumAverages = 0;
unsigned int AveragerLength = 4096;
unsigned int Fiducial = 0;
unsigned int forceCal = 0;

char * user_outputfile = NULL;



int main(int argc, char ** argv)
{
    HANDLE disk_fd = INVALID_HANDLE_VALUE; // disk file handle
    char output_file[MAX_DEVICES][128] = {"uvdma.dat", "uvdma1.dat", "uvdma2.dat", "uvdma3.dat"};


	std::cout << "uv_acquire v1.0" << std::endl << std::endl;

	// Create a class with convienient access functions to the DLL
	uvAPI *uv = new uvAPI;


	// Allocate a page aligned buffer for DMA 
    unsigned char * sysMem[SYSTEM_MEM_SIZE];
    for (int i=0;i<SYSTEM_MEM_SIZE;i++)
	{
		sysMem[i] = NULL;
        uv->X_MemAlloc((void**)&sysMem[i], DIG_BLOCK_SIZE);
		if (!sysMem[i])
		{
			std::cout << "failed to allocate block buffer" << std::endl;
			for (int j=(i-1);j>= 0;j--)
                if (sysMem[j]) {uv->X_FreeMem(sysMem[j]);}

			return  -1;

		}
	}

	// Setup the first (usually only) board in the system

    uv->setSetupDoneBit(BoardNum,0);  // this way we always re-cal the board
	acquire_parser(argc, argv);

    printf("BoardNum = %d\n",BoardNum);
    fflush(stdout);

    // Initialize settings
    uv->selClock(BoardNum, InternalClock);
    uv->setupBoard(BoardNum);
    uv->getAdcClockFreq(BoardNum);

	// Write user settings to the board
	acquire_set_session(uv);

	for (int runNum = 0; runNum < 1000 ;runNum++)
	{
		unsigned int adcClock = uv->getAdcClockFreq(BoardNum); 
        std::cout << uv->getOverruns(BoardNum) << " overruns" << std::endl;

		std::cout << "\n\nIteration " << runNum << " ADC clock freq ~=" << adcClock << std::endl;
		glitchTest *test = new glitchTest;
        uv->SetupAcquire(BoardNum,numBlocksToAcquire);
//		if (ECLTrigger >0)
//		{
//			uv->SetECLTriggerEnable(BoardNum,1);
//			printf("waiting for ecl trigger\n");
//		}
//		else
//		{
//			printf("begin readout\n");
//		}
		for	(unsigned int i=0; i<numBlocksToAcquire; i++ )
		{
			// read a block from the board
            uv->X_Read(BoardNum,sysMem[i],DIG_BLOCK_SIZE);
			if (i==0)
				printf("reading board to local memory\n");
			if ( (i%100) == 0 )
			{
				std::cout << i << "\r";
                fflush(stdout);
			}
		}
		printf ("\ndone copying from board to local memory\n");
        std::cout << uv->getOverruns(BoardNum) << " overruns" << std::endl;

		for	(unsigned int i=0; i<numBlocksToAcquire; i++ )
		{
			// testForGlitches is used to check for sharp discontinuities in the acquired data.
			// 

			test->testForGlitches(BoardNum ,uv ,sysMem[i] ,DIG_BLOCK_SIZE);

			if (test->numErrors >0)
			{
                // open the data disk file
                if (user_outputfile == NULL)	// If user did not specify name, use default uvdma.dat
                {
                    if ((disk_fd = uv->X_CreateFile(output_file[BoardNum])) < 0)
                    {
                        for (int j=0; j<SYSTEM_MEM_SIZE;j++)
                            if (sysMem[j]) {uv->X_FreeMem(sysMem[j]);}
                        delete uv;
                        exit(1);
                    }
                }
                else	// User specified name obtained from acquire_parser
                {
                    if ((disk_fd = uv->X_CreateFile(user_outputfile)) < 0)
                    {
                        for (int j=0; j<SYSTEM_MEM_SIZE;j++)
                            if (sysMem[j]) {uv->X_FreeMem(sysMem[j]);}
                        delete uv;
                        exit(1);
                    }
                }

				printf("Glitch started in block %d at %d\n",i,test->errMin);

				if (i>=1)
				{
					if (test->errMin <= 1)  // we need preceding block
					{
                        unsigned int numChannels = uv->GetNumChannels(BoardNum);
						unsigned int sampleNum = DIG_BLOCK_SIZE;
                        sampleNum /= 2; // sizeof(int16)
						sampleNum /= numChannels;  // this is numSamples per block
						sampleNum -= 100;  // show 100 preceding smaples 
                        uv->X_Write(disk_fd, sysMem[(i-1)] + (sampleNum * 2 * numChannels), (DIG_BLOCK_SIZE -  (sampleNum  * 2 * numChannels)));
                        uv->X_Write(disk_fd, sysMem[i], (DIG_BLOCK_SIZE - (100 *  2 * numChannels )));
					}
					else
					{
                        uv->X_Write(disk_fd, sysMem[i], DIG_BLOCK_SIZE);
					}
				}
				else
				{
                    uv->X_Write(disk_fd, sysMem[i], DIG_BLOCK_SIZE);
				}

                uv->X_Close(disk_fd);    // Close the disk file
                for (int j=0; j<SYSTEM_MEM_SIZE;j++)
                    if (sysMem[j]) {uv->X_FreeMem(sysMem[j]);}
				delete uv;
				exit(0);

			}

			if ( (i%100) == 0 )
			{
				std::cout << i << "\r";
                fflush(stdout);

			}
		}
		std::cout << std::endl << test->numErrors << " glitches were detected this run" << std::endl;
        if ((disk_fd = uv->X_CreateFile("uvdma.dat")) < 0)
		{
			std::cout << "failed to open uvdma.dat" << std::endl;
            for (int j=0; j<SYSTEM_MEM_SIZE;j++)
                if (sysMem[j]) {uv->X_FreeMem(sysMem[j]);}
			delete uv;
			exit(1);
		}
        uv->X_Write(disk_fd, sysMem[0], DIG_BLOCK_SIZE);
        uv->X_Close(disk_fd);    // Close the disk file

        std::cout << uv->getOverruns(BoardNum) << " overruns" << std::endl;

		delete test;


	}

    for (int j=0 ; j<SYSTEM_MEM_SIZE ; j++)
        if (sysMem[j]) {uv->X_FreeMem(sysMem[j]);}
	delete uv;
	return 0;
	
}


void acquire_parser(int argc, char ** argv)
{
    int arg_index;

    // check how many arguments, if run without arguements prints usage.
    if(argc == 1)
    {
        acquire_parser_printf();
        exit(1);
    }
    else
    {
        // make sure 2nd arguement is valid number of blocks (1st arguement is the application name)
        numBlocksToAcquire = atoi(argv[1]);

        if(numBlocksToAcquire <= 0)
        {
            std::cout << "Invalid number of blocks specified, exiting!" << std::endl;
            exit(1);
        }

        // starting at third arguement look for options
        for(arg_index=2; arg_index<argc; arg_index++)
        {

            if( strcmp(argv[arg_index], "-b") == 0 )
            {
                // make sure option is followed by (BoardNum)
                if(argc>(arg_index+1))
                {
                    arg_index++; // increment the arguement index b/c we have now taking care of the next arguement
                    BoardNum = atoi(argv[arg_index]);
                }
                else{
                    std::cout << "Invalid board number with ""-b"" option" << std::endl;
                    exit(1);
                }
            }
            else if ( strcmp(argv[arg_index], "-forceCal") == 0 )
            {
                forceCal = 1;
                std::cout << "force calibration selected" << std::endl;
            }
            else if ( strcmp(argv[arg_index], "-ic") == 0 )
            {
                InternalClock = 1;
                std::cout << "internal clock selected" << std::endl;
            }
            else if ( strcmp(argv[arg_index], "-f") == 0 )
            {
                if(argc>(arg_index+1))
                {
                    arg_index++; // increment the arguement index b/c we have now taking care of the next arguement
                    user_outputfile  = argv[arg_index];
                }
            }
            else if ( strcmp(argv[arg_index], "-scm") == 0 )
            {
                SingleChannelMode = 1;
                SingleChannelSelect = 0;
            }
            else if ( strcmp(argv[arg_index], "-scs") == 0 )
            {
                // make sure option is followed by (Chan N)
                if(argc>(arg_index+1))
                {
                    arg_index++; // increment the arguement index b/c we have now taking care of the next arguement
                    SingleChannelSelect= atoi(argv[arg_index]);
                }
                else
                {
                    std::cout << "(Chan N) must be in range 0,1 for 2-channel boards and 0,1,2,3 for 4-channel boards" << std::endl;
                    exit(1);
                }
            }
            else if ( strcmp(argv[arg_index], "-desiq") == 0 )
            {
                DESIQ = 1;
            }
            else if ( strcmp(argv[arg_index], "-desclkiq") == 0 )
            {
                DESCLKIQ = 1;
            }
            else if ( strcmp(argv[arg_index], "-ecltrig") == 0 )
            {
                ECLTrigger = 1;
            }
            else if ( strcmp(argv[arg_index], "-ecldelay") == 0 )
            {
                if(argc>(arg_index+1))
                {
                    arg_index++; // increment the arguement index b/c we have now taking care of the next arguement
                    ECLTriggerDelay  = atoi(argv[arg_index]);
                }
                else
                {
                    std::cout << "ECL trigger delay must be [0,2^32]" << std::endl;
                    exit(1);
                }
            }

            else if ( strcmp(argv[arg_index], "-dcm") == 0 )
            {
                DualChannelMode = 1;
                DualChannelSelect = 3; // Default DualChannelSelect = channel 0 channel 1
            }
            else if ( strcmp(argv[arg_index], "-dcs") == 0 )
            {
                // make sure option is followed by (Chan N)
                if(argc>(arg_index+1))
                {
                    arg_index++; // increment the arguement index b/c we have now taking care of the next arguement
                    DualChannelSelect= atoi(argv[arg_index]);
                }
                else
                {
                    std::cout << "(Chan NM) must be 01, 02, 03, 12, 13, 23" << std::endl;
                    exit(1);
                }
            }
            else if ( strcmp(argv[arg_index], "-capture_count") == 0 )
            {
                if(argc>(arg_index+1))
                {
                    arg_index++; // increment the arguement index b/c we have now taking care of the next arguement
                    CaptureCount  = atoi(argv[arg_index]);
                }
                else
                {
                    printf("setCaptureDepth must be [0,2^32] \n");
                    exit(1);
                }
            }
            else if ( strcmp(argv[arg_index], "-capture_depth") == 0 )
            {
                if(argc>(arg_index+1))
                {
                    arg_index++; // increment the arguement index b/c we have now taking care of the next arguement
                    CaptureDepth  = atoi(argv[arg_index]);
                }
                else
                {
                    std::cout << "capture_depth must a multiple of 8 up to 2^32. 0 = normal acquisition." << std::endl;
                    exit(1);
                }

                if( (CaptureDepth % 8) != 0 )
                {
                    std::cout << "capture_depth must a multiple of 8 up to 2^32. 0 = normal acquisition." << std::endl;
                    exit(1);
                }
            }
            else if ( strcmp(argv[arg_index], "-dec") == 0 )
            {
                // Make sure option is followed by the desired decimation amount
                if(argc>(arg_index+1))
                {
                    arg_index++; // increment the arguement index b/c we have now taking care of the next arguement
                    Decimation  = atoi(argv[arg_index]);
                }
                else
                {
                    std::cout << "Decimation must either be 1, 2, 4, 8" << std::endl;
                    exit(1);
                }


                if( !((Decimation == 1) ||
                    (Decimation  == 2) ||
                    (Decimation  == 4) ||
                    (Decimation  == 8) ) )
                {
                    std::cout << "Decimation must either be 1, 2, 4, 8" << std::endl;
                    exit(1);
                }
            }
            else if ( strcmp(argv[arg_index], "-pretrigger") == 0 )
            {
                if(argc>(arg_index+1))
                {
                    arg_index++; // increment the arguement index b/c we have now taking care of the next arguement
                    PretriggerMemory  = atoi(argv[arg_index]);

                }
                else
                {
                    std::cout << "pretrigger must be 0 to 4096" << std::endl;
                    exit(1);
                }
                if (PretriggerMemory > 4096)
                {
                    std::cout << "pretrigger must be 0 to 4096" << std::endl;
                    exit(1);
                }
            }
            else if( strcmp(argv[arg_index], "-ttledge") == 0 )
            {
                TriggerMode = TTL_TRIGGER_EDGE;
            }
            else if( strcmp(argv[arg_index], "-ttlinv") == 0 )
            {
                TTLInvert = 1;
            }
            else if( strcmp(argv[arg_index], "-hdiode") == 0 )
            {
                TriggerMode = HETERODYNE;
            }
            else if( strcmp(argv[arg_index], "-syncselrecord") == 0 )
            {
                TriggerMode = SYNC_SELECTIVE_RECORDING;
            }
            else if( strcmp(argv[arg_index], "-analog") == 0 )
            {
                TriggerMode = WAVEFORM_TRIGGER;
            }
            else if ( strcmp(argv[arg_index], "-analog_ch") == 0 )
            {
                if(argc>(arg_index+1))
                {
                    arg_index++; // increment the arguement index b/c we have now taking care of the next arguement
                    TriggerCh  = atoi(argv[arg_index]);

                }
                else
                {
                    std::cout << "analog_ch must be either 0,1,2,3 (4-channel boards), or 0,1 (2-channel boards)" << std::endl;
                    exit(1);
                }
                if (PretriggerMemory >2047)
                {
                    std::cout << "analog_ch must be either 0,1,2,3 (4-channel boards), or 0,1 (2-channel boards)" << std::endl;
                    exit(1);
                }
            }
            else if( strcmp(argv[arg_index], "-threshold") == 0 )
            {
                if(argc>(arg_index+1))
                {
                    arg_index++; // increment the arguement index b/c we have now taking care of the next arguement
                    TriggerThreshold  = atoi(argv[arg_index]);

                }
                else
                {
                    std::cout << "thresh_a must be 0 to 65535" << std::cout;
                    exit(1);
                }
            }

            else if( strcmp(argv[arg_index], "-hysteresis") == 0 )
            {
                if(argc>(arg_index+1))
                {
                    arg_index++; // increment the arguement index b/c we have now taking care of the next arguement
                    TriggerHysteresis  = atoi(argv[arg_index]);

                }
                else
                {
                    std::cout << "thresh_b must be 0 to 65535" << std::endl;
                    exit(1);
                }
            }
            else if ( strcmp(argv[arg_index], "-avg") == 0 )
            {
                if(argc>(arg_index+1))
                {
                    arg_index++; // increment the arguement index b/c we have now taking care of the next arguement
                    NumAverages  = atoi(argv[arg_index]);
                }
                else
                {
                    std::cout << "NumAverages must be [0,2^32]" << std::endl;
                    exit(1);
                }
            }
            else if ( strcmp(argv[arg_index], "-avg_len") == 0 )
            {
                if(argc>(arg_index+1))
                {
                    arg_index++; // increment the arguement index b/c we have now taking care of the next arguement
                    AveragerLength  = atoi(argv[arg_index]);
                }
                else
                {
                    std::cout << "AveragerLength must be [0, 2^17]" << std::endl;
                    exit(1);
                }
            }
            else if( strcmp(argv[arg_index], "-fiducial") == 0 )
            {
                Fiducial = 1;
            }
        }
    }
}

void acquire_set_session(uvAPI *pUV)
{

    if ((forceCal) && pUV->IS_ISLA216P(BoardNum))  // presently there is not an explicit AD16 calibrate command.  Changing clocks forces calibration.
    {
        {
            if (InternalClock == CLOCK_INTERNAL)
                pUV->selClock(BoardNum, CLOCK_EXTERNAL);
            else
                pUV->selClock(BoardNum, CLOCK_INTERNAL);
        }
    }
    pUV->selClock(BoardNum, InternalClock);
    if (pUV->IS_ISLA216P(BoardNum))
    {
        unsigned int selectedChannels;

        if (SingleChannelMode == 1)
        {
            switch (SingleChannelSelect) {
            case 0:
                selectedChannels = IN0;
            case 1:
                selectedChannels = IN3;  // due to part poulation on 2ch verion ch1 is on IN3
            case 2:
                selectedChannels = IN1;
            case 3:
                selectedChannels = IN2;
            default:
                selectedChannels = IN0;
            }
        }
        else if (DualChannelMode == 1)
        {
            selectedChannels = (IN0 | IN3);
        }
        else
        {
            selectedChannels = (IN0 | IN1 | IN2 | IN3);
        }
        std::cout << BoardNum << "AD16 setup" << std::endl;
        pUV->selectAdcChannels(BoardNum, selectedChannels);

        unsigned int TriggerChannel;
        switch (TriggerCh) {
        case 0:
            TriggerChannel = IN0;
        case 1:
            TriggerChannel = IN3;  // due to part poulation on 2ch verion ch1 is on IN3
        case 2:
            TriggerChannel = IN1;
        case 3:
            TriggerChannel = IN2;
        default:
            TriggerChannel = IN0;
        }
        pUV->selectTrigger(BoardNum, TriggerMode, TriggerSlope, TriggerChannel);
        pUV->configureWaveformTrigger(BoardNum, TriggerThreshold, TriggerHysteresis);
        pUV->configureSegmentedCapture(BoardNum, CaptureCount, CaptureDepth, 0);
        pUV->configureAverager(BoardNum, NumAverages, AveragerLength, 0);
        pUV->setFiducialMarks(BoardNum, Fiducial);
    }
    if (pUV->IS_adc12d2000(BoardNum))
    {
        if (SingleChannelMode == 1)	// One channel mode
        {
            if (DESCLKIQ == 1)
                pUV->ADC12D2000_Channel_Mode(BoardNum, ONE_CH_MODE, DESCLKIQ_MODE, 1);	// DESCLKIQ
            else if (DESIQ == 1)
                pUV->ADC12D2000_Channel_Mode(BoardNum, ONE_CH_MODE, DESIQ_MODE, 1);	// DESIQ
            else
                if (SingleChannelSelect == 0)
                    pUV->ADC12D2000_Channel_Mode(BoardNum, ONE_CH_MODE, DESQ_MODE, 1);	// AIN0
                else
                    pUV->ADC12D2000_Channel_Mode(BoardNum, ONE_CH_MODE, DESI_MODE, 1);	// AIN1
        }
        else
        {
            pUV->ADC12D2000_Channel_Mode(BoardNum, TWO_CH_MODE, 1, 1);	// Two channel mode
        }
        // Set ECL Trigger Delay
        pUV->SetECLTriggerDelay(BoardNum, ECLTriggerDelay);
        // Set Decimation
        pUV->setAdcDecimation(BoardNum, Decimation);
        // Set ECL Trigger
        std::cout << "ECLTrigger=" << ECLTrigger << std::endl;
        pUV->SetECLTriggerEnable(BoardNum, ECLTrigger);
    }

    //       pUV->SetAdcDecimation(BoardNum, Decimation);
    pUV->SetTTLInvert(BoardNum, TTLInvert);
    pUV->setPreTriggerMemory(BoardNum, PretriggerMemory);
}

void acquire_parser_printf()
{
    printf("\n");
    printf("glitchTest will setup the requested board with desired settings, initiate\nacquisition and check data for glitches. \n\nUsage:\n\n");
    printf("glitchTest (N blocks) [OPTIONS]\n");
    printf("\tAcquires specified number of blocks. Number of blocks option\n\t\tmust always be first.\n\n");

    printf("The following [OPTIONS] may follow the number of blocks option in any order:\n\n");

    printf("-f (filename)\t\tUse this argument to specify the name of the file to write. Must append .dat at the end of filename\n");
    printf("\n");
    printf("-ic\t\t\tBoard will use the internal clock. If not specified board uses external clock.\n");
    printf("\n");
    printf("-scm\t\t\tRun multi-channel boards in single channel mode. Not required for single channel boards.\n");
    printf("-scs (Chan N)\t\tWhen running multi-channel boards in single channel mode use\n\t\t\tchannel (Chan N).\n");
    printf("\n");
    printf("-dcm\t\t\tRun multi-channel boards in dual channel mode. Not required for dual channel boards.\n");
    printf("-dcs <NM>\t\tWhen running in dual channel mode, specify the channels to acquire from. \n\t\t\tWhere channels NM can be 01, 02, 03, 12, 13, 23.\n");
    printf("\n");
    printf("-b (BoardNum)\t\tAcquires from the specified board. Do not specify the same board\n\t\t\tfrom two separate consoles. BoardNum starts at 0.\n");
    printf("\n");
    printf("-ttledge\t\tSets acquisition to await TTL trigger edge.\n");
    printf("-ttlinv\t\t\tInverts the selective recording feature. Default is active LOW.\n");
    printf("-ecldelay (M)\t\tSet the ECL trigger delay. Default = 8000 (64us). See manual for details.\n");
    printf("-dec (factor)\t\tEnables input sample decimation. (factor) can be 1,2,4,8 or 16.\n");
    printf("-capture_count (X)\tEnable segmented captures with X number of records of length Y.\n");
    printf("-capture_depth (Y)\tEnable segmented captures with X number of records of length Y. Y must be a multiple of 8 samples.\n");
    printf("-analog\t\tFor analog waveform trigger. Use -ttlinv to specify rising or falling edge.\n");
    printf("-threshold <N>\t\tFor analog waveform trigger threshold. <N> = [0,65535].\n");
    printf("-hysteresis <N>\t\tFor analog waveform trigger hysteresis. The hysteresis value = threshold + <N>. <N> = [0,65535].\n");
    printf("-fiducial\t\tPlaces Fiducial marks in data when trig/sel record starts recording\n");
    printf("-pretrigger\t\tFor pretrigger memory after trigger is acquired. pretrigger must be 0 to 512.\n");
    printf("-avg <N> \t\t(N=0-65535)\tAverages N+1 cycles,  N=0 is flow through mode\n");
    printf("-avg_len <N>\t\tFor 50T/110T:\t1-channel mode: N=16384,8192,4096,2048,1024,512,256,128,64,32,16\n");
    printf("\t\t\t\t\t2-channel mode: N=8192,4096,2048,1024,512,256,128,64,32,16,8\n");
    printf("\t\t\t\t\t4-channel mode: N=4096,2048,1024,512,256,128,64,32,16,8,4\n");
    printf("\t\t\tFor 155T:\t1-channel mode:\tN=131072,65536,32768,16384,8192,4096,2048,1024,512,256,128,64,32,16\n");
    printf("\t\t\t\t\t2-channel mode: N=65536,32768,16384,8192,4096,2048,1024,512,256,128,64,32,16,8\n");
    printf("\t\t\t\t\t4-channel mode: N=32768,16384,8192,4096,2048,1024,512,256,128,64,32,16,8,4\n");
    printf("\t\t\tavg_len default is 4096.\n");
    //printf("-v\t\tEnable extra print statements.\n");
}
