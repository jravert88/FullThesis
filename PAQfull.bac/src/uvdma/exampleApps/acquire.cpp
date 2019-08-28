/**
@file pUV_acquire_main.cpp
* This is a simple example program which acquires 8192 blocks (8GBytes) of data into onboard memory (continuously) then reads it into system memory, and demonstrates use of the acquired data
*/

#include <iostream>
#include "uvAPI.h"
//#include "glitchTest.h"


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
Default values for SetupBoard. These may change due to acquire_parser.
*/
unsigned short BoardNum = 0;
unsigned int numBlocksToAcquire = 1;
unsigned int InternalClock = CLOCK_EXTERNAL;		// Changed default setting to 0 to match "-ic" from old version
unsigned int SingleChannelMode = 0;
unsigned int SingleChannelSelect = 1;
unsigned int ChannelSelect = 1;
unsigned int DESIQ = 0;
unsigned int DESCLKIQ = 0;
unsigned int ECLTrigger = 0;
unsigned int ECLTriggerDelay = 0;
unsigned int DualChannelMode = 0;
unsigned int DualChannelSelect = 01;
unsigned int firstchan = 0;		//default to channel 0
unsigned int secondchan = 3;	//default to channel 3
unsigned int chans = 9;			//default bitwise channel 0 and 3
unsigned int CaptureCount = 0;
unsigned int CaptureDepth = 0;
unsigned int Decimation = 1;
unsigned int PretriggerMemory = 0;
unsigned int TTLInvert = 0;
unsigned int TriggerMode = NO_TRIGGER;
unsigned int TriggerCh = 0;
unsigned int TriggerSlope = FALLING_EDGE;
unsigned int TriggerThreshold16 = 32768;
unsigned int TriggerThreshold14 = 8192;
unsigned int TriggerThreshold12 = 2048;
unsigned int TriggerHysteresis = 100;
unsigned int NumAverages = 0;
unsigned int AveragerLength = 4096;
unsigned int Fiducial = 0;
unsigned int forceCal = 0;
double Frequency = 0.0;

char * user_outputfile = NULL;
int early_exit;

bool UseLargeMem = false;

int main(int argc, char ** argv)
{
	std::cout << "acquire v1.0" << std::endl << std::endl;

    unsigned long overruns;
    int error;
	HANDLE disk_fd = INVALID_HANDLE_VALUE; // disk file handle	
    char output_file[MAX_DEVICES][128] = {"uvdma.dat", "uvdma1.dat", "uvdma2.dat", "uvdma3.dat"};

	// Create a class with convienient access functions to the DLL
    uvAPI *uv = new uvAPI;
	unsigned char * sysMem = NULL;
	// system memory buffer for large acquisitions		
    unsigned char * sysMemBig = NULL;	

	// used for accessing memory above 2GB
    size_t large_alloc_size; 

	unsigned int sysmem_size = 0;

	// Obtain user desired settings
	acquire_parser(argc, argv);

	// Initialize settings    
	if (forceCal)
		uv->setSetupDoneBit(BoardNum,0); // force full setup

//    uv->selClock(BoardNum,InternalClock);

//    std::cout <<"setupboard" << std::endl;
    uv->setupBoard(BoardNum);


//	std::cout  << std::endl  << std::endl <<"set session" << std::endl;
//	fflush(stdout);
	// Write user settings to the board
    acquire_set_session(uv);
//	std::cout  << std::endl << "set session done" << std::endl;
	fflush(stdout);

	// read the clock freq
    unsigned int adcClock = uv->getAdcClockFreq(BoardNum);
	std::cout << " ADC clock freq ~= " << adcClock << "MHz" << std::endl;
	fflush(stdout);

	// Allocate a page aligned buffer for DMA 
    error = uv->X_MemAlloc((void**)&sysMem, DIG_BLOCK_SIZE);
	if (error)
	{
		std::cout << "failed to allocate block buffer" << std::endl;
		return  -1;
	}
/*
 *    if (numBlocksToAcquire > 8192)
	{
		UseLargeMem = true;
		sysmem_size = numBlocksToAcquire - 8192;
        std::cout <<  "Acquiring " << numBlocksToAcquire << " blocks..." << std::endl;
        std::cout <<  sysmem_size << " in system memory. " << 8192 << " on board." << std::endl;

		// allocate a page-aligned buffer for writing data to disk
        // for allocation > 2GB, use a size_t, and make sure to cast the int to a double
        large_alloc_size = (size_t) (DIG_BLOCKSIZE * (double) sysmem_size);
        error = uv->X_MemAlloc((void**)&sysMemBig, large_alloc_size);

		if (error)
		{
			std::cout << "failed to allocate block buffer" << std::endl;
			return  -1;
		}
		else
        {
            std::cout << "Allocated "<< sysmem_size*(DIG_BLOCKSIZE/(1024*1024)) << "MB system buffer\n" << std::endl;
        }

		// Set the number of blocks we initially read from board to be equal to the size of the system memory buffer.
        numBlocksToAcquire = sysmem_size;
        std::cout << "Reading " << numBlocksToAcquire << " blocks into system memory" << std::endl;
    }
*/
	// open the data disk file
    if (user_outputfile == NULL)	// If user did not specify name, use default uvdma.dat
	{
//#ifdef _WINDOWS
        if ((disk_fd = uv->X_CreateFile(output_file[BoardNum])) < 0)
//#else
//		if ((disk_fd = open(output_file[BoardNum],O_WRONLY)) < 0)
//#endif
		{ 
            if(sysMem){uv->X_FreeMem(sysMem);}
			exit(1);
		}
	}
	else	// User specified name obtained from acquire_parser
	{
        if ((disk_fd = uv->X_CreateFile(user_outputfile)) < 0)
		{ 
            if(sysMem){uv->X_FreeMem(sysMem);}
			exit(1);
		}
	}

    uv->SetupAcquire(BoardNum,numBlocksToAcquire);

	//	printf("setupacquire success!\n");
	if(!UseLargeMem)	// Normal acquire < 8192 blocks
	{
		// For each block of data requested
		for	(unsigned int i=0; i<numBlocksToAcquire; i++ )
		{
			// read a block from the board
            uv->X_Read(BoardNum,sysMem,DIG_BLOCK_SIZE);

			// write block to file
//#ifdef _WINDOWS
            error = uv->X_Write(disk_fd, sysMem, DIG_BLOCKSIZE);
//#else
//            error = write(disk_fd, sysMem, DIG_BLOCKSIZE);
//#endif

            if (i%100 == 0) {
                std::cout << "Reading Block: " << i << "\r";
                fflush(stdout);
            }
            if (i%10000 == 0)
            {
                overruns = uv->getOverruns(BoardNum);
//                std::cout << "\noverruns  = " << overruns << std::endl;
            }
        }
	}
	else	// UseLargeMem acquire
	{
		// For each block of data requested
		for	(unsigned int i=0; i<numBlocksToAcquire; i++ )
		{
			// read block from board into next part of large system buffer
            // use a size_t and cast the int into a double (for >2GB) into the buffer
            large_alloc_size = (size_t) (DIG_BLOCKSIZE * (double) i); 
            error = uv->X_Read(BoardNum,  sysMemBig+large_alloc_size, DIG_BLOCKSIZE);

			// Get out of the loop if ctrl-c is recieved
			if(early_exit==1){i=numBlocksToAcquire+1;} 
		}

		// Write data stored in system memory to disk (the first data acquired)
        // For each block of data stored in system memory
        for(unsigned int i=0; i<numBlocksToAcquire; i++)
        {

            printf("\rWriting Block: %d",(i+1));                        
            fflush(stdout);

            // write block to file
            large_alloc_size = (size_t) (DIG_BLOCKSIZE * (double) i); 
            error = uv->X_Write(disk_fd, sysMemBig+large_alloc_size, DIG_BLOCKSIZE);

            // Get out of the loop if ctrl-c is recieved
            if(early_exit==1){i=numBlocksToAcquire+1;} 
        }

		// Write data still on board to disk
        // For each block of data stored in system memory
        for(unsigned int i=0; i<8192; i++)
        {

            // read block from board into small system buffer 
            error = uv->X_Read(BoardNum, sysMem, DIG_BLOCKSIZE);

            std::cout << "\rWriting Block: " << (i+1+numBlocksToAcquire);
            fflush(stdout);

            // write data block to file
            error = uv->X_Write(disk_fd, sysMem, DIG_BLOCKSIZE);
            
            // Get out of the loop if ctrl-c is recieved
            if(early_exit==1){i=numBlocksToAcquire+1;} 
        }
	}

	std::cout << std::endl;
    overruns = uv->getOverruns(BoardNum);

    std::cout << "done: " <<  overruns << " overruns occurred on Board " << BoardNum << std::endl;

    std::cout << std::endl;


	if(disk_fd)
    {
        uv->X_Close(disk_fd);    // Close the disk file
    }

	// deallocate resources
    if(sysMem){uv->X_FreeMem(sysMem);}
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
                    arg_index++; // increment the arguement index b/c we have now taken care of the next arguement
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
                InternalClock = CLOCK_INTERNAL;
                std::cout << "internal clock selected" << std::endl;
			}
			else if ( strcmp(argv[arg_index], "-freq") == 0 )
			{
				// make sure option is followed by (Chan N)
                if(argc>(arg_index+1))
                {
                    arg_index++; // increment the arguement index b/c we have now taking care of the next arguement
                    Frequency= atoi(argv[arg_index]);
					if (Frequency<=2000000000 && Frequency >= 50000000){
						std::cout << "Internal clock frequency " << Frequency << std::endl;
					}
 					else
					{
						std::cout << "Frequency selected must be between 300MHz and 2GHz for AD12 and 50MHz and 250MHz for AD16" << std::endl;
						exit(1);
					}
               }
				else
				{
                    std::cout << "Frequency selected must be between 300MHz and 2GHz for AD12 and 50MHz and 250MHz for AD16" << std::endl;
                    exit(1);
				}
			}

			else if ( strcmp(argv[arg_index], "-f") == 0 )
			{
				if(argc>(arg_index+1))
                {
                    arg_index++; // increment the arguement index b/c we have now taken care of the next arguement
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
                    arg_index++; // increment the arguement index b/c we have now taken care of the next arguement
                    SingleChannelSelect= atoi(argv[arg_index]);
                    std::cout << "SingleChannelSelect " << SingleChannelSelect << std::endl;
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
				SingleChannelMode = 1;
				SingleChannelSelect = 0;
            }
            else if ( strcmp(argv[arg_index], "-desclkiq") == 0 )
            {
                DESCLKIQ = 1;
				SingleChannelMode = 1;
				SingleChannelSelect = 0;
            }
            else if ( strcmp(argv[arg_index], "-ecltrig") == 0 )
            {
                ECLTrigger = 1;
            }
            else if ( strcmp(argv[arg_index], "-ecldelay") == 0 )
            {
                if(argc>(arg_index+1))
                {
                    arg_index++; // increment the arguement index b/c we have now taken care of the next arguement
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
                    arg_index++; // increment the arguement index b/c we have now taken care of the next arguement
					DualChannelSelect= atoi(argv[arg_index]);
					firstchan = DualChannelSelect / 10;
					secondchan = DualChannelSelect % 10;
					chans = (1 << firstchan) + (1 << secondchan);	//bitwise representation of channels
//					printf("dualchannelselect=%d\n",DualChannelSelect);
//					printf("Channels chosen = %d and %d\n",firstchan,secondchan);
//					printf("chans = %d\n",chans);
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
                    arg_index++; // increment the arguement index b/c we have now taken care of the next arguement
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
                    arg_index++; // increment the arguement index b/c we have now taken care of the next arguement
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
                    arg_index++; // increment the arguement index b/c we have now taken care of the next arguement
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
                    arg_index++; // increment the arguement index b/c we have now taken care of the next arguement
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
                    arg_index++; // increment the arguement index b/c we have now taken care of the next arguement
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
                    arg_index++; // increment the arguement index b/c we have now taken care of the next arguement
                    TriggerThreshold12  = atoi(argv[arg_index]);
					TriggerThreshold16 = TriggerThreshold12;	//this may need a better scheme
					TriggerThreshold14 = TriggerThreshold12;	//this may need a better scheme
                }
                else
                {
                    std::cout << "thresh_a must be 0 to 65535 (16bit)" << std::cout;
                    std::cout << "thresh_a must be 0 to 16383 (14bit)" << std::cout;
                    std::cout << "thresh_a must be 0 to 4095 (12bit)" << std::cout;
                    exit(1);
                }
            }

            else if( strcmp(argv[arg_index], "-hysteresis") == 0 )
            {
                if(argc>(arg_index+1))
                {
                    arg_index++; // increment the arguement index b/c we have now taken care of the next arguement
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
                    arg_index++; // increment the arguement index b/c we have now taken care of the next arguement
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

    pUV->selClock(BoardNum, InternalClock);
	int defaultchannels = pUV->GetAllChannels(BoardNum);
//	printf("All channels = %d\n",defaultchannels);
	if (pUV->IS_ISLA216P(BoardNum))
    {
        if (pUV->HAS_microsynth(BoardNum) && Frequency!=0){
			if (Frequency<=250000000){
				pUV->MICROSYNTH_freq(BoardNum, Frequency);
			}
			else{
				printf("Frequency outside range 50MHz-250MHz, using previous frequency\n");
			}
		}
        unsigned int selectedChannels = 0;

        if (SingleChannelMode == 1)
        {
            switch (SingleChannelSelect) {
            case 0:
                selectedChannels = IN0;
                break;
            case 1:
                selectedChannels = IN3;  // due to part poulation on 2ch verion ch1 is on IN3
                break;
            case 2:
                selectedChannels = IN1;
                break;
            case 3:
                selectedChannels = IN2;
                break;
            default:
                selectedChannels = IN0;
                break;
            }
        }
        else if (DualChannelMode == 1 || defaultchannels == 2)
        {
            selectedChannels = (chans);
        }
        else if (defaultchannels == 4)
        {
            selectedChannels = (IN0 | IN1 | IN2 | IN3);
        }
		else {
			printf("channel info not found, exiting\n");
			exit(1);
		}
//        std::cout << std::endl << std::endl << BoardNum << "AD16 setup selected channels= " <<  selectedChannels << "scs=" << SingleChannelSelect << std::endl << std::endl;
        pUV->selectAdcChannels(BoardNum, selectedChannels);

        unsigned int TriggerChannel;
        switch (TriggerCh) {
        case 0:
            TriggerChannel = IN0;
            break;
        case 1:
            TriggerChannel = IN3;  // due to part poulation on 2ch verion ch1 is on IN3
            break;
        case 2:
            TriggerChannel = IN1;
            break;
        case 3:
            TriggerChannel = IN2;
            break;
        default:
            TriggerChannel = IN0;
            break;
        }
//		fflush(stdout);
//		pUV->selectTrigger(BoardNum, WAVEFORM_TRIGGER,FALLING_EDGE , IN0);
        pUV->selectTrigger(BoardNum, TriggerMode, TriggerSlope, TriggerCh);			//this might need to be changed to "TriggerCh" as was changed on 12bit
        pUV->configureWaveformTrigger(BoardNum, TriggerThreshold16, TriggerHysteresis);
        pUV->configureSegmentedCapture(BoardNum, CaptureCount, CaptureDepth, 0);
		if (NumAverages > 0){
			pUV->configureAverager(BoardNum, NumAverages, AveragerLength, 0);
		}
        pUV->setFiducialMarks(BoardNum, Fiducial);
	unsigned int trigVal = pUV->isTriggerEnabled(BoardNum);
//	std::cout << "trigVal" << trigVal << std::endl;

    }
    if (pUV->IS_adc12d2000(BoardNum))
    {
        if (pUV->HAS_microsynth(BoardNum) && Frequency!=0){
			if (Frequency>=300000000){
				pUV->MICROSYNTH_freq(BoardNum, Frequency);
			}
			else{
				printf("Frequency outside range 300MHz-2GHz, using previous frequency\n");
			}
		}
		unsigned int selectedChannels = 0;
		if (SingleChannelMode == 1)	// One channel mode
        {
			if (DESIQ == 1){
				selectedChannels = 8;
			}
			else if (DESCLKIQ == 1){
				selectedChannels = 4;
			}
			else if (SingleChannelSelect == 0){
				selectedChannels = 1;
			}
			else if (SingleChannelSelect == 1){
				selectedChannels = 2;
			}
			else{
				selectedChannels = 1;
			}
			pUV->selectAdcChannels(BoardNum, selectedChannels);
		}
        else
        {
            pUV->selectAdcChannels(BoardNum, IN0 | IN1);	// Two channel mode
			printf("two channel mode\n");
		}

		unsigned int TriggerChannel;
		//printf("triggerCh:%d\n",TriggerCh);
		//printf("triggerCh:%d\n",TriggerCh);
        switch (TriggerCh) {
        case 0:
            TriggerChannel = IN0;
            break;
        case 1:
            TriggerChannel = IN3;  // due to part poulation on 2ch verion ch1 is on IN3
            break;
        case 2:
            TriggerChannel = IN1;
            break;
        case 3:
            TriggerChannel = IN2;
            break;
        default:
            TriggerChannel = IN0;
            break;
        }
		
		// Set ECL Trigger Delay
        pUV->SetECLTriggerDelay(BoardNum, ECLTriggerDelay);
        // Set Decimation
        pUV->setAdcDecimation(BoardNum, Decimation);
        // Set ECL Trigger
 //       std::cout << "ECLTrigger=" << ECLTrigger << std::endl;
        pUV->SetECLTriggerEnable(BoardNum, ECLTrigger);
		
		//printf("boardnum=%d triggermode=%d triggerslope=%d triggerch=%d triggerthreshold12=%d triggerhysteresis=%d\n",BoardNum,TriggerMode,TriggerSlope,TriggerCh,TriggerThreshold12,TriggerHysteresis);
		//printf("capturecount=%d capturedepth=%d numaverages=%d averagerlength=%d pretrigger=%d\n",CaptureCount,CaptureDepth,NumAverages,AveragerLength,PretriggerMemory);
		pUV->selectTrigger(BoardNum, TriggerMode, TriggerSlope, TriggerCh); //not using "TriggerChannel" and instead using "TriggerCh".
		pUV->configureWaveformTrigger(BoardNum, TriggerThreshold12, TriggerHysteresis);
        pUV->configureSegmentedCapture(BoardNum, CaptureCount, CaptureDepth, 0);
		if (NumAverages > 0){
			if (NumAverages > 64){
				NumAverages = 64;
				printf("!!CAUTION!!: Averages reduced to maximum for AD12 (64)\n");
			}
			pUV->configureAverager(BoardNum, NumAverages, AveragerLength, 0);
		}
	pUV->setFiducialMarks(BoardNum, Fiducial);
	//printf("finished AD12 section\n");
	//Sleep(10000);
	}

	if (pUV->IS_AD5474(BoardNum)){
		printf("AD14 found\n");
		unsigned int selectedChannels = 0;

		if (SingleChannelMode == 1)	// One channel mode
		{
			printf("Setting board to 1 channel mode. ");

			if (SingleChannelSelect == 0)
			{
				printf("Acquire IN0.\n");
				selectedChannels = IN0;
			}
			else if (SingleChannelSelect == 1)
			{
				printf("Acquire IN1.\n");
				selectedChannels = IN1;
			}
			else
			{
				printf("Invalid channel. Defaulting to acquire IN0.\n");
				selectedChannels = IN0;
			}

		pUV->selectAdcChannels(BoardNum, selectedChannels);
	}
	else
	{
                selectedChannels = 3;		//1 || 2 = 3
                printf("Setting board to 2 channel mode\n");
                pUV->selectAdcChannels(BoardNum, selectedChannels);
	}

	// Configure Trigger
	pUV->selectTrigger(BoardNum, TriggerMode, TriggerSlope, TriggerCh);

	//Configure Waveform Trigger
	pUV->configureWaveformTrigger(BoardNum, TriggerThreshold14, TriggerHysteresis);

	// Configure Segmented Capture
	pUV->configureSegmentedCapture(BoardNum, CaptureCount, CaptureDepth, 0);

	// Configure Averager
	pUV->configureAverager(BoardNum, NumAverages, AveragerLength, 0);

	// Set Decimation
	pUV->setAdcDecimation(BoardNum, Decimation);
	
	// Set Fiducial Marks
	pUV->setFiducialMarks(BoardNum, Fiducial);

	}

        if (!pUV->IS_adc12d2000(BoardNum) && !pUV->IS_ISLA216P(BoardNum) && !pUV->IS_AD5474(BoardNum)){
		printf("AD8 found\n");


                unsigned int selectedChannels = 0;

		if (SingleChannelMode == 1){	// One channel mode
			if (SingleChannelSelect == 1){
				selectedChannels = IN1;
			}
			else{
			selectedChannels = IN0;
			}
		}
	else{
		selectedChannels = 3;
	}
	//pUV->selectAdcChannels(BoardNum, selectedChannels);
        pUV->selectAdcChannels(BoardNum, selectedChannels);
	// Set ECL Trigger Delay
        pUV->SetECLTriggerDelay(BoardNum, ECLTriggerDelay);
        // Set Decimation
        pUV->setAdcDecimation(BoardNum, Decimation);
        // Set ECL Trigger
        pUV->SetECLTriggerEnable(BoardNum, ECLTrigger);
	// Configure Segmented Capture
	pUV->configureSegmentedCapture(BoardNum, CaptureCount, CaptureDepth, 0);
	}

    //       pUV->SetAdcDecimation(BoardNum, Decimation);
    pUV->SetTTLInvert(BoardNum, TTLInvert);
    pUV->setPreTriggerMemory(BoardNum, PretriggerMemory);
}

void acquire_parser_printf()
{
    printf("\n");
    printf("acquire will setup the requested board with desired settings, initiate\nacquisition and store data to disk file. \n\nUsage:\n\n");
    printf("acquire (N blocks) [OPTIONS]\n");
	printf("\t\t\tAcquires specified number of blocks. Number of blocks option must always be first.\n\n");
  
    printf("The following [OPTIONS] may follow the number of blocks option in any order:\n\n");    

    printf("-f (filename)\t\tUse this argument to specify the name of the file to write. Must append .dat at the end of filename\n");
	printf("\n");
	printf("-ic\t\t\tBoard will use the internal clock. If not specified board uses external clock.\n");
	printf("-freq\t\t\tSpecifies the internal clock frequency for boards equipped with a microsynth programmable oscillator\n");
	printf("\t\t\t300MHz-2GHz for AD12, 50MHz-250MHz for AD16\n");
	printf("\n");
    printf("-scm\t\t\tRun multi-channel boards in single channel mode. Not required for single channel boards.\n");
    printf("-scs (Chan N)\t\tWhen running multi-channel boards in single channel mode use channel (Chan N).\n");
	printf("-desiq\t\t\tdesiq mode for AD12 board.\n");
	printf("-desclkiq\t\tdesclkiq mode for AD12 board.\n");
	printf("\n");
	printf("-dcm\t\t\tRun quad channel boards in dual channel mode. Not required for dual channel boards.\n");
	printf("-dcs <NM>\t\tWhen running in dual channel mode, specify the channels to acquire from. \n\t\t\tWhere channels NM can be 01, 02, 03, 12, 13, 23.\n");
	printf("\n");
    printf("-b (BoardNum)\t\tAcquires from the specified board. Do not specify the same board\n\t\t\tfrom two separate consoles. BoardNum starts at 0.\n");
	printf("\n");
	printf("-forceCal\t\tRuns full setup and re-reads calibration info from ultra_config.dat.\n");
	printf("-hdiode\t\t\tHeterodyne trigger.\n");
	printf("-syncselrecord\t\tSelective recording.\n");
	printf("-ttledge\t\tSets acquisition to await TTL trigger edge.\n");
	printf("-ttlinv\t\t\tInverts the selective recording feature. Default is active LOW.\n");
	printf("-ecltrig\t\tAcquire once ecl trigger has been received.\n");
	printf("-ecldelay (N)\t\tSet the ECL trigger delay. Default = 8000 (64us). See manual for details.\n");
    	printf("-dec (factor)\t\tEnables input sample decimation. (factor) can be 1,2,4,8 or 16. (AD8/AD12/AD16)\n");
	printf("-capture_count (X)\tEnable segmented captures with X number of records of length Y.\n");
	printf("-capture_depth (Y)\tEnable segmented captures with X number of records of length Y. Y must be a multiple of 8 samples.\n");
	printf("-analog\t\t\tFor analog waveform trigger. Use -ttlinv to specify rising or falling edge.\n");
	printf("-analog_ch\t\tanalog waveform trigger channel");
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
