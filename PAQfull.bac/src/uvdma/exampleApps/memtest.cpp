////////////////////////
//  uv_memtest.cpp
//  Ultraview AD83000 series DMA tester
//
//
//////////////////////// 

#include <iostream>
#include "uvAPI.h"
//#include "Pcie5vDefines.h"

void INTHandler(int sig);
unsigned char GetWalkingPattern(int i);
unsigned char GetModuloPattern(int i);

void memtest_parser_printf();
void memtest_parser(int argc, char ** argv);

unsigned short BoardNum = 0;
HANDLE pBoardNum;
int early_exit;
unsigned int numBlocksToAcquire = 1;
bool WriteToFile;
bool PrintBlocks;
bool CreateTestPattern;
bool CompareReRead;
bool DmaTestMode;
enum _TestPatternType { DRAMCalPattern, WalkingOnes, Modulo, Triangle} TestPatternType;
enum _TestMode { DRAMCalMode, DRAMTestMode} TestMode;

//#define PRINT_FIRST_DATA
#define PAGE_SIZE       4096
#define MAX_BLOCKS      8192
#define samples_per_block	262144
#define PAGES_PER_BLOCK (DIG_BLOCK_SIZE/PAGE_SIZE)

// @ 333MHz DQS delay of 0-,3 are best, DQ delays of 6,7,8,9 are best
#define MIN_DQS_DELAY           0   // 0   // less than 2 has lots of errors, 5-9 are all about the same
#define MAX_DQS_DELAY           9   // 1   // 64 max,
#define MIN_DQ_DELAY            0   // 8   // higher than 18 has lots of errors, 7-14 seem to be the best
#define MAX_DQ_DELAY            15   // 9  // 64 max, 

#define NUM_DRAM_DQ             128

void ResetDRAMCalibration(unsigned long ****dq_error_hist);
bool AllocateDRAMHist(unsigned long ****dq_error_hist);
void FreeDRAMHist(unsigned long ****dq_error_hist);
bool AllocateMismatchArray(bool ***mismatch);
void FreeMismatchArray(bool ***mismatch);
void ResetBoardPointers(uvAPI * pUV);

HANDLE testfile, output_file;


void RunDRAMMemTest(uvAPI * pUV, void *pBuffer, void *pReadData);
void RunDRAMDelayCal(uvAPI * pUV, void *pBuffer, void *pReadData);
void CheckDRAMData(uvAPI * pUV, void *pBuffer, void *pReadData, int dqs_delay, int dq_delay, unsigned long ****dq_error_hist);


int num_byte_errors;


int main(int argc, char ** argv)
{

    // Create a class with convienient access functions to the DLL
    uvAPI *uv = new uvAPI;


    int dma_alloc_error;	
    unsigned long i;

    // disk file handle 
    //HANDLE disk_fd;					
    // used for accessing memory above 2GB
    //size_t large_alloc_size;  

    
    // Memory buffers for DMA operations, these are declared as void * so that the user programs work with the different
    // resolution and number of channel ADC boards without needing to be recompiled.
    void * pBuffer = NULL;	    // bounce buffer for writes/read	    
    void * pReadData = NULL;    // second buffer for comparing memory
          
    // In this application we want to check DRAM operation on a per bytes basis (e.g. regardless of resolution and number of
    // of channels on the board), so we can cast these to byte arrays to make this application simpler.
    unsigned char * pBufferChar;    
    unsigned char * pReadDataChar;





    // Install INTHandler as our software interrupt handler- Linux only.
#ifndef _WINDOWS
    signal( SIGINT, INTHandler);	
#endif
    early_exit=0;



    memtest_parser(argc, argv);   // Parse the command line input


   
    // If we are writing the data read back from the board's DRAM to disk then create the file
    if(WriteToFile)
    {
        printf("Writing to file. \n");
        output_file = uv->X_CreateFile("memtest.dat");
    }

    // allocate a page-aligned buffer for writing data to disk	
    dma_alloc_error = uv->X_MemAlloc((void**)&pBuffer, DIG_BLOCK_SIZE);

    if((dma_alloc_error) || (pBuffer == NULL))
    {
        printf("Memory allocation failed!\n");
        exit(1);
    }

    // allocate a page-aligned buffer for reading data from disk
    dma_alloc_error = uv->X_MemAlloc((void**)&pReadData, DIG_BLOCK_SIZE);

    if((dma_alloc_error) || (pReadData == NULL))
    {
        printf("Memory allocation failed!\n");
        exit(1);
    }    

    pBufferChar = (unsigned char *) pBuffer;
    pReadDataChar = (unsigned char *) pReadData;




    // Setup the board specified by ss.board_num for acquisition. SetupBoard will return false if the setup failed.
    if(!uv->setupBoard(BoardNum))
    {
        uv->X_FreeMem(pBuffer);
        uv->X_FreeMem(pReadData);
        exit(1);
    }    
  
	// Force SEL_WRITE_WHOLE_ROW false as pcie -> ddr won't work in this mode!
//	uv->SetWriteWholeRow(BoardNum, false);

    // If we are writing a test pattern to the board then create it now
    if(CreateTestPattern)
    {
        for(i=0; i<DIG_BLOCK_SIZE; i++)
        {
            if(TestPatternType == WalkingOnes)
            {
                pBufferChar[i] = GetWalkingPattern(i);
            }
            else if (TestPatternType == Modulo)
            {
                pBufferChar[i] = GetModuloPattern(i);
            }
            else if (TestPatternType == Triangle)
            {
                pBufferChar[i] = i%256;
            }
            else if (TestPatternType == DRAMCalPattern)
            {
                pBufferChar[i] = 0;
            }
        }
    }



   

    ResetBoardPointers(uv);   // Reset the board's pointers. E.g. DRAM address pointers


    // Run the DRAM delay calibration routine
    if(TestMode == DRAMCalMode)
    {        
        RunDRAMDelayCal(uv, pBuffer, pReadData);
    }
    // Run the DRAM memory test routine
    else
    {
        RunDRAMMemTest(uv, pBuffer, pReadData);
    }


    if(WriteToFile)
    {
        uv->X_Close(output_file);
    }


    //dll.hDllCheckDeviceStatus();

    

    //x_UnloadDll(&dll);       
    

    if(pBuffer){ uv->X_FreeMem(pBuffer); }
    if(pReadData){ uv->X_FreeMem(pReadData); }

    // if we recieved SIGINT (LINUX only) then terminate the calling process after allowing our acquisition to finish	
    if(early_exit==1)
    {
        printf("SIGINT exiting\n");
#ifndef _WINDOWS
        signal(SIGINT, SIG_DFL);	// specify the default action
        kill(getpid(), SIGINT);		// send SIGINT to calling process
        exit(0);
#endif
    }
	

    return 0;
}


















void INTHandler(int sig)
{
#ifndef _WINDOWS
    // ignore the interrupt, we will handle it
    signal(sig, SIG_IGN);	
    // flag acquisition loop to exit on after finishing current interation
    early_exit = 1;		
#endif
}




unsigned char GetWalkingPattern(int i)
{
    int j = i % 20;

    if(j==0){return 0x00;}
    if(j==1){return 0xFF;}
    if(j==2){return 0x01;}
    if(j==3){return 0xFE;}
    if(j==4){return 0x02;}
    if(j==5){return 0xFD;}
    if(j==6){return 0x04;}
    if(j==7){return 0xFB;}
    if(j==8){return 0x08;}
    if(j==9){return 0xF7;}
    if(j==10){return 0x10;}
    if(j==11){return 0xEF;}
    if(j==12){return 0x20;}
    if(j==13){return 0xDF;}
    if(j==14){return 0x40;}
    if(j==15){return 0xBF;}
    if(j==16){return 0x80;}
    if(j==17){return 0x7F;}
    if(j==18){return 0xFF;}
    if(j==19){return 0x00;}

    return 0x00;

}




// Credit for this code idea goes to BradyTech.com; the people who wrote Memtest86.
unsigned char GetModuloPattern(int i)
{
    int j = i % 10;
    int x = i % 100;

    if(x<10)
    {
        if(j==0){return 0x11;}
        else{return 0xEE;}
    }
    else if (x<20)
    {
        if(j==1){return 0x03;}
        else{return 0xF8;}
    }
    else if (x<30)
    {
        if(j==2){return 0xAA;}
        else{return 0x55;}
    }
    else if (x<40)
    {
        if(j==3){return 0xA3;}
        else{return 0x58;}
    }
    else if (x<50)
    {
        if(j==4){return 0xA3;}
        else{return 0x58;}
    }
    else if (x<60)
    {
        if(j==5){return 0xFF;}
        else{return 0x00;}
    }
    else if (x<70)
    {
        if(j==6){return 0xF8;}
        else{return 0x03;}
    }
    else if (x<80)
    {
        if(j==7){return 0x21;}
        else{return 0xDE;}
    }
    else if (x<90)
    {
        if(j==7){return 0x48;}
        else{return 0xB3;}
    }
    else if (x<100)
    {
        if(j==7){return 0x57;}
        else{return 0x86;}
    }

    return 0x00;
}




void memtest_parser_printf()
{
    printf("Memtest will DMA data to the on-board DRAM and then read back the same data and check for any errors.\n");
    printf("Usage: \n");
    printf("memtest -cal (-w, -m, -t) \t Run DRAM bit-based cal routine. -cal option must be followed by (-w,-m, or -t option). This option must be first.\n");
    printf("memtest -w (N blocks) \t\t Write then readback a walking ones test pattern. -w option must be followed by number of blocks.\n");
    printf("memtest -m (N blocks) \t\t Write then readback a modulo test pattern. -m option must be followed by number of blocks.\n");
    printf("memtest -t (N blocks) \t\t Write then readback a triangle test pattern. -t option must be followed by number of blocks.\n");
    printf("        -rr option checks if reads are due reading/writing by reading each block twice and comparing the read data.\n");
    printf("        -s option saves the readback data to memtest.dat\n");
    printf("        -p option prints the blocks which mismatched\n");
}



// parse the command line input for options
void memtest_parser(int argc, char ** argv)
{
    int arg_index;

    WriteToFile = false;
    PrintBlocks = false;
    CreateTestPattern = false;
    DmaTestMode = false;        // Set TRUE so that DMA rates will be printed by x_Time functions
    TestMode = DRAMTestMode;            // Set default run mode to fast page based error checking
    CompareReRead = false;

    printf("\n");

	// check how many arguments, if run without arguements prints usage.
    if(argc == 1)
    {
        memtest_parser_printf();
        exit(1);
    }
    else 
    { 
        // starting at second arguement look for options
        for(arg_index=1; arg_index<argc; arg_index++)
        {

            if( strcmp(argv[arg_index], "-cal") == 0 )
            {
                printf("Running DRAM calibration\n");
                TestMode = DRAMCalMode; // Set run mode to run the calibration test
            }
            else if( strcmp(argv[arg_index], "-w") == 0 )
            {
                // make sure option is followed by number of blocks
                if(argc>arg_index)
                {
                    numBlocksToAcquire = atoi(argv[arg_index+1]);
                    arg_index++; // increment the arguement index b/c we have now taking care of the next arguement
                    printf("Using walking ones test pattern, %d blocks\n", numBlocksToAcquire);
                    CreateTestPattern = true;
                    TestPatternType = WalkingOnes;
                }
                if(numBlocksToAcquire <= 0)
                {
                    printf("Failed to parse arguements, exiting!\n"); 
                    exit(1);
                }
            }
            else if( strcmp(argv[arg_index], "-m") == 0 )
            {
                // make sure option is followed by number of blocks
                if(argc>arg_index)
                {
                    numBlocksToAcquire = atoi(argv[arg_index+1]);
                    arg_index++; // increment the arguement index b/c we have now taking care of the next arguement
                    printf("Using modulo test pattern, %d blocks\n", numBlocksToAcquire);
                    CreateTestPattern = true;
                    TestPatternType = Modulo;
                }
                if(numBlocksToAcquire <= 0)
                {
                    printf("Failed to parse arguements, exiting!\n"); 
                    exit(1);
                }
            }
            else if( strcmp(argv[arg_index], "-t") == 0 )
            {
                // make sure option is followed by number of blocks
                if(argc>arg_index)
                {
                    numBlocksToAcquire = atoi(argv[arg_index+1]);
                    arg_index++; // increment the arguement index b/c we have now taking care of the next arguement
                    printf("Using triangle test pattern, %d blocks\n", numBlocksToAcquire);
                    CreateTestPattern = true;
                    TestPatternType = Triangle;
                }
                if(numBlocksToAcquire <= 0)
                {
                    printf("Failed to parse arguements, exiting!\n"); 
                    exit(1);
                }
            }
            else if( strcmp(argv[arg_index], "-rr") == 0 )
            {
                printf("Running comparision of re-read data\n");
                CompareReRead = true;
            }
            else if( strcmp(argv[arg_index], "-s") == 0 )
            {
                printf("Readback data will be saved to memtest.dat\n");
                WriteToFile = true;
            }
            else if( strcmp(argv[arg_index], "-p") == 0 )
            {
                printf("Mismatched blocks will be printed\n");
                PrintBlocks = true;
            }
            else
            {
                // if we get to here then we didn't detect any options at all!
                printf("No options specified! Check syntax.\n");
                exit(1);
            }

        }

    }



}





// 010711, Reset variables for DRAM delay calibration
bool AllocateDRAMHist(unsigned long ****dq_error_hist)
{
    int dqs_d, dq_d;

    // allocate the DQS delay array
    (*dq_error_hist) = (unsigned long ***) malloc(MAX_DQS_DELAY*sizeof(unsigned long *));

    if((*dq_error_hist) == NULL)
    {
        printf("Error allocating DRAM DQS delay array, exiting!\n");
        return true;
    }

    // each DQS delay array element points to an array of DQ delay elements
    for(dqs_d=0; dqs_d<MAX_DQS_DELAY; dqs_d++)
    {
        (*dq_error_hist)[dqs_d] = (unsigned long **) malloc(MAX_DQ_DELAY*sizeof(unsigned long *));
        
        if((*dq_error_hist)[dqs_d] == NULL)
        {
            printf("Error allocating DRAM DQ delay subarray, exiting!\n");
            return true;
        }
    }

    // each pair of (DQS,DQ delays) points to an array of error elements (one element for each DRAM bit) 
    for(dqs_d=0; dqs_d<MAX_DQS_DELAY; dqs_d++)
    {
        for(dq_d=0; dq_d<MAX_DQ_DELAY; dq_d++)
        {
        
            (*dq_error_hist)[dqs_d][dq_d] = (unsigned long *) malloc(NUM_DRAM_DQ*sizeof(unsigned long));
        
            if((*dq_error_hist)[dqs_d][dq_d] == NULL)
            {
                printf("Error allocating DRAM (DQS,DQ) error subarray, exiting!\n");
                return true;
            }
        }
    }

    return false;
}



void FreeDRAMHist(unsigned long ****dq_error_hist)
{
    
    int dqs_d, dq_d;

    // each pair of (DQS,DQ delays) points to an array of error elements (one element for each DRAM bit) 
    for(dqs_d=0; dqs_d<MAX_DQS_DELAY; dqs_d++)
    {
        for(dq_d=0; dq_d<MAX_DQ_DELAY; dq_d++)
        {
            if((*dq_error_hist)[dqs_d][dq_d] != NULL)
            {
                free((*dq_error_hist)[dqs_d][dq_d]);
            }
        }
    }

    // each DQS delay array element points to an array of DQ delay elements
    for(dqs_d=0; dqs_d<MAX_DQS_DELAY; dqs_d++)
    {
        if((*dq_error_hist)[dqs_d] != NULL)
        {
            free((*dq_error_hist)[dqs_d]);
        }
    }

    if((*dq_error_hist) != NULL)
    {
        free((*dq_error_hist));
    }
}





// 010611, Reset variables for DRAM delay calibration
void ResetDRAMCalibration(unsigned long ****dq_error_hist)
{
    int dqs_d, dq_d, dq_i;

    for(dqs_d=0; dqs_d<MAX_DQS_DELAY; dqs_d++)
    {
        for(dq_d=0; dq_d<MAX_DQ_DELAY; dq_d++)
        {
            for(dq_i=0; dq_i<NUM_DRAM_DQ; dq_i++)
            {
                (*dq_error_hist)[dqs_d][dq_d][dq_i] = 0;
            }
        }
    }
}




bool AllocateMismatchArray(bool ***mismatch)
{
    int page_i;

    // allocate the page array
    (*mismatch) = (bool **) malloc(PAGES_PER_BLOCK*sizeof(bool *));

    if((*mismatch) == NULL)
    {
        printf("Error allocating page error array!\n");
        return true;
    }

    // each DQS delay array element points to an array of DQ delay elements
    for(page_i=0; page_i<PAGES_PER_BLOCK; page_i++)
    {
        (*mismatch)[page_i] = (bool *) malloc(MAX_BLOCKS*sizeof(bool));
        
        if((*mismatch)[page_i] == NULL)
        {
            printf("Error allocating block error subarray!\n");
            return true;
        }
    }

    return false;
}



void FreeMismatchArray(bool ***mismatch)
{
    int i;

    // Free the mismatch array
    for(i=0; i<PAGES_PER_BLOCK; i++)
    {
        if((*mismatch)[i] != NULL)
        { 
            free((*mismatch)[i]);
        }
    }

}




void ResetBoardPointers(uvAPI * pUV)
{
//    pUV->DllApiSetPreventUnderOverRuns(BoardNum, false);     // Allow the device driver to read/write data to the board without regard for over/underruns.
    //dll.hDllApiSetPioRegister(PIO_OFFSET_BLOCKSIZE, DIG_BLOCKSIZE);     // Have to do this for D/A DMA
//	pUV->SET_BLOCKSIZE(BoardNum, DIG_BLOCK_SIZE);

//	pUV->SET_PCIE_RD_START_BLOCK(BoardNum, 0);
//	pUV->SET_PCIE_RD_END_BLOCK(BoardNum, 8191);

//	pUV->DllApiSetShadowRegister(BoardNum, PIO_OFFSET_DAC_WR_START_BLOCK, 0, 0, 0);
//	pUV->DllApiSetShadowRegister(BoardNum, PIO_OFFSET_DAC_WR_END_BLOCK, 8191, 0, 0);
//	pUV->DllApiSetShadowRegister(BoardNum, PIO_OFFSET_DAC_RD_START_BLOCK, 0, 0, 0);
//	pUV->DllApiSetShadowRegister(BoardNum, PIO_OFFSET_DAC_RD_END_BLOCK, 8191, 0, 0);
	
    //dll.hDllApiSetPioRegister(PIO_OFFSET_PCIE_RD_START_BLOCK, 0 );                      // Start PCIe DRAM reads at block 0	
	//dll.hDllApiSetPioRegister(PIO_OFFSET_PCIE_RD_END_BLOCK, pSS->blocks_on_board-1 );   // Wrap PCIe DRAM reads at last block

//	pUV->SetContinuousAutoRefresh(BoardNum, true);
    //dll.hDllSetContinuousARF(pSS, true);         // Over-ride DLL value to ensure this is set.

}











void RunDRAMMemTest(uvAPI * pUV, void *pBuffer, void *pReadData)
{

    // instantiate the time variable
    //TIME_VAR_TYPE SysTime;

    unsigned long sample, sampleread;

    int dqs_delay, dq_delay, dq_index;
    unsigned long DRAMAdjReg = 0;
    unsigned int page_index, block_index, byte_index;
    int error;

    int num_mismatch=0;
    int num_match=0;
    int num_mismatch_t=0;
    bool flag_block;


    // Error array for checking each page of each block for errors
    bool **mismatch;
    bool page_alloc_error = false;


    // If we are going to test DRAM using write/read test then allocate the array for storing mismatch data 
    page_alloc_error = AllocateMismatchArray(&mismatch);

    if(page_alloc_error)
    {
        printf("Error allocating mismatch array, Exiting!\n");
        FreeMismatchArray(&mismatch);
        exit(1);
    }



    if(DmaTestMode)
    {
        printf("Running DMA test, will not check readback data for errors!\n");
    }
    printf("\n");


    // Run the calibration type test here...
    //for(dqs_delay=MIN_DQS_DELAY; dqs_delay<MAX_DQS_DELAY; dqs_delay++)
    //{
    //    for(dq_delay=MIN_DQ_DELAY; dq_delay<MAX_DQ_DELAY; dq_delay++)
    //    {
    //        if(early_exit==1){ dq_delay=MAX_DQ_DELAY; } // if ctrl-C was recieved in Linux then exit this loop 
    //dqs_delay = DRAM_DQS_IDELAY_VAL;
    //dq_delay = DRAM_DQ_IDELAY_VAL;
    //dqs_delay = pUV->GetDRAMDQSValue(BoardNum);
  //  dq_delay = pUV->GetDRAMDQValue(BoardNum);

            //ResetBoardPointers(pSS);   // Reset the board's pointers. E.g. DRAM address pointers

            //dll.hDllDramIdelayShift(pSS->dram_dqs, pSS->dram_dq);  // Set the new delay values
    //		pUV->DramIdelayShift(BoardNum, dqs_delay, dq_delay);
            
            num_match=0;
            num_mismatch=0;

            printf("(dqs_delay=%d, dq_delay=%d)\n", dqs_delay, dq_delay);
        

    // Get time for throughput measurements
    //x_StartTime(&SysTime);

	//pBoardNum = pUV->DllApiGetCurrentDeviceHandle();

    // Write each block of data to the board
    for(block_index=0; block_index < numBlocksToAcquire; block_index++)
    {
        if(!DmaTestMode){ printf("\rWriting Block: %d", block_index); fflush(stdout); }
		{

//            // If we are reading data from a file read it into the bounce buffer first
//            error = x_Read( testfile, pBuffer, DIG_BLOCKSIZE);

        // Write the data to the board
        error = pUV->X_Write(BoardNum, pBuffer, DIG_BLOCK_SIZE);
		}

        if(early_exit==1){ block_index=numBlocksToAcquire+1; } 
    }

    // End time for throughput measurements.
    //x_EndTime(&SysTime, pSS);


//  x_FileSeek(testfile, (off_t) 0);
//	SLEEP(1); // Wait for a while to test DRAM refresh
    printf("\n");




    // Get time for throughput measurements
    //x_StartTime(&SysTime);

    // For each block of data requested
    for(block_index=0; block_index < numBlocksToAcquire; block_index++)
    {
        if(!DmaTestMode){ printf("\rReading/Checking Block: %d ",block_index); fflush(stdout); }
		//printf("print1\n");


//  error = x_Read( testfile, pBuffer, DIG_BLOCKSIZE);


        // Read the data from the board into second memory buffer
        error = pUV->X_Read(BoardNum, pReadData, DIG_BLOCK_SIZE);
		//printf("print2\n");
		Sleep(1);


        if(!DmaTestMode)
        {
            if(WriteToFile)
            {
                error = pUV->X_Write(output_file, pReadData, DIG_BLOCK_SIZE);
				//printf("print3\n");

            }

            num_mismatch_t = 0;
        
            for(page_index=0; page_index < PAGES_PER_BLOCK; page_index++)
            {   
                // Check each page of each block, just check them as char arrays because it doesn't matter
                error = memcmp( ((unsigned char *)pBuffer)+(page_index*PAGE_SIZE), ((unsigned char *)pReadData)+(page_index*PAGE_SIZE), PAGE_SIZE );
				//printf("print4\n");
                if( error != 0 )
                {
                    num_mismatch_t = 1;
                    mismatch[page_index][block_index] = true;
                }
                else
                {
                    mismatch[page_index][block_index] = false;
                }

            }

            if(num_mismatch_t == 1)
            {
                printf("Error in block %d \n", block_index);
                num_mismatch++;
            }
            else
            {
                num_match++;
            }
            
        #ifdef PRINT_FIRST_DATA


            // print the first few values from only the first block to demonstrate how to access the data
            if(block_index==0)
            {
                printf("\n");
                for(byte_index=0; byte_index<256; byte_index++) // 64 dev_config[BoardNum].samples_per_block
                {
                    if(byte_index%32==0){ printf("\n"); }

                    printf("0x%x = %x \t(%x)\n", byte_index, ((unsigned char *)pReadData)[byte_index], ((unsigned char *)pBuffer)[byte_index]);
                }

                printf("\n\n");
                for(byte_index=16384-128; byte_index<16384; byte_index++) 
                {
		            if(byte_index%32==0){ printf("\n"); }

                    printf("0x%x = %x \t(%x)\n", byte_index, ((unsigned char *)pReadData)[byte_index], ((unsigned char *)pBuffer)[byte_index]);
                }
            }
        #endif


            // Get out of the loop if ctrl-c is recieved
            if(early_exit==1){ block_index=numBlocksToAcquire+1; }
        }
		               	
    } // for each block
    
    // End time for throughput measurements.
    //x_EndTime(&SysTime, pSS);






    printf(" \n");
    printf(" \n");
    
    if(!DmaTestMode)
    {
        if(num_match == numBlocksToAcquire)
        {
            printf("All %d blocks match\n", num_match);
        }
        else
        {
            printf("%d blocks mismatch!\n", num_mismatch);
        }

        // Print the numbers of the blocks which contained errors
        if(PrintBlocks)
        { 
            for(block_index=0; block_index< numBlocksToAcquire; block_index++)
            {
                flag_block = false;

                for(page_index=0; page_index < PAGES_PER_BLOCK; page_index++)
                {   
                    if( mismatch[page_index][block_index] )
                    {
                        flag_block = true;
                    }
                }

                if(flag_block)
                {
                   printf("Block %d BAD!\n", block_index);
                
                }

            } 
        }
    }


    //}}
    


    // Free the mismatch array
    FreeMismatchArray(&mismatch);

    
}







void RunDRAMDelayCal(uvAPI * pUV, void *pBuffer, void *pReadData)
{

    int i, j, page_index, block_index;
    int dqs_delay, dq_delay, dq_index;
    int error;

    int num_mismatch=0;
    int num_match=0;
    int num_mismatch_t=0;
    bool flag_block;
    unsigned long DRAMAdjReg = 0;


    num_byte_errors = 0;


//    for(i=0; i<101; i++)
//    {
//        printf("%d \t\t 0x%x \t 0x%x \t 0x%x \t 0x%x \t 0x%x \t 0x%x \t 0x%x \t 0x%x\n", i,  sysMem[i], sysMem[i+16], sysMem[i+32], sysMem[i+48], sysMem[i+64],  sysMem[i+80],  sysMem[i+96],  sysMem[i+112]);
//        if((i%16) == 15){ printf("\n"); }
//    }
//    printf("\n");



    // Error histogram for DRAM DQ data. Store number of errors found for each DQ at each (DQS Delay, DQ Delay) combination.
    unsigned long ***dq_error_hist = NULL;
    bool dram_alloc_error = true;


    // If we are going to run the DRAM delay calibration routine then allocate the array for the histogram 
    dram_alloc_error = AllocateDRAMHist(&dq_error_hist);         // 010710, Allocate the error histogram


    if(dram_alloc_error)
    {
        // don't free dq_error_hist as it was not properly allocated.
        // free everything we have allocated to this point.
        exit(1);
    }


    ResetDRAMCalibration(&dq_error_hist);   // 010611, reset variables for DRAM DQS/DQ delay calibration

	//pBoardNum = pUV->DllApiGetCurrentDeviceHandle();

    for(dqs_delay=MIN_DQS_DELAY; dqs_delay<MAX_DQS_DELAY; dqs_delay++)
    {
        for(dq_delay=MIN_DQ_DELAY; dq_delay<MAX_DQ_DELAY; dq_delay++)
        {
            if(early_exit==1){ dq_delay=MAX_DQ_DELAY; } // if ctrl-C was recieved in Linux then exit this loop 


            //ResetBoardPointers(pSS);   // Reset the board's pointers. E.g. DRAM address pointers

            //dll.hDllDramIdelayShift(dqs_delay, dq_delay);  // Set the new delay values
    //		pUV->DramIdelayShift(BoardNum, dqs_delay, dq_delay);
            

            // Write each block of data to the board
            for(block_index=0; block_index < numBlocksToAcquire; block_index++)
            {
                printf("\rWriting Block: %d", block_index); fflush(stdout);
                pUV->X_Write(pBoardNum, pBuffer, DIG_BLOCK_SIZE); // Write the data to the board
				//SLEEP(1);
                if(early_exit==1){ block_index=numBlocksToAcquire+1; } 
            }

            //  SLEEP(1); // Wait for a while to test DRAM refresh
            printf("\n");

            // Read each block of data from the board
            for(block_index=0; block_index < numBlocksToAcquire; block_index++)
            {
                printf("\rReading/Checking Block: %d ",block_index); fflush(stdout);

                // Read the data from the board into second memory buffer
                pUV->X_Read(BoardNum, pReadData, DIG_BLOCK_SIZE);
				Sleep(1);

                if(CompareReRead)
                {
                    // Read the same block again into comparison buffer to see if they match. (If they do then re-reading the same block always matches!)
                    //dll.hDllApiSetPioRegister(PIO_OFFSET_PCIE_RD_START_BLOCK, block_index );  // Set the current PCIe read block
        //			pUV->SET_PCIE_RD_START_BLOCK(BoardNum, block_index);

                    pUV->X_Read(BoardNum, pBuffer, DIG_BLOCK_SIZE);

                }

                //Check each bit, and record in histogram array
                CheckDRAMData(pUV, pBuffer, pReadData, dqs_delay, dq_delay, &dq_error_hist);

                if(early_exit==1){ block_index=numBlocksToAcquire+1; } 

            }

        } // for each dq delay value		               	
    } // for each dqs delay value
    


    printf("\n");
    int byte_index, bit_index;
    int num_errors[NUM_DRAM_DQ/8];  // number of errors in each DQ group
    int num_errors_total;

    // Sum all errors for all bits over each DQS,DQ delay pair to see how each DQS,DQ delay pair performed
    for(dqs_delay=MIN_DQS_DELAY; dqs_delay<MAX_DQS_DELAY; dqs_delay++)
    {
        for(dq_delay=MIN_DQ_DELAY; dq_delay<MAX_DQ_DELAY; dq_delay++)
        {
            num_errors_total = 0;

            for(byte_index=0; byte_index<NUM_DRAM_DQ/8; byte_index++)
            {
                num_errors[byte_index]=0;
            
                for(bit_index=0; bit_index<8; bit_index++)
                {
                    num_errors[byte_index] += dq_error_hist[dqs_delay][dq_delay][(byte_index*8)+bit_index]; // number of erros for this DQS
                }

                num_errors_total += num_errors[byte_index]; // total number of errors

                if(num_errors[byte_index] != 0){ printf("errors in byte %d = %d\n", byte_index, num_errors[byte_index]); }
            }
            printf("(dqs_delay=%d, dq_delay=%d) \t Number Bit Errors=%d  Number Byte Errors=%d\n", dqs_delay, dq_delay, num_errors_total, num_byte_errors);
        }
    }

    // Sum all errors for all DQS,DQ delay pair for each bit to see how each bit performed.






    printf("\n");
    int sel_index=0;

    while( (sel_index>=0) && (sel_index<=112))
    {
        printf("Enter DQ index (out of range 0-112 to quit)):");
        scanf("%d", &sel_index);
        printf("\n");
        //printf("sel_index=%d\n", sel_index);

        for(dqs_delay=MIN_DQS_DELAY; dqs_delay<MAX_DQS_DELAY; dqs_delay++)
        {
            printf("DQS DELAY=%d\n", dqs_delay);

            // re-print the bit indices
            for(dq_index=sel_index; dq_index<sel_index+16; dq_index++)
            {
                printf("<%d> \t", dq_index);
            }
            printf("\n");

            for(dq_delay=MIN_DQ_DELAY; dq_delay<MAX_DQ_DELAY; dq_delay++)
            {
                printf("DQ DELAY=%d\n", dq_delay);

                //printf("\n");
                // print the bit values
                for(dq_index=sel_index; dq_index<sel_index+16; dq_index++)
                {
                    printf("%d \t", dq_error_hist[dqs_delay][dq_delay][dq_index]);
                }
                
                printf("\n");
            }
        }
    }

    // Free the error histogram
    FreeDRAMHist(&dq_error_hist);

}




// We are passed one block of data read from the board (pReadData) and one block of data that it should match (pBuffer)
// The DRAM is NUM_DRAM_DQ bits wide. We compare each set of NUM_DRAM_DQ bits to check if there is an error in that bit
void CheckDRAMData(uvAPI * pUV, void *pBuffer, void *pReadData, int dqs_delay, int dq_delay, unsigned long ****dq_error_hist)
{
    int byte_index, bit_index, dq_index;
    unsigned char ReadByte, CompareByte, ByteMask;
    bool ByteError;

    unsigned char * pBufferChar = (unsigned char *) pBuffer;
    unsigned char * pReadDataChar = (unsigned char *) pReadData;


    // Check each byte in the block
    for(byte_index=0; byte_index<samples_per_block; byte_index++)
    {
        ByteError = false;


//        // Check each bit in the byte
//        for(bit_index=0; bit_index<8; bit_index++)
//        {
//            ByteMask = (1 << bit_index);
//            CompareByte = pBuffer[byte_index] & ByteMask;
//            ReadByte = pReadData[byte_index] & ByteMask;
//
//            // If XOR of this bit is true then they don't match and it is an error
//            if(CompareByte^ReadByte)
//            {
//                // Every 16 bytes is one set of DRAM DQ bits
//                dq_index = 8*(byte_index%(NUM_DRAM_DQ/8))+bit_index;
//                (*dq_error_hist)[dqs_delay][dq_delay][dq_index]++;
//
//                ByteError = TRUE;
 //           }
//
//
//        }// check each bit

        // Just compare the whole byte, not each bit (e.g. its suspect to be a byte problem not a per DQ bit problem)
        CompareByte = pBufferChar[byte_index];
        ReadByte = pReadDataChar[byte_index];

        if(CompareByte != ReadByte)
        {
            // Every 16 bytes is one set of DRAM DQ bits. Since we are only checking bytes just mark the first bit of the bytes as an error.
            dq_index = 8*(byte_index%(NUM_DRAM_DQ/8));
            (*dq_error_hist)[dqs_delay][dq_delay][dq_index]++;

            ByteError = true;
        }


//        // on the last bit print the byte if any of the bits were in error
//        if(ByteError)
//        { 
//            num_byte_errors++;
//            if(num_byte_errors < 50)
//            {
//                //printf("byte_index=%d, (byte_index mod 16)=%d\n\n", byte_index, (byte_index%16));
//                printf("\nByte read = 0x%x (0x%x)  \t\t byte_index=%d num_byte_errors=%d\n", pReadData[byte_index], pBuffer[byte_index], byte_index, num_byte_errors );                             
//            }
//        }


        // If the byte had an error, and it is the first of a DRAM burst (BL=8) (DRAM width is 16 bytes)
        // 16B width, BL=8 -> 128 Bytes per burst. The first samples of each burst are (byte_index%128)<16
        //if(ByteError && ((byte_index%128) < 16))
        if(ByteError)
        {              
            num_byte_errors++;

            if(num_byte_errors < 20)
            {
                printf("\n");
                printf("0x%x: \t 0x%x \t (0x%x)\n", byte_index, pReadDataChar[byte_index], pBufferChar[byte_index] );                
                
                //printf("0x%x: \t(0x%x)\t(0x%x)\t(0x%x)\t(0x%x)\t(0x%x)\t(0x%x)\t(0x%x)\t(0x%x)\n", byte_index, pBuffer[byte_index], pBuffer[byte_index+16], pBuffer[byte_index+32], pBuffer[byte_index+48], pBuffer[byte_index+64], pBuffer[byte_index+80], pBuffer[byte_index+96], pBuffer[byte_index+112]);                
                //printf("0x%x: \t 0x%x \t 0x%x \t 0x%x \t 0x%x \t 0x%x \t 0x%x \t 0x%x \t 0x%x\n", byte_index, pReadData[byte_index], pReadData[byte_index+16], pReadData[byte_index+32], pReadData[byte_index+48], pReadData[byte_index+64], pReadData[byte_index+80], pReadData[byte_index+96], pReadData[byte_index+112]);                
                printf("\n");                             
            }
        }


    }// check each byte

}






