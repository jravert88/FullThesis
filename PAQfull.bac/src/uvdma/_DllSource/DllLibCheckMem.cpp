/*
DLL library file. This file used to check acquired data for discontinuities.

Last Modified 110110
110110 Change: Added quick histogram creation, and modified for >2 channels.
062810 Change: Tested checksum creation, reading checksum from board
*/

#include "DllLibMain.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <string.h>

#ifndef _WINDOWS
	#include <unistd.h>
#else
#endif


// Globals defined in DllLibMain.cpp, import them here. (012911)
extern DEVICE_HANDLE;                          // Handle to the low-level driver API
//extern short BoardNum;                           // The index of the current board that we are accessing.
extern cal_struct cal_data[MAX_DEVICES];       // Calibration data for each board
extern config_struct dev_config[MAX_DEVICES];  // Configuration data structure


//for large files
#define _FILE_OFFSET_BITS 64


#define DO_HISTOGRAM        false    // false
#define NUM_BLOCKS_HIST     10      // 10   10
#define NUM_DISCON_TO_PRINT 10      // 10    0
#define PRINT_CHECKSUM	    0	    // 1    0
#define DEBUG_BLOCKS	    -1	    // 32   -1

     
unsigned long ** hist;      // 020211
unsigned long num_bad = 0;

void PrintHist(unsigned short BoardNum, int block, void *pBuffer);
void GetBoardCalStats(unsigned short BoardNum, SAMPLE_TYPE sample[MAX_CHAN]);


void CheckMem(unsigned short BoardNum, int num_blocks, FILE_HANDLE_TYPE pFile, void *pBuffer)
{
    // Arrays for the ADC samples for each channel.
    SAMPLE_TYPE sample[MAX_CHAN], last_sample[MAX_CHAN];
    
    int chan_delta[MAX_CHAN];
    int last_error_block[MAX_CHAN], last_error_index[MAX_CHAN];
    int block_i;
    unsigned int i;
    unsigned short chan_i;
    unsigned long bytesRead;
    unsigned long chan_index;

    unsigned short CHAN_A, CHAN_B;



    if(dev_config[BoardNum].adc_res == 8)
    {
        // 061810 these are correct for 8-bit boards
        CHAN_A = 0;
        CHAN_B = 1;
    }
    else
    {
        // 012010 these are correct for 14 bit boards
        CHAN_A = 1;
        CHAN_B = 0;
    }



    // allocate the histogram array for each channel
    hist = (unsigned long **) malloc((dev_config[BoardNum].adc_chan)*sizeof(unsigned long *));

    if(hist == NULL)
    {
        printf("Error allocating histogram array, exiting!\n");
        return;
    }

    // for each channel allocate an array with an element for each adc value (histogram)
    for(chan_index=0; chan_index<dev_config[BoardNum].adc_chan; chan_index++)
    {
        hist[chan_index] = (unsigned long *) malloc((dev_config[BoardNum].max_adc_value+1)*sizeof(unsigned long));
        
        if(hist[chan_index] == NULL)
        {
            printf("Error allocating histogram array, exiting!\n");
            return;
        }
    }




    for(chan_i=0; chan_i<dev_config[BoardNum].adc_chan_used; chan_i++)
    {
        sample[chan_i] = 0;
        last_sample[chan_i] = 0;
        chan_delta[chan_i] = 0;
        
        cal_data[BoardNum].glitches[chan_i] = 0;
        cal_data[BoardNum].num_7F[chan_i] = 0;
        cal_data[BoardNum].num_less7F[chan_i] = 0;		
        cal_data[BoardNum].num_80[chan_i] = 0;
        cal_data[BoardNum].num_more80[chan_i] = 0;
    }


    
    for(block_i=0; block_i < num_blocks; block_i++)
    {

// Read the next block from the file (OS dependent)
#ifdef _WINDOWS
        if(pFile != NULL) // read from file
#else
        if(pFile != 0)
#endif
        {
#ifdef _WINDOWS	
#else
                lseek(pFile, (off_t)block_i*DIG_BLOCKSIZE, SEEK_SET);
                read(pFile, pBuffer, DIG_BLOCKSIZE);
#endif
        }
		else	// read directly from the board, critical for the 8-bit phase shift calibration
		{	
            SET_PIO_REG(BoardNum, PIO_OFFSET_PCIE_RD_START_BLOCK, block_i);
		
#ifdef _WINDOWS
                ReadFile( DEVICE_CURBOARDHANDLE, pBuffer, DIG_BLOCKSIZE, &bytesRead, NULL );	// read the current block
#else
                read(DEVICE_CURBOARDHANDLE, pBuffer, DIG_BLOCKSIZE);
#endif
		}



        if(block_i%1000 == 0){ printf("Checking block %d\n", block_i); }

        if((block_i < NUM_BLOCKS_HIST) && DO_HISTOGRAM){ PrintHist(BoardNum, block_i, pBuffer); }
  

        for( i=0; i<dev_config[BoardNum].samples_per_block; i+=1)	
        {

            if(i==0)
            {

                if(block_i == 0)
                {
                    for(chan_i=0; chan_i<dev_config[BoardNum].adc_chan_used; chan_i++)
                    {
                        chan_delta[chan_i] = 0; // no error possible on the very fist sample of entire record
                    }
                }
                else
                {
                    // Get the ith channel sample for the 0th multi-sample
                    // Compare the 1st sample of this block to last sample of the last block
                    for(chan_i=0; chan_i<dev_config[BoardNum].adc_chan_used; chan_i++)
                    {
                        sample[chan_i] = GetSample(BoardNum, pBuffer, 0, chan_i); 
                        chan_delta[chan_i] = sample[chan_i] - last_sample[chan_i];

                        // TODO: Add glitch checking here!
                        if( (chan_delta[chan_i] > (dev_config[BoardNum].adc_error_value)) || (chan_delta[chan_i] < -(dev_config[BoardNum].adc_error_value)) )
                        {
                            cal_data[BoardNum].glitches[chan_i]++;
		
                            if(cal_data[BoardNum].glitches[chan_i] < NUM_DISCON_TO_PRINT)
                            {
                                printf("blk=%d \t byte=%d \t Channel=%d \t Delta=%d\n", block_i, i, chan_i, chan_delta[chan_i]);
                            }

                            // keep track of the current location so we can know if this is a glitch or a break
                            last_error_block[chan_i] = block_i;
                            last_error_index[chan_i] = i;
                        }
                    }

                }
            }
            // If it's not the very fist sample in the block
            else
            {
                    // Get the ith channel sample for the ith multi-sample
                    for(chan_i=0; chan_i<dev_config[BoardNum].adc_chan_used; chan_i++)
                    {
                        sample[chan_i] = GetSample(BoardNum, pBuffer, i, chan_i); 
                        chan_delta[chan_i] = sample[chan_i] - GetSample(BoardNum, pBuffer, i-1, chan_i);
                        // save last sample of this block to compare with first of next block
                        last_sample[chan_i] = GetSample(BoardNum, pBuffer, dev_config[BoardNum].samples_per_block-1, chan_i);  // TODO: Only do this once!
                        
                        // check the derivative and check if its the second in a single glitch. only count glitches once.
                        if( 
                            ( (chan_delta[chan_i] > dev_config[BoardNum].adc_error_value) || (chan_delta[chan_i] < -dev_config[BoardNum].adc_error_value) )
                        &&
                        !(  ((last_error_block[chan_i] == block_i) && (last_error_index[chan_i] == i-1)) )
                          )
                        {
                            cal_data[BoardNum].glitches[chan_i]++;
	
                            if(cal_data[BoardNum].glitches[chan_i] < NUM_DISCON_TO_PRINT)
                            {
                                printf("blk=%d \t byte=%d \t Channel=%d \t Delta=%d\n", block_i, i, chan_i, chan_delta[chan_i]);
                            }

                            // keep track of the current location so we can know if this is a glitch or a break
                            last_error_block[chan_i] = block_i;
                            last_error_index[chan_i] = i;
                        }
                        
                    }
    
                    // Get statistics for board_cal, TODO: include the very first sample of each block...
                    GetBoardCalStats(BoardNum, sample);
		

            }// End block index not equal to zero

        }// End block index
    
    }// End block number



    for(chan_i=0; chan_i<dev_config[BoardNum].adc_chan_used; chan_i++)
    {
        printf("Chan %d: # Glitches = %d\n", chan_i, cal_data[BoardNum].glitches[chan_i]);
    }


    if(hist != NULL)
    {
        free(hist);
    }
}

void CheckMemWindows(unsigned short BoardNum, int num_blocks, const char * filename, void *pBuffer)
{
    // Arrays for the ADC samples for each channel.
    SAMPLE_TYPE sample[MAX_CHAN], last_sample[MAX_CHAN];
    
    int chan_delta[MAX_CHAN];
    int last_error_block[MAX_CHAN], last_error_index[MAX_CHAN];
    int block_i;
    unsigned int i;
    unsigned short chan_i;
    unsigned long bytesRead;
    unsigned long chan_index;

    unsigned short CHAN_A, CHAN_B;

	FILE *fileHandle;

    if(dev_config[BoardNum].adc_res == 8)
    {
        // 061810 these are correct for 8-bit boards
        CHAN_A = 0;
        CHAN_B = 1;
    }
    else
    {
        // 012010 these are correct for 14 bit boards
        CHAN_A = 1;
        CHAN_B = 0;
    }



    // allocate the histogram array for each channel
    hist = (unsigned long **) malloc((dev_config[BoardNum].adc_chan)*sizeof(unsigned long *));

    if(hist == NULL)
    {
        printf("Error allocating histogram array, exiting!\n");
        return;
    }

    // for each channel allocate an array with an element for each adc value (histogram)
    for(chan_index=0; chan_index<dev_config[BoardNum].adc_chan; chan_index++)
    {
        hist[chan_index] = (unsigned long *) malloc((dev_config[BoardNum].max_adc_value+1)*sizeof(unsigned long));
        
        if(hist[chan_index] == NULL)
        {
            printf("Error allocating histogram array, exiting!\n");
            return;
        }
    }




    for(chan_i=0; chan_i<dev_config[BoardNum].adc_chan_used; chan_i++)
    {
        sample[chan_i] = 0;
        last_sample[chan_i] = 0;
        chan_delta[chan_i] = 0;
        
        cal_data[BoardNum].glitches[chan_i] = 0;
        cal_data[BoardNum].num_7F[chan_i] = 0;
        cal_data[BoardNum].num_less7F[chan_i] = 0;		
        cal_data[BoardNum].num_80[chan_i] = 0;
        cal_data[BoardNum].num_more80[chan_i] = 0;
    }


	fileHandle = fopen(filename, "rb");
    
    for(block_i=0; block_i < num_blocks; block_i++)
    {
#ifdef _WINDOWS
		__int64 seekPos = DIG_BLOCKSIZE * (__int64)(block_i);
		// Read the next block from the file (OS dependent)
		_fseeki64(fileHandle,seekPos,SEEK_SET);
		fread(pBuffer,1,DIG_BLOCKSIZE,fileHandle);
#endif
        if(block_i%1000 == 0){ printf("Checking block %d\n", block_i); }

        if((block_i < NUM_BLOCKS_HIST) && DO_HISTOGRAM){ PrintHist(BoardNum, block_i, pBuffer); }
  

        for( i=0; i<dev_config[BoardNum].samples_per_block; i+=1)	
        {

            if(i==0)
            {
				//printf("first sample in block\n");
                if(block_i == 0)
                {
                    for(chan_i=0; chan_i<dev_config[BoardNum].adc_chan_used; chan_i++)
                    {
                        chan_delta[chan_i] = 0; // no error possible on the very fist sample of entire record
                    }
                }
                else
                {
                    // Get the ith channel sample for the 0th multi-sample
                    // Compare the 1st sample of this block to last sample of the last block
                    for(chan_i=0; chan_i<dev_config[BoardNum].adc_chan_used; chan_i++)
                    {
                        sample[chan_i] = GetSample(BoardNum, pBuffer, 0, chan_i); 
                        chan_delta[chan_i] = sample[chan_i] - last_sample[chan_i];
                        // TODO: Add glitch checking here!
                        if( (chan_delta[chan_i] > (dev_config[BoardNum].adc_error_value)) || (chan_delta[chan_i] < -(dev_config[BoardNum].adc_error_value)) )
                        {
                            cal_data[BoardNum].glitches[chan_i]++;
		
                            if(cal_data[BoardNum].glitches[chan_i] < NUM_DISCON_TO_PRINT)
                            {
                                printf("blk=%d \t byte=%d \t Channel=%d \t Delta=%d\n", block_i, i, chan_i, chan_delta[chan_i]);
                            }

                            // keep track of the current location so we can know if this is a glitch or a break
                            last_error_block[chan_i] = block_i;
                            last_error_index[chan_i] = i;
                        }
                    }

                }
            }
            // If it's not the very fist sample in the block
            else
            {		
                    // Get the ith channel sample for the ith multi-sample
                    for(chan_i=0; chan_i<dev_config[BoardNum].adc_chan_used; chan_i++)
                    {
                        sample[chan_i] = GetSample(BoardNum, pBuffer, i, chan_i); 
                        chan_delta[chan_i] = sample[chan_i] - GetSample(BoardNum, pBuffer, i-1, chan_i);

                        // save last sample of this block to compare with first of next block
                        last_sample[chan_i] = GetSample(BoardNum, pBuffer, dev_config[BoardNum].samples_per_block-1, chan_i);  // TODO: Only do this once!
                        
                        // check the derivative and check if its the second in a single glitch. only count glitches once.
                        if( 
                            ( (chan_delta[chan_i] > dev_config[BoardNum].adc_error_value) || (chan_delta[chan_i] < -dev_config[BoardNum].adc_error_value) )
                        &&
                        !(  ((last_error_block[chan_i] == block_i) && (last_error_index[chan_i] == i-1)) )
                          )
                        {
                            cal_data[BoardNum].glitches[chan_i]++;
	
                            if(cal_data[BoardNum].glitches[chan_i] < NUM_DISCON_TO_PRINT)
                            {
                                printf("blk=%d \t byte=%d \t Channel=%d \t Delta=%d\n", block_i, i, chan_i, chan_delta[chan_i]);
                            }

                            // keep track of the current location so we can know if this is a glitch or a break
                            last_error_block[chan_i] = block_i;
                            last_error_index[chan_i] = i;
                        }
                        
                    }
    
                    // Get statistics for board_cal, TODO: include the very first sample of each block...
                    GetBoardCalStats(BoardNum, sample);
		

            }// End block index not equal to zero

        }// End block index
    
    }// End block number



    for(chan_i=0; chan_i<dev_config[BoardNum].adc_chan_used; chan_i++)
    {
        printf("Chan %d: # Glitches = %d\n", chan_i, cal_data[BoardNum].glitches[chan_i]);
    }


    if(hist != NULL)
    {
        free(hist);
    }

	fclose(fileHandle);
}






void PrintHist(unsigned short BoardNum, int block, void *pBuffer)
{

    unsigned long sample;
    unsigned int chan_i, samp_index;


    // For each multi-sample word in the block
    for(samp_index=0; samp_index<dev_config[BoardNum].samples_per_block; samp_index++)
    {
        // For each channel
        for(chan_i=0; chan_i<dev_config[BoardNum].adc_chan_used; chan_i++)
        {
            sample = GetSample(BoardNum, pBuffer, samp_index, chan_i);             
            hist[chan_i][sample]++;
        }
    }

    
    // Print the histogram only after the last block
    if(block == NUM_BLOCKS_HIST-1)
    {
        // For each channel
        for(chan_i=0; chan_i<dev_config[BoardNum].adc_chan_used; chan_i++)
        {
            // Print the histogram array
            for(samp_index=0; samp_index<=dev_config[BoardNum].max_adc_value; samp_index++)
            {
                if(hist[chan_i][samp_index] != 0)
                {
                    printf("Chan %d, Value %d: %d\n", chan_i, samp_index, hist[chan_i][samp_index]);
                }
            }

            printf("\n");
        }
    }

/*    
    // Count the number of values that far away from 7F/80
    // Sum the histogram array
    for(samp_index=0; samp_index<=dev_config[BoardNum].max_sample_value; samp_index++)
    {
        // For each channel
        for(chan_i=0; chan_i<dev_config[BoardNum].adc_chan_used; chan_i++)
        {
            if((samp_index != 126) && (samp_index != 127) && (samp_index != 128) && (samp_index != 129))
            {
                num_bad += hist[chan_i][samp_index];
            }
        }
    }

    printf("block = %d \n", block);
    printf("num_bad = %d \n", num_bad);
*/

}


void GetBoardCalStats(unsigned short BoardNum, SAMPLE_TYPE sample[MAX_CHAN])
{
    unsigned int chan_i;

    for(chan_i=0; chan_i<dev_config[BoardNum].adc_chan_used; chan_i++)
    {
        // Get statistics for board_cal
        if(sample[chan_i] == 0x7F)
        {
            cal_data[BoardNum].num_7F[chan_i]++;
        }
        else if( (sample[chan_i] < 0x7F) && (sample[chan_i] > 0x78) )
        {
            cal_data[BoardNum].num_less7F[chan_i]++;		
        }

        if(sample[chan_i] == 0x80)
        {
            cal_data[BoardNum].num_80[chan_i]++;
        }
        else if( (sample[chan_i] > 0x80) && (sample[chan_i] < 0x88) )
        {
            cal_data[BoardNum].num_more80[chan_i]++;		
        }       
    }

    return;
}
