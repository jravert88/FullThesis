////////////////////////
//
// Test program for DMA checksum
// Last Modified: 071410
// 071410 Change: Added 8/14-bit clock frequency check
//
///////////////////////
  

#include "AppDll.h" 
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <string.h>


//#define PRINT_CHECKSUMS // print the checksum for each block, not only if there is an error


int main(int argc, char ** argv)
{

    int i,j;  
    int error = 0, error_flag = 0;	
    int num_blocks;
    unsigned int checksum_local; // locally generated 32-bit checksum calculated per DMA block
    unsigned int checksum_board; // board gernerated 32-bit checksum calcutated per DMA block prior to DMA
    unsigned int adc_underover = 0; // board generated indication of ADC underflow/overflow per DMA block
    float clk_f;
    
    HANDLE disk_fd;      // disk file handle 					
    HANDLE checksum_fd;  // disk file handle 
    HANDLE underover_fd; // disk file handle 


    // system memory buffer, declared here as an int because checksum is 32-bits wide
    unsigned int * sysMem = NULL;       
    //printf("sizeof(unsigned int) = %d\n", sizeof(unsigned int));

    // 071510
    unsigned char * databuf = NULL; 
    int error_index;
    int last_error_j = 0;
    int mult = 1;
    double page_num;


    if((argc != 2)) 
    {
        printf("usage: TestChecksum (number_of_blocks)\n");
        exit(1);
    }

    if(argc == 2)
    {
        num_blocks = atoi(argv[1]);  
        if(num_blocks ==0)
        {
            printf("Failed to parse arguements, exiting... \n"); 
            exit(1);
        }
    }




    // allocate a page-aligned buffer 	
    error = x_MemAlloc((void**)&sysMem, (size_t) DIG_BLOCKSIZE);
    if((error) || (sysMem == NULL))
    {
        printf("memory allocation failed!\n");
        exit(1);
    }
 

    // Open the file containing the acquired data and the file containing the DMA checksums read from the board
    //printf("Opening file. May take a while if the file is large.\n");   
    fflush(stdout);      

    // open the data disk file	
    if ((disk_fd = x_OpenFile("uvdma.dat")) < 0)
    { 
        printf("Error opening data file.\n");          
        exit(1);
    }

    // open the checksum disk file	
    if ((checksum_fd = x_OpenFile("checksum.dat")) < 0)
    { 
       printf("Error opening checksum file.\n");   
       exit(1);
    }

    // open the underover disk file	
    if ((underover_fd = x_OpenFile("underover.dat")) < 0)
    { 
       printf("Error opening underover file.\n");   
       exit(1);
    } 



    // For each block of data requested calculate checksum from the ADC data and compare to checksum read from board
    for(i=0; i<num_blocks; i++)
    {
        // reset to a zeroed checksum
        checksum_local = 0;

        // read data block from board into small system buffer 
        error = x_Read(disk_fd, sysMem, DIG_BLOCKSIZE);
        if(error < 0)
        { 
            printf("Error reading data file. %d\n", error);
        }

        // Calcutate the checksum from the data file
        // 4 bytes per 32-bit DMA word, checksum is 32-bits wide => (DIG_BLOCKSIZE/4) words per DMA block
        for(j=0; j<(DIG_BLOCKSIZE/4); j+=1)	
        {
            checksum_local = checksum_local ^ sysMem[j];
        }

        // Read the checksum from the board for this DMA block 
        // The checksum register on the board is updated after each DMA block is read.
        error = x_Read(checksum_fd, &checksum_board, 4);
        if(error < 0)
        { 
            printf("Error reading checksum file. %d\n", error);
        }

#ifdef PRINT_CHECKSUMS
        printf("Block %d: Local Checksum = %x \t Board Checksum = %x\n", i, checksum_local, checksum_board);
#endif

        if(checksum_local != checksum_board)
        { 
            error_flag = 1;
            printf("ERROR! Checksum Mismatch on Block %d\n", i);
        }

        // Read the ADC underflow/overflow indicator from the board for this DMA block 
        error = x_Read(underover_fd, &adc_underover, 4);

        if(error < 0)
        { 
            printf("Error reading underflow/overflow file. %d\n", error);
        }
        else
        {
            // printf("adc_underover= %x\n", adc_underover);
            
            if(adc_underover & 0x00000001)
            {
                printf("ADC overflow on block %d\n", i);
            }
            if(adc_underover & 0x00000002)
            {
                printf("ADC underflow on block %d\n", i);
            }

            // This section of code checks to see if a 14-bit board's clock was in range.
            int STRB_CNT_MULT = 2; // This is 2 for 14-bit boards
            clk_f = (double)(STRB_CNT_MULT/1024.0)*(double)(10*(adc_underover >> 16));            
            //printf("Clock Frequency = %.1f MHz\n", clk_f);     
            if( (clk_f > 450) || (clk_f < 15))  {   printf("Clock Frequency out of range! %.1f\n", clk_f);  }  

        }


    }


    if(!error_flag)
    { 
        printf("All checksums match.\n");
    }



    // Free the system memory
    if(sysMem != NULL)
    {
        x_FreeMem(sysMem);
    }

    // Close the disk file 
    x_Close(disk_fd);	

    // Close the checksum file
    x_Close(checksum_fd);

    // Close the underover file
    x_Close(underover_fd);

	
    return 0;
}




