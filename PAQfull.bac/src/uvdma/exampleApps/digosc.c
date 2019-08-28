/*
 Use the spin_button (with text field) to enter the desired block in the file you wish to examine. Then press the Update button to draw the requested block.
 Use the slider bar to select which set of bytes you wish examine. (no update necessary).
	
 Requires: GTK & X Windows library files.

 Last Modified: 021311 - Ported to use cross platform Windows DLL and Linux static library.
*/



#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <gtk/gtk.h>
#include <fcntl.h>
#include <string.h>
#include <sys/stat.h>

#include <inttypes.h>
#include <stdint.h>

//#include "AppDll.h" 

	


void digosc_parser(int argc, char ** argv);
void digosc_parser_printf();
int DIG_BLOCKSIZE = (1024 * 1024);

int block_size = DIG_BLOCKSIZE;	// blocksize for DMA transfers, all I/O requests must be this size
int adcChanUsed=4;
//int samples_per_block=block_size; //default to 1 bytes per sample
bool verbose = false;
int bits = 16; //default to 16bit


    unsigned int getSample16(unsigned int numChannels,  unsigned char *buf, size_t sample_num, unsigned int channel )
    {
        uint16_t  *uint16ptr = (uint16_t  *)buf;
        unsigned int val = uint16ptr[sample_num * numChannels + channel];
        return val;
    }
    unsigned int getSample14(unsigned int numChannels,  unsigned char *buf, size_t sample_num, unsigned int channel )
    {
        uint16_t  *uint16ptr = (uint16_t  *)buf;
        unsigned int val = uint16ptr[sample_num * numChannels + channel] & 0x3fff;
        return val;
    }
    unsigned int getSample12(unsigned int numChannels,  unsigned char *buf, size_t sample_num, unsigned int channel )
    {
        uint16_t  *uint16ptr = (uint16_t  *)buf;
        unsigned int val = uint16ptr[sample_num * numChannels + channel] & 0x0fff;
        return val;
    }
    unsigned char getSample8(unsigned int numChannels, unsigned char *buf, size_t sample_num, unsigned int channel )
    {
        unsigned char *uint8ptr = (unsigned char *)buf;
        unsigned char val = uint8ptr[sample_num * numChannels + channel];
        return val;
    }


#define CANVAS_X  1000
#define CANVAS_Y  500
#define _FILE_OFFSET_BITS 64    //for large files

// These should be moved into digosc, they are only used there and user doesn't need to change them!
// define to draw the zero value on the y-axis for the digosc application
//#define DRAW_ZERO

// Define to enable memory checking. Does not work with direct/continuous 
#define _CHECK_MEM	
//#define CHECKMEM_PRINT		10	// set to 0 for no print
//#define _SKIP_GRAPH
bool skip_graph = false;


// The total number of DMA blocks that can be displayed
// will be determined from the file size or user input (acquisition mode)
int total_blocks = 0;
int total_blocks2 = 0; 

// Keep track of window size and number of "windows" worth of points plotted
int current_width;
int current_height;
int windows_plotted;
int windows_per_dma_block;
int blocks_read=0;
int discontinuity=0;
u_int gnextsample;
int first = 0;


//102610 - Declare global zero and scale variables
float scale;

int zero_sample_value;
float scale_coefficient;



// Device Interface Variables
int board_handle;
int boardp;
char dev_name[25]  = "/dev/uvdma0";
char file_name[25] = "uvdma.dat";
char file_name2[25] = "uvdma1.dat"; 
// NOTE: IF MAX_FILES OR MAX_CHANNELS IS LARGER THAN 8, MORE COLORS NEED TO BE ADDED.
// ALSO, chan_string[] array needs to be added to
#define MAX_FILES 8
#define MAX_CHANNELS 4  //102610
int fd[MAX_FILES];
char filenames[MAX_FILES][25];
int alloc_failed[MAX_FILES];
int num_files;
// end Device Interface Variables

//Color Setup
static GdkColor red = 
{ 0, 0xffff, 0x0000, 0x0000 }
;
static GdkColor green = 
{ 0, 0x0000, 0xffff, 0x0000 }
;
static GdkColor blue = 
{ 0, 0x0000, 0x0000, 0xffff }
;
static GdkColor yellow = 
{ 0, 0xffff, 0xffff, 0x0000 }
;
static GdkColor cyan = 
{ 0, 0x0000, 0xffff, 0xffff }
;
static GdkColor pink = 
{ 0, 0xffff, 0x0000, 0xffff }
;


u_long max_blocks;


// Digosc acquire
//gboolean timer_callback (gpointer data);
gboolean timer_callback (GtkWidget * data);


//Display Defines
char * chan_string[MAX_CHANNELS];
char chan_string0[25] = "AIN 0";
char chan_string1[25] = "AIN 1";
char chan_string2[25] = "AIN 2";
char chan_string3[25] = "AIN 3";


//char reading[25]  = "Reading uvdma.dat:";
char reading[35] ="Reading ";
char continuous_acq[35] = "Continuous Acquisition Mode";
char direct_text[100] = "Direct Acquisiton ";
char blocks_text[10] = " Blocks";


// Utility Function Definitions - See Implementations for details
void plot_window(GtkWidget *widget, int start_index);
void plot(GtkWidget *widget);
void digosc_cleanup();

/*Event Handlers (Callback Functions)  - See Implementations for details*/
static int expose_event(GtkWidget *widget, GdkEventExpose *event);
static int configure_event(GtkWidget *widget, GdkEventConfigure *event);
static void destroy( GtkWidget *widget, gpointer data);
void update(GtkWidget *widget);
static void scale_value_changed(GtkWidget * widget, gpointer data);

/* GTK specific variables*/
static GdkPixmap * pixmap = NULL; 	// The "offscreen" pixelmap, global b/c it must be replaced everytime the parent window is resized
GtkWidget * blockselect_spin = NULL;
GtkWidget * singleblock_button = NULL; 	// Pointer to manage checkbox which allows selection of continous acquisition of single block
u_char block_mode;
GtkWidget * byteselect_scale = NULL;
u_char selecting_byte = FALSE;		// A boolean used to keep track of block mode versus byte selecting (single window)
int selected_start_byte = 0;           	// Used to keep track of where the slider (scale) is currently positioned

int selected_start_block= 0;		// selected block offset in the file 

short multi_file;
int num_blocks_direct;
int ttl_cnt;
unsigned char * sysMem;	 		// Allocate for the setup_bd call






// Allocate as a char array so that pointer arithmetic is byte based.
unsigned char *pBuffer[MAX_FILES];

//dll_struct dll;
// 1) When the DLL is loaded it instantiates the device driver API object and makes handles to each device.
// 2) The user should define a dll_struct object and then load the DLL by calling x_LoadDll(). 
// 3) Instantiate a setup_stuct, fill it out and pass it to x_SelectDevice().
// 4) Call x_SelectDevice() which will store a handle to that device in dll.hCurrentDevice. This is the handle that should be used for system calls.
// 5) Call dll.hDllSetupBoard() to prepare the specified board for acquisition according to the setup_stuct values.

//setup_struct ss;
// Structure containing variables used for board setup. This is declared at the user level so that the user can
// change these variables and consequently the operation of the board. The definition of setup_struct is in AD83000xDefines.h
// This structure must be updated as desired before calling SetupBoard(). See definition of structure for details.


int samplesPerBlock = DIG_BLOCKSIZE;


int main(int argc, char * argv[])
{
//    x_LoadDll(&dll);



    num_blocks_direct = 0;
    GtkWindow * window;
    GtkWidget * canvas;
    GtkWidget * update_button;
    GtkWidget * vbox;
    GtkWidget * hbox;          

    struct stat fd_stat[MAX_FILES];  
    int status;
    char temp[25] = "h";
    char tempnum[5];
    int i,j;
    max_blocks = 4*1024*1024/(block_size/1024);
    boardp = 0;

    
    // Set the strings that are going to print on the top of the screen  //CHANGE IF MAX_CHANNELS changes
    chan_string[0] = & chan_string0[0];
    chan_string[1] = & chan_string1[0];
    chan_string[2] = & chan_string2[0];
    chan_string[3] = & chan_string3[0];


    // Parse the command line inputs. This funtion will set the variables in the setup_structure, which determines the operation of the board.
    digosc_parser(argc, argv);

	
    // Regardless of usage this application will always require at least one memory buffer, allocate this buffer
    alloc_failed[0] = posix_memalign((void**) &pBuffer[0], 4096, 2*DIG_BLOCKSIZE);
	
    if(alloc_failed[0])
    {
	printf("Failed to allocate %dMB host memory buffer...exiting.\n", DIG_BLOCKSIZE/(1024*1024));
	digosc_cleanup();
        exit(-1);
    }     
    else
    {
        // If the memory has been successfully allocated, pass the buffer along with the setup struct
//        ss.pBuffer = (void *) pBuffer[0];
    }

    if (bits != 8)
    {
        samplesPerBlock /= 2;  // 2 bytes/sample
    }

    if (adcChanUsed == 0)
        adcChanUsed = 2;
    samplesPerBlock /= adcChanUsed;
    
    // Call the DLL and select device number ss.board_num, if possible
//    x_SelectDevice(&dll, &ss, ss.board_num);

	
    // Setup the board specified by ss.board_num for acquisition. SetupBoard will return false if the setup failed.
//    if(!dll.hDllSetupBoard(&ss))
//    {
//        x_FreeMem(sysMem);;
//	digosc_cleanup();
 //       exit(-1);
//    }


    switch (bits){
    case 8:
        zero_sample_value = (1 << 7);
        scale_coefficient = (256);
        break;
    case 12:
        zero_sample_value = (1 << 11);
        scale_coefficient = (4*1024);
        break;
    case 14:
        zero_sample_value = (1 << 13);
        scale_coefficient = (16*1024);
        break;
    case 16:
        zero_sample_value = (1 << 15);
        scale_coefficient = (64*1024);
        break;
    default:
        zero_sample_value = (1 << 15);
        scale_coefficient = (64*1024);
    }

    /*
    if(ss.adc_res == 8)
    {
        zero_sample_value = 0x0080;
        scale_coefficient = 256;
    }
    else if(ss.adc_res == 12)
    {
        zero_sample_value = 0x00000800;
        scale_coefficient = (4*1024);
    }
    else if(ss.adc_res == 14)
    {
        zero_sample_value = 0x00002000;
        scale_coefficient = (16*1024);
    }
    else if(ss.adc_res == 16)
    {
        zero_sample_value = 0x00007FFF;
        scale_coefficient = (64*1024);
    }
*/

    for(i=0; i<MAX_FILES; i++)
    {
	alloc_failed[i] = -1;	// init memory buffer allocation status to unallocated
	fd[i] = -1;		// init file descriptors to unallocated
    }



    if(multi_file == TRUE)
    {
        // create the filenames
        char uvdma[7] ="uvdma";
        char dotdat[7] =".dat";

        for (j=0; j<num_files; j++)
	{
            filenames[j][0]='\0';
            sprintf(tempnum, "%d", j);
            strcat(filenames[j],uvdma);  	// append uvdma

            if(j != 0 )
	    { 					// there is no uvdma0.dat, only uvdma.dat
                strcat(filenames[j],tempnum); 	// append the file number
            }
	
            strcat(filenames[j],dotdat);  	// append the .dat
        }

        // If the multi_file switch is activated from the command line, the extra buffers are allocated
        for (j=1; j<num_files; j++)
        {
            alloc_failed[j] = posix_memalign((void**) &pBuffer[j], 4096, 2*DIG_BLOCKSIZE); // was 4*block_size for dual channel 8-bit
      
            if(alloc_failed[j])
            {
                printf("Memory Allocation Failed!\n");
                digosc_cleanup();
                exit(-1);
            }
        }
    }



    printf("Ignore error messages regarding board setup if reading from file.\n");
    printf("Press the Update button after each time selected block is changed.\n");
    printf("***The displayed data may not be accurate until update is pressed***\n");


 /*   if((boardp != 0) && (num_blocks_direct == 0))
    {
	// Set vars to passed to the setup routines to prepare the board(s) for capture
	ss.num_devices = 1;
	ss.blocks_to_acq = 64;	
	ss.pBuffer = pBuffer[0];
        setup_bd(&boardp, &ss);
    }*/


    printf("\n");

	if (skip_graph == false)
	{
	    gtk_init (&argc, &argv); // initialize the
	
	    //Create a new window object, cast the return value from GtkWidget to GtkWindow
	    window = GTK_WINDOW( gtk_window_new (GTK_WINDOW_TOPLEVEL) );
	    //Set the default size
	    gtk_window_set_default_size(window, CANVAS_X, CANVAS_Y);


	    //Create packing objects for controlling placement of widgets
	    vbox = gtk_vbox_new(FALSE, 0);
	    hbox = gtk_hbox_new(FALSE, 0);
	}

//printf("boardp = %d\n", boardp);

    // if no board is open then read from file
    if(boardp == 0)
    {
        if(multi_file == FALSE)
	{  
            //for a single file, only open one file
            fd[0] = open(file_name, O_RDWR);

            if(fd[0]<1)
            {
                printf("FAILED TO OPEN %s, EXITING....\n",file_name);
                digosc_cleanup();
                exit(-1);
            }
            status = fstat(fd[0], &fd_stat[0]);
            total_blocks = (int)(fd_stat[0].st_size/block_size);
	
            printf("file_name = %s. total_blocks = %d.\n", file_name, total_blocks);
        }
        else if(multi_file == TRUE)
	{ 
            // for multiple files, open a fd for each file.
            for (j=0; j<num_files; j++)
	    {
                int temp_blocks;
		char color_name[25];
                char * t_return;

                fd[j] = open(filenames[j], O_RDWR);
                status = fstat(fd[j],&fd_stat[j]);

                if(status == -1)
		{
                    printf("FAILED TO OPEN %s, EXITING.... \n ",filenames[j]);
                    digosc_cleanup();
                    exit(-1);
                }

                //Setup the string with the color names- to print to command line
                t_return = strcpy(color_name, "color failure");
		switch (j)
		{
                    case 0 : 
			t_return=strcpy(color_name,"Black");
			break;
                    case 1 : 
			t_return=strcpy(color_name,"Red"); 
			break;
                    case 2 : 
			t_return=strcpy(color_name,"Blue");
			break;
                    case 3 : 
			t_return=strcpy(color_name,"Green"); 
			break;
                    case 4 : 
			t_return=strcpy(color_name,"Orange");
			break;
                    case 5 : 
			t_return=strcpy(color_name,"Yellow"); 
			break;
                    case 6 : 
			t_return=strcpy(color_name,"Pink");
			break;
                    case 7 : 
			t_return=strcpy(color_name,"Cyan");
			break;
                    case 8 : 
			t_return=strcpy(color_name,"Violet"); 
			break;
                }
                if(j==0){total_blocks = (int)(fd_stat[j].st_size/block_size);}

                // We need to make sure that total_blocks is the number of blocks in the smallest file
                temp_blocks = (int)(fd_stat[j].st_size/block_size);    

                if(temp_blocks < total_blocks){ total_blocks = temp_blocks;} 
                printf("filename = %s.  file size = %dMB.  total blocks = %d. Waveform Color: %s \n", filenames[j], (int)fd_stat[j].st_size/(1024*1024), total_blocks, color_name);
            }
        }
    }

if (skip_graph == false)
{
    update_button = gtk_button_new_with_label("Update");
    gtk_box_pack_start(GTK_BOX(hbox), update_button, FALSE, FALSE, 0);	

    blockselect_spin = gtk_spin_button_new_with_range(0, total_blocks-1, 1); // total_blocks	//min block=0 max block=blocks_in_file-1 step=1
    gtk_box_pack_start(GTK_BOX(hbox), blockselect_spin, FALSE, TRUE, 0);


    singleblock_button = gtk_check_button_new_with_label("Single Block");
    //gtk_box_pack_start(GTK_BOX(hbox), singleblock_button, FALSE, FALSE, 0);	
    gtk_toggle_button_set_active( GTK_TOGGLE_BUTTON(singleblock_button), TRUE);
    block_mode = TRUE;

    // Set scale range [0, 512K] with  step size

    byteselect_scale = gtk_hscale_new_with_range(0,samplesPerBlock, 512);
    gtk_range_set_increments(GTK_RANGE(byteselect_scale), 512, 512);                    // Clicking on slider increments by 512
    gtk_range_set_update_policy(GTK_RANGE(byteselect_scale), GTK_UPDATE_DISCONTINUOUS); // Request update only when slider is released
    gtk_box_pack_start(GTK_BOX(hbox), byteselect_scale, TRUE, TRUE, 0);


    gtk_box_pack_start(GTK_BOX(vbox), hbox, FALSE, FALSE, 0);  //Place the hbox in the vbox


    // Create drawing area (canvas)
    canvas = gtk_drawing_area_new();
    gtk_drawing_area_size( GTK_DRAWING_AREA(canvas), CANVAS_X, CANVAS_Y);
    gtk_box_pack_start(GTK_BOX(vbox), canvas, TRUE, TRUE, 0);

    gtk_container_add(GTK_CONTAINER(window), vbox);

    // Configure Event Handlers
    g_signal_connect(G_OBJECT(window), "destroy", G_CALLBACK(destroy), NULL);
    g_signal_connect(G_OBJECT(canvas), "expose_event", G_CALLBACK(expose_event), NULL);
    g_signal_connect(G_OBJECT(canvas), "configure_event", G_CALLBACK(configure_event), NULL);
    g_signal_connect(G_OBJECT(update_button), "clicked", G_CALLBACK(update), NULL);
    g_signal_connect(G_OBJECT(byteselect_scale), "value-changed", G_CALLBACK(scale_value_changed), NULL);

    // Display the main window and all sub Widgets
    gtk_widget_show_all( GTK_WIDGET(window) );

    /*if((boardp != 0) && (num_blocks_direct == 0)) //continuous mode
        g_timeout_add(100,  (gboolean (*) (void *))timer_callback, GTK_WIDGET(window));*/
}
    strcat(reading,file_name);

    if (skip_graph == false)
    {
    printf("going to gtk loop\n");
    gtk_main ();
    }
    if (skip_graph == true)
    {
        printf("Checking Memory\n");
    // dll.hDllCheckMem(total_blocks, fd[0], pBuffer[0]);
    }


    if(fd[0] != -1) {close(fd[0]);}
        if( multi_file == TRUE)
        {
            for (j=1; j<num_files; j++)
            {
                if(fd[j] != -1) {close(fd[j]);}
            }
        }

    digosc_cleanup();

    return 0;
}




 
// void update(GtkWidget *widget)
// callback function for a press of the update button
// PARAMS: Widget * widget -  provides access to the main window so it can be queued for a "repaint"
void update(GtkWidget *widget)
{
    int bytes_read=0;
    int j;
    int mult_bytes_read[MAX_FILES];
    off_t file_offset;

    // Read the check button to see if user wants a single block transfer or continous acquisition
    // block_mode = gtk_toggle_button_get_active( GTK_TOGGLE_BUTTON(singleblock_button));

    // Ensure that selecting_byte is deasserted
    selecting_byte = FALSE;

    selected_start_block = (int)gtk_spin_button_get_value(GTK_SPIN_BUTTON(blockselect_spin));


//printf("boardp = %d\n", boardp);
    if(boardp == 0)
    {
        // Reading from one file
        if (multi_file == FALSE)
        {
//printf("reading from single file\n");

            file_offset = (off_t) selected_start_block;
            file_offset = file_offset * (off_t) DIG_BLOCKSIZE;
            // For a single file, only read from one of the FDs
 	    lseek(fd[0], file_offset, SEEK_SET);// move to requested block
  
//printf("selected_start_block=%d total_blocks=%d block_size=%d  ss.samples_per_block=%d\n", selected_start_block, total_blocks, block_size, ss.samples_per_block);
	    if(selected_start_block >= total_blocks-1) 
	    { 
	        selected_start_block = total_blocks-1;
  	        bytes_read = read(fd[0], pBuffer[0], DIG_BLOCKSIZE); // read requested block 
  	    }
  	    else
  	    {
  	        bytes_read = read(fd[0], pBuffer[0], DIG_BLOCKSIZE); // read requested block
//printf("bytes_read=%d\n", bytes_read);
  	        bytes_read = read(fd[0], pBuffer[0]+DIG_BLOCKSIZE, DIG_BLOCKSIZE); // read next block
//printf("bytes_read=%d\n", bytes_read);
  	    }
	}

        // Reading multiple files
	else
        {
            // For multiple files, read from each of the FDs.
            for (j=0; j<num_files; j++)
	    {
                lseek(fd[j], (selected_start_block*block_size),SEEK_SET);// move to requested block
  
                if(selected_start_block >= total_blocks-1) 
		{ 
		    selected_start_block = total_blocks-1;
  		    mult_bytes_read[j] = read(fd[j], pBuffer[j], block_size); // read requested block 
  		}
  		else
  		{
  		    mult_bytes_read[j] = read(fd[j], pBuffer[j], block_size); // read requested block
            mult_bytes_read[j] = read(fd[j], pBuffer[j] + samplesPerBlock, block_size); // read next block
  		}
            }
        }
    }
    else
    {
        if(num_blocks_direct != 0)
	{
    //        dll.hDllApiSetPioRegister(PIO_OFFSET_PCIE_RD_START_BLOCK, selected_start_block ); // move to requested block

            if(selected_start_block >= total_blocks-1)
	    {
                selected_start_block = total_blocks-1;
    //        bytes_read = read(boardp, pBuffer[0], block_size); // read requested block
            }
            else
	    { 
                read(boardp, pBuffer[0], block_size);
          //      bytes_read = read(boardp, pBuffer[0] + ss.samples_per_block, block_size);
            }
        }
        //Continuous acquisition mode is now handled by the timeout function
    }	



    if(multi_file == TRUE)
    {
        for (j=0; j<num_files; j++)
	{
            if(mult_bytes_read[j] == -1) { printf("-A Read Error Occured-\n"); }
        }
        printf("block %d read.\n", selected_start_block); 
    }
    else
    {
        if(bytes_read == -1) { printf("Single File Read Error!\n"); }
        else { printf("block %d read.\n", selected_start_block);  }
    }

    // Reset to zero windows plotted
    windows_plotted = 0;
    plot(widget);
    blocks_read = total_blocks+1; // ensure that in the expose_event() we don't just keep plotting blocks
}




// void plot(GtkWidget *widget)
// Utility function used to manage both block mode plotting and continuous acquistion (one window intervals)
// This functions provides minimal flow control and prepares for a new plot of data by clearing the pixel map
// PARAMS: Widget * widget -  provides access to the main window so it can be queued for a "repaint"
void plot(GtkWidget *widget)
{
    int start_index = 0;

    // Clear the current pixel map
    gdk_draw_rectangle(pixmap, widget->style->white_gc, TRUE, 0, 0, current_width, current_height); 

    if(block_mode)
    {
        // Update the starting index for continuation of plotting an entire block
        start_index = windows_plotted*current_width;
    }

    //usleep(50000);     // Heres the spot to place a sleep command

    // Plot the next window
    plot_window(widget, start_index);
}





// Utility function used to plot one window worth of data.
// PARAMS: Widget * widget -  provides access to the main window so it can be queued for a "repaint"
// int start_index -  is the index buffer where the plotting should start.  The last data point
// plotted will be pBuffer[start_index+current_width]
void plot_window(GtkWidget *widget, int start_index)
{
    int i, file_iter, chan_iter, j, index;
    int x_pixel_offset;
    //float scale; 102610
    u_int sample0[MAX_FILES], nextsample0[MAX_FILES], zero;
    u_int sample1[MAX_FILES], nextsample1[MAX_FILES];

    // Plotting function plots lines, so we need to pull 2 points
    // 2-D array to hold samples and nextsamples. One dimention is files, the other dimention is channels
    int samples[MAX_FILES][MAX_CHANNELS], nextsamples[MAX_FILES][MAX_CHANNELS];

    // Create an array to match the arrays above to hold the colors for plotting
    GdkGC * gc_color[MAX_FILES][MAX_CHANNELS];



    // --------------DISPLAY SETUP--------------

    // NOTE: IF MAX_FILES OR MAX_CHANNELS IS LARGER THAN 8, MORE COLORS NEED TO BE ADDED
    GdkFont *font;
    font = gdk_font_load ("-Adobe-Helvetica-Bold-R-Normal--*-140-*-*-*-*-*-*");
    GdkColor colors[MAX_FILES];
    colors[1]= red;
    colors[2]= blue;
    colors[3]= green;
    gdk_color_parse ("Orange", &colors[4]);
    colors[5]= yellow;
    colors[6]= pink;
    colors[7]= cyan;
    gdk_color_parse ("DarkViolet", &colors[8]);

    GdkGC* gc[MAX_FILES];
    GdkColormap *colormap[MAX_FILES];
 
    for (j=1; j<8; j++){
        gc[j]=gdk_gc_new(widget->window);
        colormap[j] = gdk_gc_get_colormap(gc[j]);
        gdk_rgb_find_color(colormap[j], &colors[j]);
        gdk_gc_set_foreground(gc[j], &colors[j]);
    }


    // This section of code sets up the gc_color array. It is 2D. 
    // If we are just reading one file, each channel is a different color.
    // If we are reading multiple files, each file's data is a different color.
    if( multi_file == FALSE)
    {
        // black is different for some reason
        gc_color[0][0] = widget->style->black_gc;

        // Assign each channel a different color
        for(chan_iter=1; chan_iter< adcChanUsed; chan_iter++)
        {
            gc_color[0][chan_iter] = gc[chan_iter];
        }

    }
    else  
    {
        // black is different for some reason
        for(chan_iter=0; chan_iter< adcChanUsed; chan_iter++)
        {
            gc_color[0][i] = widget->style->black_gc;
        }

        for(file_iter=1; file_iter< adcChanUsed; file_iter++)
        {
            for(chan_iter=0; chan_iter< adcChanUsed; chan_iter++)
            {
                gc_color[file_iter][chan_iter] = gc[file_iter];
            }

        }
    }
    // -------------- END DISPLAY SETUP--------------


    index = start_index;
 

    // ------------- MAIN PLOTTING LOOP--------------
    // Plot a point for each "column" in current window width  
    for(i=0; (i<current_width-1) && (index < 2*block_size); i++)
    {

        index = start_index + i;


        // For loop to read from each file        
        for(file_iter=0; file_iter < num_files ; file_iter++)
        {

            // For loop to read each channel
            for(chan_iter=0; chan_iter < adcChanUsed; chan_iter++)
            {
		if (bits == 8){
            if (adcChanUsed == 0){
                adcChanUsed = 2;
			}
                // Use the GetSample function to extract the samples from the data buffer NS 102610
                    samples[file_iter][chan_iter] = getSample8(adcChanUsed,pBuffer[file_iter], index, chan_iter); 		//dll.hDll.GetSample
                    nextsamples[file_iter][chan_iter] = getSample8(adcChanUsed,pBuffer[file_iter], index+1, chan_iter);
		}
		if (bits == 12){
            if (adcChanUsed == 0){
                adcChanUsed = 2;
			}
                // Use the GetSample function to extract the samples from the data buffer NS 102610
                    samples[file_iter][chan_iter] = getSample12(adcChanUsed,pBuffer[file_iter], index, chan_iter); 		//dll.hDll.GetSample
                    nextsamples[file_iter][chan_iter] = getSample12(adcChanUsed,pBuffer[file_iter], index+1, chan_iter);
		}
		if (bits == 14){
            if (adcChanUsed == 0){
                adcChanUsed = 2;
			}
                // Use the GetSample function to extract the samples from the data buffer NS 102610
                    samples[file_iter][chan_iter] = getSample14(adcChanUsed,pBuffer[file_iter], index, chan_iter); 		//dll.hDll.GetSample
                    nextsamples[file_iter][chan_iter] = getSample14(adcChanUsed,pBuffer[file_iter], index+1, chan_iter);
		}
		if (bits == 16){
            if (adcChanUsed == 0){
                adcChanUsed = 4;
			}
                // Use the GetSample function to extract the samples from the data buffer NS 102610
                    samples[file_iter][chan_iter] = getSample16(adcChanUsed,pBuffer[file_iter], index, chan_iter); 		//dll.hDll.GetSample
                    nextsamples[file_iter][chan_iter] = getSample16(adcChanUsed,pBuffer[file_iter], index+1, chan_iter);
		}

                // In window the y-value "grows downward" => reflect the y-value
                samples[file_iter][chan_iter] = (unsigned int)(current_height - samples[file_iter][chan_iter]*scale - 40);
                nextsamples[file_iter][chan_iter] = (unsigned int)(current_height - nextsamples[file_iter][chan_iter]*scale - 40); 

            }

        }

        // Set the value for the zero line (if applicable)
        zero = (u_int)(current_height - (u_int) zero_sample_value*scale - 40) ;
        


        //-------------Plotting Section--------------------------------

        // ----------------------------------
        // Plot text @ the top of the window to explain to the user what is going on 

        // Set x_pixel_offset to zero to start writing at the left edge of the screen
        x_pixel_offset = 0;

        // If we are just viewing a single set of data
        if( multi_file == FALSE)
        {
            //continuous acquisition mode
            if ((boardp != 0) && (num_blocks_direct == 0))
            {				
                gdk_draw_string(pixmap, font, widget->style->black_gc , 0, 15, continuous_acq);
                x_pixel_offset = x_pixel_offset + 300;
            }
            
            //reading from file
            else
            {						
	        gdk_draw_string(pixmap, font, widget->style->black_gc , 0, 15, reading);
                x_pixel_offset = x_pixel_offset + 145;

	    }

            for( chan_iter=0; chan_iter < adcChanUsed; chan_iter++)		//ss.adcChanUsed
            {
                gdk_draw_string(pixmap, font, gc_color[0][chan_iter] , x_pixel_offset, 15, chan_string[chan_iter]);
                x_pixel_offset = x_pixel_offset + 45;
            }

        }
        else  
        {
            //reading from file						
	    gdk_draw_string(pixmap, font, widget->style->black_gc , 0, 15, reading);
            x_pixel_offset = x_pixel_offset + 145;

            for( file_iter=0; file_iter < num_files; file_iter++)
            {
                gdk_draw_string(pixmap, font, gc_color[file_iter][0] , x_pixel_offset, 15, filenames[0]);
                x_pixel_offset = x_pixel_offset + 90;
            }

        }



        // ----------------------------------
        // Plot data  

#ifdef DRAW_ZERO
        gdk_draw_line(pixmap, widget->style->black_gc, i, zero, (i+1), zero);
#endif 


        // For loop to read from each file        
        for(file_iter=0; file_iter < num_files ; file_iter++)
        {

            // For loop to read each channel
            for(chan_iter=0; chan_iter < adcChanUsed; chan_iter++)		//ss.adcChanUsed
            {
                gdk_draw_line(pixmap,
                              gc_color[file_iter][chan_iter],
                              i,
                              samples[file_iter][chan_iter],
                              (i+1),
                              nextsamples[file_iter][chan_iter]
			     );
            }

        }
        
	//-------------Plotting Section--------------------------------


    } 

    // ------------- MAIN PLOTTING LOOP------------------------------------


    if(block_mode)
    {
        // windows_plotted = windows_per_dma_block causes the end of plotting
        // Increment the number of "windows" worth of data we have printed and slow the plotting process
        windows_plotted++;
    }


    gtk_widget_queue_draw_area(widget, 0, 0, current_width, current_height);   // Redraw the canvas
}




// Event Handler for window exposure
// This function is implicitly called by the plot_window function via the call to gtk_widget_queue_draw_area(),
// which request a redraw of the drawing area (canvas)
// PARAMS: Widget * widget -  provides access to the main window
// GdkEventExpose *event - provides information about the event	
static int expose_event(GtkWidget *widget, GdkEventExpose *event)
{
   
  // Place the pixel map in the window
  gdk_draw_pixmap(widget->window, widget->style->fg_gc[GTK_WIDGET_STATE(widget)],
		  pixmap, event->area.x, event->area.y,
		  event->area.x, event->area.y,
		  event->area.width, event->area.height);

    if(first == 0)
    {
		
#ifdef _CHECK_MEM
//        printf("Checking Memory\n");
//	CheckMem(0,total_blocks, fd[0], pBuffer[0]);        //dll.hDllCheckMem
#endif
        first = 1;
    }


 // This function is called on the initial creation of the window, don't allow recursive
 // calls to plot_window function until we have started the update process (windows_plotted 
 // would never get incremented ->infinite loop)!
 // Also if selecting_byte is asserted we dont want recursive calls becuase we already plotted the requested screen
  if((windows_plotted > 0) && !selecting_byte)
  {
    // Call plot() if still have data to plot
    if( (windows_plotted < windows_per_dma_block) && (boardp == 0) )          
      {
        plot(widget);
//printf("calling plot wp=%d, wpdb=%d\n",windows_plotted,  windows_per_dma_block);
      }
     else
      {
	// if reading from file then "update" until all blocks read
	// note that if update button is pressed we set blocks_read
	// to total_blocks+1, therefore this is run only inital creation of window
/*
	if(blocks_read < total_blocks)
	 {
		if(read(fd, pBuffer, block_size))
		{
		  printf("block %d read\r", blocks_read);
		  fflush(stdout);	
		}
   		// Reset to zero windows plotted
   		windows_plotted = 0;
   		plot(widget);
   		blocks_read++;
	   //update(widget);
	 }
*/

      }			
  }
  
  return FALSE;
}





// Event Handler for a change in the value of the slider
// PARAMS: Widget * widget -  provides access to the main window
static void scale_value_changed(GtkWidget * widget, gpointer data)
{
    // Save this value so that if the window is resized we can re-plot it
    selected_start_byte = (int)gtk_range_get_value(GTK_RANGE(widget));  
    // Set this var so that we plot just a single window and dont do the entire block mode process
    selecting_byte = TRUE;      


    gtk_toggle_button_set_active( GTK_TOGGLE_BUTTON(singleblock_button), TRUE);
    block_mode = TRUE;

    // Clear the current pixel map
    gdk_draw_rectangle(pixmap, widget->style->white_gc, TRUE, 0, 0, current_width,current_height); 


    //printf("ssb=%d\n", selected_start_byte);

    plot_window( GTK_WIDGET(widget) , selected_start_byte);

}





// Configure event handler for the GTK window
// PARAMS: Widget * widget -  provides access to the main window
//	gpointer data - not used	
static int configure_event(GtkWidget *widget, GdkEventConfigure *event)
{
    if(pixmap)
    {
        // Dump the old one
        g_object_unref(pixmap);
    }

    // Allocate a new pixmap using the new window size
    pixmap = gdk_pixmap_new(widget->window, widget->allocation.width, widget->allocation.height, -1);
	
    gdk_draw_rectangle(pixmap, widget->style->white_gc, TRUE, 0, 0, widget->allocation.width,
									widget->allocation.height);
	
    //Save the current size of the window for later use
    current_width = widget->allocation.width;
    current_height = widget->allocation.height;

    //102610
    scale = (float) (current_height/1.2)/ scale_coefficient;



    //Record number of "windows" worth of data we will need to diplay per DMA block
    //windows_per_dma_block = buffer_LENGTH/current_width;  //AD8-1500
    windows_per_dma_block = samplesPerBlock/(current_width); 	//block_size/(current_width);	//ss.samples_per_block


    // Also in case the user was byte selecting and just resizing we need to repaint the requested data
    // The widget we recieved was a GtkDrawingArea, so we can't just use gtk_range_get_value() to get the start_index
    // => used the saved value
    plot_window( GTK_WIDGET(widget) , selected_start_byte );
  
    return TRUE;
}



// Destroy event handler for the GTK window.
// PARAMS: Widget * widget -  provides access to the main window
// gpointer data - not used 	
static void destroy ( GtkWidget *widget, gpointer data)
{
    gtk_main_quit();
}



// Event Handler that is called periodically. Used in continuous acquisition mode
// PARAMS: gpointer data - is the widget. Used to redraw the screen after we acquire new data. 	
gboolean timer_callback (GtkWidget * data) 
{

	int bytes_read;

	// Set vars to passed to the setup routines to prepare the board(s) for capture
//	ss.num_devices = 1;
//	ss.blocks_to_acquire = 64;
//    ss.pBuffer = pBuffer[0];
//        //setup_bd(&boardp, &ss); DEBUG IF ADDING CONTINUOUS ACQUIRE

//    	read(boardp, pBuffer[0], block_size);
//    	bytes_read = read(boardp, pBuffer[0]+ ss.samples_per_block, block_size);
    	windows_plotted = 0;
   	plot(data);
    	gtk_widget_queue_draw(data);

    	return TRUE;
}


void digosc_cleanup()
{
	int i;

	for(i=0; i< MAX_FILES; i++)
	{
	  if(pBuffer[i] != NULL)
	  {
	    free(pBuffer[i]);
	  }
	}


	return;
}





void digosc_parser_printf()
{
    printf("\n");
    printf("digosc reads A/D data stored in a disk file and displays it as a waveform. Usage:\n");
    printf("digosc          Reads from uvdma.dat.\n");    
    printf("-f (FileName)   Reads from FileName.\n");
    printf("-m (NumFiles)   Read from multiple files- uvdma.dat, uvdma1.dat, uvdma2.dat...\n");
    printf("-8,-12,-14,-16  Look at the data from 8bit, 12bit, 14bit, or 16bit board type.\n");
    printf("-scm            Interpret file data as single channel data. Not required for single channel boards. This should be the same as option supplied to acquire.\n");
    printf("-dcm            Interpret the data as dual channel data. Only required for 4 channel boards acquiring in 2 channel mode.\n");
    printf("-v              Enable extra print statements.\n");
    printf("-skip_graph     Disable plotting and check for glitches.\n");
//    printf("-d (N blocks)   Direct acquisition mode. Acquires the (N blocks) directly from board. NOTE: Direct mode only work with a single board.\n");
}



void digosc_parser(int argc, char ** argv) //setup_struct *pSS
{

    int arg_index;

    multi_file = false;    // default to 1 file
    num_files = 1;         // default to 1 file

    // check how many arguments, if run without arguments prints usage.
    if(argc == 1)
    {
        digosc_parser_printf();
	exit(1);
    }
    else 
    { 
        // starting at 2nd arguement look for options
        for(arg_index=1; arg_index<argc; arg_index++)
        {

            if( strcmp(argv[arg_index], "-f") == 0 )
            {
                // make sure option is followed by (FileName)
                if(argc > arg_index)
                {
                    arg_index++; // increment the arguement index b/c we have now taking care of the next arguement
                    strcpy(file_name, argv[arg_index]);
                    printf("reading from %s.\n", file_name);
                }
                else
                {
                    printf("-f option must be followed by FileName\n");
                    digosc_cleanup();
                    exit(1);
                }
            }
            else if( strcmp(argv[arg_index], "-m") == 0 )
            {
                // make sure option is followed by (NumFiles)
                if(argc > arg_index)
                {
                    arg_index++; // increment the arguement index b/c we have now taking care of the next arguement
                    num_files = atoi(argv[arg_index]);

                    printf("Reading from %d files...\n", num_files);
                    multi_file = true;
                }
                else
                {
                    printf("-m option must be followed by NumFiles\n");
                    digosc_cleanup();
                    exit(-1);
                }

                if(num_files > MAX_FILES)
	        {
                   printf("Number of files requested for open is greater than MAX_FILES. MAX_FILES=%d. Exiting. \n",MAX_FILES);
                   digosc_cleanup();
                   exit(-1);
                }
            }

//8,12,14,16 bits
            else if( strcmp(argv[arg_index], "-8") == 0 )
            {
                printf("8 bit board selected\n");
                bits = 8;
                adcChanUsed = 2;
            }
            else if( strcmp(argv[arg_index], "-12") == 0 )
            {
                printf("12 bit board selected\n");
                bits = 12;
                adcChanUsed = 2;

            }
            else if( strcmp(argv[arg_index], "-14") == 0 )
            {
                printf("14 bit board selected\n");
 		bits = 14;
        adcChanUsed = 2;
            }
            else if( strcmp(argv[arg_index], "-16") == 0 )
            {
                printf("16 bit board selected\n");
                adcChanUsed = 4;
        bits = 16;
            }
//8,12,14,16 bits

            else if( strcmp(argv[arg_index], "-scm") == 0 )
            {
                //printf("Running board in single channel mode\n"); 
        adcChanUsed = 1;
            }
            else if( strcmp(argv[arg_index], "-dcm") == 0 )
            {
                //printf("Running board in dual channel mode\n"); 
        adcChanUsed = 2;
            }
            else if( strcmp(argv[arg_index], "-v") == 0 )
            {
                //printf("Running in verbose mode\n"); 
                verbose = true;
            }
            else if( strcmp(argv[arg_index], "-skip_graph") == 0 )
            {
                skip_graph = true;
            }

            else
            {
                printf("Invalid option specified! Check syntax.\n");
                exit(1);
            }
        }
    }

}






