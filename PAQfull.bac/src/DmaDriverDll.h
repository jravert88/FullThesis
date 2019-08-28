// -------------------------------------------------------------------------
// 
// PRODUCT:			DMA Driver
// MODULE NAME:		DmaDriverDll.h
// 
// MODULE DESCRIPTION: 
// 
// Contains defines, structures and exported functions for the DLL-like interface.
// 
// $Revision:  $
//
// ------------------------- CONFIDENTIAL ----------------------------------
// 
//              Copyright (c) 2011 by Northwest Logic, Inc.   
//                       All rights reserved. 
// 
// Trade Secret of Northwest Logic, Inc.  Do not disclose. 
// 
// Use of this source code in any form or means is permitted only 
// with a valid, written license agreement with Northwest Logic, Inc. 
// 
// Licensee shall keep all information contained herein confidential  
// and shall protect same in whole or in part from disclosure and  
// dissemination to all third parties. 
// 
// 
//                        Northwest Logic, Inc. 
//                  1100 NW Compton Drive, Suite 100 
//                      Beaverton, OR 97006, USA 
//   
//                        Ph:  +1 503 533 5800 
//                        Fax: +1 503 533 5900 
//                      E-Mail: info@nwlogic.com 
//                           www.nwlogic.com 
// 
// -------------------------------------------------------------------------

#ifdef WIN32

	#include "Setupapi.h"
#else
	//#include "../DmaDriverCli/StdTypes.h"
	#include "StdTypes.h"
#endif // Windows vs. Linux

//#include "../../Include/version.h"
//#include "../../Include/DmaDriverIoctl.h"
#include "version.h"
#include "DmaDriverIoctl.h"

#ifndef _DMADRIVERDLL_H
#define _DMADRIVERDLL_H


// Defines
//
#ifdef WIN32
	// Maximum number of boards
	#define MAXIMUM_NUMBER_OF_BOARDS   4

	// The following ifdef block is the standard way of creating macros which make exporting
	// from a DLL simpler. All files within this DLL are compiled with the DMADRIVERDLL_EXPORTS
	// symbol defined on the command line. this symbol should not be defined on any project
	// that uses this DLL. This way any other project whose source files include this file see
	// DMADRIVERDLL_API functions as being imported from a DLL, whereas this DLL sees symbols
	// defined with this macro as being exported.
	#ifdef DMADRIVERDLL_EXPORTS
		#define DMADRIVERDLL_API extern "C" __declspec(dllexport)
	#else
		#define DMADRIVERDLL_API __declspec(dllimport)
	#endif

#else
	// Maximum number of boards
	#define MAXIMUM_NUMBER_OF_BOARDS   4
	#define DMADRIVERDLL_API

#endif // Windows vs. Linux

// DMA_INFO_STRUCT
//
// DMA Info Structure - Contains information on the type and 
//	quantity of DMA Engines
typedef struct _DMA_INFO_STRUCT
{
	char		PacketSendEngineCount;
	char		PacketRecvEngineCount;
	char		PacketSendEngine[MAX_NUM_DMA_ENGINES];
	char		PacketRecvEngine[MAX_NUM_DMA_ENGINES];
	char		DLLMajorVersion;
	char		DLLMinorVersion;
	char		DLLSubMinorVersion;
	char		DLLBuildNumberVersion;
	char		AddressablePacketMode;
} DMA_INFO_STRUCT, *PDMA_INFO_STRUCT;

#define	S2C_DIRECTION		TRUE
#define C2S_DIRECTION		FALSE

typedef struct _PACKET_GENCHK_STRUCT
{
	ULONG32		Control;			// Packet Generator Control DWORD
	ULONG32		NumPackets;			// Count of packet to generate, 0 = infinite
	ULONG32		DataSeed;			// Data Seed pattern
	ULONG32		UserCtrlStatSeed;	// Seed for the User Control/Status fields
	ULONG32 	PacketLength[4];	// Packet Length array
} PACKET_GENCHK_STRUCT, *PPACKET_GENCHK_STRUCT;


// STAT_STRUCT
//
// Status Structure - Status Information from an IOCTL transaction
typedef struct _STAT_STRUCT
{
    ULONGLONG   CompletedByteCount; // Number of bytes transfered
} STAT_STRUCT, *PSTAT_STRUCT;


// ----------------------
// Externally Visible API

// ConnectToBoard
//
// Connects to a board.  'board' is used to select between
//   multiple board instances (more than 1 board may be present);
//   If the board class has not been created, it does it now.

DMADRIVERDLL_API UINT ConnectToBoard
    (
    UINT	board,
	PDMA_INFO_STRUCT	pDmaInfo
    );



// DisconnectFromBoard
//
// Disconnect from a board.  'board' is used to select between
//   multiple board instances (more than 1 board may be present);
// If the board class has not been created, return;  
//   otherwise, call DisconnectFromBoard.  Does not delete the board class.

DMADRIVERDLL_API UINT DisconnectFromBoard
    (
    UINT board                  // Board to target
    );



// GetBoardCfg
//
// Get Board Configuration from the Driver
//
// The Driver auto discovers the board's resources during Driver
//   initialization (via hardware capabilities and configuration
//   register advertisements) and places the information in a
//   BOARD_CONFIG_STRUCT structure. BOARD_CONFIG_STRUCT provides
//   the necessary information about the board so that the
//   application knows what resources are available and how to
//   access them.
//
// GetBoardCfg gets a copy of the current BOARD_CONFIG_STRUCT
//   structure kept by the Driver.  No hardware accesses are
//   initiated by calling GetBoardCfg.

DMADRIVERDLL_API UINT GetBoardCfg
    (
    UINT board,                 // Board to target
    BOARD_CONFIG_STRUCT* Board  // Returned structure
    );


// GetDMAEngineCap
//
// Gets the DMA Engine Capabilities of DMA Engine number DMAEngine
//   on board number 'board' and returns them in DMACap.
//
// DMA Engine Capabilities are defined as follows::
//   DmaCap[    0] : 1 == Engine Present; 0 == Engine not present
//   DmaCap[ 2: 1] : Direction
//                   00 == System to Card Engine (Write to Card)
//                   01 == Card to System Engine (Read from Card)
//                   10 == Reserved
//                   11 == Bidirectional (Write and Read)
//   DmaCap[    3] : Reserved
//   DmaCap[ 5: 4] : Programming Model
//                   00 - Block DMA - Not Supported
//                   01 - Packet DMA
//                   10 - Reserved
//                   11 - Reserved
//   DmaCap[ 7: 6] : Reserved
//   DmaCap[15: 8] : EngineNumber[7:0]
//                   Unique identifying number for this DMA Engine
//   DmaCap[23:16] : ImplCardAddresWidth[7:0]
//                   Size in bytes of the Card Address space that this
//                   DMA Engine is connected to == 2^ImplCardAddresWidth.
//                   ImplCardAddresWidth == 0 indicates that the DMA Engine
//                   is connected to a Stream/FIFO and does not implement
//                   addressable Card Memory
//   DmaCap[31:24] : Reserved
//
// Returns DMA Capabilities in the DMACap sent to the call.

DMADRIVERDLL_API UINT GetDMAEngineCap
	(
    UINT board,				// Board number to target
    ULONG DMAEngine,		// DMA Engine number to use
    PDMA_CAP_STRUCT DMACap	// Returned DMA Engine Capabilitie
	);



// GetDmaPerf
//
// Gets DMA Performance information from board 'board'
//   and DMA Engine 'EngineNum'.
//
// The DMA Engine and Driver record performance information
//   when DMA operations are in process.  This information
//   is obtained by calling GetDmaPerf.
//
// Returns the DMA performance status in the DMA_STAT_STRUCT sent to the call.

DMADRIVERDLL_API UINT GetDmaPerf
    (
    UINT board,             // Board number to target
    int EngineOffset,		// DMA Engine number offset to use
	UINT TypeDirection,     // DMA Type (Block / Packet) & Direction Flags
    PDMA_STAT_STRUCT Status // Returned performance metrics
    );

// WRITE PCI Configuration space
//
// Sends a WRITE_PCI_CONFIG_IOCTL call to the driver.
//
// Copies to 'Offset' from the beginning of the PCI Configuration space for our 'board'
//  from application's memory 'Buffer' for the 'Length' specified.
// 
// Returns Completion status in the STAT_STRUCT sent to the call.
//
// Function added as of version 4.6.x.x
//
DMADRIVERDLL_API UINT WritePCIConfig
    (
    UINT board,             // Board number to target
    BYTE *Buffer,           // Data buffer
    ULONG Offset,			// Offset in PCI Config space to start transfer
    ULONG Length,			// Byte length of transfer
    PSTAT_STRUCT Status     // Completion Status
    );

// READ PCI Configuration space
//
// Sends a READ_PCI_CONFIG_IOCTL call to the driver.
//
// Copies from 'Offset' from the beginning of the PCI Configuration space for our 'board'
//  to application's memory 'Buffer' for the 'Length' specified.
// 
// Returns Completion status in the STAT_STRUCT sent to the call.
//
// Function added as of version 4.6.x.x
//
DMADRIVERDLL_API UINT ReadPCIConfig
    (
    UINT board,             // Board number to target
    BYTE *Buffer,           // Data buffer
    ULONG Offset,			// Offset in PCI Config space to start transfer
    ULONG Length,			// Byte length of transfer
    PSTAT_STRUCT Status     // Completion Status
    );


// DoMem
//
// Sends a DoMem IOCTL call to the driver.
//
// Uses the system CPU to perform a memory copy between 
//   application's memory 'Buffer' starting at byte offset 'Offset' and
//   board 'board' Base Addres Register (BAR) 'BarNum' starting at byte offset
//   'CardOffset'.  The copy operation length in bytes is specified by 'Length'.
// 
// DoMem is primarily intended for control reads and writes.
//   DoMem is not useful for high bandwidth data transfers because
//   the system CPU uses very small burst sizes which results in
//   poor efficiency.
// 
// Returns Completion status in the STAT_STRUCT sent to the call.

DMADRIVERDLL_API UINT DoMem
    (
    UINT board,             // Board number to target
    UINT Rd_Wr_n,           // 1==Read, 0==Write
    UINT BarNum,            // Base Address Register (BAR) to access
    BYTE *Buffer,           // Data buffer
    ULONGLONG Offset,       // Offset in data buffer to start transfer
    ULONGLONG CardOffset,   // Offset in BAR to start transfer
    ULONGLONG Length,       // Byte length of transfer
    PSTAT_STRUCT Status     // Completion Status
    );


// PacketReceive
//
// Use for FIFO Packet DMA only: 

DMADRIVERDLL_API UINT PacketReceive
    (
    UINT	board,			// Board number to target
    int		EngineOffset,	// DMA Engine number offset to use
	ULONG * BufferToken,	// Token for the returned buffer
	void *	Buffer,			// Pointer to the recived packet buffer
	ULONG *	Length			// Length of the received packet
    );

// PacketReceiveEx - Extended functionality - UserStatus
//
// Send a PACKET_RECEIVE_IOCTL call to the driver and waits for a completion
//
// Returns Completion status.
DMADRIVERDLL_API UINT PacketReceiveEx
	(
	UINT		board, 		// Board to target
	int			EngineNum,  // DMA Engine number offset to use
	ULONGLONG *	UserStatus, // User Status returned from the EOP DMA Descriptor
	ULONG *		BufferToken, // Token for the returned buffer
	void *		Buffer,   	// Pointer to the recived packet buffer
	ULONG *		Length		// Length of the received packet
	);

// PacketReturnReceive
//
// Use for FIFO Packet DMA only: 

DMADRIVERDLL_API UINT PacketReturnReceive
	(
    UINT	board,			// Board number to target
    int		EngineOffset,	// DMA Engine number offset to use
	ULONG * BufferToken		// Token for the buffer to return
	);

// PacketReceives
//
// Multiple Packet receive function for use with FIFO Packet DMA only: 
DMADRIVERDLL_API UINT PacketReceives
	(
    UINT	board,			// Board number to target
	int		EngineNum,      // DMA Engine number offset to use
	PPACKET_RECVS_STRUCT	pPacketRecvs // Pointer to Packet Receives struct
	);

// PacketSend
//
// Use for FIFO Packet DMA only: 

DMADRIVERDLL_API UINT PacketSend
    (
    UINT	board,			// Board number to target
    int		EngineOffset,	// DMA Engine number offset to use
    ULONGLONG CardOffset,   // Address of Memory on the card
	UCHAR *	Buffer,			// Pointer to the packet buffer to send
	ULONG 	Length			// Length of the send packet
    );

// PacketSendEx	- Extended Functionality - User Control
//
// Send a PACKET_SEND_IOCTL call to the driver
//
// Returns Completion status.

DMADRIVERDLL_API UINT PacketSendEx
	(
	UINT	board,      	// Board to target
	int		EngineOffset,	// DMA Engine number offset to use
	ULONGLONG UserControl,  // User Control to set in the first DMA Descriptor
	ULONGLONG CardOffset,   // Address of Memory on the card
	UCHAR *	 Buffer,        // Pointer to the packet buffer to send
	ULONG 	Length			// Length of the send packet
	);

// PacketRead
//
// Use for Addressable Packet DMA only: 
DMADRIVERDLL_API UINT PacketRead
	(
	UINT 	board,          // Board to target
	int		EngineOffset,   // DMA Engine number offset to use
	ULONGLONG * UserStatus, // User Status returned from the EOP DMA Descriptor
	ULONGLONG CardOffset,   // Card Address to start read from
	UCHAR *	Buffer,         // Address of data buffer
	ULONG *	Length			// Length to Read
	);

// PacketWrite
//
// Use for Addressable Packet DMA only: 
DMADRIVERDLL_API UINT PacketWrite
	(
	UINT 	board,			// Board to target
	int		EngineOffset,	// DMA Engine number offset to use
	ULONGLONG UserControl,  // User Control to set in the first DMA Descriptor
	ULONGLONG CardOffset,   // Card Address to start write to
	UCHAR *	Buffer,         // Address of data buffer
	ULONG 	Length			// Length of data packet
	);

// PacketReadEx
//
// Use for Addressable Packet DMA only: 
DMADRIVERDLL_API UINT PacketReadEx
	(
	UINT 	board,          // Board to target
	int		EngineOffset,   // DMA Engine number offset to use
	ULONGLONG * UserStatus, // User Status returned from the EOP DMA Descriptor
	ULONGLONG CardOffset,   // Card Address to start read from
	ULONG	Mode,			// Control Mode Flags
	UCHAR *	Buffer,         // Address of data buffer
	ULONG *	Length			// Length to Read
	);

// PacketWriteEx
//
// Use for Addressable Packet DMA only: 
DMADRIVERDLL_API UINT PacketWriteEx
	(
	UINT 	board,			// Board to target
	int		EngineOffset,	// DMA Engine number offset to use
	ULONGLONG UserControl,  // User Control to set in the first DMA Descriptor
	ULONGLONG CardOffset,   // Card Address to start write to
	ULONG	Mode,			// Control Mode Flags
	UCHAR *	Buffer,         // Address of data buffer
	ULONG 	Length			// Length of data packet
	);

// SetupPacketMode
//
// Use for Packet DMA only: 

DMADRIVERDLL_API UINT SetupPacketMode
    (
    UINT	board,          // Board number to target
    int		EngineOffset,	// DMA Engine number offset to use
    BYTE *	Buffer,         // Data buffer (FIFO Mode)
    ULONG *	BufferSize,     // Size of the Packet Recieve Data Buffer requested (FIFO Mode)
    ULONG *	MaxPacketSize,	// Length of largest packet (FIFO Mode)
    int		PacketMode,		// Sets mode, FIFO or Addressable Packet mode
    int		NumberDescriptors	// Number of DMA Descriptors to allocate (Addressable mode)	
    );

// ShutdownPacketMode
//
// Use for Packet DMA only: 
DMADRIVERDLL_API UINT ShutdownPacketMode
	(
    UINT	board,			// Board number to target
	int		EngineOffset    // DMA Engine number offset to use
	);

// SetupPacketGenerator
//
// Use for Packet DMA only: 

DMADRIVERDLL_API UINT SetupPacketGenerator
    (
    UINT	board,          // Board number to target
    int		EngineOffset,	// DMA Engine number offset to use
	BOOLEAN	SendDirection,	// True for Pattern Checker, False for Pattern Generator
	PPACKET_GENCHK_STRUCT pPacketGenChk		// pointer to Packet Generator / Pattern Checker structure
    );

#endif  // _DMADRIVERDLL_H
