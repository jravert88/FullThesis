// -------------------------------------------------------------------------
// 
// PRODUCT:			DMA Driver
// MODULE NAME:		DmaDriverDll.cpp
// 
// MODULE DESCRIPTION: 
// 
// Contains the function entry points for the DLL-like applications.
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

#include <malloc.h>
#include "DmaDriverDll.h"

#include <unistd.h>
#include <signal.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <errno.h>
#include <fcntl.h>
#include <memory.h>

#include <stdint.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <errno.h>
#include <fcntl.h>
#include <memory.h>
#include "StdTypes.h"


// BOARD_INFO_STRUCT
//
// Board Info Structure - Contains information for all the board on this system and 
// the state of the driver interface
//
typedef struct _BOARD_INFO_STRUCT
{
	DMA_INFO_STRUCT	DmaInfo;
	bool			AttachedToDriver;
	int 			hDevice;
	int				AllocationMode[MAX_NUM_DMA_ENGINES];
} BOARD_INFO_STRUCT, *PBOARD_INFO_STRUCT;

				   
BOARD_INFO_STRUCT	gBoardInfo[MAXIMUM_NUMBER_OF_BOARDS];

#define DBG		1

//--------------------------------------------------------------------
//
//	Private DLL like Interface
//
//--------------------------------------------------------------------



// ConnectToBoard
//
// This connects to the board you are interested in talking to.
UINT ConnectToBoard(
    UINT 				board,  		// Board to target
	PDMA_INFO_STRUCT	pDmaInfo
   )
{
	DMA_CAP_STRUCT			DMACap;
	char					i;
	char                    DeviceName[sizeof(PROCFS_PATH) + sizeof(PROCFS_FB_NAME) + 3];
	char *					pDeviceName = &DeviceName[0];

	// Make sure the "Board" is valid
	if (board >= MAXIMUM_NUMBER_OF_BOARDS)
	{
		printf("%d is an invalid board number\n", board);
		return STATUS_INVALID_BOARDNUM;
	}

	gBoardInfo[board].AttachedToDriver = false;
	gBoardInfo[board].hDevice = INVALID_HANDLE_VALUE;

	
	
	sprintf(pDeviceName, "%s%d/%s", PROCFS_PATH, board, PROCFS_FB_NAME);

	//----DEBUG-----
	printf("About to connect...\n");
	printf("Name: %s\n", DeviceName);
	//----DEBUG-----

	gBoardInfo[board].hDevice = open(DeviceName, O_RDWR);
	if (gBoardInfo[board].hDevice == -1)
	{
		if (errno == EACCES)
		{
			printf("Failed to connect to driver, insufficient priviledge\n");
		}
		printf("Failed to connect to device driver: %s\n", DeviceName);
		return STATUS_INCOMPLETE;
	}
	
	// set flag for other function calls
	gBoardInfo[board].AttachedToDriver = true;
	
	// Assume no DMA Engines found
	gBoardInfo[board].DmaInfo.PacketRecvEngineCount = 0;
	gBoardInfo[board].DmaInfo.PacketSendEngineCount = 0;
	gBoardInfo[board].DmaInfo.AddressablePacketMode = false;
	
	// Get DMA Engine cap to extract the engine numbers for the packet and block mode engines
	for (i = 0; i < MAX_NUM_DMA_ENGINES; i++)
	{
		gBoardInfo[board].DmaInfo.PacketRecvEngine[i] = -1;
		gBoardInfo[board].DmaInfo.PacketSendEngine[i] = -1;
		gBoardInfo[board].AllocationMode[i] = DMA_MODE_NOT_SET;
	
		if (GetDMAEngineCap(board, i, &DMACap) == STATUS_SUCCESSFUL)
		{
			//printf("Engine %u Capabilites: 0x%lx\n", DMACap.EngineNum, DMACap.DmaCapabilities);
			if ((DMACap.DmaCapabilities & DMA_CAP_ENGINE_PRESENT) == DMA_CAP_ENGINE_PRESENT)
			{
				if ((DMACap.DmaCapabilities & DMA_CAP_ENGINE_TYPE_MASK) & DMA_CAP_PACKET_DMA)
				{
					//printf("Found Packet Engine[%d] ", i);
					if ((DMACap.DmaCapabilities & DMA_CAP_ENGINE_TYPE_MASK) & DMA_CAP_ADDRESSABLE_PACKET_DMA)
					{
						//printf("Addressable Packet Mode supported, ");
						gBoardInfo[board].DmaInfo.AddressablePacketMode = true;
					}
					if ((DMACap.DmaCapabilities & DMA_CAP_DIRECTION_MASK) == DMA_CAP_SYSTEM_TO_CARD)
					{
						gBoardInfo[board].DmaInfo.PacketSendEngine[gBoardInfo[board].DmaInfo.PacketSendEngineCount++] = i;
						//printf("of type S2C (Send)\n");
					}
					else if ((DMACap.DmaCapabilities & DMA_CAP_DIRECTION_MASK) == DMA_CAP_CARD_TO_SYSTEM)
					{
						gBoardInfo[board].DmaInfo.PacketRecvEngine[gBoardInfo[board].DmaInfo.PacketRecvEngineCount++] = i;
						//printf("of type C2S (Recv)\n");
					}
					else
					{
						//printf("of invalid type\n");
					}
				}
				else
				{
					//printf("of invalid type\n");
				}
			}
		}
	}
	gBoardInfo[board].DmaInfo.DLLMajorVersion = VER_MAJOR_NUM;
	gBoardInfo[board].DmaInfo.DLLMinorVersion = VER_MINOR_NUM;
	gBoardInfo[board].DmaInfo.DLLSubMinorVersion = VER_SUBMINOR_NUM;
	gBoardInfo[board].DmaInfo.DLLBuildNumberVersion = VER_BUILD_NUM;
		
	memcpy(pDmaInfo, &gBoardInfo[board].DmaInfo, sizeof(DMA_INFO_STRUCT));
	return STATUS_SUCCESSFUL;
}

// DisconnectFromBoard
//
// Disconnect from a board.
// Cleanup any global data structures that have been created.
UINT DisconnectFromBoard(
	UINT 				board   		// Board to target
	)
{
	int		i;

	// Make sure the 'Board' is valid
	if (board >= MAXIMUM_NUMBER_OF_BOARDS)
	{
		return STATUS_INVALID_BOARDNUM;
	}

	gBoardInfo[board].AttachedToDriver = false;
	
	if (gBoardInfo[board].hDevice != INVALID_HANDLE_VALUE)
	{
		// Make sure the Packet Mode engines are shutdown
		for (i = 0; i < gBoardInfo[board].DmaInfo.PacketRecvEngineCount; i++)
		{
			if (gBoardInfo[board].AllocationMode[i] != DMA_MODE_NOT_SET)
			{
				ShutdownPacketMode(board, i);
			}
		}
		close(gBoardInfo[board].hDevice);
		gBoardInfo[board].hDevice = INVALID_HANDLE_VALUE;
	}
	
	// Reset to no DMA Engines found
	gBoardInfo[board].DmaInfo.PacketRecvEngineCount = 0;
	gBoardInfo[board].DmaInfo.PacketSendEngineCount = 0;
	return STATUS_SUCCESSFUL;
}


// GetBoardCfg
//
// Sends a GetBoardCfg IOCTL call to the driver.
// Returns the data in the BOARD_CONFIG_STRUCT sent to the call.
UINT GetBoardCfg
	(
	UINT 				board,     		// Board to target
	BOARD_CONFIG_STRUCT* Board
	)
{
	UINT			status;

	// Make sure the "Board" is valid
	if (board >= MAXIMUM_NUMBER_OF_BOARDS)
	{
		return STATUS_INVALID_BOARDNUM;
	}

	// Send GET_BOARD_CONFIG_IOCTL
	status = ioctl (gBoardInfo[board].hDevice, GET_BOARD_CONFIG_IOCTL_BASE, Board);
	if (status != 0)
	{
		// ioctl failed
#if DBG
		printf("GetBoardCfg IOCTL call failed.  Status=0x%x, Error = 0x%x\n", status, errno);
#endif // DBG
		return errno;
	}
	return STATUS_SUCCESSFUL;
}

// WritePCIConfig
//
// Sends a WRITE_PCI_CONFIG_IOCTL call to the driver.
//
// Copies to 'Offset' from the beginning of the PCI Configuration space for our 'board'
//  from application's memory 'Buffer' for the 'Length' specified.
//
// The STAT_STRUCT is updated with the status information returned by the driver
// for this transaction.
//
// Returns: Processing status of the call.

UINT WritePCIConfig
	(
	UINT board,             // Board number to target
	BYTE *Buffer,           // Data buffer
	ULONG Offset,			// Offset in PCI Config space to start transfer
	ULONG Length,			// Byte length of transfer
	PSTAT_STRUCT Status     // Completion Status
	)
{
	RW_PCI_CONFIG_STRUCT	PCIConfig;
    DWORD 			bytesReturned = 0;
    UINT			status;
	
	if (board >= MAXIMUM_NUMBER_OF_BOARDS)
	{
		return STATUS_INVALID_BOARDNUM;
	}
	PCIConfig.Offset 	= Offset;
	PCIConfig.Length	= Length;
	PCIConfig.Buffer	= Buffer;
	// Send WRITE_PCI_CONFIG
	status = ioctl (gBoardInfo[board].hDevice, WRITE_PCI_CONFIG_IOCTL, &PCIConfig);
	if (status != 0)
	{
		// ioctl failed
		Status->CompletedByteCount = 0;
#if DBG
		printf("WritePCIConfig IOCTL call failed.  Status=0x%x, Error = 0x%x\n", status, errno);
#endif // DBG
		return errno;
	}
	// save the returned size
	Status->CompletedByteCount = bytesReturned;
	return STATUS_SUCCESSFUL;
}

// ReadPCIConfig
//
// Sends a READ_PCI_CONFIG_IOCTL call to the driver.
//
// Copies from 'Offset' from the beginning of the PCI Configuration space for our 'board'
//  to application's memory 'Buffer' for the 'Length' specified.
//
// The STAT_STRUCT is updated with the status information returned by the driver
// for this transaction.
//
// Returns: Processing status of the call.

UINT ReadPCIConfig
	(
	UINT board,             // Board number to target
	BYTE *Buffer,           // Data buffer
	ULONG Offset,			// Offset in PCI Config space to start transfer
	ULONG Length,			// Byte length of transfer
	PSTAT_STRUCT Status     // Completion Status
	)
{
	RW_PCI_CONFIG_STRUCT	PCIConfig;
    DWORD 			bytesReturned = 0;
    UINT			status;
	
	if (board >= MAXIMUM_NUMBER_OF_BOARDS)
	{
		return STATUS_INVALID_BOARDNUM;
	}
	PCIConfig.Offset 	= Offset;
	PCIConfig.Length	= Length;
	PCIConfig.Buffer	= Buffer;
	// Send WRITE_PCI_CONFIG
	status = ioctl (gBoardInfo[board].hDevice, READ_PCI_CONFIG_IOCTL, &PCIConfig);
	if (status != 0)
	{
		// ioctl failed
		Status->CompletedByteCount = 0;
#if DBG
		printf("ReadPCIConfig IOCTL call failed.  Status=0x%x, Error = 0x%x\n", status, errno);
#endif // DBG
		return errno;
	}
	// save the returned size
	Status->CompletedByteCount = bytesReturned;
	return STATUS_SUCCESSFUL;
}

// GetDMAEngineCap
//
// Returns capabilities of DMA Engine.
UINT GetDMAEngineCap
	(
	UINT 				board,      	// Board to target
	ULONG 				EngineNum,
    PDMA_CAP_STRUCT 	DMACap
	)
{
	UINT			status;

	// Make sure the "Board" is valid
	if (board >= MAXIMUM_NUMBER_OF_BOARDS)
	{
		return STATUS_INVALID_BOARDNUM;
	}

	// Send GET_DMA_ENGINE_CAP_IOCTL
	DMACap->EngineNum = EngineNum;
	status = ioctl (gBoardInfo[board].hDevice, GET_DMA_ENGINE_CAP_IOCTL_BASE, DMACap);
	if (status != 0)
	{
		// ioctl failed
#if DBG
		printf("GetDMACap IOCTL call failed.  Status=0x%x, Error = 0x%x\n", status, errno);
#endif // DBG
		return errno;
	}
	return STATUS_SUCCESSFUL;
}

// DoMem
//
// Sends a DoMem IOCTL call to the driver.
// Returns Completion status in the STAT_STRUCT sent to the call.

UINT DoMem
	(
	UINT 				board,  		// Board to target
	UINT 				Rd_Wr_n,         
	UINT 				BarNum,          
	BYTE *				Buffer,         
	ULONGLONG 			Offset,     
	ULONGLONG 			CardOffset, 
	ULONGLONG 			Length,     
	PSTAT_STRUCT 		Status   
	)
{
	DO_MEM_STRUCT 	doMemStruct;
	DWORD 			bytesReturned = 0;
	DWORD 			ioctlCode;
	UINT			status;

	// Make sure the 'Board' is valid
	if (board >= MAXIMUM_NUMBER_OF_BOARDS)
	{
		return STATUS_INVALID_BOARDNUM;
	}

	// fill in the doMem Structure
	doMemStruct.BarNum = 		BarNum;
	doMemStruct.Offset = 		Offset;
	doMemStruct.CardOffset = 	CardOffset;
	doMemStruct.Length = 		Length;
	doMemStruct.Buffer = 		Buffer;
	// Default to write mode
	ioctlCode = DO_MEM_WRITE_ACCESS_IOCTL_BASE;
	// determine the ioctl code
	if (Rd_Wr_n == READ_FROM_CARD)
	{
		ioctlCode = DO_MEM_READ_ACCESS_IOCTL_BASE;
	}
	
	// Send DoMem IOCTL
	status = ioctl (gBoardInfo[board].hDevice, ioctlCode, &doMemStruct);
	if (status != 0)
	{
		// ioctl failed
		Status->CompletedByteCount = 0;
#if DBG
		printf("DoMem IOCTL call failed.  Status=0x%x, Error = 0x%x\n", status, errno);
#endif // DBG
		return errno;
	}
	// save the returned size
	Status->CompletedByteCount = bytesReturned;
	return STATUS_SUCCESSFUL;
}

// GetDmaPerf
//
// Gets DMA Performance numbers from the board.
UINT GetDmaPerf
	(
	UINT 				board,       	// Board to target
	int 				EngineNumOffset,
	UINT 				TypeDirection,	// DMA Type & Direction Flags
	PDMA_STAT_STRUCT 	Status
	)
{
	DWORD			bytesReturned = 0;
	int				EngineNum = 0;
	UINT			status;

	// Make sure the "Board" is valid
	if (board >= MAXIMUM_NUMBER_OF_BOARDS)
	{
		return STATUS_INVALID_BOARDNUM;
	}

	// determine the ioctl code
	if ((TypeDirection & DMA_CAP_DIRECTION_MASK) == DMA_CAP_CARD_TO_SYSTEM)
	{
		if ((TypeDirection & DMA_CAP_ENGINE_TYPE_MASK) == DMA_CAP_PACKET_DMA)
		{
			if (EngineNumOffset < gBoardInfo[board].DmaInfo.PacketRecvEngineCount)
			{
				EngineNum = gBoardInfo[board].DmaInfo.PacketRecvEngine[EngineNumOffset];
			}
		}
	}
	else // Direction is S2C
	{
		if ((TypeDirection & DMA_CAP_ENGINE_TYPE_MASK) == DMA_CAP_PACKET_DMA)
		{
			if (EngineNumOffset < gBoardInfo[board].DmaInfo.PacketSendEngineCount)
			{
				EngineNum = gBoardInfo[board].DmaInfo.PacketSendEngine[EngineNumOffset];
			}
		}
	}
	
	Status->EngineNum = EngineNum;
	
	// Send GET_PERF_IOCTL IOCTL
	status =  ioctl (gBoardInfo[board].hDevice, GET_PERF_IOCTL_BASE, Status);
	if (status == 0)
	{
		if ((TypeDirection & DMA_CAP_ENGINE_TYPE_MASK) == DMA_CAP_PACKET_DMA)
		{
			// For Packet DMA Engines strip off the low nibble.
			Status->CompletedByteCount &= -16;
			Status->DriverTime &= -16;
			Status->HardwareTime &= -16;
		}
	}
	else
	{
		// ioctl failed
#if DBG
		printf("GetDmaPerf IOCTL call failed.  Status=0x%x, Error = 0x%x\n", status, errno);
#endif // DBG
		return errno;
	}
	return STATUS_SUCCESSFUL;
}


//**************************************************
// FIFO Packet Mode Function calls
//**************************************************

// PacketReceive
//
// Send a PACKET_RECEIVE_IOCTL call to the driver and waits for a completion
//
// Returns Completion status.
UINT PacketReceive
	(
	UINT 				board,   		// Board to target
	int					EngineNum,      // DMA Engine number offset to use
	ULONG *				BufferToken,
	void *				Buffer,           
	ULONG *				Length
	)
{
	return (PacketReceiveEx(board, EngineNum, NULL, BufferToken, Buffer, Length));
}

// PacketReceiveEx - Extended functionality - UserStatus
//
// Send a PACKET_RECEIVE_IOCTL call to the driver and waits for a completion
//
// Returns Completion status.
UINT PacketReceiveEx
	(
	UINT 				board,   		// Board to target
	int					EngineNum,      // DMA Engine number offset to use
	ULONGLONG * 		UserStatus,   	// User Status returned from the EOP DMA Descriptor
	ULONG *				BufferToken,
	void *				Buffer,           
	ULONG *				Length
	)
{
	PACKET_RECEIVE_STRUCT	PacketRecv;
	DWORD			bytesReturned = 0;
	ULONG *			pulBuffer = (ULONG *)Buffer;
	UINT			status;

	// Make sure the 'Board" is valid
	if (board >= MAXIMUM_NUMBER_OF_BOARDS)
	{
		return STATUS_INVALID_BOARDNUM;
	}

	if (EngineNum < gBoardInfo[board].DmaInfo.PacketRecvEngineCount)
	{
		PacketRecv.EngineNum = gBoardInfo[board].DmaInfo.PacketRecvEngine[EngineNum];
		// Indicate Rx Token (if any)
		PacketRecv.RxToken = *BufferToken;
		PacketRecv.Address = 0;
		PacketRecv.Length = 0;	/* Indicate a normal recieve */
	
		*BufferToken = (ULONG)-1;
		*Length = 0;
	
		// Send Packet Mode Release space
		status = ioctl(gBoardInfo[board].hDevice, PACKET_RECEIVE_IOCTL_BASE,  &PacketRecv);
		if (status != 0)
		{
			// ioctl failed
#if DBG
			printf("Packet Rx failed.  Status=0x%x, Error = 0x%x\n", status, errno);
#endif // DBG
			return errno;
		}
		if (PacketRecv.RxToken != -1)
		{
			if (UserStatus != NULL)
			{
				*UserStatus = PacketRecv.UserStatus;
			}
			*BufferToken = PacketRecv.RxToken;
			*pulBuffer = (ULONG)PacketRecv.Address;
			*Length = PacketRecv.Length;
		}
		else
		{
#if DBG
			printf("Received Packet FAILED, Bad Token!\n");
#endif // DBG
			return STATUS_INCOMPLETE;
		}
	}
	else
	{
		return STATUS_INVALID_MODE;
	}
	return STATUS_SUCCESSFUL;
}

// PacketReturnReceive
//
// Send a PACKET_RECEIVE_IOCTL call to the driver to return a buffer token
//
// Returns Completion status.

UINT PacketReturnReceive
	(
	UINT 				board, 			// Board to target
	int					EngineNum,      // DMA Engine number offset to use
	ULONG * 			BufferToken
	)
{
	PACKET_RECEIVE_STRUCT	PacketRecv;
	DWORD			bytesReturned = 0;
	UINT			status;

	// Make sure the 'Board" is valid
	if (board >= MAXIMUM_NUMBER_OF_BOARDS)
	{
		return STATUS_INVALID_BOARDNUM;
	}

	if (EngineNum < gBoardInfo[board].DmaInfo.PacketRecvEngineCount)
	{
		PacketRecv.EngineNum = gBoardInfo[board].DmaInfo.PacketRecvEngine[EngineNum];
		// Indicate Rx Token
		PacketRecv.RxToken = *BufferToken;
		PacketRecv.Address = 0;
		PacketRecv.Length = -1;	/* Indicate a Return Receive */
	
		// For the Packet Return we use the standard Packet Receive call but
		// we do not specify a Out or Return buffer
		status = ioctl(gBoardInfo[board].hDevice, PACKET_RECEIVE_IOCTL_BASE, &PacketRecv);
		if (status != 0)
		{
			// ioctl failed
#if DBG
			printf("Packet Return Rx failed for Token %d. Status=0x%x, Error = 0x%x\n", PacketRecv.RxToken, status, errno);
#endif // DBG
			return errno;
		}
	}
	else
	{
		return STATUS_INVALID_MODE;
	}
	return STATUS_SUCCESSFUL;
}


// PacketReceives -
//
// Send a PACKET_RECEIVES_IOCTL call to the driver and waits for a completion
//
// Returns Completion status.

UINT PacketReceives
	(
	UINT 	board, 			// Board to target
	int		EngineNum,      // DMA Engine number offset to use
	PPACKET_RECVS_STRUCT	pPacketRecvs
	)
{
	DWORD					PacketSize = 0;
	DWORD					bytesReturned = 0;
	DWORD					LastErrorStatus = 0;
	UINT					status;

	// Make sure the 'Board" is valid
	if (board >= MAXIMUM_NUMBER_OF_BOARDS)
	{
		return STATUS_INVALID_BOARDNUM;
	}

	if (EngineNum < gBoardInfo[board].DmaInfo.PacketRecvEngineCount)
	{
		pPacketRecvs->EngineNum = gBoardInfo[board].DmaInfo.PacketRecvEngine[EngineNum];

		PacketSize = sizeof(PACKET_RECVS_STRUCT) + 
			(pPacketRecvs->AvailNumEntries * sizeof(PACKET_ENTRY_STRUCT));

		// Send Packet Recieve multiples request
		status = ioctl(gBoardInfo[board].hDevice, PACKET_RECEIVES_IOCTL_BASE, pPacketRecvs);
		// Make sure we returned something useful
		if (status != 0)
		{
			// ioctl failed
#if DBG
			printf("Packet Recvs failed, Status=0x%x, Error = 0x%x\n", status, errno);
#endif // DBG
			return errno;
		}
	}
	else
	{
#if DBG
		printf("Packet Recvs failed, Incorrect DMA Engine (%d)\n", EngineNum);
#endif // DBG
		return STATUS_INVALID_MODE;
	}
	return STATUS_SUCCESSFUL;
}


// PacketSend
//
// Send a PACKET_SEND_IOCTL call to the driver
//
// Returns Completion status.

UINT PacketSend
	(
	UINT 				board,      		// Board to target
	int					EngineOffset,		// DMA Engine number offset to use
	ULONGLONG 			CardOffset,   
	UCHAR *				Buffer,           
	ULONG 				Length
	)
{
	return (PacketSendEx(board, EngineOffset, 0, CardOffset, Buffer, Length));
}

// PacketSendEx	- Extended Functionality - User Control
//
// Send a PACKET_SEND_IOCTL call to the driver
//
// Returns Completion status.

UINT PacketSendEx
	(
	UINT 				board,      		// Board to target
	int					EngineOffset,		// DMA Engine number offset to use
	ULONGLONG 			UserControl,   	// User Control to set in the first DMA Descriptor
	ULONGLONG 			CardOffset,   
	UCHAR *				Buffer,           
	ULONG 				Length
	)
{
	PACKET_SEND_STRUCT	PacketSend;
	DWORD			bytesReturned = 0;
	UINT			status;

	// Make sure the 'Board" is valid
	if (board >= MAXIMUM_NUMBER_OF_BOARDS)
	{
		return STATUS_INVALID_BOARDNUM;
	}

	if (EngineOffset < gBoardInfo[board].DmaInfo.PacketSendEngineCount)
	{
		// Select a Packet Send DMA Engine 
		PacketSend.EngineNum = gBoardInfo[board].DmaInfo.PacketSendEngine[EngineOffset];
		PacketSend.Length = Length;
		PacketSend.UserControl = UserControl;
		PacketSend.BufferAddress = Buffer;
		PacketSend.CardOffset = CardOffset;
	
		status = ioctl(gBoardInfo[board].hDevice, PACKET_SEND_IOCTL_BASE, &PacketSend);
		if (status != 0)
		{
			// ioctl failed
#if DBG
			printf("Packet Send failed. Status=0x%x, Error = 0x%x\n", status, errno);
#endif // DBG
			return errno;
		}
	}
	else
	{
		printf("DLL: Packet Send failed. No Packet Send Engine\n");
		return STATUS_INVALID_MODE;
	}
	return STATUS_SUCCESSFUL;
}


//**************************************************
// Addressable Packet Mode Function calls
//**************************************************

// PacketRead
//
// Sends a PACKET_READ_IOCTL via the PacketReadEx function
//
// Returns Completion status.

UINT PacketRead
	(
	UINT 				board,     		// Board to target
	int					EngineNum,     	// DMA Engine number offset to use
	ULONGLONG * 		UserStatus,   	// User Status returned from the EOP DMA Descriptor
	ULONGLONG 			CardOffset,   	// Card Address to start read from
	UCHAR *				Buffer,         // Address of data buffer
	ULONG *				Length			// Length to Read
	)
{
	return (PacketReadEx(board,	EngineNum, UserStatus, CardOffset, 0, Buffer, Length));
}

// PacketReadEx
//
// Send a PACKET_READ_IOCTL call to the driver and waits for a completion
//
// Returns Completion status.

UINT PacketReadEx
	(
	UINT 				board,     		// Board to target
	int					EngineNum,     	// DMA Engine number offset to use
	ULONGLONG * 		UserStatus,   	// User Status returned from the EOP DMA Descriptor
	ULONGLONG 			CardOffset,   	// Card Address to start read from
	ULONG				Mode,			// Control Mode Flags
	UCHAR *				Buffer,         // Address of data buffer
	ULONG *				Length			// Length to Read
	)
{
	PACKET_READ_WRITE_STRUCT	PacketRead;
	DWORD			bytesReturned = 0;
	UINT			status;

	// Make sure the "Board" is valid
	if (board >= MAXIMUM_NUMBER_OF_BOARDS)
	{
		return STATUS_INVALID_BOARDNUM;
	}

	if (EngineNum < gBoardInfo[board].DmaInfo.PacketRecvEngineCount)
	{
		PacketRead.EngineNum = gBoardInfo[board].DmaInfo.PacketRecvEngine[EngineNum];
		PacketRead.CardOffset = CardOffset;
		PacketRead.UserInfo = 0;
		PacketRead.BufferAddress = Buffer;
		PacketRead.Length = *Length;
	
		*Length = 0;
	
		// Send Packet Read
		status = ioctl(gBoardInfo[board].hDevice, PACKET_READ_IOCTL_BASE,  &PacketRead);
		if (status != 0)
		{
			// ioctl failed
#if DBG
			printf("Packet Read failed. Status = 0x%x, Error = 0x%x\n", status, errno);
#endif // DBG
			return errno;
		}
		*UserStatus = PacketRead.UserInfo;
		*Length = PacketRead.Length;
	}
	else
	{
		return STATUS_INVALID_MODE;
	}
	return STATUS_SUCCESSFUL;
}


// PacketWrite
//
// Sends a PACKET_WRITE_IOCTL via the PacketWriteEX function call to the driver
//
// Returns Completion status.

UINT PacketWrite
	(
	UINT 				board,			// Board to target
	int					EngineOffset,	// DMA Engine number offset to use
	ULONGLONG 			UserControl,   	// User Control to set in the first DMA Descriptor
	ULONGLONG 			CardOffset,   	// Card Address to start write to
	UCHAR *				Buffer,         // Address of data buffer
	ULONG 				Length			// Length of data packet
	)
{
	return (PacketWriteEx(board, EngineOffset, UserControl, CardOffset, 0, Buffer, Length));
}

// PacketWriteEx
//
// Send a PACKET_WRITE_IOCTL call to the driver
//
// Returns Completion status.

UINT PacketWriteEx
	(
	UINT 				board,			// Board to target
	int					EngineOffset,	// DMA Engine number offset to use
	ULONGLONG 			UserControl,   	// User Control to set in the first DMA Descriptor
	ULONGLONG 			CardOffset,   	// Card Address to start write to
	ULONG				Mode,			// Control Mode Flags
	UCHAR *				Buffer,         // Address of data buffer
	ULONG 				Length			// Length of data packet
	)
{
	PACKET_READ_WRITE_STRUCT	PacketWrite;
	DWORD			bytesReturned = 0;
	UINT			status;

	// Make sure the "Board" is valid
	if (board >= MAXIMUM_NUMBER_OF_BOARDS)
	{
		return STATUS_INVALID_BOARDNUM;
	}

	if (EngineOffset < gBoardInfo[board].DmaInfo.PacketSendEngineCount)
	{
		// Select a Packet Write DMA Engine 
		PacketWrite.EngineNum = gBoardInfo[board].DmaInfo.PacketSendEngine[EngineOffset];
		PacketWrite.Length = 	Length;
		PacketWrite.BufferAddress = Buffer;
		PacketWrite.CardOffset = CardOffset;
		PacketWrite.UserInfo = 	UserControl;
	
		status = ioctl(gBoardInfo[board].hDevice, PACKET_WRITE_IOCTL_BASE, &PacketWrite);
		if (status != 0)
		{
			// ioctl failed
#if DBG
			printf("Packet Write failed.  Status=0x%x, Error = 0x%x\n", status, errno);
#endif // DBG
			return errno;
		}
	}
	else
	{
#if DBG
		printf("DLL: Packet Write failed. No Packet Write Engine\n");
#endif // DBG
		return STATUS_INVALID_MODE;
	}
	return STATUS_SUCCESSFUL;
}

//**************************************************
// Common Packet Mode Function calls
//**************************************************

// SetupPacket
//
// Sends two PACKET_BUF_ALLOC_IOCTL calls to the driver to setup the recieve buffer
//  and intialize the descriptors for sending packets
//
// Returns Completion status.

UINT SetupPacketMode
	(
	UINT 				board,   			// Board to target
	int					EngineOffset,		// DMA Engine number offset to use
    BYTE *				Buffer,				// Address of Pool Buffer (FIFO Mode)
	ULONG*				BufferSize,			// Buffer Pool Size (FIFO Mode)
	ULONG *				MaxPacketSize,		// Largest Packet Size expected (FIFO Mode)
	int					PacketMode,			// Sets mode, FIFO or Addressable
	int					NumberDescriptors	// Number of DMA Descriptors to allocate (Addressable mode)	
	)
{
	BUF_ALLOC_STRUCT	BufAlloc;
	DWORD			bytesReturned = 0;
	UINT			status;

	// Make sure the "Board" is valid
	if (board >= MAXIMUM_NUMBER_OF_BOARDS)
	{
		return STATUS_INVALID_BOARDNUM;
	}

	if (EngineOffset < (int)gBoardInfo[board].DmaInfo.PacketRecvEngineCount)
	{
		// Set the DMA Engine we want to allocate for
		BufAlloc.EngineNum = gBoardInfo[board].DmaInfo.PacketRecvEngine[EngineOffset];
	
		if ((PacketMode == PACKET_MODE_FIFO) ||
			(PacketMode == PACKET_MODE_STREAMING))
		{
			// The user is requestion FIFO Packet mode
			// In this case the application suppies a buffer for the reception of packets
			// for the desginated DMA Engine.
			BufAlloc.AllocationMode = PacketMode;
			// Allocate the size of...
			BufAlloc.Length = *BufferSize;
			// Allocate the number of decriptors based on the Maximum Packet Size we can handle
			BufAlloc.MaxPacketSize = *MaxPacketSize;
			BufAlloc.BufferAddress = Buffer;
			BufAlloc.NumberDescriptors = 0;
	
			// Send Packet Mode Allocate space
			status = ioctl(gBoardInfo[board].hDevice, PACKET_BUF_ALLOC_IOCTL_BASE, &BufAlloc);
			if (status != 0)
			{
				// ioctl failed
#if DBG
				printf("FIFO Packet Mode setup failed. Status=0x%x, Error = 0x%x\n", status, errno);
#endif // DBG
				return errno;
			}
		}
		else if (PacketMode == PACKET_MODE_ADDRESSABLE)
		{
			// This is an application allocated the buffer
			BufAlloc.AllocationMode = PacketMode;
			BufAlloc.Length = 0;
			BufAlloc.MaxPacketSize = 0;
			BufAlloc.BufferAddress = NULL;
			BufAlloc.NumberDescriptors = NumberDescriptors;
	
			// Send Packet Mode Allocate space
			status = ioctl(gBoardInfo[board].hDevice, PACKET_BUF_ALLOC_IOCTL_BASE,	&BufAlloc);
			if (status != 0)
			{
				// ioctl failed
#if DBG
				printf("Addressable Packet Mode setup failed. Status=0x%x, Error = 0x%x\n", status, errno);
#endif // DBG
				return errno;
			}
		}
		else
		{
			return STATUS_INVALID_MODE;
		}
		gBoardInfo[board].AllocationMode[EngineOffset] = PacketMode;
	}
	else
	{
#if DBG
		printf("No Packet Mode DMA Engines found\n");
#endif // DBG
		return STATUS_INVALID_MODE;
	}
	return STATUS_SUCCESSFUL;
}

// ShutdownPacketMode
//
// Sends PACKET_BUF_DEALLOC_IOCTL calls to the driver to teardown the recieve buffer
//  and teardown the descriptors for sending packets
//
// Returns Completion status.

UINT ShutdownPacketMode
	(
	UINT 				board,    		// Board to target
	int					EngineOffset	// DMA Engine number offset to use
	)
{
	BUF_DEALLOC_STRUCT	BufDeAlloc;
	DWORD			bytesReturned = 0;
	UINT			status;

	// Make sure the "Board" is valid
	if (board >= MAXIMUM_NUMBER_OF_BOARDS)
	{
		return STATUS_INVALID_BOARDNUM;
	}

	if (gBoardInfo[board].AllocationMode[EngineOffset] != DMA_MODE_NOT_SET)
	{
		if (EngineOffset < gBoardInfo[board].DmaInfo.PacketRecvEngineCount)
		{
			// Set the allocation mode to what we used above
			BufDeAlloc.AllocationMode = gBoardInfo[board].AllocationMode[EngineOffset];
			// Set the DMA Engine we want to de-allocate
			BufDeAlloc.EngineNum = gBoardInfo[board].DmaInfo.PacketRecvEngine[EngineOffset];
			// Return the Buffer Address we recieved from the Allocate call
			BufDeAlloc.RxBufferAddress = 0;
		
			// Send Packet Mode Release
			status = ioctl(gBoardInfo[board].hDevice, PACKET_BUF_RELEASE_IOCTL_BASE, &BufDeAlloc);
			if (status != 0)
			{
				// ioctl failed
#if DBG
				printf("Packet Rx buffer DeAllocate failed.  Status=0x%x, Error = 0x%x\n", status, errno);
#endif // DBG
				return errno;
			}
			gBoardInfo[board].AllocationMode[EngineOffset] = DMA_MODE_NOT_SET;
			return STATUS_SUCCESSFUL;
		}
	}
	return STATUS_INVALID_MODE;
}


// SetupPacketGenerator
//
// Send PACKET_GENERATOR_CONTROL_IOCTL calls to the driver to 
//	setup the onboard (chip) Packet Generator
//
// Returns Completion status.

UINT SetupPacketGenerator
	(
	UINT 				board,      	// Board to target
	int					EngineOffset,	// DMA Engine number offset to use
	bool				Direction,
	PPACKET_GENCHK_STRUCT pPacketGenChk		// pointer to Packet Generator / Pattern Checker structure
	)
{
	PACKET_GEN_CTRL_STRUCT	PacketGen;
	DWORD			bytesReturned = 0;
	UINT			status;

	// Make sure the "Board" is valid
	if (board >= MAXIMUM_NUMBER_OF_BOARDS)
	{
		return STATUS_INVALID_BOARDNUM;
	}

	if (Direction == S2C_DIRECTION)
	{
		if (EngineOffset < gBoardInfo[board].DmaInfo.PacketSendEngineCount)
		{
			// Set the DMA Engine to the Receive Packet DMA Engine
			PacketGen.EngineNum = gBoardInfo[board].DmaInfo.PacketSendEngine[EngineOffset];
		}
		else
		{
#if DBG
			printf("Packet Checker Control failed. No Packet Send Engine found\n");
#endif // DBG
			return STATUS_INVALID_MODE;
		}
	}
	else
	{
		if (EngineOffset < gBoardInfo[board].DmaInfo.PacketRecvEngineCount)
		{
			// Set the DMA Engine to the Send Packet DMA Engine
			PacketGen.EngineNum = gBoardInfo[board].DmaInfo.PacketRecvEngine[EngineOffset];
		}
		else
		{
#if DBG
			printf("Packet Generator Control failed. No Packet Receive Engine found\n");
#endif // DBG
			return STATUS_INVALID_MODE;
		}
	}

	// Set the Packet Generator Control DWORD
	// The last two values are the rate clocks
	PacketGen.Control = pPacketGenChk->Control;
	// Set the number of packets to generate
	PacketGen.NumPackets = pPacketGenChk->NumPackets;
	// Set the Data Pattern Seed
	PacketGen.DataSeed = pPacketGenChk->DataSeed;
	// Set the User Control / Status pattern seed
	PacketGen.UserCtrlStatSeed = pPacketGenChk->UserCtrlStatSeed;
	// Allocate the number of decriptors based on the Maximum Packet Size we can handle
	PacketGen.PacketLength[0] = pPacketGenChk->PacketLength[0];
	PacketGen.PacketLength[1] = pPacketGenChk->PacketLength[0];
	PacketGen.PacketLength[2] = pPacketGenChk->PacketLength[0];
	PacketGen.PacketLength[3] = pPacketGenChk->PacketLength[0];

	// Send Packet Generator / Checker command
	status = ioctl(gBoardInfo[board].hDevice, PACKET_GENERATOR_CONTROL_IOCTL_BASE,	&PacketGen);
	if (status != 0)
	{
		// ioctl failed
#if DBG
		printf("Packet Generator Control failed.  Status=0x%x, Error = 0x%x\n", status, errno);
#endif // DBG
		return errno;
	}
	return STATUS_SUCCESSFUL;
}


