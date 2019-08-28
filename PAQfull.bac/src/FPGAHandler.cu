/*
 * FPGAHandler.cu

 *
 *  Created on: Aug 22, 2014
 *      Author: adm85
 */
#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <ctime>
#include "DmaDriverDll.h"
#include "DmaDriverIoctl.h"
#include "stdafx.h"
#include "StdTypes.h"
#include "FPGAHandler.h"
#include "Environment.h"



using namespace std;
using namespace PAQ_SOQPSK;

namespace PAQ_SOQPSK{
	//-----------------------------------------------------------------------------------------------------------------------------------------------
	//														CONSTRUCTOR / DESTRUCTOR
	//-----------------------------------------------------------------------------------------------------------------------------------------------

	/**
	 * Initializes clas properties
	 */
	FPGAHandler::FPGAHandler() {
		//Board connection properties
		boardNum = 0;
		barNum = 0;

		//Samples Reader Properties
		engineOffset = 0;
		sz = sysconf(_SC_PAGESIZE);
		packetMode = PACKET_MODE_ADDRESSABLE;
		numDescriptors = MAX_DMA_DESCRIPTORS;
		mode = NORMAL_MODE;

		userStatus = 0;
		userControl = 0;

		//Connect to the board
		connectToBoard();

		//Set up Addressable Packet Mode on the DMA core
		setupBoardPacketMode();

		if(posix_memalign((void **)&bitDecArray, getPageSize(), BIT_DEC_BUFFER_SIZE_IN_BYTES1)) {
			//Buffer allocation failed.
			cout << "Bit Decision Array buffer allocation FAILED." << endl;
			free(bitDecArray);
			exit(1);
		}
	}

	FPGAHandler::~FPGAHandler() {
		//Turn off Packet Mode (frees resources on driver side)
		shutdownBoardPacketMode();

		//Disconnect from the DMA core
		disconnectFromBoard();

		delete [] bitDecArray;
	}

	//-----------------------------------------------------------------------------------------------------------------------------------------------
	//														GENERAL FPGA METHODS
	//-----------------------------------------------------------------------------------------------------------------------------------------------
	/**
	 * Calls the FPGA driver ConnectToBoard() method.
	 * Optionally prints debug information. This method is called by the constructor,
	 * and should not have to be called by the user.
	 */
	void FPGAHandler::connectToBoard() {
		//Call driver method
		uint boardConnected = ConnectToBoard(boardNum, &dmaInfo);

		if(DEBUG_MODE) {
			if(boardConnected == STATUS_SUCCESSFUL) {
				cout << "Successfully connected to FPGA" << endl;
			} else {
				cout << "***FAILED to connect to FPGA***" << endl;
			}
		}
	}

	/**
	 * Calls the FPGA driver DisconnectFromBoard() method.
	 * Optionally prints debug information.
	 * This board is called by the class destructor, and should not have to be called
	 * by the user.
	 */
	void FPGAHandler::disconnectFromBoard() {
		DisconnectFromBoard(boardNum);

		if(DEBUG_MODE) {
			cout << "Successfully disconnected from FPGA" << endl;
		}
	}

	/**
	 * Sets up the Packet Mode so that we can perform reads from the SDRAM memory on the FPGA.
	 * This method is automatically called by the constructor, and should not have to be called by the user.
	 */
	void FPGAHandler::setupBoardPacketMode() {
		uint iStat = SetupPacketMode(boardNum,
					   	   	   	     engineOffset,
					   	   	   	     (UCHAR *)NULL,
					   	   	   	     (ULONG *)NULL,
					   	   	   	     NULL,
					   	   	   	     packetMode,
					   	   	   	     numDescriptors);

		if(DEBUG_MODE) {
			if(iStat == STATUS_SUCCESSFUL) {
				cout << "Addressable Packet Mode setup successful" << endl;
			} else {
				cout << "***Addressable Packet Mode setup FAILED***" << endl;
				printf("Stat value is 0x%02X\n", iStat);
			}
		}
	}

	/**
	 * Disables Addressable Packet Mode on the DMA core (on the FPGA). This apparently
	 * frees up resources used by the DMA driver. This operation is called automatically
	 * by the class destructor and should not have to be called by the user.
	 */
	void FPGAHandler::shutdownBoardPacketMode() {
		uint iStat = ShutdownPacketMode(boardNum, engineOffset);

		if(DEBUG_MODE) {
			if(iStat == STATUS_SUCCESSFUL) {
				cout << "Addressable Packet Mode shutdown successful" << endl;
			} else {
				cout << "***Addressable Packet Mode shutdown FAILED" << endl;
				printf("Stat value is 0x%02X\n", iStat);
			}
		}
	}

	/**
	 * Handles register operations.
	 * To populate readWriteBar, use FPGAHandler::DO_MEM_READ and FPGAHandler::DO_MEM_WRITE.
	 * It is HIGHLY recommended to use posix_memalign to allocate the space for regBuffer.
	 * bufferSize is the size of regBuffer in bytes. This is the number of bytes that will be written/read
	 *
	 */
	uint FPGAHandler::regRW(uint readWriteBar, unsigned char* regBuffer, ULONGLONG regAddr, ULONGLONG bufferSize) {
		uint iStat = DoMem(boardNum,
					  	   readWriteBar,
					  	   barNum,
					  	   regBuffer,
					  	   0, //Starting write address offset in memBuffer (we always want this to be 0)
					  	   regAddr,
					  	   bufferSize,
					  	   &statusInfo);

		if(DEBUG_MODE) {
			if(iStat == STATUS_SUCCESSFUL) {
				//cout << "Register operation successful" << endl;
			} else {
				cout << "***Register operation FAILED***" << endl;
				printf("Stat value is 0x%02X\n", iStat);
			}
		}

		return iStat;
	}

	/**
	 * Reads from the SDRAM memory on the FPGA card using the PacketReadEx() driver function.
	 * It is HIGHLY recommended to use posix_memalign to allocate the space for regBuffer.
	 */
	uint FPGAHandler::memRead(ULONGLONG memAddr, unsigned char* memBuffer, ULONG* bufferLength) {
		uint iStat = PacketReadEx(boardNum,
								  engineOffset,
								  &userStatus,
								  memAddr,
								  mode,
								  memBuffer,
								  bufferLength);

		if(DEBUG_MODE) {
			if(iStat == STATUS_SUCCESSFUL) {
				//cout << "SDRAM read operation successful" << endl;
			} else {
				cout << "***SDRAM read operation FAILED***" << endl;
				printf("Stat value is 0x%02X\n", iStat);
			}
		}

		return iStat;
	}

	/**
	 * Writes to the SDRAM memory on the FPGA card using the PacketWriteEx() driver function.
	 * It is HIGHLY recommended to use posix_memalign to allocate the space for regBuffer.
	 */
	uint FPGAHandler::memWrite(ULONGLONG memAddr, unsigned char* memBuffer, ULONG bufferLength) {
		uint iStat = PacketWriteEx(boardNum,
								   engineOffset,
								   userControl,
								   memAddr,
								   mode,
								   memBuffer,
								   bufferLength);

		if(DEBUG_MODE) {
			if(iStat == STATUS_SUCCESSFUL) {
				//cout << "SDRAM write operation successful" << endl;
			} else {
				cout << "***SDRAM write operation FAILED***" << endl;
				printf("Stat value is 0x%02X\n", iStat);
			}
		}

		return iStat;
	}

	//-----------------------------------------------------------------------------------------------------------------------------------------------
	//														GETTERS / SETTERS
	//-----------------------------------------------------------------------------------------------------------------------------------------------
	long FPGAHandler::getPageSize() {
		return sz;
	}

}
