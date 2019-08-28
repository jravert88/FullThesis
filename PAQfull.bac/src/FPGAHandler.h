/*
 * FPGAHandler.h
 *
 *  Created on: Aug 22, 2014
 *      Author: adm85
 */

#ifndef FPGAHandler_H_
#define FPGAHandler_H_

#include "DmaDriverDll.h"
#include "DmaDriverIoctl.h"
#include "stdafx.h"
#include "StdTypes.h"

//#define MAX_DMA_DESCRIPTORS 8192
//#define BUFFER_CHUNK_SIZE 31457280
//#define NORMAL_MODE 0
////#define REGISTER_SIZE_IN_BYTES 16

namespace PAQ_SOQPSK{
class FPGAHandler {
public:
	//Constants
	static const uint MAX_DMA_DESCRIPTORS = 8192;
	static const uint DMA_TRANSFER_SIZE = 31457280;
	static const uint MEMORY_BUFFER_SIZE_IN_BYTES = 157286400;
	static const uint NORMAL_MODE = 0;

	static const ULONGLONG MEMORY_I_SAMPLES_TOP_ADDR		= 0x00000000;
	static const ULONGLONG MEMORY_I_SAMPLES_BOTTOM_ADDR		= 0x09600000;
	static const ULONGLONG MEMORY_Q_SAMPLES_TOP_ADDR		= 0x12C00000;
	static const ULONGLONG MEMORY_Q_SAMPLES_BOTTOM_ADDR		= 0x1C200000;
	static const ULONGLONG MEMORY_BIT_DEC_TOP_ADDR			= 0x25800000;
	static const ULONGLONG MEMORY_BIT_DEC_BOTTOM_ADDR		= 0x26A2E840;

	static const ULONGLONG REG_MEMORY_SAMPLES_ADDR    		= 0x8080;
	static const ULONGLONG REG_HOST_WRITE_REGION_ADDR 		= 0x8090;
	static const ULONGLONG REG_BITDEC_TRANSMIT_ACTIVE_ADDR 	= 0x80A0;
	static const ULONGLONG REG_BITDEC_SLOW_THRESHOLD_ADDR	= 0x80B0;
	static const ULONGLONG REG_BITDEC_UNDERFLOWS_ADDR		= 0x80C0;
	static const uint REGISTER_SIZE_IN_BYTES 				= 16;
	static const uint TOP_SECTION_WRITE_FLAG				= 0;
	static const uint BOTTOM_SECTION_WRITE_FLAG				= 1;

	static const UINT DO_MEM_READ  = 1;
	static const UINT DO_MEM_WRITE = 0;

	static const uint BIT_DEC_BUFFER_SIZE_IN_BYTES = 19064896;
	static const uint BIT_DEC_INITIAL_SLOW_THRESHOLD = 3000000;
	static const float CHUNK_PROCESS_PERIOD = 1.906501818181818;

	static const ULONGLONG FPGA_SCRATCH_REG_ADDR = 0x00008080;
	static const ULONGLONG HOST_WRITE_REGION_REG_ADDR = 0x00008090;
	static const ULONGLONG BIT_DECISION_ACTIVATE_REG = 0x000080A0;
	static const ULONGLONG TRANSMIT_SLOW_THRESHOLD_REG = 0x000080B0;
	static const ULONGLONG BIT_DEC_TOP_AREA_ADDR = 0x25800000;
	static const ULONGLONG BIT_DEC_BOTTOM_AREA_ADDR = 0x26A2E840;

	static const uint BUFFER_CHUNK_SIZE = 150*1048576;
	static const uint BIT_DEC_BUFFER_SIZE_IN_BYTES1 = 19064896;
	static const uint PN11_LENGTH = 2047;
	//static const uint DO_MEM_READ 1
//	static const uint DO_MEM_WRITE 0;
	static const uint BIT_DEC_TOP_AREA = 0;
	static const uint BIT_DEC_BOTTOM_AREA = 1;

	static const float BIT_DECISION_WRITE_PERIOD = 1.906501818181818;

	//Constructor/Destructor
	FPGAHandler();
	virtual ~FPGAHandler();

	//Register read/write
	uint regRW(uint readWriteBar, unsigned char* regBuffer, ULONGLONG regAddr, ULONGLONG bufferSize);

	//Memory Read/Write
	uint memRead(ULONGLONG memAddr, unsigned char* memBuffer, ULONG* bufferLength);
	uint memWrite(ULONGLONG memAddr, unsigned char* memBuffer, ULONG bufferLength);

	//Bit decision writing methods


	//Getter/Setter
	long getPageSize();

	// Moved from Private
	uint boardNum;
	ULONG engineOffset;
	ULONGLONG userControl;
	ULONG mode;
	STAT_STRUCT statusInfo;


	unsigned char* bitDecArray;

private:
	//General methods
	void connectToBoard();
	void disconnectFromBoard();
	void setupBoardPacketMode();
	void shutdownBoardPacketMode();


	//Connection to board
	DMA_INFO_STRUCT dmaInfo;

	uint barNum;

	//Samples Reader Properties

	long sz;
	int packetMode;
	int numDescriptors;

	ULONGLONG userStatus;


};
}
#endif /* FPGAHandler_H_ */
