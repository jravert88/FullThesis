// -------------------------------------------------------------------------
// 
// PRODUCT:			DMA Driver
// MODULE NAME:		StdTypes.h
// 
// MODULE DESCRIPTION: 
// 
// Contains standard c extension defines and typedefs.
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

/*! @file StdTypes.h
*/

#ifndef __STANDARD_TYPES__h_
#define __STANDARD_TYPES__h_

#include <stdint.h>

typedef unsigned int    uint32_t;
typedef unsigned short  uint16_t;
typedef unsigned char   uint8_t;

#ifndef TRUE
# define TRUE                       1
#define true						TRUE
# define FALSE                      0
#define false						FALSE
#endif /* TRUE */

#ifndef boolean_t
typedef unsigned char               boolean_t;
#endif // boolean_t

#ifndef __cplusplus
#ifndef bool
typedef boolean_t                   bool;
#endif // bool
#endif // Not C++

#ifndef BOOLEAN
typedef bool		               BOOLEAN;
#endif // boolean_t

#ifndef ULONG32
typedef uint32_t                    ULONG32;
#endif // ULONG32

#ifndef DWORD
typedef uint32_t                    DWORD;
#endif // DWORD

#ifndef ULONG64
typedef uint64_t                    ULONG64;
#endif // ULONG64

#ifndef ULONGLONG
//typedef uint64_t                    ULONGLONG;
typedef long long unsigned int		ULONGLONG;
#endif // ULONG64

#ifndef PVOID
typedef void *                      PVOID;
#endif // PVOID

#ifndef UCHAR
typedef unsigned char               UCHAR;
#endif // UCHAR

#ifndef BYTE
typedef unsigned char               BYTE;
#endif // BYTE

#ifndef USHORT
typedef unsigned int                USHORT;
#endif // USHORT

#ifndef UINT
typedef unsigned int                UINT;
#endif // UINT

#ifndef ULONG
typedef unsigned long               ULONG;
#endif // ULONG

#ifndef LONG
typedef long                        LONG;
#endif // LONG

#ifndef SIZE_T
//#define SIZE_T(arg)                 ((size_t)(arg))
#define SIZE_T                 		size_t
//#typedef size_t 					SIZE_T;
#endif // SIZE_T

#ifndef Sleep
#define Sleep(a)                    usleep(a * 1000)
#endif // Sleep

#ifndef SleepEx
#define SleepEx(a, b)               usleep(a * 1000)
#endif // SleepEx

#ifndef LPTHREAD_START_ROUTINE
typedef void *(*LPTHREAD_START_ROUTINE)(void *);
#endif // LPTHREAD_START_ROUTINE

#define	INVALID_HANDLE_VALUE		-1

#endif /* !defined(__STANDARD_TYPES_h__) */

