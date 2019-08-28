// -------------------------------------------------------------------------
// 
// PRODUCT:			DMA Driver
// MODULE NAME:		version.h
// 
// MODULE DESCRIPTION: 
// 
// Contains the master version number and macro defines.
// 
// $Revision:  $
//
// ------------------------- CONFIDENTIAL ----------------------------------
// 
//              Copyright (c) 2013 by Northwest Logic, Inc.    
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

#ifndef _VERSION_H_
#define _VERSION_H_

#ifdef WIN32   // Windows Version------------------------------

#define VER_MAJOR_NUM               4
#define VER_MINOR_NUM               7
#define VER_SUBMINOR_NUM			0
#define VER_BUILD_NUM               0000

#else  // Linux version

#define VER_MAJOR_NUM               4
#define VER_MINOR_NUM               7
#define VER_SUBMINOR_NUM			0
#define VER_BUILD_NUM               0000

#endif // Windows vs. Linux differences

#define STRINGIZE2(s) #s
#define STRINGIZE(s) STRINGIZE2(s)

#define VER_MAJOR_NUM_STR       STRINGIZE(VER_MAJOR_NUM)
#define VER_MINOR_NUM_STR       STRINGIZE(VER_MINOR_NUM)
#define VER_SUBMINOR_NUM_STR    STRINGIZE(VER_SUBMINOR_NUM)
#define VER_BUILD_NUM_STR       STRINGIZE(VER_BUILD_NUM)

#define VER_PRODUCTVERSION			VER_MAJOR_NUM, VER_MINOR_NUM, VER_SUBMINOR_NUM, VER_BUILD_NUM
#define VER_PRODUCTVERSION_STR		VER_MAJOR_NUM_STR "." VER_MINOR_NUM_STR "." VER_SUBMINOR_NUM_STR "." VER_BUILD_NUM_STR

#define VER_LEGALCOPYRIGHT_YEARS    "2008-2013"
#define VER_PRODUCTNAME_STR         "Northwest Logic PCI Express Core"
#define VER_INTERNALNAME_STR        "DMADriver.sys"
#define VER_COMPANYNAME_STR         "Northwest Logic, Inc."
#define VER_FILEDESCRIPTION_STR     "PCI Express DMA Reference Driver"
#define VER_LEGALCOPYRIGHT_STR      VER_LEGALCOPYRIGHT_YEARS " " VER_COMPANYNAME_STR

//
#endif

