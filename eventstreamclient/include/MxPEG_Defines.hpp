//           ///          //                                  C++ Cross Platform
//          /////        ////
//         /// XXX     XXX ///            ///////////   /////////     ///   ///
//        ///    XXX XXX    ///         ///             ///    ///   ///  ///
//       ///       XXX       ///         /////////     ///      //  //////
//      ///      XXX XXX      ///               ///   ///    ///   ///  ///
//     ////    XXX     XXX    ////    ////////////  //////////    ///   ///
//    ////                     ////
//   ////  M  O  B  O  T  I  X  ////////////////////////////////////////////////
//  //// Security Vision Systems //////////////////////////////////////////////
//
//
///////
     //
     //  MOBOTIX EventStream Client SDK 
     //
     //  See LICENSE_SDK included in the SDK package
     //
     //  Copyright (c) 2005 - 2016, MOBOTIX AG
     //  All rights reserved.
     //
///////
    

#ifndef MXPEG_DEFINES_HPP_
#define MXPEG_DEFINES_HPP_

//#include "MxPEG_Logger.hpp"

#if defined (_MSC_VER)
#include <Windows.h>
#include <Time.h>
#else
#include <sys/time.h>
#endif

#include <stdint.h>
#include <cstddef>

#ifndef SAFEFREE
#define SAFEFREE(x) do { if(x != nullptr) { free(x); x=nullptr;} } while(0)
#endif

#define UNFIX_X(_var, _fix) ( (double)(_var)/(uint32_t)(1<<_fix) )

typedef enum t_MxPEG_ReturnCode {
   er_noData 		= -4,
   er_invalidType 	= -3,
   er_notInitialized= -2,
   er_Error 		= -1,
   er_Success 		=  1
} MxPEG_ReturnCode;

typedef enum t_MxPEG_VideoMode {
	vm_MxPEG,
	vm_MJPEG
} MxPEG_VideoMode;

typedef enum t_MxPEG_Sensor {
	s_left,
	s_right,
	s_both,
	s_auto,
	s_thermal
} MxPEG_Sensor;


namespace ie { namespace MxPEG {
typedef struct tMxPEGFrame
{
   uint8_t * data;                      //pointer to SOI of encoded MxPEG frame
   size_t size;                         //size of image data
} MxPEG_Frame;
}}

#endif //MXPEG_DEFINES_HPP_
