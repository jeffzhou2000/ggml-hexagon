/**=============================================================================

@file
   qprintf.h

@brief
   API, macros and struct definitions for qprintf utilities available from C.

Copyright (c) 2017, 2020 QUALCOMM Technologies Incorporated.
All Rights Reserved Qualcomm Proprietary
=============================================================================**/

#ifndef qprintf_H
#define qprintf_H


#ifdef __cplusplus
extern "C"
{
#endif

#include "hexagon_types.h"
#include "stdlib.h"

#ifdef BUILDING_SO
/// MACRO enables function to be visible in shared-library case.
#define qprintf_API __attribute__ ((visibility ("default")))
#else
/// MACRO empty for non-shared-library case.
#define qprintf_API
#endif

/**
 * @defgroup  Masks Common masks controlling which bytes to display.
 * @{
 */

/// Display all bytes
#define QPRINTF_MASK_ALL -1ull

/// Display no bytes
#define QPRINTF_MASK_NONE 0ull

/// Display even bytes
#define QPRINTF_MASK_EVEN_8  0x5555555555555555ull

/// Display odd bytes
#define QPRINTF_MASK_ODD_8   0xaaaaaaaaaaaaaaaaull

/// Display even 16-bit elements
#define QPRINTF_MASK_EVEN_16 0x3333333333333333ull

/// Display odd 16-bit elements
#define QPRINTF_MASK_ODD_16  0xccccccccccccccccull

/// Display even 32-bit elements
#define QPRINTF_MASK_EVEN_32   0x0f0f0f0f0f0f0f0full

/// Display odd 32-bit elements
#define QPRINTF_MASK_ODD_32    0xf0f0f0f0f0f0f0f0ull

/**
 * @}
 */

/**
 * @defgroup  C_functions qprintf functions
 * @{
 */
//---------------------------------------------------------------------------
/// @brief
///   Set the mask controlling which bytes to display when printing out an HVX
///   register.
///
///   If the nth bit of mask is set, the nth byte of HVX will be displayed.
///   When printing HVX as 16-bit or 32-bit elements, only the bit corresponding
///   to the lowest byte of the element controls whether the element will be
///   printed out or not.
///
/// @param high
///   Mask for upper 64 bytes of HVX vector.
///
/// @param low
///   Mask for lower 64 bytes of HVX vector.
///
/// @return
///   None.
///
/// Example:
///
/// * Display the 32-bit odd elements of the 64 most significant bytes and the even
///   bytes of the 64 least significant bytes of HVX vectors printed with option %%m.
/// @code
///     // From C before invoking your assembly routine
///   	qprintf_set_mask(QPRINTF_MASK_ODD_32,QPRINTF_MASK_EVEN_8);
///
///     // From assembly
///     qprintf("v0: %mx",v0);
/// @endcode
///
/// See also \ref assembly-hvx-registers for more assembly examples using %%m.
//---------------------------------------------------------------------------
qprintf_API void qprintf_set_mask(unsigned long long high, unsigned long long low);

//---------------------------------------------------------------------------
/// @brief
///   Print a V register.
///
/// @param msg
///   Character string used to display V register.
///
/// @param V
///   HVX vector register to display.
///
/// @return
///   None.
///
/// Example: See \ref c-hvx-registers for usage examples.
//---------------------------------------------------------------------------
void qprintf_V(const char* msg, HVX_Vector V);

//---------------------------------------------------------------------------
/// @brief
///   Print a Q register. see documentation for details on supported format.
///
/// @param msg
///   Character string used to display Q register.
///
/// @param Q
///   HVX predicate register to display.
///
/// @return
///   None.
///
/// Example: See \ref c-predicate-registers for usage examples.
//---------------------------------------------------------------------------
void qprintf_Q(const char* msg, HVX_VectorPred Q);

//---------------------------------------------------------------------------
/// @brief
///   Display all HVX registers.
///
/// @return
///   None.
///
/// Example: See \ref c-register-dump for usage examples.
//---------------------------------------------------------------------------
extern qprintf_API void qprintf_V_all(void);


//---------------------------------------------------------------------------
/// @brief
///   Display all scalar registers.
///
/// @return
///   None.
///
/// Example: See \ref c-register-dump for usage examples.
//---------------------------------------------------------------------------
extern qprintf_API void qprintf_R_all(void);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif

#endif  // #ifndef qprintf_H
