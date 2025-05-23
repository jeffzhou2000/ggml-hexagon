/**=============================================================================

@file
   qprintf_asm.h

@brief
   Extend printf support to assembly.

Copyright (c) 2017,2020 QUALCOMM Technologies Incorporated.
All Rights Reserved Qualcomm Proprietary
=============================================================================**/

/**
 * @defgroup  ASM_function qprintf routine assembly-callable
 * @{
 */

//---------------------------------------------------------------------------
/// @brief
///   Assembly macro for displaying registers along with a message and
///   filename[linenumber].
///
/// @param MSG
///   Message to display.
///
/// @return
///   none.
///
/// Example: See \ref assembly-support for usage examples.
//---------------------------------------------------------------------------
#define qprintf(MSG,...) qprintf_macro #__FILE__, #__LINE__, MSG, #__VA_ARGS__

/**
 * @}
 */

//! @cond Doxygen_Suppress

.set STACK_SIZE, 24
.macro	qprintf_macro FILE_NAME LINE_NUMBER MSG ARGS
.data
1:
.string "\MSG\()\0"
2:
.string "\ARGS\()\0"
3:
.string "\LINE_NUMBER\()\0"
4:
.string "\FILE_NAME\()\0"
.text
  {
   allocframe(#STACK_SIZE)                 // sp[STACK_SIZE]=r31:30  (sp refering to sp after stack allocation)
   memd(r29 + #(-STACK_SIZE-8)) = r29:28   // sp[0]=r29:28
   r28 = ADD(PC,##1b@PCREL)
   } {
   memw(r29 + #8) = r28                    // sp[8]=&msg
   r28 = ADD(PC,##2b@PCREL)
   } {
   memw(r29 + #12) = r28                   // sp[12]=&args
   r28 = #\LINE_NUMBER\()
   } {
   memw(r29 + #16) = r28                   // sp[12]=line_number
   r28 = ADD(PC,##4b@PCREL)
   } {
   memw(r29 + #20) = r28                   // sp[16]=&file_name
   call qprintf_asm
   } {
   r28 = memw(r29 + #0)
   deallocframe
   }
.endm

//! @endcond