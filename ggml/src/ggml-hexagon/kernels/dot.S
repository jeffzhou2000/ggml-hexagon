/**=============================================================================
@file
    qhblas_f_vector_dot_af.S

@brief
    Calculates dot product of two input float vectors.

    Function prototype
        
        int32_t qhblas_f_vector_dot_af(float_a8_t *input_1, float_a8_t *input_2, float *output, uint32_t size);

    Reference C code
        
        int32_t qhblas_f_vector_dot_af(float_a8_t *input_1, float_a8_t *input_2, float *output, uint32_t size)
        {
            if ((input_1 == NULL) || (input_2 == NULL) || (output == NULL) || (size == 0))
            {
                return -1;
            }

            float dot = 0;
            for (uint32_t i = 0; i < size; ++i)
            {
                dot += input_1[i] * input_2[i];
            }

            *output = dot;
            return 0;
        }

Copyright (c) 2019 Qualcomm Technologies Incorporated.
All Rights Reserved. Qualcomm Proprietary and Confidential.
=============================================================================**/

/*============================================================================*/

    .p2align 2
    .p2align 4,,15
    .global qhblas_f_vector_dot_af
    .type qhblas_f_vector_dot_af, @function

/*============================================================================*/

#define DC_PREFETCH_AHEAD    64                                    // number of bytes for DCFETCH
#define L2_PREFETCH_AHEAD    256                                   // number of bytes for L2FETCH
#define L2FETCH_CONFIG       0x0100FF00+(L2_PREFETCH_AHEAD/256)    // [stride = 256 : width = 255 : height = bytes/256]
#define L2_PREFETCH_ELEMS    L2_PREFETCH_AHEAD/8                   // number of elements to prefetch with L2FETCH

/*============================================================================*/

qhblas_f_vector_dot_af:
{
    p0 = !cmp.eq(r0,#0)                           // input_1 != NULL
    p0 = !cmp.eq(r1,#0)                           // input_2 != NULL
    p0 = !cmp.eq(r2,#0)                           // output != NULL
    p0 = cmp.gtu(r3,#0)                           // size > 0
    if (!p0.new) jump:nt .L_ret
}
{
    r10 = #0
    r3 = lsr(r3,#1)                               // size / 2
    p1 = tstbit(r3,#0)                            // check for odd size
    if(cmp.eq(r3.new,#0)) jump:nt .L_do_one
}
{
    r7:6 = #0
    r9:8 = #0
    r5 = add(r3,#7)                               // (size / 2) + 7
    p2 = cmp.gtu(r3,#L2_PREFETCH_ELEMS)           // check whether we can do l2fetch
}
{
    r5 = lsr(r5,#3)                               // ceil(size / 2)
    r14 = mux(p2,r3,#0)                           // set l2fetch counter
}
{
    r13:12 = combine(##L2FETCH_CONFIG,#8)         // set l2fetch config and max number of iterations for .L_loop_do_two
    loop1(.L_prefetch_loop_do_two,r5)
}
    .falign
.L_prefetch_loop_do_two:
{
    dcfetch(r0+#DC_PREFETCH_AHEAD)                // prefetch ahead for input_1
    r5 = min(r12,r3)                              // min(8, size / 2)
}
{
    dcfetch(r1+#DC_PREFETCH_AHEAD)                // prefetch ahead for input_2
    loop0(.L_loop_do_two,r5)
    p2 = cmp.eq(r3,r14)                           // check whether to do l2fetch
    if (!p2.new) jump:t .L_loop_do_two
}
{
    r5 = add(r3,#-L2_PREFETCH_ELEMS)              // number of elements left to prefetch ahead
    r15 = add(r0,#L2_PREFETCH_AHEAD)              // input_1 addr for l2fetch
}
{
    p2 = cmp.gtu(r5,#L2_PREFETCH_ELEMS)           // check whether we can continue to do l2fetch
    r15 = add(r1,#L2_PREFETCH_AHEAD)              // input_2 addr for l2fetch
    l2fetch(r15,r13)
}
{
    if (p2) r14 = add(r14,#-L2_PREFETCH_ELEMS)    // adjust l2fetch counter
    if (!p2) r14 = #0                             // there are no more bytes left to prefetch ahead
    l2fetch(r15,r13)
}
    .falign
.L_loop_do_two:
{
    r7:6 = memd(r0++#8)
    r9:8 = memd(r1++#8)
    r10 += sfmpy(r7,r9)
}
{
    r10 += sfmpy(r6,r8)
    r3 = add(r3,#-1)                              // adjust (size / 2)
}:endloop0:endloop1
{
    r10 += sfmpy(r7,r9)
    if (!p1) jump:nt .L_ret
}
    .falign
.L_do_one:
{
    r4 = memw(r0)
    r5 = memw(r1)
}
{
    r10 += sfmpy(r4,r5)
}
    .falign
.L_ret:
{
    if (p0) memw(r2) = r10
    r0 = mux(p0,#0,#-1)
    jumpr r31
}
    .size   qhblas_f_vector_dot_af, .-qhblas_f_vector_dot_af
