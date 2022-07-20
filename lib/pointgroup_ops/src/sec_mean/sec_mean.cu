/*
Segment Operations (mean, max, min) (no bp)
Written by Li Jiang
All Rights Reserved 2020.
*/

#include <stdio.h>
#include <math.h>
#include "sec_mean.h"

/* ================================== sec_mean ================================== */
__global__ void sec_mean_cuda_(int nProposal, int C, float *inp, int *offsets, float *out){
    for(int p_id = blockIdx.x; p_id < nProposal; p_id += gridDim.x){
        int start = offsets[p_id];
        int end = offsets[p_id + 1];

        float count = (float)(end - start);

        for(int plane = threadIdx.x; plane < C; plane += blockDim.x){
            float mean = 0;
            for(int i = start; i < end; i++){
                mean += (inp[i * C + plane] / count);
            }
            out[p_id * C + plane] = mean;
        }
    }
}

//input: inp (N, C) float
//input: offsets (nProposal + 1) int
//output: out (nProposal, C) float
void sec_mean_cuda(int nProposal, int C, float *inp, int *offsets, float *out){
    sec_mean_cuda_<<<std::min(nProposal, (int)32768), std::min(C, (int)32)>>>(nProposal, C, inp, offsets, out);
}


__global__ void sec_mean_bp_cuda_(int nProposal, int C, float *d_out, int *offsets, float *d_inp){
    for(int p_id = blockIdx.x; p_id < nProposal; p_id += gridDim.x){
        int start = offsets[p_id];
        int end = offsets[p_id + 1];

        float count = (float)(end - start);

        for(int plane = threadIdx.x; plane < C; plane += blockDim.x){
            for(int i = start; i < end; i++){
                d_inp[i * C + plane] = d_out[p_id * C + plane] / count;
            }
        }
    }
}

// f = 0.25(f1 + f2 + f3 + f4)
// d_f1 = d_f * 0.25
//input: d_out (nProposal, C), float
//output: d_inp (N, C), float
void sec_mean_bp_cuda(int nProposal, int C, float *d_out, int *offsets, float *d_inp){
    sec_mean_bp_cuda_<<<std::min(nProposal, (int)32768), std::min(C, (int)32)>>>(nProposal, C, d_out, offsets, d_inp);
}


/* ================================== sec_min ================================== */
__global__ void sec_min_cuda_(int nProposal, int C, float *inp, int *offsets, float *out){
    for(int p_id = blockIdx.x; p_id < nProposal; p_id += gridDim.x){
        int start = offsets[p_id];
        int end = offsets[p_id + 1];

        for(int plane = threadIdx.x; plane < C; plane += blockDim.x){
            float min_val = 1e50;
            for(int i = start; i < end; i++){
                if(inp[i * C + plane] < min_val){
                    min_val = inp[i * C + plane];
                }
            }
            out[p_id * C + plane] = min_val;
        }
    }
}

//input: inp (N, C) float
//input: offsets (nProposal + 1) int
//output: out (nProposal, C) float
void sec_min_cuda(int nProposal, int C, float *inp, int *offsets, float *out){
    sec_min_cuda_<<<std::min(nProposal, (int)32768), std::min(C, (int)32)>>>(nProposal, C, inp, offsets, out);
}


/* ================================== sec_max ================================== */
__global__ void sec_max_cuda_(int nProposal, int C, float *inp, int *offsets, float *out){
    for(int p_id = blockIdx.x; p_id < nProposal; p_id += gridDim.x){
        int start = offsets[p_id];
        int end = offsets[p_id + 1];

        for(int plane = threadIdx.x; plane < C; plane += blockDim.x){
            float max_val = -1e50;
            for(int i = start; i < end; i++){
                if(inp[i * C + plane] > max_val){
                    max_val = inp[i * C + plane];
                }
            }
            out[p_id * C + plane] = max_val;
        }
    }
}

//input: inp (N, C) float
//input: offsets (nProposal + 1) int
//output: out (nProposal, C) float
void sec_max_cuda(int nProposal, int C, float *inp, int *offsets, float *out){
    sec_max_cuda_<<<std::min(nProposal, (int)32768), std::min(C, (int)32)>>>(nProposal, C, inp, offsets, out);
}