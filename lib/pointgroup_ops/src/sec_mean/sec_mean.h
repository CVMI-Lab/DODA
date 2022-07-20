/*
Segment Operations (mean, max, min)
Written by Li Jiang
All Rights Reserved 2020.
*/

#ifndef SEC_MEAN_H
#define SEC_MEAN_H
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

#include "../datatype/datatype.h"

void sec_mean(at::Tensor inp_tensor, at::Tensor offsets_tensor, at::Tensor out_tensor, int nProposal, int C);
void sec_mean_cuda(int nProposal, int C, float *inp, int *offsets, float *out);

void sec_mean_bp(at::Tensor d_inp_tensor, at::Tensor offsets_tensor, at::Tensor d_out_tensor, int nProposal, int C);
void sec_mean_bp_cuda(int nProposal, int C, float *d_out, int *offsets, float *d_inp);

void sec_min(at::Tensor inp_tensor, at::Tensor offsets_tensor, at::Tensor out_tensor, int nProposal, int C);
void sec_min_cuda(int nProposal, int C, float *inp, int *offsets, float *out);

void sec_max(at::Tensor inp_tensor, at::Tensor offsets_tensor, at::Tensor out_tensor, int nProposal, int C);
void sec_max_cuda(int nProposal, int C, float *inp, int *offsets, float *out);


#endif //SEC_MEAN_H
