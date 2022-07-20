#include "knn.h"
#include "../cuda_utils.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void knn_batch_cuda_(int n, int m, int k, const float *__restrict__ xyz, const float *__restrict__ query_xyz, const int *__restrict__ batch_idxs, const int *__restrict__ query_batch_offsets, int *__restrict__ idx) {
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= n) return;

    xyz += pt_idx * 3;
    idx += pt_idx * k;

    float ox = xyz[0];
    float oy = xyz[1];
    float oz = xyz[2];

    float best[40];
    int besti[40];
    for(int i = 0; i < k; i++){
        best[i] = 1e20;
        besti[i] = 0;
    }

    int batch_idx = batch_idxs[pt_idx];
    int start = query_batch_offsets[batch_idx];
    int end = query_batch_offsets[batch_idx + 1];

    for (int i = start; i < end; ++i) {
        float x = query_xyz[i * 3 + 0];
        float y = query_xyz[i * 3 + 1];
        float z = query_xyz[i * 3 + 2];
        float d2 = (ox - x) * (ox - x) + (oy - y) * (oy - y) + (oz - z) * (oz - z);
        for(int p = 0; p < k; p++){
            if(d2 < best[p]){
                for(int q = k - 1; q > p; q--){
                    best[q] = best[q - 1];
                    besti[q] = besti[q - 1];
                }
                best[p] = d2;
                besti[p] = i;
                break;
            }
        }
    }

    for(int i = 0; i < k; i++){
        idx[i] = besti[i];
    }
}


void knn_batch_cuda(int n, int m, int k, const float *xyz, const float *query_xyz, const int *batch_idxs, const int *query_batch_offsets, int *idx, cudaStream_t stream) {
    // param xyz: (n, 3), float
    // param query_xyz: (m, 3), float
    // param batch_idxs: (n), int
    // param query_batch_offsets: (B + 1), int, offsets[-1] = m
    // param idx: (n, k), int

    cudaError_t err;

    dim3 blocks(DIVUP(n, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    knn_batch_cuda_<<<blocks, threads, 0, stream>>>(n, m, k, xyz, query_xyz, batch_idxs, query_batch_offsets, idx);
    // cudaDeviceSynchronize();  // for using printf in kernel function

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}