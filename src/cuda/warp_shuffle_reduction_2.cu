#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define GRID_SIZE 1024
#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE) //8

__global__ void warp_shuffle_redution_2(float* d_in, float* d_out, int N)
{
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    //[STEP1]grid-stride loopで各スレッドが部分和を作る
    int total = blockDim.x * gridDim.x;
    int start = blockDim.x * blockIdx.x+ tid;
    float val = 0.0f;
    for( int idx = start; idx < N; idx += total)
    {
        val += d_in[idx];
    }

    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);

    // [Step 2] 各warpの代表(lane_id==0)が SMEM に部分和を書き出す
    __shared__ float warp_sums[WARPS_PER_BLOCK];
    if(lane_id == 0)
    {
        warp_sums[warp_id] = val;
    }
    __syncthreads();
    // [Step 3] 最初の warp(warp_id==0)だけが、warp_sums を再 reduction
    if(warp_id == 0)
    {
        // warp_sums は 8 要素しかないので、lane_id < 8 だけ意味のある値
        val = (lane_id < WARPS_PER_BLOCK) ? warp_sums[lane_id] : 0.0f;
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);

        // [Step 4] ブロック代表が atomicAdd
        if(lane_id == 0)
        {
            atomicAdd(d_out, val);

        }
    }
}