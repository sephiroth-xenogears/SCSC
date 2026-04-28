// Step 1-C Warp Shuffle Reduction
// ベース: shared_reduction_gsl.cu (2026-04-27 完成版)
// 新規要素:
//   [1] warp 内 reduction を __shfl_down_sync で実装(★YK 担当)
//   [2] warp 間集約は SMEM 経由で 1 warp に集めてから再 shuffle
//   [3] grid-stride loop は維持(昨日の達成を継続活用)
//
// 期待性能: grid-stride loop版とほぼ同等(memory bound のため)
// 真の目的: Warp Shuffle 技法の習得(L4 への布石)

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define GRID_SIZE 1024
#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)  //8

__global__ void warp_shuffle_reduction(float* d_in, float* d_out, int N)
{
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;   //どのWARPか（0〜7）
    int lane_id = tid % WARP_SIZE;   //warp内のレーン

    //[step1]grid-stride loopで各スレッドが部分和を作る
    int total_threads = blockDim.x * gridDim.x;
    int start = blockDim.x * blockIdx.x + tid;
    float val = 0.0f;
    for(int idx = start; idx < N; idx += total_threads)
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

        // 8要素を warp shuffle で集約(offset = 4, 2, 1 の 3 回)
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

int main(int argc, char* argv[])
{
    int N = (argc > 1) ? atoi(argv[1]) : (1 << 24);   //デフォルト16M
    size_t size = N * sizeof(float);

    printf("=== Warp Shuffle Reduction ===\n");
    printf("N = %d (%.2f M elements)\n", N,N / 1.0e6);
    printf("Memory size: %.2f MB\n",size / 1.0e6);
    printf("Grid: %d blocks x %d threads (FIXED)\n",GRID_SIZE,BLOCK_SIZE);
    printf("Warps per block : %d\n", WARPS_PER_BLOCK);
    printf("Elements per thread: %.2f\n", (double)N / (GRID_SIZE * BLOCK_SIZE));

    //ホスト側
    float* h_a = (float*)malloc(size);
    float h_c;
    for(int i = 0; i < N; i++) h_a[i] = 1.0f;

    //デバイス側
    float *d_a, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_c, sizeof(float));
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    //CUDA Event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //Warm-up
    for (int i = 0; i < 3; i++ )
    {
        cudaMemset(d_c, 0, sizeof(float));
        warp_shuffle_reduction<<<GRID_SIZE,BLOCK_SIZE>>>(d_a, d_c, N);
    }
    cudaDeviceSynchronize();

    //エラーチェック
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Kernel Launch error: %s\n",cudaGetErrorString(err));
        return 1;
    }
    err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        printf("Kernel execution error:%s\n",cudaGetErrorString(err));
        return 1;
    }

    //本計測
    const int RUNS = 10;
    float total_ms = 0.0f;
    for(int i = 0; i < RUNS; i++)
    {
        cudaMemset(d_c, 0, sizeof(float));
        cudaEventRecord(start);
        warp_shuffle_reduction<<<GRID_SIZE,BLOCK_SIZE>>>(d_a, d_c, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    float arg_ms = total_ms / RUNS;
    
    cudaMemcpy(&h_c, d_c, sizeof(float), cudaMemcpyDeviceToHost);

    float expected = (float)N;
    float rel_err = fabsf(h_c - expected) / expected;
    bool correct = rel_err < 1e-4f;

    double gb_per_s = (double)size / (arg_ms / 1000.0) / 1.0e9;

    printf("\n---Result---\n");
    printf("Avg time: %.4f ms (over %d runs)\n",arg_ms, RUNS);
    printf("Bandwidth: %.2f GB/s\n",gb_per_s);
    printf("Got: %.2f, Expected: %.2f, Rel err: %.2e\n",h_c, expected, rel_err);
    printf("Verification: %s\n", correct ? "PASS" : "FAIL");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_a);
    cudaFree(d_c);
    free(h_a);

    return 0;

}