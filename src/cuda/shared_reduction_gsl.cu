// Step 1-B'' Shared Memory Reduction grid-stride loop版
// ベース: shared_reduction_scale.cu (2026-04-27)
// 変更点:
//   [1] grid-stride loop により各スレッドが複数要素を処理
//   [2] グリッド数を固定(GRID_SIZE=1024)、N に比例しない
//   [3] スレッドあたりの仕事量を増やし、起動コスト・atomic競合を削減
//
// 設計判断:
//   - GRID_SIZE=1024 は Jetson AGX Orin (16 SM × 各SM最大2048スレッド) を意識
//   - 1024ブロック × 256スレッド = 262144 スレッド
//   - N=16M なら各スレッドが 64要素処理
//   - atomicAdd 競合は 1024 回(従来 65536 回 → 1/64)

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// グリッド数固定(N に依存しない)
#define GRID_SIZE 1024
#define BLOCK_SIZE 256

__global__ void shared_memory_reduction_gsl(float* d_in, float* d_out, int N)
{
    // [1] s_data サイズ BLOCK_SIZE と一致
    // [2] grid-stride loop: 各スレッドが N/total_threads 個の要素を加算
    // [3] tree reduction: stride 半減で各段階で半分のスレッドが加算
    // [4] tid==0 ガード + atomicAdd でブロック間集約
    __shared__ float s_data[BLOCK_SIZE];
    int tid = threadIdx.x;

    // [grid-stride loop] 各スレッドが複数要素を加算
    int total_threads = blockDim.x * gridDim.x;
    int start = blockDim.x * blockIdx.x + tid;

    float sum = 0.0f;
    for (int idx = start; idx < N; idx += total_threads) {
        sum += d_in[idx];
    }
    s_data[tid] = sum;
    __syncthreads();

    // [tree reduction] 既存と同じ
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(d_out, s_data[0]);
    }
}

int main(int argc, char* argv[])
{
    int N = (argc > 1) ? atoi(argv[1]) : (1 << 20);
    size_t size = N * sizeof(float);

    printf("=== Shared Memory Reduction (grid-stride loop) ===\n");
    printf("N = %d (%.2f M elements)\n", N, N / 1.0e6);
    printf("Memory size: %.2f MB\n", size / 1.0e6);
    printf("Grid: %d blocks x %d threads = %d threads (FIXED)\n",
           GRID_SIZE, BLOCK_SIZE, GRID_SIZE * BLOCK_SIZE);
    printf("Elements per thread: %.2f\n",
           (double)N / (GRID_SIZE * BLOCK_SIZE));

    // ホスト側
    float* h_a = (float*)malloc(size);
    float h_c;
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
    }

    // デバイス側
    float *d_a, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_c, sizeof(float));
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    // CUDA Event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up
    for (int i = 0; i < 3; i++) {
        cudaMemset(d_c, 0, sizeof(float));
        shared_memory_reduction_gsl<<<GRID_SIZE, BLOCK_SIZE>>>(d_a, d_c, N);
    }
    cudaDeviceSynchronize();

    // エラーチェック
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel Launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // 本計測
    const int RUNS = 10;
    float total_ms = 0.0f;
    for (int i = 0; i < RUNS; i++) {
        cudaMemset(d_c, 0, sizeof(float));
        cudaEventRecord(start);
        shared_memory_reduction_gsl<<<GRID_SIZE, BLOCK_SIZE>>>(d_a, d_c, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    float avg_ms = total_ms / RUNS;

    // 結果取得
    cudaMemcpy(&h_c, d_c, sizeof(float), cudaMemcpyDeviceToHost);

    float expected = (float)N;
    float rel_err = fabsf(h_c - expected) / expected;
    bool correct = rel_err < 1e-4f;

    double gb_per_s = (double)size / (avg_ms / 1000.0) / 1.0e9;

    printf("\n--- Results ---\n");
    printf("Avg time: %.4f ms (over %d runs)\n", avg_ms, RUNS);
    printf("Bandwidth: %.2f GB/s\n", gb_per_s);
    printf("Got: %.2f, Expected: %.2f, Rel err: %.2e\n", h_c, expected, rel_err);
    printf("Verification: %s\n", correct ? "PASS" : "FAIL");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_c);
    free(h_a);

    return 0;
}