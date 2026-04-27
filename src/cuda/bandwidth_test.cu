// メモリ帯域天井測定 (bandwidthTest代替)
// 純粋なdevice-to-deviceメモリコピーで実効帯域の天井を測る
// Reductionと違い計算オーバーヘッドなし → 純粋なメモリ律速

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

// 単純コピーカーネル(各スレッド1要素)
__global__ void copy_kernel_simple(float* d_in, float* d_out, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        d_out[i] = d_in[i];
    }
}

// grid-stride loop版コピーカーネル
__global__ void copy_kernel_gsl(float* d_in, float* d_out, int N)
{
    int tid = threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;
    int start = blockDim.x * blockIdx.x + tid;

    for (int idx = start; idx < N; idx += total_threads) {
        d_out[idx] = d_in[idx];
    }
}

double measure_kernel(void (*kernel)(float*, float*, int),
                      float* d_in, float* d_out, int N,
                      int grid, int block, const char* name)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up
    for (int i = 0; i < 3; i++) {
        kernel<<<grid, block>>>(d_in, d_out, N);
    }
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in %s: %s\n", name, cudaGetErrorString(err));
        return -1.0;
    }

    // 本計測
    const int RUNS = 10;
    float total_ms = 0.0f;
    for (int i = 0; i < RUNS; i++) {
        cudaEventRecord(start);
        kernel<<<grid, block>>>(d_in, d_out, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    float avg_ms = total_ms / RUNS;

    // copyは「読み + 書き」の両方なので2倍
    size_t bytes = (size_t)N * sizeof(float) * 2;
    double gb_per_s = (double)bytes / (avg_ms / 1000.0) / 1.0e9;

    printf("[%s] N=%d, time=%.4f ms, BW=%.2f GB/s (R+W)\n",
           name, N, avg_ms, gb_per_s);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return gb_per_s;
}

int main(int argc, char* argv[])
{
    int N = (argc > 1) ? atoi(argv[1]) : (1 << 24);  // デフォルト16M
    size_t size = N * sizeof(float);

    printf("=== Memory Bandwidth Ceiling Test ===\n");
    printf("N = %d (%.2f M elements)\n", N, N / 1.0e6);
    printf("Memory size: %.2f MB (each buffer)\n", size / 1.0e6);
    printf("Total transfer: %.2f MB (read + write)\n", size * 2 / 1.0e6);
    printf("\n");

    float* h_a = (float*)malloc(size);
    for (int i = 0; i < N; i++) h_a[i] = 1.0f;

    float *d_a, *d_b;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    // 単純版: 各スレッド1要素
    int grid_simple = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    measure_kernel(copy_kernel_simple, d_a, d_b, N,
                   grid_simple, BLOCK_SIZE, "Simple    ");

    // grid-stride loop版: グリッド固定
    measure_kernel(copy_kernel_gsl, d_a, d_b, N,
                   1024, BLOCK_SIZE, "Grid-stride");

    // cudaMemcpy版(参考)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < 3; i++) {
        cudaMemcpy(d_b, d_a, size, cudaMemcpyDeviceToDevice);
    }
    cudaDeviceSynchronize();

    const int RUNS = 10;
    float total_ms = 0.0f;
    for (int i = 0; i < RUNS; i++) {
        cudaEventRecord(start);
        cudaMemcpy(d_b, d_a, size, cudaMemcpyDeviceToDevice);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    float avg_ms = total_ms / RUNS;
    double gb_per_s = (double)size * 2 / (avg_ms / 1000.0) / 1.0e9;
    printf("[cudaMemcpy ] N=%d, time=%.4f ms, BW=%.2f GB/s (R+W)\n",
           N, avg_ms, gb_per_s);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    free(h_a);

    return 0;
}