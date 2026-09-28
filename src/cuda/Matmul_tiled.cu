// matmul_tiled.cu
// Tiled matrix multiplication (PMPP Ch.5 pattern) with a naive baseline.
//
// Purpose: solidify the tiling pattern on a halo-free problem before adding
//          halo to the blur kernel (論点K).
//
// C = A * B   where  A: M x K,  B: K x N,  C: M x N   (row-major, float)
//
// Build:  nvcc -O3 -arch=sm_87 matmul_tiled.cu -o matmul_tiled    (Orin = sm_87)
//         (drop -arch or set to your GPU; sm_87 = Jetson AGX Orin)
// Run:    ./matmul_tiled

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK(call)                                                            \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(_e));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define TILE_WIDTH 16     // 16x16 = 256 threads/block. Try 32 as an experiment.
#define RUNS       50     // timed iterations (warmup is separate)
// ---------------------------------------------------------------------------
// Naive kernel: each thread reads a full row of A and column of B from global.
// Low arithmetic intensity -> memory bound. This is the "before".
// ---------------------------------------------------------------------------

__global__ void matmul_naive(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int M, int N, int K){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < M && col < N){
       float acc = 0.0f;
       for(int k = 0; k < K; k++)
          acc += A[row * K + k] * B[k * N + col];
       C[row * N + col] = acc;
    }

}
// ---------------------------------------------------------------------------
// Tiled kernel: cooperatively stage a TILE_WIDTH x TILE_WIDTH block of A and B
// into shared memory, reuse it TILE_WIDTH times, then advance to the next tile
// along K. This raises arithmetic intensity -> pushes toward compute bound.
// ---------------------------------------------------------------------------
__global__ void matmul_tiled(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int M, int N, int K){
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + ty;     //output row thread owns
    int col = blockIdx.x * TILE_WIDTH + tx;     //output col thread owns

    float acc = 0.0f;

    int numphases = (K + TILE_WIDTH - 1) / TILE_WIDTH;    //ceil
    for(int p = 0; p < numphases; ++p){
        // --- Cooperative load: A tile (rows fixed by block, cols = p*TILE + tx)
        int acol = p * TILE_WIDTH + tx; 
        As[ty][tx] = (row < M && acol < K) ? A[row * K + acol] : 0.0f;

        // --- Cooperative load: B tile (cols fixed by block, rows = p*TILE + ty)
        int brow = p * TILE_WIDTH + ty;
        Bs[ty][tx] = (brow < K && col < N) ? B[brow * N + col] : 0.0f;
        
        // SYNC #1: all loads visible before anyone reads the tiles.
        // Remove this -> threads read stale/garbage shared mem. Silent wrong answer.
        __syncthreads();
        
        // Multiply the staged tiles. Out-of-range loads were zero-padded, so the
        // last partial phase needs no special handling: zeros don't affect the sum.
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k)
            acc += As[ty][k] * Bs[k][tx];

        // SYNC #2: nobody overwrites the tiles for phase p+1 until all threads
        // have finished computing on phase p. Remove this -> write-before-read
        // race. Also silent wrong answer.
        __syncthreads();            
    
    }
    if(row < M && col < N)
      C[row * N + col] = acc;

}

// ---------------------------------------------------------------------------
// CPU reference (triple loop). O(MNK); fine for correctness on ~1024^3.
// ---------------------------------------------------------------------------
// Reference accumulates in double and stores double: a trustworthy "gold"
// answer, so the check measures the kernel's error, not the reference's own
// rounding.
void matmul_cpu(const float* A, const float* B, double* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k)
                acc += (double)A[i * K + k] * (double)B[k * N + j];
            C[i * N + j] = acc;
        }
}

// Norm-wise relative error: ||ref - got||_2 / ||ref||_2.
// Robust for mean-zero data: a handful of near-zero entries can't dominate the
// way a per-element |ref-got|/|ref| metric lets them. Expect ~1e-5 for correct
// float matmul; ~1e-1 or NaN signals a real bug.
double normwise_rel_err(const double* ref, const float* got, int n) {
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = ref[i] - (double)got[i];
        num += d * d;
        den += ref[i] * ref[i];
    }
    return sqrt(num) / sqrt(den);
}

// time a launch lambda over RUNS, return ms/iter
template <typename Launch>
float time_kernel(Launch launch) {
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    launch();                       // warmup (also triggers JIT / cache fill)
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaEventRecord(start));
    for (int r = 0; r < RUNS; ++r)  // note: r < RUNS, runs exactly RUNS times
        launch();
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    return ms / RUNS;
}

int main() {
    const int M = 1024, N = 1024, K = 1024;

    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    float *hA = (float*)malloc(bytesA);
    float *hB = (float*)malloc(bytesB);
    float  *hC   = (float*)malloc(bytesC);                          // device result
    double *hRef = (double*)malloc((size_t)M * N * sizeof(double));  // gold (double)

    srand(0);
    for (int i = 0; i < M * K; ++i) hA[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < K * N; ++i) hB[i] = (float)rand() / RAND_MAX - 0.5f;

    printf("Computing CPU reference (%dx%dx%d)...\n", M, N, K);
    matmul_cpu(hA, hB, hRef, M, N, K);

    float *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, bytesA));
    CHECK(cudaMalloc(&dB, bytesB));
    CHECK(cudaMalloc(&dC, bytesC));
    CHECK(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));

    // FLOP count: each C element = K multiply-adds = 2K flops.
    double gflop = 2.0 * (double)M * N * K / 1e9;

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH,
              (M + TILE_WIDTH - 1) / TILE_WIDTH);

    // --- Naive ---
    float msNaive = time_kernel([&]{
        matmul_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
    });
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost));
    double errNaive = normwise_rel_err(hRef, hC, M * N);

    // --- Tiled ---
    float msTiled = time_kernel([&]{
        matmul_tiled<<<grid, block>>>(dA, dB, dC, M, N, K);
    });
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost));
    double errTiled = normwise_rel_err(hRef, hC, M * N);

    printf("\n%-8s %10s %12s %14s  %s\n", "kernel", "ms/iter", "GFLOP/s", "rel_err(L2)", "status");
    printf("%-8s %10.3f %12.1f %14.2e  %s\n", "naive", msNaive, gflop / (msNaive / 1e3),
           errNaive, errNaive < 1e-3 ? "PASS" : "FAIL");
    printf("%-8s %10.3f %12.1f %14.2e  %s\n", "tiled", msTiled, gflop / (msTiled / 1e3),
           errTiled, errTiled < 1e-3 ? "PASS" : "FAIL");
    printf("\nspeedup (naive/tiled): %.2fx\n", msNaive / msTiled);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC); free(hRef);
    return 0;
}