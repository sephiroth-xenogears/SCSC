// blur_tiled.cu
// Shared-memory tiled grayscale blur (PMPP Ch.5 pattern) with a naive baseline.
//
// New concept vs. matmul tiling: HALO (ghost cells). Computing a TILE x TILE
// output block needs a (TILE + 2*R) x (TILE + 2*R) input region, because each
// output pixel averages its (2R+1) x (2R+1) neighborhood. The shared tile is
// therefore LARGER than the thread block, which forces two index subtleties:
//   (1) fewer threads than shared cells -> strided cooperative load
//   (2) shared coords are offset from global coords by R (the halo border)
//
// Boundary: average -> "count valid neighbors and divide by the count"
// (NOT zero-pad-and-divide-by-9; that would darken the edges). Out-of-image
// halo cells are loaded as 0 but excluded from the count at compute time.
//
// Build:  nvcc -O3 -arch=sm_87 blur_tiled.cu -o blur_tiled    (Orin = sm_87)
// Run:    ./blur_tiled
//
// Experiment: recompile with different BLUR_RADIUS (1, 4, 8). At R=1 the L2
// cache already absorbs the neighborhood overlap, so tiled may barely beat
// naive. As R grows the overlap exceeds cache and explicit tiling should pull
// ahead. That crossover is the whole point of Ch.5.

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

#define TILE 16               // output tile dim; block = TILE x TILE
#define BLUR_RADIUS 8         //R:1 => 3×3、4 => 9×9、8 => 17×17
#define R    BLUR_RADIUS
#define SH   (TILE + 2 * R)   //shared tile dim (output tile + halo)
#define RUNS  50

// ---------------------------------------------------------------------------
// Naive: each thread reads its (2R+1)^2 neighborhood straight from global.
// Relies on L1/L2 to absorb the overlap between neighboring threads.
// ---------------------------------------------------------------------------
__global__ void blur_naive(const float* __restrict__ in, float* __restrict__ out, int W, int H){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(row < H && col < W){
       float sum = 0.0f;
       int   cnt = 0;
       for(int dy = -R; dy <= R; ++dy )
          for(int dx = -R; dx <= R; ++dx){
              int gy = row + dy, gx = col + dx;
              if(gx >= 0 && gx < W && gy >= 0 && gy < H){
                sum += in[gy * W + gx];
                cnt++;
              } 
          }
        out[row * W + col] = sum / cnt;  
    }
}

// ---------------------------------------------------------------------------
// Tiled: cooperatively stage a (TILE+2R)^2 region (tile + halo) into shared
// memory ONCE, then every output pixel reads its neighborhood from shared.
// ---------------------------------------------------------------------------
__global__ void blur_tiled(const float* __restrict__ in, float* __restrict__ out,int W,int H){
    __shared__ float s[SH][SH];

    int tx = threadIdx.x , ty = threadIdx.y;
    int col = blockIdx.x * TILE + tx;
    int row = blockIdx.y * TILE + ty;
    // Global coord of the shared tile's (0,0): block origin shifted up/left by R.
    int gx0 = blockIdx.x * TILE - R;
    int gy0 = blockIdx.y * TILE - R;

    // Cooperative strided load: TILE*TILE threads fill SH*SH shared cells.
    // Each thread walks linear indices tId, tId + (TILE*TILE), ... until done.
    int tid = ty * TILE + tx;
    for(int i = tid; i < SH * SH; i += TILE * TILE){
        int sy = i / SH, sx = i % SH;       // position inside shared tile
        int gy = gy0 + sy, gx = gx0 + sx;   // corresponding global pixel
        float v = 0.0f;                      // out-of-image halo -> 0 (unused)
        if(gx >= 0 && gx < W && gy >= 0 && gy < H){
           v = in[gy * W + gx]; 
        }
        s[sy][sx] = v;
    }
    __syncthreads();
    if(row < H && col < W){
        // This thread's center sits at shared (ty+R, tx+R). A neighbor at
        // offset (dy,dx) is at shared (ty+R+dy, tx+R+dx). Validity is judged
        // by the neighbor's GLOBAL coord, exactly as in naive -> same count.
        float sum = 0.0f;
        int cnt = 0;
        for(int dy = -R; dy <= R; ++dy)
           for(int dx = -R; dx <= R; ++dx){
              int gy = row + dy, gx = col + dx;
              if(gy >= 0 && gy < H && gx >= 0 && gx < W){
                sum += s[ty + R + dy][tx + R + dx];
                cnt++;
              }
           }
        out[row * W + col] = sum/cnt;      
    }
}
// ---------------------------------------------------------------------------
void blur_cpu(const float* in, float* out, int W, int H) {
    for (int row = 0; row < H; ++row)
        for (int col = 0; col < W; ++col) {
            float sum = 0.0f;
            int   cnt = 0;
            for (int dy = -R; dy <= R; ++dy)
                for (int dx = -R; dx <= R; ++dx) {
                    int gy = row + dy, gx = col + dx;
                    if (gx >= 0 && gx < W && gy >= 0 && gy < H) {
                        sum += in[gy * W + gx];
                        cnt++;
                    }
                }
            out[row * W + col] = sum / cnt;
        }
}

float max_abs_err(const float* a, const float* b, int n) {
    float worst = 0.0f;
    for (int i = 0; i < n; ++i) {
        float e = fabsf(a[i] - b[i]);
        if (e > worst) worst = e;
    }
    return worst;
}

template <typename Launch>
float time_kernel(Launch launch) {
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    launch();                            // warmup
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(start));
    for (int r = 0; r < RUNS; ++r)
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
    const int W = 2048, H = 2048;        // 16MB float >> Orin L2 (~4MB)
    size_t n = (size_t)W * H;
    size_t bytes = n * sizeof(float);

    float *hIn  = (float*)malloc(bytes);
    float *hOut = (float*)malloc(bytes);
    float *hRef = (float*)malloc(bytes);

    srand(0);
    for (size_t i = 0; i < n; ++i) hIn[i] = (float)rand() / RAND_MAX;

    printf("Blur %dx%d, radius R=%d (%dx%d window), shared tile %dx%d\n",
           W, H, R, 2 * R + 1, 2 * R + 1, SH, SH);
    blur_cpu(hIn, hRef, W, H);

    float *dIn, *dOut;
    CHECK(cudaMalloc(&dIn, bytes));
    CHECK(cudaMalloc(&dOut, bytes));
    CHECK(cudaMemcpy(dIn, hIn, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((W + TILE - 1) / TILE, (H + TILE - 1) / TILE);

    // Effective bandwidth = logical traffic (1 read + 1 write per pixel) / time.
    // Same numerator for both kernels, so higher GB/s == faster.
    double gb = 2.0 * (double)n * sizeof(float) / 1e9;

    float msNaive = time_kernel([&]{ blur_naive<<<grid, block>>>(dIn, dOut, W, H); });
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(hOut, dOut, bytes, cudaMemcpyDeviceToHost));
    float errNaive = max_abs_err(hRef, hOut, n);

    float msTiled = time_kernel([&]{ blur_tiled<<<grid, block>>>(dIn, dOut, W, H); });
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(hOut, dOut, bytes, cudaMemcpyDeviceToHost));
    float errTiled = max_abs_err(hRef, hOut, n);

    printf("\n%-8s %10s %12s %14s  %s\n", "kernel", "ms/iter", "GB/s(eff)", "max_abs_err", "status");
    printf("%-8s %10.3f %12.1f %14.2e  %s\n", "naive", msNaive, gb / (msNaive / 1e3),
           errNaive, errNaive < 1e-4f ? "PASS" : "FAIL");
    printf("%-8s %10.3f %12.1f %14.2e  %s\n", "tiled", msTiled, gb / (msTiled / 1e3),
           errTiled, errTiled < 1e-4f ? "PASS" : "FAIL");
    printf("\nspeedup (naive/tiled): %.2fx\n", msNaive / msTiled);

    cudaFree(dIn); cudaFree(dOut);
    free(hIn); free(hOut); free(hRef);
    return 0;
}