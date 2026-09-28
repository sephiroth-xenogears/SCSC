// separable_blur.cu
// Separable (two-pass) box blur vs. a naive 2D box blur.
//
// Key idea (a DIFFERENT axis from tiling): a 2D box average factors into a
// horizontal 1D pass followed by a vertical 1D pass. Reads per output pixel
// drop from (2R+1)^2 to 2*(2R+1).  R=8: 289 -> 34.
//
// Why it works: the 2D uniform kernel is a rank-1 outer product
// (w2d[dy][dx] = w_v[dy] * w_h[dx]). Separable <=> 2D kernel is rank-1.
// Box and Gaussian are rank-1; a general (non-outer-product) kernel is not.
//
// Cost trade: the separable path writes an intermediate buffer (pass 1) and
// reads it back (pass 2) -> one extra global round trip. So small R may not
// pay; large R wins big. That crossover is the measurement target.
//
// Edge caveat (IMPORTANT): with "divide by valid count" border handling, the
// separable result does NOT bit-match the monolithic 2D box at the borders
// (within R px of the edge). Interior is identical. The vertical pass weights
// each row equally, while the 2D version weights by actual valid-pixel count.
// This is BY DESIGN, not a bug. We verify separable against a separable CPU
// reference (exact), and separately report how far it sits from the 2D box.
//
// Build: nvcc -O3 -arch=sm_87 separable_blur.cu -o separable_blur
// Run:   ./separable_blur     (recompile with BLUR_RADIUS = 1, 4, 8)

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

#define TILE        16
#define BLUR_RADIUS 8                  // R: 1 => 3x3, 4 => 9x9, 8 => 17x17
#define R           BLUR_RADIUS
#define RUNS        50
// ---------------------------------------------------------------------------
// Naive 2D box (yesterday's kernel). (2R+1)^2 reads per output.
// ---------------------------------------------------------------------------
__global__ void blur_2d(const float* __restrict__ in, float* out, int W, int H ){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < H && col < W){
        float sum = 0.0f;
        int cnt = 0;
        for(int dy = -R; dy <= R; ++dy)
           for(int dx = -R;dx <= R; ++dx){
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
// Separable pass 1: horizontal 1D box -> intermediate buffer. (2R+1) reads.
// Access in[row][col+dx]: for fixed dx, consecutive col => coalesced.
// ---------------------------------------------------------------------------
__global__ void blur_h(const float* __restrict__ in, float* __restrict__ tmp, int W,int H){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(row < H && col < W){
        float sum = 0.0f;
        int cnt = 0;
        for(int dx = -R; dx <= R; ++dx){
            int gx = col + dx;
            if(gx >= 0&& gx < W){
                sum += in[row * W + gx];
                cnt++;
            }
        }
        tmp[row * W + col] = sum / cnt;
    }
}
// ---------------------------------------------------------------------------
// Separable pass 2: vertical 1D box over tmp -> output. (2R+1) reads.
// Access tmp[row+dy][col]: for fixed dy, consecutive col => coalesced.
// ---------------------------------------------------------------------------
__global__ void blur_v(const float* __restrict__ tmp, float* __restrict__ out,int W, int H){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(row < H && col < W){
        float sum = 0.0f;
        int cnt = 0;
        for(int dy = -R; dy <= R; ++dy){
            int gy = row + dy;
            if(gy >= 0 && gy < H){
                sum += tmp[gy * W + col];
                cnt++;
            }
        }
        out[row * W + col] = sum / cnt; 
    }
}
// ---------------------------------------------------------------------------
// CPU references (own logic for each, since they differ at borders).
// ---------------------------------------------------------------------------
void ref_2d(const float* in, float* out, int W, int H) {
    for (int row = 0; row < H; ++row)
        for (int col = 0; col < W; ++col) {
            float sum = 0.0f; int cnt = 0;
            for (int dy = -R; dy <= R; ++dy)
                for (int dx = -R; dx <= R; ++dx) {
                    int gy = row+dy, gx = col+dx;
                    if (gx>=0 && gx<W && gy>=0 && gy<H) { sum += in[gy*W+gx]; cnt++; }
                }
            out[row*W+col] = sum / cnt;
        }
}

void ref_sep(const float* in, float* out, float* tmp, int W, int H) {
    for (int row = 0; row < H; ++row)        // horizontal pass
        for (int col = 0; col < W; ++col) {
            float sum = 0.0f; int cnt = 0;
            for (int dx = -R; dx <= R; ++dx) {
                int gx = col+dx;
                if (gx>=0 && gx<W) { sum += in[row*W+gx]; cnt++; }
            }
            tmp[row*W+col] = sum / cnt;
        }
    for (int row = 0; row < H; ++row)        // vertical pass
        for (int col = 0; col < W; ++col) {
            float sum = 0.0f; int cnt = 0;
            for (int dy = -R; dy <= R; ++dy) {
                int gy = row+dy;
                if (gy>=0 && gy<H) { sum += tmp[gy*W+col]; cnt++; }
            }
            out[row*W+col] = sum / cnt;
        }
}

float max_abs_err(const float* a, const float* b, int n) {
    float w = 0.0f;
    for (int i = 0; i < n; ++i) { float e = fabsf(a[i]-b[i]); if (e>w) w=e; }
    return w;
}

// max abs diff restricted to the interior (>= R px from every edge)
float max_abs_err_interior(const float* a, const float* b, int W, int H) {
    float w = 0.0f;
    for (int row = R; row < H-R; ++row)
        for (int col = R; col < W-R; ++col) {
            float e = fabsf(a[row*W+col] - b[row*W+col]);
            if (e > w) w = e;
        }
    return w;
}

template <typename Launch>
float time_kernel(Launch launch) {
    cudaEvent_t s, e;
    CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));
    launch(); CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(s));
    for (int r = 0; r < RUNS; ++r) launch();
    CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
    float ms = 0.0f; CHECK(cudaEventElapsedTime(&ms, s, e));
    CHECK(cudaEventDestroy(s)); CHECK(cudaEventDestroy(e));
    return ms / RUNS;
}

int main() {
    const int W = 2048, H = 2048;
    size_t n = (size_t)W * H, bytes = n * sizeof(float);

    float *hIn = (float*)malloc(bytes);
    float *hOut = (float*)malloc(bytes);
    float *hTmp = (float*)malloc(bytes);
    float *ref2 = (float*)malloc(bytes);   // 2D reference
    float *refS = (float*)malloc(bytes);   // separable reference

    srand(0);
    for (size_t i = 0; i < n; ++i) hIn[i] = (float)rand() / RAND_MAX;

    printf("Blur %dx%d, R=%d (%dx%d window)\n", W, H, R, 2*R+1, 2*R+1);
    printf("reads/output: 2D = %d, separable = %d\n\n", (2*R+1)*(2*R+1), 2*(2*R+1));

    ref_2d(hIn, ref2, W, H);
    ref_sep(hIn, refS, hTmp, W, H);

    float *dIn, *dOut, *dTmp;
    CHECK(cudaMalloc(&dIn, bytes));
    CHECK(cudaMalloc(&dOut, bytes));
    CHECK(cudaMalloc(&dTmp, bytes));
    CHECK(cudaMemcpy(dIn, hIn, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((W+TILE-1)/TILE, (H+TILE-1)/TILE);
    double gb = 2.0 * (double)n * sizeof(float) / 1e9;   // logical 1R+1W

    // --- naive 2D ---
    float ms2d = time_kernel([&]{ blur_2d<<<grid,block>>>(dIn,dOut,W,H); });
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(hOut, dOut, bytes, cudaMemcpyDeviceToHost));
    float e2d = max_abs_err(ref2, hOut, n);

    // --- separable (pass1 + pass2) ---
    float msSep = time_kernel([&]{
        blur_h<<<grid,block>>>(dIn, dTmp, W, H);
        blur_v<<<grid,block>>>(dTmp, dOut, W, H);
    });
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(hOut, dOut, bytes, cudaMemcpyDeviceToHost));
    float eSep = max_abs_err(refS, hOut, n);                  // vs separable ref
    float dSepVs2dFull = max_abs_err(ref2, hOut, n);          // vs 2D box (full)
    float dSepVs2dIn   = max_abs_err_interior(ref2, hOut, W, H); // vs 2D (interior)

    printf("%-12s %10s %12s %14s  %s\n", "kernel", "ms/iter", "GB/s(eff)", "err_vs_ownref", "status");
    printf("%-12s %10.3f %12.1f %14.2e  %s\n", "2D naive", ms2d, gb/(ms2d/1e3),
           e2d, e2d < 1e-4f ? "PASS" : "FAIL");
    printf("%-12s %10.3f %12.1f %14.2e  %s\n", "separable", msSep, gb/(msSep/1e3),
           eSep, eSep < 1e-4f ? "PASS" : "FAIL");
    printf("\nspeedup (2D / separable): %.2fx\n", ms2d / msSep);

    printf("\nseparable vs 2D box:  full max diff = %.2e,  interior max diff = %.2e\n",
           dSepVs2dFull, dSepVs2dIn);
    printf("(interior ~0 = same math; full nonzero = edge-only difference, expected)\n");

    cudaFree(dIn); cudaFree(dOut); cudaFree(dTmp);
    free(hIn); free(hOut); free(hTmp); free(ref2); free(refS);
    return 0;
}