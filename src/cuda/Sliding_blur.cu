// sliding_blur.cu
// Box blur three ways, to show the arithmetic-reduction arc end to end:
//   1) naive 2D      : (2R+1)^2 reads/output         O(R^2)
//   2) separable     : 2*(2R+1) reads/output         O(R)   [coalesced]
//   3) sliding window: O(1) per output via running sum       [H pass uncoalesced]
//
// Sliding idea: moving one pixel right, the window loses one pixel on the left
// and gains one on the right. Keep a running sum: sum += entering - leaving.
// Cost per output is constant, independent of R.
//
// Two consequences this file is built to expose:
//  (a) Parallelism: the running sum is SEQUENTIAL along a row/column, so the
//      parallel unit must be a whole row (H pass) or column (V pass), not a
//      pixel. That makes the H pass UNCOALESCED (neighbor threads = neighbor
//      rows = addresses W apart). The V pass stays coalesced.
//  (b) Accuracy: the running sum carries float rounding error forward, so the
//      result is NOT bit-identical to the recompute-each-window versions.
//      (It stays well-behaved because a SLIDING sum is bounded, unlike a
//      prefix sum which would drift badly.)
//
// Build: nvcc -O3 -arch=sm_87 sliding_blur.cu -o sliding_blur
// Run:   ./sliding_blur     (recompile with BLUR_RADIUS = 1, 4, 8)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK(call)                                                            \
    do { cudaError_t _e = (call);                                              \
         if (_e != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(_e)); exit(EXIT_FAILURE); } } while (0)

#define TILE        16
#define BLOCK1D     256
#define BLUR_RADIUS 7
#define R           BLUR_RADIUS
#define RUNS        50
// ---- 1) naive 2D -----------------------------------------------------------
__global__ void blur_2d(const float* __restrict__ in,float* __restrict__ out,int W,int H){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(row < H && col < W){
        float sum = 0.0f;
        int cnt = 0;
        for(int dy = -R; dy <= R; ++dy)
           for(int dx = -R; dx <= R;++dx){
              int gy = row + dy, gx = col + dx;
              if(gx >= 0 && gx < W && gy >= 0 && gy < H){
                sum += in[gy * W + gx];
                cnt++;
              }
           }
        out[row * W + col] = sum / cnt;
    } 
}
// ---- 2) separable, recompute each window (O(R), coalesced) ------------------
__global__ void blur_h_sep(const float* __restrict__ in, float* __restrict__ tmp, int W, int H){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(row < H && col < W){
        float sum = 0.0f;
        int cnt = 0;
        for(int dx = -R; dx <= R; ++dx){
            int gx = col + dx;
            if(gx >= 0 && gx < W){
                sum += in[row * W + gx];
                cnt++;
            }
        }
        tmp[row * W + col] = sum/cnt;
    }
}
__global__ void blur_v_sep(const float* __restrict__ tmp, float* __restrict__ out, int W, int H){
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

// ---- 3) sliding window, running sum (O(1)) ---------------------------------
// H pass: one thread per ROW, marches left->right. UNCOALESCED across threads.
__global__ void blur_h_slide(const float* __restrict__ in, float* __restrict__ tmp, int W, int H){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= H)return;
    const float* irow = in + (size_t)row * W;
    float* trow = tmp + (size_t)row * W;
    float sum  = 0.0f;
    int cnt = 0;
    int rend = (R < W-1)? R : W - 1;
    for(int j = 0; j <= rend; ++j){
        sum += irow[j];
        cnt++; 
    }
    for(int c = 0; c < W; ++c){
        trow[c] = sum / cnt;
        int add = c + R + 1;
        if(add < W){
            sum += irow[add];
            cnt++;
        }     
        int rem = c - R;
        if(rem >= 0){
            sum -= irow[rem];
            cnt--;
        }
    }

}

// V pass: one thread per COLUMN, marches top->bottom. COALESCED across threads.
__global__ void blur_v_slide(const float* __restrict__ tmp, float* __restrict__ out, int W, int H){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(col >= W)return;
    float sum = 0.0f;
    int cnt = 0;
    int rend = (R < H - 1) ? R : H - 1;
    for(int i = 0; i <= rend; i++){
        sum += tmp[(size_t)i * W + col];
        cnt++;
    }
    for(int r = 0; r < H; r++){
        out[(size_t)r * W + col] = sum / cnt;
        int add = r + R + 1;
        if(add < H){
            sum += tmp[(size_t)add * W + col];
            cnt++;
        }
        int rem = r - R;
        if(rem >= 0){
            sum -= tmp[(size_t)rem * W +col];
            cnt--;
        }
    }
}
// ---- CPU references --------------------------------------------------------
void ref_2d(const float* in, float* out, int W, int H) {
    for (int row=0; row<H; ++row) for (int col=0; col<W; ++col) {
        float sum=0.0f; int cnt=0;
        for (int dy=-R; dy<=R; ++dy) for (int dx=-R; dx<=R; ++dx) {
            int gy=row+dy, gx=col+dx;
            if (gx>=0&&gx<W&&gy>=0&&gy<H){ sum+=in[gy*W+gx]; cnt++; }
        }
        out[row*W+col]=sum/cnt;
    }
}
void ref_sep(const float* in, float* out, float* tmp, int W, int H) {
    for (int row=0; row<H; ++row) for (int col=0; col<W; ++col) {
        float sum=0.0f; int cnt=0;
        for (int dx=-R; dx<=R; ++dx){ int gx=col+dx; if (gx>=0&&gx<W){ sum+=in[row*W+gx]; cnt++; } }
        tmp[row*W+col]=sum/cnt;
    }
    for (int row=0; row<H; ++row) for (int col=0; col<W; ++col) {
        float sum=0.0f; int cnt=0;
        for (int dy=-R; dy<=R; ++dy){ int gy=row+dy; if (gy>=0&&gy<H){ sum+=tmp[gy*W+col]; cnt++; } }
        out[row*W+col]=sum/cnt;
    }
}

float max_abs_err(const float* a, const float* b, int n) {
    float w=0.0f; for (int i=0;i<n;++i){ float e=fabsf(a[i]-b[i]); if(e>w)w=e; } return w;
}

template <typename Launch> float time_kernel(Launch launch) {
    cudaEvent_t s,e; CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));
    launch(); CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(s));
    for (int r=0;r<RUNS;++r) launch();
    CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
    float ms=0.0f; CHECK(cudaEventElapsedTime(&ms,s,e));
    CHECK(cudaEventDestroy(s)); CHECK(cudaEventDestroy(e));
    return ms/RUNS;
}

int main() {
    const int W=2048, H=2048;
    size_t n=(size_t)W*H, bytes=n*sizeof(float);
    float *hIn=(float*)malloc(bytes), *hOut=(float*)malloc(bytes), *hTmp=(float*)malloc(bytes);
    float *ref2=(float*)malloc(bytes), *refS=(float*)malloc(bytes);

    srand(0);
    for (size_t i=0;i<n;++i) hIn[i]=(float)rand()/RAND_MAX;

    printf("Blur %dx%d, R=%d (%dx%d window)\n", W,H,R,2*R+1,2*R+1);
    printf("reads/output: 2D=%d, separable=%d, sliding=O(1)\n\n", (2*R+1)*(2*R+1), 2*(2*R+1));
    ref_2d(hIn, ref2, W, H);
    ref_sep(hIn, refS, hTmp, W, H);

    float *dIn,*dOut,*dTmp;
    CHECK(cudaMalloc(&dIn,bytes)); CHECK(cudaMalloc(&dOut,bytes)); CHECK(cudaMalloc(&dTmp,bytes));
    CHECK(cudaMemcpy(dIn,hIn,bytes,cudaMemcpyHostToDevice));

    dim3 b2(TILE,TILE), g2((W+TILE-1)/TILE,(H+TILE-1)/TILE);
    int gRows=(H+BLOCK1D-1)/BLOCK1D, gCols=(W+BLOCK1D-1)/BLOCK1D;
    double gb = 2.0*(double)n*sizeof(float)/1e9;

    float ms2d = time_kernel([&]{ blur_2d<<<g2,b2>>>(dIn,dOut,W,H); });
    CHECK(cudaGetLastError()); CHECK(cudaMemcpy(hOut,dOut,bytes,cudaMemcpyDeviceToHost));
    float e2d = max_abs_err(ref2,hOut,n);

    float msSep = time_kernel([&]{
        blur_h_sep<<<g2,b2>>>(dIn,dTmp,W,H);
        blur_v_sep<<<g2,b2>>>(dTmp,dOut,W,H);
    });
    CHECK(cudaGetLastError()); CHECK(cudaMemcpy(hOut,dOut,bytes,cudaMemcpyDeviceToHost));
    float eSep = max_abs_err(refS,hOut,n);

    float msSlide = time_kernel([&]{
        blur_h_slide<<<gRows,BLOCK1D>>>(dIn,dTmp,W,H);
        blur_v_slide<<<gCols,BLOCK1D>>>(dTmp,dOut,W,H);
    });
    CHECK(cudaGetLastError()); CHECK(cudaMemcpy(hOut,dOut,bytes,cudaMemcpyDeviceToHost));
    float eSlide = max_abs_err(refS,hOut,n);   // vs separable ref; expect small but NONZERO

    printf("%-12s %10s %12s %14s  %s\n","kernel","ms/iter","GB/s(eff)","err","status");
    printf("%-12s %10.3f %12.1f %14.2e  %s\n","2D naive",ms2d, gb/(ms2d/1e3), e2d, e2d<1e-4f?"PASS":"FAIL");
    printf("%-12s %10.3f %12.1f %14.2e  %s\n","separable",msSep, gb/(msSep/1e3), eSep, eSep<1e-4f?"PASS":"FAIL");
    printf("%-12s %10.3f %12.1f %14.2e  %s\n","sliding",msSlide, gb/(msSlide/1e3), eSlide, eSlide<1e-2f?"PASS":"FAIL");
    printf("\nspeedup vs 2D:  separable %.2fx,  sliding %.2fx\n", ms2d/msSep, ms2d/msSlide);
    printf("sliding vs separable speedup: %.2fx\n", msSep/msSlide);

    cudaFree(dIn); cudaFree(dOut); cudaFree(dTmp);
    free(hIn); free(hOut); free(hTmp); free(ref2); free(refS);
    return 0;
}