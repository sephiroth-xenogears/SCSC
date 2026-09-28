// tiled_separable_blur.cu
// The blur arc finale: TILING x SEPARABLE (two stacked optimizations).
//
//   naive 2D       : (2R+1)^2 reads/output                       O(R^2)
//   separable      : 2*(2R+1) reads/output, coalesced            O(R)
//   tiled separable : separable, but each 1D pass stages a tile  O(R) from shared
//                     (with 1D halo) into shared memory first, so input reuse
//                     is explicit instead of left to the L2 cache.
//
// Why tile SEPARABLE and not the O(1) sliding window?
//   The sliding window's running sum is sequential along a row -> only 1 thread
//   per row -> tiny parallelism. Tiling fixes coalescing but NOT parallelism,
//   so tiled-sliding would still lose. Separable has full parallelism AND
//   coalescing, so it is the right thing to tile. The GPU wants parallelism +
//   coalesced access, not minimal arithmetic.
//
// New idea vs the 2D blur tile: the halo is 1D. The horizontal pass only blurs
// in x, so its shared tile is wider by 2R in x only: [TILE_Y][TILE_X + 2R].
// The vertical pass is taller by 2R in y only: [TILE_Y + 2R][TILE_X].
//
// Build: nvcc -O3 -arch=sm_87 tiled_separable_blur.cu -o tiled_separable_blur
// Run:   ./tiled_separable_blur    (recompile with BLUR_RADIUS = 1, 4, 8)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK(call)                                                            \
    do { cudaError_t _e=(call);                                                \
         if (_e!=cudaSuccess){ fprintf(stderr,"CUDA error %s:%d: %s\n",        \
             __FILE__,__LINE__,cudaGetErrorString(_e)); exit(1);} } while(0)

#define BLUR_RADIUS 10
#define R           BLUR_RADIUS
#define TILE_X      32          // output cols per block
#define TILE_Y      8           // output rows per block  (32*8 = 256 threads)
#define RUNS        50

// ---- naive 2D (baseline) ---------------------------------------------------
__global__ void blur_2d(const float* __restrict__ in, float* __restrict__ out, int W, int H) {
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if (row<H && col<W) {
        float sum=0.0f; int cnt=0;
        for (int dy=-R; dy<=R; ++dy) for (int dx=-R; dx<=R; ++dx) {
            int gy=row+dy, gx=col+dx;
            if (gx>=0&&gx<W&&gy>=0&&gy<H){ sum+=in[gy*W+gx]; cnt++; }
        }
        out[row*W+col]=sum/cnt;
    }
}

// ---- plain separable (O(R), coalesced, relies on L2 for reuse) --------------
__global__ void blur_h(const float* __restrict__ in, float* __restrict__ tmp, int W, int H){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < H && col < W){
        float s = 0;
        int c = 0;
        for(int dx = -R; dx <= R; ++dx){
            int gx = col + dx;
            if(gx >= 0 && gx < W){
                s += in[row * W + gx];
                c++;
            }
        }
        tmp[row * W + col] = s/c;
 
    }
}

__global__ void blur_v(const float* __restrict__ tmp,float* __restrict__ out, int W, int H){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(row < H && col < W){
        float s = 0;
        int c = 0;
        for(int dy = -R; dy <= R; ++dy){
            int gy = row + dy;
            if(gy >= 0 && gy < H){
                s += tmp[gy * W + col];
                c++;
            }
        }
        out[row * W + col] = s/c;
    }
}
// ---- tiled separable: horizontal pass (1D halo in x) -----------------------
__global__ void blur_h_tiled(const float* __restrict__ in, float* __restrict__ tmp, int W, int H){
    __shared__ float s[TILE_Y][TILE_X + 2 * R];
    const int SW = TILE_X + 2 * R;        // shared width (tile + x-halo)

    int tx = threadIdx.x, ty = threadIdx.y;
    int col = blockIdx.x * TILE_X + tx;
    int row = blockIdx.y * TILE_Y + ty;
    int gx0 = blockIdx.x * TILE_X - R;
    // cooperative coalesced load of TILE_Y x SW cells
    int tid = ty * TILE_X + tx;
    for(int i = tid; i < TILE_Y * SW; i += TILE_X * TILE_Y){
        int sy = i / SW, sx = i % SW;
        int gy = blockIdx.y * TILE_Y + sy , gx = gx0 + sx;
        float v = 0.0f;
        if(gy < H && gx >= 0 && gx < W) v = in[gy * W + gx];
        s[sy][sx] = v;
    }
    __syncthreads();
    if(row < H && col < W){
        // this thread's center sits at shared x-index (tx+R); window = tx..tx+2R
        float sum = 0.0f;
        int cnt = 0;
        for(int dx = -R; dx <= R; ++dx){
            int gx = col + dx;

            if(gx >= 0 && gx < W){
                sum += s[ty][tx + R + dx];
                cnt++;
            }
        }
        tmp[row * W + col] = sum / cnt;
    }
}
// ---- tiled separable: vertical pass (1D halo in y) -------------------------
__global__ void blur_v_tiled(const float* __restrict__ tmp, float* __restrict__ out, int W, int H){
    __shared__ float s[TILE_Y + 2 * R][TILE_X];
    const int SH = TILE_Y + 2 * R;
    int tx = threadIdx.x, ty = threadIdx.y;
    int col = blockIdx.x * TILE_X + tx;
    int row = blockIdx.y * TILE_Y + ty;
    int gy0 = blockIdx.y * TILE_Y - R;

    int tid = ty * TILE_X + tx;
    for(int i = tid; i < SH * TILE_X; i += TILE_X * TILE_Y){
        int sy = i / TILE_X, sx = i % TILE_X;
        int gy = gy0 + sy, gx = blockIdx.x * TILE_X + sx;
        float v = 0.0f;
        if(gx < W && gy >= 0 && gy < H)
        v = tmp[gy * W + gx];
        s[sy][sx] = v;
    }
    __syncthreads();
    
    if(row < H && col < W){
        float sum = 0.0f;
        int cnt = 0;
        for(int dy = -R; dy <= R; ++dy){
            int gy = row + dy;
            if(gy >= 0 && gy < H){
                sum += s[ty + R + dy][tx];
                cnt++;
            }
        }
        out[row * W + col] = sum / cnt;
    }
}
// ---- CPU references --------------------------------------------------------
void ref_2d(const float* in, float* out, int W, int H) {
    for (int row=0;row<H;++row) for (int col=0;col<W;++col){
        float s=0; int c=0;
        for(int dy=-R;dy<=R;++dy)for(int dx=-R;dx<=R;++dx){int gy=row+dy,gx=col+dx;
            if(gx>=0&&gx<W&&gy>=0&&gy<H){s+=in[gy*W+gx];c++;}}
        out[row*W+col]=s/c; }
}
void ref_sep(const float* in, float* out, float* tmp, int W, int H) {
    for (int row=0;row<H;++row) for (int col=0;col<W;++col){ float s=0; int c=0;
        for(int dx=-R;dx<=R;++dx){int gx=col+dx; if(gx>=0&&gx<W){s+=in[row*W+gx];c++;}}
        tmp[row*W+col]=s/c; }
    for (int row=0;row<H;++row) for (int col=0;col<W;++col){ float s=0; int c=0;
        for(int dy=-R;dy<=R;++dy){int gy=row+dy; if(gy>=0&&gy<H){s+=tmp[gy*W+col];c++;}}
        out[row*W+col]=s/c; }
}
float max_abs_err(const float* a,const float* b,int n){ float w=0;
    for(int i=0;i<n;++i){float e=fabsf(a[i]-b[i]); if(e>w)w=e;} return w; }

template <typename L> float time_kernel(L launch){
    cudaEvent_t s,e; CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));
    launch(); CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(s));
    for(int r=0;r<RUNS;++r) launch();
    CHECK(cudaEventRecord(e)); CHECK(cudaEventSynchronize(e));
    float ms=0; CHECK(cudaEventElapsedTime(&ms,s,e));
    CHECK(cudaEventDestroy(s)); CHECK(cudaEventDestroy(e)); return ms/RUNS;
}

int main(){
    const int W=2048,H=2048; size_t n=(size_t)W*H, bytes=n*sizeof(float);
    float *hIn=(float*)malloc(bytes), *hOut=(float*)malloc(bytes), *hTmp=(float*)malloc(bytes);
    float *ref2=(float*)malloc(bytes), *refS=(float*)malloc(bytes);
    srand(0); for(size_t i=0;i<n;++i) hIn[i]=(float)rand()/RAND_MAX;

    printf("Blur %dx%d, R=%d (%dx%d window), tile %dx%d\n", W,H,R,2*R+1,2*R+1,TILE_X,TILE_Y);
    ref_2d(hIn,ref2,W,H); ref_sep(hIn,refS,hTmp,W,H);

    float *dIn,*dOut,*dTmp;
    CHECK(cudaMalloc(&dIn,bytes)); CHECK(cudaMalloc(&dOut,bytes)); CHECK(cudaMalloc(&dTmp,bytes));
    CHECK(cudaMemcpy(dIn,hIn,bytes,cudaMemcpyHostToDevice));

    dim3 b2(32,8), g2((W+31)/32,(H+7)/8);                  // 2D blocks (also used by tiled)
    dim3 bt(TILE_X,TILE_Y), gt((W+TILE_X-1)/TILE_X,(H+TILE_Y-1)/TILE_Y);
    double gb = 2.0*(double)n*sizeof(float)/1e9;

    float ms2d=time_kernel([&]{ blur_2d<<<g2,b2>>>(dIn,dOut,W,H); });
    CHECK(cudaGetLastError()); CHECK(cudaMemcpy(hOut,dOut,bytes,cudaMemcpyDeviceToHost));
    float e2d=max_abs_err(ref2,hOut,n);

    float msSep=time_kernel([&]{ blur_h<<<g2,b2>>>(dIn,dTmp,W,H); blur_v<<<g2,b2>>>(dTmp,dOut,W,H); });
    CHECK(cudaGetLastError()); CHECK(cudaMemcpy(hOut,dOut,bytes,cudaMemcpyDeviceToHost));
    float eSep=max_abs_err(refS,hOut,n);

    float msTil=time_kernel([&]{ blur_h_tiled<<<gt,bt>>>(dIn,dTmp,W,H); blur_v_tiled<<<gt,bt>>>(dTmp,dOut,W,H); });
    CHECK(cudaGetLastError()); CHECK(cudaMemcpy(hOut,dOut,bytes,cudaMemcpyDeviceToHost));
    float eTil=max_abs_err(refS,hOut,n);

    printf("\n%-16s %10s %12s %14s  %s\n","kernel","ms/iter","GB/s(eff)","err","status");
    printf("%-16s %10.3f %12.1f %14.2e  %s\n","2D naive",ms2d,gb/(ms2d/1e3),e2d,e2d<1e-4f?"PASS":"FAIL");
    printf("%-16s %10.3f %12.1f %14.2e  %s\n","separable",msSep,gb/(msSep/1e3),eSep,eSep<1e-4f?"PASS":"FAIL");
    printf("%-16s %10.3f %12.1f %14.2e  %s\n","tiled separable",msTil,gb/(msTil/1e3),eTil,eTil<1e-4f?"PASS":"FAIL");
    printf("\nspeedup vs 2D:  separable %.2fx,  tiled %.2fx\n", ms2d/msSep, ms2d/msTil);
    printf("tiled vs separable: %.2fx\n", msSep/msTil);

    cudaFree(dIn);cudaFree(dOut);cudaFree(dTmp);
    free(hIn);free(hOut);free(hTmp);free(ref2);free(refS);
    return 0;
}