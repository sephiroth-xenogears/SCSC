// =============================================================
// 第2段: SoA (Struct of Arrays) 版
//
// 第1段との唯一の違い: データレイアウト。
//   AoS: [a00,a01,a02,a11,a12,a22][a00,...]...  (1行列が48byte連続)
//   SoA: [a00,a00,a00,...][a01,a01,...]...      (同じ要素が連続)
// 計算も並列性も第1段と完全に同一。変えたのは「並べ方」だけ。
//
// なぜ速くなるか: ワープ32スレッドが各自の担当行列の a00 を読むとき、
//   SoA では a00 が連続アドレスに並ぶ → 1トランザクションでコアレス。
//   AoS では48byteストライドでバラけてトランザクションが分割。
//
// ★予測(原則6): 第1段 28.6 GB/s (=ピーク178の16%) に対し、
//   コアレス成立で 100〜150 GB/s に跳ねる見込み(3〜5倍)。
//   178に張り付かないなら次の容疑は倍精度の演算律速 → 切り分けへ。
//
// ビルド: nvcc -O3 -arch=sm_87 inv3_spd_soa.cu -o inv3_spd_soa
// 計測:   ncu --set full ./inv3_spd_soa 1048576
//   → global load efficiency が第1段から大きく改善するはず
// =============================================================
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

// SoA: 6本の独立した配列。同じフィールドが連続して並ぶ。
struct SoA
{
    double *a00,*a01,*a02,*a11,*a12,*a22;/* data */
};

#define CUDA_CHECK(x) do{ cudaError_t cuda_status__=(x); if(cuda_status__!=cudaSuccess){ \
  printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(cuda_status__)); \
  exit(1);} }while(0)

// --- カーネル: 1スレッド=1行列, ただし読み書きはSoAで連続アクセス ---
__global__ void inv3_soa_kernel(SoA A, SoA Ai, int N){
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if(n >= N) return;
    // 読み込み: 各配列から自分のnを読む。
    // ワープ内の隣接スレッド(n, n+1, ...)は連続アドレスを読む → コアレス。
    double a00 = A.a00[n], a01 = A.a01[n], a02 = A.a02[n], a11 = A.a11[n], a12 = A.a12[n], a22 = A.a22[n];
    
    // 計算: 第0段/第1段と完全に同一の閉形式・分岐レス。
    double c00 = a11 * a22 - a12 * a12;
    double c01 = a02 * a12 - a01 * a22;
    double c02 = a01 * a12 - a02 * a11;
    double c11 = a00 * a22 - a02 * a02;
    double c12 = a02 * a01 - a00 * a12;
    double c22 = a00 * a11 - a01 * a01;
    double det = a00 * c00 + a01 * c01 + a02  * c02;
    double invdet = 1.0/det;

    // 書き込み: SoA なのでこちらも連続 → コアレス。
    Ai.a00[n] = c00 * invdet;
    Ai.a01[n] = c01 * invdet;
    Ai.a02[n] = c02 * invdet;
    Ai.a11[n] = c11 * invdet;
    Ai.a12[n] = c12 * invdet;
    Ai.a22[n] = c22 * invdet;

}

// --- ホスト CPU リファレンス(照合用, 第0段と同一) ---
struct Sym3
{
    double a00, a01, a02, a11, a12, a22;
    /* data */
};

static Sym3 make_spd(double m[9], double eps){
    auto col = [&](int i, int k){return m[k * 3 + i];};
    auto dot = [&](int i, int j){return col(i,0) * col(j,0) + col(i,1) * col(j,1) + col(i,2) * col(j,2);};
    Sym3 A;
    A.a00 = dot(0,0) + eps;
    A.a01 = dot(0,1);
    A.a02 = dot(0,2);
    A.a11 = dot(1,1) + eps;
    A.a12 = dot(1,2);
    A.a22 = dot(2,2) + eps;
    return A;
}

static Sym3 inv3_spd_cpu(const Sym3& A){
    double c00 = A.a11 * A.a22 - A.a12 * A.a12;
    double c01 = A.a02 * A.a12 - A.a01 * A.a22;
    double c02 = A.a01 * A.a12 - A.a02 * A.a11;
    double c11 = A.a00 * A.a22 - A.a02 * A.a02;
    double c12 = A.a02 * A.a01 - A.a00 * A.a12;
    double c22 = A.a00 * A.a11 - A.a01 * A.a01;
    double det = A.a00 * c00 + A.a01 * c01 + A.a02 * c02;
    double id = 1.0/det;
    Sym3 R;
    R.a00 = c00 * id;
    R.a01 = c01 * id;
    R.a02 = c02 * id;
    R.a11 = c11 * id;
    R.a12 = c12 * id;
    R.a22 = c22 * id;
    return R;
}

// SoA のデバイス配列を確保
static void alloc_soa(SoA& s, int N){
    size_t b = (size_t)N * sizeof(double);
    CUDA_CHECK(cudaMalloc(&s.a00,b));
    CUDA_CHECK(cudaMalloc(&s.a01,b));
    CUDA_CHECK(cudaMalloc(&s.a02,b));
    CUDA_CHECK(cudaMalloc(&s.a11,b));
    CUDA_CHECK(cudaMalloc(&s.a12,b));
    CUDA_CHECK(cudaMalloc(&s.a22,b));
}

int main(int argc, char** argv){
    int N = (argc > 1)? atoi(argv[1]) : 1048576;
    double eps = 1e-3;
    srand(12345);
    size_t b = (size_t)N * sizeof(double);
    double *h00 = (double*)malloc(b), *h01 = (double*)malloc(b), *h02 = (double*)malloc(b),
    *h11 = (double*)malloc(b), *h12 = (double*)malloc(b), *h22 = (double*)malloc(b);
    Sym3 *hRef = (Sym3*)malloc((size_t)N * sizeof(Sym3));

    for(int n = 0; n < N; n++){
        double m[9];
        for(int k = 0; k < 9; k++) m[k] = (double)rand() / RAND_MAX * 2.0 - 1.0;
        Sym3 A = make_spd(m,eps);
            h00[n] = A.a00;
            h01[n] = A.a01;
            h02[n] = A.a02;
            h11[n] = A.a11;
            h12[n] = A.a12;
            h22[n] = A.a22;
            hRef[n] = inv3_spd_cpu(A);
        }
        
        SoA dA, dAi;
        alloc_soa(dA, N);
        alloc_soa(dAi,N);
        CUDA_CHECK(cudaMemcpy(dA.a00, h00, b, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dA.a01, h01, b, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dA.a02, h02, b, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dA.a11, h11, b, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dA.a12, h12, b, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dA.a22, h22, b, cudaMemcpyHostToDevice));

        int threads = 256, blocks = (N + threads - 1) / threads;
        cudaEvent_t evStart, evStop;
        cudaEventCreate(&evStart);
        cudaEventCreate(&evStop);
        inv3_soa_kernel<<<blocks, threads>>>(dA, dAi, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEventRecord(evStart);
        inv3_soa_kernel<<<blocks, threads>>>(dA, dAi, N);
        cudaEventRecord(evStop);
        CUDA_CHECK(cudaEventSynchronize(evStop));
        float ms = 0;
        cudaEventElapsedTime(&ms, evStart, evStop);
        // 結果回収(SoA → ホスト)して照合
        double *r00 = (double*)malloc(b), *r01 = (double*)malloc(b), *r02 = (double*)malloc(b),
               *r11 = (double*)malloc(b), *r12 = (double*)malloc(b), *r22 = (double*)malloc(b);
        CUDA_CHECK(cudaMemcpy(r00, dAi.a00, b, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(r01, dAi.a01, b, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(r02, dAi.a02, b, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(r11, dAi.a11, b, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(r12, dAi.a12, b, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(r22, dAi.a22, b, cudaMemcpyDeviceToHost));

        double worst = 0;
        int fails = 0;

        for(int n = 0; n < N; n++){
            double d = 0;
            d = fmax(d, fabs(r00[n] - hRef[n].a00));
            d = fmax(d, fabs(r01[n] - hRef[n].a01));
            d = fmax(d, fabs(r02[n] - hRef[n].a02));
            d = fmax(d, fabs(r11[n] - hRef[n].a11));
            d = fmax(d, fabs(r12[n] - hRef[n].a12));
            d = fmax(d, fabs(r22[n] - hRef[n].a22));
            worst = fmax(worst, d);
            if(d > 1e-9) fails++;
        }
    double gbps = (2.0 * (size_t)N * 6 * sizeof(double)) / (ms / 1e3) / 1e9;
    printf("[SoA] N=%d  kernel=%.3f ms  effBW=%.1f GB/s  worst=%.3e  fails=%d => %s\n",
    N, ms, gbps, worst, fails, (fails==0&&worst<1e-9)?"PASS":"FAIL");
    printf("  (第1段 AoS=28.6 GB/s と比較。コアレス成立で跳ねたか?)\n");

    return 0;
}