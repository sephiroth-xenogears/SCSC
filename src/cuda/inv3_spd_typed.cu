// inv3_spd_typed.cu — 第2段の謎解き: double→float 実験
//
// 設計方針:
//   (1) 余因子展開は inv3_spd_core<T> に一箇所化(__host__ __device__)。
//       GPU カーネルも CPU リファレンスも同じ式を呼ぶ(DRY、写経ミスの構造的封じ)。
//   (2) 行列生成は double で一度だけ行い(単一の真実)、float 経路へはキャストで供給。
//   (3) 検証は常に double で実施: 残差 R = A(double)·Ainv - I の最大絶対値。
//       物差し(原則10)は型実験の影響を受けない場所に置く。
//   (4) run<double>() と run<float>() を同一バイナリで連続実行。
//       双方とも SoA レイアウト・同一グリッド構成 → 差は「型」のみ。
//
// ビルド:  nvcc -O3 -arch=sm_87 inv3_spd_typed.cu -o inv3_spd_typed
// 実行:    ./inv3_spd_typed
// プロファイル: sudo /usr/local/cuda-12.6/bin/ncu \
//              --section SpeedOfLight --section ComputeWorkloadAnalysis \
//              ./inv3_spd_typed
//   (カーネルが double 版・float 版の2つ出る。テンプレート実引数で区別できる)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_runtime.h>
// ---- エラーチェック(内部変数は err_ : 以前の e 衝突の教訓) ----
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error: %s @ %s:%d\n",                      \
                    cudaGetErrorString(err_), __FILE__, __LINE__);           \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

// ---- パラメータ ----
static const int N = 1 << 20;     // 1,048,576 行列(第2段と同一)
static const int BLOCK = 256;     
static const int NREP  = 20;      //計時はNREP回平均(ウォームアップ別)
static const double EPS = 0.1;    //A = MᵀM + EPS·I(正定値・条件数の調整弁)

// ---- SoA: 対称3x3の上三角6要素 ----
template <typename T>
struct SoA6
{
    T *a00, *a01, *a02, *a11, *a12, *a22;/* data */
};

// =====================================================================
// 核心: 余因子展開による SPD 3x3 逆行列(唯一の定義箇所)
//   A⁻¹ = adj(A)/det。SPD なので逆行列も対称 → 上三角6要素のみ返す。
// =====================================================================
template <typename T>
__host__ __device__ inline void inv3_spd_core(
    T a00, T a01, T a02, T a11, T a12, T a22,
    T &i00, T &i01, T &i02, T &i11, T &i12, T &i22){
    const T c00 = a11 * a22 - a12 * a12;
    const T c01 = a02 * a12 - a01 * a22;
    const T c02 = a01 * a12 - a02 * a11;
    const T c11 = a00 * a22 - a02 * a02;
    const T c12 = a01 * a02 - a00 * a12;
    const T c22 = a00 * a11 - a01 * a01;

    const T det = a00 * c00 + a01 * c01 + a02 * c02;
    const T r = T(1) / det;     // 除算は一度、以後は乗算6回

    i00 = c00 * r;
    i01 = c01 * r;
    i02 = c02 * r;
    i11 = c11 * r;
    i12 = c12 * r;
    i22 = c22 * r;

}

// ---- GPU カーネル(A案: 1スレッド=1行列、SoA) ----
template <typename T>
__global__ void inv3_soa_kernel(SoA6<T> A, SoA6<T> Ainv, int n){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)return;

    T i00, i01, i02, i11, i12, i22;
    inv3_spd_core<T>(A.a00[i], A.a01[i], A.a02[i],
                     A.a11[i], A.a12[i], A.a22[i],
                     i00, i01, i02, i11, i12, i22);
    Ainv.a00[i] = i00;
    Ainv.a01[i] = i01;
    Ainv.a02[i] = i02;
    Ainv.a11[i] = i11;
    Ainv.a12[i] = i12;
    Ainv.a22[i] = i22;
}

// ---- SPD 行列群の生成(double、単一の真実) ----
// M を一様乱数 [-1,1] で作り A = MᵀM + EPS·I。
static void gen_spd_double(std::vector<double> h[6], int n){
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);

    for(int k = 0; k < 6; ++k) h[k].resize(n);

    for(int i = 0; i < n; ++i){
        double m[3][3];
        for(int r = 0; r < 3; ++r)
           for(int c = 0; c < 3; ++c)
              m[r][c] = uni(rng);
    // A = MᵀM + EPS·I(上三角のみ)
    double a[3][3];
    for(int r = 0; r < 3; ++r)
       for(int c = r; c < 3; ++c){
        double s = 0.0;
        for(int k = 0; k < 3; ++k) s += m[k][r] * m[k][c];
        a[r][c] = s + (r == c ? EPS : 0.0);
       }
    h[0][i] = a[0][0];
    h[1][i] = a[0][1];
    h[2][i] = a[0][2];
    h[3][i] = a[1][1];
    h[4][i] = a[1][2];
    h[5][i] = a[2][2];
    }
}

// ---- 検証(常に double): worst = max |A·Ainv − I| ----
// A は生成時の double 値、Ainv は各経路の結果を double に持ち上げて評価。
static double residual_worst(const std::vector<double> hA[6], const std::vector<double>hI[6], int n){
    double worst = 0.0;
    for(int i = 0; i < n; ++i){
        const double a[3][3] = {
            { hA[0][i], hA[1][i], hA[2][i] },
            { hA[1][i], hA[3][i], hA[4][i] },
            { hA[2][i], hA[4][i], hA[5][i] }
        };
        const double v[3][3] = {
            { hI[0][i], hI[1][i], hI[2][i] },
            { hI[1][i], hI[3][i], hI[4][i] },
            { hI[2][i], hI[4][i], hI[5][i] }
        };
        for(int r = 0; r < 3; ++r)
           for(int c = 0; c < 3; ++c){
            double s = 0.0;
            for(int k = 0; k < 3; ++k) s += a[r][k] * v[k][c];
            const double d = std::fabs(s - (r == c ? 1.0 : 0.0));
            if(d > worst) worst = d;
           }
    }
    return worst;
}
// ---- 1経路の実行: 型 T で GPU 実行し、時間・帯域・残差を報告 ----
template<typename T>
static void run(const char *label, const std::vector<double> hA[6], std::vector<double>hOut[6],double tol){
    const size_t bytesT = sizeof(T) * (size_t)N;

    // ホスト側: double → T へ変換(生成は一度きり、経路ごとにキャスト)
    std::vector<T> hin[6], hout[6];
    for(int k = 0; k < 6; ++k){
        hin[k].resize(N);
        hout[k].resize(N);
        for(int i = 0; i < N; ++i)hin[k][i] = (T)hA[k][i];
    }

    //デバイス確保・転送
    SoA6<T> dA{}, dI{};
    T **pa[6] = { &dA.a00, &dA.a01, &dA.a02, &dA.a11, &dA.a12, &dA.a22 };
    T **pi[6] = { &dI.a00, &dI.a01, &dI.a02, &dI.a11, &dI.a12, &dI.a22 };
    for(int k = 0; k < 6; ++k){
        CUDA_CHECK(cudaMalloc(pa[k], bytesT));
        CUDA_CHECK(cudaMalloc(pi[k], bytesT));
        CUDA_CHECK(cudaMemcpy(*pa[k], hin[k].data(), bytesT,cudaMemcpyHostToDevice));
    }

        const int grid = (N + BLOCK - 1) / BLOCK;

        // ウォームアップ1回 → NREP回を event 計時
        inv3_soa_kernel<<<grid, BLOCK>>>(dA, dI, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t ev0, ev1;
        CUDA_CHECK(cudaEventCreate(&ev0));
        CUDA_CHECK(cudaEventCreate(&ev1));
        CUDA_CHECK(cudaEventRecord(ev0));
        for(int r = 0; r < NREP; ++r)
           inv3_soa_kernel<T><<<grid, BLOCK>>>(dA, dI, N);
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));

        float ms_total = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms_total, ev0, ev1));
        const double ms = ms_total / NREP;

        // 実効帯域: 読み6 + 書き6 要素/行列
        const double bytes_moved = 12.0 * sizeof(T) * (double)N;
        const double effBW = bytes_moved / (ms * 1e-3) / 1e9;     // GB/s
        // 概算 30 flop/行列 → 演算強度 AI = 30 / (12·sizeof(T))
        const double gflops = 30.0 * (double)N / (ms * 1e-3) / 1e9;

        // 結果回収 → double へ持ち上げ → 残差検証
        for(int k = 0; k < 6; ++k){
            CUDA_CHECK(cudaMemcpy(hout[k].data(), *pi[k], bytesT, cudaMemcpyDeviceToHost));
            hOut[k].resize(N);
            for(int i = 0; i < N; ++i)hOut[k][i] = (double)hout[k][i];
        }
        const double worst = residual_worst(hA, hOut, N);
        const bool   pass  = (worst < tol);
        printf("[%s] N=%d  kernel=%.3f ms  effBW=%.1f GB/s  ~%.1f GFLOPS  "
       "AI=%.3f flop/B  worst=%.3e  => %s (tol=%.0e)\n",
        label, N, ms, effBW, gflops,
        30.0 / (12.0 * sizeof(T)), worst, pass ? "PASS" : "FAIL", tol);
    for(int k = 0; k < 6; ++k){
        CUDA_CHECK(cudaFree(*pa[k]));
        CUDA_CHECK(cudaFree(*pi[k]));
    }
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
}

int main(){
    printf("=== inv3_spd typed A/B: double vs float (SoA, 1thread=1matrix) ===\n");
    printf("生成: A = MtM + %.2f*I (double, seed=42) / 検証: doubleで A*Ainv-I\n\n",
           EPS);

    static std::vector<double> hA[6];
    gen_spd_double(hA, N);
    
    // CPUリファレンス(double, 共有コア関数経由)— 物差しの自己検証
    {
        static std::vector<double> hRef[6];
        for(int k = 0; k < 6; ++k)hRef[k].resize(N);
        for(int i = 0; i < N; ++i){
            inv3_spd_core<double>(hA[0][i], hA[1][i], hA[2][i],
                                  hA[3][i], hA[4][i], hA[5][i],
                                  hRef[0][i], hRef[1][i], hRef[2][i],
                                  hRef[3][i], hRef[4][i], hRef[5][i]);
        }
        const double worst = residual_worst(hA, hRef, N);
        printf("[CPU ref double] worst=%.3e  => %s\n\n",
               worst, worst < 1e-9 ? "PASS" : "FAIL");     
    }

    static std::vector<double> hOutD[6], hOutF[6];

    run<double>("GPU double", hA, hOutD, 1e-9);    // 第2段の再現(アンカー)
    run<float>("GPU float", hA, hOutF, 1e-3);      // ★本実験

    // 型間の直接比較: ESKF の double/float 判断材料
    double diff = 0.0;
    for(int k = 0; k < 6; ++k)
        for(int i = 0; i < N; ++i){
            const double d = std::fabs(hOutD[k][i] - hOutF[k][i]);
            if(d > diff) diff = d;
        }
    printf("\n[double vs float] 逆行列要素の最大差 = %.3e\n", diff);
    printf("(EPS=%.2f の条件数での値。共分散行列が悪条件ならここが伸びる)\n", EPS);

    return 0;
}
