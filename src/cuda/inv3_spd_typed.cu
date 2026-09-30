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
#include <algorithm>
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
static const double EPS_LIST[] = { 1e-1, 1e-2, 1e-3, 1e-4,1e-5 };
static const int EPS_COUNT = sizeof(EPS_LIST)/sizeof(EPS_LIST[0]);
static const double EPS = 0.1;    //A = MᵀM + EPS·I(正定値・条件数の調整弁)

// ---- SoA: 対称3x3の上三角6要素 ----
template <typename T>
struct SoA6
{
    T* p[6];  // a00,a01,a02,a11,a12,a22 の6配列

};
struct RunResult { double worst, ms, effBW, gflops; };

// 対称3x3の上三角6要素のインデックス（この順序が全コードの唯一の約束）
//   [A00 A01 A02]
//   [A01 A11 A12]
//   [A02 A12 A22]
enum { I00 = 0, I01 = 1, I02 = 2, I11 = 3, I12 = 4, I22 = 5 };




// =====================================================================
// 核心: 余因子展開による SPD 3x3 逆行列(唯一の定義箇所)
//   A⁻¹ = adj(A)/det。SPD なので逆行列も対称 → 上三角6要素のみ返す。
// =====================================================================
template <typename T>
__host__ __device__ inline void inv3_spd_core(
    T a[6],T inv[6]){
    const T c00 = a[I11] * a[I22] - a[I12] * a[I12];
    const T c01 = a[I02] * a[I12] - a[I01] * a[I22];
    const T c02 = a[I01] * a[I12] - a[I02] * a[I11];
    const T c11 = a[I00] * a[I22] - a[I02] * a[I02];
    const T c12 = a[I01] * a[I02] - a[I00] * a[I12];
    const T c22 = a[I00] * a[I11] - a[I01] * a[I01];

    const T det = a[I00] * c00 + a[I01] * c01 + a[I02] * c02;
    const T r = T(1) / det;     // 除算は一度、以後は乗算6回

    inv[I00] = c00 * r;
    inv[I01] = c01 * r;
    inv[I02] = c02 * r;
    inv[I11] = c11 * r;
    inv[I12] = c12 * r;
    inv[I22] = c22 * r;

}

// ---- GPU カーネル(A案: 1スレッド=1行列、SoA) ----
template <typename T>
__global__ void inv3_soa_kernel(SoA6<T> A, SoA6<T> Ainv, int n){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)return;

    T a[6], inv[6];
    #pragma unroll
    for(int k = 0; k < 6; ++k)a[k] = A.p[k][i];

    inv3_spd_core<T>(a, inv);

    #pragma unroll
    for(int k = 0; k < 6; ++k)Ainv.p[k][i] = inv[k];
}

// ---- SPD 行列群の生成(double、単一の真実) ----
// M を一様乱数 [-1,1] で作り A = MᵀM + EPS·I。
static void gen_spd_double(std::vector<double> h[6], int n, double eps){
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
        a[r][c] = s + (r == c ? eps : 0.0);
       }
    h[0][i] = a[0][0];
    h[1][i] = a[0][1];
    h[2][i] = a[0][2];
    h[3][i] = a[1][1];
    h[4][i] = a[1][2];
    h[5][i] = a[2][2];
    }
}

// ---- 対称3x3の固有値（解析解、昇順で返す）----
// 特性方程式を三角関数で解く方法。SPD前提なので実固有値が3つ。
static void eig3_sym(const double a[6], double lam[3]){
    // トレースを引いて偏差行列 B = A - qI にする（q = tr(A)/3）
    const double q = (a[I00] + a[I11] + a[I22]) / 3.0;

    const double b00 = a[I00] - q, b11 = a[I11] - q, b22 = a[I22] - q;
    const double b01 = a[I01], b02 = a[I02], b12 = a[I12];

    //p2 = ||B|||_F^2 / 6 相当（非対角は2回数える）
    const double p2 = (b00 * b00 + b11 * b11 + b22 * b22 + 2.0 * (b01 * b01 + b02 * b02 + b12 * b12)) / 6.0;
    const double p = std::sqrt(p2);

    if(p < 1e-300){ // Aがほぼq*I(三重固有値)
        lam[0] = lam[1] = lam[2] = q;
        return;
    }

    // det(B/p) = cos(3θ) の計算
    const double d00 = b00 / p, d11 = b11 / p, d22 = b22 / p;
    const double d01 = b01 / p, d02 = b02 / p, d12 = b12 / p;
          double r = ( d00 * (d11 * d22 - d12 * d12) 
                     - d01 * (d01 * d22 - d12 * d02) 
                     + d02 * (d01 * d12 - d11 * d02)) / 2.0;

if(r <= -1.0) r = -1.0;
else if(r >= 1.0) r = 1.0;

    const double phi = std::acos(r) / 3.0;
    const double PI = 3.14159265358979323846;

    //降順に出る
    const double e0 = q + 2.0 * p * std::cos(phi);
    const double e2 = q + 2.0 * p * std::cos(phi + (2.0 * PI / 3.0));
    const double e1 = 3.0 * q - e0 - e2; // トレース保存で中間値を出す

    //昇順に詰め直す
    lam[0] = e2; lam[1] = e1; lam[2] = e0;
}

static void cond_stats(const std::vector<double> hA[6], int n, double& kap_min, double& kap_med, double& kap_max){
    std::vector<double> ks(n);
    for(int i = 0; i < n; ++i){
        double a[6], lam[3];
        for(int k = 0; k < 6; ++k) a[k] = hA[k][i];
        eig3_sym(a, lam);
        ks[i] = lam[2] / lam[0];
    }
    std::sort(ks.begin(), ks.end());
    kap_min = ks.front();
    kap_med = ks[n/2];
    kap_max = ks.back();
}

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
static RunResult run(const std::vector<double> hA[6], std::vector<double>hOut[6]){
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
    for(int k = 0; k < 6; ++k){
        CUDA_CHECK(cudaMalloc(&dA.p[k], bytesT));
        CUDA_CHECK(cudaMalloc(&dI.p[k], bytesT));
        CUDA_CHECK(cudaMemcpy(dA.p[k], hin[k].data(), bytesT,cudaMemcpyHostToDevice));
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
            CUDA_CHECK(cudaMemcpy(hout[k].data(), dI.p[k], bytesT, cudaMemcpyDeviceToHost));
            hOut[k].resize(N);
            for(int i = 0; i < N; ++i)hOut[k][i] = (double)hout[k][i];
        }
        const double worst = residual_worst(hA, hOut, N);

    // デバイス解放
        for(int k = 0; k < 6; ++k){
        CUDA_CHECK(cudaFree(dA.p[k]));
        CUDA_CHECK(cudaFree(dI.p[k]));
    }
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    return { worst, ms, effBW, gflops };
}



int main(){
    printf("=== inv3_spd typed A/B: double vs float (SoA, 1thread=1matrix) ===\n");
    printf("生成: A = MtM + eps*I (double, seed=42) / 検証: doubleで A*Ainv-I\n\n");
printf("%-8s | %-10s | %-10s | %-10s | %-12s | %-12s | %-12s | %-12s\n",
       "EPS", "kap_min", "kap_med", "kap_max",
       "CPU worst", "GPUdbl worst", "GPUflt worst", "max|d-f|");
printf("---------+------------+------------+------------+"
       "--------------+--------------+--------------+-------------\n");

    for(int e = 0; e < EPS_COUNT; ++e){
        const double eps = EPS_LIST[e];
    
        static std::vector<double> hA[6];
        gen_spd_double(hA, N, eps);
// ---- 検算: 固有値の和=トレース、積=行列式（最初のEPSのみ）----
        if(e == 0){
            double a[6], lam[3];
            for(int k = 0; k < 6; ++k)a[k] = hA[k][0];
            eig3_sym(a, lam);
            const double tr = a[I00] + a[I11] + a[I22];
            const double det = a[I00] * (a[I11] * a[I22] - a[I12] * a[I12])
                             - a[I01] * (a[I01] * a[I22] - a[I12] * a[I02])
                             + a[I02] * (a[I01] * a[I12] - a[I11] * a[I02]);
            fprintf(stderr, " check: tr=%.6e vs sum(lam)=%.6e, det=%.6e vs prod(lam)=%.6e\n",
                    tr, lam[0]+lam[1]+lam[2], det, lam[0]*lam[1]*lam[2]);
            fprintf(stderr, "lam = %.6e %.6e %.6e\n", lam[0], lam[1], lam[2]);
        }

        double kap_min, kap_med, kap_max;
        cond_stats(hA, N, kap_min, kap_med, kap_max);
    
    // CPUリファレンス(double, 共有コア関数経由)— 物差しの自己検証
        static std::vector<double> hRef[6];
        for(int k = 0; k < 6; ++k)hRef[k].resize(N);
        for(int i = 0; i < N; ++i){
            double a[6], inv[6];
            for(int k = 0; k < 6; ++k)a[k] = hA[k][i];
            inv3_spd_core<double>(a, inv);
            for(int k = 0; k < 6; ++k)hRef[k][i] = inv[k];
        }
        const double w_cpu = residual_worst(hA, hRef, N);
        static std::vector<double> hOutD[6], hOutF[6];
        const RunResult r_dbl = run<double>(hA, hOutD);
        const RunResult r_flt = run<float>(hA, hOutF);
        double diff = 0.0;
        for(int k = 0; k < 6; ++k)
            for(int i = 0; i < N; ++i){
                const double d = std::fabs(hOutD[k][i] - hOutF[k][i]);
                if(d > diff) diff = d;
            }
printf("%-8.1e | %-10.3e | %-10.3e | %-10.3e | %-12.3e | %-12.3e | %-12.3e | %-12.3e\n",
       eps, kap_min, kap_med, kap_max, w_cpu, r_dbl.worst, r_flt.worst, diff);
printf("         time: dbl %.3f ms (%.1f GB/s, %.1f GFLOP/s) | flt %.3f ms (%.1f GB/s, %.1f GFLOP/s) | dbl/flt %.2fx\n",
       r_dbl.ms, r_dbl.effBW, r_dbl.gflops,
       r_flt.ms, r_flt.effBW, r_flt.gflops,
       r_dbl.ms / r_flt.ms);
        fflush(stdout);
    }
    return 0;

}
