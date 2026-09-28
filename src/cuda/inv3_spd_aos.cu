// =============================================================
// 第1段: 素朴な GPU 版 (AoS, 1スレッド1行列)
//
// 目的: わざと「遅い」ものを作る。AoS レイアウトでコアレスが崩れ、
//       並列性(軸1)は足りているのにメモリパターン(軸3)で律速する
//       ことを Nsight で観測する。第2段(SoA)との対比の基準点。
//
// 第0段(CPU)と同じ余因子展開。結果はCPUリファレンスと照合してPASS判定。
//
// ★予測(原則6: 計測前に述べる) -- Jetson で Nsight Compute を回す前に:
//   - global load efficiency: 大きく低下する見込み。
//     理由: ワープ32スレッドが各自の担当行列(AoS=48byte間隔)の
//     同じフィールド(例 a00)を読むと、アドレスが48byteストライドで
//     飛び、1トランザクションにまとまらない。
//   - occupancy(占有率): 高く出るはず(スレッドは大量に立つ)。
//     →「並列性はあるのにメモリで律速」= occupancy高 × bandwidth低
//   - 実効帯域(GB/s): ピークの数分の一に留まる見込み。
//   この3点を Nsight で確認し、第2段(SoA)で改善するかを叩く。
//
// ビルド: nvcc -O3 -arch=sm_87 inv3_spd_aos.cu -o inv3_spd_aos
//   (Jetson AGX Orin は sm_87)
// 計測:   ncu --set full ./inv3_spd_aos 1048576
//   特に: l1tex__t_sectors_pipe_lsu_mem_global_op_ld (load)
//         smsp__sass_average_data_bytes_per_sector_mem_global_op_ld (efficiency)
//         gpu__time_duration / 帯域
// =============================================================
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

// --- AoS: 第0段の Sym3 をそのまま GPU に持っていく(=遅い版) ---
// レイアウト: [a00,a01,a02,a11,a12,a22] が1行列ぶん連続。
// 行列n の a00 と 行列n+1 の a00 は 48byte(6*double) 離れる。
struct sym3
{
    double a00,a01,a02,a11,a12,a22;/* data */
};

#define CUDA_CHECK(x) do{ cudaError_t cuda_status__=(x); if(cuda_status__!=cudaSuccess){ \
  printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(cuda_status__)); \
  exit(1);} }while(0)

// --- デバイス側: 3x3 SPD 逆行列(第0段と同一の閉形式・分岐レス) ---
__device__ __forceinline__ sym3 inv3_spd_dev(const sym3& A){
    double c00 = A.a11 * A.a22 - A.a12 * A.a12;
    double c01 = A.a02 * A.a12 - A.a01 * A.a22;
    double c02 = A.a01 * A.a12 - A.a02 * A.a11;
    double c11 = A.a00 * A.a22 - A.a02 * A.a02;
    double c12 = A.a02 * A.a01 - A.a00 * A.a12;
    double c22 = A.a00 * A.a11 - A.a01 * A.a01;
    double det = A.a00 * c00 + A.a01 * c01 + A.a02 * c02;

    double invdet = 1.0 / det;        // 除算1回 → 以降は乗算(GPUで除算は高コスト)
    sym3 R;
    R.a00 = c00 * invdet;
    R.a01 = c01 * invdet;
    R.a02 = c02 * invdet;
    R.a11 = c11 * invdet;
    R.a12 = c12 * invdet;
    R.a22 = c22 * invdet;
    return R;
}

__global__ void inv3_aos_kernel(const sym3* __restrict__ A,sym3* __restrict__ Ai, int N){
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if(n >= N) return;
    // AoS: A[n] を読むと 48byte の構造体を1スレッドが丸ごと触る。
    // ワープ内の隣接スレッドは 48byte 離れた構造体を読む → コアレス崩れ。
    Ai[n] = inv3_spd_dev(A[n]);
}

// --- ホスト側 CPU リファレンス(第0段と同一, 照合用) ---
static sym3 make_spd(double m[9], double eps){
    auto col = [&](int i, int k){return m[k * 3 + i];};
    auto dot = [&](int i, int j){return col(i,0) * col(j,0) + col(i,1) + col(j,1) + col(i,2) * col(j,2);};
    sym3 A;
    A.a00 = dot(0,0) + eps;
    A.a01 = dot(0,1);
    A.a02 = dot(0,2);
    A.a11 = dot(1,1) + eps;
    A.a12 = dot(1,2);
    A.a22 = dot(2,2) + eps;
    return A;
}

static sym3 inv3_spd_cpu(const sym3& A){
    double c00=A.a11*A.a22-A.a12*A.a12, c01=A.a02*A.a12-A.a01*A.a22,
           c02=A.a01*A.a12-A.a02*A.a11, c11=A.a00*A.a22-A.a02*A.a02,
           c12=A.a02*A.a01-A.a00*A.a12, c22=A.a00*A.a11-A.a01*A.a01;
    double det=A.a00*c00+A.a01*c01+A.a02*c02, id=1.0/det;
    sym3 R; 
    R.a00=c00*id; 
    R.a01=c01*id; 
    R.a02=c02*id;
    R.a11=c11*id; 
    R.a12=c12*id; 
    R.a22=c22*id; 
    return R;
}

int main(int argc, char** argv){
    int N = (argc > 1)? atoi(argv[1]) : 1048576;
    double eps = 1e-3;
    srand(12345);

    size_t bytes = (size_t)N * sizeof(sym3);
    sym3 *hA = (sym3*)malloc(bytes), *hAi = (sym3*)malloc(bytes), *hRef = (sym3*)malloc(bytes);
    for(int n = 0; n < N; n++){
        double m[9];
        for(int k = 0; k < 9; k++){
            m[k] = (double)rand() / RAND_MAX * 2.0 - 1.0;
            hA[n] = make_spd(m,eps);
            hRef[n] = inv3_spd_cpu(hA[n]);

        }
    sym3 *da, *dAi;
    CUDA_CHECK(cudaMalloc(&da, bytes));
    CUDA_CHECK(cudaMalloc(&dAi, bytes));
    CUDA_CHECK(cudaMemcpy(da, dAi, bytes, cudaMemcpyHostToDevice));

    int threads = 256, blocks = (N + threads - 1)/threads;
    // ウォームアップ + 計測(簡易。詳細は Nsight で)
    cudaEvent_t evStart, evStop;
    cudaEventCreate(&evStart);
    cudaEventCreate(&evStop);
    inv3_aos_kernel<<<blocks,threads>>>(da, dAi, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(evStart);
    inv3_aos_kernel<<<blocks,threads>>>(da, dAi, N);
    cudaEventRecord(evStop);
    CUDA_CHECK(cudaEventSynchronize(evStop));
    float ms = 0;
    cudaEventElapsedTime(&ms, evStart, evStop);

    CUDA_CHECK(cudaMemcpy(hAi, dAi, bytes, cudaMemcpyDeviceToHost));
    // CPU リファレンスと照合
    double worst = 0;
    int fails = 0;
    for(int n = 0; n < N; n++){
        double d = 0;
        d = fmax(d, fabs(hAi[n].a00 - hRef[n].a00));
        d = fmax(d, fabs(hAi[n].a01 - hRef[n].a01));
        d = fmax(d, fabs(hAi[n].a02 - hRef[n].a02));
        d = fmax(d, fabs(hAi[n].a11 - hRef[n].a11));
        d = fmax(d, fabs(hAi[n].a12 - hRef[n].a12));
        d = fmax(d, fabs(hAi[n].a22 - hRef[n].a22));
        worst = fmax(worst, d);
        if(d > 1e-9)fails++;

        // 帯域: 読み(A) + 書き(Ai) = 2 * bytes を ms で割る
        double gbps = (2.0*bytes)/(ms/1e3)/1e9;
        printf("[AoS] N=%d  kernel=%.3f ms  effBW=%.1f GB/s  worst=%.3e  fails=%d => %s\n",
              N, ms, gbps, worst, fails, (fails==0&&worst<1e-9)?"PASS":"FAIL");
        printf("  (予測: Jetson の ~178GB/s ピークに対し、AoS なのでコアレス崩れで低めに出るはず)\n");

    cudaFree(da); 
    cudaFree(dAi); 
    free(hA); 
    free(hAi); 
    free(hRef);
    return 0;
    }

   }
}