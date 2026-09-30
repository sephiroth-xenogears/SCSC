#include <cmath>
#include <cstdio>
#include <cstdlib>  
#include <vector>
#include <random>
#include <cuda_runtime.h>   

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error: %s @ %s:%d\n",                      \
                    cudaGetErrorString(err_), __FILE__, __LINE__);           \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

template <typename T>
__host__ __device__ void rodrigues(const T w[3], T theta, T R[9]){
    const T wx = w[0], wy = w[1], wz = w[2];
    const T c = cos(theta), s = sin(theta), t = 1 - c;

    R[0] = t * wx * wx + c;
    R[1] = t * wx * wy - s * wz;
    R[2] = t * wx * wz + s * wy;

    R[3] = t * wy * wx + s * wz;
    R[4] = t * wy * wy + c;
    R[5] = t * wy * wz - s * wx;

    R[6] = t * wz * wx - s * wy;
    R[7] = t * wz * wy + s * wx;
    R[8] = t * wz * wz + c;
}
template <typename T>
__global__ void rodrigues_kernel(const T* w, const T* theta, T* R, int n){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    rodrigues<T>(&w[3*idx], theta[idx], &R[9*idx]);
}

int main(){
    const int n = 1 << 20; 
    std::vector<double> h_w(3 * n);  //回転軸
    std::vector<double> h_theta(n);   //角度
    std::vector<double> h_R(9 * n);    //出力の受け皿：GPUの結果（1回転あたり9個）
    std::vector<double> h_R_ref(9 * n); // CPUで計算した結果の受け皿
    const double PI = 3.14159265358979323846;

    h_w[0] = 0.0;   // 0番目の回転の wx
    h_w[1] = 0.0;   //              wy
    h_w[2] = 1.0;   //              wz
    h_theta[0] = PI / 2.0;
    std::mt19937 rng(42); // 乱数生成器
    std::uniform_real_distribution<double> dist(-1.0, 1.0); //軸の成分用
    std::uniform_real_distribution<double> dist_theta(0.0, 2.0 * PI); //角度用
    for(int i = 1; i < n; ++i){
        double x = dist(rng), y = dist(rng), z = dist(rng);
        double norm = std::sqrt(x*x + y*y + z*z);
        if( norm < 1e-8) { // ほぼゼロベクトルの場合はデフォルトの軸を使う
            h_w[3*i + 0] = 0.0;
            h_w[3*i + 1] = 0.0;
            h_w[3*i + 2] = 1.0;
        }
        else {
            h_w[3*i + 0] = x / norm;
            h_w[3*i + 1] = y / norm;
            h_w[3*i + 2] = z / norm;
        }
        h_theta[i] = dist_theta(rng);  
    }
    //デバイス確保・転送
    double *d_w = nullptr;
    CUDA_CHECK(cudaMalloc(&d_w, sizeof(double) * 3 * n));
    double *d_theta = nullptr;
    CUDA_CHECK(cudaMalloc(&d_theta, sizeof(double) * n));
    double *d_R = nullptr;
    CUDA_CHECK(cudaMalloc(&d_R, sizeof(double) * 9 * n));
    CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), sizeof(double) * 3 * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_theta, h_theta.data(), sizeof(double) * n, cudaMemcpyHostToDevice));
    // GPUで計算
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;
    rodrigues_kernel<<<numBlocks, blockSize>>>(d_w, d_theta, d_R, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    // 結果をホストにコピー
    CUDA_CHECK(cudaMemcpy(h_R.data(), d_R, sizeof(double) * 9 * n, cudaMemcpyDeviceToHost));

    // 3x3 の形で表示
    printf("R = exp([w]theta), w=(%f, %f, %f), theta=%f\n", h_w[0], h_w[1], h_w[2], h_theta[0]);
    for(int r = 0; r < 3; ++r){
        printf("%12.4e %12.4e %12.4e\n", h_R[3*r + 0], h_R[3*r + 1], h_R[3*r + 2]);
    }
    // CPU リファレンス（同じコア関数を CPU で呼ぶ）
    for(int i = 0; i < n; ++i)
    rodrigues<double>(&h_w[3*i], h_theta[i], &h_R_ref[9*i]);

   // 最大誤差
    double worst = 0.0;
    int nan_count = 0;
    for(int k = 0; k < 9 * n; ++k){
    if(std::isnan(h_R[k]) || std::isnan(h_R_ref[k])){
        nan_count++;
        continue;
    }
    const double d = std::fabs(h_R[k] - h_R_ref[k]);
    if(d > worst) worst = d; 
    }
    printf("max |R_gpu - R_cpu| = %.3e\n", worst);
    printf("NaN count: %d\n", nan_count);
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_theta));
    CUDA_CHECK(cudaFree(d_R));  

return 0;
}

    

