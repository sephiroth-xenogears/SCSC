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

template <typename T> 
__global__ void so3_check_kernel(const T* R, T* orth_err, T* det_err, int n){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    const T* Rmat = &R[9*idx];
    T RtR[9];
    // RtR = R^T * R
    for(int i = 0; i < 3; ++i){ 
        for(int j = 0; j < 3; ++j){
            RtR[3*i + j] = 0;
            for(int k = 0; k < 3; ++k){
                RtR[3*i + j] += Rmat[3*k + i] * Rmat[3*k + j];
            }
        }
    }
    // orth_err = ||RtR - I||_F
    T err = 0;
    for(int i = 0; i < 3; ++i){
        for(int j = 0; j < 3; ++j){
            T diff = RtR[3*i + j] - (i == j ? 1 : 0);
            err += diff * diff;
        }
    }
    orth_err[idx] = sqrt(err);
    // det_err = |det(R) - 1|
    T det = Rmat[0] * (Rmat[4] * Rmat[8] - Rmat[5] * Rmat[7]) -
             Rmat[1] * (Rmat[3] * Rmat[8] - Rmat[5] * Rmat[6]) +
             Rmat[2] * (Rmat[3] * Rmat[7] - Rmat[4] * Rmat[6]);
    det_err[idx] = fabs(det - 1);
}            
int main() {
    const int n = 1 << 20; // 2^20 matrices
    std::vector<double> h_w(3 * n);  // rotation axes
    std::vector<double> h_theta(n);   // angles
    std::vector<double> h_R(9 * n);    // output: R matrices
    std::vector<double> h_orth_err(n); // orthogonality errors
    std::vector<double> h_det_err(n);  // determinant errors        
    const double PI = 3.14159265358979323846;
    h_w[0] = 0.0;   // first rotation axis wx
    h_w[1] = 0.0;   //              wy
    h_w[2] = 1.0;   //              wz
    h_theta[0] = PI / 2.0; // first rotation angle
    std::mt19937 rng(42); // random number generator
    std::uniform_real_distribution<double> dist(-1.0, 1.0); // for axis components
    std::uniform_real_distribution<double> dist_theta(0.0, 2.0 * PI); // for angles
    for(int i = 1; i < n; ++i){
        double x = dist(rng), y = dist(rng), z = dist(rng);
        double norm = std::sqrt(x*x + y*y + z*z);
        if( norm < 1e-8) { // if nearly zero vector, use default axis
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
    // Allocate device memory and transfer data 
    double *d_w = nullptr, *d_theta = nullptr, *d_R = nullptr, *d_orth_err = nullptr, *d_det_err = nullptr;
    CUDA_CHECK(cudaMalloc(&d_w, 3 * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_theta, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_R, 9 * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_orth_err, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_det_err, n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), 3 * n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_theta, h_theta.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    // Launch kernel to compute R matrices
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    rodrigues_kernel<<<numBlocks, blockSize>>>(d_w, d_theta, d_R, n);
    int max_orth_index = -1, max_det_index = -1;
    CUDA_CHECK(cudaGetLastError());

    const int bad = 500;
    double tmp[9];
    CUDA_CHECK(cudaMemcpy(tmp, d_R + bad * 9, 9 * sizeof(double), cudaMemcpyDeviceToHost));
    for(int k = 0; k < 9; ++k)tmp[k] *= 1.001;
    CUDA_CHECK(cudaMemcpy(d_R + bad * 9, tmp, 9 * sizeof(double), cudaMemcpyHostToDevice));
    // Launch kernel to check orthogonality and determinant
    so3_check_kernel<<<numBlocks, blockSize>>>(d_R, d_orth_err, d_det_err, n);
    CUDA_CHECK(cudaGetLastError());
    // Copy results back to host
    //CUDA_CHECK(cudaMemcpy(h_R.data(), d_R, 9 * n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_orth_err.data(), d_orth_err, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_det_err.data(), d_det_err, n * sizeof(double), cudaMemcpyDeviceToHost));
    double max_orth_err = 0.0, max_det_err = 0.0;
    for(int i = 0; i < n; ++i){
        if(h_orth_err[i] > max_orth_err) max_orth_err = h_orth_err[i];
        if(h_det_err[i] > max_det_err) max_det_err = h_det_err[i];
        if(std::isnan(h_orth_err[i]) || std::isnan(h_det_err[i])){
            printf("NaN detected at index %d\n", i);
        }
        if(max_orth_index < 0 && h_orth_err[i] > 1e-6) max_orth_index = i;
        if(max_det_index < 0 && h_det_err[i] > 1e-6) max_det_index = i;       
    }
    printf("Max orthogonality error: %e\n", max_orth_err);
    printf("Max determinant error: %e\n", max_det_err);
    printf("First index with orthogonality error > 1e-6: %d\n", max_orth_index);
    printf("First index with determinant error > 1e-6: %d\n", max_det_index);
    // Free device memory
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_theta));
    CUDA_CHECK(cudaFree(d_R));
    CUDA_CHECK(cudaFree(d_orth_err));
    CUDA_CHECK(cudaFree(d_det_err));        
    return 0;
}