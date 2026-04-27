#include <cuda_runtime.h> 
#include <stdio.h>
#include <math.h>

__global__ void shared_memory_reduction(float* d_in, float* d_out, int N)
{
    __shared__ float s_data[256]; //項目2
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    //項目6:範囲外は単位元 0.0f
    s_data[tid] = (i < N)? d_in[i] : 0.0f;
    __syncthreads();   //項目4:ロード(tid) → reduction(別tid)の間

    // 項目3:Sequential Addressinng
    for(int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if(tid < stride)
        {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();  //項目4：　段階N（tid）→段階N+1(別tid)の間

    }
    // 項目1,5: thread 0が atomicAdd,s_data[0]のみ最終合計
    if (tid == 0)
    {
        atomicAdd(d_out, s_data[0]);
    }
    
}
int main(void)
{
    int N = 1024;
    size_t size = N * sizeof(float);

    float* h_a = (float*)malloc(size);
    float h_c;

    for(int i = 0; i < N; i++)
    {
        h_a[i] = (float)i;
    }

    float* d_a;
    float* d_c;
    cudaMalloc(&d_a,size);
    cudaMalloc(&d_c,sizeof(float));
    cudaMemset(d_c, 0, sizeof(float));
    
    cudaMemcpy(d_a,h_a,size,cudaMemcpyHostToDevice);

    int threadPerBlock = 256;
    int blocksPerGrid = (N + threadPerBlock -1) / threadPerBlock;

    shared_memory_reduction<<<blocksPerGrid,threadPerBlock>>>(d_a,d_c,N);
    
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        printf("Kernel launch error: %s\n",cudaGetErrorString(err));
        return 1;
    }
    err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        printf("Kernel execution error: %s\n",cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(&h_c,d_c,sizeof(float),cudaMemcpyDeviceToHost);
    
    float expected = 0.0f;
    for(int i = 0; i < N; i++)
    {
        expected += h_a[i];
    }

    bool correct = fabsf(h_c - expected) < 1e-3f;

    printf("%s (got %f,expected %f)\n", correct? "PASS":"FAIL",h_c, expected);

    cudaFree(d_a);
    cudaFree(d_c);

    free(h_a);
}