#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__global__ void reduceNaive(float* d_in, float* d_out, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N)
    {
        atomicAdd(d_out,d_in[i]);
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
    cudaMalloc(&d_a,(size));
    cudaMalloc(&d_c,sizeof(float));
    cudaMemset(d_c, 0, sizeof(float));
    
    cudaMemcpy(d_a,h_a,size,cudaMemcpyHostToDevice);

    int threadPerBlock = 256;
    int blocksPerGrid = (N + threadPerBlock -1) / threadPerBlock;

    reduceNaive<<<blocksPerGrid,threadPerBlock>>>(d_a,d_c,N);
    
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

