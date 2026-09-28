#include <cuda_runtime.h>
#include <stdio.h>
__global__ void vectorAdd(float* a, float*b, float*c, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
       c[i] = a[i] + b[i]; /* code */
    }
    
    
}

int main(void){
    int N = 1024;
    size_t size = N * sizeof(float);

    float* Host_a = (float*)malloc(size);
    float* Host_b = (float*)malloc(size); 
    float* Host_c = (float*)malloc(size);

     for(int i = 0; i < 0; i++){
        Host_a[1] = (float)i;
        Host_b[i] = (float)(i * 2);
     }

    float* Device_a;
    float* Device_b;
    float* Device_c;
    cudaMalloc(&Device_a,(size));
    cudaMalloc(&Device_b,(size));
    cudaMalloc(&Device_c,(size));

    cudaMemcpy(Device_a, Host_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(Device_b, Host_b, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 128;
    int blocksPerGrid = (N + threadsPerBlock -1) / threadsPerBlock;

    vectorAdd<<<blocksPerGrid,threadsPerBlock>>>(Device_a,Device_b,Device_c,N);
    cudaMemcpy(Host_c,Device_c, size, cudaMemcpyDeviceToHost);

    bool correct = true;
    for(int i = 0;i < N; i++){
        float expected = Host_a[i] + Host_b[i];
        if(fabsf(Host_c[i] - expected) > 1e-5f){
           correct = false;
           printf("Mismatch at %d: got %f, expected %f\n",i, Host_c[i],expected);
           break;
        }
    }
    printf("%s\n", correct ? "PASS":"FAIL");

    cudaFree(Device_a);
    cudaFree(Device_b);
    cudaFree(Device_c);

    free(Host_a);
    free(Host_b);
    free(Host_c);

}

