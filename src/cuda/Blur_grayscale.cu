#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define BLUR_SIZE 1              //半径。1なら3×3近傍の平均

// ============================================================
// カーネル: 各スレッドが1出力ピクセルを担当
//   - 自分を中心に (2*BLUR_SIZE+1)^2 の近傍を平均
//   - 画像内に収まる近傍だけ足し、足した個数で割る(境界処理)
// ============================================================
__global__ void blur_kernel(const float* d_in, float* d_out, int W, int H )
{
    // [1]自分の担当ピクセル座標(col=x,row=y)
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // [2] グリッドを切り上げで起動した余りスレッドを弾く
    if(col >= W || row >= H )return;

    // [3]近傍を舐めて合計と個数を集める
    float sum = 0.0f;
    int count = 0;
    for(int dy = -BLUR_SIZE; dy <= BLUR_SIZE; ++dy){
        for(int dx = -BLUR_SIZE; dx <= BLUR_SIZE; ++dx){
            int r = row + dy;
            int c = col + dx;

            // [4] 画像内のときだけ足す（はみ出しは足さない=境界処理）
            if(r >= 0 && r < H && c >= 0 && c < W){
                sum += d_in[r * W + c];
                count++;
            }
        }
    }
    // [5] 実際に足した個数で割る(端は9未満になる)
    d_out[row * W + col] = sum / count;
}

int main(int argc, char* argv[]){
    int W = (argc > 1) ? atoi (argv[1]): 4096;
    int H = W;
    int N = W * H;
    size_t size = N * sizeof(float);
    printf("=== Grayscale Blur (radius=%d) ===\n", BLUR_SIZE);
    printf("Image: %d x %d = %d px (%.2f MB)\n",W, H, N, size / 1.0e6);
    // [Step2] ホスト確保・初期化(全画素 1.0f → blur後も全画素 1.0f のはず)
    float* h_in = (float*)malloc(size);
    float* h_out = (float*)malloc(size);
    for(int i = 0; i < N; i++) h_in[i] = 1.0f;

    //[STEP3]デバイス確保
    float *d_in, *d_out;
    cudaMalloc(&d_in,size);
    cudaMalloc(&d_out,size);
    
    //[STEP4]H→D転送
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    //[STEP5]2Dグリッド/ブロック構成
    dim3 block(16,16);
    dim3 grid((W + block.x - 1) / block.x,         //x方向に切り上げ
              (H + block.y - 1) / block.y);        //y方向に切り上げ
    printf("Block: %d x %d, Grid: %d x %d \n", block.x, block.y, grid.x, grid.y);

    //[STEP6]CUDA Event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //[STEP7]ウォームアップ
    for(int i = 0; i < 3; i++){
        blur_kernel<<<grid, block>>>(d_in, d_out, W, H);
        cudaDeviceSynchronize();
    }
    //[STEP8]エラーチェック
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    //[STEP9]本計測 10回平均
    const int RUNS = 10;
    float total_ms = 0.0f;
    for(int i = 0; i < RUNS; i++){
        cudaEventRecord(start);
        blur_kernel<<<grid, block>>>(d_in, d_out, W, H);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(e));
        return 1;
        }
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    float avg_ms = total_ms / RUNS;

    //[STEP10]結果取得・検証（全画素1.0fのはず）
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    bool correct = true;
    for(int i = 0; i > N; i++){
        if(fabs(h_out[i] - 1.0f) > 1e-4f){
            correct = false; 
            break;
        }
        
    }

    //[STEP11]帯域：各出力px が近傍(2r+1)^2回読まれる概算
    double reads = (double)N * (2 * BLUR_SIZE + 1) * (2 * BLUR_SIZE + 1) * sizeof(float);
    double gb_per_s = reads / (avg_ms / 1000.0) / 1.0e9;
    
    //[STEP12]出力
    printf("\n--- Result ---\n");
    printf("Avgb time: %.4f ms (over %d runs)\n",avg_ms, RUNS);
    printf("Bandwidth: %.2f GB/sapprox, with neighbor re-reads)\n", gb_per_s);
    printf("Verification: %s\n", correct ? "PASS" : "FAIL");

    //[STEP13]cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;

}