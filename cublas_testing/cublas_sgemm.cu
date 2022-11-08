#include<cublas_v2.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<math.h>
#include<iostream>
#include <climits>
#include<stdio.h>

bool verify_solution(float *a, float *b, float *c, int n){
    float temp;
    float epsilon = 0.001;
    float diff = 0.0;
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            temp = 0;
            for(int k = 0; k < n; k++){
                temp += a[k * n + i] * b[j * n + k];
            }
            diff = fabs(c[j*n+i] - temp);
            if(diff>epsilon) return false;
        }
    }
    return true;
}

int main(){
    int array[10] = {1088, 2112, 3136, 4160, 5184, 6208, 7232, 8256, 9280, 10304};
    float milliseconds;
    for(int i = 0; i < 10; i++){
        int n = array[i];
        int bytes = n * n * sizeof(float);

        float *h_a, *h_b, *h_c;
        float *d_a, *d_b, *d_c;

        h_a = (float*)malloc(bytes);
        h_b = (float*)malloc(bytes);
        h_c = (float*)malloc(bytes);
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);

        //generate the martix
        for (int i = 0; i < n * n; ++i) {
            h_a[i] = float(rand() % 255 - 127) / 127;
            h_b[i] = float(rand() % 255 - 127) / 127;
        }        

        cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

        // cudaEvent used to record the time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        //cublas handle
        cublasHandle_t handle;
        cublasCreate(&handle);

        //scalaing factors
        float alpha = 1.0f;
        float beta = 0.0f;

        //caclute c = (alpha*a)*b + (beta*c)
        //(m*k) * （k*n） = (m*n)
        //handle, cublasOperation_t transa, cublasOperation_t transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc
        cudaEventRecord(start);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_a, n, d_b, n, &beta, d_c, n);
        cudaEventRecord(stop);

        cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);


        //get the running time milliseconds
        cudaEventSynchronize(start);
        cudaEventSynchronize(stop);
        milliseconds = 0.0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        //bool verify = verify_solution(h_a, h_b, h_c, n);

        std::cout<<"n= "<<n<<std::endl;
        
        /*
        if(verify == true){
            std::cout<<"ok"<<std::endl;
        }
        else{
            std::cout<<"failed"<<std::endl;
        }
        */
        
        float second = milliseconds/1e3;
        std::cout<<"milliseconds = "<<milliseconds<<std::endl;
        std::cout<<"Time = "<<second<<std::endl;
        std::cout<<"Performance = "<<2*(n/1e3)*(n/1e3)*(n/1e3)/second<<std::endl;
        printf("\n");
        
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        free(h_a);
        free(h_b);
        free(h_c);

        cublasDestroy(handle);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
}