#include<iostream>
#include<cuda_runtime.h>
#include<cublas_v2.h>

#include <sys/time.h>

void vector_init(float *a, int n){
    for(int i = 0; i < n; i++){
        a[i] = float(rand() % 10000)/1000;
    }
}

int main(){
    int n = 1 <<10;//1 million
    int bytes = n *sizeof(float);
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b;

    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);
    cudaMallocManaged(&d_a, bytes);
    cudaMallocManaged(&d_b, bytes);

    vector_init(h_a, n);
    vector_init(h_b, n);

    // cudaEvent used to record the time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //use for cublas
    cublasHandle_t handle;
    cublasCreate(&handle);

    //copy the vectors over to the device
    cublasSetVector(n, sizeof(float), h_a, 1, d_a, 1);
    cublasSetVector(n, sizeof(float), h_b, 1, d_b, 1);

    const float scale = 2.0f;

    //do cublas saxpy
    //saxpy y = a*x + y
    cudaEventRecord(start);
    cublasSaxpy(handle, n, &scale, d_a, 1, d_b, 1);
    cudaEventRecord(stop);

    //copy the result from device to host
    cublasGetVector(n, sizeof(float), d_b, 1, h_c, 1);

    //get the running time milliseconds
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float maxError = 0.0f;
    for (int i = 0; i < n; i++)
        maxError = fmax(maxError, fabs(h_c[i]-(scale * h_a[i] + h_b[i])));
    std::cout << "Max error: " << maxError << std::endl;
    std::cout << "Time:" << milliseconds << std::endl;
    std::cout<< "Performace: "<< (2*n) /milliseconds/1000000.0<< std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    free(h_a);
    free(h_b);
    free(h_c);

    cublasDestroy(handle);

    return 0;
}