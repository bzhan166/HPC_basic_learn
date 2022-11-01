#include<iostream>
#include<cuda_runtime.h>
#include<cublas_v2.h>

#include <sys/time.h>

void vector_init(float *a, int n){
    for(int i = 0; i < n; i++){
        a[i] = float(rand() % 1000)/1000;
    }
}

int main(){
    int n = 1 <<20;//1048576
    int bytes = n *sizeof(float);
    float *h_a, *h_b;
    float *d_a, *d_b;
    float result = 0.0;

    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);

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


    //do cublas sdot
    cudaEventRecord(start);
    cublasSdot(handle,n,d_a,1,d_b,1,&result);
    cudaEventRecord(stop);

    //get the running time milliseconds
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float diff = 0.0f;
    float sum = 0.0f;

    for(int i = 0; i < n; i++){
        sum += h_a[i]*h_b[i]; 
    }
    diff = fabs(sum - result);
        
    
    std::cout<<"h_a[1]= "<<h_a[0]<<" "<<"h_b[1]= "<<h_b[0]<<" "<<"result= "<<result<<" "<<std::endl;
    std::cout << "Diff: " << diff << std::endl;
    std::cout << "Time:" << milliseconds << std::endl;
    std::cout<< "Performace: "<< (2*n) /milliseconds/1000000.0<< std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    free(h_a);
    free(h_b);

    cublasDestroy(handle);

    return 0;
}