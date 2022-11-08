#include<iostream>
#include<cuda_runtime.h>
#include<cublas_v2.h>

#include <sys/time.h>

void vector_init(float *a, int n){
    for(int i = 0; i < n; i++){
        a[i] = float(rand() % 255-127)/127;
    }
}

int main(){
    int array[10] = {1088, 2112, 3136, 4160, 5184, 6208, 7232, 8256, 9280, 10304};
    for(int i = 0; i< 10; i++){
        int n = array[i];
        int bytes = n *sizeof(float);
        float *h_a, *h_b, *h_c;
        float *d_a, *d_b;
        float result = 0.0;
        float time = 0;

        h_a = (float*)malloc(bytes);
        h_b = (float*)malloc(bytes);
        h_c = (float*)malloc(bytes);
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        float milliseconds;

        // cudaEvent used to record the time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        //use for cublas
        cublasHandle_t handle;
        cublasCreate(&handle);

        float diff = 0.0f;
        for(int i = 0; i<100;i++){
            vector_init(h_a, n);
            vector_init(h_b, n);

            //copy the vectors over to the device
            cublasSetVector(n, sizeof(float), h_a, 1, d_a, 1);
            cublasSetVector(n, sizeof(float), h_b, 1, d_b, 1);

            //do cublas sdot
            cudaEventRecord(start);
            cublasSdot(handle,n,d_a,1,d_b,1,&result);
            cudaEventRecord(stop);

            //copy the result from device to host
            cublasGetVector(n, sizeof(float), d_b, 1, h_c, 1);

            //get the running time milliseconds
            cudaEventSynchronize(stop);
            milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            time += milliseconds;

            float sum = 0.0f;

            for(int i = 0; i < n; i++){
                sum += h_a[i]*h_b[i]; 
            }
            diff += fabs(sum - result);
        }
        std::cout<<"n= "<<n<<std::endl;
        std::cout<<"Diff= "<<diff/100<<std::endl;
        std::cout<<"Performance= "<<2*n*100/(time*1e6)<<std::endl;

        cudaFree(d_a);
        cudaFree(d_b);
        free(h_a);
        free(h_b);
        free(h_c);

        cublasDestroy(handle);
    }
    return 0;
}