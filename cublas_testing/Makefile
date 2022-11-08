default: cublas_saxpy cublas_sdot cublas_sgemm

cublas_saxpy:
	nvcc cublas_saxpy.cu -lcublas -o test/cublas_saxpy

cublas_sdot:
	nvcc cublas_sdot.cu -lcublas -o test/cublas_sdot

cublas_sgemm:
	nvcc cublas_sgemm.cu -lcublas -o test/cublas_sgemm

