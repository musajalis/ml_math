#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <sys/time.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d code=%d\n", __FILE__, __LINE__, err); \
        exit(1); \
    } \
}

__global__ void sample_init(float* arr, int size, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed + idx, 0, 0, &state);
        arr[idx] = curand_uniform(&state) * 10.0f;
    }
}

__global__ void matrix_mult(float* mat1, int mat1_width,
                            float* mat2, int mat2_width,
                            float* result, int mat1_height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < mat1_height && col < mat2_width) {
        float sum = 0.0f;
        for (int k = 0; k < mat1_width; k++) {
            sum += mat1[row * mat1_width + k] * mat2[k * mat2_width + col];
        }
        result[row * mat2_width + col] = sum;
    }
}

float* cuda_matrix_generator(int width, int height, bool sample) {
    int size = width * height;
    float* h_arr = (float*)malloc(size * sizeof(float));
    if (!h_arr) return NULL;

    float* d_arr;
    CHECK_CUDA(cudaMalloc(&d_arr, size * sizeof(float)));

    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);

    if (sample) {
        sample_init<<<grid, block>>>(d_arr, size, time(NULL));
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    } else {
        CHECK_CUDA(cudaMemset(d_arr, 0, size * sizeof(float)));
    }

    CHECK_CUDA(cudaMemcpy(h_arr, d_arr, size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_arr));

    return h_arr;
}

float* cuda_matrix_mult(float* h_A, float* h_B, int m, int k, int n) {
    float *d_A, *d_B, *d_C;
    size_t size_A = m * k * sizeof(float);
    size_t size_B = k * n * sizeof(float);
    size_t size_C = m * n * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k, &alpha,
                d_B, n,
                d_A, k,
                &beta, d_C, n);

    float* h_C = (float*)malloc(size_C);
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return h_C;
}

void print_matrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main() {
    int mat1_w = 48000, mat1_h = 23000;
    int mat2_w = 26000, mat2_h = 48000;

    float* mat1 = cuda_matrix_generator(mat1_w, mat1_h, true);
    float* mat2 = cuda_matrix_generator(mat2_w, mat2_h, true);

    if (!mat1 || !mat2) {
        printf("Matrix generation failed!\n");
        free(mat1); free(mat2);
        return 1;
    }

    struct timeval start, end;
    gettimeofday(&start, NULL);

    float* result = cuda_matrix_mult(mat1, mat2, mat1_h, mat1_w, mat2_w);

    gettimeofday(&end, NULL);
    double elapsed = (end.tv_sec - start.tv_sec) +
                    (end.tv_usec - start.tv_usec) / 1000000.0;

    if (!result) {
        printf("\nMultiplication failed!\n");
    } else {
        printf("\nOperation completed in: %.3f seconds\n", elapsed);
    }

    free(mat1); free(mat2); free(result);
    return 0;
}
