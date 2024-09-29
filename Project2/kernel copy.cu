#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>
#include <algorithm>
#include <math.h>

int nextPowerOf2(int size) {
    if (size == 0) return 1;
    return pow(2, ceil(log2(size)));
}

// CUDA Error Check Macro
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void BitonicSortCUDA(int* arrGpu, int size, int i, int j) {
    // Shared memory for faster access
    // extern __shared__ int sharedMem[];

    // Calculate thread index and map it to array index
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // // Copy data to shared memory (each thread loads one element)
    // if (idx < size) {
    //     sharedMem[threadIdx.x] = arrGpu[idx];
    // }
    // __syncthreads();

    // Perform the bitonic sort
    int two_pow_i = 1 << i;
    int two_pow_j = 1 << j;

    // Calculate indices based on thread index
    int first_index = idx; //threadIdx.x;
    int second_index = first_index ^ two_pow_j;

    if (second_index > first_index && second_index < size) {
        int first_val = arrGpu[first_index];
        int second_val = arrGpu[second_index];

        // Sorting condition based on bitonic pattern
        if ((two_pow_i & first_index) == 0) {
            if (first_val > second_val) {
                // Swap if out of order
                arrGpu[first_index] = second_val;
                arrGpu[second_index] = first_val;
            }
        } else {
            if (first_val < second_val) {
                // Swap if out of order
                arrGpu[first_index] = second_val;
                arrGpu[second_index] = first_val;
            }
        }
    }

    // Copy sorted data back to global memory
    // if (idx < size) {
    //     arrGpu[idx] = sharedMem[threadIdx.x];
    // }
    __syncthreads();
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <array_size>\n", argv[0]);
        return 1;
    }

    int size = atoi(argv[1]);

    srand(time(NULL));

    // ======================================================================
    // arCpu contains the input random array
    // arrSortedGpu should contain the sorted array copied from GPU to CPU
    // ======================================================================
    int* arrCpu = (int*)malloc(size * sizeof(int));
    int* arrSortedGpu = (int*)malloc(size * sizeof(int));

    for (int i = 0; i < size; i++) {
        arrCpu[i] = rand() % 1000;
    }

    // Print the original array
    // printf("Original Array:\n");
    // for (int i = 0; i < size; i++) {
    //     printf("%d ", arrCpu[i]);
    // }
    // printf("\n");

    float gpuTime, h2dTime, d2hTime, cpuTime = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // ======================================================================
    // Transfer data (arr_cpu) to device
    // ======================================================================

    int *arrGpu;
    cudaMallocManaged(&arrGpu, size * sizeof(int));
    cudaMemcpy(arrGpu, arrCpu, size * sizeof(int), cudaMemcpyHostToDevice);

    // ======================================================================
    // End your code
    // ======================================================================

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&h2dTime, start, stop);

    cudaEventRecord(start);
    
    // ======================================================================
    // Perform bitonic sort on GPU
    // ======================================================================

    int threadsPerBlock = 1024;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    // size_t sharedMemSize = threadsPerBlock * sizeof(int);

    // Launch the kernel
    // BitonicSortCUDA<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(arrGpu, size);

    int log_len = 31 - __builtin_clz(size);

    for (int i = 1; i <= log_len; i++) 
    {
        for (int j = i-1; j >= 0; j--) 
        {
            BitonicSortCUDA<<<blocksPerGrid, threadsPerBlock>>>(arrGpu, size, i, j);
        }
    }

    // BitonicSortCUDA<<<blocksPerGrid, threadsPerBlock>>>(arrGpu, size);


    // ======================================================================
    // End your code
    // ======================================================================

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);
    cudaEventRecord(start);

    // ======================================================================
    // Transfer sorted data back to host (copied to arr_sorted_gpu)
    // ======================================================================

    cudaMemcpy(arrSortedGpu, arrGpu, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(arrGpu);

    // ======================================================================
    // End your code
    // ======================================================================

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d2hTime, start, stop);

    // Print the sorted array (from GPU)
    // printf("Sorted Array (GPU):\n");
    // for (int i = 0; i < size; i++) {
    //     printf("%d ", arrSortedGpu[i]);
    // }
    // printf("\n");


    auto startTime = std::chrono::high_resolution_clock::now();
    
    // CPU sort for performance comparison
    std::sort(arrCpu, arrCpu + size);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    cpuTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    cpuTime = cpuTime / 1000;

    int match = 1;
    for (int i = 0; i < size; i++) {
        if (arrSortedGpu[i] != arrCpu[i]) {
            match = 0;
            break;
        }
    }

    free(arrCpu);
    free(arrSortedGpu);

    if (match)
        printf("\033[1;32mFUNCTIONAL SUCCESS\n\033[0m");
    else {
        printf("\033[1;31mFUNCTIONCAL FAIL\n\033[0m");
        return 0;
    }
    
    printf("\033[1;34mArray size         :\033[0m %d\n", size);
    printf("\033[1;34mCPU Sort Time (ms) :\033[0m %f\n", cpuTime);
    float gpuTotalTime = h2dTime + gpuTime + d2hTime;
    int speedup = (gpuTotalTime > cpuTime) ? (gpuTotalTime/cpuTime) : (cpuTime/gpuTotalTime);
    float meps = size / (gpuTotalTime * 0.001) / 1e6;
    printf("\033[1;34mGPU Sort Time (ms) :\033[0m %f\n", gpuTotalTime);
    printf("\033[1;34mGPU Sort Speed     :\033[0m %f million elements per second\n", meps);
    if (gpuTotalTime < cpuTime) {
        printf("\033[1;32mPERF PASSING\n\033[0m");
        printf("\033[1;34mGPU Sort is \033[1;32m %dx \033[1;34mfaster than CPU !!!\033[0m\n", speedup);
        printf("\033[1;34mH2D Transfer Time (ms):\033[0m %f\n", h2dTime);
        printf("\033[1;34mKernel Time (ms)      :\033[0m %f\n", gpuTime);
        printf("\033[1;34mD2H Transfer Time (ms):\033[0m %f\n", d2hTime);
    } else {
        printf("\033[1;31mPERF FAILING\n\033[0m");
        printf("\033[1;34mGPU Sort is \033[1;31m%dx \033[1;34mslower than CPU, optimize further!\n", speedup);
        printf("\033[1;34mH2D Transfer Time (ms):\033[0m %f\n", h2dTime);
        printf("\033[1;34mKernel Time (ms)      :\033[0m %f\n", gpuTime);
        printf("\033[1;34mD2H Transfer Time (ms):\033[0m %f\n", d2hTime);
        return 0;
    }

    return 0;
}

