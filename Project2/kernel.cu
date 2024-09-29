#define CHUNK_SIZE 512
#define CHUNK_LOG 9

#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>
#include <algorithm>
#include <math.h>

// Get new array size with padding if its length is not a power of 2
int nextPowerOf2(int size) {
    if (size == 0) return 1;
    size--;
    size |= size >> 1;
    size |= size >> 2;
    size |= size >> 4;
    size |= size >> 8;
    size |= size >> 16;
    return size + 1;
}

// Pad the input array in parallel
// Sets every element between [size, newSize-1] to zero. 
__global__ void padArrayKernel(int* arrGpu, int size, int newSize) {
    int idx = min(size + blockIdx.x*CHUNK_SIZE + threadIdx.x, newSize-1);
    arrGpu[idx] = 0;  // Assign padValue to the new elements
}

// Global Memory Bitonic Sort
__global__ void BitonicSortCUDA(int* arrGpu, int size, int i, int j) {
    // Calculate thread index and map it to array index
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Calculate indices based on thread index
    int first_index = idx;
    int second_index = first_index ^ (1 << j);

    // Only compare when first index < second_index
    if (second_index > first_index && second_index < size) {
        int first_val = arrGpu[first_index];
        int second_val = arrGpu[second_index];

        // Sorting condition based on bitonic pattern
        int two_pow_i = 1 << i;
        bool ascending = (two_pow_i & first_index) == 0;
        if ((ascending && first_val > second_val) || (!ascending && first_val < second_val)) {
            arrGpu[first_index] = second_val;
            arrGpu[second_index] = first_val;
        }
    }
}

// Shared Memory Bitonic Sort w/ nested for loop
// Run this first when values of i/j are both small
__global__ void BitonicSortSharedCUDA(int* arrGpu, int size) {
    // Shared memory for faster access
    __shared__ int sharedMem[CHUNK_SIZE];
    // This thread handles element #(bid.x*CHUNK_SIZE + tid.x) of overall array
    int idx = blockIdx.x*CHUNK_SIZE + threadIdx.x;
    // Copy data to shared memory (each thread loads one element)
    sharedMem[threadIdx.x] = arrGpu[idx];
    __syncthreads();
    for (int i = 1; i < CHUNK_LOG; i++) 
    {
        // Perform the bitonic sort
        int two_pow_i = 1 << i;
        int first_index = threadIdx.x;
        bool ascending = (two_pow_i & first_index) == 0;
        for (int j = i-1; j >= 0; j--)
        {
            // Calculate indices based on thread index
            // These indices are into the shared mem array
            int second_index = first_index ^ (1 << j);

            if (second_index > first_index && second_index < blockDim.x)
            {
                int first_val = sharedMem[first_index];
                int second_val = sharedMem[second_index];

                // Sorting condition based on bitonic pattern
                if ((ascending && first_val > second_val) || (!ascending && first_val < second_val)) {
                    sharedMem[first_index] = second_val;
                    sharedMem[second_index] = first_val;
                    
                }
            }
            __syncthreads();
        }
    }
    // Copy sorted data back to global memory
    arrGpu[idx] = sharedMem[threadIdx.x];
}

// Shared memory bitonic sort with single for loop
// Run this after calling global memory bitonic sort on large values of i/j
__global__ void BitonicSortSharedCUDASingleLoop(int* arrGpu, int size, int i) {
    // Shared memory for faster access
    __shared__ int sharedMem[CHUNK_SIZE];
    // This thread handles element #(bid.x*CHUNK_SIZE + tid.x) of overall array
    int idx = blockIdx.x*CHUNK_SIZE + threadIdx.x;
    // Copy data to shared memory (each thread loads one element)
    sharedMem[threadIdx.x] = arrGpu[idx];
    __syncthreads();
    
    // Perform the bitonic sort
    int two_pow_i = 1 << i;
    int first_index = threadIdx.x;
    bool ascending = (two_pow_i & idx) == 0;
    for (int j = CHUNK_LOG - 1; j >= 0; j--)
    {
        // Calculate indices based on thread index
        // These indices are into the shared mem array
        int second_index = first_index ^ (1 << j);

        if (second_index > first_index && second_index < blockDim.x)
        {
            int first_val = sharedMem[first_index];
            int second_val = sharedMem[second_index];

            // Sorting condition based on bitonic pattern
            if ((ascending && first_val > second_val) || (!ascending && first_val < second_val)) {
                sharedMem[first_index] = second_val;
                sharedMem[second_index] = first_val;
            }
        }
        __syncthreads();   
    }
    
    // Copy sorted data back to global memory
    arrGpu[idx] = sharedMem[threadIdx.x];
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
    int* arrCpu;
    int* arrSortedGpu;

    // Changed this as allowed in Ed discussion
    // Using pinned memory for host memory arrays
    cudaMallocHost((void**)&arrCpu, size * sizeof(int));
    cudaMallocHost((void**)&arrSortedGpu, size * sizeof(int));

    for (int i = 0; i < size; i++) {
        arrCpu[i] = rand() % 1000;
    }

    float gpuTime, h2dTime, d2hTime, cpuTime = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // ======================================================================
    // Transfer data (arr_cpu) to device
    // ======================================================================
    int newSize = nextPowerOf2(size); // Size used in calculations
    int *arrGpu;
    cudaMalloc(&arrGpu, newSize << 2);
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

    // Example Computations for threads/blocks:
    // N = 10 mil
    // N-padded = 16,777,216
    // threadsPerBlock = 1024
    // blocksPerGrid = 16384
    // Padding = no remainder
    // sharedMemSize = 4096B

    // Set gpu specs for padding
    int threadsPerBlock = CHUNK_SIZE;
    int blocksPerGrid = ((newSize-size) + threadsPerBlock - 1) >> CHUNK_LOG;

    // For small arrays use 1 block and set threads to exact length
    if (newSize < CHUNK_SIZE) {
        threadsPerBlock = newSize;
        blocksPerGrid = 1;
    }

    // Pad the Array
    padArrayKernel<<<blocksPerGrid, threadsPerBlock>>>(arrGpu, size, newSize);

    // Set gpu specs for sorting
    blocksPerGrid = (newSize + threadsPerBlock - 1) >> CHUNK_LOG;

    // For small arrays we only need 1 block
    if (newSize < CHUNK_SIZE) {
        blocksPerGrid = 1;
    }

    size_t sharedMemSize = threadsPerBlock * sizeof(int);
    int log_len = 31 - __builtin_clz(newSize);

    // For i = 0 to CHUNK_LOG-1, use shared memory since all elements compared are < 1024 apart
    BitonicSortSharedCUDA<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(arrGpu, newSize);

    // For i = CHUNK_LOG to log_len, we call bitonic sort with global memory
    // when i is large (>= CHUNK_LOG) and shared memory when i is small (< CHUNK_LOG)
    for (int i = CHUNK_LOG; i <= log_len; i++) 
    {
        for (int j = i-1; j >= CHUNK_LOG; j--)
        {
            // Only use global memory when comparing elements that are 1024+ indices apart
            BitonicSortCUDA<<<blocksPerGrid, threadsPerBlock>>>(arrGpu, newSize, i, j);
        }
        // For j = CHUNK_LOG-1 to 0, use shared memory since all elements compared are < 1024 apart
        BitonicSortSharedCUDASingleLoop<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(arrGpu, newSize, i);
    }

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

    // Offset so we don't copy the padding elements
    cudaMemcpy(arrSortedGpu, arrGpu + (newSize - size), size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(arrGpu);

    // ======================================================================
    // End your code
    // ======================================================================

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d2hTime, start, stop);

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

    // Changed this as allowed in Ed discussion
    cudaFreeHost(arrCpu);
    cudaFreeHost(arrSortedGpu);

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

