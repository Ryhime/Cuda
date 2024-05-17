#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h> 

__global__ void vectorAdd(int* a, int* b, int* c){
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

void withGPU(){
    clock_t start = clock();
    int a[] = {1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3};
    int b[] = {4,5,6,4,5,6,4,5,6,4,5,6,4,5,6,4,5,6,4,5,6,4,5,6,4,5,6,4,5,6,4,5,6,4,5,6,4,5,6,4,5,6,4,5,6};
    int c[sizeof(a)/sizeof(int)] = {0};

    // Create pointers refering to the memory in the GPU
    int* cudaA = 0;
    int* cudaB = 0;
    int* cudaC = 0;

    // Allocate memory in the GPU
    cudaMalloc(&cudaA, sizeof(a));
    cudaMalloc(&cudaB, sizeof(b));
    cudaMalloc(&cudaC, sizeof(c));

    // Copy the allocated memory on the CPU to the GPU
    cudaMemcpy(cudaA, a, sizeof(a), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaB, b, sizeof(b), cudaMemcpyHostToDevice);
    // Becauce c can be zeroed out since it is the result

    // GRID_SIZE, BLOCK_SIZE
    // Look into why we are using only one grid and the difference between the two
    vectorAdd<<<1,sizeof(a)/sizeof(int)>>>(cudaA,cudaB,cudaC);


    // Bring from the GPU back to the CPU
    cudaMemcpy(c,cudaC,sizeof(c),cudaMemcpyDeviceToHost);
    for (int i=0;i<sizeof(c)/sizeof(int);i++){
        printf("%d\n",c[i]);
    }
    clock_t end = clock();
    printf("The program took %f seconds to finish with a GPU.\n",(double)(end-start)/CLOCKS_PER_SEC);
}

void withCPU(){
    clock_t start = clock();
    int a[] = {1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3};
    int b[] = {4,5,6,4,5,6,4,5,6,4,5,6,4,5,6,4,5,6,4,5,6,4,5,6,4,5,6,4,5,6,4,5,6,4,5,6,4,5,6,4,5,6,4,5,6};
    int c[sizeof(a)/sizeof(int)] = {0};

    for (int i=0;i<sizeof(a)/sizeof(int);i++){
        c[i] = a[i]+b[i];
    }

    for (int i=0;i<sizeof(a)/sizeof(int);i++){
        printf("%d\n",c[i]);
    }

    clock_t end = clock();
    printf("The program took %f seconds to finish with a CPU.\n",(double)(end-start)/CLOCKS_PER_SEC);
}


int main(){
    withGPU();
    withCPU();
    return 0;  
}