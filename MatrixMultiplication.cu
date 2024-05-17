#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void multiply(int* a,int* b,int sharedDim,int bCols,int* c){
    // Row
    int row = blockIdx.x;
    // Col
    int col = threadIdx.x;
    int sm = 0;
    for (int i=0;i<sharedDim;i++){
        sm+=a[row*sharedDim+i]*b[i*bCols+col];
    }
    c[row*bCols+col] = sm;
}
int main(){
    int aRows = 1;
    int aColsBRows = 3;
    int bCols = 2;

    // Define the matrices
    int a[aRows][aColsBRows] = {{1,2,3}};
    int b[aColsBRows][bCols] = {{1,2},{1,2},{1,1}};
    int c[aRows][bCols] = {0};

    // Create memory on the GPU for them
    int* gpuA = 0;
    int* gpuB = 0;
    int* gpuC = 0;

    cudaMalloc(&gpuA,sizeof(a));
    cudaMalloc(&gpuB,sizeof(b));
    cudaMalloc(&gpuC,sizeof(c));

    // Copy memory to GPU
    cudaMemcpy(gpuA,a,sizeof(a),cudaMemcpyHostToDevice);
    cudaMemcpy(gpuB,b,sizeof(b),cudaMemcpyHostToDevice);

    multiply<<<aRows,bCols>>>(gpuA,gpuB,aColsBRows,bCols,gpuC);

    // Get C from GPU
    cudaMemcpy(c,gpuC,sizeof(c),cudaMemcpyDeviceToHost);

    for (int i=0;i<aRows;i++){
        printf("|");
        for (int k=0;k<bCols;k++){
            printf("%d",c[i][k]);
            if (k<bCols-1) printf(" ");
        }
        printf("|\n");
    }
}