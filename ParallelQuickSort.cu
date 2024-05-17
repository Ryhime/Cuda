#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
int partition(int* arr, int startIndex, int endIndex){
    int toSwitchToIndex = startIndex;
    int pivotIndex = endIndex;
    for (int i=0;i<endIndex-startIndex;i++){
        if (arr[])
    }
}

int main(){
    // Create the array
    int a[] = {5,78,1,5,7,34,78,4,7,4,2};
    partition(a,0,sizeof(a)/sizeof(int)-1);


    // Show Results
    for (int i=0;i<sizeof(a)/sizeof(int);i++){
        printf("%d ",a[i]);
    }
}