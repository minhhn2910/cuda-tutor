#include <stdio.h>
#include <cuda_runtime.h>

/**
 * CUDA Kernel Device code
 *
 * Computes c = a + b;
 */
__global__ void add(int* a, int* b, int* c)
{
	*c = *a + *b; 
}

/**
 * Host main routine
 */
int main(void)
{
    // Allocate the host input  A
    int *h_A = (int *)malloc(sizeof(int));

    // Allocate the host input  B
    int *h_B = (int *)malloc(sizeof(int));

    // Allocate the host output C
    int *h_C = (int *)malloc(sizeof(int));

	*h_A = 1;
	*h_B = 1;
    
    // Allocate the device inputs
    int *d_A = NULL;
    cudaMalloc((void **)&d_A, sizeof(int));
    int *d_B = NULL;
    cudaMalloc((void **)&d_B, sizeof(int));
    // Allocate the device output C
    int *d_C = NULL;
    cudaMalloc((void **)&d_C, sizeof(int));
   
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    cudaMemcpy(d_A, h_A, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(int), cudaMemcpyHostToDevice);
  
    add<<<1, 1>>>(d_A, d_B, d_C );
    printf("Copy output data from the CUDA device to host memory\n");
    cudaMemcpy(h_C, d_C, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Done, result : %d\n", *h_C);
    return 0;
}

