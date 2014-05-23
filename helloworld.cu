#include <stdio.h>
#include <cuda_runtime.h>

/**
 * CUDA Kernel Device code
 * done nothing
 */
__global__ void kernel(void)	{}

/**
 * Host main routine
 */
int main(void)
{
   kernel<<<1,1>>>();
   printf("Hello World\n");
    return 0;
}

