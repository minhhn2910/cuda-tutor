#include <stdio.h>
#include <cuda_runtime.h>
#include "timer.h"
/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
	for(int j =0 ; j< 100; j++)
      	 	C[i] = A[i] + B[i] + j ;
    }
}

/**check function**/
void check_result (float* h_A,float* h_B,float* h_C,int numElements)
{
	float* temp_result = (float*)malloc(numElements*sizeof(float));
	printf("checking result \n");

	GpuTimer timer ;//= new GpuTimer();	
	timer.Start();

	for(int i = 0; i < numElements; i++)
		for(int j =0;j<100;j++)
			temp_result[i] = h_A[i] + h_B[i]+j;

	timer.Stop();
	
	printf("CPU time : %f ms\n", timer.Elapsed());

//check for differences between host and device results

	for (int i = 0; i < numElements; i++)
	{
        	if (fabs(temp_result[i]- h_C[i]) > 1e-3)
        	{
	
        	    fprintf(stderr, "Result verification failed at element %d! Where CPU=%f   GPU=%f \n", i,temp_result[i],h_C[i]);
        	    return;
        	}
    	}
    printf("Test PASSED\n");
	
}

/**
 * Host main routine
 */
int main(int argc, char** argv)
{
   // Print the vector length to be used, and compute its size
    int numElements = 50000;
    if(argc >1)
	numElements = atoi(argv[1]);

    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host input vector A
    float *h_A = (float *)malloc(size);

    // Allocate the host input vector B
    float *h_B = (float *)malloc(size);

    // Allocate the host output vector C
    float *h_C = (float *)malloc(size);

    float *ref_C = (float*) malloc(size);
    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // Allocate the device input vector A
    float *d_A = NULL;
    cudaMalloc((void **)&d_A, size);
    // Allocate the device input vector B
    float *d_B = NULL;
    cudaMalloc((void **)&d_B, size);
    // Allocate the device output vector C
    float *d_C = NULL;
    cudaMalloc((void **)&d_C, size);
    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);




    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    GpuTimer timer_kernel;

    timer_kernel.Start();

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    
    timer_kernel.Stop();
    printf("GPU time : %f ms\n", timer_kernel.Elapsed());



  
    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);



    check_result(h_A, h_B, h_C, numElements);


    // Verify that the result vector is correct
   // Reset the device and exit
    cudaDeviceReset();

    printf("Done\n");
    return 0;
}

