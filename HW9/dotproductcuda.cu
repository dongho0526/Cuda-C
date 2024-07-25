
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>


const int N = 32 * 1024;
const int threadsPerBlock = 64;
const int blocksPerGrid =
(N + threadsPerBlock - 1) / threadsPerBlock;


__global__ void dot(float* a, float* b, float* c) {
	__shared__ float cache[threadsPerBlock];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;

	float   temp = 0;
	
	temp = a[tid] * b[tid];
	
	

	// set the cache values
	cache[cacheIndex] = temp;
	

	// synchronize threads in this block
	__syncthreads();

	// for reductions, threadsPerBlock must be a power of 2
	// because of the following code
	int i = blockDim.x / 2;  //i=512
	
	while (i != 0) {
		if (cacheIndex < i) {
			 cache[cacheIndex] += cache[cacheIndex + i];
		}
		__syncthreads();
		i /= 2;
	}

	if (cacheIndex == 0) {
		c[blockIdx.x] = cache[cacheIndex];
	}
}



__global__ void sum(float* a) {
	extern __shared__ float cache[];
	int cacheIndex = threadIdx.x;


	// set the cache values
	cache[cacheIndex] = a[cacheIndex];

	
	// synchronize threads in this block
	__syncthreads();

	// for reductions, threadsPerBlock must be a power of 2
	// because of the following code
	int i = blocksPerGrid / 2;
	
	while (i != 0) {
		if (cacheIndex < i) {
			cache[cacheIndex] += cache[cacheIndex + i];
		}

		__syncthreads();
		
		i /= 2;
	}

	if (cacheIndex == 0) {
		a[cacheIndex] = cache[0];
	}

	
}


int main(void) {
	float* a, * b, c, * partial_c;
	float* dev_a, * dev_b, * dev_partial_c;

	// allocate memory on the cpu side
	a = (float*)malloc(N * sizeof(float));
	b = (float*)malloc(N * sizeof(float));
	partial_c = (float*)malloc(blocksPerGrid * sizeof(float));

	// allocate the memory on the GPU
	cudaMalloc((void**)&dev_a,
		N * sizeof(float));
	cudaMalloc((void**)&dev_b,
		N * sizeof(float));
	cudaMalloc((void**)&dev_partial_c,
		blocksPerGrid * sizeof(float));

	// fill in the host memory with data
	for (int i = 0; i < N; i++) {
		a[i] = 1;
		b[i] = 1;
	}
	float temp=0.0;
	for (int i = 0; i < N; i++) {
		temp += a[i] * b[i];
	}
	printf(" SUM in C = %f \n", temp);

	// copy the arrays 'a' and 'b' to the GPU
	cudaMemcpy(dev_a, a, N * sizeof(float),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(float),
		cudaMemcpyHostToDevice);
	
	printf("N= %d \n ", N);
	printf("threadsPerBlock=%d blocksPerGrid=%d \n", threadsPerBlock, blocksPerGrid);

	dot << <blocksPerGrid, threadsPerBlock >> > (dev_a, dev_b, dev_partial_c);

	 sum << <1, blocksPerGrid, blocksPerGrid *sizeof(float) >> > (dev_partial_c);

	// copy the array 'c' back from the GPU to the CPU

	 /*
	cudaMemcpy(partial_c, dev_partial_c,
		blocksPerGrid * sizeof(float),
		cudaMemcpyDeviceToHost);
		*/
	
	cudaMemcpy(partial_c, dev_partial_c,
		sizeof(float),
		cudaMemcpyDeviceToHost);
	
	// finish up on the CPU side
	c = 0;


	//printf("Total sum c=%f \n", c);
	printf("Total sum c=%f \n", partial_c[0]);
	// free memory on the gpu side
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_partial_c);

	// free memory on the cpu side
	free(a);
	free(b);
	free(partial_c);
}
