
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"

#include <stdio.h>
cudaError_t invert8WithCuda(unsigned char *input, unsigned char *output, int xSize, int ySize);

void invert8WithC(unsigned char *input, unsigned char *output, int xSize, int ySize);

int verify(unsigned char *input, unsigned char *output, int xSize, int ySize);



int verify(unsigned char *GoldInput, unsigned char *CudaInput, int xSize, int ySize) {
	for (int i = 0; i<xSize*ySize; i++) {
		if (GoldInput[i] != CudaInput[i]) {
			printf("GoldInput[%d] = %d CInput[%d]=%d \n", i, GoldInput[i], i, CudaInput[i]);
			return(1);
		}
	}
	return(0);
}


void invert8WithC(unsigned char *input, unsigned char *output, int xSize, int ySize)
{
    for (int i = 0; i<xSize*ySize; i++) {
		output[i] = 255 - input[i];
	}
}


__global__ void invert8WithCudaSimple(unsigned char* input, unsigned char* output, int xSize, int ySize)
{
	int idx = threadIdx.x;
	int startindex, endindex;

	
	startindex = idx*xSize/2;
	endindex = startindex + xSize/2;  
	

	// printf("idx=%d \t ", idx);
	for (int i = startindex; i < endindex; i++) {
		output[i] = 255 - input[i];
	}
}

__global__ void kernelInvert8_1024(unsigned char *input, unsigned char *output, int xSize, int ySize, int size)
{
	int xWidth = xSize; // bockDim.x = 512, gridDim.x = 1
	int xLoc = (blockIdx.x * blockDim.x + threadIdx.x); // threadIx : 0 to 511
	int yLoc = blockIdx.y*blockDim.y + threadIdx.y; // // blockIdy= 0 to 511, blockDim.y = 1, 


	int index = xLoc + yLoc*xWidth;
	
	//printf("blockIdx =%d \n", blockIdx.x);
	output[index] = 255 - input[index];
	output[index + 512 ] = 255 - input[index + 512];
	output[index +  512*1024] = 255 - input[index  +512 * 1024];
	output[index + 512 + 512 * 1024] = 255 - input[index + 512 + 512 * 1024];
	
}

__global__ void kernelInvert8(unsigned char* input, unsigned char* output, int xSize, int ySize, int size)
{
	int xWidth = xSize; // bockDim.x = 512, gridDim.x = 1
	int xLoc = (blockIdx.x * blockDim.x + threadIdx.x); // threadIx : 0 to 511
	int yLoc = blockIdx.y * blockDim.y + threadIdx.y; // // blockIdy= 0 to 511, blockDim.y = 1, 


	int index = xLoc + yLoc * xWidth;

	if(index < size)
		output[index] = 255 - input[index];

}

int main()
{
	unsigned char *input, *CudaOutput, *GoldOutput;
	int xSize, ySize;

	xSize = 512;
	ySize = 512;
	input = new unsigned char[xSize*ySize];
	CudaOutput = new unsigned char[xSize*ySize];
	GoldOutput = new unsigned char[xSize*ySize];
	int i, j;
	printf("xSize=%d ySize=%d \n", xSize, ySize);

	FILE *fp;

	fp = fopen("usc.raw", "rb");

	fread(input, xSize, ySize, fp);
	
			
	invert8WithC(input, GoldOutput, xSize, ySize);
	// Add vectors in parallel.
	cudaError_t cudaStatus = invert8WithCuda(input, CudaOutput, xSize, ySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "invert8WithCuda failed!");
		return 1;
	}

	int error = verify(GoldOutput, CudaOutput, xSize, ySize);

	if (error != 0)
		printf("Verify Failed \n");
	else
		printf("Verify Successful \n");

	fp = fopen("COutput.raw", "wb");
	fwrite(GoldOutput, xSize, ySize, fp);
	fclose(fp);

	fp = fopen("CudaOutput.raw", "w");
	fwrite(CudaOutput, xSize, ySize, fp);
	fclose(fp);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	delete[] CudaOutput;
	delete[] GoldOutput;
	delete[] input;

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t invert8WithCuda(unsigned char *input, unsigned char *output, int xSize, int ySize)
{
	unsigned char *dev_input = 0;
	unsigned char *dev_output = 0;

	//	cudaProfilerInitialize();
	unsigned int xysize = xSize*ySize;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.

	cudaDeviceProp prop;
	int count;

	dim3 blocks, threads;

	// Block size = 512x1
	threads.x = 32; // BlockDim.x 
	threads.y = 32; //BlockDim. y
	threads.z = 1;

	blocks.x = (xSize + threads.x -1) / threads.x;
	blocks.y = (ySize + threads.y - 1) / threads.y;

	///blocks.x = xSize/threads.x; // blocks.x = 32
	//blocks.y = ySize/threads.y; // blocks.y = 32
	blocks.z = 1;


	//printf(" Number of threads launched = %d \n", blocks.x * blocks.y * threads.x * threads.y);
	//printf(" Number of pixels  = %d \n", xSize*ySize);

	//
	//// threads launched : threads.x*threads.y*block.x*blocky
	//// 
	////512x512 : along X 512/8 = 64 thread blocks Alon gY 64 blocks
	//
	//// blocks.x = (xSize + threads.x - 1) / (threads.x); //1  # of blockalongx
	//// blocks.y = (ySize + threads.y - 1) / (threads.y); //512
	//printf("blocks.x = %d blocks.y=%d \n", blocks.x, blocks.y);
	//printf("threads.x = %d threads.y=%d \n", threads.x, threads.y);



	cudaGetDeviceCount(&count);
	printf("Count =  %d\n", count);

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	
	// Allocate GPU buffers for two input     .
	

	
	cudaEventRecord(start, 0);

	cudaStatus = cudaMalloc((void**)&dev_input, xysize * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_output, xysize * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_input, input, xysize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	
	
	// Launch a kernel on the GPU with one thread for each element.
	//invert8WithCudaSimple << <1, threads>> >(dev_input, dev_output, xSize, ySize);

	kernelInvert8 << <blocks, threads >> > (dev_input, dev_output, xSize, ySize, xSize*ySize);


	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching invert8Kernel!\n", cudaStatus);
		goto Error;
	}
	
	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(output, dev_output, xysize * sizeof(char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float cudaElapsedTime;
	cudaEventElapsedTime(&cudaElapsedTime, start, stop);
	printf("Time for execution = %3.1f ms \n", cudaElapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

Error:
	cudaFree(dev_input);
	cudaFree(dev_output);

	return cudaStatus;
}


