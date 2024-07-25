
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "timer.h"
#include "Timer1.h"


#define PRINTC_CUDA_ATOMIC_SHARED

#define XSIZE 1024
#define YSIZE 1024
#define HISTOBINS 8
cudaError_t histoWithCuda(unsigned int* c, unsigned int* a, int xSize, int ySize);
cudaError_t histoWithCudaAtomic(unsigned int* c, unsigned int* a, int xSize, int ySize);
cudaError_t histoSharedMem(unsigned int* hIn, unsigned int* hOut, int xSize, int ySize);

void histoWithC(unsigned int* c, unsigned int* histo, int xSize, int ySize)
{
	int temp;
	for (int i = 0; i < ySize; i++)
		for (int j = 0; j < xSize; j++) {
			temp = c[i * xSize + j] % HISTOBINS;
			histo[temp]++;
		}
}


__global__ void histoKernel(unsigned int* c, unsigned int* a, int size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int index = j * gridDim.x * blockDim.x + i;
	int temp;
	if (index < size) {
		temp = c[index] % HISTOBINS;
		a[temp] = a[temp] + 1;
	}

}

__global__ void histoAtomicKernel(unsigned int* c, unsigned int* a, int size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int index = j * gridDim.x * blockDim.x + i;
	int temp;
	if (index < size) {
		temp = c[index] % HISTOBINS;
		atomicAdd(&a[temp], 1);
	}

}


__global__ void histoAccm(unsigned int* c, unsigned int* a, int size)
{
	int j = threadIdx.x;
	for (int i = j; i < size; i+=HISTOBINS) {
		a[j] += c[i];
	}
}

//************************************ HW11 **************************************************
__global__ void histoSharedKernel(unsigned int* c, unsigned int* a, int xSize, int ySize)
{
	// Shared memory allocation
	__shared__ unsigned int sharedHist[HISTOBINS];

	// Initialize shared memory histogram bins to 0
	int tid = threadIdx.x + threadIdx.y * blockDim.x;
	if (tid < HISTOBINS) {
		sharedHist[tid] = 0;
	}
	__syncthreads();

	// Calculate global index
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	if (i < xSize && j < ySize) {
		int index = j * xSize + i;
		int temp = c[index] % HISTOBINS;
		atomicAdd(&sharedHist[temp], 1);
	}
	__syncthreads();

	// Add shared histogram bins to global memory histogram
	if (tid < HISTOBINS) {
		atomicAdd(&a[tid], sharedHist[tid]);
	}
}


int verify(unsigned int* sOut, unsigned int* cudaOut) {

	for (int i = 0; i < HISTOBINS; i++) {
		if (sOut[i] != cudaOut[i]) {
			fprintf(stderr, "verify failed! cOut[%d]=%d cudaOut[%d]=%d \n", i, sOut[i], i, cudaOut[i]);
			return 0;
		}
	}
	return 1;
}


int main()
{
	int xSize, ySize;
	xSize = XSIZE; ySize = YSIZE;
	unsigned int* histoCTable, * histoCudaTableAtomic, * histoCudaTable;
	unsigned int* histoCudaSharedTableSync, * histoCudaSharedMemBlock;
	unsigned int* in;
	in = new unsigned int[xSize * ySize];
	histoCTable = new unsigned int[HISTOBINS];
	histoCudaTable = new unsigned int[HISTOBINS];
	histoCudaTableAtomic = new unsigned int[HISTOBINS];
	histoCudaSharedMemBlock = new unsigned int[HISTOBINS];
	int histoBins = HISTOBINS;

	float dCpuTime;
	int loopCount;
	CPerfCounter counter;

	for (int i = 0; i < ySize; i++)
		for (int j = 0; j < xSize; j++)
			in[i * xSize + j] = (i * xSize + j) % 256;

	for (int i = 0; i < HISTOBINS; i++) {
		histoCTable[i] = 0;
		histoCudaTable[i] = 0;
		histoCudaTableAtomic[i] = 0;
		histoCudaSharedMemBlock[i] = 0;
	}

	dCpuTime = 0.0f;
	int verifyResult;

	dCpuTime = 0.0f;
	for (loopCount = 0; loopCount < 100; loopCount++) {
		for (int i = 0; i < HISTOBINS; i++) {
			histoCTable[i] = 0;
		}
		counter.Reset();
		counter.Start();
		histoWithC(in, histoCTable, xSize, ySize);
		counter.Stop();
		dCpuTime += counter.GetElapsedTime();
		
	}

	//dCpuTime = counter.GetElapsedTime()/(double)loopCount;
	printf("Histo C Performance (ms) = %f \n", dCpuTime / (double)loopCount * 1000.0);
	

	// square vectors in parallel.

	cudaError_t cudaStatus = histoWithCuda(in, histoCudaTable, xSize, ySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaHistogram failed! \n");
		return 1;
	}
	verifyResult = verify(histoCTable, histoCudaTable);
	if (verifyResult == 0) {
		fprintf(stderr, "Verify histoCudaTable Failed \n");
	}
	else {
		fprintf(stderr, "Verify histoCudaTable Successfule \n");
	}


	cudaStatus = histoWithCudaAtomic(in, histoCudaTableAtomic, xSize, ySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaHistogram failed! \n");
		return 1;
	}
	verifyResult = verify(histoCTable, histoCudaTableAtomic);
	if (verifyResult == 0) {
		fprintf(stderr, "Verify histoCudaTableAtomic Failed \n");
	}
	else {
		fprintf(stderr, "Verify histoCudaTableAtomic Successfule \n");
	}


	cudaStatus = histoSharedMem(in, histoCudaSharedMemBlock, xSize, ySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSharedMemHistogram failed! \n");
		return 1;
	}

	verifyResult = verify(histoCTable, histoCudaSharedMemBlock);
	if (verifyResult == 0) {
		fprintf(stderr, "Verify histoSharedMem Failed \n");
	}
	else {
		fprintf(stderr, "Verify histoSharedMem Successfule \n");
	}


	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed! \n");
		return 1;
	}



	printf("index\t\tCTable\t\tCudaTable\tCudaTableAtomic\tCudaSharedTable\n");
	for (int i = 0; i < HISTOBINS; i++) {
		printf("%d\t\t%d\t\t%d\t\t\t%d\t\t%d\t\t%d \n", i, histoCTable[i], histoCudaTable[i], histoCudaTableAtomic[i], histoCudaSharedTableSync[i], histoCudaSharedMemBlock[i]);
	}

	delete[] histoCudaTable;
	delete[] histoCTable;
	delete[] in;

	return 0;
}

//  function for using CUDA to square vectors in parallel.
cudaError_t histoWithCuda(unsigned int* hIn, unsigned int* hOut, int xSize, int ySize)
{
	unsigned int* dev_in = 0;
	unsigned int* dev_out = 0;
	GpuTimer timer;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaEventRecord(start, 0); // To measure performance

	cudaStatus = cudaMalloc((void**)&dev_in, xSize * ySize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_out, HISTOBINS * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaMemset((void*)&dev_out, 0, HISTOBINS * sizeof(int));

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_in, hIn, xSize * ySize * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	dim3 numThreads(32, 32);
	dim3 numBlocks((xSize + numThreads.x - 1) / numThreads.x, (ySize + numThreads.y - 1) / numThreads.y);

	timer.Start();
	histoKernel << <numBlocks, numThreads >> > (dev_in, dev_out, xSize * ySize);
	cudaStatus = cudaDeviceSynchronize();
	timer.Stop();
	printf("With histoKernel Time  elapsed=%g ms\n", timer.Elapsed());

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(hOut, dev_out, HISTOBINS * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	//printf("Time elapsed=%g ms\n", timer.Elapsed());

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float cudaElapsedTime;
	//cudaEventElapsedTime(&cudaElapsedTime, start, stop);
	//printf("Time for execution = %3.1f ms \n", cudaElapsedTime); 
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

Error:
	cudaFree(dev_in);
	cudaFree(dev_out);

	return cudaStatus;
}

//  function for using CUDA to square vectors in parallel.
cudaError_t histoWithCudaAtomic(unsigned int* hIn, unsigned int* hOut, int xSize, int ySize)
{
	unsigned int* dev_in = 0;
	unsigned int* dev_out = 0;
	GpuTimer timer;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaEventRecord(start, 0); // To measure performance

	cudaStatus = cudaMalloc((void**)&dev_in, xSize * ySize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_out, HISTOBINS * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaMemset((void*)&dev_out, 0, HISTOBINS * sizeof(int));

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_in, hIn, xSize * ySize * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	dim3 numThreads(32, 32);
	dim3 numBlocks((xSize + numThreads.x - 1) / numThreads.x, (ySize + numThreads.y - 1) / numThreads.y);

	timer.Start();
	histoAtomicKernel << <numBlocks, numThreads >> > (dev_in, dev_out, xSize * ySize);
	cudaStatus = cudaDeviceSynchronize();
	timer.Stop();
	printf("With histoAtomicKernel Time  elapsed=%g ms\n", timer.Elapsed());


	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(hOut, dev_out, HISTOBINS * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	//printf("Time elapsed=%g ms\n", timer.Elapsed());

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float cudaElapsedTime;
	//cudaEventElapsedTime(&cudaElapsedTime, start, stop);
	//printf("Time for execution = %3.1f ms \n", cudaElapsedTime); 
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

Error:
	cudaFree(dev_in);
	cudaFree(dev_out);

	return cudaStatus;
}


cudaError_t histoSharedMem(unsigned int* hIn, unsigned int* hOut, int xSize, int ySize)
{
	//__shared__ sharedMem[];
	unsigned int* dev_in = 0;
	unsigned int* dev_out = 0;
	unsigned int* dev_histoBlock_out = 0;
	GpuTimer timer;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}


	dim3 numThreads(32, 32);
	dim3 numBlocks((xSize + numThreads.x - 1) / numThreads.x, (ySize + numThreads.y - 1) / numThreads.y);

	cudaEventRecord(start, 0); // To measure performance

	//To store intermediate results from blocks
	cudaStatus = cudaMalloc((void**)&dev_histoBlock_out, numBlocks.x * numBlocks.y* HISTOBINS * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaMemset((void*)&dev_histoBlock_out, 0, HISTOBINS * numBlocks.x * numBlocks.y* sizeof(int));

	cudaStatus = cudaMalloc((void**)&dev_in, xSize * ySize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_out, HISTOBINS * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaMemset((void*)&dev_out, 0, HISTOBINS * sizeof(int));

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_in, hIn, xSize * ySize * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	timer.Start();
	histoSharedKernel << <numBlocks, numThreads >> > (dev_in, dev_histoBlock_out, xSize, ySize);
	histoAccm<< <1, HISTOBINS >> > (dev_histoBlock_out, dev_out, numBlocks.x * numBlocks.y* HISTOBINS);
	cudaStatus = cudaDeviceSynchronize();
	timer.Stop();
	printf("With histoSharedKernel  Time  elapsed=%g ms\n", timer.Elapsed());

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(hOut, dev_out, HISTOBINS * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	//printf("Time elapsed=%g ms\n", timer.Elapsed());

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);


	cudaEventDestroy(start);
	cudaEventDestroy(stop);

Error:
	cudaFree(dev_in);
	cudaFree(dev_out);

	return cudaStatus;
}
