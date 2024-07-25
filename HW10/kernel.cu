
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#define max(a,b) ((a > b) ? a : b)
#define PI 3.141592

cudaError_t Conv_separableWithCuda(unsigned char* input, unsigned char* output, float* filterX, float* filterY, int kernel_x, int kernel_y, int xSize, int ySize);

void boxcarFilter(float* filter, int kernel_x, int kernel_y);
void gaussianFilter(float* filter, int kernel_x, int kernel_y);

int verify(unsigned char* input, unsigned char* output, int xSize, int ySize);

void insertpad(unsigned char* inputPtr, unsigned char* outputPtr, int xSize, int ySize, int kernel_x, int kernel_y, char* pad);

int verify(unsigned char* Goldoutput, unsigned char* Cudaoutput, int xSize, int ySize) {
	for (int i = 0; i < xSize * ySize; i++) {
		if (abs(Goldoutput[i] - Cudaoutput[i]) > 1) {
			printf("Goldoutput[%d] = %d Cudaoutput[%d]=%d \n", i, Goldoutput[i], i, Cudaoutput[i]);
			return(1);
		}
	}
	return(0);
}

void insertpad(unsigned char* inputPtr, unsigned char* outputPtr, int xSize, int ySize, int kernel_x, int kernel_y, char* pad) {
	int i, j;
	if (pad == "boundary") {
		for (i = 0; i < kernel_x / 2; i++) {
			for (j = 0; j < kernel_y / 2; j++) {
				*(outputPtr + i + j * (xSize + kernel_x / 2 * 2)) = *(inputPtr);
				*(outputPtr + (xSize + kernel_x / 2 + i) + j * (xSize + kernel_x / 2 * 2)) = *(inputPtr + xSize - 1);
				*(outputPtr + i + (ySize + kernel_y / 2 + j) * (xSize + kernel_x / 2 * 2)) = *(inputPtr + (ySize - 1) * xSize);
				*(outputPtr + (xSize + kernel_x / 2 + i) + (ySize + kernel_y / 2 + j) * (xSize + kernel_x / 2 * 2)) = *(inputPtr + xSize - 1 + (ySize - 1) * xSize);
			}
			for (j = 0; j < ySize; j++) {
				*(outputPtr + i + (j + kernel_y / 2) * (xSize + kernel_x / 2 * 2)) = *(inputPtr + j * xSize);
				*(outputPtr + xSize + kernel_x / 2 + i + (j + kernel_y / 2) * (xSize + kernel_x / 2 * 2)) = *(inputPtr + xSize - 1 + j * xSize);
			}
		}
		for (j = 0; j < kernel_y / 2; j++) {
			for (i = 0; i < xSize; i++) {
				*(outputPtr + i + kernel_x / 2 + j * (xSize + kernel_x / 2 * 2)) = *(inputPtr + i);
				*(outputPtr + i + kernel_x / 2 + (j + ySize + kernel_y / 2) * (xSize + kernel_x / 2 * 2)) = *(inputPtr + i);
			}
		}

	}
	else if (pad == "zero") {
		for (i = 0; i < kernel_x / 2; i++) {
			for (j = 0; j < kernel_y / 2; j++) {
				*(outputPtr + i + j * (xSize + kernel_x / 2 * 2)) = 0;
				*(outputPtr + (xSize + kernel_x / 2 + i) + j * (xSize + kernel_x / 2 * 2)) = 0;
				*(outputPtr + i + (ySize + kernel_y / 2 + j) * (xSize + kernel_x / 2 * 2)) = 0;
				*(outputPtr + (xSize + kernel_x / 2 + i) + (ySize + kernel_y / 2 + j) * (xSize + kernel_x / 2 * 2)) = 0;
			}
			for (j = 0; j < ySize; j++) {
				*(outputPtr + i + (j + kernel_y / 2) * (xSize + kernel_x / 2 * 2)) = 0;
				*(outputPtr + xSize + kernel_x / 2 + i + (j + kernel_y / 2) * (xSize + kernel_x / 2 * 2)) = 0;
			}
		}
		for (j = 0; j < kernel_y / 2; j++) {
			for (i = 0; i < xSize; i++) {
				*(outputPtr + i + kernel_x / 2 + j * (xSize + kernel_x / 2 * 2)) = 0;
				*(outputPtr + i + kernel_x / 2 + (j + ySize + kernel_y / 2) * (xSize + kernel_x / 2 * 2)) = 0;
			}
		}

	}
	else if (pad == "mirror") {
		for (i = 0; i < kernel_x / 2; i++) {
			for (j = 0; j < kernel_y / 2; j++) {
				*(outputPtr + i + j * (xSize + kernel_x / 2 * 2)) = *(inputPtr + kernel_x / 2 - i + (kernel_y / 2 - j) * xSize);
				*(outputPtr + (xSize + kernel_x / 2 + i) + j * (xSize + kernel_x / 2 * 2)) = *(inputPtr + (xSize - 1 - i) + (kernel_y / 2 - j) * xSize);
				*(outputPtr + i + (ySize + kernel_y / 2 + j) * (xSize + kernel_x / 2 * 2)) = *(inputPtr + kernel_x / 2 - i + (ySize - 1 - j) * xSize);
				*(outputPtr + (xSize + kernel_x / 2 + i) + (ySize + kernel_y / 2 + j) * (xSize + kernel_x / 2 * 2)) = *(inputPtr + (xSize - 1 - i) + (ySize - 1 - j) * xSize);
			}
			for (j = 0; j < ySize; j++) {
				*(outputPtr + i + (j + kernel_y / 2) * (xSize + kernel_x / 2 * 2)) = *(inputPtr + kernel_x / 2 - i + j * xSize);
				*(outputPtr + xSize + kernel_x / 2 + i + (j + kernel_y / 2) * (xSize + kernel_x / 2 * 2)) = *(inputPtr + xSize - 1 - i + j * xSize);
			}
		}
		for (j = 0; j < kernel_y / 2; j++) {
			for (i = 0; i < xSize; i++) {
				*(outputPtr + i + kernel_x / 2 + j * (xSize + kernel_x / 2 * 2)) = *(inputPtr + i + (kernel_y / 2 - j) * xSize);
				*(outputPtr + i + kernel_x / 2 + (j + ySize + kernel_y / 2) * (xSize + kernel_x / 2 * 2)) = *(inputPtr + i + (ySize - 1 - j) * xSize);
			}
		}


	}
	for (i = 0; i < xSize; i++) {
		for (j = 0; j < ySize; j++) {
			*(outputPtr + i + kernel_x / 2 + (j + kernel_y / 2) * (xSize + kernel_x / 2 * 2)) = *(inputPtr + i + j * ySize);
		}
	}

}
void boxcarFilter(float* filter, int kernel_x, int kernel_y) {
	int i;
	for (i = 0; i < kernel_x * kernel_y; i++) {
		*(filter + i) = 1.0;
	}
}

void gaussianFilter(float* filter, int kernel_x, int kernel_y) {
	int x, y;
	float sum = 0.0;
	float sigma = 1.0;
	int half_x = kernel_x / 2;
	int half_y = kernel_y / 2;
	float s = 2.0 * sigma * sigma;

	// Compute the filter values
	for (y = -half_y; y <= half_y; y++) {
		for (x = -half_x; x <= half_x; x++) {
			float r = sqrt(x * x + y * y);
			float value = (exp(-(r * r) / s)) / (PI * s);
			filter[(y + half_y) * kernel_x + (x + half_x)] = value;
			sum += value;
		}
	}

	for (y = 0; y < kernel_y; y++) {
		for (x = 0; x < kernel_x; x++) {
			filter[y * kernel_x + x] /= sum;
		}
	}
}



__global__ void kernelConv_separable(unsigned char* input, unsigned char* output, float* filterX, float* filterY, int kernel_x, int kernel_y, int xSize, int ySize)
{
	int bstartx = blockIdx.x * xSize / gridDim.x;
	int bstarty = blockIdx.y * ySize / gridDim.y;
	int tstartx = bstartx + threadIdx.x;
	int tstarty = bstarty + threadIdx.y;
	int i;

	__shared__ unsigned char s[32][32];

	if (tstartx < (xSize + kernel_x / 2 * 2) && tstarty < (ySize + kernel_y / 2 * 2)) {
		s[threadIdx.x][threadIdx.y] = input[tstartx + tstarty * (xSize + kernel_x / 2 * 2)];
	}

	__syncthreads();

	if (tstartx < xSize && tstarty < ySize) {
		float sum = 0.0;
		float sum1 = 0.0;
		for (int r = 0; r < kernel_y; r++) {
			for (int c = 0; c < kernel_x; c++) {
				sum += s[threadIdx.x + c][threadIdx.y + r] * filterX[c];
			}
			sum1 += sum * filterY[r];
			sum = 0.0;
		}
		output[tstarty * xSize + tstartx] = (unsigned char)(sum1 / kernel_x / kernel_y);
	}

}

int main()
{
	unsigned char* input, * padPtr, * CudaOutput;
	float* filter, * filterX, * filterY;
	int xSize, ySize;
	int smSize;
	int kernel_x, kernel_y;
	char* pad = "boundary"; // zero / mirror / boundary
	char* filter_name = "gaussian"; // gaussian / boxcar
	xSize = 512;
	ySize = 512;
	kernel_x = 15;
	kernel_y = 15;
	input = new unsigned char[xSize * ySize];
	padPtr = new unsigned char[(xSize + kernel_x / 2 * 2) * (ySize + kernel_y / 2 * 2)];
	CudaOutput = new unsigned char[xSize * ySize];
	filter = new float[kernel_x * kernel_y];
	filterX = new float[kernel_x];
	filterY = new float[kernel_y];
	int i, j;
	printf("pad type : %s, %s filter size : %d x %d\n", pad, filter_name, kernel_x, kernel_y);
	printf("image xSize=%d image ySize=%d \n", xSize, ySize);

	FILE* fp;

	fp = fopen("usc.raw", "rb");

	fread(input, xSize, ySize, fp);

	fclose(fp);
	insertpad(input, padPtr, xSize, ySize, kernel_x, kernel_y, pad);

	if (filter_name == "boxcar")
	{
		boxcarFilter(filter, kernel_x, kernel_y);
		boxcarFilter(filterX, kernel_x, 1);
		boxcarFilter(filterY, 1, kernel_y);
	}
	else if (filter_name == "gaussian")
	{
		gaussianFilter(filter, kernel_x, kernel_y);
		gaussianFilter(filterX, kernel_x, 1);
		gaussianFilter(filterY, 1, kernel_y);
	}

	
	cudaError_t cudaStatus = Conv_separableWithCuda(padPtr, CudaOutput, filterX, filterY, kernel_x, kernel_y, xSize, ySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Conv_separableWithCuda failed!");
		return 1;
	}


	fp = fopen("CudaOutput.raw", "wb");
	fwrite(CudaOutput, xSize, ySize, fp);
	fclose(fp);

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	delete[] CudaOutput;
	delete[] padPtr;
	delete[] input;

	return 0;
}

// Helper function for using CUDA to Median5tap_ vectors in parallel.
cudaError_t Conv_separableWithCuda(unsigned char* input, unsigned char* output, float* filterX, float* filterY, int kernel_x, int kernel_y, int xSize, int ySize)
{
	unsigned char* dev_input = 0;
	unsigned char* dev_output = 0;
	float* dev_filterX = 0;
	float* dev_filterY = 0;
	unsigned int xysize = xSize * ySize;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaError_t cudaStatus;
	cudaDeviceProp prop;

	dim3 blocks, threads;

	threads.x = 16;
	threads.y = 16;
	blocks.x = xSize / 16;
	blocks.y = ySize / 16;
	printf("blocks.x = %d blocks.y=%d \n", blocks.x, blocks.y);
	printf("threads.x = %d threads.y=%d \n", threads.x, threads.y);

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaEventRecord(start, 0);
	// Allocate GPU buffers for two input     .
	cudaStatus = cudaMalloc((void**)&dev_input, (xSize + kernel_x) * (ySize + kernel_y) * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_output, xysize * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_filterX, kernel_x * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_filterY, kernel_y * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_input, input, (xSize + kernel_x) * (ySize + kernel_y), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_filterX, filterX, kernel_x * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_filterY, filterY, kernel_y * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaProfilerStart();

	kernelConv_separable << <blocks, threads >> > (dev_input, dev_output, dev_filterX, dev_filterY, kernel_x, kernel_y, xSize, ySize);
	cudaProfilerStop();
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Conv_separableKernel!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMemcpy(output, dev_output, xysize * sizeof(char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);


	float cudaElapsedTime;
	cudaEventElapsedTime(&cudaElapsedTime, start, stop);
	printf("Time for execution = %3.6f ms \n", cudaElapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

Error:
	cudaFree(dev_input);
	cudaFree(dev_output);
	cudaFree(dev_filterX);
	cudaFree(dev_filterY);
	return cudaStatus;
}
