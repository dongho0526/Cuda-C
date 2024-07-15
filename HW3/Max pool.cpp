#include "stdafx.h"

#include <stdio.h>
#include <stdlib.h>
#include "Timer.h"

#include <emmintrin.h>
#include <intrin.h>


#define min(a,b) ((a < b) ? a : b)
#define max(a,b) ((a > b) ? a : b)

#define MAX_FILTER_WIDTH 32
#define MAX_FILTER_HEIGHT 32
void maxpool(unsigned char* inputPtr, unsigned char* outputMaxpoolptr, int kernel_width, int kernel_height, int xSize, int ySize);
void maxpoolSSE(unsigned char* inputPtr, unsigned char* outputMaxpoolptr, int kernel_width, int kernel_height, int xSize, int ySize);
void CacheFlush(__m128i* src, unsigned int countVect) {
	int j;
	for (j = 0; j < countVect; j++) {
		_mm_clflush(src + j);
	}
}


int _tmain(int argc, _TCHAR* argv[])
{
	FILE* input_fp, * output_fp;
	unsigned char* inputPtr;					//Input Image
	unsigned char* outputMaxpoolPtr, *outputSSEPtr;				//Output Image
	float* filterPtr;
	int xSize, ySize;
	int kernel_width, kernel_height;
	int firTapSize;
	int i;

	int scale;
	int j;

	int buffer_size;

	float dCpuTime;
	int loopCount;
	char temp3;

	CPerfCounter counter;

	xSize = 512;
	ySize = 512;
	kernel_width = 2;
	kernel_height = 2;

	buffer_size = xSize * ySize * sizeof(char);
	printf("buffer_size : %d \n", buffer_size);

	inputPtr = new unsigned char[buffer_size];
	outputMaxpoolPtr = new unsigned char[buffer_size];
	outputSSEPtr = new unsigned char[buffer_size];

	/*for (i = 0; i < kernel_height; i++) {
		for (j = 0; j < kernel_width; j++) {
			*(filterPtr + i * kernel_width + j) = 1.0;
		}
	}*/

	scale = kernel_width * kernel_height;


	/*************************************************************************************
	* Read the input image
	*************************************************************************************/

	int err = fopen_s(&input_fp, "usc.raw", "rb");
	if (err != 0) {
		printf("Error: Input file can not be opened\n");
		exit(-1);
	}

	if (fread(inputPtr, xSize, ySize, input_fp) == 0) {
		printf("Error: Input file can not be read\n");
		exit(-1);
	}


	fclose(input_fp);

	/*****************************************************
	* Call generic C invert8
	*****************************************************/

	counter.Reset();
	counter.Start();
	for (loopCount = 0; loopCount < 100; loopCount++) {
		CacheFlush((__m128i*)inputPtr, buffer_size / 16);
		maxpool(inputPtr, outputMaxpoolPtr, kernel_width, kernel_height, xSize, ySize);
	}
	counter.Stop();
	dCpuTime = counter.GetElapsedTime() / (double)loopCount;
	printf("C MaxPool Performance (ms) = %f \n", dCpuTime * 1000.0);

	err = fopen_s(&output_fp, "MaxPool_outfile.raw", "wb");
	if (err != 0) {
		printf("Error: output file can not be opened\n");
		exit(-1);
	}


	if (fwrite(outputMaxpoolPtr, xSize/kernel_width, ySize/kernel_height, output_fp) == 0)
	{
		printf("file write error: MaxPool_outfile.raw\n");
		exit(-1);
	}/* fi */

	fclose(output_fp);
	//////////////////SSE STRAT
	counter.Reset();
	counter.Start();
	for (loopCount = 0; loopCount < 100; loopCount++) {
		CacheFlush((__m128i*)inputPtr, buffer_size / 16);
		maxpoolSSE(inputPtr, outputSSEPtr, kernel_width, kernel_height, xSize, ySize);
	}
	counter.Stop();
	dCpuTime = counter.GetElapsedTime() / (double)loopCount;
	printf("SSE MaxPool Performance (ms) = %f \n", dCpuTime * 1000.0);

	err = fopen_s(&output_fp, "MaxPoolSSE.raw", "wb");
	if (err != 0) {
		printf("Error: output file can not be opened\n");
		exit(-1);
	}


	if (fwrite(outputSSEPtr, xSize / kernel_width, ySize / kernel_height, output_fp) == 0)
	{
		printf("file write error: MaxPoolSSE.raw\n");
		exit(-1);
	}/* fi */

	fclose(output_fp);

	/* free the allocated memories */
	delete[] inputPtr;
	delete[] outputMaxpoolPtr;
	delete[] outputSSEPtr;

	return 0;
}

void maxpoolSSE(unsigned char* inputPtr, unsigned char* outputMaxpoolptr, int kernel_width, int kernel_height, int xSize, int ySize) {
	int i, j;
	__m128i* src128_ptr, * dst128_ptr;
	__m128i zero, temp0, temp1, temp01, temp11, temp2, temp21, temp3, temp31, temp4, temp41, temp5, temp51, temp6, data;
	__m128i maskoe = _mm_set_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
	__m128i maskep = _mm_set_epi8(15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0);
	src128_ptr = (__m128i*)(inputPtr);
	dst128_ptr = (__m128i*)(outputMaxpoolptr);
	for (i = 0; i < ySize; i += 2) {
		for (j = 0; j < xSize / 16; j += 2) {
			temp0 = *(src128_ptr + i * xSize / 16 + j);
			temp1 = *(src128_ptr + (i + 1) * xSize / 16 + j);

			temp01 = *(src128_ptr + i * xSize / 16 + j + 1);
			temp11 = *(src128_ptr + (i + 1) * xSize / 16 + j + 1);

			temp2 = _mm_max_epi8(temp0, temp1);
			temp21 = _mm_max_epi8(temp01, temp11);

			temp3 = _mm_shuffle_epi8(temp2, maskoe);
			temp31 = _mm_shuffle_epi8(temp21, maskoe);

			temp4 = _mm_max_epi8(temp2, temp3);
			temp41 = _mm_max_epi8(temp21, temp31);

			temp5 = _mm_shuffle_epi8(temp4, maskep);
			temp51 = _mm_shuffle_epi8(temp41, maskep);

			temp6 = _mm_unpacklo_epi8(temp5, temp51);
			data = _mm_shuffle_epi8(temp6, maskep);
			*(dst128_ptr + i / 2 * xSize / 32 + j / 2) = data;

		}

	}
}

void maxpool(unsigned char* inputPtr, unsigned char* outputMaxpoolptr, int kernel_width, int kernel_height, int xSize, int ySize) {
	int i, j;
	for (i = 0; i < (xSize / kernel_width); i++) {
		for (j = 0; j < (ySize / kernel_height); j++) {
			*(outputMaxpoolptr + j * xSize / kernel_width + i) = max(max(*(inputPtr + 2 * i + (2 * j) * xSize), *(inputPtr + 2 * i - 1 + (2 * j) * xSize)), max(*(inputPtr + 2 * i - 1 + (2 * j - 1) * xSize), *(inputPtr + 2 * i + (2 * j - 1) * xSize)));
		}
	}
}


