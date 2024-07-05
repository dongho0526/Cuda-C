// verticalMedianFilter5TapSSE.cpp : This file contains the 'main' function. Program execution begins and ends there.
//


#include <iostream>
#include <fstream>
#include <fstream>
#include <stdio.h>
#include <emmintrin.h>
#include "Timer.h"

#include <math.h>
#include <cmath>

#define min(a,b) ((a < b) ? a : b)
#define max(a,b) ((a > b) ? a : b)

#define MAX_FILTER_WIDTH 32
#define MAX_FILTER_HEIGHT 32

// Defined Function
// cMedian3	1
// cMedian5	2
#define	USEDFUNCTION 2

void sseMedian3(unsigned char* __restrict inputPtr, unsigned char* outputPtr, int xSize, int ySize);
void sseMedian5(unsigned char* __restrict inputPtr, unsigned char* outputPtr, int xSize, int ySize);
void cMedian3(unsigned char* __restrict inputPtr, unsigned char* outputCPtr, int xSize, int ySize);
void cMedian5(unsigned char* __restrict inputPtr, unsigned char* outputCPtr, int xSize, int ySize);
int verify(unsigned char* cInput, unsigned char* SSEOutput, int xSize, int ySize);

void avg_pooling_sse(unsigned char* input, unsigned char* output, int width, int height);
void max_pooling_sse(unsigned char* input, unsigned char* output, int width, int height);


void CacheFlush(__m128i* src, unsigned int countVect) {
	int j;
	for (j = 0; j < countVect; j++) {
		_mm_clflush(src + j);
	}
}

int verify(unsigned char* cInput, unsigned char* SSEOutput, int xSize, int ySize);

int main()

{

	FILE* input_fp, * output_fp;
	unsigned char* inputPtr, * outputPtr, * outputCPtr;				//Input Image
	unsigned char* outputPtr_avg, * outputPtr_max;
	int xSize, ySize;
	int buffer_size;
	float dCpuTime;
	int loopCount;
	CPerfCounter counter;

	xSize = 512;
	ySize = 512;

	buffer_size = xSize * ySize * sizeof(char);
	inputPtr = new unsigned char[buffer_size];
	outputPtr = new unsigned char[buffer_size];
	outputCPtr = new unsigned char[buffer_size];
	outputPtr_avg = new unsigned char[buffer_size];
	outputPtr_max = new unsigned char[buffer_size];

#if USEDFUNCTION == 1
	printf("Used function Median3 \n\n");
#elif USEDFUNCTION == 2
	printf("Used function Median5 \n\n");
#endif 

	/*************************************************************************************
	* Read the input image
	*************************************************************************************/
	fopen_s(&input_fp, "usc.raw", "rb");

	if (fread(inputPtr, xSize, ySize, input_fp) == 0) {
		printf("Error: Input file can not be read\n");
		exit(-1);
	}
	fclose(input_fp);

	/*****************************************************
	* Median 3-tap filter
	*****************************************************/

	dCpuTime = 0.0;

	for (loopCount = 0; loopCount < 10000; loopCount++) {

		CacheFlush((__m128i*)inputPtr, buffer_size / 16);

		counter.Reset();
		counter.Start();

#if	USEDFUNCTION == 1
		sseMedian3(inputPtr, outputPtr, xSize, ySize);
#elif USEDFUNCTION == 2	
		sseMedian5(inputPtr, outputPtr, xSize, ySize);
#endif	
		CacheFlush((__m128i*)outputPtr, buffer_size / 16);
		counter.Stop();

		dCpuTime += counter.GetElapsedTime();

	}


	dCpuTime = dCpuTime / (double)loopCount;
	printf("SSE Median Performance (ms) = %f \n", dCpuTime * 1000.0);

	//CAll C module for verification
#if	USEDFUNCTION == 1
	cMedian3(inputPtr, outputCPtr, xSize, ySize);
#elif USEDFUNCTION == 2	
	cMedian5(inputPtr, outputCPtr, xSize, ySize);
#endif	

	int error = verify(outputCPtr, outputPtr, xSize, ySize);
	if (error != 0)
		printf("Verify Failed \n");
	else
		printf("Verify Successful \n");


	fopen_s(&output_fp, "out_simd.raw", "wb");

	if (fwrite(outputPtr, xSize, ySize, output_fp) == 0) {
		printf("file write error: C_5_tap_simd.raw\n");
		exit(-1);
	}/* fi */



	dCpuTime = 0.0;

	for (loopCount = 0; loopCount < 10000; loopCount++) {

		CacheFlush((__m128i*)inputPtr, buffer_size / 16);

		counter.Reset();
		counter.Start();


		avg_pooling_sse(inputPtr, outputPtr_avg, xSize, ySize);
	
		CacheFlush((__m128i*)outputPtr_avg, buffer_size / 16);
		counter.Stop();

		dCpuTime += counter.GetElapsedTime();

	}
	fopen_s(&output_fp, "out_avg.raw", "wb");

	dCpuTime = dCpuTime / (double)loopCount;
	printf("AVG Pooling Performance (ms) = %f \n", dCpuTime * 1000.0);

	if (fwrite(outputPtr_avg, xSize, ySize, output_fp) == 0) {
		printf("file write error: out_avg.raw\n");
		exit(-1);
	}/* fi */
	fclose(output_fp);







	dCpuTime = 0.0;

	for (loopCount = 0; loopCount < 10000; loopCount++) {

		CacheFlush((__m128i*)inputPtr, buffer_size / 16);

		counter.Reset();
		counter.Start();


		max_pooling_sse(inputPtr, outputPtr_max, xSize, ySize);

		CacheFlush((__m128i*)outputPtr_max, buffer_size / 16);
		counter.Stop();

		dCpuTime += counter.GetElapsedTime();

	}
	fopen_s(&output_fp, "out_max.raw", "wb");

	dCpuTime = dCpuTime / (double)loopCount;
	printf("Max Pooling Performance (ms) = %f \n", dCpuTime * 1000.0);

	if (fwrite(outputPtr_max, xSize, ySize, output_fp) == 0) {
		printf("file write error: out_max.raw\n");
		exit(-1);
	}/* fi */
	fclose(output_fp);

	
	/* free the allocated memories */
	free(inputPtr);
	free(outputPtr);
	free(outputCPtr);
	return(0);



}


int verify(unsigned char* COutput, unsigned char* SIMDOutput, int xSize, int ySize) {
	for (int i = 0; i < xSize * ySize; i++) {
		if (COutput[i] != SIMDOutput[i]) {
			printf("COutput[%d] = %d SIMDOutput[%d]=%d \n", i, COutput[i], i, SIMDOutput[i]);
			return(1);
		}
	}
	return(0);
}


void cMedian3(unsigned char* __restrict inputPtr, unsigned char* outputCPtr, int xSize, int ySize) {

	int temp0, temp1, temp2, temp3, temp4, temp5, temp6;
	int x_count;
	int y_count;
	for (y_count = 0; y_count < ySize - 2; y_count++) { //-2 because of 3 -tap

		for (x_count = 0; x_count < xSize; x_count++) {
			temp0 = *(inputPtr + y_count * xSize + x_count);
			temp1 = *(inputPtr + (y_count + 1) * xSize + x_count);
			temp2 = *(inputPtr + (y_count + 2) * xSize + x_count);
			temp3 = (temp0 > temp1) ? temp0 : temp1;
			temp4 = (temp0 < temp1) ? temp0 : temp1;
			temp5 = (temp2 < temp3) ? temp2 : temp3;
			temp6 = (temp5 > temp4) ? temp5 : temp4;
			*(outputCPtr + y_count * xSize + x_count) = temp6;
		}
	}

}
void sseMedian3(unsigned char* __restrict inputPtr, unsigned char* outputPtr, int xSize, int ySize) {

	int nsrcwidth = xSize >> 4;

	int x_count;
	int y_count;

	__m128i a;
	__m128i* src128Ptr;
	__m128i* dst128Ptr;

	__m128i temp0, temp1, temp2, temp3, temp4, temp5, temp6;

	for (y_count = 0; y_count < ySize - 2; y_count++) { //-2 because of 3 -tap

		for (x_count = 0; x_count < nsrcwidth; x_count++) {


			src128Ptr = (__m128i*)(inputPtr + x_count * 16 + y_count * xSize);
			dst128Ptr = (__m128i*)(outputPtr + x_count * 16 + y_count * xSize);

			temp0 = *src128Ptr;
			temp1 = *(src128Ptr + 1 * nsrcwidth);
			temp2 = *(src128Ptr + 2 * nsrcwidth);

			temp3 = _mm_max_epu8(temp0, temp1); //a>b ,a
			temp4 = _mm_min_epu8(temp0, temp1);//b
			temp5 = _mm_min_epu8(temp2, temp3);//
			temp6 = _mm_max_epu8(temp4, temp5);//d

			*dst128Ptr = temp6;
		}
	}

}

void cMedian5(unsigned char* __restrict inputPtr, unsigned char* outputCPtr, int xSize, int ySize) {

	int tempA1, tempA2, tempA3, tempA4, tempB1, tempB2, tempB3, tempB4, tempB5, tempB6, tempB7;
	int tempC1, tempC2, tempC3, tempC4, tempC5, tempC6, tempD1, tempD2, tempD3, tempD4, tempE1, tempE2;
	int x_count;
	int y_count;
	for (y_count = 0; y_count < ySize - 4; y_count++) { //-4 because of 5 -tap

		for (x_count = 0; x_count < xSize; x_count++) {
			tempA1 = *(inputPtr + (y_count)*xSize + x_count);
			tempB1 = *(inputPtr + (y_count + 1) * xSize + x_count);
			tempC1 = *(inputPtr + (y_count + 2) * xSize + x_count);
			tempD1 = *(inputPtr + (y_count + 3) * xSize + x_count);
			tempE1 = *(inputPtr + (y_count + 4) * xSize + x_count);

			tempA2 = min(tempA1, tempB1);
			tempB2 = max(tempA1, tempB1);

			tempB3 = min(tempB2, tempC1);
			tempC2 = max(tempB2, tempC1);
			tempA3 = min(tempA2, tempB3);
			tempB4 = max(tempA2, tempB3);

			tempC3 = min(tempC2, tempD1);
			tempD2 = max(tempC2, tempD1);
			tempB5 = min(tempB4, tempC3);
			tempC4 = max(tempB4, tempC3);
			tempD3 = min(tempD2, tempE1);
			tempC5 = min(tempC4, tempD3);

			tempB6 = max(tempA3, tempB5);
			tempC6 = max(tempB6, tempC5); // third highest value

			*(outputCPtr + y_count * xSize + x_count) = tempC6;
		}
	}

}

void sseMedian5(unsigned char* __restrict inputPtr, unsigned char* outputPtr, int xSize, int ySize) {

	int nsrcwidth = xSize >> 4;

	int x_count;
	int y_count;

	__m128i a;
	__m128i* src128Ptr;
	__m128i* dst128Ptr;

	__m128i tempA1, tempA2, tempA3, tempA4, tempB1, tempB2, tempB3, tempB4, tempB5, tempB6, tempB7;
	__m128i tempC1, tempC2, tempC3, tempC4, tempC5, tempC6, tempD1, tempD2, tempD3, tempD4, tempE1, tempE2;


	for (y_count = 0; y_count < ySize - 4; y_count++) { //-4 because of 5 -tap

#pragma unroll 64
		for (x_count = 0; x_count < nsrcwidth; x_count++) {

			src128Ptr = (__m128i*)(inputPtr + x_count * 16 + y_count * xSize);
			dst128Ptr = (__m128i*)(outputPtr + x_count * 16 + y_count * xSize);

			tempA1 = *src128Ptr;
			tempB1 = *(src128Ptr + 1 * nsrcwidth);
			tempC1 = *(src128Ptr + 2 * nsrcwidth);
			tempD1 = *(src128Ptr + 3 * nsrcwidth);
			tempE1 = *(src128Ptr + 4 * nsrcwidth);

			tempA2 = _mm_min_epu8(tempA1, tempB1);
			tempB2 = _mm_max_epu8(tempA1, tempB1);
			tempB3 = _mm_min_epu8(tempB2, tempC1);
			tempC2 = _mm_max_epu8(tempB2, tempC1);
			tempC3 = _mm_min_epu8(tempC2, tempD1);
			tempD2 = _mm_max_epu8(tempC2, tempD1);
			tempD3 = _mm_min_epu8(tempD2, tempE1);
			//tempE2 = _mm_max_epu8(tempD2, tempE1); // highest value

			tempA3 = _mm_min_epu8(tempA2, tempB3);
			tempB4 = _mm_max_epu8(tempA2, tempB3);
			tempB5 = _mm_min_epu8(tempB4, tempC3);
			tempC4 = _mm_max_epu8(tempB4, tempC3);
			tempC5 = _mm_min_epu8(tempC4, tempD3);
			//tempD4 = _mm_max_epu8(tempC4, tempD3); // second highest value

			//tempA4 = _mm_min_epu8(tempA3, tempB5);
			tempB6 = _mm_max_epu8(tempA3, tempB5);
			//tempB7 = _mm_min_epu8(tempB6, tempC5);
			tempC6 = _mm_max_epu8(tempB6, tempC5); // third highest value

			*dst128Ptr = tempC6;
		}
	}

}






void avg_pooling_sse(unsigned char* input, unsigned char* output, int width, int height) {
	int out_width = width / 2;
	int out_height = height / 2;

	for (int y = 0; y < out_height; ++y) {
		for (int x = 0; x < out_width; ++x) {
			int idx = (y * 2) * width + (x * 2);

			// Load four 8-bit values (2x2 pooling window) into SSE registers
			__m128i top_left = _mm_cvtsi32_si128(*(int*)&input[idx]);
			__m128i top_right = _mm_cvtsi32_si128(*(int*)&input[idx + 1]);
			__m128i bottom_left = _mm_cvtsi32_si128(*(int*)&input[idx + width]);
			__m128i bottom_right = _mm_cvtsi32_si128(*(int*)&input[idx + width + 1]);

			// Unpack 8-bit integers to 16-bit integers for addition without overflow
			__m128i tl = _mm_unpacklo_epi8(top_left, _mm_setzero_si128());
			__m128i tr = _mm_unpacklo_epi8(top_right, _mm_setzero_si128());
			__m128i bl = _mm_unpacklo_epi8(bottom_left, _mm_setzero_si128());
			__m128i br = _mm_unpacklo_epi8(bottom_right, _mm_setzero_si128());

			// Sum the four 16-bit integers
			__m128i sum1 = _mm_add_epi16(tl, tr);
			__m128i sum2 = _mm_add_epi16(bl, br);
			__m128i sum = _mm_add_epi16(sum1, sum2);

			// Divide by 4 to compute the average (shift right by 2)
			__m128i avg = _mm_srli_epi16(sum, 2);

			// Pack the 16-bit averages back to 8-bit integers
			__m128i result = _mm_packus_epi16(avg, _mm_setzero_si128());

			// Store the result in the output array
			output[y * out_width + x] = (unsigned char)_mm_cvtsi128_si32(result);
		}
	}
}


void max_pooling_sse(unsigned char* input, unsigned char* output, int width, int height) {
	int out_width = width / 2;
	int out_height = height / 2;

	for (int y = 0; y < out_height; ++y) {
		for (int x = 0; x < out_width; ++x) {
			int idx = (y * 2) * width + (x * 2);

			// Load four 8-bit values (2x2 pooling window) into SSE registers
			__m128i top_left = _mm_cvtsi32_si128(*(int*)&input[idx]);
			__m128i top_right = _mm_cvtsi32_si128(*(int*)&input[idx + 1]);
			__m128i bottom_left = _mm_cvtsi32_si128(*(int*)&input[idx + width]);
			__m128i bottom_right = _mm_cvtsi32_si128(*(int*)&input[idx + width + 1]);

			// Unpack 8-bit integers to 16-bit integers for comparison
			__m128i tl = _mm_unpacklo_epi8(top_left, _mm_setzero_si128());
			__m128i tr = _mm_unpacklo_epi8(top_right, _mm_setzero_si128());
			__m128i bl = _mm_unpacklo_epi8(bottom_left, _mm_setzero_si128());
			__m128i br = _mm_unpacklo_epi8(bottom_right, _mm_setzero_si128());

			// Compare pairs and find max values
			__m128i max1 = _mm_max_epu8(tl, tr);
			__m128i max2 = _mm_max_epu8(bl, br);
			__m128i max = _mm_max_epu8(max1, max2);

			// Pack the 16-bit maximums back to 8-bit integers
			__m128i result = _mm_packus_epi16(max, _mm_setzero_si128());

			// Store the result in the output array
			output[y * out_width + x] = (unsigned char)_mm_cvtsi128_si32(result);
		}
	}
}