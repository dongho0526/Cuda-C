// intel8sse.cpp : Defines the entry point for the console application.
//

// invert8.cpp : Defines the entry point for the console application.
//

//#include "stdafx.h"

#include <iostream>
#include <stdlib.h>
#include "Timer.h"
#include <tmmintrin.h>
#include <emmintrin.h>
#include <mmintrin.h>
#include <immintrin.h>
#include <omp.h>


void invert8(unsigned char* inputPtr, unsigned char* outputPtr, int xSize, int ySize);

void invert8_SIMD(unsigned char* inputPtr, unsigned char* outputPtr, int xSize, int ySize);

int verify(unsigned char* cOutput, unsigned char* SSEOutput, int xSize, int ySize);

void
invert8_256SIMD(
	
	unsigned char* src_ptr,
	unsigned char* dst_ptr,
	int width,
	int height, 
	int td);


void CacheFlush(__m128i* src, unsigned int countVect) {
	unsigned int j;
	for (j = 0; j < countVect; j++) {
		_mm_clflush(src + j);
	}
}

int main()
{
	FILE* input_fp, * output_fp;
	unsigned char* inputPtr;					//Input Image
	unsigned char* outCPtr, * outSIMDPtr;				//Output Image

	int xSize, ySize;

	int buffer_size;

	double dCpuTime;
	int loopCount;


	CPerfCounter counter;


	xSize = 512;
	ySize = 512;

	buffer_size = xSize * ySize * sizeof(char);
	printf("buffer_size : %d \n", buffer_size);

	inputPtr = new unsigned char[buffer_size];
	outCPtr = new unsigned char[buffer_size];
	outSIMDPtr = new unsigned char[buffer_size];

	/*************************************************************************************
	* Read the input image
	*************************************************************************************/
	errno_t err;
	err = fopen_s(&input_fp, "usc.raw", "r");
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

	dCpuTime = 0.0f;
	for (loopCount = 0; loopCount < 1000; loopCount++) {
		CacheFlush((__m128i*)inputPtr, buffer_size / 16);
		CacheFlush((__m128i*)outCPtr, buffer_size / 16);
		counter.Reset();
		counter.Start();
		invert8(inputPtr, outCPtr, xSize, ySize);
		counter.Stop();
		dCpuTime += counter.GetElapsedTime();
	}

	printf("C Performance (ms) = %f \n", dCpuTime / (double)loopCount * 1000.0);

	if ((fopen_s(&output_fp, "outfile_C.raw", "wb")) != 0) {
		printf("Error: Output file can not be opened\n");
		exit(-1);
	}
	if (fwrite(outCPtr, xSize, ySize, output_fp) == 0)
	{
		printf("file write error: outfile_C.raw\n");
		exit(-1);
	}/* fi */

	int td, tdmax, tdySize, ySizeLast;
	td = 1;
	dCpuTime = 0.0f;
	for (loopCount = 0; loopCount < 1000; loopCount++) {
		CacheFlush((__m128i*)inputPtr, buffer_size / 16);
		CacheFlush((__m128i*)outSIMDPtr, buffer_size / 16);
		counter.Reset();
		counter.Start();
		invert8_256SIMD(inputPtr, outSIMDPtr, xSize, ySize, td);
		counter.Stop();
		dCpuTime += counter.GetElapsedTime();
	}

	printf("SIMD AVX Performance (ms) = %f \n", dCpuTime / (double)loopCount * 1000.0);

	
	//int nthreads = omp_get_num_threads();
	tdmax = omp_get_max_threads();
	printf("tdmax threads = %d\n", tdmax);
	tdySize = ySize / tdmax;
	printf("tdySize = %d\n", tdySize);
	if (ySize % tdmax != 0)
		ySizeLast = ySize - tdySize * tdmax;
	else
		ySizeLast = 0;

	dCpuTime = 0.0f;
	for (loopCount = 0; loopCount < 1000; loopCount++) {
		CacheFlush((__m128i*)inputPtr, buffer_size / 16);
		CacheFlush((__m128i*)outSIMDPtr, buffer_size / 16);
		counter.Reset();
		counter.Start();
		//invert8_256SIMD(inputPtr, outSIMDPtr, xSize, ySize, td);

#pragma omp parallel
		{
		    td = omp_get_thread_num();
			invert8_256SIMD(inputPtr + td *tdySize*xSize, outSIMDPtr+ td * tdySize*xSize, xSize, tdySize, td);
		}
		invert8_256SIMD(inputPtr + tdmax * tdySize * xSize, outSIMDPtr + tdmax * tdySize * xSize, xSize, ySizeLast, td);
		
		counter.Stop();
		dCpuTime += counter.GetElapsedTime();
	}

	printf("SIMD multi proc Performance (ms) = %f \n", dCpuTime / (double)loopCount * 1000.0);


	/*********************************************************
	* Write the verify.raw
	*********************************************************/

	if ((fopen_s(&output_fp, "outfile_SIMD.raw", "wb")) != 0) {
		printf("Error: Output file can not be opened\n");
		exit(-1);
	}
	if (fwrite(outSIMDPtr, xSize, ySize, output_fp) == 0)
	{
		printf("file write error: outfile_SIMD.raw\n");
		exit(-1);
	}/* fi */

	fclose(output_fp);


	int error = verify(outCPtr, outSIMDPtr, xSize, ySize);

	if (error != 0)
		printf("Verify Failed \n");
	else
		printf("Verify Successful \n");

	/* free the allocated memories */
	free(inputPtr);
	free(outCPtr);
	free(outSIMDPtr);

	return 0;
}

//USING restrict
void
invert8_SIMD(
	//	unsigned char *__restrict src_ptr, 
	//	unsigned char *__restrict dst_ptr, 

	unsigned char* src_ptr,
	unsigned char* dst_ptr,
	int width,
	int height)
{
	int m;
	__m128i* src128_ptr, * dst128_ptr;
	__m128i data, temp0;
	__m128i const128_ff;

	
	const128_ff = _mm_set1_epi8(0xff);
	src128_ptr = (__m128i*) (src_ptr);
	dst128_ptr = (__m128i*) (dst_ptr);

	for (m = 0; m < height * width / 16; m++) {
		data = *(src128_ptr++);
		temp0 = _mm_subs_epi8(const128_ff, data);
		*(dst128_ptr++) = temp0;
	}

}


//USING restrict
void
invert8_256SIMD(
	//	unsigned char *__restrict src_ptr, 
	//	unsigned char *__restrict dst_ptr, 

	unsigned char* src_ptr,
	unsigned char* dst_ptr,
	int width,
	int height,
	int td)
{
	int m;
	__m256i* src256_ptr, * dst256_ptr;
	__m256i data, temp0;
	__m256i const256_ff;

	//printf("td = % d\n", td);

	const256_ff = _mm256_set1_epi8(0xff);
	src256_ptr = (__m256i*) (src_ptr);
	dst256_ptr = (__m256i*) (dst_ptr);

	for (m = 0; m < height * width / 32; m++) {
		data = *(src256_ptr++);
		temp0 = _mm256_subs_epi8(const256_ff, data);
		*(dst256_ptr++) = temp0;
	}

}


void invert8(unsigned char* inputPtr, unsigned char* outputPtr, int xSize, int ySize)
{
	int i, j;
	unsigned char inData, outData;
	for (i = 0; i < ySize; i++)
		for (j = 0; j < xSize; j++) {
			inData = *(inputPtr + i * xSize + j);
			outData = 255 - inData;
			*(outputPtr + i * xSize + j) = outData;
		}
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


