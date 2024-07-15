// invert8IACA.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "Timer.h"
#include <emmintrin.h>

using namespace std;

void invert8(unsigned char* inputPtr, unsigned char* outputPtr, int xSize, int ySize);
void invert8SIMD(unsigned char* __restrict inputPtr, unsigned char* __restrict outputPtr, int xSize, int ySize);
int verify(unsigned char* cOutput, unsigned char* SSEOutput, int xSize, int ySize);

void CacheFlush(__m128i* src, unsigned int countVect) {
	int j;
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

	buffer_size = xSize * ySize;
	printf("buffer_size : %d \n", buffer_size);

	//buffer_size = ((((xSize+16) * ySize * sizeof(char) -1) | 0x3fff) +1);
	//	inputPtr = new unsigned char[buffer_size];
	//	outSIMDPtr =  new unsigned char[buffer_size];
	
	// allocate heap storage larger than SIZE
	 outCPtr = new unsigned char[buffer_size];
	__m128i* input128Ptr = new __m128i[buffer_size];
	__m128i* output128Ptr = new __m128i[buffer_size];
	inputPtr = (unsigned char*)input128Ptr;
	outSIMDPtr = (unsigned char*)output128Ptr;;


	//	prinmtf("input_ptr=%d \n". inputPtr);


	/*************************************************************************************
	* Read the input image
	*************************************************************************************/
	int err1 = fopen_s(&input_fp, "usc.raw", "rb");
	if (err1 != 0) {
		printf("Error: Input file can not be opened\n");
		exit(-1);
	}

	if (fread(inputPtr, xSize, ySize, input_fp) == 0) {
		printf("Error: Input file can not be read\n");
		exit(-1);
	}

	fclose(input_fp);


	dCpuTime = 0.0f;
	for (loopCount = 0; loopCount < 100; loopCount++) {
		CacheFlush((__m128i*)inputPtr, buffer_size / 16);
		CacheFlush((__m128i*)outCPtr, buffer_size / 16);
		counter.Reset();
		counter.Start();
		invert8(inputPtr, outCPtr, xSize, ySize);
		counter.Stop();
		dCpuTime += counter.GetElapsedTime();
	}
	dCpuTime = dCpuTime /(double)loopCount;
	printf("C Performance (ms) = %f \n", dCpuTime * 1000.0);

	err1 = fopen_s(&output_fp, "outfile_C.raw", "wb");
	if (err1 != 0) {
		printf("Error: Output file can not be opened\n");
		exit(-1);
	}
	if (fwrite(outCPtr, xSize, ySize, output_fp) == 0)
	{
		printf("file write error: outfile_C.raw\n");
		exit(-1);
	}/* fi */

	fclose(output_fp);

	dCpuTime = 0.0f;
	for (loopCount = 0; loopCount < 100; loopCount++) {
		CacheFlush((__m128i*)inputPtr, buffer_size / 16);
		CacheFlush((__m128i*)outSIMDPtr, buffer_size / 16);
		counter.Reset();
		counter.Start();
		invert8SIMD(inputPtr, outSIMDPtr, xSize, ySize);
		counter.Stop();
		dCpuTime += counter.GetElapsedTime();
	}
	dCpuTime = dCpuTime / (double)loopCount;
	printf("SIMD Performance (ms) = %f \n", dCpuTime  * 1000.0);


	/*********************************************************
	* Write the verify.raw
	*********************************************************/

	err1 = fopen_s(&output_fp, "outfile_SIMD.raw", "wb");
	if (err1 != 0) {
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
	delete[] inputPtr;
	delete[] outCPtr;
	delete[] outSIMDPtr;
	//int aa;
	//cin >> aa;
}

//unsigned char *__restrict src_ptr, 
//unsigned char *__restrict dst_ptr, 
void
invert8SIMD(
	unsigned char* __restrict src_ptr,
	unsigned char* __restrict dst_ptr,
	int width,
	int height)
{
	int m, n;
	__m128i* src128_ptr, * dst128_ptr;
	__m128i data, temp0;

	__m128i const128_ff;

	const128_ff = _mm_set1_epi8(0xff);
	src128_ptr = (__m128i*) (src_ptr);
	dst128_ptr = (__m128i*) (dst_ptr);
	m = 0;
	for (m = 0; m < height; m++) {


		//	#pragma unroll(16)
		for (n = 0; n < width; n += 16) {

			data = *src128_ptr++;
			temp0 = _mm_xor_si128(data, const128_ff);
			//temp0 = _mm_sub_epi8(const128_ff, data);

			*dst128_ptr++ = temp0;


		}
	}

}


void invert8(unsigned char* inputPtr, unsigned char* outputPtr, int xSize, int ySize)
{
	int i, j;
	unsigned char inData;
	for (i = 0; i < ySize; i++)
		for (j = 0; j < xSize; j++) {

			inData = *(inputPtr + i * xSize + j);
			*(outputPtr + i * xSize + +j) = 255 - inData;
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