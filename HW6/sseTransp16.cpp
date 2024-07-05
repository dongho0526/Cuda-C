
#include <stdio.h>
#include <stdlib.h>
#include "Timer.h"
#include <emmintrin.h>


using namespace std;


void transp16(unsigned short* inputPtr, unsigned short* outputPtr, int xSize, int ySize);
void transp16SIMD(unsigned short* __restrict inputPtr, unsigned short* __restrict outputPtr, int xSize, int ySize);
int verify(unsigned short* cOutput, unsigned short* SSEOutput, int xSize, int ySize);


void CacheFlush(__m128i* src, unsigned int countVect) {
	int j;
	for (j = 0; j < countVect; j++) {
		_mm_clflush(src + j);
	}
}

int main()
{
	FILE* input_fp, * output_fp;	//Input Image
	unsigned short* inputPtr, * outCPtr, * outSIMDPtr;				//Output Image
	unsigned short* input16Ptr;

	int xSize, ySize;
	int i, j;

	int buffer_size;

	float dCpuTime;
	int loopCount;
	char temp3;

	CPerfCounter counter;


	xSize = 512;
	ySize = 512;

	buffer_size = xSize * ySize;
	printf("buffer_size : %d \n", buffer_size);

	input16Ptr = new unsigned short[buffer_size];

	inputPtr = new unsigned short[buffer_size];
	outCPtr = new unsigned short[buffer_size];
	outSIMDPtr = new unsigned short[buffer_size];

	/*************************************************************************************
	* Read the input image
	*************************************************************************************/
	if (fopen_s(&input_fp, "usc16.raw", "rb") != 0) {
		printf("Error: Input file can not be opened\n");
		exit(-1);
	}

	if (fread(inputPtr, xSize * 2, ySize, input_fp) == 0) {
		printf("Error: Input file can not be read\n");
		exit(-1);
	}
	fclose(input_fp);

	/*****************************************************
	* Call generic C transpose
	*****************************************************/

	dCpuTime = 0.0f;
	for (loopCount = 0; loopCount < 100; loopCount++) {
		CacheFlush((__m128i*)inputPtr, buffer_size / 16);
		CacheFlush((__m128i*)outCPtr, buffer_size / 16);
		counter.Reset();
		counter.Start();
		transp16(inputPtr, outCPtr, xSize, ySize);
		counter.Stop();
		dCpuTime += counter.GetElapsedTime();
	}
	//dCpuTime = counter.GetElapsedTime()/(double)loopCount;
	printf("C Performance (ms) = %f \n", dCpuTime / (double)loopCount * 1000.0);

	if (fopen_s(&output_fp, "outfile_C.raw", "wb") != 0) {
		printf("Error: Output file can not be opened\n");
		exit(-1);
	}
	if (fwrite(outCPtr, xSize * 2, ySize, output_fp) == 0)
	{
		printf("file write error: outfile_C.raw\n");
		exit(-1);
	}/* fi */

	dCpuTime = 0.0f;
	for (loopCount = 0; loopCount < 100; loopCount++) {
		CacheFlush((__m128i*)inputPtr, buffer_size / 16);
		CacheFlush((__m128i*)outSIMDPtr, buffer_size / 16);
		counter.Reset();
		counter.Start();
		transp16SIMD(inputPtr, outSIMDPtr, xSize, ySize);
		counter.Stop();
		dCpuTime += counter.GetElapsedTime();
	}

	printf("SIMD Performance (ms) = %f \n", dCpuTime / (double)loopCount * 1000.0);


	/*********************************************************
	* Write the verify.raw
	*********************************************************/

	if (fopen_s(&output_fp, "outfile_SIMD.raw", "wb") != 0) {
		printf("Error: Output file can not be opened\n");
		exit(-1);
	}

	if (fwrite(outSIMDPtr, xSize * 2, ySize, output_fp) == 0)
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
}

void
transp16SIMD(
	unsigned short* __restrict src_ptr,
	unsigned short* __restrict dst_ptr,
	int width,
	int height)
{
	int i, j, k, l, m, n;
	__m128i* src128_ptr, * dst128_ptr;
	__m128i data[8], temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7;
	__m128i temp8, temp9, temp10, temp11, temp12, temp13, temp14, temp15;
	__m128i tempa, tempb, tempc, tempd, tempe, tempf, tempg, temph;
	int offset_x, offset_y, width_temp, height_temp;
	__m128i const* src128_const_ptr;


	/*---- End of declarations ----*/
	offset_x = width >> 3;
	offset_y = height >> 3;


	/*******************************************************
	* FUNCTION:
	*   Process a data block, in 8x8 blocks

	*******************************************************/

	// for (m = num_thread; m<num_thread + CILK_TRANSP_PROCESS; m+=8){
	for (m = 0; m < height; m += 8) {
		for (n = 0; n < width; n += 8) {
			src128_ptr = (__m128i*)(src_ptr + m * width + n);
			dst128_ptr = (__m128i*)(dst_ptr + n * height + m);

			// Load 8x8 block
			temp0 = _mm_loadu_si128(src128_ptr);
			temp1 = _mm_loadu_si128(src128_ptr + offset_x);
			temp2 = _mm_loadu_si128(src128_ptr + 2 * offset_x);
			temp3 = _mm_loadu_si128(src128_ptr + 3 * offset_x);
			temp4 = _mm_loadu_si128(src128_ptr + 4 * offset_x);
			temp5 = _mm_loadu_si128(src128_ptr + 5 * offset_x);
			temp6 = _mm_loadu_si128(src128_ptr + 6 * offset_x);
			temp7 = _mm_loadu_si128(src128_ptr + 7 * offset_x);

			// Transpose 8x8 block
			temp8 = _mm_unpacklo_epi16(temp0, temp1);
			temp9 = _mm_unpackhi_epi16(temp0, temp1);
			temp10 = _mm_unpacklo_epi16(temp2, temp3);
			temp11 = _mm_unpackhi_epi16(temp2, temp3);
			temp12 = _mm_unpacklo_epi16(temp4, temp5);
			temp13 = _mm_unpackhi_epi16(temp4, temp5);
			temp14 = _mm_unpacklo_epi16(temp6, temp7);
			temp15 = _mm_unpackhi_epi16(temp6, temp7);

			tempa = _mm_unpacklo_epi32(temp8, temp12);
			tempb = _mm_unpackhi_epi32(temp8, temp12);
			tempc = _mm_unpacklo_epi32(temp9, temp13);
			tempd = _mm_unpackhi_epi32(temp9, temp13);
			tempe = _mm_unpacklo_epi32(temp10, temp14);
			tempf = _mm_unpackhi_epi32(temp10, temp14);
			tempg = _mm_unpacklo_epi32(temp11, temp15);
			temph = _mm_unpackhi_epi32(temp11, temp15);

			temp0 = _mm_unpacklo_epi64(tempa, tempe);
			temp1 = _mm_unpackhi_epi64(tempa, tempe);
			temp2 = _mm_unpacklo_epi64(tempb, tempf);
			temp3 = _mm_unpackhi_epi64(tempb, tempf);
			temp4 = _mm_unpacklo_epi64(tempc, tempg);
			temp5 = _mm_unpackhi_epi64(tempc, tempg);
			temp6 = _mm_unpacklo_epi64(tempd, temph);
			temp7 = _mm_unpackhi_epi64(tempd, temph);

			// Store transposed 8x8 block
			_mm_storeu_si128(dst128_ptr, temp0);
			_mm_storeu_si128(dst128_ptr + offset_y, temp1);
			_mm_storeu_si128(dst128_ptr + 2 * offset_y, temp2);
			_mm_storeu_si128(dst128_ptr + 3 * offset_y, temp3);
			_mm_storeu_si128(dst128_ptr + 4 * offset_y, temp4);
			_mm_storeu_si128(dst128_ptr + 5 * offset_y, temp5);
			_mm_storeu_si128(dst128_ptr + 6 * offset_y, temp6);
			_mm_storeu_si128(dst128_ptr + 7 * offset_y, temp7);
		}
	}

}

int verify(unsigned short* COutput, unsigned short* SIMDOutput, int xSize, int ySize) {
	for (int i = 0; i < xSize * ySize; i++) {
		if (COutput[i] != SIMDOutput[i]) {
			printf("COutput[%d] = %d SIMDOutput[%d]=%d \n", i, COutput[i], i, SIMDOutput[i]);
			return(1);
		}
	}
	return(0);
}

void transp16(unsigned short* inputPtr, unsigned short* outputPtr, int xSize, int ySize)
{

	int i, j, k, l;
	int blockSize = 32;
	int xCounter, yCounter;
	unsigned short inData, outData;
	xCounter = xSize / blockSize;
	yCounter = ySize / blockSize;
	for (i = 0; i < ySize; i += blockSize)
		for (j = 0; j < xSize; j += blockSize)
			for (k = 0; k < blockSize; k++)
				for (l = 0; l < blockSize; l++) {
						inData = *(inputPtr + (i + k) * xSize + j + l);
					*(outputPtr + (j + l) * ySize + i + k) = inData;
				}
}
\




