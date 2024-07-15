#include "conv8.h"


#include "stdafx.h"

#include <stdio.h>
#include <stdlib.h>
#include "Timer.h"

#include <emmintrin.h>


#define min(a,b) ((a < b) ? a : b)
#define max(a,b) ((a > b) ? a : b)

#define MAX_FILTER_WIDTH 32
#define MAX_FILTER_HEIGHT 32
void conv8(unsigned char* inputPtr, unsigned char* outputConvPtr, float* filter, int kernel_width, int kernel_height,  int xSize, int ySize, int scale, int pad);

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
	unsigned char* outputConvPtr;				//Output Image
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
	kernel_width = 15;
	kernel_height = 15;

	buffer_size = xSize * ySize * sizeof(char);
	printf("buffer_size : %d \n", buffer_size);

	inputPtr = new unsigned char[buffer_size];
	outputConvPtr = new unsigned char[buffer_size];

	filterPtr = new  float[MAX_FILTER_WIDTH * MAX_FILTER_HEIGHT * sizeof(short)];

	for (i = 0; i < kernel_height; i++)
		for (j = 0; j < kernel_width; j++)
			*(filterPtr + i * kernel_width + j) = 1.0;

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
	int pad;
	pad = 1;
	counter.Reset();
	counter.Start();
	for (loopCount = 0; loopCount < 100; loopCount++) {
		CacheFlush((__m128i*)inputPtr, buffer_size / 16);
		conv8(inputPtr, outputConvPtr, filterPtr, kernel_width, kernel_height, xSize, ySize, scale, pad);
	}
	counter.Stop();
	dCpuTime = counter.GetElapsedTime() / (double)loopCount;
	printf("C 2DConv Performance (ms) = %f \n", dCpuTime * 1000.0);

	err = fopen_s(&output_fp, "convOutfile.raw", "wb");
	if (err != 0) {
		printf("Error: output file can not be opened\n");
		exit(-1);
	}


	if (fwrite(outputConvPtr, xSize, ySize, output_fp) == 0)
	{
		printf("file write error: convOutfile.raw\n");
		exit(-1);
	}/* fi */

	fclose(output_fp);
	/* free the allocated memories */
	free(inputPtr);
	free(outputConvPtr);
	free(filterPtr);

	return 0;
}

void conv8(unsigned char* inputPtr, unsigned char* outputConvPtr, float* filterPtr, int kernel_width, int kernel_height,
	int xSize, int ySize, int scale, int pad)
{

	int i, j;
	float sum;
	int image_r, image_c;
	int flag = 0, image_value;
	float temp0;
	int filter_value, xyflag;
	int yLoc, xLoc, r, c;

	for (yLoc = 0; yLoc < ySize; yLoc++)
		for (xLoc = 0; xLoc < xSize; xLoc++) {
			sum = 0.0;
			for (r = -kernel_height / 2; r <= kernel_height / 2; r++)	 // row wise
				for (c = -kernel_width / 2; c <= kernel_width / 2; c++) { // col wise
				
					image_c = xLoc + c;
					image_r = yLoc + r;
					if (pad == 1) {
						image_c = min(max(image_c, 0), (xSize - 1));
						image_r = min(max(image_r, 0), (ySize - 1));
						image_value = (inputPtr[image_r * xSize + image_c]);
					}
					else {
						xyflag = (image_c < 0) || (image_c > xSize - 1) || (image_r < 0) || (image_r > ySize - 1);
						image_value = xyflag ? 0 : (inputPtr[image_r * xSize + image_c]);
					}

					filter_value = filterPtr[(r + kernel_height / 2) * kernel_width + c + kernel_width / 2];

					sum += (float)(image_value * filter_value);

				}

			temp0 = ((sum + 0.5f) / scale);

			temp0 = (temp0 < 0.0) ? 0.0 : temp0;
			temp0 = (temp0 > 255.0) ? 255.0 : temp0;

			outputConvPtr[yLoc * xSize + xLoc] = (unsigned char)(temp0);
		}

}

