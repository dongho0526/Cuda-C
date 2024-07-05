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

void conv8_separable(unsigned char* inputPtr, unsigned char* outputConvPtr, int kernel_width, int kernel_height, int xSize, int ySize, int pad);


void sobel(unsigned char* inputPtr, unsigned char* outputConvPtr, float* PtrGx, float* PtrGy, char xory, int xSize, int ySize, int scale, int pad);

void sobel_edge_detector(
	unsigned char* inputPtr,
	unsigned char* outputPtr,
	int width,
	int height);

void CacheFlush(__m128i* src, unsigned int countVect) {
	int j;
	for (j = 0; j < countVect; j++) {
		_mm_clflush(src + j);
	}
}


int _tmain(int argc, _TCHAR* argv[])
{	
	int change = 15;
	FILE* input_fp, * output_fp;
	unsigned char* inputPtr;					//Input Image
	unsigned char* outputConvPtr;				//Output Image
	float* filterPtr;
	int* filterIntPtr;
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
	kernel_width = change;
	kernel_height = change;

	buffer_size = xSize * ySize * sizeof(char);
	printf("buffer_size : %d \n", buffer_size);

	inputPtr = new unsigned char[buffer_size];
	outputConvPtr = new unsigned char[buffer_size];

	filterPtr = new  float[MAX_FILTER_WIDTH * MAX_FILTER_HEIGHT ];
	filterIntPtr = new  int [MAX_FILTER_WIDTH * MAX_FILTER_HEIGHT ];

	/* Set all the coefficient to 1.0. Boxcr filter*/
	for (i = 0; i < kernel_height; i++)
		for (j = 0; j < kernel_width; j++)
			*(filterPtr + i * kernel_width + j) = 1.0;
	//Normalize convolution results

	scale = kernel_width * kernel_height;

	//Fixed point keren


	for (i = 0; i < kernel_height; i++)
		for (j = 0; j < kernel_width; j++)
			*(filterIntPtr + i * kernel_width + j) = (int) (65536.0/scale);


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
	pad = 0; // Boundary pad with zeros
	dCpuTime = 0.0;
	for (loopCount = 0; loopCount < 100; loopCount++) {
		CacheFlush((__m128i*)inputPtr, buffer_size / 16);
		CacheFlush((__m128i*)outputConvPtr, buffer_size / 16);
		counter.Reset();
		counter.Start();
		conv8(inputPtr, outputConvPtr, filterPtr, kernel_width, kernel_height, xSize, ySize, scale, pad);
		counter.Stop();
		dCpuTime += counter.GetElapsedTime();
	}


	printf("C Performance (ms) = %f \n", dCpuTime / (double)loopCount * 1000.0);


	err = fopen_s(&output_fp, "convOutfile_15.raw", "wb");
	if (err != 0) {
		printf("Error: output file can not be opened\n");
		exit(-1);
	}


	if (fwrite(outputConvPtr, xSize, ySize, output_fp) == 0)
	{
		printf("file write error: convOutfile.raw\n");
		exit(-1);
	}/* fi */
	


	float* PtrGx = new float[9];
	float* PtrGy = new float[9];

	int Gx[3][3] = { {-1, 0, 1}, {-2, 0, 2} , {-1, 0, 1} };
	int Gy[3][3] = { {1, 2, 1}, {0, 0, 0}, {-1, -2, -1} };

	for (i = 0; i < 3; i++)
		for (j = 0; j < 3; j++) {
			*(PtrGx + i * 3 + j) = (int)65536 * Gx[i][j];
			*(PtrGy + i * 3 + j) = (int)65536 * Gy[i][j];
		}

	free(outputConvPtr);
	outputConvPtr = new unsigned char[buffer_size];

	//////////////////////////////////////////////////
	//Sobel filter
	
	pad = 0; // Boundary pad with zeros
	dCpuTime = 0.0;
	for (loopCount = 0; loopCount < 100; loopCount++) {
		CacheFlush((__m128i*)inputPtr, buffer_size / 16);
		CacheFlush((__m128i*)outputConvPtr, buffer_size / 16);
		counter.Reset();
		counter.Start();
		sobel_edge_detector(inputPtr, outputConvPtr, xSize, ySize);
		counter.Stop();
		dCpuTime += counter.GetElapsedTime();
	}
	/////////////////////////////////////////////////

	printf("Sobel Filter C Performance (ms) = %f \n", dCpuTime / (double)loopCount * 1000.0);

	err = fopen_s(&output_fp, "convOutfile_sobel.raw", "wb");
	if (err != 0) {
		printf("Error: output file can not be opened\n");
		exit(-1);
	}


	if (fwrite(outputConvPtr, xSize, ySize, output_fp) == 0)
	{
		printf("file write error: convOutfile_sobel.raw\n");
		exit(-1);
	}/* fi */

	free(outputConvPtr);
	outputConvPtr = new unsigned char[buffer_size];
	
	pad = 0; // Boundary pad with zeros
	dCpuTime = 0.0;
	for (loopCount = 0; loopCount < 100; loopCount++) {
		CacheFlush((__m128i*)inputPtr, buffer_size / 16);
		CacheFlush((__m128i*)outputConvPtr, buffer_size / 16);
		counter.Reset();
		counter.Start();
		conv8_separable(inputPtr, outputConvPtr, kernel_width, kernel_height, xSize, ySize, pad);
		counter.Stop();
		dCpuTime += counter.GetElapsedTime();
	}

	printf("C Performance (ms) = %f \n", dCpuTime / (double)loopCount * 1000.0);

	
	err = fopen_s(&output_fp, "convOutfile_separable.raw", "wb");
	if (err != 0) {
		printf("Error: output file can not be opened\n");
		exit(-1);
	}


	if (fwrite(outputConvPtr, xSize, ySize, output_fp) == 0)
	{
		printf("file write error: convOutfile_separable.raw\n");
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
					xyflag = (image_c < 0) || (image_c > xSize - 1) || (image_r < 0) || (image_r > ySize - 1);

					if (pad == 0) {
						image_value = xyflag ? 0 : (inputPtr[image_r * xSize + image_c]);
					}

					//	else {} //Define other types of pads

					filter_value = filterPtr[(r + kernel_height / 2) * kernel_width + c + kernel_width / 2];
					//printf("%lf", filter_value);
					sum += (float)(image_value * filter_value);

				}

			//temp0 = sum >> 16;
			temp0 = ((sum + 0.5f) / scale);

			temp0 = (temp0 < 0.0) ? 0.0 : temp0;
			temp0 = (temp0 > 255.0) ? 255.0 : temp0;

			outputConvPtr[yLoc * xSize + xLoc] = (unsigned char)(temp0);
		}

}




void conv8_separable(unsigned char* inputPtr, unsigned char* outputConvPtr, int kernel_width, int kernel_height, int xSize, int ySize, int pad) {
	int x, y, k;
	float sum;
	int image_r, image_c;
	int image_value;
	float temp0;
	int xyflag;

	// Boxcar filter value (for separable filter, the sum of filter elements should be 1)
	float boxcar_value_width = 1.0f / kernel_width;
	float boxcar_value_height = 1.0f / kernel_height;

	// Temporary buffer to store the intermediate results after row convolution
	float* tempBuffer = (float*)malloc(xSize * ySize * sizeof(float));

	// Row-wise convolution
	for (y = 0; y < ySize; y++) {
		for (x = 0; x < xSize; x++) {
			sum = 0.0f;
			for (k = -kernel_width / 2; k <= kernel_width / 2; k++) {
				image_c = x + k;
				xyflag = (image_c < 0) || (image_c >= xSize);

				if (pad == 0) {
					image_value = xyflag ? 0 : inputPtr[y * xSize + image_c];
				}
				else {
					// Add other padding methods if necessary
					image_value = 0;
				}

				sum += (float)(image_value * boxcar_value_width);
			}
			tempBuffer[y * xSize + x] = sum;
		}
	}

	// Column-wise convolution
	for (y = 0; y < ySize; y++) {
		for (x = 0; x < xSize; x++) {
			sum = 0.0f;
			for (k = -kernel_height / 2; k <= kernel_height / 2; k++) {
				image_r = y + k;
				xyflag = (image_r < 0) || (image_r >= ySize);

				if (pad == 0) {
					image_value = xyflag ? 0 : tempBuffer[image_r * xSize + x];
				}
				else {
					// Add other padding methods if necessary
					image_value = 0;
				}

				sum += (float)(image_value * boxcar_value_height);
			}

			// Normalize and clip the result
			temp0 = ((sum + 0.5f));
			temp0 = (temp0 < 0.0f) ? 0.0f : temp0;
			temp0 = (temp0 > 255.0f) ? 255.0f : temp0;

			outputConvPtr[y * xSize + x] = (unsigned char)(temp0);
		}
	}

	free(tempBuffer);
}

void sobel_edge_detector(
	unsigned char* inputPtr,
	unsigned char* outputPtr,
	int width,
	int height)
{
	int x, y;
	float gx, gy;
	float G;

	// Sobel kernels
	float Gx[3][3] = {
		{-1.0f, 0.0f, 1.0f},
		{-2.0f, 0.0f, 2.0f},
		{-1.0f, 0.0f, 1.0f}
	};

	float Gy[3][3] = {
		{-1.0f, -2.0f, -1.0f},
		 {0.0f,  0.0f,  0.0f},
		 {1.0f,  2.0f,  1.0f}
	};

	for (y = 1; y < height - 1; y++) {
		for (x = 1; x < width - 1; x++) {
			gx = 0.0f;
			gy = 0.0f;

			// Apply Sobel kernels
			for (int r = -1; r <= 1; r++) {
				for (int c = -1; c <= 1; c++) {
					int pixel = (int)inputPtr[(y + r) * width + (x + c)];
					gx += pixel * Gx[r + 1][c + 1];
					gy += pixel * Gy[r + 1][c + 1];
				}
			}

			// Calculate the magnitude of the gradient
			G = gy;

			// Normalize to the range [0, 255]
			G = (G > 255.0f) ? 255.0f : G;
			G = (G < 0.0f) ? 0.0f : G;

			outputPtr[y * width + x] = (unsigned char)G;
		}
	}
}