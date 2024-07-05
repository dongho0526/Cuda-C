// verticalMedianFilter3Tap.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"




#include <iostream>
#include <fstream>
#include <fstream>
#include <stdio.h>
#include <emmintrin.h>
#include "Timer.h"


#include <math.h>
#include <cmath>



void cMedian3(unsigned char*__restrict inputPtr, unsigned char* outputCPtr, int xSize, int ySize);
void hMedian3(unsigned char* __restrict inputPtr, unsigned char* outputCPtr, int xSize, int ySize);
void vertical5(unsigned char* __restrict inputPtr, unsigned char* outputCPtr, int xSize, int ySize);
void horizontal5(unsigned char* __restrict inputPtr, unsigned char* outputCPtr, int xSize, int ySize);
void median3x3(unsigned char* __restrict inputPtr, unsigned char* outputCPtr, int xSize, int ySize);


void CacheFlush(__m128i *src, unsigned int countVect) {
	int j;
	for (j = 0; j<countVect; j++) {
		_mm_clflush(src + j);
	}
}

int main()

{

	FILE * input_fp, *output_fp;
	unsigned char *inputPtr, *outputCPtr;					//Input Image
	int xSize, ySize;
	int buffer_size;
	float dCpuTime;
	int loopCount;
	CPerfCounter counter;

	xSize = 512;
	ySize = 512;

	buffer_size = xSize*ySize * sizeof(char);
	inputPtr = new unsigned char[buffer_size];
	outputCPtr = new unsigned char[buffer_size];

	/*************************************************************************************
	* Read the input image
	*************************************************************************************/
	fopen_s(&input_fp, "usc.raw", "rb");

	if (fread(inputPtr, xSize, ySize, input_fp) == 0) {
		printf("Error: Input file can not be read\n");
		exit(-1);
	}
	fclose(input_fp);

	counter.Reset();
	counter.Start();
	//CAll C module for verification
	for (loopCount = 0; loopCount < 1; loopCount++)
		//vertical5(inputPtr, outputCPtr, xSize, ySize);
		median3x3(inputPtr, outputCPtr, xSize, ySize);
		//cMedian3(inputPtr, outputCPtr, xSize, ySize);
		//horizontal5(inputPtr, outputCPtr, xSize, ySize);
		

	counter.Stop();
	dCpuTime = counter.GetElapsedTime() / (double)loopCount;
	printf("C performance (ms) = %f \n", dCpuTime * 1000.0);

	fopen_s(&output_fp, "outC_3x3.raw", "wb");

	if (fwrite(outputCPtr, xSize, ySize, output_fp) == 0){
		printf("file write error: C_5_tap_simd.raw\n");
		exit(-1);
	}/* fi */

	fclose(output_fp);
	/* free the allocated memories */
	free(inputPtr);
	free(outputCPtr);
	return(0);
}



void cMedian3(unsigned char*__restrict inputPtr, unsigned char* outputCPtr, int xSize, int ySize) {

	int temp0, temp1, temp2, temp3, temp4, temp5, temp6;
	int x_count;
	int y_count;
	for (y_count = 0; y_count < ySize - 2; y_count++) { //-2 because of 3 -tap

		for (x_count = 0; x_count < xSize; x_count++) {
			temp0 = *(inputPtr + y_count*xSize + x_count);
			temp1 = *(inputPtr + (y_count+1)*xSize + x_count);
			temp2 = *(inputPtr + (y_count+2)*xSize + x_count);
			if (temp0 > temp1) std::swap(temp0, temp1);
			if (temp1 > temp2) std::swap(temp1, temp2);
			if (temp0 > temp1) std::swap(temp1, temp2);
			*(outputCPtr + y_count*xSize + x_count) = temp1;
		}
	}

}


void hMedian3(unsigned char* __restrict inputPtr, unsigned char* outputCPtr, int xSize, int ySize) {

	int temp0, temp1, temp2, temp3, temp4, temp5, temp6;
	int x_count;
	int y_count;
	for (y_count = 0; y_count < ySize; y_count++) { //-2 because of 3 -tap

		for (x_count = 0; x_count < xSize-2; x_count++) {
			temp0 = *(inputPtr + y_count * xSize + x_count);
			temp1 = *(inputPtr + (y_count) * xSize + x_count+1);
			temp2 = *(inputPtr + (y_count) * xSize + x_count+2);
			if (temp0 > temp1) std::swap(temp0, temp1);
			if (temp1 > temp2) std::swap(temp1, temp2);
			if (temp0 > temp1) std::swap(temp1, temp2);
			*(outputCPtr + y_count * xSize + x_count) = temp1;
		}
	}

}


void horizontal5(unsigned char* __restrict inputPtr, unsigned char* outputCPtr, int xSize, int ySize) {
	int temp0, temp1, temp2, temp3, temp4;
	int x_count;
	int y_count;
	for(y_count = 0; y_count < ySize-4; y_count++)
		for (x_count = 0; x_count < xSize; x_count++) {
			temp0 = inputPtr[y_count * xSize + x_count];
			temp1 = inputPtr[(y_count + 1) * xSize + x_count];
			temp2 = inputPtr[(y_count + 2) * xSize + x_count];
			temp3 = inputPtr[(y_count + 3) * xSize + x_count];
			temp4 = inputPtr[(y_count + 4) * xSize + x_count];
			if (temp0 > temp1) std::swap(temp0, temp1);
			if (temp2 > temp3) std::swap(temp2, temp3);
			if (temp0 > temp2)std::swap(temp0, temp2);
			if (temp1 > temp3)std::swap(temp1, temp3);
			if (temp4 > temp2)std::swap(temp4, temp2);
			if (temp4 > temp0)std::swap(temp4, temp0);
			if (temp4 > temp1)std::swap(temp4, temp1);

			*(outputCPtr + y_count * xSize + x_count) = temp4;
		}
}


void vertical5(unsigned char* __restrict inputPtr, unsigned char* outputCPtr, int xSize, int ySize) {
	int temp0, temp1, temp2, temp3, temp4;
	int x_count;
	int y_count;
	for (y_count = 0; y_count < ySize; y_count++) {
		for (x_count = 0; x_count < xSize - 4; x_count++) {
			temp0 = inputPtr[y_count * xSize + x_count];
			temp1 = inputPtr[(y_count) * xSize + x_count+1];
			temp2 = inputPtr[(y_count) * xSize + x_count+2];
			temp3 = inputPtr[(y_count) * xSize + x_count+3];
			temp4 = inputPtr[(y_count) * xSize + x_count+4];
			if (temp0 > temp1) std::swap(temp0, temp1);
			if (temp2 > temp3) std::swap(temp2, temp3);
			if (temp0 > temp2)std::swap(temp0, temp2);
			if (temp1 > temp3)std::swap(temp1, temp3);
			if (temp4 > temp2)std::swap(temp4, temp2);
			if (temp4 > temp0)std::swap(temp4, temp0);
			if (temp4 > temp1)std::swap(temp4, temp1);

			*(outputCPtr + y_count * xSize + x_count) = temp4;
		}
	}
}

void median3x3(unsigned char* __restrict inputPtr, unsigned char* outputCPtr, int xSize, int ySize) {
	int temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8;
	int x_count;
	int y_count;
	for (y_count = 0; y_count < ySize - 2; y_count++) {
		for (x_count = 0; x_count < xSize - 2; x_count++) {
			temp0 = inputPtr[y_count * xSize + x_count];
			temp1 = inputPtr[y_count * xSize + x_count+1];
			temp2 = inputPtr[y_count * xSize + x_count+2];
			temp3 = inputPtr[(y_count+1) * xSize + x_count];
			temp4 = inputPtr[(y_count+1) * xSize + x_count+1];
			temp5 = inputPtr[(y_count+1) * xSize + x_count+2];
			temp6 = inputPtr[(y_count+2) * xSize + x_count];
			temp7 = inputPtr[(y_count+2) * xSize + x_count+1];
			temp7 = inputPtr[(y_count+2) * xSize + x_count+2];
			if (temp0 > temp1) std::swap(temp0, temp1);
			if (temp3 > temp4) std::swap(temp3, temp4);
			if (temp6 > temp7) std::swap(temp6, temp7);
			if (temp1 > temp2) std::swap(temp1, temp2);
			if (temp4 > temp5) std::swap(temp4, temp5);
			if (temp7 > temp8) std::swap(temp7, temp8);
			if (temp0 > temp1) std::swap(temp0, temp1);
			if (temp3 > temp4) std::swap(temp3, temp4);
			if (temp6 > temp7) std::swap(temp6, temp7);

			if (temp0 > temp3) std::swap(temp0, temp3);
			if (temp3 > temp6) std::swap(temp3, temp6);
			if (temp0 > temp3) std::swap(temp0, temp3);
			if (temp1 > temp4) std::swap(temp1, temp4);
			if (temp4 > temp7) std::swap(temp4, temp7);
			if (temp1 > temp4) std::swap(temp1, temp4);
			if (temp2 > temp5) std::swap(temp2, temp5);
			if (temp5 > temp8) std::swap(temp5, temp8);
			if (temp2 > temp5) std::swap(temp2, temp5);

			if (temp2 > temp4) std::swap(temp2, temp4);
			if (temp4 > temp6) std::swap(temp4, temp6);
			if (temp2 > temp4) std::swap(temp2, temp4);
			if (temp1 > temp3) std::swap(temp1, temp3);
			if (temp3 > temp5) std::swap(temp3, temp5);
			if (temp1 > temp3) std::swap(temp1, temp3);

			if (temp3 > temp4) std::swap(temp3, temp4);

			*(outputCPtr + y_count * xSize + x_count) = temp4;
		}
	}
}