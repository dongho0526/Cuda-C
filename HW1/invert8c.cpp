#include <stdio.h>
#include <stdlib.h>

#include "Timer.h"

void invert8(unsigned char* inputPtr, unsigned char* outputPtr, int xSize, int ySize);
void transpose8(unsigned char* inputPtr, unsigned char* outputPtr, int xSize, int ySize);
void transpose_block(unsigned char* inputPtr, unsigned char* outputPtr, int xSize, int ySize);

void flip_x(unsigned char* inputPtr, unsigned char* outputPtr, int xSize, int ySize);
void flip_y(unsigned char* inputPtr, unsigned char* outputPtr, int xSize, int ySize);

int main()
{
	FILE* input_fp, * output_fp;
	unsigned char* inputPtr;					//Input Image
	unsigned char* outCPtr;				//Output Image

	int xSize, ySize;
	int i, j;

	int buffer_size;

	float dCpuTime;
	int loopCount;
	char temp3;

	CPerfCounter counter;


	xSize = 128;
	ySize = 128;

	buffer_size = xSize * ySize * sizeof(char);
	printf("buffer_size : %d \n", buffer_size);

	inputPtr = new unsigned char[buffer_size];
	outCPtr = new unsigned char[buffer_size];

	/*************************************************************************************
	* Read the input image
	*************************************************************************************/



	int err = fopen_s(&input_fp, "usc.raw", "r");
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
	for (loopCount = 0; loopCount < 100; loopCount++) {
		counter.Reset();
		counter.Start();
		transpose8(inputPtr, outCPtr, xSize, ySize);
		counter.Stop();
		dCpuTime += counter.GetElapsedTime();
	}


	//dCpuTime = counter.GetElapsedTime()/(double)loopCount;
	printf("Transpose Performance (ms) = %f \n", dCpuTime / (double)loopCount * 1000.0);

	err = fopen_s(&output_fp, "outfile_transpose.raw", "wb");
	if (err != 0) {
		printf("Error: Output file can not be opened\n");
		exit(-1);
	}

	if (fwrite(outCPtr, xSize, ySize, output_fp) == 0)
	{
		printf("file write error: outfile_transpose.raw\n");
		exit(-1);
	}/* fi */

	printf("Saving image done %d %d \n ", xSize, ySize);

	/* free the allocated memories */
	delete[] inputPtr;
	delete[] outCPtr;

	return 0;
}
void invert8(unsigned char* inputPtr, unsigned char* outputPtr, int xSize, int ySize)
{
	int i, j;
	unsigned char inData, outData;


	for (i = 0; i < ySize; i++)
		for (j = 0; j < xSize; j++) {
			inData = *(inputPtr + i * xSize + j);
			outData = 255 - inData;
			//outData = inData;
			*(outputPtr + i * xSize + j) = outData;
		}
}


void transpose8(unsigned char* inputPtr, unsigned char* outputPtr, int xSize, int ySize) {
	int i, j, k;
	unsigned char inData, outData;
	for (i = 0; i < ySize; i++)
		for (j = 0; j < xSize; j++) {
			//for (k = 0; k < 3 && (i + k) < ySize; k++)
				//*(outputPtr + (j * ySize) + (i + k)) = *(inputPtr + ((i + k) * xSize) + j);
				*(outputPtr + j * ySize + i) = *(inputPtr + i * xSize + j);

		}
}

void transpose_block(unsigned char* inputPtr, unsigned char* outputPtr, int xSize, int ySize)
{

	int i, j, k, l;
	int blockSize = 128;
	int xCounter, yCounter;
	unsigned char inData, outData;
	xCounter = xSize / blockSize;
	yCounter = ySize / blockSize;

	for (i = 0; i < ySize; i += blockSize)
		for (j = 0; j < xSize; j += blockSize)
			for (k = 0; k < blockSize; k++)
				for (l = 0; l < blockSize; l++) {
					int a = (i  + k) * xSize + (j + l);
					int b = (j + l) * ySize + (i + k);
					outputPtr[a] = inputPtr[b];
				}
}

void flip_x(unsigned char* inputPtr, unsigned char* outputPtr, int xSize, int ySize)
{
	int i, j, k, l;
	unsigned char inData, outData;
	for (i = 0; i< ySize; i++)
		for (j = 0; j < xSize; j++) {
			*(outputPtr + i * xSize + j) = *(inputPtr + (ySize - 1 - i) * xSize + j);
		}
}

void flip_y(unsigned char* inputPtr, unsigned char* outputPtr, int xSize, int ySize)
{
	int i, j, k, l;
	unsigned char inData, outData;
	for (i = 0; i < ySize; i++)
		for (j = 0; j < xSize; j++) {
			*(outputPtr + i * xSize + j) = *(inputPtr + i * xSize + (xSize - 1 - j));
		}
}