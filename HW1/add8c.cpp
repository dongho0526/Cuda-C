// add8c.cpp : Defines the entry point for the console application.
//

#include <stdio.h>
#include <stdlib.h>

#include "Timer.h"


void add8(unsigned char *inputPtr0, unsigned char *inputPtr1, unsigned char *outputPtr, int xSize, int ySize);
void add8_2(unsigned char* inputPtr0, unsigned char* inputPtr1, unsigned char* outputPtr, int xSize, int ySize);
void add8_3(unsigned char* inputPtr0, unsigned char* inputPtr1, unsigned char* outputPtr, int xSize, int ySize);
int verify(unsigned char *input0, unsigned char *input1, unsigned char *outputPtr, int xSize, int ySize);


int main()
{
	FILE * input_fp, *output_fp;
	unsigned char *inputPtr0;					//Input Image
	unsigned char *inputPtr1;					//Input Image
	unsigned char  *outCPtr;				//Output Image

	int xSize, ySize;
	int i, j;

	int buffer_size;

	float dCpuTime;
	int loopCount;
	char temp3;

	CPerfCounter counter;


	xSize = 512;
	ySize = 512;

	buffer_size = xSize*ySize * sizeof(char);
	printf("buffer_size : %d \n", buffer_size);

	inputPtr0 = new unsigned char[buffer_size];
	inputPtr1 = new unsigned char[buffer_size];
	outCPtr = new unsigned char[buffer_size];


	for (i = 0; i < 16; i++)
		*(inputPtr0 + i) = i;

	unsigned int* input32_ptr;
	input32_ptr = (unsigned int*)inputPtr0;

	


	/*************************************************************************************
	* Read the input image
	*************************************************************************************/
	int err = fopen_s(&input_fp, "usc.raw", "r");
	if (err != 0) {
		printf("Error: Input file usc.raw can not be opened\n");
		exit(-1);
	}

	if (fread(inputPtr0, xSize, ySize, input_fp) == 0) {
		printf("Error: Input file can not be read\n");
		exit(-1);
	}

	fclose(input_fp);

	// err = fopen_s(&input_fp, "outfile_C.raw", "r");
	err = fopen_s(&input_fp, "circle8.raw", "rb"); // barbaragray.raw
	if (err != 0) {
		//printf("%s", input_fp);
		printf("Error: Input file circle8.raw can not be opened\n");
		exit(-1);
	}

	if (fread(inputPtr1, xSize, ySize, input_fp) == 0) {
		printf("Error: Input file circle8.raw can not be read\n");
		exit(-1);
	}

	fclose(input_fp);

	/*****************************************************
	* Call generic C invert8
	*****************************************************/

	dCpuTime = 0.0f;
	for (loopCount = 0; loopCount<1000; loopCount++) {

		counter.Reset();
		counter.Start();
		add8_2(inputPtr0, inputPtr1, outCPtr, xSize, ySize);
		counter.Stop();
		dCpuTime += counter.GetElapsedTime();
	}


	//dCpuTime = counter.GetElapsedTime()/(double)loopCount;
	printf("%d", loopCount);
	printf("add8 C Performance (ms) = %f \n", dCpuTime / (double)loopCount*1000.0);

	err = fopen_s(&output_fp, "outResult.raw", "wb");
	if (err != 0) {
		printf("Error: Output file can not be opened\n");
		exit(-1);
	}

	if (fwrite(outCPtr, xSize, ySize, output_fp) == 0)
	{
		printf("file write error: outResult.raw\n");
		exit(-1);
	}/* fi */


	int error = verify(inputPtr0, inputPtr1, outCPtr, xSize, ySize);

	if (error != 0)
		printf("Verify Failed \n");
	else
		printf("Verify Successful \n");


	 /* free the allocated memories */
	delete [] inputPtr0;
	delete [] inputPtr1;
	delete [] outCPtr;
	return 0;
}
void add8(unsigned char *inputPtr0, unsigned char *inputPtr1, unsigned char *outputPtr, int xSize, int ySize)
{
	int i, j;
	unsigned char inData0, inData1;
	unsigned char outData;
	unsigned int temp2;

	//unsigned int outData32;
	//unsigned int inData032, inData132;


	for (i = 0; i<ySize; i++)
		//#pragma omp parallel
		for (j = 0; j<xSize; j++) {
			inData0 = *(inputPtr0 + i*xSize + j);
			inData1 = *(inputPtr1 + i*xSize + j);
			temp2 = inData0 + inData1;
			outData=(unsigned char)(temp2 / 2);
			*(outputPtr + i*xSize + j) = outData;
		}
}

void add8_2(unsigned char* inputPtr0, unsigned char* inputPtr1, unsigned char* outputPtr, int xSize, int ySize)
{
	int i, j;
	unsigned char inData0, inData1;
	unsigned char outData;
	unsigned int temp2;

	//unsigned int outData32;
	//unsigned int inData032, inData132;


	for (i = 0; i < ySize; i++)
		//#pragma omp parallel
		for (j = 0; j < xSize; j++) {
			inData0 = *(inputPtr0 + i * xSize + j);
			inData1 = *(inputPtr1 + i * xSize + j);
			temp2 = inData0 + inData1;
			outData = (unsigned short)(temp2);
			*(outputPtr + i * xSize + j) = outData;
		}
}

void add8_3(unsigned char* inputPtr0, unsigned char* inputPtr1, unsigned char* outputPtr, int xSize, int ySize)
{
	int i, j;
	unsigned char inData0, inData1;
	unsigned char outData;
	unsigned int temp2;

	for (i = 0; i < ySize; i++)
		for (j = 0; j < xSize; j++) {
			inData0 = *(inputPtr0 + i * xSize + j);
			inData1 = *(inputPtr1 + i * xSize + j);
			temp2 = inData0 + inData1;
			temp2 = temp2 > 255 ? 255 : temp2;
			outData = (unsigned char)temp2;
			*(outputPtr + i * xSize + j) = outData;
		}
}

int verify(unsigned char *inPut0, unsigned char *inPut1, unsigned char *outPut, int xSize, int ySize) {
	int temp;
	for (int i = 0; i<xSize*ySize; i++) {
		//temp = (inPut0[i] + inPut1[i]) / 2;
		temp = (inPut0[i] + inPut1[i]);
		if (temp != (int) (outPut[i])) {
			//printf("i=%d, inPut0[%d] = %d inPut1[%d] = %d, temp = %d, outPut[%d]=%d \n", i, i, inPut0[i], i, inPut1[i], temp, i, outPut[i]);
			return(0);
		}
	}
	return(0);
}
