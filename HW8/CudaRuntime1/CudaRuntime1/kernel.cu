
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"

#include <stdio.h>

#define thread_num 1024
#define block_num 1
#define block_num_x 1
#define block_num_y 512

cudaError_t Add(unsigned char *input, unsigned char *output, int xSize, int ySize);
cudaError_t FlipX(unsigned char* input, unsigned char* output, int xSize, int ySize);
cudaError_t FlipY(unsigned char* input, unsigned char* output, int xSize, int ySize);


__global__ void AddKernel_1thread(unsigned char* input, unsigned char* output, int xSize, int ySize)
{

    int idx = threadIdx.x; // thread => 0
    
    // Ensure that we only have one thread and one block
    if (idx == 0) {
        for (int i = 0; i < xSize * ySize; i++) {
            int temp = input[i] + input[i];
            output[i] = (temp > 255) ? 255 : temp;
        }
    }

}

__global__ void FlipYKernel_1thread(unsigned char* input, unsigned char* output, int xSize, int ySize) {
    int idx = threadIdx.x;
    int idy = threadIdx.y;

    if (idx == 0 && idy == 0) {
        for (int y = 0; y < ySize; ++y) {
            for (int x = 0; x < xSize; ++x) {
                int inputIndex = y * xSize + x;
                int outputIndex = y * xSize + (xSize - 1 - x);
                output[outputIndex] = input[inputIndex];
                
            }
        }
    }

}

__global__ void FlipXKernel_1thread(unsigned char* input, unsigned char* output, int xSize, int ySize) {
    int idx = threadIdx.x;
    int idy = threadIdx.y;

    if (idx == 0 && idy == 0) {
        for (int y = 0; y < ySize; ++y) {
            for (int x = 0; x < xSize; ++x) {
                int inputIndex = y * xSize + x;
                int outputIndex = (ySize - y - 1) * xSize + x;
                output[outputIndex] = input[inputIndex];
            }
        }
    }
}


__global__ void AddKernel_1024(unsigned char* input, unsigned char* output, int xSize, int ySize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // 각 스레드의 고유 인덱스
    int xySize = xSize * ySize;
    int numThreads = blockDim.x * gridDim.x; // 전체 스레드 수

    for (int i = idx; i < xySize; i += numThreads) {
        int temp = input[i] + input[i];
        output[i] = (temp > 255) ? 255 : temp;
    }
}

__global__ void FlipXKernel_1024(unsigned char* input, unsigned char* output, int xSize, int ySize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int xySize = xSize * ySize;
    int numThreadsx = blockDim.x * gridDim.x;

    for (int i = idx; i < xySize; i += numThreadsx) {
        int x = i % xSize;
        int y = i / xSize;
        int inputIndex = y * xSize + x;
        int outputIndex = (ySize - 1 - y) * xSize + x;
        output[outputIndex] = input[inputIndex];
    }

}

__global__ void FlipYKernel_1024(unsigned char* input, unsigned char* output, int xSize, int ySize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int xySize = xSize * ySize;
    int numThreadsx = blockDim.x * gridDim.x;

    for (int i = idx; i < xySize; i += numThreadsx) {
        int x = i % xSize;
        int y = i / xSize;
        int inputIndex = y * xSize + x;
        int outputIndex = y * xSize + (xSize - 1 - x);
        output[outputIndex] = input[inputIndex];
    }
}

__global__ void AddKernel_512x512(unsigned char* input, unsigned char* output, int xSize, int ySize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < xSize && y < ySize) {
        int idx = y * xSize + x;
        int temp = input[idx] + input[idx];
        output[idx] = (temp > 255) ? 255 : temp;
    }
}

__global__ void FlipXKernel_512x512(unsigned char* input, unsigned char* output, int xSize, int ySize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < xSize && y < ySize) {
        int inputIndex = y * xSize + x;
        int outputIndex = (ySize-1-y) * xSize + x;
        output[outputIndex] = input[inputIndex];
    }
}

__global__ void FlipYKernel_512x512(unsigned char* input, unsigned char* output, int xSize, int ySize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < xSize && y < ySize) {
        int inputIndex = y * xSize + x;
        int outputIndex = y * xSize + (xSize - 1 - x);
        output[outputIndex] = input[inputIndex];
    }

}

// Zblock은 blockSize => (xSize, 1) gridSize => (ySize, 1)

__global__ void AddKernel_Zblock(unsigned char* input, unsigned char* output, int xSize, int ySize)
{
    int x = threadIdx.x;
    int y = blockIdx.x;

    if (x < xSize && y < ySize) {
        int idx = y * xSize + x;
        int temp = input[idx] + input[idx];
        output[idx] = (temp > 255) ? 255 : temp;
    }

}

__global__ void FlipXKernel_Zblock(unsigned char* input, unsigned char* output, int xSize, int ySize)
{
    int x = threadIdx.x;
    int y = blockIdx.x;

    if (x < xSize && y < ySize) {
        int inputIndex = y * xSize + x;
        int outputIndex = (ySize - 1 - y) * xSize + x;
        output[outputIndex] = input[inputIndex];
    }
}

__global__ void FlipYKernel_Zblock(unsigned char* input, unsigned char* output, int xSize, int ySize)
{
    int x = threadIdx.x;
    int y = blockIdx.x;

    if (x < xSize && y < ySize) {
        int inputIndex = y * xSize + x;
        int outputIndex = y * xSize + (xSize - 1 - x);
        output[outputIndex] = input[inputIndex];
    }

}

// Mblock은 blockSize => (xSize/blocksize, ySize/blocksize) gridSize => (16, 16)

__global__ void AddKernel_Mblock(unsigned char* input, unsigned char* output, int xSize, int ySize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < xSize && y < ySize) {
        int idx = y * xSize + x;
        int temp = input[idx] + input[idx];
        output[idx] = (temp > 255) ? 255 : temp;
    }
}

__global__ void FlipXKernel_Mblock(unsigned char* input, unsigned char* output, int xSize, int ySize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < xSize && y < ySize) {
        int inputIndex = y * xSize + x;
        int outputIndex = (ySize - 1 - y) * xSize + x;
        output[outputIndex] = input[inputIndex];
    }
}

__global__ void FlipYKernel_Mblock(unsigned char* input, unsigned char* output, int xSize, int ySize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < xSize && y < ySize) {
        int inputIndex = y * xSize + x;
        int outputIndex = y * xSize + (xSize - 1 - x);
        output[outputIndex] = input[inputIndex];
    }

}


int main()
{
    unsigned char* input, * CudaOutput, * CudaOutput_X, * CudaOutput_Y;
    int xSize, ySize;
    
    xSize = 512;
    ySize = 512;

    input = new unsigned char[xSize * ySize];
    CudaOutput = new unsigned char[xSize * ySize];
    CudaOutput_X = new unsigned char[xSize * ySize];
    CudaOutput_Y = new unsigned char[xSize * ySize];

    int i, j;

    FILE* fp;
    fp = fopen("usc.raw", "rb");
    fread(input, xSize, ySize, fp);

    //Add(input, CudaOutput, xSize, ySize);
    
    cudaError_t cudaStatus = Add(input, CudaOutput, xSize, ySize);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Add failed!\n");
        return 1;
    }

    fp = fopen("Output_add.raw", "wb");
    fwrite(CudaOutput, xSize, ySize, fp);
    fclose(fp);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    
    cudaError_t cudaStatus_X = FlipX(input, CudaOutput_X, xSize, ySize);

    if (cudaStatus_X != cudaSuccess) {
        fprintf(stderr, "FlipX failed\n");
        return 1;
    }

    fp = fopen("Output_FlipX.raw", "wb");
    fwrite(CudaOutput_X, xSize, ySize, fp);
    fclose(fp);

    cudaStatus_X = cudaDeviceReset();
    if (cudaStatus_X != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }


    cudaError_t cudaStatus_Y = FlipY(input, CudaOutput_Y, xSize, ySize);

    if (cudaStatus_Y != cudaSuccess) {
        fprintf(stderr, "FlipY failed\n");
        return 1;
    }

    fp = fopen("Output_FlipY.raw", "wb");
    fwrite(CudaOutput_Y, xSize, ySize, fp);
    fclose(fp);

    cudaStatus_Y = cudaDeviceReset();
    if (cudaStatus_Y != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }



    delete[] CudaOutput;
    delete[] CudaOutput_X;
    delete[] CudaOutput_Y;
    delete[] input;

    return 0;
}

cudaError_t Add(unsigned char* input, unsigned char* output, int xSize, int ySize)
{
    unsigned char* dev_input = 0;
    unsigned char* dev_output = 0;

    unsigned int xySize = xSize * ySize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaError_t cudaStatus;

    cudaDeviceProp prop;
    int count;

    cudaGetDeviceCount(&count);

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }


    cudaEventRecord(start, 0);

    cudaStatus = cudaMalloc((void**)&dev_input, xySize * sizeof(char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        //goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_output, xySize * sizeof(char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        //goto Error;
    }

    cudaStatus = cudaMemcpy(dev_input, input, xySize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        //goto Error;
    }

    dim3 blockSize(16, 16); // 1D 블록, 1024 스레드
    dim3 gridSize(32, 32); // 1D 그리드, 1 블록

    //AddKernel_1thread << <1, 1 >> > (dev_input, dev_output, xSize, ySize);
    //AddKernel << < blockSize, gridSize >> > (dev_input, dev_output, xSize, ySize);
    //AddKernel_512x512 << < blockSize, gridSize >> > (dev_input, dev_output, xSize, ySize);
    //AddKernel_Zblock << <blockSize, gridSize >> > (dev_input, dev_output, xSize, ySize);
    AddKernel_Mblock << <blockSize, gridSize >> > (dev_input, dev_output, xSize, ySize);

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching invert8Kernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(output, dev_output, xySize * sizeof(char), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float cudaElapsedTime;
    cudaEventElapsedTime(&cudaElapsedTime, start, stop);
    printf("Add - Time for execution = %3.1f ms \n", cudaElapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

Error:
    cudaFree(dev_input);
    cudaFree(dev_output);

    return cudaStatus;
}




cudaError_t FlipY(unsigned char* input, unsigned char* output, int xSize, int ySize)
{
    unsigned char* dev_input = 0;
    unsigned char* dev_output = 0;

    unsigned int xySize = xSize * ySize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaError_t cudaStatus_X;

    cudaDeviceProp prop;
    int count;

    cudaGetDeviceCount(&count);

    cudaStatus_X = cudaSetDevice(0);
    if (cudaStatus_X != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }


    cudaEventRecord(start, 0);

    cudaStatus_X = cudaMalloc((void**)&dev_input, xySize * sizeof(char));
    if (cudaStatus_X != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        //goto Error;
    }

    cudaStatus_X = cudaMalloc((void**)&dev_output, xySize * sizeof(char));
    if (cudaStatus_X != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        //goto Error;
    }

    cudaStatus_X = cudaMemcpy(dev_input, input, xySize, cudaMemcpyHostToDevice);
    if (cudaStatus_X != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        //goto Error;
    }

    dim3 blockSize(16, 16); // 1D 블록, 1024 스레드
    dim3 gridSize(32, 32); // 1D 그리드, 1 블록

    //FlipYKernel << <blockSize, gridSize >> > (dev_input, dev_output, xSize, ySize);
    //FlipYKernel_1024 << <blockSize, gridSize >> > (dev_input, dev_output, xSize, ySize);
    //FlipYKernel_512x512 << <blockSize, gridSize >> > (dev_input, dev_output, xSize, ySize);
    //FlipYKernel_Zblock << <blockSize, gridSize >> > (dev_input, dev_output, xSize, ySize);
    FlipYKernel_Mblock << <blockSize, gridSize >> > (dev_input, dev_output, xSize, ySize);

    cudaStatus_X = cudaDeviceSynchronize();
    if (cudaStatus_X != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching invert8Kernel!\n", cudaStatus_X);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus_X = cudaMemcpy(output, dev_output, xySize * sizeof(char), cudaMemcpyDeviceToHost);
    if (cudaStatus_X != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float cudaElapsedTime;
    cudaEventElapsedTime(&cudaElapsedTime, start, stop);
    printf("FlipY - Time for execution = %3.1f ms \n", cudaElapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

Error:
    cudaFree(dev_input);
    cudaFree(dev_output);

    return cudaStatus_X;
}


cudaError_t FlipX(unsigned char* input, unsigned char* output, int xSize, int ySize)
{
    unsigned char* dev_input = 0;
    unsigned char* dev_output = 0;

    unsigned int xySize = xSize * ySize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaError_t cudaStatus_X;

    cudaDeviceProp prop;
    int count;


    cudaGetDeviceCount(&count);

    cudaStatus_X = cudaSetDevice(0);
    if (cudaStatus_X != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }


    cudaEventRecord(start, 0);

    cudaStatus_X = cudaMalloc((void**)&dev_input, xySize * sizeof(char));
    if (cudaStatus_X != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        //goto Error;
    }

    cudaStatus_X = cudaMalloc((void**)&dev_output, xySize * sizeof(char));
    if (cudaStatus_X != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        //goto Error;
    }

    cudaStatus_X = cudaMemcpy(dev_input, input, xySize, cudaMemcpyHostToDevice);
    if (cudaStatus_X != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        //goto Error;
    }

    dim3 blockSize(32, 32); // 1D 블록, 1024 스레드
    dim3 gridSize(16, 16); // 1D 그리드, 1 블록

    //FlipXKernel_1thread << <blcokSize, gridSize >> > (dev_input, dev_output, xSize, ySize);
    //FlipXKernel_1024 << <blockSize, gridSize >> > (dev_input, dev_output, xSize, ySize);
    //FlipXKernel_512x512 << <blockSize, gridSize >> > (dev_input, dev_output, xSize, ySize);
    //FlipXKernel_Zblock << <blockSize, gridSize >> > (dev_input, dev_output, xSize, ySize);
    FlipXKernel_Mblock << <blockSize, gridSize >> > (dev_input, dev_output, xSize, ySize);

    cudaStatus_X = cudaDeviceSynchronize();
    if (cudaStatus_X != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching invert8Kernel!\n", cudaStatus_X);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus_X = cudaMemcpy(output, dev_output, xySize * sizeof(char), cudaMemcpyDeviceToHost);
    if (cudaStatus_X != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float cudaElapsedTime;
    cudaEventElapsedTime(&cudaElapsedTime, start, stop);
    printf("FlipX - Time for execution = %3.1f ms \n", cudaElapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

Error:
    cudaFree(dev_input);
    cudaFree(dev_output);

    return cudaStatus_X;
}
