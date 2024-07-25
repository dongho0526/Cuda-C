/*
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"

#include <stdio.h>

#define N (1024*1024)
#define FULL_DATA_SIZE (N*20)
#define CHUNK_SIZE (N/3) // Smaller chunk size to better utilize streams

__global__ void kernel(int* a, int* b, int* c, int dataSize) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < dataSize) {
        int idx1 = (idx + 1) % 256;
        int idx2 = (idx + 2) % 256;
        float as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
        float bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
        c[idx] = (as + bs) / 2;
    }
}

int main(void) {
    cudaDeviceProp prop;
    int whichDevice;
    cudaGetDevice(&whichDevice);
    cudaGetDeviceProperties(&prop, whichDevice);
    if (!prop.deviceOverlap) {
        printf("Device will not handle overlaps, so no speed up from streams\n");
        return 0;
    }

    cudaEvent_t start, stop;
    float elapsedTime;

    cudaStream_t stream0, stream1, stream2;
    int* host_a, * host_b, * host_c;
    int* dev_a0, * dev_b0, * dev_c0;
    int* dev_a1, * dev_b1, * dev_c1;
    int* dev_a2, * dev_b2, * dev_c2;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    cudaMalloc((void**)&dev_a0, CHUNK_SIZE * sizeof(int));
    cudaMalloc((void**)&dev_b0, CHUNK_SIZE * sizeof(int));
    cudaMalloc((void**)&dev_c0, CHUNK_SIZE * sizeof(int));

    cudaMalloc((void**)&dev_a1, CHUNK_SIZE * sizeof(int));
    cudaMalloc((void**)&dev_b1, CHUNK_SIZE * sizeof(int));
    cudaMalloc((void**)&dev_c1, CHUNK_SIZE * sizeof(int));

    cudaMalloc((void**)&dev_a2, CHUNK_SIZE * sizeof(int));
    cudaMalloc((void**)&dev_b2, CHUNK_SIZE * sizeof(int));
    cudaMalloc((void**)&dev_c2, CHUNK_SIZE * sizeof(int));

    cudaHostAlloc((void**)&host_a, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_b, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_c, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);

    for (int i = 0; i < FULL_DATA_SIZE; i++) {
        host_a[i] = rand();
        host_b[i] = rand();
    }

    cudaEventRecord(start, 0);

    for (int i = 0; i < FULL_DATA_SIZE; i += CHUNK_SIZE * 3) {
        // 1st stream
        cudaMemcpyAsync(dev_a0, host_a + i, CHUNK_SIZE * sizeof(int), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(dev_b0, host_b + i, CHUNK_SIZE * sizeof(int), cudaMemcpyHostToDevice, stream0);
        kernel << <CHUNK_SIZE / 256, 256, 0, stream0 >> > (dev_a0, dev_b0, dev_c0, CHUNK_SIZE);
        cudaMemcpyAsync(host_c + i, dev_c0, CHUNK_SIZE * sizeof(int), cudaMemcpyDeviceToHost, stream0);

        // 2nd stream
        if (i + CHUNK_SIZE < FULL_DATA_SIZE) {
            cudaMemcpyAsync(dev_a1, host_a + i + CHUNK_SIZE, CHUNK_SIZE * sizeof(int), cudaMemcpyHostToDevice, stream1);
            cudaMemcpyAsync(dev_b1, host_b + i + CHUNK_SIZE, CHUNK_SIZE * sizeof(int), cudaMemcpyHostToDevice, stream1);
            kernel << <CHUNK_SIZE / 256, 256, 0, stream1 >> > (dev_a1, dev_b1, dev_c1, CHUNK_SIZE);
            cudaMemcpyAsync(host_c + i + CHUNK_SIZE, dev_c1, CHUNK_SIZE * sizeof(int), cudaMemcpyDeviceToHost, stream1);
        }

        // 3rd stream
        if (i + 2 * CHUNK_SIZE < FULL_DATA_SIZE) {
            cudaMemcpyAsync(dev_a2, host_a + i + 2 * CHUNK_SIZE, CHUNK_SIZE * sizeof(int), cudaMemcpyHostToDevice, stream2);
            cudaMemcpyAsync(dev_b2, host_b + i + 2 * CHUNK_SIZE, CHUNK_SIZE * sizeof(int), cudaMemcpyHostToDevice, stream2);
            kernel << <CHUNK_SIZE / 256, 256, 0, stream2 >> > (dev_a2, dev_b2, dev_c2, CHUNK_SIZE);
            cudaMemcpyAsync(host_c + i + 2 * CHUNK_SIZE, dev_c2, CHUNK_SIZE * sizeof(int), cudaMemcpyDeviceToHost, stream2);
        }
    }

    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time taken:  %3.1f ms\n", elapsedTime);

    cudaFreeHost(host_a);
    cudaFreeHost(host_b);
    cudaFreeHost(host_c);
    cudaFree(dev_a0);
    cudaFree(dev_b0);
    cudaFree(dev_c0);
    cudaFree(dev_a1);
    cudaFree(dev_b1);
    cudaFree(dev_c1);
    cudaFree(dev_a2);
    cudaFree(dev_b2);
    cudaFree(dev_c2);
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return 0;
}
*/
/*
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"

#include <stdio.h>

//#define SINGLE_STREAM
//#define DOUBLE_STREAM

#define N   (1024*1024)
#define FULL_DATA_SIZE   (N*20)


__global__ void kernel(int* a, int* b, int* c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        int idx1 = (idx + 1) % 256;
        int idx2 = (idx + 2) % 256;
        float   as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
        float   bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
        c[idx] = (as + bs) / 2;
    }
}


int main(void) {

    cudaDeviceProp  prop;
    int whichDevice;
    (cudaGetDevice(&whichDevice));
    (cudaGetDeviceProperties(&prop, whichDevice));
    if (!prop.deviceOverlap) {
        printf("Device will not handle overlaps, so no speed up from streams\n");
        return 0;
    }

    cudaEvent_t     start, stop;
    float           elapsedTime;

    cudaStream_t    stream0, stream1;// , stream2;
    int* host_a, * host_b, * host_c;

    //  GPU beffers for stream N
    int* dev_a0, * dev_b0, * dev_c0; // buffers
    int* dev_a1, * dev_b1, * dev_c1;
    //int* dev_a2, * dev_b2, * dev_c2;

    // start the timers
    (cudaEventCreate(&start));
    (cudaEventCreate(&stop));

    // initialize the stream
    (cudaStreamCreate(&stream0));
    (cudaStreamCreate(&stream1));
    //(cudaStreamCreate(&stream2));

    // allocate the memory on the GPU
    (cudaMalloc((void**)&dev_a0,
        N * sizeof(int)));
    (cudaMalloc((void**)&dev_b0,
        N * sizeof(int)));
    (cudaMalloc((void**)&dev_c0,
        N * sizeof(int)));

    (cudaMalloc((void**)&dev_a1,
        N * sizeof(int)));
    (cudaMalloc((void**)&dev_b1,
        N * sizeof(int)));
    (cudaMalloc((void**)&dev_c1,
        N * sizeof(int)));

    //(cudaMalloc((void**)&dev_a2,
    //    N * sizeof(int)));
    //(cudaMalloc((void**)&dev_b2,
    //    N * sizeof(int)));
    //(cudaMalloc((void**)&dev_c2,
    //    N * sizeof(int)));

    // allocate host locked memory, used to stream
    (cudaHostAlloc((void**)&host_a,
        FULL_DATA_SIZE * sizeof(int),
        cudaHostAllocDefault));
    (cudaHostAlloc((void**)&host_b,
        FULL_DATA_SIZE * sizeof(int),
        cudaHostAllocDefault));
    (cudaHostAlloc((void**)&host_c,
        FULL_DATA_SIZE * sizeof(int),
        cudaHostAllocDefault));

    for (int i = 0; i < FULL_DATA_SIZE; i++) {
        host_a[i] = rand();
        host_b[i] = rand();
    }

    (cudaEventRecord(start, 0));
    // now loop over full data, in bite-sized chunks
    for (int i = 0; i < FULL_DATA_SIZE; i += N * 2) {
        //1st work
        // copy the locked memory to the device, async
        (cudaMemcpyAsync(dev_a0, host_a + i,
            N * sizeof(int),
            cudaMemcpyHostToDevice,
            stream0));
        (cudaMemcpyAsync(dev_a1, host_a + i + N,
            N * sizeof(int),
            cudaMemcpyHostToDevice,
            stream1));

        //(cudaMemcpyAsync(dev_a2, host_a + i + N*2,
        //    N * sizeof(int),
        //    cudaMemcpyHostToDevice,
        //    stream2));


        (cudaMemcpyAsync(dev_b0, host_b + i,
            N * sizeof(int),
            cudaMemcpyHostToDevice,
            stream0));
        (cudaMemcpyAsync(dev_b1, host_b + i + N,
            N * sizeof(int),
            cudaMemcpyHostToDevice,
            stream1));
        //(cudaMemcpyAsync(dev_b2, host_b + i + N*2,
        //    N * sizeof(int),
        //    cudaMemcpyHostToDevice,
        //    stream2));

        kernel << <N / 256, 256, 0, stream0 >> > (dev_a0, dev_b0, dev_c0);

        kernel << <N / 256, 256, 0, stream1 >> > (dev_a1, dev_b1, dev_c1);

        //kernel << <N / 256, 256, 0, stream2 >> > (dev_a2, dev_b2, dev_c2);

        // copy the data from device to locked memory
        (cudaMemcpyAsync(host_c + i, dev_c0,
            N * sizeof(int),
            cudaMemcpyDeviceToHost,
            stream0));


        // copy the data from device to locked memory
        (cudaMemcpyAsync(host_c + i + N, dev_c1,
            N * sizeof(int),
            cudaMemcpyDeviceToHost,
            stream1));

        //(cudaMemcpyAsync(host_c + i + N, dev_c2,
        //    N * sizeof(int),
        //    cudaMemcpyDeviceToHost,
        //    stream2));

    }
    // copy result chunk from locked to full buffer
    (cudaStreamSynchronize(stream0));
    (cudaStreamSynchronize(stream1));
    //(cudaStreamSynchronize(stream2));

    (cudaEventRecord(stop, 0));

    (cudaEventSynchronize(stop));
    (cudaEventElapsedTime(&elapsedTime,
        start, stop));
    printf("Time taken:  %3.1f ms\n", elapsedTime);

    // cleanup the streams and memory
    (cudaFreeHost(host_a));
    (cudaFreeHost(host_b));
    (cudaFreeHost(host_c));
    (cudaFree(dev_a0));
    (cudaFree(dev_b0));
    (cudaFree(dev_c0));
    (cudaFree(dev_a1));
    (cudaFree(dev_b1));
    (cudaFree(dev_c1));
    //(cudaFree(dev_a2));
    //(cudaFree(dev_b2));
    //(cudaFree(dev_c2));
    (cudaStreamDestroy(stream0));
    (cudaStreamDestroy(stream1));
    //(cudaStreamDestroy(stream2));

    return 0;

}

*/

/*

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"

#include <stdio.h>

#define N   (1024*1024)
#define FULL_DATA_SIZE   (N*20)

__global__ void kernel(int* a, int* b, int* c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        int idx1 = (idx + 1) % 256;
        int idx2 = (idx + 2) % 256;
        float   as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
        float   bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
        c[idx] = (as + bs) / 2;
    }
}

int main(void) {

    cudaDeviceProp  prop;
    int whichDevice;
    cudaGetDevice(&whichDevice);
    cudaGetDeviceProperties(&prop, whichDevice);
    if (!prop.deviceOverlap) {
        printf("Device will not handle overlaps, so no speed up from streams\n");
        return 0;
    }

    cudaEvent_t     start, stop;
    float           elapsedTime;

    cudaStream_t    stream0, stream1, stream2, stream3;
    int* host_a, * host_b, * host_c;

    //  GPU buffers for streams
    int* dev_a0, * dev_b0, * dev_c0;
    int* dev_a1, * dev_b1, * dev_c1;
    int* dev_a2, * dev_b2, * dev_c2;
    int* dev_a3, * dev_b3, * dev_c3;

    // start the timers
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // initialize the streams
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);

    // allocate the memory on the GPU
    cudaMalloc((void**)&dev_a0, N * sizeof(int));
    cudaMalloc((void**)&dev_b0, N * sizeof(int));
    cudaMalloc((void**)&dev_c0, N * sizeof(int));

    cudaMalloc((void**)&dev_a1, N * sizeof(int));
    cudaMalloc((void**)&dev_b1, N * sizeof(int));
    cudaMalloc((void**)&dev_c1, N * sizeof(int));

    cudaMalloc((void**)&dev_a2, N * sizeof(int));
    cudaMalloc((void**)&dev_b2, N * sizeof(int));
    cudaMalloc((void**)&dev_c2, N * sizeof(int));

    cudaMalloc((void**)&dev_a3, N * sizeof(int));
    cudaMalloc((void**)&dev_b3, N * sizeof(int));
    cudaMalloc((void**)&dev_c3, N * sizeof(int));

    // allocate host locked memory, used to stream
    cudaHostAlloc((void**)&host_a, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_b, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_c, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);

    for (int i = 0; i < FULL_DATA_SIZE; i++) {
        host_a[i] = rand();
        host_b[i] = rand();
    }

    cudaEventRecord(start, 0);
    // now loop over full data, in bite-sized chunks
    for (int i = 0; i < FULL_DATA_SIZE; i += N) {
        int stream_idx = (i / N) % 4;

        cudaStream_t current_stream;
        int* dev_a;
        int* dev_b;
        int* dev_c;

        if (stream_idx == 0) {
            current_stream = stream0;
            dev_a = dev_a0;
            dev_b = dev_b0;
            dev_c = dev_c0;
        }
        else if (stream_idx == 1) {
            current_stream = stream1;
            dev_a = dev_a1;
            dev_b = dev_b1;
            dev_c = dev_c1;
        }
        else if (stream_idx == 2) {
            current_stream = stream2;
            dev_a = dev_a2;
            dev_b = dev_b2;
            dev_c = dev_c2;
        }
        else {
            current_stream = stream3;
            dev_a = dev_a3;
            dev_b = dev_b3;
            dev_c = dev_c3;
        }

        // copy the locked memory to the device, async
        cudaMemcpyAsync(dev_a, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, current_stream);
        cudaMemcpyAsync(dev_b, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, current_stream);

        kernel << <N / 256, 256, 0, current_stream >> > (dev_a, dev_b, dev_c);

        // copy the data from device to locked memory
        cudaMemcpyAsync(host_c + i, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost, current_stream);
    }

    // synchronize the streams
    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamSynchronize(stream3);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time taken:  %3.1f ms\n", elapsedTime);

    // cleanup the streams and memory
    cudaFreeHost(host_a);
    cudaFreeHost(host_b);
    cudaFreeHost(host_c);
    cudaFree(dev_a0);
    cudaFree(dev_b0);
    cudaFree(dev_c0);
    cudaFree(dev_a1);
    cudaFree(dev_b1);
    cudaFree(dev_c1);
    cudaFree(dev_a2);
    cudaFree(dev_b2);
    cudaFree(dev_c2);
    cudaFree(dev_a3);
    cudaFree(dev_b3);
    cudaFree(dev_c3);
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);

    return 0;
}


*/


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"

#include <stdio.h>

#define N   (1024*1024)
#define FULL_DATA_SIZE   (N*20)
#define NUM_STREAMS 2
#define CHUNK_SIZE (N / NUM_STREAMS)

__global__ void kernel(int* a, int* b, int* c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < CHUNK_SIZE) {
        int idx1 = (idx + 1) % 256;
        int idx2 = (idx + 2) % 256;
        float   as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
        float   bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
        c[idx] = (as + bs) / 2;
    }
}

int main(void) {

    cudaDeviceProp  prop;
    int whichDevice;
    cudaGetDevice(&whichDevice);
    cudaGetDeviceProperties(&prop, whichDevice);
    if (!prop.deviceOverlap) {
        printf("Device will not handle overlaps, so no speed up from streams\n");
        return 0;
    }

    cudaEvent_t     start, stop;
    float           elapsedTime;

    cudaStream_t    streams[NUM_STREAMS];
    int* host_a, * host_b, * host_c;

    //  GPU buffers for streams
    int* dev_a[NUM_STREAMS], * dev_b[NUM_STREAMS], * dev_c[NUM_STREAMS];

    // start the timers
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // initialize the streams and allocate the memory on the GPU
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
        cudaMalloc((void**)&dev_a[i], CHUNK_SIZE * sizeof(int));
        cudaMalloc((void**)&dev_b[i], CHUNK_SIZE * sizeof(int));
        cudaMalloc((void**)&dev_c[i], CHUNK_SIZE * sizeof(int));
    }

    // allocate host locked memory, used to stream
    cudaHostAlloc((void**)&host_a, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_b, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_c, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);

    for (int i = 0; i < FULL_DATA_SIZE; i++) {
        host_a[i] = rand();
        host_b[i] = rand();
    }

    cudaEventRecord(start, 0);
    // now loop over full data, in bite-sized chunks
    for (int i = 0; i < FULL_DATA_SIZE; i += N * 2)
    {
        int idx = (i / N) % NUM_STREAMS;
        (cudaMemcpyAsync(dev_a[idx], host_a + i,
            CHUNK_SIZE * sizeof(int),
            cudaMemcpyHostToDevice,
            streams[idx]));
        (cudaMemcpyAsync(dev_b[idx], host_b + i,
            CHUNK_SIZE * sizeof(int),
            cudaMemcpyHostToDevice,
            streams[idx]));

        kernel << <CHUNK_SIZE / 256, 256, 0, streams[idx] >> > (dev_a[idx], dev_b[idx], dev_c[idx]);

        cudaMemcpyAsync(host_c + i, dev_c[idx], CHUNK_SIZE * sizeof(int), cudaMemcpyDeviceToHost, streams[idx]);
    }

    // synchronize the streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time taken:  %3.1f ms\n", elapsedTime);

    printf("Number of Streams: %d\n", NUM_STREAMS);
    printf("Full Data Size: %d\n", FULL_DATA_SIZE);
    printf("Data Chunk Size (FULL_DATA_SIZE/(20 * STREAM_NUM)): %d\n", CHUNK_SIZE);
    printf("Time taken:  %3.1f ms\n", elapsedTime);

    // cleanup the streams and memory
    cudaFreeHost(host_a);
    cudaFreeHost(host_b);
    cudaFreeHost(host_c);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaFree(dev_a[i]);
        cudaFree(dev_b[i]);
        cudaFree(dev_c[i]);
        cudaStreamDestroy(streams[i]);
    }

    return 0;
}



