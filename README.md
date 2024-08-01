## Setup Environment
  - Visual Studio
  - CUDA
  - Intel one API toolkit
  - ImageJ : To view Raw images

--------------------


## C Programming

### HW1
- Implementation
    - Invert
    - Add two Images

### HW2
- Implementation
    - FlipX, Flipy
    - Image Transpose

### HW3
- Implementation
    - Max and Average Pooling
    - Matrix Multiply
    - Transpose

### HW4
- Implementation
    - Median Filter
        - vertical, horizontal
        - 3-tap, 5-tap, 3x3
    - RGB to GraySepia color Conversion

### HW5
- Implementation
    - 2D Convolution (with fixed point)
      - 5x5, 7x7, 9x9, 13x13, 15x15
      - Zero padding
      - Boundary Extension
      - Reflection
    - Sobel edge detection
      - SQRT and absolute summation
    - Boxcar filtering
    - SIMD Implementation - Add, FlipX, FlipY, median 3-tap

### HW6
- Implementation (SSE/AVX Programming)
  - Vertical FIR (16-tap)
  - Transpose
  - Average Pooling using streaming instruction
    - 2x2
  - Max Pooling using streaming instruction

-------------------


## Cuda programming
### HW7
- Implementation
  - Invert

### HW8
- Implementation : Different Number of Module and Block
  - Invert8
  - Add
  - FlipX
  - FlipY

### HW9
- Implementation
  - Transpose
    - With and Without 2D block mechanism
    - Different 2D thread Blocksize
  - Maxpool
    - 2D thread block
    - Differen 2D thread Blocksize
  - Median Filter
    - 3x3
    - 5-tap
    - With and Without Shared Memory

### HW10
- Implementation
  - 2D Convolution
    - Zero Padding
    - Boundary Padding
    - Mirroring
    - With Shared Memory
    - Using Gaussian and Boxcar Filter
    - Separable

### HW11
- Implementation
  - Histogram
    - With Shared Memory
### HW12
- Implementation
   - Improve the performance of the double stream program

