// rgbtogssse.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
// Load an image and save it in PNG and JPG format using stb_image libraries
#include <stdio.h>
#include <stdlib.h>
#include "Timer.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


int main(void) {

    float dCpuTime = 0.0;
    int loopcount;
    

    int width, height, channels;
    unsigned char* img = stbi_load("sky.jpg", &width, &height, &channels, 0);
    //unsigned char* img = stbi_load("Shapes.png", &width, &height, &channels, 0);
    if (img == NULL) {
        printf("Error in loading the image\n");
        exit(1);
    }
    printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n", width, height, channels);


    // Convert the input image to gray
    int img_size = width * height * channels;
    int gray_channels = channels == 4 ? 2 : 1;
    int gray_img_size = width * height * gray_channels;

    printf("gray_channels = % d, channels = % d \n", gray_channels, channels);
    unsigned char* gray_img = new unsigned char[gray_img_size];
    if (gray_img == NULL) {
        printf("Unable to allocate memory for the gray image.\n");
        exit(1);
    }

    int i;
    unsigned char* p, * pg;
    p = (unsigned char*)img;
    pg = (unsigned char*)gray_img;

    printf("image size = %d\n", img_size);

    printf("\n------gray image--------\n");

    CPerfCounter counter;
    counter.Reset();
    counter.Start();
    for (loopcount = 0; loopcount < 1; loopcount++) {
        for (i = 0; i < img_size; i += channels) {
            *pg = (unsigned char)((*p + *(p + 1) + *(p + 2)) / 3);
            //printf(" i=%d pg= %d \n", i, *pg);

            if (channels == 4) {
                *(pg + 1) = *(p + 3);
            }
            p += channels;
            pg += gray_channels;
        }
    }
    counter.Stop();
    dCpuTime += counter.GetElapsedTime();
    printf("Fixed-point Performance (ms) = %f \n", dCpuTime * 1000.0);
    stbi_write_jpg("sky_gray1.jpg", width, height, gray_channels, gray_img, 100);

    p = (unsigned char*)img;
    pg = (unsigned char*)gray_img;
   

    counter.Reset();
    counter.Start();
    for (loopcount = 0; loopcount < 1; loopcount++) {
        for (i = 0; i < img_size; i += channels) {
            *pg = (unsigned char)((*p + *(p + 1) + *(p + 2)) / 3.0);
            //printf(" i=%d pg= %d \n", i, *pg);

            if (channels == 4) {
                *(pg + 1) = *(p + 3);
            }
            p += channels;
            pg += gray_channels;
        }
    }
    counter.Stop();
    dCpuTime += counter.GetElapsedTime();
    printf("Floating-point Performance (ms) = %f \n", dCpuTime * 1000.0);



    stbi_write_jpg("sky_gray2.jpg", width, height, gray_channels, gray_img, 100);
   // stbi_write_png("Shapes_gray.png", width, height, gray_channels, gray_img, width * gray_channels);

    // Convert the input image to sepia
    unsigned char* sepia_img = new unsigned char[img_size];
    if (sepia_img == NULL) {
        printf("Unable to allocate memory for the sepia image.\n");
        exit(1);
    }
    p = (unsigned char*)img; 
    pg = (unsigned char*)sepia_img;

    const int bit8 = 8;
    
    printf("\n------sefia image--------\n");
    counter.Reset();
    counter.Start();
    for (loopcount = 0; loopcount < 1; loopcount++) {
        for (i = 0; i < img_size / channels; i++) {
            int sepia_red_r = (393 << bit8) / 1000;
            int sepia_red_g = (769 << bit8) / 1000;
            int sepia_red_b = (180 << bit8) / 1000;

            int sepia_green_r = (349 << bit8) / 1000;
            int sepia_green_g = (686 << bit8) / 1000;
            int sepia_green_b = (168 << bit8) / 1000;

            int sepia_blue_r = (272 << bit8) / 1000;
            int sepia_blue_g = (534 << bit8) / 1000;
            int sepia_blue_b = (131 << bit8) / 1000;


            uint16_t red = 0;
            uint16_t green = 0;
            uint16_t blue = 0;
            

            for (i = 0; i < img_size / channels; i++) {

                red = ((sepia_red_r * *p + sepia_red_g * *(p + 1) + sepia_red_b * *(p + 2)) >> bit8);
                green = ((sepia_green_r * *p + sepia_green_g * *(p + 1) + sepia_green_b * *(p + 2)) >> bit8);
                blue = ((sepia_blue_r * *p + sepia_blue_g * *(p + 1) + sepia_blue_b * *(p + 2)) >> 8);
                *pg = (unsigned char)(red > 255 ? 255 : red);
                *(pg + 1) = (unsigned char)(green > 255 ? 255 : green);
                *(pg + 2) = (unsigned char)(blue > 255 ? 255 : blue);

                if (channels == 4) {
                    *(pg + 3) = *(p + 3);
                }
                p += channels;
                pg += channels;
            }
        }
    }
    counter.Stop();
    dCpuTime += counter.GetElapsedTime();
    printf("Fixed-point Performance (ms) = %f \n", dCpuTime * 1000.0);
    stbi_write_jpg("sky_sepia1.jpg", width, height, channels, sepia_img, 100);


    p = (unsigned char*)img;
    pg = (unsigned char*)sepia_img;

    counter.Reset();
    counter.Start();
    for (loopcount = 0; loopcount < 1; loopcount++) {
        for (i = 0; i < img_size / channels; i++) {
            *pg = (unsigned char)fmin(0.393 * *p + 0.769 * *(p + 1) + 0.189 * *(p + 2), 255.0);         // red
            *(pg + 1) = (unsigned char)fmin(0.349 * *p + 0.686 * *(p + 1) + 0.168 * *(p + 2), 255.0);   // green
            *(pg + 2) = (unsigned char)fmin(0.272 * *p + 0.534 * *(p + 1) + 0.131 * *(p + 2), 255.0);   // blue        
            if (channels == 4) {
                *(pg + 3) = *(p + 3);
            }
            p += channels;
            pg += channels;
        }
        
    }
    counter.Stop();
    dCpuTime += counter.GetElapsedTime();
    printf("Floating-point Performance (ms) = %f \n", dCpuTime * 1000.0);

   
    stbi_write_jpg("sky_sepia2.jpg", width, height, channels, sepia_img, 100);

    stbi_image_free(img);
    delete[] gray_img;
    delete[] sepia_img;
}


