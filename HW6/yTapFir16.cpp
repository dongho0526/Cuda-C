
/* Include headers */
#include "fir16.h"



void imagePad(
	  unsigned short *src_ptr,
	  unsigned short *dst_ptr,
	  int xSize,
	  int ySize,
	  int kernel_size)
{

	int x_count, y_count, f_count;
	int accm;

	// Pad top
	for(y_count=0; y_count< kernel_size/2 ; y_count++) {

		for (x_count = 0; x_count < xSize; x_count++){
			
			*(dst_ptr + (y_count) *  xSize + x_count) = *(src_ptr  + x_count) ;

		}

	}

	//Copy original image
	for(y_count=0; y_count< ySize; y_count++) {

		for (x_count = 0; x_count < xSize; x_count++){
			
			*(dst_ptr + (y_count+kernel_size/2) *  xSize + x_count) = *(src_ptr + y_count *  xSize + x_count) ;
		}

	}

	//Pad bottome
	for(y_count=0; y_count< kernel_size/2 ; y_count++) {

		for (x_count = 0; x_count < xSize; x_count++){
			
			*(dst_ptr + (y_count +ySize ) *  xSize + x_count) = *(src_ptr + ySize*xSize + x_count) ;
		}

	}

}


void fir16_tight_loop_C(
	  unsigned short *src_ptr,
	  unsigned short *dst_ptr,
	  int xSize,
	  int ySize,
	  int* kernel_ptr,
	  int kernel_size,
	  int scale)
{

	int x_count, y_count, f_count;
	int accm;
	for(y_count=0; y_count< ySize; y_count++) {

		for (x_count = 0; x_count < xSize; x_count++){
			accm = 0;
			for (f_count = 0; f_count < kernel_size; f_count++){
				accm += *(src_ptr + (y_count+f_count) *  xSize + x_count) * kernel_ptr[f_count] ;
			}
			accm >>= scale;
			if (accm < 0)
				accm = 0;
			if (accm > 32768)
				accm = 32768;

			*(dst_ptr + y_count *  xSize + x_count) = (unsigned short) accm;
		}

	}
}

void yfir16_tight_loop_SIMD(
	  unsigned short *src_ptr,
	  unsigned short *dst_ptr,
	  int xSize,
	  int ySize,
	  int* kernel_ptr,
	  int kernel_size,
	  int scale)
{
	int y_count, x_count, k_count;

	int i;
	 int count;

	__m128i* x128_dst_ptr;

	__m128i output_x15_x12, output_x11_x8, output_x7_x4, output_x3_x0;
	__m128i output_x7_x0, output_x15_x8;

	__m128i x7_x0, x15_x8, x15_x12,x11_x8,x7_x4,x3_x0 ;	
	__m128i k_coef[MAX_FILTER_SIZE];
	__m128i acc_x15_x12, acc_x11_x8, acc_x7_x4, acc_x3_x0;

	__m128i scale128, zero128;

	__m128i max32767, mul_x15_x12, mul_x11_x8,mul_x7_x4,mul_x3_x0;

	__m128i const* x128_src_ptr;

	//set coef value
	scale128 = _mm_cvtsi32_si128(scale);
	zero128 = _mm_setzero_si128();

	max32767 = _mm_set1_epi16(32767);


	/******************************************
	* Axial filter process
	******************************************/

	
	//printf("thread=%d \n", thread);
	for (i=0; i<kernel_size; i++)
	{			
		int temp = kernel_ptr[i];			
		k_coef[i] = _mm_set_epi32(temp, temp,	temp, temp);
	}
	
	for(y_count=0; y_count< ySize; y_count++) {

		for (x_count = 0; x_count < xSize; x_count+=8)
		{
			x128_dst_ptr = (__m128i *)(dst_ptr + y_count*xSize+ x_count);
			acc_x15_x12 = zero128;
			acc_x11_x8 = zero128;
			acc_x7_x4 = zero128;
			acc_x3_x0 = zero128;


			for (k_count = 0; k_count < kernel_size; k_count++)
			{
				x128_src_ptr =(__m128i*) (src_ptr + (y_count+k_count)*xSize + x_count); 
				x7_x0 = _mm_loadu_si128(x128_src_ptr);
								
				//unpack
				x3_x0 = _mm_unpacklo_epi16(x7_x0, zero128);
				x7_x4 = _mm_unpackhi_epi16(x7_x0, zero128);
				
				//multiply
				mul_x3_x0 = _mm_mullo_epi32(x3_x0, k_coef[k_count]);
				mul_x7_x4 = _mm_mullo_epi32(x7_x4, k_coef[k_count]);
				
//acc
				acc_x3_x0 = _mm_add_epi32(acc_x3_x0, mul_x3_x0);
				acc_x7_x4 = _mm_add_epi32(acc_x7_x4, mul_x7_x4);


			} // end of l


			//output preparation - shift
			output_x3_x0 = _mm_srl_epi32(acc_x3_x0, scale128);
			output_x7_x4 = _mm_srl_epi32(acc_x7_x4, scale128);
			/*
			output_x3_x0 = _mm_min_epi32(output_x3_x0, max32767);
			output_x7_x4 = _mm_min_epi32(output_x7_x4, max32767);
			*/
			// output
			output_x7_x0 = _mm_packus_epi32(output_x3_x0 , output_x7_x4);
			
			//output_x7_x0 = _mm_max_epi16(output_x7_x0, zero128);
			_mm_storeu_si128(x128_dst_ptr, output_x7_x0);
			
		}  //end of j	

	}

}





void xfir16_tight_loop_SIMD(
	unsigned short* src_ptr,
	unsigned short* dst_ptr,
	int xSize,
	int ySize,
	int* kernel_ptr,
	int kernel_size,
	int scale)
{
	int y_count, x_count, k_count;
	int i;

	__m128i* x128_dst_ptr;

	__m128i x15_x0;
	__m128i k_coef[MAX_FILTER_SIZE];
	__m128i acc_x15_x0;

	__m128i scale128, zero128;

	__m128i const* x128_src_ptr;

	// Set coefficients
	scale128 = _mm_cvtsi32_si128(scale);
	zero128 = _mm_setzero_si128();

	// Load filter coefficients
	for (i = 0; i < kernel_size; i++) {
		int temp = kernel_ptr[i];
		k_coef[i] = _mm_set1_epi16(temp);  // Set 16-bit values
	}

	for (y_count = 0; y_count < ySize; y_count++) {
		for (x_count = 0; x_count < xSize; x_count += 8) {
			x128_dst_ptr = (__m128i*)(dst_ptr + y_count * xSize + x_count);
			acc_x15_x0 = zero128;

			for (k_count = 0; k_count < kernel_size; k_count++) {
				int x_offset = x_count + k_count - kernel_size / 2;
				if (x_offset < 0 || x_offset >= xSize) continue;

				x128_src_ptr = (const __m128i*)(src_ptr + y_count * xSize + x_offset);
				x15_x0 = _mm_loadu_si128(x128_src_ptr);

				// Multiply and accumulate
				__m128i mul_lo = _mm_mullo_epi16(x15_x0, k_coef[k_count]);
				acc_x15_x0 = _mm_add_epi16(acc_x15_x0, mul_lo);
			}

			// Output preparation - shift right by scale
			__m128i output_x15_x0 = _mm_srai_epi16(acc_x15_x0, scale);

			// Saturate to [0, 65535]
			output_x15_x0 = _mm_packus_epi16(output_x15_x0, zero128);

			// Store the result
			_mm_storeu_si128(x128_dst_ptr, output_x15_x0);
		}
	}
}







void yfir8_tight_loop_SIMD(
	unsigned char* src_ptr,
	unsigned char* dst_ptr,
	int xSize,
	int ySize,
	int* kernel_ptr,
	int kernel_size,
	int scale)
{
	int y_count, x_count, k_count;
	int i;

	__m128i* x128_dst_ptr;
	__m128i output_x15_x0;
	__m128i x15_x0;
	__m128i k_coef[MAX_FILTER_SIZE];
	__m128i acc_x15_x0;
	__m128i scale128, zero128, max255;
	__m128i mul_x15_x0;

	const __m128i* x128_src_ptr;

	// Set coefficients
	scale128 = _mm_cvtsi32_si128(scale);
	zero128 = _mm_setzero_si128();
	max255 = _mm_set1_epi8(255);

	// Load filter coefficients
	for (i = 0; i < kernel_size; i++) {
		int temp = kernel_ptr[i];
		k_coef[i] = _mm_set1_epi16(temp);  // Set 16-bit values
	}

	for (y_count = 0; y_count < ySize; y_count++) {
		for (x_count = 0; x_count < xSize; x_count += 16) {
			x128_dst_ptr = (__m128i*)(dst_ptr + y_count * xSize + x_count);
			acc_x15_x0 = zero128;

			for (k_count = 0; k_count < kernel_size; k_count++) {
				x128_src_ptr = (const __m128i*)(src_ptr + (y_count + k_count) * xSize + x_count);
				x15_x0 = _mm_loadu_si128(x128_src_ptr);

				// Unpack 8-bit integers to 16-bit integers for multiplication
				__m128i x7_x0_lo = _mm_unpacklo_epi8(x15_x0, zero128);
				__m128i x7_x0_hi = _mm_unpackhi_epi8(x15_x0, zero128);

				// Multiply and accumulate
				__m128i mul_lo = _mm_mullo_epi16(x7_x0_lo, k_coef[k_count]);
				__m128i mul_hi = _mm_mullo_epi16(x7_x0_hi, k_coef[k_count]);

				acc_x15_x0 = _mm_add_epi16(acc_x15_x0, mul_lo);
				acc_x15_x0 = _mm_add_epi16(acc_x15_x0, mul_hi);
			}

			// Output preparation - shift right by scale
			output_x15_x0 = _mm_srai_epi16(acc_x15_x0, scale);

			// Saturate to [0, 255]
			output_x15_x0 = _mm_packus_epi16(output_x15_x0, zero128);

			// Store the result
			_mm_storeu_si128(x128_dst_ptr, output_x15_x0);
		}
	}
}