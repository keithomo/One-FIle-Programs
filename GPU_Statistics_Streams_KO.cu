/************************************************************************************
Program: 	GPU_Statistics_Streams_KO
Author:		Keith Omogrosso
Institution:National Radio Astronomy Observatory (Green Bank)
Date:		08/16/2016
Purpose:	To efficiently calculate mean, rms, skewness, and kurtosis from reformatted guppi data (one channel only). Input file can be obtained by running rcog_read_gupp_alone.cc for one channel. Output file  contains mean, rms, skewness, and kurtosis for a predifined time period of .01048576 s. Constant variable are not meant to be changed.
-------------
Modified by:	Keith Omogrosso
Date Modified:	12/27/2016
Modification:	Input can be modified. The input of this code has to be uint8_t. This performs on 4 sets of data at the same time. It was designed to perform statistics for 4 polarizations of light. Each value from each polarization came as a cluster of 4. This is why the structures of char4 and float2 were used as variable types. So, incoming data needs to be spaced every 4 bytes, each a byte long. If each of these Dataset variables were one byte a piece, the file might look like this in bytes: [Dataset1] [Dataset2] [Dataset3] [Dataset4] [Dataset1] [Dataset2] [Dataset3] [Dataset4] etc...   Also, this program takes a large data set and breaks it into smaller chunks to process in parallel instead of processing the whole file sequentially. It cannot break the input into many chunks. About 16 is the limit I have found for a decent NVIDIA GPU. 
*************************************************************************************/

// To run: $ [program name] 1

#include<cuda.h> 
#include<stdio.h>
#include <memory.h>
#include <string.h>
#include <cufft.h>
#include <cuComplex.h>
#include <stdlib.h>
#include <iostream>
#include <unistd.h>
#include "../../../Cudapractice/common/book.h"
#define FFT_SIZE 512 
#define NPOL 4
#define CUDA_CALL(call) \
do { \
    cudaError_t err = call; \
    if (cudaSuccess != err) { \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.", \
                 __FILE__, __LINE__, cudaGetErrorString(err) ); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUFFT_CALL(call) \
do { \
    cufftResult status = call; \
    if (status != CUFFT_SUCCESS) \
    { \
        printf("cufft Error %d on line %d\n", status, __LINE__); \
    } \
    } while(0)
const int samp_T0 = 65536; // number of samples in one time resolution for statistics
const int nT0 = 8; // cannot go above 16 because streams > 16 = slower in this situation.
const int hist_size = 256;
const int nMom = 3;

__global__ 
void power_arr(char4 *in, float2 *m2, int shift)
{ 
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x + shift*samp_T0;
	// calculate real^2 + imaginary^2  = power
    m2[idx].x = ((float)in[idx].x) * ((float)in[idx].x) + ((float)in[idx].y) * ((float)in[idx].y);
    m2[idx].y = ((float)in[idx].z) * ((float)in[idx].z) + ((float)in[idx].w) * ((float)in[idx].w);
}

__global__
void max_min(float2 *m2, int2 *maximum, int2 *minimum, float2 *resizer, int shift)
{	
	int idx = threadIdx.x * blockDim.x + shift*samp_T0;
	int threadMx = 0;
	int threadMy = 0;
	int threadmx = 0;
	int threadmy = 0;
	for (int ii = 0; ii < samp_T0/blockDim.x; ii++){ // fastest method
		if (threadMx < m2[idx + ii].x){threadMx = (int)m2[idx + ii].x;}
		if (threadMy < m2[idx + ii].y){threadMy = (int)m2[idx + ii].y;}
		if (threadmx > m2[idx + ii].x){threadmx = (int)m2[idx + ii].x;}
		if (threadmy > m2[idx + ii].y){threadmy = (int)m2[idx + ii].y;}
	}
	__syncthreads();
	atomicMax( &maximum[shift].x, threadMx);
	atomicMax( &maximum[shift].y, threadMy);
	atomicMin( &minimum[shift].x, threadmx);
	atomicMin( &minimum[shift].y, threadmy);
	__syncthreads();
	if (idx == shift*samp_T0){ //1.001 just to make division work later
		resizer[shift].x = ((float)(maximum[shift].x - minimum[shift].x)/hist_size)*1.001; 
		resizer[shift].y = ((float)(maximum[shift].y - minimum[shift].y)/hist_size)*1.001; 
	}
}

__global__
void mean_npol(float2 *m2, float2 *mean, int shift) 
{
	float mx = 0, my = 0;
	int idx = threadIdx.x * blockDim.x + shift*samp_T0;
	for (int ii = 0; ii < (samp_T0/blockDim.x); ii++){
		mx += m2[idx + ii].x;
		my += m2[idx + ii].y;
	}
	__syncthreads();
	atomicAdd( &mean[shift].x, mx);
	atomicAdd( &mean[shift].y, my);
	__syncthreads();
	if (idx == shift*samp_T0){
		mean[shift].x /= (samp_T0); 
		mean[shift].y /= (samp_T0); 
	}
}

__global__  
void histo_kernel( float2 *m2, int2 *histogram, int2 *minimum, float2 *resizer, int shift)
{
	__shared__ int2 tempH[hist_size];
	tempH[threadIdx.x] = make_int2(0., 0.);
	__syncthreads();
	int idx = threadIdx.x + (blockIdx.x*blockDim.x) + shift*samp_T0;
	int offset = blockDim.x*gridDim.x;
	while(idx < (1+shift)*samp_T0){
		atomicAdd(&tempH[(int)((m2[idx].x-minimum[shift].x)/resizer[shift].x)].x,1); 
		atomicAdd(&tempH[(int)((m2[idx].y-minimum[shift].y)/resizer[shift].y)].y,1); 
		idx += offset;
	}	
	__syncthreads();
	atomicAdd( &(histogram[threadIdx.x + shift*hist_size].x), tempH[threadIdx.x].x);
	atomicAdd( &(histogram[threadIdx.x + shift*hist_size].y), tempH[threadIdx.x].y);
}

__global__ 
void moment_order(float2 *mean, int2 *histogram, float2 *moment, int2 *minimum, float2 *resizer, int shift)
{
	int order = threadIdx.x; // 0 = moment 2; 1 = moment 3; 2 = moment 4.
	for (int ii = 0; ii < hist_size; ii++){
		moment[order + shift*3].x += ((float)( powf( ((ii+minimum[shift].x)*resizer[shift].x) - mean[shift].x, order+2)) * histogram[ii + shift*hist_size].x); 
		moment[order + shift*3].y += ((float)( powf( ((ii+minimum[shift].y)*resizer[shift].y) - mean[shift].y, order+2)) * histogram[ii + shift*hist_size].y); 
	}
	moment[order + shift*3].x /= (samp_T0);
	moment[order + shift*3].y /= (samp_T0);
}

/********************************************************/
/************************* MAIN *************************/
/********************************************************/

int main(int argc, char *argv[])
{	
	cudaError_t cudaError;
	FILE *fp_in;
	FILE *fp_out;
	char filename[59] = "../rcog/red_gup.dat";
	int outlet = 1;
	dim3 hgrd(16);
	dim3 pwrgrd(samp_T0/hist_size);
	dim3 hist(hist_size);
	dim3 grd(NPOL);

	// memory
	char4 *h_input;
	float2 *h_m2;
	char4 *d_input;
	float2 *d_m2;
	float2 *d_mean;
	int2 *d_maximum;
	int2 *d_minimum;
	float2 *d_resizer; // sizes the histogram and calibrates statistics. 
	int2 *d_histogram;
	float2 *d_moment;
	float2 *h_mean;
	float2 *h_moment;
	int2 *h_histogram;
	int2 *h_maximum;
	struct rsk {
		float2 rms;
		float2 skew;
		float2 kurt;
	};
	struct rsk h_rsk[nT0];

	// Host Allocation (page-locked)
	CUDA_CALL(cudaHostAlloc((void**) &h_input, nT0*samp_T0*sizeof(char4), cudaHostAllocDefault));
	CUDA_CALL(cudaHostAlloc((void**) &h_m2, nT0*samp_T0*sizeof(float2), cudaHostAllocDefault));
	CUDA_CALL(cudaHostAlloc((void**) &h_mean, nT0*sizeof(float2), cudaHostAllocDefault));
	CUDA_CALL(cudaHostAlloc((void**) &h_moment, nT0*nMom*sizeof(float2), cudaHostAllocDefault));
	CUDA_CALL(cudaHostAlloc((void**) &h_histogram, nT0*hist_size*sizeof(int2), cudaHostAllocDefault));
	CUDA_CALL(cudaHostAlloc((void**) &h_maximum, nT0*sizeof(int2), cudaHostAllocDefault));

	// Device Allocation
	CUDA_CALL(cudaMalloc((void**) &d_input, nT0*samp_T0*sizeof(char4)));
	CUDA_CALL(cudaMalloc((void**) &d_m2, nT0*samp_T0*sizeof(float2)));
	CUDA_CALL(cudaMalloc((void**) &d_mean, nT0*sizeof(float2)));
	CUDA_CALL(cudaMalloc((void**) &d_maximum, nT0*sizeof(int2)));
	CUDA_CALL(cudaMalloc((void**) &d_minimum, nT0*sizeof(int2)));
	CUDA_CALL(cudaMalloc((void**) &d_resizer, nT0*sizeof(float2)));
	CUDA_CALL(cudaMalloc((void**) &d_histogram, nT0*hist_size*sizeof(int2)));
	CUDA_CALL(cudaMalloc((void**) &d_moment, nT0*nMom*sizeof(float2)));
	
	// time keeps
	float elapsedTime=0, elapsedTimeK=0, elapsedTimeM=0, elapsedTimeO=0;
	static cudaEvent_t start_mem, stop_mem;
	static cudaEvent_t start_ker, stop_ker;
	static cudaEvent_t start_o, stop_o;
	cudaEventCreate(&start_mem);
	cudaEventCreate(&stop_mem);
	cudaEventCreate(&start_ker);
	cudaEventCreate(&stop_ker);
	cudaEventCreate(&start_o);
	cudaEventCreate(&stop_o);

/*
What this program is going to do first. First make it so it only runs one packet of data for the given time resolution. Then increase the number of packets to maximum load and then test to see the speed improvements. 
*/

/////////////////////////////// command line args ////////////////////////////////////////////

	if (argc>=2) { sscanf( argv[1], "%d", &outlet );} 
	else {
      	printf("FATAL: main(): command line argument <outlet> not specified\n");
      	return 0;
	}
	if (outlet != 0 &&outlet != 1 && outlet !=2 && outlet != 3){printf("outlet err\n");return 0;} 
  	printf("outlet = %d\n",outlet);

////////////////////////////////// main loop ///////////////////////////////////////////////

	if (!(fp_in = fopen(filename,"r"))){
		printf("FATAL: main(); couldn't open file'\n");
		return -1;
	}if (!(fp_out = fopen("rcog_stats_test_out.dat","w"))){
		printf("FATAL: main(); couldn't open file'\n");
		return -1;
	}

	cudaStream_t stream[nT0];

	int ii = 0;
	while(!feof(fp_in)){

		// read file
		cudaEventRecord(start_mem, 0);
		//fseek(fp_in, 1024*samp_T0*sizeof(char4), SEEK_SET); // sample number 1025
		fread( h_input, nT0*samp_T0*sizeof(char4), 1, fp_in);
		cudaEventRecord(stop_mem, 0);
		cudaEventSynchronize(stop_mem);
		cudaEventRecord(start_ker, 0);
		CUDA_CALL(cudaMemset(d_m2, 0, nT0*samp_T0*sizeof(float2)));
		CUDA_CALL(cudaMemset(d_moment, 0, nT0*nMom*sizeof(float2)));
		CUDA_CALL(cudaMemset(d_mean, 0, nT0*sizeof(float2)));
		CUDA_CALL(cudaMemset(d_histogram, 0, nT0*hist_size*sizeof(int2)));
		CUDA_CALL(cudaMemset(d_maximum, 0, nT0*sizeof(int2)));
		CUDA_CALL(cudaMemset(d_minimum, 0, nT0*sizeof(int2)));
		CUDA_CALL(cudaMemset(d_resizer, 0, nT0*sizeof(float2)));
		memset(h_rsk, 0, nT0*sizeof(struct rsk));

		for (int shift = 0; shift < nT0; shift++){ // start streams
			CUDA_CALL(cudaStreamCreate( &stream[shift]));
			CUDA_CALL(cudaMemcpyAsync(d_input + shift*samp_T0, h_input + shift*samp_T0, samp_T0*sizeof(char4), cudaMemcpyHostToDevice, stream[shift]));
			power_arr<<<pwrgrd,hist,0,stream[shift]>>>(d_input, d_m2, shift);
			max_min<<<1,256,0,stream[shift]>>>(d_m2, d_maximum, d_minimum, d_resizer, shift);
			mean_npol<<<1,256,0,stream[shift]>>>(d_m2, d_mean, shift); 
			histo_kernel<<<hgrd,hist,0,stream[shift]>>>(d_m2, d_histogram, d_minimum, d_resizer, shift);
			moment_order<<<1,3,0,stream[shift]>>>(d_mean, d_histogram, d_moment, d_minimum, d_resizer, shift);
		}
		for (int shift = 0; shift < nT0; shift++){
			cudaStreamSynchronize( stream[shift]);
			cudaMemcpyAsync(h_mean + shift, d_mean + shift, sizeof(float2), cudaMemcpyDeviceToHost, stream[shift]);
			cudaMemcpyAsync(h_maximum + shift, d_maximum + shift, sizeof(int2), cudaMemcpyDeviceToHost, stream[shift]);
			cudaMemcpyAsync(h_histogram + shift*hist_size, d_histogram + shift*hist_size, hist_size*sizeof(int2), cudaMemcpyDeviceToHost, stream[shift]);
			cudaMemcpyAsync(h_moment + shift*nMom, d_moment + shift*nMom, nMom*sizeof(float2), cudaMemcpyDeviceToHost, stream[shift]);
			cudaStreamSynchronize( stream[shift]);
		}

		cudaEventRecord(stop_ker,0);
		cudaEventSynchronize(stop_ker);

		//printf("\nOn the cpu now\n");
		for (int nn = 0; nn < nT0; nn++){ // calc for multiple streams
			printf("\nshift %d:::\n",nn);
			h_rsk[nn].rms.x = sqrt(abs(h_moment[nn*3+0].x));
			h_rsk[nn].rms.y = sqrt(abs(h_moment[nn*3+0].y));
			h_rsk[nn].skew.x = h_moment[nn*3+1].x/(pow(h_moment[nn*3+0].x, 1.5));
			h_rsk[nn].skew.y = h_moment[nn*3+1].y/(pow(h_moment[nn*3+0].y, 1.5));
			h_rsk[nn].kurt.x = (h_moment[nn*3+2].x/(pow(h_moment[nn*3+0].x, 2))) -3;
			h_rsk[nn].kurt.y = (h_moment[nn*3+2].y/(pow(h_moment[nn*3+0].y, 2))) -3;

			int2 histocount = make_int2(0., 0.);
			for (int oo=nn*hist_size; oo<((nn+1)*hist_size); oo++){
				histocount.x += h_histogram[oo].x;
				histocount.y += h_histogram[oo].y;
			} //printf("histocount %d\n", histocount.x); 
			//if (histocount.x != samp_T0){printf("histogram.x out of range\n");return 0;}
			//if (histocount.y != samp_T0){printf("histogram.y out of range\n");return 0;}
		}

		cudaEventRecord(start_o, 0);
		if (outlet == 1){
			for (int nn = 0; nn < nT0; nn++){ // display for multiple streams
				printf("M.x = %f; ", h_mean[nn].x);
				printf("M.y = %f; \n", h_mean[nn].y); 
				printf("^.x = %d; ", h_maximum[nn].x);
				printf("^.y = %d; \n", h_maximum[nn].y);
				printf("R.x = %f; ", h_rsk[nn].rms.x);
				printf("R.y = %f; \n", h_rsk[nn].rms.y);
				printf("S.x = %f; ", h_rsk[nn].skew.x);
				printf("S.y = %f; \n", h_rsk[nn].skew.y);
				printf("K.x = %f; ", h_rsk[nn].kurt.x);
				printf("K.y = %f; \n", h_rsk[nn].kurt.y);
				for (int kk = 0; kk < hist_size; kk++){
					fprintf(stdout, "%d ", h_histogram[kk + nn*hist_size].y);
				}fprintf(stdout, "\nThe other one\n\n");
			}
		}
		if (outlet == 2){
			for (int ee = 0; ee < nT0; ee++){ 
				fwrite(&h_mean[ee], sizeof(float2), 1, fp_out); 
				fwrite(&h_maximum[ee], sizeof(int2), 1, fp_out);
				fwrite(&h_rsk[ee], sizeof(struct rsk), 1, fp_out);
			}
		}

		cudaEventRecord(stop_o, 0);
		cudaEventSynchronize(stop_o);
		cudaEventElapsedTime( &elapsedTime, start_ker, stop_ker);
		elapsedTimeK += elapsedTime;
		cudaEventElapsedTime( &elapsedTime, start_mem, stop_mem);
		elapsedTimeM += elapsedTime;
		cudaEventElapsedTime( &elapsedTime, start_o, stop_o);
		elapsedTimeO += elapsedTime;
		
	ii++;
	if (ii > 99){break;}
	}

	printf("\nKernel calls took: %3.1f ms\n", elapsedTimeK);
	printf("All Memory transfers took: %3.1f ms\n", elapsedTimeM);
	printf("Writing to files and calculations took: %3.1f ms\n", elapsedTimeO);

	cudaEventDestroy(stop_mem);
	cudaEventDestroy(start_ker);
	cudaEventDestroy(stop_ker);
	cudaStreamDestroy(*stream);

	cudaFree(d_resizer);
	cudaFree(d_m2);
	cudaFree(d_input);
	cudaFree(d_mean);
	cudaFree(d_maximum);
	cudaFree(d_minimum);
	cudaFree(d_histogram);
	cudaFree(d_moment);
	cudaFreeHost(h_input);
	cudaFreeHost(h_mean);
	cudaFreeHost(h_histogram);
	cudaFreeHost(h_moment);
	cudaFreeHost(h_m2);
	cudaFreeHost(h_maximum);

	fclose(fp_in);
	fclose(fp_out);
	return 0;
}

