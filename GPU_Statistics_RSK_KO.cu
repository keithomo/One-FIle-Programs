/************************************************************************************
Program: 	GPU_Statistics_RSK_KO
Author:		Keith Omogrosso
Institution:National Radio Astronomy Observatory (Green Bank)
Date:		08/16/2016
Purpose:	To efficiently calculate mean, rms, skewness, and kurtosis from reformatted guppi data (one channel only). Input file can be obtained by running rcog_read_gupp_alone.cc for one channel. Output file  contains mean, rms, skewness, and kurtosis for a predifined time period of .01048576 s. Constant variable are not meant to be changed.
-------------
Modified by:	Keith Omogrosso
Date Modified:	12/27/2016
Modification:	Input can be modified. The input of this code has to be uint8_t. This performs on 4 sets of data at the same time. It was designed to perform statistics for 4 polarizations of light. Each value from each polarization came as a cluster of 4. This is why the structures of char4 and float2 were used as variable types. So, incoming data needs to be spaced every 4 bytes, each a byte long. If each of these Dataset variables were one byte a piece, the file might look like this in bytes: [Dataset1] [Dataset2] [Dataset3] [Dataset4] [Dataset1] [Dataset2] [Dataset3] [Dataset4] etc...
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
#include "../Cudapractice/common/book.h"
#define FFT_SIZE 512 
#define NPOL 4

const int samp_T0 = 65536; // number of samples in one time resolution for statistics
const int nT0 = 1024;
const int hist_size = 256;

struct rsk {
	float2 rms;
	float2 skew;
	float2 kurt;
};

// (1 / 6)
__global__
void power_arr(char4 *in, float2 *m2)
{ // threads*blocks = samples.
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	// calculate real^2 + imaginary^2  = power
    m2[idx].x = ((float)in[idx].x) * ((float)in[idx].x) + ((float)in[idx].y) * ((float)in[idx].y);
    m2[idx].y = ((float)in[idx].z) * ((float)in[idx].z) + ((float)in[idx].w) * ((float)in[idx].w);
}

// (2 / 6)
__global__
void mean_npol(float2 *m2, float2 *mean) 
{ // <<< time samples, samp_T0/256 >>>
	float mx = 0, my = 0;
	int idx = threadIdx.x * blockDim.x + blockIdx.x * samp_T0;
	for (int ii = 0; ii < samp_T0/blockDim.x; ii++){
		mx += m2[idx + ii].x;
		my += m2[idx + ii].y;
	}
	__syncthreads();
	atomicAdd( &mean[blockIdx.x].x, mx);
	atomicAdd( &mean[blockIdx.x].y, my);
	__syncthreads();
	if (idx == blockIdx.x*samp_T0){
		mean[blockIdx.x].x /= (samp_T0); 
		mean[blockIdx.x].y /= (samp_T0); 
	}
}

// (3 / 6)
__global__
void max_min(float2 *m2, int2 *maximum, int2 *minimum, float2 *resizer)
{ // <<< time samples, samp_T0/256 >>>
	int idx = ((blockIdx.x * blockDim.x) + threadIdx.x) * 256;
	int threadMx = 0;
	int threadMy = 0;
	int threadmx = 0;
	int threadmy = 0;
	for (int ii = 0; ii < 256; ii++){
		if (threadMx < m2[idx + ii].x){threadMx = (int)m2[idx + ii].x;}
		if (threadMy < m2[idx + ii].y){threadMy = (int)m2[idx + ii].y;}
		if (threadmx > m2[idx + ii].x){threadmx = (int)m2[idx + ii].x;}
		if (threadmy > m2[idx + ii].y){threadmy = (int)m2[idx + ii].y;}
	}
	__syncthreads();
	atomicMax( &maximum[blockIdx.x].x, threadMx);
	atomicMax( &maximum[blockIdx.x].y, threadMy);
	atomicMin( &minimum[blockIdx.x].x, threadmx);
	atomicMin( &minimum[blockIdx.x].y, threadmy);
	__syncthreads();
	if (idx == blockIdx.x*blockDim.x*256){
		resizer[blockIdx.x].x = ((float)(maximum[blockIdx.x].x - minimum[blockIdx.x].x)/hist_size)*1.001; //1.001 just to make division work later
		resizer[blockIdx.x].y = ((float)(maximum[blockIdx.x].y - minimum[blockIdx.x].y)/hist_size)*1.001; 
	}
}

// (4 / 6)
__global__  // TESTED WITH RFI AND GAUSSIAN WHITE NOISE. GOOD FUNCTION
void histo_kernel( float2 *m2, int2 *histogram, int2 *minimum, float2 *resizer){
	__shared__ int2 tempH[hist_size];
	tempH[threadIdx.x] = make_int2(0., 0.);
	__syncthreads();
	int idx = threadIdx.x + (blockIdx.x*blockDim.x);
	int jump = floorf((blockIdx.x*blockDim.x)/samp_T0); // what time sample each threadblock is in. cause 256*256 = samp_T0.
	for (int zz = 0; zz < nT0; zz++){ // worked around programming issue
		if (jump == zz){
			atomicAdd(&tempH[(int)((m2[idx].x-minimum[zz].x)/resizer[zz].x )].x,1);
			atomicAdd(&tempH[(int)((m2[idx].y-minimum[zz].y)/resizer[zz].y )].y,1); 
			__syncthreads();
			atomicAdd( &(histogram[threadIdx.x + zz*hist_size].x), tempH[threadIdx.x].x);
			atomicAdd( &(histogram[threadIdx.x + zz*hist_size].y), tempH[threadIdx.x].y);
		}
	}
}

// (5 / 6)
__global__  // modified from other code
void moment_order(float2 *mean, int2 *histogram, float2 *moment, int2 *minimum, float2 *resizer)
{
	int next = blockIdx.x;
	int order = threadIdx.x; // can take out a thread. You do not need 1st moment
	for (int ii = 0; ii < hist_size; ii++){
		moment[order+next*3].x += ((float)( powf( ((ii+minimum[next].x)*resizer[next].x) - mean[next].x, order+2)) * histogram[ii+next*hist_size].x); 
		moment[order+next*3].y += ((float)( powf( ((ii+minimum[next].y)*resizer[next].y) - mean[next].y, order+2)) * histogram[ii+next*hist_size].y); 
	}
	moment[order+next*3].x /= (samp_T0);
	moment[order+next*3].y /= (samp_T0);
}

// (6 / 6)
__global__
void final(float2 *moment, struct rsk *d_rsk)
{
	int idx = blockIdx.x; // <<<1024 "timeseries", 6 "made faster">>>
	if(threadIdx.x==0){ d_rsk[idx].rms.x = sqrtf(fabsf(moment[idx*3+0].x));}
	if(threadIdx.x==1){ d_rsk[idx].rms.y = sqrtf(fabsf(moment[idx*3+0].y));}
	if(threadIdx.x==2){ d_rsk[idx].skew.x = moment[idx*3+1].x/(powf(moment[idx*3+0].x, 1.5));}
	if(threadIdx.x==3){ d_rsk[idx].skew.y = moment[idx*3+1].y/(powf(moment[idx*3+0].y, 1.5));}
	if(threadIdx.x==4){ d_rsk[idx].kurt.x = (moment[idx*3+2].x/(powf(moment[idx*3+0].x, 2))) -3;}
	if(threadIdx.x==5){ d_rsk[idx].kurt.y = (moment[idx*3+2].y/(powf(moment[idx*3+0].y, 2))) -3;}
}



/********************************************************/
/************************* MAIN *************************/
/********************************************************/

int main(int argc, char *argv[])
{	
	FILE *fp_in;
	FILE *fp_out;
	cudaError_t cudaError;
	char filename[59] = "../FRSC_KO/CUDA_KO/rcog/red_gup.dat"; // replace for different infile
	int outlet = 3;
	int nT0 = 512; // number of samp_T0's. This can be modified.
	int nMom = 3; // 4 moments to be calculated
	dim3 pwrgrd(nT0*samp_T0/hist_size);
	dim3 hist(hist_size);
	dim3 nto(nT0);
	dim3 nmom(nMom);

	// memory
	char4 *h_input;
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
	struct rsk h_rsk[nT0], *d_rsk;	

	// Host Allocation (page-locked)
	cudaHostAlloc((void**) &h_input, nT0*samp_T0*sizeof(char4), cudaHostAllocDefault);
	cudaHostAlloc((void**) &h_mean, nT0*sizeof(float2), cudaHostAllocDefault); 
	cudaHostAlloc((void**) &h_moment, nT0*nMom*sizeof(float2), cudaHostAllocDefault); //-
	cudaHostAlloc((void**) &h_histogram, nT0*hist_size*sizeof(int2), cudaHostAllocDefault); //-
	cudaHostAlloc((void**) &h_maximum, nT0*sizeof(int2), cudaHostAllocDefault); //-
	cudaHostAlloc((void**) &h_rsk, nT0*sizeof(struct rsk), cudaHostAllocDefault);

	// Device Allocation
	cudaMalloc((void**)&d_input, nT0*samp_T0*sizeof(char4));
	cudaMalloc((void**)&d_m2, nT0*samp_T0*sizeof(float2));
	cudaMalloc((void**)&d_mean, nT0*sizeof(float2));
	cudaMalloc((void**)&d_maximum, nT0*sizeof(int2));
	cudaMalloc((void**)&d_minimum, nT0*sizeof(int2));
	cudaMalloc((void**)&d_resizer, nT0*sizeof(float2));
	cudaMalloc((void**)&d_histogram, nT0*hist_size*sizeof(int2));
	cudaMalloc((void**)&d_moment, nT0*nMom*sizeof(float2));
	cudaMalloc((void**)&d_rsk, nT0*sizeof(struct rsk));
	
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

////////////////////////////////// command line args //////////////////////////////////////////

	if (argc>=2) { sscanf( argv[1], "%d", &outlet );} 
	else {
      	printf("FATAL: main(): command line argument <outlet> not specified\n");
      	return 0;
	}
	if (outlet != 1 && outlet !=2 && outlet != 3){printf("outlet err\n");return 0;} 
  	printf("outlet = %d\n",outlet);

////////////////////////////////////// main loop /////////////////////////////////////////////

	if (!(fp_in = fopen(filename,"r"))){
		printf("FATAL: main(); couldn't open file'\n");
		return -1;
	}if (!(fp_out = fopen("rcog_rsk.dat","w"))){
		printf("FATAL: main(); couldn't open file'\n");
		return -1;
	}
	int ii = 0;
	while(!feof(fp_in)){	
		printf("NUMBER: %d\n", ii);
		cudaMemset(d_m2, 0, nT0*samp_T0*sizeof(float2));
		cudaMemset(d_mean, 0, nT0*sizeof(float2));
		cudaMemset(d_moment, 0, nT0*nMom*sizeof(float2));
		cudaMemset(d_histogram, 0, nT0*hist_size*sizeof(int2));
		cudaMemset(d_maximum, 0, nT0*sizeof(int2));
		cudaMemset(d_minimum, 0, nT0*sizeof(int2));
		cudaMemset(d_resizer, 0, nT0*sizeof(float2));
		cudaMemset(d_rsk, 0, nT0*sizeof(struct rsk));

		// read file + CUDA
		cudaEventRecord(start_mem, 0);
		fread( h_input, nT0*samp_T0*sizeof(char4), 1, fp_in);	
		cudaEventRecord(start_ker, 0); 
		cudaMemcpy(d_input, h_input, nT0*samp_T0*sizeof(char4), cudaMemcpyHostToDevice);
		power_arr<<<pwrgrd,hist>>>(d_input, d_m2); 
		mean_npol<<<nto,hist>>>(d_m2, d_mean);// <<< time samples, samp_T0/256 >>>
		max_min<<<nto,hist>>>(d_m2, d_maximum, d_minimum, d_resizer); 	
		histo_kernel<<<pwrgrd,hist>>>(d_m2, d_histogram, d_minimum, d_resizer); 
		moment_order<<<nto,nmom>>>(d_mean, d_histogram, d_moment, d_minimum, d_resizer);
		final<<<nto,6>>>(d_moment, d_rsk);  
		cudaMemcpy(h_rsk, d_rsk, nT0*sizeof(struct rsk), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_mean, d_mean, nT0*sizeof(float2), cudaMemcpyDeviceToHost);
		cudaEventRecord(stop_ker,0);
		cudaEventSynchronize(stop_ker);
		
		if (outlet == 1){ // Diagnostics
			cudaMemcpy(h_maximum, d_maximum, nT0*sizeof(float2), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_moment, d_moment, nT0*nMom*sizeof(float2), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_histogram, d_histogram, nT0*hist_size*sizeof(float2), cudaMemcpyDeviceToHost);
			int next = 0;
			for (next = 0; next < 6; next++){
			printf("\nStat Sample: %d;\n", next);
			printf("M.x = %f; ", h_mean[next].x);
			printf("M.y = %f; \n", h_mean[next].y); 
			printf("^.x = %d; ",h_maximum[next].x);
			printf("^.y = %d; \n",h_maximum[next].y);
			printf("R.x = %f; ", h_rsk[next].rms.x);
			printf("R.y = %f; \n", h_rsk[next].rms.y);
			printf("S.x = %f; ", h_rsk[next].skew.x);
			printf("S.y = %f; \n", h_rsk[next].skew.y);
			printf("K.x = %f; ", h_rsk[next].kurt.x);
			printf("K.y = %f; \n", h_rsk[next].kurt.y);/*
			printf("moment 2 = %f; ",h_moment[next*nMom + 1].x);
			printf("moment 2 = %f; \n",h_moment[next*nMom + 1].y);
			printf("moment 3 = %f; ",h_moment[next*nMom + 2].x);
			printf("moment 3 = %f; \n",h_moment[next*nMom + 2].y);
			printf("moment 4 = %f; ",h_moment[next*nMom + 3].x);
			printf("moment 4 = %f; \n",h_moment[next*nMom + 3].y);*/
			for (int kk = 0; kk < hist_size; kk++){
				fprintf(stdout, "%d ", h_histogram[kk + next*hist_size].x);
			}printf("\n");
			}
		}
		if (outlet == 2){ // write to outfile
			for (int jj = 0; jj < nT0; jj++){
				fwrite(&h_mean[jj], sizeof(float2), 1, fp_out);
				fwrite(&h_rsk[jj], sizeof(struct rsk), 1, fp_out);
			}
		}

		// Timing addition
		cudaEventRecord(stop_mem, 0);
		cudaEventSynchronize(stop_mem);
		cudaEventElapsedTime( &elapsedTime, start_ker, stop_ker);
		elapsedTimeK += elapsedTime;
		cudaEventElapsedTime( &elapsedTime, start_mem, stop_mem);
		elapsedTimeM += elapsedTime;
	ii++;	
	}

////////////////////////////////////// Cleaning up ///////////////////////////////////////////////

	printf("\nKernel calls took: %3.3f ms\n", elapsedTimeK);
	printf("All Memory transfers took: %3.3f ms\n", elapsedTimeM - (elapsedTimeK));
	printf("Writing to files and calculations took: %3.3f ms\n", elapsedTimeO);

	cudaEventDestroy(stop_mem);
	cudaEventDestroy(start_ker);
	cudaEventDestroy(stop_ker);

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
	cudaFreeHost(h_maximum);

	fclose(fp_in);
	fclose(fp_out);
	return 0;
}

/*************** FOR ERROR CHECKING INLINE *************
cudaError_t cudaError;
cudaError = cudaGetLastError();
if(cudaError != cudaSuccess)
{
	printf("  cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
	return 0;
}
********************************************************/
// FUTURE WORK ON THIS CODE MAY WANT TO CONSIDER CUDA DYNAMIC PARALLIZATION FOR EFFICIENCY.
