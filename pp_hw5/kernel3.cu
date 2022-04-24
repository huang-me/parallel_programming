#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__device__ int mandel(float c_re, float c_im, int maxIteration) {
	float z_re = c_re, z_im = c_im;
    int iter;
	for (iter = 0; iter < maxIteration; ++iter) {

	    if (z_re * z_re + z_im * z_im > 4.f)
		    break;

		float new_re = z_re * z_re - z_im * z_im;
		float new_im = 2.f * z_re * z_im;
		z_re = c_re + new_re;
		z_im = c_im + new_im;
	}

	return iter;
}

__global__ void mandelKernel(int* img, int x0, int y0, float dx, float dy, int maxIterations, int width, int height) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	if(j >= width || i >= height) return;
	float x = (float)x0 + j * dx;
	float y = (float)y0 + i * dy;
	int index = i * width + j;
	img[index] = mandel(x, y, maxIterations);
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (
			float upperX, 
			float upperY, 
			float lowerX, 
			float lowerY, 
			int* img, 
			int resX, 
			int resY, 
			int maxIterations)
{
	// calculate dx & dy
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;
	// initialize memory of device
	int *imgKernel, *imgHost;
	//cudaMallocPitch((void**) &imgKernel, &imgKernelSize, sizeof(int) * resX, resY);
	cudaHostAlloc((void**) &imgHost, sizeof(int) * resX * resY, cudaHostAllocWriteCombined | cudaHostAllocMapped);
	cudaHostGetDevicePointer(&imgKernel, imgHost, 0);
	// setup threads
	int blockWidth = 32;
	bool remX = resX & (blockWidth - 1), remY = resY & (blockWidth - 1);
	dim3 blockCnt((resX / blockWidth + remX), (resY / blockWidth + remY));
	dim3 threadsPerBlock(blockWidth, blockWidth);
	// call device function
	mandelKernel<<<blockCnt, threadsPerBlock>>> \
		(imgKernel, lowerX, lowerY, stepX, stepY, maxIterations, resX, resY);
	cudaDeviceSynchronize();
	// copy result back to host
	// memcpy(img, imgHost, sizeof(int) * resX * resY);
	cudaMemcpy(img, imgHost, sizeof(int) * resX * resY, cudaMemcpyHostToHost);
}
