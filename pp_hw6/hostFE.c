#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth;
	
	int err;
	cl_kernel kernel = clCreateKernel(*program, "convolution", &err);

	cl_mem input, output, filterKernel;
	input = clCreateBuffer(*context, CL_MEM_READ_WRITE, sizeof(float) * imageWidth * imageHeight, NULL, NULL);
	output = clCreateBuffer(*context, CL_MEM_READ_WRITE, sizeof(float) * imageWidth * imageHeight, NULL, NULL);
	filterKernel = clCreateBuffer(*context, CL_MEM_READ_WRITE, sizeof(float) * filterSize, NULL, NULL);

	cl_command_queue commands = clCreateCommandQueue(*context, *device, 0, &err);
	clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(float) * imageWidth * imageHeight, inputImage, 0, NULL, NULL);
	clEnqueueWriteBuffer(commands, filterKernel, CL_TRUE, 0, sizeof(float) * filterSize, filter, 0, NULL, NULL);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void**) &input);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void**) &filterKernel);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), (void**) &output);
	clSetKernelArg(kernel, 3, sizeof(int), &filterWidth);
	clSetKernelArg(kernel, 4, sizeof(int), &imageWidth);
	clSetKernelArg(kernel, 5, sizeof(int), &imageHeight);

	size_t global[2] = {imageWidth, imageHeight}, local[2] = {8, 8};
	clEnqueueNDRangeKernel(commands, kernel, 2, NULL, global, local, 0, NULL, NULL);

	clFinish(commands);
	clEnqueueReadBuffer(commands, output, CL_TRUE, 0, sizeof(float) * imageWidth * imageHeight, outputImage, 0, NULL, NULL);


	clReleaseMemObject(input);
	clReleaseMemObject(output);
	clReleaseMemObject(filterKernel);
	clReleaseCommandQueue(commands);
}
