__kernel void convolution(	__global float *input, \
							__global float *filter, \
							__global float *output, \
							int filterWidth, \
							int imgWidth, \
							int imgHeight)
{
	int x = get_global_id(0), y = get_global_id(1);
	int hf = filterWidth / 2, mid = hf + 1;
	float curr = 0.f;
	for(int i = -1 * hf; i <= hf; i++) {
		for(int j = -1 * hf; j <= hf; j++) {
			if(i + x >= 0 && i + x < imgWidth && j + y >= 0 && j + y < imgHeight) {
				curr += input[i + x + (j + y) * imgWidth] * filter[i + hf + (j + hf) * filterWidth];
			}
		}
	}
	output[x + y * imgWidth] = curr;
	return;   
}
