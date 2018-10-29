// (4b) settings of sampler to read pixel values from image: 
// * coordinates are pixel-coordinates
// * no interpolation between pixels
// * pixel values from outside of image are taken from edge instead
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE;


// operation types
#define OP_DILATE 0
#define OP_ERODE 1


// (1) kernel with 3 arguments: input and output image, and operation (dilation=0 or erosion=1)
__kernel void morphOpKernel(__read_only image2d_t in, int op, __write_only image2d_t out)
{
	// (2) IDs of work-item represent x and y coordinates in image
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	
	// extreme pixel value (max/min for dilation/erosion) inside structuring element
	float extremePxVal = op == OP_DILATE ? 0.0f : 1.0f; // initialize extreme value with min when searching for max value and vice versa
	
	// (3) structuring element is square of size 3x3: therefore we simply walk the 3x3 neighborhood of the current location
	for(int i=-1; i<=1; ++i)
	{
		for(int j=-1; j<=1; ++j)
		{
			// (4a) take pixel value at location (x+i, y+j)
			const float pxVal = read_imagef(in, sampler, (int2)(x + i, y + j)).s0;
			
			// (5) depending on operation, search max or min pixel value
			switch(op)
			{
				// for dilation, take max
				case OP_DILATE:
					extremePxVal = max(extremePxVal, pxVal);
				break;
				
				// for erosion, take min
				case OP_ERODE:
					extremePxVal = min(extremePxVal, pxVal);
				break;
			}
		}
	}
	
	// (6) write value of pixel to output image at location (x, y)
	write_imagef(out, (int2)(x, y), (float4)(extremePxVal, 0.0f, 0.0f, 0.0f));
}
