import cv2
import numpy as np
import pyopencl as cl


class MorphOp:
	"operation types"
	DILATE = 0
	ERODE = 1


def applyMorphOp(imgIn, op):
	"apply morphological operation to image using GPU"
	
	# (1) setup OpenCL
	platforms = cl.get_platforms() # a platform corresponds to a driver (e.g. AMD)
	platform = platforms[0] # take first platform
	devices = platform.get_devices(cl.device_type.GPU) # get GPU devices of selected platform
	device = devices[0] # take first GPU
	context = cl.Context([device]) # put selected GPU into context object
	queue = cl.CommandQueue(context, device) # create command queue for selected GPU and context

	# (2) get shape of input image, allocate memory for output to which result can be copied to
	shape = imgIn.T.shape
	imgOut = np.empty_like(imgIn)	
	
	# (2) create image buffers which hold images for OpenCL
	imgInBuf = cl.Image(context, cl.mem_flags.READ_ONLY, cl.ImageFormat(cl.channel_order.LUMINANCE, cl.channel_type.UNORM_INT8), shape=shape) # holds a gray-valued image of given shape
	imgOutBuf = cl.Image(context, cl.mem_flags.WRITE_ONLY, cl.ImageFormat(cl.channel_order.LUMINANCE, cl.channel_type.UNORM_INT8), shape=shape) # placeholder for gray-valued image of given shape
	
	# (3) load and compile OpenCL program
	program = cl.Program(context, open('kernel.cl').read()).build()

	# (3) from OpenCL program, get kernel object and set arguments (input image, operation type, output image)
	kernel = cl.Kernel(program, 'morphOpKernel') # name of function according to kernel.py
	kernel.set_arg(0, imgInBuf) # input image buffer
	kernel.set_arg(1, np.uint32(op)) # operation type passed as an integer value (dilate=0, erode=1)
	kernel.set_arg(2, imgOutBuf) # output image buffer
	
	# (4) copy image to device, execute kernel, copy data back
	cl.enqueue_copy(queue, imgInBuf, imgIn, origin=(0, 0), region=shape, is_blocking=False) # copy image from CPU to GPU
	cl.enqueue_nd_range_kernel(queue, kernel, shape, None) # execute kernel, work is distributed across shape[0]*shape[1] work-items (one work-item per pixel of the image)
	cl.enqueue_copy(queue, imgOut, imgOutBuf, origin=(0, 0), region=shape, is_blocking=True) # wait until finished copying resulting image back from GPU to CPU
	
	return imgOut


def main():
	"test implementation: read file 'in.png', apply dilation and erosion, write output images"

	# read image
	img = cv2.imread('in.png', cv2.IMREAD_GRAYSCALE)
	
	# dilate
	dilate = applyMorphOp(img, MorphOp.DILATE)
	cv2.imwrite('dilate.png', dilate)
	
	# erode
	erode = applyMorphOp(img, MorphOp.ERODE)
	cv2.imwrite('erode.png', erode)


if __name__ == '__main__':
	main()