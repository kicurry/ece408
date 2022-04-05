#include    <wb.h>

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)


#define Mask_width  5
#define Mask_radius Mask_width/2
#define Max_Channel 3
#define O_Tile_Width 8
#define CEIL(x, y) (((x) + (y) - 1) / (y))

//@@ INSERT CODE HERE

// Report error location and terminate, if "cudaError != SUCCESS".
#define CSC(err)	__cudaSafeCall(err, __FILE__, __LINE__)
// Report error location and terminate, if "cudaError != SUCCESS" occured.
#define CCE()		__cudaCheckError(__FILE__, __LINE__)

inline void __cudaSafeCall(cudaError err, const char *file, const int line) {
	if (cudaSuccess != err) {
		fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line,
				cudaGetErrorString(err));
		exit(-1);
	}
	return;
}

inline void __cudaCheckError(const char *file, const int line) {
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line,
				cudaGetErrorString(err));
		exit(-1);
	}
	// More careful checking. However, this will affect performance.
	// Comment out if needed.
	//#define safer
	#ifdef safer
	err = cudaDeviceSynchronize();
	if (cudaSuccess != err) {
		fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n", file,
				line, cudaGetErrorString(err));
		exit(-1);
	}
	#endif
	return;
}

// constant memory for mask array
// NOTE: 1D linear memory
__constant__ float M[Mask_width * Mask_width];

__device__ float clamp(float x, float start, float end) {
    float tmp = x > start ? x : start;
    return tmp < end ? tmp : end;
}

// NOTE: 'pitch/channels == width' indicates that image data is aligned!!!
__global__ void convolution_2D_tiled_kernel(float *P, float *N, 
                                            int height, int width, int pitch, int channels, 
                                            int MaskWidth) {
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;

    // index in output tiles of result matrix P
    int row_o = blockIdx.x * O_Tile_Width + tidx;
    int col_o = blockIdx.y * O_Tile_Width + tidy;

    // index in input tiles (corresponding to index in N)
    int row_i = row_o - Mask_radius;
    int col_i = col_o - Mask_radius;

    __shared__ float N_ds[O_Tile_Width + Mask_width - 1][O_Tile_Width + Mask_width - 1][Max_Channel];
    // load N elements from global memory to shared memory
    if (row_i >= 0 && row_i < height && col_i >= 0 && col_i < width) {
        for (int c = 0;c < channels;c ++) {
            // N_ds[tidx][tidy][c] = N[channels * (row_i * pitch + col_i) + c];
            N_ds[tidx][tidy][c] = N[row_i * pitch + col_i * channels + c];
        }
    } else {
        for (int c = 0;c < channels;c ++) {
            N_ds[tidx][tidy][c] = 0.0f;
        }
    }

    if (tidx < O_Tile_Width && tidy < O_Tile_Width) {
        float output[Max_Channel];
        for (int c = 0;c < channels;c ++) {
            output[c] = 0.0f;
            for (int i = 0;i < MaskWidth;i ++) {
                for (int j = 0;j < MaskWidth;j ++) {                
                    output[c] += M[i * MaskWidth + j] * N_ds[tidx + i][tidy + j][c];
                }
            }
        }

        // the height and width may not be multiples of Tile_Width
        if (row_o < height && col_o < width) {
            for (int c = 0;c < channels;c ++) {
                // P[channels * (row_o * pitch + col_o) + c] = clamp(output[c], 0.0f, 1.0f);
                P[row_o * pitch + col_o * channels + c] = clamp(output[c], 0.0f, 1.0f);
            }
        }
    }
}

int main(int argc, char* argv[]) {
    wbArg_t arg;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    // float * deviceMaskData;

    arg = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(arg, 0);
    inputMaskFile = wbArg_getInputFile(arg, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    // print some information about image
    wbLog(TRACE, "height = ", imageHeight, ", width = ", imageWidth, ", pitch = ", wbImage_getPitch(inputImage));

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    // cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    // cudaMemcpy(deviceMaskData,
    //            hostMaskData,
    //            maskRows * maskColumns * sizeof(float),
    //            cudaMemcpyHostToDevice);
    
    // use constant memory alternatively
    CSC(cudaMemcpyToSymbol(M, hostMaskData, maskRows * maskColumns * sizeof(float)));
    wbTime_stop(Copy, "Copying data to the GPU");


    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE
    /*  1. what is 'wbImage_t' ?
     *  'struct st_wbImage_t'@lib/wbImage.h:line 6
     *  typedef struct st_wbImage_t {
     *      int width;
     *      int height;
     *      int channels;
     *      int pitch;
     *      float * data;
     *  } * wbImage_t;
     *  'wbImage_t' is pointer type of 'struct st_wbImage_t'
     *
     *  2. what is wbImage_getData() ?
     *  '#define wbImage_getData(img) ((img)->data)'@lib/wbImage.h:line 20
     *  channel(c) of pixel(i,j) <=> data[3 * i * pitch + 3 * j + c]
     */

    dim3 threads(O_Tile_Width + Mask_width - 1, O_Tile_Width + Mask_width - 1, 1);
    dim3 blocks(CEIL(imageHeight, O_Tile_Width), CEIL(imageWidth, O_Tile_Width), 1);
    convolution_2D_tiled_kernel<<<blocks, threads>>>(deviceOutputImageData, deviceInputImageData,
                                                     imageHeight, imageWidth, wbImage_getPitch(inputImage), imageChannels,
                                                     Mask_width);
    CCE();
    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(arg, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    // cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}
