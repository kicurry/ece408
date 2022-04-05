## constant memory
### 1. global constant memory
``` C++
// declared in file scope(global scope) 
__constant__ float M[Mask_width * Mask_width];

// memcpy to constant memory in host code
cudaMemcpyToSymbol(M, hostMaskData, maskRows * maskColumns * sizeof(float));

// Correct way to use: just like global variable
__global__ void convolution_2D_tiled_kernel(float *P, float *N, 
                                            int height, int width, int channels, 
                                            int MaskWidth) {
    // ...
        do something with M[i]
    // ...
}

// A wrong case!!!
__global__ void convolution_2D_tiled_kernel(float *P, float *N, 
                                            int height, int width, int channels, 
                                            int MaskWidth, const float *M) {
    // ...
}

// NEVER pass it as a parameter in host code!!!!
convolution_2D_tiled_kernel<<<blocks, threads>>>(deviceOutputImageData, deviceInputImageData, imageHeight, imageWidth, imageChannels, Mask_width, M);
```
### 2. `__restrict__` keyword

you can use `const float * __restrict__ M` as one of the parameters during your kernel launch. 

This informs the compiler that the contents of the mask array are constants and will only be accessed through pointer variable `M`.

```C++
__global__ void convolution_2D_tiled_kernel(float *P, float *N, 
                                            int height, int width, int channels, 
                                            int MaskWidth, 
                                            const float * __restrict__ M) {
    // ...
}

// allocate device memory and copy
cudaMalloc(&deviceMaskData, maskRows * maskColumns * sizeof(float));
cudaMemcpy(deviceMaskData,
            hostMaskData,
            maskRows * maskColumns * sizeof(float),
            cudaMemcpyHostToDevice);
```

## Pitch
Textbook may be misleading, it doesnot consider channels when illustrating pitch!!!

In fact, pitch equals to `aligned(channels * width)`