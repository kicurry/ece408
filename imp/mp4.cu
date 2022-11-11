// MP 4 Reduction
// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];

#include    <wb.h>

#define BLOCK_SIZE 64 //@@ You can change this

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

// monolithic-style load
// reduction tree
__global__ void total(float * input, float * output, int len) {
    //@@ Load a segment of the input vector into shared memory
    //@@ Traverse the reduction tree
    //@@ Write the computed sum of the block to the output vector at the 
    //@@ correct index
    
    // declare shared memory
    __shared__ float partialSum[BLOCK_SIZE << 1];

    // load input to shared memory
    int tidx = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tidx;
    if (idx < len) {
        // only the last warp has thread divergence
        partialSum[tidx] = input[idx];
    }

    // reduction
    for (unsigned int stride = blockDim.x >> 1;stride >= 1;stride >>= 1) {
        __syncthreads();
        if (tidx < stride && idx < len) {
            partialSum[tidx] += partialSum[tidx + stride];
        }
    }

    // store in global output
    output[blockIdx.x] = partialSum[0];
}

// monolithic-style load
// Step 1: reduction tree to warpSize
// Step 2: warpReduction by shuffle instructions
__global__ void total_by_shfl(float * input, float * output, int len) {    
    // declare shared memory
    __shared__ float partialSum[BLOCK_SIZE << 1];

    // load input to shared memory
    int tidx = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tidx;
    if (idx < len) {
        // only the last warp has thread divergence
        partialSum[tidx] = input[idx];
    }

    // reduction
    for (unsigned int stride = blockDim.x >> 1;stride >= 32;stride >>= 1) {
        __syncthreads();
        if (tidx < stride && idx < len) {
            partialSum[tidx] += partialSum[tidx + stride];
        }
    }

    register float value = partialSum[tidx];
    for (unsigned int stride = 16;stride >= 1;stride >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, stride);
    }

    // store in global output
    if (tidx == 0) {
        output[blockIdx.x] = value;
    }
}

// monolithic-style load
// multi-pass warpReduction(by shuffle instructions)
__global__ void total_by_shfl2(float * input, float * output, int len) {    
    // declare shared memory
    static __shared__ float partialSum[BLOCK_SIZE >> 4];

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int laneId = threadIdx.x & 0x1f;
    int warpId = threadIdx.x / warpSize;

    // warp reduction
    register float value = 0; 
    if (idx < len) {
       value = input[idx];
    }
    for (unsigned int stride = 16;stride >= 1;stride /= 2) {
        value += __shfl_down_sync(0xffffffff, value, stride);
    }
    // communication among warps through shared memory
    if (laneId == 0) {
        partialSum[warpId] = value;
    }
    __syncthreads();
    value = (threadIdx.x < (BLOCK_SIZE >> 4)) ? partialSum[threadIdx.x] : 0;

    // the 1st warp reduction
    value += __shfl_down_sync(0xf, value, 2);
    value += __shfl_down_sync(0xf, value, 1);
    // store in global output
    if (threadIdx.x == 0) {
        output[blockIdx.x] = value;
    }
}

// grid-stride-style load
// multi-pass warpReduction(by shuffle instructions)
__global__ void total_by_shfl3(float * input, float * output, int len) {    
    static __shared__ float partialSum[BLOCK_SIZE >> 4];
    
    volatile register float sum = float(0);
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = idx;i < len;i += gridDim.x * blockDim.x) {
        sum += input[i];
    }

    int laneId = threadIdx.x & 0x1f;
    int warpId = threadIdx.x / warpSize;

    // warp reduction 
    for (unsigned int stride = 16;stride >= 1;stride /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, stride);
    }
    // communication among warps through shared memory
    if (laneId == 0) {
        partialSum[warpId] = sum;
    }
    __syncthreads();
    sum = (threadIdx.x < (BLOCK_SIZE >> 4)) ? partialSum[threadIdx.x] : 0;

    // the 1st warp reduction
    sum += __shfl_down_sync(0xf, sum, 2);
    sum += __shfl_down_sync(0xf, sum, 1);
    // store in global output
    if (threadIdx.x == 0) {
        output[blockIdx.x] = sum;
    }
}

// grid-stride-style load
// pure reduction tree
// NOTE: gridSize must equal to (# of Multiprocessors)*(max # of Blcoks in SM(in reality))
__global__ void total2(float * input, float * output, int len) {    
    static __shared__ float partialSum[BLOCK_SIZE << 1];
    
    volatile register float sum = float(0);
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = idx;i < len;i += gridDim.x * blockDim.x) {
        sum += input[i];
    }
    if(idx < len){
        partialSum[threadIdx.x] = sum;
    }
    // reduction
    for (unsigned int stride = blockDim.x >> 1;stride >= 1;stride >>= 1) {
        __syncthreads();
        if (threadIdx.x < stride && idx < len) {
            partialSum[threadIdx.x] += partialSum[threadIdx.x + stride];
        }
    }
    // store in global output
    output[blockIdx.x] = partialSum[0];
}

int main(int argc, char ** argv) {
    int ii;
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numInputElements; // number of elements in the input list
    int numOutputElements; // number of elements in the output list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numInputElements);

    numOutputElements = numInputElements / (BLOCK_SIZE<<1);
    // numOutputElements = 360; // for tatal2
    if (numInputElements % (BLOCK_SIZE<<1)) {
        numOutputElements++;
    }
    hostOutput = (float*) malloc(numOutputElements * sizeof(float));

    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numInputElements);
    wbLog(TRACE, "The number of output elements in the input is ", numOutputElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    cudaMalloc((void **)&deviceInput, numInputElements * sizeof(float));
    cudaMalloc((void **)&deviceOutput, numOutputElements * sizeof(float));

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    cudaMemcpy(deviceInput, hostInput, numInputElements * sizeof(float), cudaMemcpyHostToDevice);

    wbTime_stop(GPU, "Copying input memory to the GPU.");
    //@@ Initialize the grid and block dimensions here
    dim3 grid(numOutputElements, 1, 1);
    dim3 block(BLOCK_SIZE << 1, 1, 1);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    total<<<grid, block>>>(deviceInput, deviceOutput, numInputElements);
    // total_by_shfl<<<grid, block>>>(deviceInput, deviceOutput, numInputElements);
    // total_by_shfl2<<<grid, block>>>(deviceInput, deviceOutput, numInputElements);
    // total_by_shfl3<<<grid, block>>>(deviceInput, deviceOutput, numInputElements);
    
    // total2<<<grid, block>>>(deviceInput, deviceOutput, numInputElements);    // need to set numOutputElements = 360

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostOutput, deviceOutput, numOutputElements * sizeof(float), cudaMemcpyDeviceToHost);

    wbTime_stop(Copy, "Copying output memory to the CPU");

    /********************************************************************
     * Reduce output vector on the host
     * NOTE: One could also perform the reduction of the output vector
     * recursively and support any size input. For simplicity, we do not
     * require that for this lab.
     ********************************************************************/
    for (ii = 1; ii < numOutputElements; ii++) {
        hostOutput[0] += hostOutput[ii];
    }

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, 1);

    free(hostInput);
    free(hostOutput);

    return 0;
}

