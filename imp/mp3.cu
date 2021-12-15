#include    <wb.h>

#define BLOCK_WIDTH 32

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

#define ceil(X, BASE) (((X) + (BASE) - 1) / (BASE))

// Compute C = A * B
__global__ void matrixMultiplyShared(float * A, float * B, float * C,
			             int numARows, int numAColumns,
			             int numBRows, int numBColumns,
			             int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
    //@@ You have to use shared memory for this MP
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = bx * BLOCK_WIDTH + tx;
    int col = by * BLOCK_WIDTH + ty;

    __shared__ float subsetA[BLOCK_WIDTH][BLOCK_WIDTH];
    __shared__ float subsetB[BLOCK_WIDTH][BLOCK_WIDTH];

    int phNum = ceil(numAColumns, BLOCK_WIDTH);
    float Cval = 0;
    for (int ph = 0;ph < phNum;ph ++) {
        // Initialize shared memory array 
        subsetA[tx][ty] = 0;
        subsetB[tx][ty] = 0;
        // Load
        if (row < numCRows && ph * BLOCK_WIDTH + ty < numAColumns) {
            subsetA[tx][ty] = A[row * numAColumns + ph * BLOCK_WIDTH + ty]; 
        }
        if (col < numCColumns && ph * BLOCK_WIDTH + tx < numBRows) {
            subsetB[tx][ty] = B[(ph * BLOCK_WIDTH + tx) * numBColumns + col];
        }
        __syncthreads();

        // Compute
        for (int k = 0;k < BLOCK_WIDTH;k ++) {
            Cval += subsetA[tx][k] * subsetB[k][ty];
        }
        __syncthreads();
    }
    if (row < numCRows && col < numCColumns) {
        C[row * numCColumns + col] = Cval;
    }
}

/*
 *   ./bin/MP3 -e ../test_data/mp03/0/output.raw -i ../test_data/mp03/0/input0.raw,../test_data/mp03/0/input1.raw -o ../test_data/mp03/0/res.raw -t matrix
 */
int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostA; // The A matrix
    float * hostB; // The B matrix
    float * hostC; // The output C matrix
    float * deviceA;
    float * deviceB;
    float * deviceC;
    int numARows; // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows; // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows; // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set this)

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float *) wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB = (float *) wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
    //@@ Set numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;
    //@@ Allocate the hostC matrix
    hostC = (float *) malloc(numCRows * numCColumns * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
    wbLog(TRACE, "The number of phase are ", ceil(numAColumns, BLOCK_WIDTH));

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    cudaMalloc((void **)&deviceA, numARows * numAColumns * sizeof(float));
    cudaMalloc((void **)&deviceB, numBRows * numBColumns * sizeof(float));
    cudaMalloc((void **)&deviceC, numCRows * numCColumns * sizeof(float));

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);

    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    //@@ Initialize the grid and block dimensions here
    dim3 DimGrid(ceil(numCRows, BLOCK_WIDTH), ceil(numCColumns, BLOCK_WIDTH), 1);
    dim3 DimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    matrixMultiplyShared<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, 
        numARows, numAColumns,
        numBRows, numBColumns,
        numCRows, numCColumns);

    cudaThreadSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");
    
    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost);

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}

