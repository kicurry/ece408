### Device Query

Suppose there is a declaration `cudaDeviceProp deviceProp` and then information about my own device is list as below:

| Member variable           | Description                  | Value                                |
| ------------------------- | ---------------------------- | ------------------------------------ |
| deviceProp.name           | name                         | NVIDIA GeForce RTX 3060 Laptop GPU   |
| deviceProp.major          | Computational Capabilities   | 8.6                                  |
| deviceProp.totalGlobalMem | Maximum global memory size   | 6441926656B = 6143.5MB ≈ 6GB |
| deviceProp.totalConstMem  | Maximum constant memory size | 65536B = 64KB                        |

#### CUDA thread organization

| Member variable             | Description              | Value                                    |
| --------------------------- | ------------------------ | ---------------------------------------- |
| deviceProp.maxGridSize[3]   | Maximum grid dimensions  | 2147483647 × 65535 × 65535 |
| deviceProp.maxThreadsDim[3] | Maximum block dimensions | 1024 × 1024 × 64           |

#### Limitation about multiprocessors

| Member variable                        | Description                                         | Value           |
| -------------------------------------- | --------------------------------------------------- | --------------- |
| deviceProp.sharedMemPerBlock           | Maximum shared memory size per block                | 49152 = 48KB    |
| deviceProp.maxThreadsPerBlock          | Maximum threads per block                           | 1024            |
|                                        |                                                     |                 |
| deviceProp.multiProcessorCount         | Number of multiprocessors(eg. SM) on device         | 30              |
| deviceProp.maxBlocksPerMultiProcessor  | Maximum number of resident block per multiprocessor | 16              |
| deviceProp.maxThreadsPerMultiProcessor | Maximum resident threads per multiprocessor         | 1536            |
| deviceProp.sharedMemPerMultiprocessor  | Shared memory available per multiprocessor in bytes | 102400B = 100KB |
| deviceProp.regsPerMultiprocessor       | 32-bit registers available per multiprocessor       | 65536           |
| deviceProp.warpSize                    | Warp size                                           | 32              |

NOTE:

- Block in here indicate **SM block** but not a kind of thread organzation in CUDA execution model
- See more details in Lecture1&2 to understand how SMs schedule thread.
