### Tiling Matrix Multiplication

#### Special case: width of matrix is multiple of tile width  

for each thread block: 

- allocate two $\text{BLOCK\_WIDTH}^2$ 2D-arrays to save a tile(aka. subset) of matrix $A,B$ respectively for each phase.

  NOTE: Where two shared memory arrays comes from?

  We divide the matrix into multiple small matrices whose size is equal to size of tile and want each thread can compute one element in result matrix $C$.

  ![mp3_1](./assets/mp3_1.png)

- for each phase:

  - Using current tile to update shared memory array.

    Because size of a tile is equal to size of a shared memory array, we can let each thread load value in relative postion of current tile.

  - Compute inner products in current tile.

#### Boundary check: matrix with arbitrarywidth

Suppose there is matrix multiplication $A_{m\times k}\times B_{k\times n} = C_{m\times n}$.

Consider a fixed thread block $B_{t\times t}$ resiponsible for computing $C'_{t\times t}$(sub-matrix of $C$ same as the figure above). We do boundary check only in two case:

- Load data from $A$ or $B$

  Coordinates cannot exceed the width(height) of $A$ or $B$ 

-  Compute

  Only threads whose coordinates are within the range of the matrix need to calculate.

---

### Limiting factor to parallelism: Memory

In my MP3's code, I orgnaze the CUDA kernal as **32×32**  thread block. And for each thread block, I allocate 32×32×4B = 4KB of shared memory to save subset of matrix $M, N$ respectively, say 8KB in total for each thread block.

On the one hand, recall MP1(see `MP1.md`), we know that maximum # of block per SM is **16** indicated by `deviceProp.maxBlocksPerMultiProcessor`. Therefore, we need 8KB×16=128KB of the shared memory to allows **16** blocks to simultaneously reside in an SM. But `deviceProp.sharedMemPerMultiprocessor` shows that GeForce RTX 3060 only has 100K-bytes of the shared memory per SM. In other word, 32×32 Tiles can not make best use of parallelism ability of SM.

On the other hand, GeForce RTX 3060 only allows **1536** threads in each SM. This constraint limits the number of blocks in each SM to ⌊1536/(32×32)⌋= 1.Consequently, only 1×8KB = 8KB of the shared memory will be used!

#### Set BLOCK_SIZE to other values
- BLOCK_SIZE = 8
    
  8×8×4B×2=512B, 512B×16=8KB << **100KB**
  
  1536/(8×8)=24 > **16**

- BLOCK_SIZE = 16
  
  16×16×4B×2=2KB, 2KB×16=32KB < **100KB**

  1536/(16×16)=6 < **16**