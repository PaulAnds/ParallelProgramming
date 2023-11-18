# ACTIVITY 1 PARCIAL 3

Programación de paralelismo

**Paul Andres Solis Villanueva**

University of Advanced Technologies

Leonardo Juárez Zucco

17/11/2023

# CHECKING THE DATA LAYOUT OF SHARED MEMORY

## Square Shared Memory

Using shared memory to cache global data with square dimensions can be used in a convenient manner. The simplicity of a square array allows for straightforward calculation of 1D memory offsets from 2D thread indices. The passage introduces a shared memory tile with 32 elements in each dimension, stored in row-major order. The layout of the 1D data and the logical 2D shared memory view with a mapping between 4-byte data elements and banks are illustrated in Figure 5-12.

Declaring a 2D shared memory variable statically:

```cuda
__shared__ int tile[N][N];
```

Because the shared memory tile is square, it can be accessed from a 2D thread block with neighboring threads accessing neighboring elements in either the x or y dimension using the following expressions:

```cuda
tile[threadIdx.y][threadIdx.x]
tile[threadIdx.x][threadIdx.y]
```

It emphasizes the importance of paying attention to how threads map to shared memory banks. Optimally, threads in the same warp (Column) should access separate banks to minimize conflicts. Since elements in shared memory belonging to different banks are stored consecutively, it is best to have threads with consecutive values of threadIdx.x accessing consecutive locations in shared memory. 

___

### Accessing Row-Major versus Column-Major
 
The kernel discussed in the passage involves writing global thread indices to a 2D shared memory array in row-major order and then reading and storing those values to global memory. The shared memory array is declared statically, and global thread indices are calculated for each thread.

Synchronization points (__syncthreads()) are used to ensure that all threads have stored data in shared memory before proceeding to global memory operations.

Kernels:

- setRowReadRow writes and reads shared memory in row-major order
- setColReadCol performs the same operations in column-major order.

The row-major order kernel is free of bank conflicts, while the column-major order kernel has a 16-way bank conflict on a Kepler device.The choice of memory access pattern significantly impacts performance. The passage demonstrates that accessing shared memory in row-major order performs better than column-major order due to neighboring threads referencing neighboring words.Performance is measured using __nvprof__, showing that the row-major order kernel is better than the column-major order kernel. 

___

### Writing Row-Major and Reading Column-Major

The shared memory is a static 2D array named 'tile' with dimensions [BDIMY][BDIMX].

```cpp
//Shared memory writes are performed in row-major order.
tile[threadIdx.y][threadIdx.x] = idx;
//Shared memory reads are done in column-major order.
out[idx] = tile[threadIdx.x][threadIdx.y];
```

Global memory index is computed 
```cpp
unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;.
```

Metrics reported after checking memory transactions for this kernel:

- Shared load transactions per request: 16.000000
- Shared store transactions per request: 1.000000

The store operation in shared memory is conflict-free.
The load operation reports a 16-way conflict, indicating potential inefficiencies or contention during shared memory reads.

___

### Dynamic Shared Memory

Dynamically declared shared memory in CUDA C.

Dynamic shared memory can be declared either outside the kernel for global file scope or inside the kernel for kernel scope. It is declared as an unsized 1D array. Calculation of memory access indices is necessary based on 2D thread indices. 


Two indices, row_idx and col_idx, are maintained for row-major and column-major memory offsets.Shared memory is written in row-major order and then read in column-major order.After synchronization, the content of shared memory is assigned to global memory in row-major order.

Kernel Code Example:

```cpp
__global__ void setRowReadColDyn(int *out) {
    // dynamic shared memory
    extern __shared__ int tile[];
    // mapping from thread index to global memory index
    unsigned int row_idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int col_idx = threadIdx.x * blockDim.y + threadIdx.y;
    // shared memory store operation
    tile[row_idx] = row_idx;
    // wait for all threads to complete
    __syncthreads();
    // shared memory load operation
    out[row_idx] = tile[col_idx];
}
```

The shared memory size must be specified when launching the kernel. For example:
```cpp
                                //shared memory size
setRowReadColDyn<<<grid, block, BDIMX * BDIMY * sizeof(int)>>>(d_C);
```

Results include 16-way conflict for shared load transactions and 1-way conflict for shared store transactions.The results of the dynamic shared memory example are compared to the example using statically declared shared memory. The dynamic shared memory example shows a 16-way conflict for read operations, while the write operations are conflict-free.

___

### Padding Statically Declared Shared Memory

Padding arrays is one strategy to prevent bank conflicts in shared memory.

Statically declared shared memory can be padded by adding a column to the 2D shared memory allocation. The kernel called setRowReadColPad that addresses a 16-way conflict observed during column-major order operations is then padding one element in each row of the shared memory array, conflicts in both reading and writing operations are eliminated.

Shared Memory Declaration:

The shared memory array is declared as `__shared__ int tile[BDIMY][BDIMX+IPAD];` , indicating the addition of padding. The kernel includes a calculation to map thread indices to global memory offsets.
```cpp
unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
```

The reported statistics indicate conflict-free transactions 
(1 shared_load_transactions_per_request and 1 shared_store_transactions_per_request).

- For Fermi devices, it is stated that padding one column is necessary to resolve bank conflicts.
- For Kepler devices, the need for padding may vary, and determining the proper number of padding elements for 64-bit access mode requires testing.

___

### Padding Dynamically Declared Shared Memory

Padding is applied to dynamically declared shared memory arrays to optimize memory access patterns in CUDA programming.
When working with dynamically declared shared memory arrays, index conversion from 2D thread indices to 1D memory indices involves skipping padded memory spaces.

Formulas:
```cpp
row_idx = threadIdx.y * (blockDim.x + 1) + threadIdx.x;
col_idx = threadIdx.x * (blockDim.x + 1) + threadIdx.y;
```

The kernel named setRowReadColDynPad utilizes dynamically allocated shared memory. Three indices are used for different memory access patterns:

- __row_idx__: Index for row-major writes to shared memory.
- __col_idx__: Index for column-major reads from shared memory.
- __g_idx__: Index for coalesced accesses to unpadded global memory.

Results for the kernel show that memory transactions are similar to statically declared shared memory padding, indicating that both types of shared memory can be effectively padded for performance improvement.


Both statically and dynamically declared shared memory can benefit from padding, leading to improved memory access patterns and enhanced performance.

___

### Comparing the Performance of the Square Shared Memory Kernels

Performance implications of different CUDA C kernels implemented are observed for a square matrix using shared memory based on elapsed times obtained from running various kernels giving us the results of:

- Kernels using padding show improved performance due to reduced bank conflicts. The elapsed times for these kernels indicate better efficiency.

- Kernels with dynamically declared shared memory have a little bit of higher time. The elapsed times for these kernels are slightly higher compared to others.

Different ordering of read and write operations on a 2D matrix can result in a transposed matrix.

Examples of the output matrices for different kernels are provided, demonstrating how the ordering of read and write operations affects whether the kernel generates a transposed matrix.

## Comments and Opinions

Understanding the reported metrics is crucial for optimizing the kernel's performance, especially addressing the reported conflict in shared memory load transactions. If you can read the nvprof and figure out what takes longer than the other functions and try to optimize it as much as possible youll end up with a really good code time. Its a really usefull tool in order to know what is going on, of course you also have to know how your code works both in software and hardware in order to know why its taking so long.

## Rectangular Shared Memory

Rectangular shared memory
- its a more general case of 2D shared memory where the number of rows and columns in an array are not equal. 

```cpp
__shared__ int tile[Row][Col];
```

Unlike a square shared memory implementation, switching thread coordinates to reference a rectangular array during a transpose operation is not straightforward. Doing so could lead to memory access violations. Re-implement kernels by recalculating access indices based on matrix dimensions.

For illustration purposes, the passage considers a rectangular shared memory array with 32 elements per row and 16 elements per column. The dimensions are defined using macros:

```cpp
#define BDIMX 32
#define BDIMY 16

__shared__ int tile[BDIMY][BDIMX];
```

Launching the kernel with only one grid and one 2D block, both using the same size as the rectangular shared memory array:

```cpp
dim3 block(BDIMX, BDIMY);
dim3 grid(1, 1);
```

Consider adjustments needed when working with rectangular shared memory in CUDA C, emphasizing the importance of adapting kernel implementations to accommodate varying array dimensions.

___

### Accessing Row-Major versus Accessing Column-Major

In the setRowReadRow kernel:

```cpp
//The innermost dimension of the shared memory array  matches the innermost dimension of the 2D thread block.
__shared__ int tile[BDIMY][BDIMX];
```

In the setColReadCol kernel:

```cpp
// The innermost dimension of the shared memory array  matches the outermost dimension of the 2D thread block.
__shared__ int tile[BDIMX][BDIMY];
```

For setRowReadRow kernel:

- shared_load_transactions_per_request 1.000000
- shared_store_transactions_per_request 1.000000

For setColReadCol kernel:

-  shared_load_transactions_per_request 8.000000
-  shared_store_transactions_per_request 8.000000


Both shared load and store transactions are serviced by eight transactions, indicating an eight-way conflict in the Col kernel compared to only one in the Row kernel. This difference is due to the Kepler K40's bank width being eight words, 16 4-byte data elements in a column are arranged into eight banks, resulting in the observed eight-way conflict.

___

### Writing Row-Major and Reading Column-Major

The kernel transposes a matrix using shared memory. The shared memory array is declared as a 2D array: 
```cpp
__shared__ int tile[BDIMY][BDIMX];
```

Three main memory operations are performed in the kernel:

- Write to a shared memory row with each warp to avoid bank conflicts.
- Read from a shared memory column with each warp to perform a matrix transpose.
- Write to a global memory row from each warp with coalesced access.

The 2D thread index of the current thread is converted to a 1D global thread ID to ensure coalesced global memory accesses. Then calculate the new coordinates in the transposed matrix using

```cpp
unsigned int irow = idx / blockDim.y;
unsigned int icol = idx % blockDim.y;
```

Initialize the shared memory tile by storing global thread IDs linearly:

```cpp
__global__ void setRowReadCol(int *out) {
  // ... (shared memory declaration, thread indexing, and coordinate calculation)
  tile[threadIdx.y][threadIdx.x] = idx;
  __syncthreads();
  out[idx] = tile[icol][irow];
}
```

Memory transactions reported by nvprof:

- shared_load_transactions_per_request 8.000000
- shared_store_transactions_per_request 1.000000


The load operation reports an eight-way conflict, while the store operation is conflict-free.

___

### Dynamically Declared Shared Memory

Dynamic shared memory can only be declared as a 1D array. When writing by rows and reading by columns, a new index is required to convert from 2D thread coordinates to 1D shared memory indices.

```cpp
//formula:
col_idx = icol * blockDim.x + irow.
```
The conversion yields column-major access to shared memory, leading to bank conflicts.

The setRowReadColDyn kernel code demonstrates the usage of dynamic shared memory. It uses dynamic shared memory tile and performs operations to showcase the conversion and access pattern. Shared memory size must be specified as part of the kernel launch using the triple-chevron syntax. 

`(<<<grid, block, size>>>)`

Transactions with nvprof alues are:

- shared_load_transactions_per_request: 8.000000 for read operations.
- shared_store_transactions_per_request: 1.000000 for write operations.

The write operation is conflict-free. The read operation reports an eight-way conflict. Dynamically allocating shared memory does not affect bank conflicts.

___

### Padding Statically Declared Shared Memory

Shared memory padding can be used to resolve bank conflicts for rectangular shared memory. For Kepler devices, it is necessary to calculate the number of padding elements needed.

A macro, `#define NPAD 2` , is used to define the number of padding columns added to each row. The padded static shared memory is declared as 

`__shared__ int tile[BDIMY][BDIMX + NPAD];`


A kernel named setRowReadColPad is similar to setRowReadCol but includes shared memory padding. Operations involve mapping 2D thread index to linear memory, converting index to transposed (row, col), and performing shared memory store and load operations.

Experimenting with different values of NPAD and changing the number of padding data elements (from two to one) affects the reported memory transactions.

___

### Padding Dynamically Declared Shared Memory

Padding techniques can be applied to dynamic shared memory kernels with rectangular shared memory regions.

Code Example
```cpp
__global__ void setRowReadColDynPad(int *out) {
// dynamic shared memory
extern __shared__ int tile[];
// mapping from thread index to global memory index
unsigned int g_idx = threadIdx.y * blockDim.x + threadIdx.x;
// convert idx to transposed (row, col)
unsigned int irow = g_idx / blockDim.y;
unsigned int icol = g_idx % blockDim.y;
unsigned int row_idx = threadIdx.y * (blockDim.x + IPAD) + threadIdx.x;
// convert back to smem idx to access the transposed element
unsigned int col_idx = icol * (blockDim.x + IPAD) + irow;
// shared memory store operation
tile[row_idx] = g_idx;
// wait for all threads to complete
__syncthreads();
// shared memory load operation
out[g_idx] = tile[col_idx];
}
```

The code includes calculations for g_idx, irow, icol, row_idx, and col_idx. The kernel uses dynamic shared memory (extern __shared__ int tile[]). The shared memory store operation is performed with tile[row_idx] = g_idx. The shared memory load operation is executed with out[g_idx] = tile[col_idx].

The purpose of shared memory padding is to reduce transactions per request, as confirmed by the reported metrics:
- shared_load_transactions_per_request: 1.000000
- shared_store_transactions_per_request: 1.000000

The successful reduction in transactions per request indicates that shared memory padding works as expected.

___

### Comparing the Performance of the Rectangular Shared Memory Kernels

The nvprof command is used to measure the elapsed time for all kernels implemented in this section using rectangular arrays.

Kernels using shared memory padding gain performance by removing bank conflicts.
Kernels with dynamic shared memory report a small amount of overhead.

Comparing the following Kernels
- setRowReadColDyn
- setRowReadColDynPad
- setRowReadCol
- setRowReadColPad
- setRowReadRow.

The nvprof output provides the elapsed time breakdown for each kernel, including the percentage of time, total time, number of calls, average time, minimum time, and maximum time.

The contents of the 2D matrix generated by each kernel are listed, showing the results of the original matrix and the transpose operations performed by the other kernels using rectangular shared memory arrays.Showing the grid and block configurations for the kernels, specifying the dimensions used in each case.

## Comments and Opinions

At the end comparing all of the kernels used in these passages we can see that the one that takes the longest is the setRowReadColDyn while the fastest is the RowReadRow which isnt dynamic. All the kernels are used depending on alot of varients, wheather what hardware you are using and what software you are going to use of run (if its code). In conclution we can say that padding does reduce even if its a little of the time that its used to process the kernel, both for dynamic or static kernels.