# Resumen Capitulo 4

Programación de paralelismo

Paul Andres Solis Villanueva

University of Advanced Technologies

Leonardo Juárez Zucco

08/11/2023

Programming in Parallel with CUDA

# INTRODUCING THE CUDA MEMORY MODEL

Memory access and management are crucial aspects of programming languages, with a significant impact on high-performance computing in modern accelerators. 

High-speed data loading and storage are essential for many workloads, and having low-latency, high-bandwidth memory can enhance performance. However, obtaining large, high-performance memory is not always possible. In such cases, optimizing latency and bandwidth using the memory model becomes important, considering the hardware memory subsystem. 

The CUDA memory model integrates host and device memory systems, providing explicit control over data placement in the entire memory hierarchy to achieve optimal performance.

## Benefits of a Memory Hierarchy

Locality in computer applications, where they tend to access specific data or code within their address space. This principle includes temporal locality, where frequently used data is revisited, and spatial locality, where nearby memory locations are also accessed.

To optimize performance, computers used today use a memory hierarchy with various memory levels 

They differ in 

* latencies
* capacities
* costs

Frequently used data resides in low-latency, low-capacity memory, while less frequently accessed data is stored in high-latency, high-capacity storage.

Both CPUs and GPUs follow similar memory hierarchy principles, but the CUDA programming model offers more control over GPU memory behavior, allowing programmers to fine-tune performance for their specific requirements.

### Opinions and Comments

We can see in the section that there’s many types of memories for many different uses. Some memories are faster than others, some have more latency, some have more capacity, some have more cost. Others have more space, others have limited space so here we have to figure out which ones we want to use for what reasons.

## CUDA Memory Model

Two types of memory: programmable and non-programmable. 

- Programmable memory allows explicit control over data placement 
- non-programmable memory relies on automatic techniques for data placement.

In the CPU memory hierarchy, L1 and L2 cache fall under non-programmable memory. However, the CUDA memory model provides various types of programmable memory, including

- registers 
- shared memory 
- local memory
- constant memory
- texture memory
- global memory. 
 
These memory spaces have different scopes, lifetimes, and caching behaviors.

#### Registers

Registers on a GPU are the fastest memory space and are primarily used for storing automatic variables in kernels without additional type qualifiers. Arrays can also be stored in registers if their indices are constant and known at compile time.

Register variables are private to each thread and have the same lifetime as the kernel they are declared in. 

The number of registers per thread varies across GPU architectures, with 63 registers on Fermi GPUs and 255 on Kepler GPUs. Using fewer registers in kernels can lead to improved performance by allowing more thread blocks to coexist on an SM, increasing occupancy.

If a kernel exceeds the hardware register limit, the surplus registers spill over to local memory, potentially impacting performance negatively. Efficient management of registers and resources is critical for achieving peak GPU performance in computational tasks.

#### Local Memory

Variables in a kernel that exceed the allocated register space for that kernel will spill into local memory. This can happen to various types of variables, including local arrays referenced with indices whose values cannot be determined at compile-time, large local structures or arrays that would consume too many registers, and any variable that surpasses the kernel's register limit.

Local memory accesses are characterized by high latency and low bandwidth. These characteristics make local memory subject to the requirements for efficient memory access, which are described in detail in the section titled "Memory Access Patterns".

Furthermore, for GPUs with compute capability 2.0 and higher, local memory data is also cached in a per-SM L1 and per-device L2 cache. This caching mechanism can help improve the access speed to local memory on these GPUs.

#### Shared Memory

Shared memory, marked with the __ shared __ attribute, resides on the GPU chip, offering high bandwidth and low latency compared to other memory types. However, each SM has limited shared memory, so overuse can lower performance by reducing the active warp count.

Shared memory is scoped within a kernel function but shares its lifetime with a thread block. 

Upon a thread block's completion, its shared memory allocation is released for reassignment to other blocks. This enables inter-thread communication by sharing data. 

Make sure to synchronized using the __syncthreads() function to prevent data hazards and potential performance issues.

Using the cudaFuncSetCacheConfig function allows for flexibility in determining memory partitioning based on the kernel's specific needs, with various cache configurations available.

#### Constant Memory

Constant memory is a special type of memory that resides in the device memory and is cached in a dedicated constant cache per SM. To declare a variable as constant, the "constant" attribute is used. It's important to note that constant variables must be declared with a global scope outside of any kernels.

Constant Memory only has a maximum of 64KB of storage for all compute capabiliteies. This memory is declared statically and is visible to all kernels in the same compilation unit.

Kernels in CUDA can only reading from constant memory. Therefore, constant memory needs to be initialized by the host using the "cudaMemcpyToSymbol" function. This function copies data from the host's memory to the memory pointed to by the constant variable on the device.

Optimal performance of constant memory is achieved when all threads within a warp read from the same memory address.However, when threads within a warp read from different memory addresses and only do so once, constant memory wont be the most efficient choice. This is because a single read from constant memory broadcasts the data to all threads in the warp, and the benefits of constant memory caching are not fully used.

#### Texture Memory

Texture memory is stored in device memory and cached in a per-SM, read-only cache. It's a type of global memory accessed through a dedicated read-only cache with hardware filtering capabilities. 

Texture memory is optimized for 2D spatial data, providing the best performance for threads in a warp that access 2D data. While it offers a performance advantage for some applications due to caching and filtering hardware, it can be slower than global memory for other applications.

#### Global Memory

Global memory in GPU programming is significant due to its size and common use, despite having high latency. It can be accessed by any SM throughout an application's lifetime. 

Global variables can be declared statically or dynamically. 

Concurrent access by multiple threads from different blocks can lead to undefined program behavior.

Global memory transactions use 32-byte, 64-byte, or 128-byte memory transactions, which must be naturally aligned. Optimizing transactions is vital, as it depends on memory address distribution and alignment. The number of transactions and throughput efficiency are influenced by the device's compute capability.

#### GPU Caches

GPU caches, similar to CPU caches, are a type of non-programmable memory. In GPU devices, there are four distinct types of caches: 

- L1
- L2
- Read-only constant
- Read-only texture.

Each SM is equipped with one L1 cache, and there is a single L2 cache that is shared among all SMs. Both the L1 and L2 caches serve the purpose of storing data in local and global memory, including handling register spills.

Unlike CPUs, where both memory loads and stores can be cached, GPUs function differently. On GPUs, only memory load operations can be cached, while memory store operations cannot be cached.

Each SM in a GPU has two specialized caches designed to enhance read performance from specific memory spaces in device memory. These are the read-only constant cache and the read-only texture cache. These caches are instrumental in optimizing read operations from their respective memory spaces.

#### Static Global Memory

A key point highlighted is the strict separation between host and device code in CUDA, preventing them from directly accessing each other's variables. There is a workaround to allow host code to access a device global variable using cudaMemcpyToSymbol, but it operates through the CUDA runtime API rather than directly accessing the variable's address.

Limitation of using cudaMemcpy to transfer data into a device variable due to device variables being represented as symbols are shown. Additionally, it mentions an exception to this rule in CUDA pinned memory, which can be accessed directly by both host and device code. 

### Opinions and Comments

In the section where showed the register, the shared, the local, the constant, the texture and the global memory. These are used for different scopes. They have different lifetimes, and they can change their behavior. It explains that the CPU has the L1, the L2 and their non-programmable memories, and they once mentioned before our program old memories. We go through all the memories starting at the register. It is the fastest memory, but it has the least amount of space. The local memory is used when the register memory is full, so for bigger variables. The shared memory is used for all the thread blocks to use. It’s in the GPU chip, which means it’s fast, but not as fast as the register memory. We have the constant memory which has 64 kB and is a static memory, the GPU can only read from this memory. the texture memory which is optimized forward to the data so more matrices. And we have the global memory which can be either dynamic or static with high latency, but it also has a lot of space.

# MEMORY MANAGEMENT

## Memory Allocation and Deallocation

In this model, the host and device have their own separate memory spaces. Kernel functions operate within the device memory space, and the CUDA runtime provides essential functions for allocating and deallocating device memory. 

The cudaMalloc() function is used to allocate global memory on the host, which reserves a specific amount of memory on the device and provides a pointer to it. This allocated memory is appropriately aligned for various variable types. 

However, cudaMalloc() can return cudaErrorMemoryAllocation in case of failure.

The memory allocated using cudaMalloc() is not automatically cleared. Programmers are responsible for populating this memory either by transferring data from the host or initializing it using the 

- cudaMemset() function, which can fill a specified number of bytes in device memory with a given value. When the application no longer requires a piece of allocated global memory 
- it can be deallocated using cudaFree(). 

It's essential to ensure that the memory was previously allocated using appropriate functions, such as cudaMalloc, as cudaFree() will return cudaErrorInvalidDevicePointer if used incorrectly or on already freed memory.

Importantly, device memory allocation and deallocation are costly operations.

### Opinions and Comments

In the section, we can see that the host and the device are after separate memory spaces. The GPU has its space that he uses to run functions for Cuda for allocating and deallocating memory. Memory allocation is used with cudaMalloc the problem with it and said it’s not automatically clear so we had to use a CUDA free in order to free it.


## Memory Transfer

In CUDA, global memory allocation is a critical step, and data transfer between the host and device is facilitated through the cudaMemcpy function, which copies a specified number of bytes from the source to the destination memory location. 

The direction of the copy is determined by the kind parameter, with options like 
- cudaMemcpyHostToHost
- cudaMemcpyHostToDevice
- cudaMemcpyDeviceToHost
- cudaMemcpyDeviceToDevice

It's important to note that cudaMemcpy typically exhibits synchronous behavior, meaning it may block until the data transfer is complete.

The kernel includes an example of memory transfer where global memory is allocated using cudaMalloc, and data is transferred to the device using cudaMemcpy with the direction specified as cudaMemcpyHostToDevice. Data is transferred back to the host using cudaMemcpy with a direction of cudaMemcpyDeviceToHost.

There is a substantial difference in bandwidth between the GPU and its memory, compared to the bandwidth between the CPU and GPU through the PCIe bus. Excessive data transfers between the host and device can lead to reduced application performance. 

### Opinions and Comments

In the section we look at memory transfer with the use of cudaMemcpy, which can be used as host host host to device, device to host, device to device. Also to recall there’s a difference with the bandwidth on the GPU and the memory, meaning that with a lot of data sending between each other and can lower the performance so we have to be careful with that.


## Pinned Memory

By default, host memory is pageable (it can undergo page fault operations that relocate data within host virtual memory under the control of the operating system). Virtual memory provides the illusion of more main memory than physically available, similar to how an L1 cache offers the illusion of more on-chip memory. However, GPUs cannot safely access pageable host memory because they lack control over when the operating system may move this data.

When transferring data from pageable host memory to device memory, the CUDA driver first allocates temporary page-locked or pinned host memory. 

The source host data is copied to this pinned memory, which is then used for transferring data to device memory. This approach significantly enhances data access bandwidth for the device. However, dont use excessive allocation of pinned memory, as it may impact the overall performance of the host system by reducing the available pageable memory for virtual memory data storage.

To make easier the allocation of pinned host memory, use cudaMallocHost function and underscores its importance in achieving improved data transfer performance. The code shows that switching from pageable to pinned host memory reduces the transfer time significantly, from 6.94 ms to 5.3485 ms.

### Opinions and Comments

In the section, I find it kind of weird that with virtual memory you can make the hardware think there’s more memory when there’s not. So as it says, it would be bad if you use all of it since it doesn’t really exist. I don’t see how it would work. I don't get how it works, but it would still be interesting to learn it since you’re tricking the device to say that it has more storage than it actually does.

## Zero-Copy Memory

This type of memory permits both the host and device to access the same memory, making it valuable for CUDA kernels. 

Key advantages include the ability to use host memory when device memory is insufficient, eliminate the need for explicit data transfers, and enhance PCIe transfer rates. However, it's important to synchronize memory accesses between the host and device to prevent undefined behavior. 

Zero-copy memory is essentially pinned (non-pageable) memory mapped into the device's address space, allocated using cudaHostAlloc with different flags, with cudaHostAllocMapped being particularly relevant. The cudaHostGetDevicePointer function allows getting the device pointer for mapped pinned memory.

Zero-copy memory is useful for small data sharing between the host and device but can significantly slow performance with frequent read/write operations, primarily due to PCIe bus latency. In practice, it's suggested that zero-copy memory is suitable for scenarios involving limited data sharing, while for larger datasets, especially with discrete GPUs connected via the PCIe bus, it may not be the most efficient choice and could lead to performance issues.

Integrated architectures, which use CPUs and GPUs on a single chip, benefit significantly from zero-copy memory, as it eliminates data transfers over the PCIe bus, enhancing performance and programmability. 

In contrast, discrete systems, where devices connect to the host through PCIe, find zero-copy memory advantageous in specific cases. However, in integrated architectures, synchronization of memory access is essential to prevent data hazards. It's important not to excessively rely on zero-copy memory since device kernels may suffer from slow performance due to high-latency issues.

### Opinions and Comments

Zero copy memory it’s kind of interesting since it kind of goes with the M1 processors the MacBook Pros have, which makes sense why it has higher speeds than the regular computer. Since both the graphic and the processor are in the same chip, it makes sense to have some processes that do things our way faster.

## Unified Virtual Addressing

UVA is a feature supported by devices with compute capability 2.0 and later, first introduced in CUDA 4.0, and it is compatible with 64-bit Linux systems. 

UVA fundamentally changes the way memory is managed by allowing host memory and device memory to coexist within a single virtual address space. This means that the distinction between pointers referring to host memory and those referring to device memory becomes invisible to the application code.

One significant advantage of UVA is that it ensures that pinned host memory, allocated using cudaHostAlloc, has identical host and device pointers. As a result, you can directly pass the returned pointer to a kernel function, getting rid of the need to separately acquire the device pointer and manage two different pointers for the same data.

In essence, UVA is a valuable feature that enhances the efficiency and clarity of CUDA programming by unifying host and device memory in a shared virtual address space, ultimately leading to improved code readability and maintainability.

### Opinions and Comments

Here we were using even more virtual things, tricking the computer that the host memory, and the device memory are in the same area. So the distinction and pointers are not there for this way of managing memory. So I can see now how virtual addressing is really helpful in these types of situations but I still don’t see how the hardware can be tricked since it’s a physical thing.

## Unified Memory

In CUDA 6.0, Unified Memory was introduced to simplify memory management in the CUDA programming model. It creates a shared memory pool accessible from both CPU and GPU with a single memory address, automating data migration for improved performance and simplified coding. 

Unified Memory relies on Unified Virtual Addressing (UVA) support but stands out with automatic data migration, unlike UVA. Managed memory is automatically managed by the system and can be used alongside device-specific allocations.

Managed memory can be allocated statically or dynamically and offers the benefits of automatic data migration and duplicate pointer elimination.

In CUDA 6.0, device code cannot call cudaMallocManaged; all managed memory must be allocated from the host or declared in global scope.

### Opinions and Comments

Unified memory uses what we used before with the unified virtual addressing and it’s for later versions. What I see here is that it can be either said to be static or dynamic and depending on how to use it it offers benefits. I would like to see a better example maybe in class if we have time but besides that it's still confusing for me.

# MEMORY ACCESS PATTERNS

## Aligned and Coalesced Access

Global memory, which serves as a logical memory space accessible from GPU kernels, with the initial data residing in physical DRAM. Memory requests typically flow between device DRAM and on-chip SM memory using either 128-byte or 32-byte transactions, and all accesses pass through the L2 cache, with the option of passing through the L1 cache, depending on the GPU's architecture.

The L1 cache operates in 128-byte lines that align with device memory segments, making aligned memory access a crucial consideration. 

Aligned memory access ensures that the first address of a memory transaction is an even multiple of the cache granularity, meaning it minimizing wasted bandwidth. Coalesced memory access is emphasized, where all 32 threads in a warp access a contiguous chunk of memory, ultimately enhancing global memory throughput.

To maximize global memory performance, it is essential to organize memory operations to be both aligned and coalesced. Misaligned and uncoalesced memory access may need multiple transactions, leading to wasted bandwidth.

### Opinions and Comments

Here is something I can agree on, in order to have better global memory performance you have to have everything organized with that said aligned and coalesced . If it’s not that way, then it would make performance worse.

## Global Memory Reads

#### Three Cache/Buffer Paths:

SM data can be pipelined through three cache/buffer paths: 

- L1/L2 cache
- constant cache
- read-only cache.

L1/L2 cache is the default path, while the other two paths require more management by the application.

#### Factors Affecting L1 Caching for Global Memory Loads:

Whether global memory load operations pass through the L1 cache depends on two factors: 
- device compute capability
- compiler options.

On Fermi GPUs (compute capability 2.x) and Kepler K40 or later GPUs (compute capability 3.5 and up), L1 caching for global memory loads can be enabled or disabled with compiler flags.

#### L1 Cache Management:

The L1 cache can be disabled with the compiler flag "-Xptxas -dlcm=cg" or explicitly enabled with "-Xptxas -dlcm=ca."

When L1 cache is disabled, load requests to global memory go directly to the L2 cache.

#### Cached Loads:

Cached load operations pass through L1 cache and are serviced by device memory transactions at the granularity of an L1 cache line (128 bytes).

#### Uncached Loads:

Uncached loads do not pass through the L1 cache and are performed at the granularity of memory segments (32 bytes), which can lead to better bus utilization for misaligned or uncoalesced memory accesses.

#### Analysis of Misaligned Accesses:

The use of offsets to misalign memory loads can significantly impact global load efficiency, causing additional memory transactions and decreased performance.

#### Disabling L1 Cache:

Disabling the L1 cache for global memory loads with the "-Xptxas -dlcm=cg" flag can affect performance, particularly for misaligned access patterns.

#### Read-Only Cache:

For GPUs of compute capability 3.5 and higher, the read-only cache can support global memory loads as an alternative to the L1 cache.

The granularity of loads through the read-only cache is 32 bytes, making it useful for scattered reads.

#### Directing Memory Reads to Read-Only Cache:

Two ways to direct memory reads through the read-only cache: 

- using the function "__ldg"
- applying const restrict qualifiers to pointers.

### Opinions and Comments

In this section of global memory reads, we can see the cache used for the L1 L2, the constant and the read-only. If it goes through the L1, it’s either a device computer capability or through the compiler options. I still see a lot of details. Every single thing in part of the CUDA code has to go through and I’m slowly losing faith on how to do it. I’m trying to understand if they’re still just a lot of information. I feel like it’s still a lot better with practice.

## Global Memory Writes

Transaction granularity, a single four-segment transaction is more efficient than two one-segment transactions for addresses within the same 128-byte region. Alignment's impact on memory access is showning difference between aligned and misaligned accesses. 

The example illustrates the effects of misalignment on memory store efficiency, and sample outputs with different offsets (0, 11, 128) demonstrate performance variations. The use of nvprof are for assessing efficiency and reveals that, in the misaligned case with an offset of 11, store efficiency drops to 80 percent due to a combination of four-segment and one-segment transactions for a 128-byte write request. 

### Opinions and Comments

Here in global memory writes, we can see many examples which kind of help with your not getting part. We can see how it affects the misalignment of memory and what the outputs are. I can see it, I can read it, I get it visually, but I feel when I put it to practice it’s still not going to be there.

## Array of Structures versus Structure of Arrays

AoS and SoA Data Organization:

As a C programmer, it's important to understand two ways of organizing data: 

- Array of Structures (AoS) 
- Structure of Arrays (SoA)

These approaches leverage the struct and array data types to store structured data efficiently.

In the AoS approach, you define a structure named innerStruct containing float fields x and y.
Data is stored in an array of these structures, spatially close together, which promotes good cache locality on the CPU. Storing data in AoS format can result in a 50 percent loss of bandwidth when only one field (e.g., x) is accessed, as both fields are loaded into cache.

In the SoA approach, you define a structure named innerArray with separate arrays for each field (x and y).
Data for each field is stored in its own array, improving GPU memory bandwidth utilization and coalesced memory access.
Storing data in SoA fashion is more efficient for GPU memory access compared to AoS.

Accessing data in AoS and SoA layouts has performance implications.

Two kernels are compared: 

- AoS data layout
- SoA data layout

The C code for a kernel (testInnerStruct) implemented with an AoS layout.
It defines input length, allocates host and device memory, and performs data initialization.

A warm-up kernel is used to measure the performance of the testInnerStruct kernel.

The C code for a kernel (testInnerArray) implemented with an SoA layout.
It defines input length, allocates global memory, and provides a similar code structure as the AoS example.

Performance is slightly better with SoA, especially for larger input sizes. Using SoA data layout results in 100 percent efficiency for global load and store operations.

### Opinions and Comments

Now we’re in data organizations, we're using the AOS and the SOA. In the examples in the AOS we use inner struct while in the SOA we use inner array. The inner array improves memory and memory access while in the AOS there specially closer together which promotes good cache locality. I can see that somehow, if their benefits and somehow up there downs.

## Performance Tuning

There are two main goals:

- Aligned and Coalesced Memory Accesses: Make sure data access is efficient and well-organized to reduce memory waste.
- Concurrent Memory Operations: Ensure there are enough simultaneous memory operations to hide delays in accessing data.

"Unrolling Techniques" is where loops with memory operations are expanded to improve performance. It's shown that unrolling can significantly boost speed.

To find the best setup for a kernel, you should experiment with grid and block sizes. The ideal block size might not always be the one with the most parallelism due to hardware limitations.

Maximizing the performance of device memory operations hinges on two key factors. 

First, efficient utilization of memory bandwidth is vital, requiring well-organized and coalesced memory access patterns to minimize waste and ensure data transfer in a streamlined manner. 

Second, enhancing the number of concurrent memory operations is crucial, achievable through techniques like unrolling to boost parallelism and modifying kernel launch configurations to expose more parallelism to Streaming Multiprocessors. 

### Opinions and Comments

Performing tuning I can see that I will be stuck here a lot, sending it to trial and error to see which settings are best for your hardware. Since there’s no real example or something you can copy, you really have to go through the code and try many different versions of what block sites are you trying to do in order to get the most optimized version of it.

# WHAT BANDWIDTH CAN A KERNEL ACHIEVE?

## Memory Bandwidth

Memory Bandwidth Sensitivity: Kernels are typically highly dependent on memory bandwidth, which means that they are often limited by the rate at which data can be read from and written to memory.

Focus on Memory Bandwidth Metrics: When optimizing kernels, it is important to pay attention to memory bandwidth metrics. This involves understanding how data in global memory is organized and accessed by the kernel.

Types of Bandwidth:

- Theoretical Bandwidth: This represents the maximum bandwidth that the hardware can theoretically achieve. 
- Effective Bandwidth: This is the actual measured bandwidth that a kernel attains in practice. It is calculated using a specific formula:

   ``` 
   effective bandwidth (GB/s) = (bytes read + bytes written) × 10^9 / time elapsed 
   ```

Example Calculation: Using the example of how to calculate effective bandwidth for a specific scenario, involving a 2048 x 2048 matrix containing 4-byte integers transferred to and from the device. The formula used is:

   ``` 
   effective bandwidth (GB/s) = (2048 * 2048 * 4 * 2 * 10^9) / time elapsed 
   ```

### Opinions and Comments

Here we see two equations for memory bandwidth, which basically means what maximum bandwidth theoretically the hard-working reach and we use a specific scenario involving a 2000 x 2000 matrix and use the formula in order to calculate how much bandwidth is gonna have.

## Matrix Transpose Problem

Matrix transpose involves exchanging each row with the corresponding column.
It is a basic operation in linear algebra but finds applications in various fields.

A host-based implementation of an out-of-place transpose algorithm for single-precision floating-point values is provided.
The function transposeHost is shown, which calculates the transpose of a matrix stored in a 1D array.

The input and output layouts are compared, highlighting that reads are accessed by rows in the original matrix, leading to coalesced access, while writes are accessed by columns in the transposed matrix, resulting in strided access.
Strided access is discussed as the worst memory access pattern for performance on GPUs.

Enabling L1 cache can impact performance in transpose operations. Two versions of the transpose kernel are introduced: 

- one that reads by rows and stores by columns 
- one that reads by columns and stores by rows.

The second version may perform better when L1 cache is enabled because it can take advantage of caching.

Before implementing the matrix transpose kernel, it is recommended to create upper and lower bound kernels to assess performance.
The performance of these copy kernels is compared to theoretical peak bandwidth.

Two naive transpose kernels are presented: 

- one that loads by row and stores by column
- one that loads by column and stores by row.

Diagonal-based transpose kernels are introduced, where block IDs are interpreted differently to optimize memory access patterns.
Performance of diagonal-based kernels is compared to their Cartesian counterparts.

Adjusting the block size is shown to impact performance significantly.
Using "thin" blocks with dimensions (8, 32) is found to be optimal for certain implementations, even surpassing the upper-bound copy kernel in performance. The effective bandwidth achieved by various kernels is measured and compared in terms of their ratio to peak bandwidth.

### Opinions and Comments

Here we use the Transpose problem, which we did an exercise on for tests, and basically shows you two ways function can be introduced. And all of them show how I read the rows and stores by columns, and then the other one shows how it reads by columns and stores by rows. And at the end, we can see that it’s faster to do it the second way. 

# MATRIX ADDITION WITH UNIFIED MEMORY

Unified Memory in CUDA programming is emphasizing the role in simplifying memory management for GPU applications. Always do revisions to the main function of a matrix addition program, aimed at leveraging Unified Memory effectively. These revisions include replacing host and device memory allocations with managed memory allocations and eliminating explicit memory copies. 

Managed memory is allocated for input matrices (A and B) and an output array (gpuRef) using cudaMallocManaged. The input matrices are then initialized on the host, and the matrix addition kernel (sumMatrixGPU) is invoked with pointers to managed memory, with synchronization achieved using cudaDeviceSynchronize().

Unified Memory offers several advantages, such as reducing code complexity and eliminating explicit memory copies. The passage compares the performance of the managed memory approach with a manual memory management version (sumMatrixGPUManaged.cu and sumMatrixGPUManual.cu). Both versions run a warm-up kernel to enhance timing accuracy. The managed version's performance is demonstrated to be nearly as fast as the manual version, which explicitly manages memory.

Profiling tools like nvprof to analyze Unified Memory performance can be helpful to analize your prosses. It provides metrics related to data transfer between the host and the device, including bytes transferred and CPU page faults. 

### Opinions and Comments

At the end of the chapter, we see how simplifying memory management for GPU applications is important. We use a cudaMalloc manager in order to allocate memory from two matrices and an output array, the advantages it says it’s that it has less code complexity, and eliminates explicit memory copies, which leaves more memory space for the rest of the process. Profiling tools also work that we used on the test and it helps you know what is less optimal or what uses more time to be executed for your code.