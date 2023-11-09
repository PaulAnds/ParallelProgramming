# Resumen Capitulo 3

Programación de paralelismo

Paul Andres Solis Villanueva

University of Advanced Technologies

Leonardo Juárez Zucco

04/11/2023

Programming in Parallel with CUDA

# INTRODUCING THE CUDA EXECUTION MODEL

## GPU Architecture Overview

The Sm in a gpu are made to be able to execute hundreds of threads. In a GPU there are multiple SM’s. The parameters in a kernel specify how many threads and blocks are used to be distributed on the SMs for the execution. There are multiple blocks in an SM. The threads in the blocks work individually to work on the overall process that the SM is asked to do.

CUDA uses SIMT (Single instruction multiple thread) architecture with warps, which are groups of 32 threads. All threads in a warp execute the same instruction at the same time, making it parallelism. SIMT is similar to SIMD (Multiple data), where they both use parallelism by running instructions to multiple execution units. The difference is that in a SIMD requires that the elements execute together in a unified synchronous group, in comparison to SIMT where multiple threads in a warp execute independently. This lets you write thread level parallel code as well as data-parallel code which makes it faster than SIMD.  

Key Differences:

* Each thread has its own instruction address counter
* Each thread has its own register state
* Each thread can have an independent execution path

A thread block is scheduled on one SM:
Visually it would be like this

Software Hardware

Thread = CUDA Core

Thread Block = SM

Grid = Device

Shared memory and registers are crucial resources. Shared memory is divided among thread blocks, and registers are assigned to individual threads. Threads within a block can cooperate, but they don't always execute simultaneously, leading to varying progress and latency. CUDA allows synchronization within a block but lacks inter-block synchronization mechanisms.

Warps within a block can be scheduled in any order, limited by SM resources. When a warp idles, the SM can schedule another available warp without overhead, as resources are shared among all threads and blocks on the SM.

### Opinions and Comments

At first I thought that the SIMD was gonna be better since it was doing the whole data at once, but it makes more sense that the SIMT is better since it synchronizes the threats instead of the whole data. This makes it so that the use of Cuda makes the programming way better and faster than it would be by using a regular CPU.

## The Fermi Architecture


The Fermi architecture was a significant advancement in GPU computing, providing the features necessary for demanding HPC ( high-performance computing) applications. It introduced   

* 512 CUDA cores in 16 SM 
* 384-bit GDDR5 DRAM memory interfaces
* 6 GB of on-board memory
* host interface connected the GPU to the CPU via the PCI Express bus
* GigaThread engine served as a global scheduler
* 768 KB L2 cache shared by all 16 SMs.  

Each SM consisted of 

* execution units (CUDA cores) 
* scheduler and dispatcher units
*  shared memory
*  a register file
*  L1 cache 

SFUs (Special function units) were used for intrinsic instructions. Each SM had two warp schedulers and two instruction dispatch units, enabling the simultaneous handling of 48 warps per SM. One feature of Fermi was the 64 KB on-chip configurable memory, distributed between shared memory and L1 cache, making better  performance for high-performance applications. The Fermi architecture also supported kernel execution, allowing multiple kernels to run on the same GPU simultaneously, making it more versatile for programmers.

### Opinions and Comments

Opinions about the Fermi in architecture are normal, from what I know about computers. This seems a little slower also with a knowledge to already have a future architecture, but it still seems good by the time it came out. He uses the shared memory register file all the memories of the CUDA Cores would need And I’m pretty sure it made a lot of big advancements on its time.

## The Kepler Architecture

The Kepler GPU architecture, released in 2012 introduces key features such as 

   * Enhanced SMs
   * Dynamic Parallelism 
   * Hyper-Q. 

The Kepler K20X chip includes 15 streaming multiprocessors (SMs) and six 64-bit memory controllers.
 Each Kepler SM unit 

   * 192 single-precision CUDA cores
   *  64 double-precision units
   *  32 special function units (SFU)
   *  32 load/store units.

 It also features four warp schedulers and eight instruction dispatchers per SM, allowing concurrent execution of four warps.


Dynamic Parallelism is a new feature that enables dynamic launching of grids and nested kernels on the GPU, making it easier to optimize recursive and data-dependent execution patterns. 

Hyper-Q increases hardware connections between the CPU and GPU, allowing CPU cores to run more tasks on the GPU concurrently, improving GPU utilization and reducing CPU idle time.

Kepler GPUs provide 32 hardware work queues, enabling more concurrency on the GPU, leading to increased overall performance. Overall, the Kepler architecture offers significant improvements in power efficiency, programmability, and performance compared to previous Fermi designs.

### Opinions and Comments

Comparing the Kepler architecture with the Fermi architecture Comparing the Kepler architecture with the Furman architecture you can tell which ones are better. The Kepler architecture has way more cores for control logic as soon as the image that I put, and contain multiple order cubes at the same time, and compare it to the Fermi . From the looks of it, it looks like the Kepler almost doubled processing power, and made advancements way faster than if they were to use the other architecture.

## Profile-Driven Optimization

Profiling involves analyzing program performance by measuring aspects like 

   * memory and time complexity
   *  instruction usage
   *  function call patterns. 

It's crucial for optimizing HPC application code. Developing an HPC application usually involves two major steps:

   * Developing the code for correctness
   * Improving the code for performance

In CUDA programming, profile-driven development is essential because it helps identify performance bottlenecks(points in a system or process that restrict its speed or efficiency, preventing it from performing at its full potential) and resource utilization. Two primary profiling tools in CUDA are nvvp (visual profiler) and nvprof (command-line profiler).

Nvvp visualizes program activity on both the CPU and GPU, identifies bottlenecks, and suggests actions for improvement. 

Nvprof collects data on the command line, including kernel execution and memory transfers. Selecting appropriate performance metrics and comparing them to theoretical peak performance is key to identifying bottlenecks.

There are three common limiters to performance for a kernel that you may encounter:

   * Memory bandwidth
   * Compute resources
   * Instruction and memory latency

Understanding hardware is essential for improving the performance of code, whether in C or CUDA programming, as it enables programmers to make their code effectively and fully leverage the capabilities of their hardware.

### Opinions and Comments

In this section, we use nvvp and nvprof which help out to identify any limiters of the process in the GPU. I also just learned about bottlenecks from outside because I was trying to make sure my GPU and processor were fine for each other. De limiters mentions are basically just how much memory they have how much you’re hard-working due, and the Latin scene that is made by not having a correct code.

# UNDERSTANDING THE NATURE OF WARP EXECUTION

In the context of launching a kernel, there is a distinction between the software and hardware perspectives of thread execution. While it may appear that all threads in a kernel run simultaneously from a software viewpoint, this is not entirely accurate from a hardware standpoint. The hardware groups 32 threads into a single execution unit called a "warp." 

## Warps and Thread Blocks

In a Single Instruction Multiple Thread (SIMT) execution environment, warps are the basic execution units within a Streaming Multiprocessor (SM). Thread blocks in a grid are divided into warps, where each warp contains 32 consecutive threads executing the same instruction. 

Threads can be organized in one, two, or three dimensions, but they are managed one-dimensionally from a hardware perspective. A unique identifier for each thread in a block is calculated using built-in variables. The number of warps in a thread block can be determined by specific formulas. If thread block size is not an even multiple of warp size, some threads in the last warp are left inactive.  

For example, a one-dimensional thread block with 128 threads will be organized into 4 warps as follows:

Warp 0: thread 0, thread 1, thread 2, ... thread 31

Warp 1: thread 32, thread 33, thread 34, ... thread 63

Warp 3: thread 64, thread 65, thread 66, ... thread 95

Warp 4: thread 96, thread 97, thread 98, ... thread 127

Given a 2D thread block, a unique identifier for each thread in a block can be calculated using the built-in threadIdx andblockDim variables:

threadIdx.y * blockDim.x + threadIdx.x.

The same calculation for a 3D thread block is as follows:

threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x

### Opinions and Comments

There is a magic number which is 32, in other words. There are 32 threats to the execution of a GPU. One thing I found interesting is that you need to have your words divided in a factor of 32 otherwise it’s good to leave out some threads, or have extra space it’s not needed making it not as efficient at it should have been.

## Warp Divergence

Control flow is a fundamental aspect of high-level programming languages, including traditional C-style constructs like 

If(cond) {

} else {

}

for, and while loops.

GPUs and CPUs handle control flow differently. CPUs use complex hardware for branch prediction, which predicts the path an application's control flow will take, reducing performance penalties if the prediction is correct. 

GPUs lack complex branch prediction mechanisms, and all threads in a warp (a group of threads that execute in lockstep) must execute identical instructions, leading to potential issues with warp divergence.  

Warp divergence occurs when threads within a warp take different paths through conditional statements, causing some threads to be disabled. This can reduce parallelism within a warp and impact performance.

Suppose for 16 threads in a warp executing this code, cond is true, but for the other 16 cond is false. Then half of the warp will need to execute the instructions in the if block, and the other half will need to execute the instructions in the else block.

Branch divergence is specific to a warp, and different conditional values in different warps do not cause warp divergence. To avoid warp divergence and achieve optimal GPU utilization, you can distribute data so that all threads in the same warp follow the same control path in an application.

Branch Efficiency is defined as the ratio of non-divergent branches to total branches, and can be calculated using the following formula:
  
Overall:

   * Warp divergence occurs when threads within a warp take different code paths.
   * Different if-then-else branches are executed serially.
   * Try to adjust branch granularity to be a multiple of warp size to avoid warp divergence
   * Different warps can execute different code with no penalty on performance.

### Opinions and Comments

My opinions on the warp divergence is that it’s going to be a little annoying to code without using if since most of the code now, and they involves using, if any else, if I want to get used to this summer, I have to start learning to do things without using conditionals. I’m pretty sure this is what makes a CUDA hard. making people not wanting to do this or challenging people to do it to make things more efficient and faster.

## Resource Partitioning

The local execution context of a warp mainly consists of the following resources:

   * Program counters
   * Registers
   * Shared memory

Each warp processed by a SM on a GPU is maintained on-chip throughout its lifetime, and switching from one execution context to another has no cost. Each SM has 32-bit registers and shared memory, which are distributed among threads and thread blocks. 

The number of thread blocks and warps that can be on an SM at the same time depends on the available registers and shared memory. If a kernel requires too many registers or shared memory and there aren't enough on the SM, the kernel launch will fail.  

A thread block is called active when it has been allocated compute resources. Within an active block, there are active warps, which can be classified into 

   * selected warps (actively executing)
   * stalled warps (not ready for execution)
   * eligible warps (ready for execution).

Warp schedulers on an SM select active warps and dispatch them to execution units. These schedulers help manage and prioritize warps for execution. Warps are eligible for execution if 32 CUDA cores are available, and all required instruction arguments are ready.
Compute resources are distributed among warps and are kept on-chip throughout a warp's lifetime, enabling fast context switching. 

To maximize GPU utilization and hide latency due to warps stalling, it's essential to keep a large number of active warps.

### Opinions and Comments

No, I see what they were talking about with on-chip. It’s faster for data to get there and be processed if you have a memory right next to you and what they were saying about having a smaller and more compact GPU for the reason that if it’s closer to each other it’s gonna process a lot faster. but now, as we talked in class, we have hardware limitations, so the only thing left to do is make faster software more optimized, or go into quantum programming with quantum computers.

## Latency Hiding

Latency is the time between the issuance and completion of an instruction.Utilization of functional units in a GPU is closely tied to the number of resident warps (groups of threads). Full compute resource utilization is achieved when all warp schedulers have an eligible warp at every clock cycle, allowing the latency of instructions to be hidden by other concurrent instructions.

Latency hiding is particularly important in CUDA programming because GPUs are designed to handle many concurrent and lightweight threads to maximize throughput, unlike CPUs, which focus on minimizing latency for a few threads at a time.

Bandwidth vs throughput:

   * Bandwidth typically represents a theoretical maximum data transfer rate, while throughput refers to the actual achieved data transfer rate.
   * Bandwidth measures the highest potential amount of data transferred per unit of time, while throughput can be used to describe the rate of any kind of operations completed per unit of time, such as instructions executed per cycle.

Instructions can be categorized into two types: Arithmetic instructions and Memory instructions.

   * Arithmetic instruction latency is the time it takes for an arithmetic operation to produce its output (typically 10-20 cycles).
   * Memory instruction latency is the time between issuing a load or store operation and the data arriving at its destination (typically 400-800 cycles).

Little's Law, originally from queue theory, is applied to GPUs to estimate the required parallelism to hide latency for arithmetic operations.The required parallelism for arithmetic operations can be shown in terms of the number of operations or the number of warps.

Number of Required Warps = Latency × Throughput

There are two ways to increase parallelism:

   *  Instruction-level parallelism (ILP) by having more independent instructions within a thread 
   * Thread-level parallelism (TLP) by having more concurrently eligible threads.

The required parallelism for memory operations is shown as the number of bytes per cycle needed to hide memory latency. Memory bandwidth is converted from gigabytes per second to gigabytes per cycle using memory frequency. 

An example calculation for Fermi GPUs shows that around 74 KB of memory I/O in-flight is needed to achieve full utilization, requiring a specific number of threads and warps.  

The number of active warps per SM depends on the execution configuration and resource constraints, like the parameters used in a kernel. Balancing these factors help achieve optimal latency hiding and resource utilization.

### Opinions and Comments

Usually in CUDA and everything else relating to the GPU, you want to have the least latency, for that we use optimizations, we fix code, and we try to maximize or throughput. Now I know the difference of latency between CPU and GPU. The CPU focuses on having almost no latency. In comparison with the GPU, it doesn’t so that’s all we have to do manually with code and try to make things faster for the GPU.

## Occupancy

GPU occupancy in CUDA programming measures the utilization of GPU resources. Occupancy is the ratio of active warps to the maximum allowed on a SM. You can calculate it using the formula: 

occupancy = active warps / maximum warps.

To find the maximum warps per SM for your GPU, use cudaGetDeviceProperties. The CUDA Toolkit provides the Occupancy Calculator to help you optimize grid and block dimensions for your kernel.

Managing the number of registers per thread can impact performance; you can control this with --maxrregcount=NUM in the nvcc compiler.

Optimizing thread block configuration and resource usage is essential to enhance occupancy, but it's not the sole factor for performance improvement.

Using these guidelines will help your application scale on current and future devices:

   *  Keep the number of threads per block a multiple of warp size (32).
   *  Avoid small block sizes: Start with at least 128 or 256 threads per block.
   *  Adjust block size up or down according to kernel resource requirements.
   * Keep the number of blocks much greater than the number of SMs to expose sufficient parallelism to your device.
   * Conduct experiments to discover the best execution configuration and resource usage.

### Opinions and Comments

For this section occupancy, it’s just focusing that we are using all the warps, if there are any warps that are not active, that means we are not using every resource we have, making the programs slower. That’s why we want to have as many warps active as we have total warps.

## Synchronization

It operates at two levels: 

   * system-level. waits for all work to complete on the host and device
   * block-level. waits for all threads in a block to reach the same execution point

cudaDeviceSynchronize blocks the host until all CUDA operations are done, potentially returning errors. __syncthreads synchronizes threads within a block, but overuse can harm performance and cause idle warps.

Sharing data among threads requires care to avoid race conditions, where unordered memory accesses occur. Threads in different blocks don't synchronize. Global synchronization points are used at the end of kernel executions to enable scalability across GPUs, allowing blocks to execute in any order.

### Opinions and Comments

Here we show the different types of synchronization, that we have the other system levels, which is basically waiting for the host and device to be on match, and we have the block level which waits for the thread blocks to be all at the same page. One is more general, and one is more specific or more detailed inside the warps.

## Scalability

Scalability is a desirable feature for parallel applications because it means that adding more hardware improves the application's speed.

In the context of CUDA, an application is considered scalable if running on two SMs reduces execution time by half compared to running on one SM.

   * Scalable parallel programs efficiently use all compute resources to enhance performance. They use everything in hardware to effectively execute the software.
   * Scalability depends on both algorithm design and hardware features.

Transparent scalability allows an application to run on varying numbers of compute cores without code changes, making it versatile and reducing the burden on developers.

Scalability is better than efficiency, an example is that, a code that is inefficient but has an scalable system can work better than an efficient code with an unscalable system overtime.

CUDA programs distribute thread blocks among multiple SMs, and the independence of execution order makes them scalable across different numbers of compute cores.  

The image shows how CUDA architecture's scalability works, showing that applications can run on different GPU configurations without code changes, and the execution time scales with available resources.

### Opinions and Comments

I find this true since my computer is not the best I could have really optimized code but with my GPU, it could be slower than someone that has a 4090 TI. That makes me want to try to optimize even more in comparison to someone who has a higher graphics since their code can be run a lot faster just because of what parts they have in their GPU.

# EXPOSING PARALLELISM

The goal is to better understand warp execution in CUDA by analyzing the "sumMatrixOnGPU2D" kernel with different execution configurations.

Grid and block heuristics are essential skills for CUDA programmers.

The provided 2D matrix summation kernel takes two input matrices, A and B, and produces an output matrix C. It operates on large matrices with dimensions of 16,384 elements in each dimension (nx = 1<<14, ny = 1<<14).  

The code allows for configurable block dimensions from the command line, with 'dimx' and 'dimy' being set based on command-line arguments.

The dimensions of the grid and block are calculated based on the input matrix dimensions, ensuring that the kernel processes all elements correctly.

The code should be compiled using the 'nvcc' command with optimization flags and a specific target architecture. The generated "sumMatrix" executable will be used for experimenting with different block and grid configurations in subsequent sections.

### Opinions and Comments

I can see that it’s really important to have the right dimensions for blocks since if you don’t, I could have some left over threads that are not doing anything or half code that is waiting to be processed making a higher latency. For this reason it’s good practice to know your hardware in order for you to specify what year dimensions of your blocks or wipes are going to be.

## Checking Active Warps with nvprof

To assess the performance of a GPU program, it's essential to establish a reference result as a baseline. This is done by testing various thread block configurations. The thread block configurations tested include (32,32), (32,16), (16,32), and (16,16). These configurations determine the dimensions of the thread blocks used in the program.

The performance results of running the "sumMatrix" program with these thread block configurations on a Tesla M2070 GPU are provided. We can see that the slowest is the 32,32 block and the fastest is the 32,16 configuration. The second configuration (32,16) has more thread blocks saying that it exposes more parallelism, which is a likely reason for its better performance.

The achieved occupancy of a kernel, which measures the utilization of GPU resources, is analyzed using nvprof.  The achieved occupancy results for different thread block configurations are provided, and it's observed that the (16,16) configuration has the highest achieved occupancy but is not the fastest, indicating that a higher occupancy doesn't always translate to better performance.

### Opinions and Comments

In the section of checking, active warps, and there is a clear example that the 32,32 was slower than 32,16, at first from plain sight this was really hard for me to process because there were more resources in the 32,32 although explain that it was Facyson 32,16 because they had more thread blocks being used. Which at the end as I bolded it, a higher occupancy doesn’t always translate to better performance. 

## Checking Memory Operations with nvprof

Memory operations in the "sumMatrix" kernel (C[idx] = A[idx] + B[idx]) 

The "sumMatrix" kernel involves three memory operations: two memory loads and one memory store. Memory read efficiency is evaluated using the "gld_throughput", and it varies based on execution configurations. The results show that higher load throughput doesn't always lead to better performance.

Checking global load efficiency using the "gld_efficiency" metric reveals that the load efficiency for configurations with smaller block sizes in the innermost dimension is lower, impacting overall performance. 

The last two cases, with block sizes in the innermost dimension being half of a warp, have lower load efficiency and don't achieve improved performance despite higher throughput.

Grid and block heuristics, the innermost dimension should always be a multiple of the warp size.

### Opinions and Comments

throughput meaning, what actual data was passed through, here in the section we have an example of how higher load through output doesn’t always lead to better performance, and it also shows that we have to be in a factor of 32, which is the warp size.

## Exposing More Parallelism

The innermost dimension of a block (block.x) should be a multiple of the warp size to improve load efficiency.

Testing different thread configurations revealed the following:

   * The (256, 8) block configuration is invalid due to exceeding the hardware limit.
   * The best results were achieved with the (128, 2) block configuration.
   * The (64, 2) configuration, while launching the most thread blocks, was not the fastest.
   * The (64, 4) configuration, having the same number of thread blocks as the best case, underperformed, emphasizing the importance of the innermost dimension of a thread block.

Exposing more parallelism remains crucial for performance optimization. Achieved occupancy metrics were measured to analyze thread block performance. 

Surprisingly, the (64, 2) configuration had the lowest achieved occupancy due to hardware limits on the maximum number of thread blocks. The (128, 2) and (256, 2) configurations, which performed best, had nearly identical achieved occupancy.

Further performance improvements were achieved by setting block.y to 1, reducing the size of each thread block and launching more thread blocks to process the same data. The (256, 1) configuration outperformed (128, 1).

Achieved occupancy, load throughput, and load efficiency metrics were examined to find the best balance of several related metrics for overall performance. It's noted that no single metric is directly equivalent to improved performance, and a balance of metrics is required to achieve the best overall performance.

   * In most cases, no single metric can prescribe optimal performance.
   * Which metric or event most directly relates to overall performance depends on the nature of the kernel code.
   * Seek a good balance among related metrics and events.
   * Check the kernel from different angles to find a balance among the related metrics.
   * Grid/block heuristics provide a good starting point for performance tuning.

### Opinions and Comments

I can see now that exceeding hardware limits will not let you do it. That’s why we have to know how many warps how many SM’s and how many blocks we are good to use for a kernel and we have to see of the three things that we have to check in order to have the most efficient code which are the archived occupancy load throughput, and load efficiency. With those, see how your hardware is gonna work best.

# AVOIDING BRANCH DIVERGENCE

## The Parallel Reduction Problem

Parallel summation of an array of integers with N elements.

Sequential sum algorithm:  

   * Summing elements in a loop.
   * Simple and intuitive for small data.

Need for parallelization with a large dataset:

   * Parallelization can speed up the summation process.
   * Associative and commutative properties of addition enable parallelism.

For parallel addition, distribute the input vector into smaller chunks then calculate partial sums for each chunk using multiple threads and lastly combine partial results to have the final result.

In an Iterative pairwise a chunk contains only a pair of elements. Threads sum two elements in a pair, storing partial results in the input vector, and halves the number of input values in each iteration until a final sum is obtained.
Two types of pairwise parallel sum implementations:

   * Neighbored pair: Elements are paired with their immediate neighbor.
   * Interleaved pair: Paired elements are separated by a given stride.

Recursive implementation of the interleaved pair approach:

   * Continues to reduce the data size recursively.
   * Achieves parallel summation by dividing the input in half at each step.

Generalization of the problem: The reduction problem is any commutative and associative operation can replace addition, like max, min, average, or product. Parallel reduction is the parallel execution of a commutative and associative operation across a vector and a fundamental operation in many parallel algorithms.

### Opinions and Comments

Here we can see another way of optimizing or making more efficient code. I like looking at the figures, are you sure we can visually see how the blocks are processed and the better way of visualizing how they should be executed. In the first one we can see that there’s a lot of open space and the second one we can see a more optimized version of what it does.

## Divergence in Parallel Reduction

A Parallel reduction kernel using the neighbored pair approach. Threads add adjacent elements to create partial sums. Two global memory arrays are employed: 

One for the input array 

One for thread block partial sums. 

Reduction is in-place, and synchronization is ensured by __syncthreads.

The concept of "stride" is used to control which elements are summed in each reduction round. Stride starts at 1 and doubles after each round. Importantly, there is no synchronization between thread blocks, necessitating the sequential summation of partial sums on the host.
  
Code snippets for a CUDA program are provided, demonstrating the main function and kernel configurations. Sample results for CPU and GPU reduction times are shown, serving as a performance baseline for future optimization efforts. This discussion presents the key principles and initial implementation of parallel reduction in CUDA programming.

### Opinions and Comments

Here in the image, we can see that there are many blocks doing many different processes, and they all come to one variable at the end. That’s where it all sums up and it shoots it to the host. This is how to make things faster instead of the host or the CPU. One by one we have the GPU to use its many cores in order to make this a parallel program. We are showing the concept of strides that is used to control what elements are summed in order to send it to the host.

## Improving Divergence in Parallel Reduction

In the kernel "reduceNeighbored," there's a conditional statement: if ((tid % (2 * stride)) == 0. This condition leads to highly divergent warps as it only holds true for even-numbered threads.

The array access index for each thread is adjusted in the new kernel called "reduceNeighboredLess." It sets the index as int index = 2 * stride * tid, effectively eliminating warp divergence.  

This adjustment means that in a thread block with 512 threads, the first 8 warps execute the first round of reduction, while the remaining 8 warps do nothing. In the second round, the first 4 warps do the reduction, and the remaining 12 warps remain idle, eliminating warp divergence.

The new kernel is integrated into the main function after the first kernel call, and it proves to be 1.26 times faster than the original implementation.

"inst_per_warp" measures the average number of instructions executed by each warp, and "gld_throughput" checks memory load throughput, highlighting the improved efficiency of the new implementation.

### Opinions and Comments

Here we can see another way, we are using thread IDs now in order to know what process to go and an order. The image we can see the first eight wipes execute the first time. Then with the thread ID we put the first and the second together and the third and the fourth together, etc., until we end up with one last warp, doing the last process, making this more efficient and organized.

## Reducing with Interleaved Pairs

Offers an alternative method for reduction operations. Unlike the neighboring approach, the interleaved pair reverses the striding of elements. It begins with a stride set at half of the thread block size and then reduces it by half in each iteration. During each round, individual threads are responsible for adding two elements separated by the current stride to generate partial sums.  

The kernel code for interleaved reduction reveals the implementation details. The stride between two elements is initialized at half of the thread block size and is progressively halved with each round.

A performance test comparing the interleaved reduction approach with previously used kernels. The results show that the interleaved implementation is significantly faster, with specific timing data provided. It outperforms the first implementation by 1.69 times and the second by 1.34 times. This enhanced performance is attributed to the global memory load and store patterns in reduceInterleaved.

Notably, the interleaved reduction approach achieves this performance boost while maintaining the same amount of warp divergence as the reduceNeighboredLess approach, indicating that the improved performance does not come at the expense of increased divergence.

### Opinions and Comments

My opinion here it’s that it’s better than the last way of reduction. Instead of grabbing the pairs and combining them in one thread ID, we are grabbing the first four and pairing them with the next four. I feel this is better because they live no space between the Warriors and it’s just a faster way of processing the data that it’s shown.

# UNROLLING LOOPS

Loop unrolling optimizes loop execution by duplicating the loop body in the code, reducing the number of loop iterations. It improves performance by reducing instruction overhead, minimizing condition checks, and allowing simultaneous memory operations. 

In CUDA, unrolling aims to increase concurrent operations and bandwidth utilization, hiding latency for better performance.
Reducing with Unrolling

Cyclic partitioning is where each thread block in the "reduceInterleaved" kernel processes a single data block. It proposes a new kernel, "reduceUnrolling2," which optimizes performance by having each thread block process two data blocks instead of one. This approach reduces memory latency and better utilizes hardware resources, as each thread works on multiple data blocks, processing a single element from each.

The code for the "reduceUnrolling2" with adjustments made to the global array index and execution configuration. The grid size is reduced by half to accommodate the change, resulting in a significant performance boost. The modified kernel runs 3.42 times faster than the original implementation, highlighting the effectiveness of unrolling in enhancing parallel processing efficiency.

Two additional implementations of unrolled kernels in the "reduceInteger.cu" file: "reduceUnrolling4," where each thread block handles four data blocks, and "reduceUnrolling8," where each thread block handles eight data blocks. The results of these implementations consistently show improved performance as the number of independent memory load/store operations in a single thread increases. 

There is a relationship between unrolling and device read throughput and higher unrolling levels resulting in increased memory read throughput. 

### Opinions and Comments

Unrolling code is still something that I’m trying to understand. I know that you’re trying to make it so there’s less loops in order to reduce latency. We saw examples of it as taking one loop and making it into 14 IF’s with no else. This makes it so the code has to run between the functions since the functions already loop, getting rid of latency since it doesn’t have to process the code many times making the threads wait for other threats to finish.

## Reducing with Unrolled Warps

In CUDA programming, "__syncthreads" ensures proper intra-block synchronization, particularly in reduction kernels, where it guarantees that all threads complete writing to global memory before advancing to the next round.

Optimization is achieved by unrolling the last 6 iterations when there are 32 or fewer threads, eliminating the need for synchronization logic. The "volatile" qualifier is essential to maintain data integrity in this multi-threaded environment. A practical example demonstrates how "warp unrolling" significantly boosts kernel performance, resulting in faster execution times.

The "stall_sync" metric confirms the effectiveness of these optimizations by reducing warp stalling during "__syncthreads" synchronization.

In essence, careful synchronization and optimization techniques improve CUDA kernel performance, making programming more efficient and effective.

### Opinions and Comments

We are getting deeper into unrolling, and even getting rid of syncthreads. It’s going to be hard for me to understand this or even put it to practice since the programmer is already understanding how memory is moved in this GPU. I feel like once you get that, Once you understand perfectly, how everything moves and, where everything is then you can code without parameters and without loops.


## Reducing with Complete Unrolling

It's crucial to consider the hardware constraints of Fermi or Kepler GPUs, which impose a limit of 1024 threads per block. The provided CUDA kernel, named reduceCompleteUnrollWarps8, is introduced as a solution. This kernel uses loop unrolling with a factor of 8, allowing the simultaneous processing of eight data elements, thus better efficiency and performance.

Throughout the kernel code, synchronization is carefully managed using multiple calls to __syncthreads(). Synchronization is essential in parallel processing to ensure that threads within a block complete their tasks in an order.

The optimized kernel is reported to execute 1.06 times faster than a previous version and 9.16 times faster than the original implementation. This demonstrates the effectiveness of loop unrolling in improving the efficiency of GPU-based reduction operations.

### Opinions and Comments

And here we can see how the code already makes it, so it’s more than nine times faster than the original code. It’s a lot more specific in the sense that it tells it exactly what it’s going to do. Comparing this to the first one makes it seem that the first one is cleaner, it's better, not so much for a GPU. Since the GPU has a different way of understanding and processing things, it may not be the best or fastest way to do it as in the first one. That’s why we are doing it how we are doing it now, since we have to grab the GPU by hand.

## Reducing with Template Functions

This section discusses techniques for optimizing parallel reduction in CUDA, primarily focusing on template functions to minimize branch overhead and improve performance. Template functions enable efficient code unrolling and dynamic block size specification.

Compile-time evaluation of if statements allows for automatic removal of unnecessary code, ensuring an efficient inner loop. The "reduceUnrolling8" kernel significantly enhances performance by handling eight data blocks per thread. Metrics like "gld_efficiency" and "gst_efficiency" can gauge memory load/store efficiency. In essence, this section underscores the benefits of template functions, compile-time optimization, and well-chosen block sizes to enhance parallel reduction efficiency in CUDA.

### Opinions and Comments

Now we are getting into template functions here to make it more efficient. We are using the reduce on rolling eight which at the end shows a speed up of 9.35 at the end this is the fastest thing you can do, the fastest ‘function’ you can use so it’s best to use this one. But I’m guessing there are certain things that are best to use with other functions or other ways then just always using the template

# DYNAMIC PARALLELISM

CUDA Dynamic Parallelism, a game-changing feature in GPU computing. Previously, all kernels were invoked from the host thread, but Dynamic Parallelism allows the creation of new GPU kernels directly on the GPU. This enables more flexible and hierarchical concurrency within GPU kernels, simplifying recursive algorithms and improving GPU resource utilization.

Dynamic Parallelism allows decisions on the number of blocks and grids to be made at runtime, adapting to data-driven scenarios and workloads, which optimizes performance. It also reduces the need to transfer control and data between the host and GPU, streamlining the process.

## Nested Execution

Dynamic parallelism allows the use of familiar kernel execution concepts for GPU kernel invocation. It distinguishes between parent and child kernel executions, where a child grid must complete before the parent is considered done. Proper nesting and synchronization are crucial.  

Grid launches in a device thread are visible within a thread block, and the block isn't considered complete until all child grids are done. When a parent launches a child grid, the child begins execution only when the parent thread block explicitly synchronizes with it. Parent and child grids share global and constant memory but have separate local and shared memory, with weak consistency guarantees. Shared and local memory are private and not visible between parent and child grids. Passing a local memory pointer as an argument when launching a child grid is invalid.

### Opinions and Comments
I’m thinking this has correlation to the use of__device__. This now leads to the use of a GPU calling to itself to do a certain function. This makes it more efficient since the CPU is slower, in the sense that while a CPU is calling the GPU to do a function the GPUs already finished it since I called it by himself.

## Nested Hello World on the GPU

"nestedHelloWorld". In the example, the host application initiates a parent grid with 8 threads in a single thread block. Thread 0 in this parent grid invokes a child grid with half as many threads, and this recursive process continues until only one thread remains.

The kernel's core logic involves printing "Hello World" for each thread and checking if it should terminate. If the thread count at a nested layer is greater than one, thread 0 invokes a child grid with a reduced number of threads. To compile the code, the "-rdc=true" flag is used to generate relocatable device code, a requirement for dynamic parallelism.

The output of the nested kernel execution displays the recursion depth and the number of threads in each block, showing how the child grids are nested. When the parent grid is invoked with two blocks instead of one, the output still shows child grids with only one block due to the kernel launch configuration.

The passage hints at the possibility of exploring different launching strategies to generate parallelism but doesn't delve into specific details. In summary,  this passage introduces dynamic parallelism in CUDA programming and provides a practical example using the "nestedHelloWorld"  kernel to show its application.

### Opinions and Comments

We have more into dynamic parallelism since we are shown that the RDC equals true flag which apparently it’s required for it to work. Where it is shown many nested, executions, and shows which ones are better we used the hello world in order just to show the speed and process of an execution of a code.

## Nested Reduction
Recursive reduction kernel  introduces the concept of reduction as a recursive function and shows how dynamic parallelism in CUDA simplifies its implementation.

The steps involved, such as converting global memory addresses and performing in-place reductions and the creation of child grids is also described, emphasizing their role in the reduction process.

An initial implementation of the recursive reduction kernel with dynamic parallelism is found to be slow due to a high number of kernel invocations and synchronizations, which negatively impact efficiency.

To address this performance issue, an optimized version of the kernel is introduced, which eliminates unnecessary synchronization operations. This optimization significantly improves the overall efficiency of the recursive reduction process, making it more suitable for practical applications.

Further optimization is explored, focusing on reducing the number of child grid launches and using idle threads more effectively. This results in even better performance, addressing some of the inefficiencies in the initial implementation. Efficient dynamic parallelism implementations in CUDA and emphasizes the need to minimize overhead and in-block synchronizations to achieve better performance. 

### Opinions and Comments

Lastly, to finish chapter 3, we can see how nested operations take the least time, as we can see. In the image, we can see how it specified the grid and the blog, the times it took to finish the process and the functions it used. In this specific example, we can see that the nested did the worst, but the nested to the best just before the GPU neighbored.