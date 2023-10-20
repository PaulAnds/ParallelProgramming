# Programming in Parallel with CUDA
### Introduction to GPU Kernels and Hardware

There are going to be 3 examples of C++ CUDA code that we will see. First is going to be a single thread, then multithread and finally it's going to use thousands of threads. The book then will explain how the architecture of a PC is constructed and an introduction to NVIDIA GPU to their hardware and how the model of CUDA programming works.

#### 1.1
PC’s now have multiple “computing CPU cores” from 2-X, it's said that to get the best out of the PC is to use parallel programming to run the code faster.

Examples of tools:
-   OpenMP
-   C++11 thread class < threads >
-
These tools help the code run faster by running parts of the code at the same time with the use of the cores. To make this better the book explains that by the use of multiple PC’s communicating with tools like Message Passing Interface (MPI). This is called high-performance computing or HPC, and around 25 of these PC’s with 8-Core CPUs would be needed to increment a factor of 200 in performance.

This works but the problem (besides the cost) is the overheating and power-hungry this system will be. The book explains that they will use a NVIDIA RTX 2070 GPU (In my case I have a NVIDIA GTX 970 GPU), and it also has the increment of a factor of 200 in performance as if it were the other system of HPC.

They also explain that the GPU is 10 times faster with the use of its internal memory, compared to a regular PC. This helps with the weaker CPU power and now just adds more of a focus to the limited memory bandwidth to worry about.

KERNEL FUNCTIONS:
- The functions in C++ code that actually run in the GPU.
-
This book uses a more modern take on C++ code rather than all the example codes that the NVIDIA Software Development Kit (SDK) uses, which could be more outdated.

### Opinions and Comments

It's interesting that they compare the use of 25 PC’s with 8 cores to be the same or even worse than the computing power of a NVIDIA RTX 2070. Meaning that the CUDA cores in reality are, as they say, faster and better than regular CPU cores. What is interesting is that the limited factor focused on after switching to the GPU is the limited memory bandwidth. The next focus would be to add more memory but the question I have now is, how would you do it?

### 1.2

The first example we will look at is with the trapezoid rule calculating the “integral of sin(x) from 0 to π”. 

![](https://lh4.googleusercontent.com/_JsIKs_BSUlGGFlfLs-KX6YRVJQcWfoHoTfXR3m3Kbq6L8X8nAMzzyQKrv5QI1BfuQaXAdBqKQuREo7DSFdAHjngGb0muuve5JvWI_gie_IlkYxtPY5mUql0cqsXY2xQgKJKGHcCI9_fLBk2M4mKzIQ)

This example was used because it's simple but it also is expensive for the computer. Basically what we do is divide the area under the curve into trapezoids called steps in the code and add the area of every individual step in order to get the integral of the function.

(command line arguments = argc) (argv = array of strings or array of arrays)

The code uses values of the steps as big as 10^9, which makes it hard enough for the computer to process and check the time it would take the CPU to do it.

-   This **first** version is called **cpusum** made by a single thread in the PC.


- The **second** version will be **ompsum** which will use the tool OpenMP sharing the loop over steps in multiple CPU threads divided equally between the cores. The ompsum version will show the best way a PC can operate without the use or help of a GPU.

-   The **third** version will be **gpusum**, which will use CUDA and the power of the GPU and the use of 1,000,000,000 threads in order to calculate the integral of sin(x).

For the ompsum version, or the second version, they used an Intel quad-core processor with the use of either four or eight threads. The speed-up factor of the with threads was 3.8 (running 8 cores instead of 4 meant that the pc was running two threads on each core which was supported by the Intel CPU, the factor to gain was a 2.

For the last example, or the gpusum, they parallelise the code with the use of multiple threads. They made it so that each thread calculates the area of each trapezoid of the 1,000,000,000 trapezoids, using a total of that many threads. The speed-up factor is now up to 960 rather than the 3.8 previously mentioned with lasting a total of **1882.707** ms.



To sum it up

-   CPUSUM = 1818.959 ms for 1,000,000 steps

-   OMPSUM = 477.961 ms for 1,000,000 steps

-   GPUSUM = 1882.707 ms for 1,000,000,000 steps


CUDA uses 4-byte floats while the other version uses a 8-byte float which makes it less accurate but by 8 significant figures, you can call it close enough.

It takes longer to read and write to memory than to make the calculations, and we will see how the GPU is delivering several TFlops/sec. (floating point operations per second, *for measurement in programming to check performance* ).**

![](https://lh3.googleusercontent.com/kxLDq_veDo2WBe84e4F06QKygObeNtZbijdG_qA3aAoifRLn4LbuAxZYkGK9LC_I82g2SY0NrKuOekmpauvfIzUzlb7n0YXAqWgiTqv2qHQvVmmOyatvzZX1Wl-Bnqxpqk-rUrbmz_VwzbI-jfgu0EU)

Another reason to use CUDA is to easily be able to scale up 3D versions of a problem that could have 1000 points in a Cartesian grid.

### Opinions and Comments

Opinions on this section is that it really is shown how fast a GPU is compared to the use of multiple threads on a CPU. The way parallel programming can make things way faster and efficient has helped a lot on complicated calculations. Things instead of going from 1 second to .5 seconds can become really big when a normal calculation takes around 4 hours to 1 hour saving a lot of time.

### 1.3
Computer code can be written by following language rules, but compiled code is run through physical hardware, on the actual PC.

![](https://lh4.googleusercontent.com/lh8-MnH-m7p6Wd7EY8OoUc7oBADd71aR1B2kRqCqNqiY4fZkMOiww1ClTDxJi7kc3i-DT4KIgTqzcRgLG92h9NoPWfY33nrtqxYOdNgayLHEDK0qrWumu5TRJZU8YqKABV2cWvev1HKQzRuhZMoZnog)

-   Master Clock: the one that tells everyone to do the next step. First computer made it with 2.2 MHz in 1981, doubling every 3 years, peaking at 4 GHz in 2002.

-   Memory: Holds the program data and the machine code instruction output. The code is treated as data read by either load/save, or fetch unit. Load/save can only write back to the main memory.

-   Load/Save: controlled by the execute logic, reads and writes data and sends it to main memory. Adds it or takes it from the registers file

-   Register File: All the data stored in the CPU, the “Heart”

-   ALU or Arithmetic Logic Unit: The execute unit tells it what calculations to make and what to do, part of the CPU that actually calculates things.

-   Execute: Decodes the compiled instructions sent by the fetch unit, tells the ALU to calculate what's necessary and transfers the result back to memory.

-   Fetch: Gets instructions from memory and passes it to the execution unit, holds the address of the instruction to do them in order, or in the necessary one to continue.
### Opinions and Comments
I get a better understanding of how a CPU works now. Used to think that a cpu just had programs that did everything but now seeing how the hardware actually works and what it does helped me understand a little bit better. Still have questions on how a piece of hardware can do these things but now I have a better idea on what is going on inside of a CPU and what the parts/ units are called.

### 1.4
In the chart you can see how Moore’s law is being followed by the computing power of a CPU. The frequency has stopped increasing since 2002 when we reached the 4GHz but performance per core is still increasing.


![](https://lh3.googleusercontent.com/GnApEMh24DvRr-EMPcAuLqQ2wNTB2Ccqxta0N5Fa03uugcWEz4Owcq65Cx4LEOvtBGJxCsfrC8M4OH3iCgYV-yuWvjWI6wEoCqbo4E0XyhyNqaiN-NS8sihur5iWPxh8SDIlCfBVUlMxFA1PAOW9RSk)

Increased the number of compute power by 1,000,000 in the last 30 years. Adding that most powerful computing system having millions of cores, have helped transform society.

### Opinions and Comments

I find it interesting, put in perspective, that when we started this in 1970 we had close to no frequency and that in 30 years that peak was reached. In ⅓ of someone's lifetime we have reached the maximum capability of frequency reached by computers. Giving everyone in the next generation by 2020 to have almost this technology at their disposal everywhere they go. The way the hardware was developed really dives into this peak that has been reached because of how these parts can’t actually be any closer together physically, and can’t be made any smaller.

### 1.5

When your computer processes information, it doesn't do it instantly. It's more like a series of steps that data and instructions go through. There's a delay between asking for something and actually getting it; we call this delay "latency."

But to make things seem faster, computers use a combination of caching (like storing things in quick-access places) and pipelining (where they start doing the next step before finishing the current one). This way, even if the first thing takes a bit of time, the following ones happen quickly.![](https://lh5.googleusercontent.com/lfRfFRkBejJlrm-OvmuumIJnyqgNDW4IzF4zNAaeRzFqdAZE9odFdPKZjF9A1e2PAJl4PgvDv-4j0sKe5SbnesNRvQlBuTARd8WVnmCn6kxT3s_2-yIuN_BAYYK-dFtcYxiHa9FCUTVtRwPPxX55KHI)

In computers, we have memory caches (like mini-storages) to keep data close, and they use these clever tricks to keep things running smoothly. In modern computers, you have different levels of these caches (L1, L2, L3 caches) to speed up how quickly you can access data.

### Opinions and Comments

Having a computer predict what is going to happen next can be really helpful, because it saves time as more code can pass quickly, which I see as good, but also if it's not accurate enough it will just waste time and power to do something that at the end it will not use.

### 1.6

Intel CPUs have noteworthy parallel capabilities through vector instructions, like SSE, AVX2, and AVX-512, which can significantly speed up floating-point calculations. These vector instructions use wide registers to handle multiple data elements simultaneously.

-   Pentium III SSE - 128-bit registers 4 4-byte floats.

-   Intel CPUs support AVX2 - 256-bit registers

-   AVX-512 - 512-byte registers 16 floats or 8 doubles.


### Opinions and Comments

Considering that before you could only hold to 4, 4 BYTE floats is crazy compared to the 16 floats or 8 doubles now. Having a register of 128 bit increase to the 4094 bits or 512 bytes is a huge jump done with Intel, and showing the advancement we have made with CPU’s.

### 1.7

Evolution of GPU over time focusing on NVIDIA hardware.

GPUs were initially designed for high-performance graphics in gaming, handling massive parallel calculations for pixels about 1.25 x 10 ^ 8 pixel calculations per second.

They store image data in frame buffers, with RGB values 3-bytes per pixel. Eventually, this GPU power was harnessed for general-purpose computing, leading to the emergence of GPGPU (general purpose computing on graphics processing units) in 2001, and it became mainstream with NVIDIA's toolkit in 2007.

NVIDIA offers three GPU classes:

-   GeForce for gaming

-   Tesla for scientific computing

-   Quadro for high-end workstations


With varying performance and features. Compute Capability (CC) values define a GPU's capabilities, and each generation brings software advancements. The most recent generation is Ampere with CC 8.0.

### Opinions and Comments

Having 3 different classes for the GPU seems good to me, because you can't really get into all the benefits of every CPU in one, since the space required to do all of them at the same time is quite difficult considering the space in a hardware chip.

### 1.8

Pascal architecture:

They start with basic compute cores that do math quickly. These cores are grouped into sets of 32, called warp-engines and execute the same instruction at every clock-cycle. In. Warp-engines have extra math tools (special function units (SFUs)) and resources they share.

*Pascal GPUs, warp-engines have 8 SFUs and either 16 or one FP64 unit.*

Warp-engines join up to create symmetric multiprocessors (SMs) with about 128 compute cores each (four wrap-engines). When you run a game, it's divided into small groups called thread blocks, and each block runs on one SM. SMs have additional memory and shared resources. With (__syncthreads(),) one can see the use of communication of threads inside a block, which can’t happen between threads in different blocks.![](https://lh3.googleusercontent.com/vKT0Y05N8HiR6yudTGQcC8WARXC4jk5zdf--RYkwoRJKFDkzWWuz_fpNZxM3kUTFu4-pFJ2ennt5bF03ooTwYAIvIQz-yMLD2KS9IjG7_aBOLwaPH0yRULTszYt-8Nkqk1HlAZLStViJG0djOERtyag)

Example:

*GTX 1080 has 20 SMs containing a total of 20 x 128 = 2560 compute-cores.*

Hardware will groups the cores into warps of 32 cores called Warp engine (WE), which adds IO(double precision units) and 8 special units(SFU), which are functions like exp or sin()

Groups of WE = SM

Groups of SM = GPU

### Opinions and Comments
All cores are in the same hardware chip. That's why it's more difficult to add cores in a small space, because it takes up space for each core, making a chip with multiple cores either bigger, or having smaller physically sized cores.

#### 1.9

GPU hierarchy

In CUDA, the main memory stores program code and data. Data transfers between the CPU and GPU main memory can be slow, so it's important to minimize them.

1.  Main memory holds what the CPU can read and write from.

2.  Constant memory is fast and is used for data that all threads need. 64 KB

3.  Texture memory is optimized for certain types of data and can make data look smoother. 2D arrays. read only. fast 1D interpolation, 2D bilinear interpolation or 3D trilinear interpolation

4.  Local memory is used when there's not enough space in the registers for temporary data.

5.  Shared memory helps threads within a group communicate efficiently. 32 KB and 64 KB of shared memory.

6.  Register files are used for fast data storage, but each thread has a limit on how much it can use. 64K 32-bit registers
    
![](https://lh6.googleusercontent.com/xd_hEP_KRVoc2Kxqm-oHbvM5mLWz_MJvZDJDe8_ZBfQJHje3PHyyo3QNwOCaMTpH1xeBDFZMedFPCx3H8iqbSn7v2AucxsVW1VE2lR_g_RdH3wpuLSvz-MIUAmQ2ZFPCj2sPiqHkvbsyu4tHuQ24U8o)

To ensure things run efficiently, it's important that threads access memory in an organized way, which helps maintain high performance.

The caches work most effectively if the 32 threads in a warp access 32-bit variables in up to 32 adjacent memory locations and the starting location is aligned on a 32-word memory boundary.

*This memory addressing patterns is called **memory coalescing** in CUDA documentation.*
### Opinions and Comments

Makes sense that the most optimal use of these threads in warps is focused on a factor of 32, so it can use its full potential without having an extra block with using some threads but not all. At the end that can't be efficient since that non completed block has less threads to communicate with each other lowering efficiency.

#### 1.10

Designing efficient CUDA kernels involves determining the optimal number of threads (Nthreads) for a specific problem, which should generally be as large as possible.

*This is what we need to do in order to make better code ^^.*

To process a 2D image having nx × ny pixels, a good choice is to put Nthreads = nx × ny.

NThreads should be as big as possible, take use of as many threads as possible.

Setting Nthreads equal to the number of GPU cores (Ncores) is not sufficient, as modern GPUs can effectively hide latencies by switching between threads.

For instance, an RTX 2070 GPU with 36 SM units can handle 2304 cores but maintains a large number of resident threads (Nres) for latency hiding.

The actual number of threads run in waves (Nwaves) is determined by Nres, Nsm, and Nwarp, with the goal of minimizing the number of threads in a kernel launch. Keep in mind that Nres and Nsm values may vary between GPU generations and models but not the Nwarp = 2.

### Opinions and Comments

Having to determine how many threads are needed to make a process can be easier made with knowing how many threads are accessible with the GPU you are working with. And I wonder if not just using all of them for every process can be a good idea. I imagine there is a number lower than the maximum that could be a better use than all of the ones accessible.

#### 1.11

In CUDA, thread blocks are groups of threads that run on the same SM, and their size should be a multiple of the warp size (typically 32).

Threads within the same block can communicate and synchronize using shared or global device memory. 

![](https://lh5.googleusercontent.com/GIStQLwvIjneq_2ymoyWGunoUeLKwm1Kz15a721Hfrib-Anf15gIjUqIALYVAf3uqIUoZ15U1IdbVqe_PeWoZRRo6rlZuZx2VWsdHGUfft2aGse6oTjFZGBiDbLxsMmRgn7St3dfNG-GWzmh0O8Ncas)

**Threads in different blocks cannot communicate** or synchronize during kernel execution. The launch configuration specifies the block size and the number of blocks in a grid. The total number of threads is threads × blocks. The block size should be adjusted to ensure Nthreads ≥ N (threads you actually want our kernel to run), with rounding up.

*thread block size to be a sub multiple of 1024; often it is 256.*

*launch configuration* with two values, the thread block size and the number of thread blocks.

Waves, though not explicitly discussed in CUDA documentation, play a role in dispatching threads to SMs for execution, emphasizing the importance of making blocks a multiple of the SM count on the GPU.

### Opinions and Comments

Besided blocks, threads are also divided in warps, the warp is always (now) at 32 but the block can vary depending on what the programmer sets it as. So you can see that a warp is made up of threads but a block is made out of warps. Checking that the warp and the block are multiples of 32, making 256 8 warps per block.

#### 1.12

NVIDIA defines occupancy as the ratio of active threads to the maximum capacity of SM units, often expressed as a percentage. Achieving 100% occupancy means running full wavefronts on SMs.

However, limited shared memory and registers can make full occupancy challenging. NVIDIA GPUs support up to 32 registers per thread for full occupancy. Lower occupancy can be acceptable, especially for compute-bound tasks with substantial shared memory usage.

Experimenting using global memory instead of shared memory and relying on L1 caching for speed may be a good compromise on modern GPUs.

### Opinions and Comments

This was similar to my thought and confirming that using all the threads doesn't mean achieving 100%. One has to test what memory to use for what, and the number of threads in each block to reach the most optimized way a code will run, varying from code to code.

## Thinking and Coding in Parallel

“Trivially parallel” is something that is done at the same time but not necessarily parallel programming, like using 4 instances of a calculator to do different things. True parallel programming requires many processing cores working together to complete a single task.

#### 2.1

Different computer architectures, as classified in Flynn's taxonomy, significantly influence the optimization of programming.

**SISD** architecture is typical for single-threaded processors, allowing rapid task switching but essentially executing one operation per clock cycle.

**SIMD** architectures, exemplified by vector processors like early CRAY supercomputers and Intel's SSE instructions, can operate on multiple data items simultaneously, improving performance for certain computations.

**MIMD** architecture, seen in modern multicore PCs and clusters, involves separate CPUs for parallel tasks, making effective use of software like MPI or OpenMP.

**MISD** is rarely used, primarily for specialized systems needing redundancy.

**SIMT**, introduced by NVIDIA for GPUs, offers parallelism through a large number of threads, permitting both common and divergent operations. In parallel programming, the SIMD/T architecture is vital, particularly for optimizing operations on multiple data items, such as loops, where minimizing dependencies between loop passes is crucial for efficiency.

**Atomic operations** to perform the required operation serially. Without them threads could override a global variable.

Instead of using them in the example, they made it so the threads save all the results in an array, and later added all the array together for the answer, to not slow the process down with the atomic operations.

![](https://lh6.googleusercontent.com/bbWb9PPWSlN1Rw3mj6CLRRxpXk2nSi6Rza8MsTRbyEfE8Lo5hve_MO4p3bdJeMFI5SIh_7EA7CK2qu_Hf8ubLwRL7QnuDWcIxVBf9_16NoqEQFO5rHWfce9he4znO-_zYEpytKXA5JpSdah70IsV_p4)

1.  cuda_runtime.h which provides basic support for CUDA

2.  thrust/device_vector.h provides a container class like std::vector for 1D


arrays in GPU memory.

![](https://lh5.googleusercontent.com/VP-d-PvBKqmPgKiPOp0gcETS5vY4P9hv2l8j5TkD6Przd-Qt4T1MYhe7p5GheakitERNKgfBgWv6T5YDrH1Qge3qLCQ4xxyAhn5D8KKIhaJ4A-oGaPSMIvbjAeKbaqZTYQV_TV5_xYtGG-sctrv8lfQ)



__ device __ = it's going to be used in the GPU, alone if only function for GPU

__ host __ = a version that is used in the CPU, only needed if used in CPU AND GPU



CUDA inline is the default for __ device __ functions.



CUDA kernels are declared using __ global __ instead of __ device __ this reflects their dual nature – callable by the host but running on the GPU.

must be declared void

Imagine a kernel working as a loop, as it's going to run “thread x block” times.

*When looking at a kernel code, you must imagine that the code is being run simultaneously by all the threads. Once you have mastered this concept, you will be a parallel programmer*

The member function data() does this job for thrust **host_vector** objects. Use the undocumented data().get() member function of **device_vectors.**

gpu_sin<<<blocks,threads>>>

Uses int variables for the blocks and threads. Blocks need to be a factor of 256 or 32 x 8.

Threads should be a factor of 32, and maximum 1024.

For most kernels a good starting point is <<<4*Nsm, 256>>> where Nsm is the number of SMs on the target GPU.

![](https://lh6.googleusercontent.com/xo7QApih_GvpCwQkleRQ_fAyBVsv7ND9ociV6GTs4YhwmpBLVDFhVcYUwQQHmG5b3F-GzTARTllIpFY0Jj0aJeH_rfa4V5_BYGwC9fh9iprYA-vf8FPbl03BB98x6zbXQx80ipt2tR3FUfKMrJM7dII)

*D2H (device to host) transfer.*

Firstly we perform the required additions on the GPU and secondly we copy the result from GPU memory to CPU memory.

NVIDIA also uses the term lane to refer to the rank(step) of a thread within its particular 32-thread warp.

The GPU hardware allocates all the threads in any particular thread block to a single SM unit on the GPU, and these threads are run together very tightly on warp-engines as warps of 32 threads.

![](https://lh4.googleusercontent.com/GsQ1R3ZDOzpRLfLe2BeUIrBWiS2yImuSvGhrt6ny16_6iFkv5Z-UB8Dgi6V7s2i6tCyEo0xSM_xRhuhHWJoJ62exdOO8OrV9ec3pmiZithWqgNGD0aQb6BixGsX_4713KUEz3SgyeUQ0n2aw-xm_VsE)




This is **NOT** used because gridDim would have given the same values to step, but hardware wise it would not have been efficient, because threads that are NEXT to each other would have had values that are 3906251 values apart.

![](https://lh5.googleusercontent.com/rsS9s533saEu8nlcNiyygXUBN1x564rCwHRTSTwzPXIiwqyMKq29BiFN7zyfjTInI896a45VNfvxZXHgF3rTg_IL7ArHZLNDXj03Q8yELZVvAIhHe2cmewaghRKzJ0upsU8v23sZW49L0ibpWODmDPc)

Kernels use dim3 variables even though if you give them simple int variables, that's why we use the .x values

This technique of using a while loop with indices having a grid-size stride between passes through the loop is called “thread-linear addressing” and is common in CUDA code.

### Opinions and Comments

At the end the use of variables in both device and host is different but use the input global in order to have both of them in one area. So the CPU can call on the GPU and pass a pointer of variables it can use to calculate. Then have the GPU pass the values back to the CPU.

#### 2.2

The four arguments enclosed in <<< >>> brackets for kernel functions are:

1.  The first argument defines the grid of thread blocks, impacting how work is distributed across the GPU's cores and memory hierarchy.

2.  The second argument determines the number of threads in a block, influencing how efficiently the GPU utilizes its thread execution units and shared memory.

3.  The third argument, shared memory size, can be dynamically allocated or statically declared, optimizing memory usage depending on the hardware's capabilities.

4.  The fourth argument specifies the CUDA stream, essential for advanced applications utilizing multiple simultaneous kernels, which can enhance GPU utilization and overlap computation with memory transfers based on the hardware's capabilities.




### Opinions and Comments

I've only seen the first two used from what the teacher has shown. The other two must be more advanced and for way more optimization. But I'm sure it has to do mainly with what hardware you have, so studying that can give you more experience on what values to put in the parameters.

#### 2.3

3D Kernel Launches now the dim3 are going to be fully used.

Using a global declaration is actually an easy way to create GPU arrays of known size, but we will rarely use it.

Disadvantage 1: The array dimensions must be set at compile time not run time

Disadvantage 2: Declaring variables with file scope is a deeply depreciated programming style because it leads to unstructured code where functions can easily cause unwanted side effects.

![](https://lh6.googleusercontent.com/xBzaM-97OkqvSLbtQ4GOdofBnUnhx5QFzLzFio5QdecVjde-8IcVuiM61_v0cPhetsVctcfIOIF8aK8vghc2xUONicY4gXegrtXq3ptbjEbVHevcIag1V7ehmHdW3wrHQC9LBSHXQOSP41sUxakDFTs)

*Since 1234567 = 512*2411+135 we have picked the 135th thread in the 2412th block. The first 4 x-y slices account for 2048 blocks so our pick is in the 364th block in the 4–5 slice pair. Next since 364 = 22*16 + 12 we conclude that our thread is in the 12th block in the set of 16 blocks that spans the index range [0-511,168-175,5-6]. This 12th block spans [352-383,176-183,5-6] and since the 135th thread is offset by [7,4,0] from this position we find an index set of [359,180,5]*

![](https://lh4.googleusercontent.com/BdZrcJeq6fyhTJV-YQNeq6YVdmQn3ZJadxT1gYXIZi3cxTLBHib1yNV7nvrVfO4ygxgBhGf52ZuLovfBhCBTyq7McglJsRspsEcDIoEqJskW0zHiDN72PFm39_pNhhZ3mp2L5UQCAjWd_B5UrT6wTyY)

### Opinions and Comments

In my opinion, calculating what you need for everything and the way to look for a specific thread is still kind of difficult for me to understand but I can see there is a base for it. I wonder if there is a way to see which individual thread caused a problem, and if there is a way to fix that thread specifically.

#### 2.4

The GPU's hardware design enables efficient and optimized programming by allowing multiple threads to execute independently, hiding latency by interleaving tasks and minimizing stalls, unlike traditional CPUs where shared registers lead to costly task switching. This architecture encourages programmers to rely on compilers for optimization rather than writing instructions in an unnatural order, ultimately enhancing code maintainability.

![](https://lh3.googleusercontent.com/5ZA-UKVj8qMUSVn591TdUH2w-MSOo2pgeIi0PBvaJf5Kn8nH2LGONxE-jH8yozyEEHqaeGZr2wat8W59MpVleW71ZIwl8tkmp624M6BJL5J5aFU1MczehAHZiMnwcXEMKfm0EpN9k6-nrj2BAbEj5p0)

In GPU each warp engine can process several active warps in an interleaved fashion, if one of its active warps stalls, a warp-engine will switch to another active warp capable of running with no loss of cycles.

Hardware limitations, such as the number of available registers and shared memory, impact the optimization of GPU programming.

Achieving full occupancy, where each thread needs only 25 cycles of computation between memory accesses, is crucial to hide global memory latency. Factors like thread block size, the number of thread blocks, and compiler-generated register allocation influence this optimization.

![](https://lh4.googleusercontent.com/TQXrIWc_6E2G5jvz-xCcGfMPX5GxmBafQZ6GllTtR9LacBs58bhPpgtsWEBdQ9v6C-Wyn1UU2-28bHVsyMfBX1PUrLkUhF7LaQZWbucV6-id3jXexcYfqSYEcVBhAbxqg4X3amqeHOC6cibVHxVKG7M)

For memory-bound kernels, **full occupancy is particularly vital**. Keeping code compact and straightforward, using separate kernels for long calculations, and understanding global memory preservation between kernel launches are key strategies for efficiently utilizing GPU hardware.

### Opinions and Comments

So the point at the end of the use of the CPU is for all the active wraps to be used. You can't have anything idle or not doing anything. That's why calculating how many threads in a block in a wrap in a grid needs to have DEPENDING on the hardware you're using so it varies from person to person.

#### 2.5

Parallel programming for GPUs running vast numbers of threads does require some rethinking of your approach to coding. Methods that work well in single CPUs or codes running a small number of independent threads may need to be rethought.

**Aka. DON'T USE IF STATEMENTS WITH MULTIPLE FUNCTIONS!**

If you have an if, what happens is the threads are divided into 2 groups, one takes on the first statement of the if while the others are idle, and when done then the other group goes to the second statement making the first group idle. This is the opposite of the parallel programming we want to do.

### Opinions and Comments

Just dont use if’s. If it's not simple, figure out a way.

#### 2.6

The parallel reduce operation:

The first point to recognise is that each data item just requires a single adition. We want to use as many threads as possible in order to hide memory latency efficiency

The host doesn't wait for the kernel function to finish and continues on with the code.

cudaDeviceSynchronize makes it so the CPU waits for the GPU to finish processing.

The results show that the GPU's 32-bit floating point calculation maintains accuracy, while the host calculation with a float variable loses accuracy due to rounding errors. However, the GPU's reduction kernel is inefficient due to frequent global memory accesses, and the host's iterative approach could be improved.

![](https://lh6.googleusercontent.com/kRGj6UGdKx1zMYulg4mPBE2cvAzOsQ4N5ihgX7In_br2bCpQRHd8mC79IDUdtMHFobYA-od0A0kS04u2B3VVSOAP-66Utb2TYXsouoVPVhW44c9u6RdpGmJAfMNCS_FScaW0W9InXLTDOLPly2j2eNk)

The change in the second code shows how they save up the results of the GPU in an additional array to not waste time on adding the results while it's getting them.

![](https://lh4.googleusercontent.com/8gRQ9KnEME07J39lUnUYxOhkWSV8L7dzHstuN7A8IS1UdRLgsSWm1D04Xfe4ImSla1PnKSWpsF54E45r-rI6NQVpOKbWNh5626d6A_u0EPXYEiQ6aXZ-42Do5wT_vo98Yey2DR_PLy9vMjoXPMv8D-o)

SM units distribute shared memory among resident thread blocks, with each thread having access to a limited amount. This shared memory is faster than global memory but is a finite resource, and excessive use may **reduce thread occupancy or lead to kernel launch failures.**

![](https://lh5.googleusercontent.com/3qBLShV3YooC3uReo7KCO13LNZN6Wi3A8CW1cN6q0OstZ8cHwAUYSisk9aKZIQ-dnD86tGzMvFrW7Y7SviYEyHo2rKv2NTGnS9vQFfNiiHVkplWne8Yyd70uhhSei3hOKh6xqCII9MZD3XMA0fUQi8E)

While shared memory was crucial in early CUDA tutorials due to limited global memory caching, more recent GPUs have improved caching, and devices with CC 7.0 and above can allocate a single fast memory resource to both L1 and shared memory in varying proportions for different kernels.

This code was faster because in many GPUs the number of SM units is not a power of 2. For example, theGPU used in the book has 36 SMs so to keep all SMs equally busy it is better to use 288 rather than 256 for the number of user set values of blocks.

![](https://lh5.googleusercontent.com/mD37vwgN22IyR9VUyokVmn977u6s_y4e_Yqh6JjuK43xK6CewPgI15D3fZ0F8Bj1ysxvyhTpcKfKtZR5qXX9uUmzvGws8gmed_zgxxH-RZ4OiymR9BpjgguvK4APldBzUdzUYANF_HquAYm8qvtjBdo)


![](https://lh4.googleusercontent.com/Uf4KRdl-MoxjqlERXenBG9qc84ug3lwakI_BeB2mFtVaZFBDmgjXErDOFmGYdaXgPLgHuAlGUcxoSU__ocsKgkaMo2yhp_9v5zolOqo44GUNehnEqVvStDSsWu-FXAoFALs0XyzlS6CBYXnVTZbmHa0)

This code has made it faster by a factor of 70, compared to the first code we analyzed. Since it made the blocks to a value that is not to the power of 2, focusing on what hardware he has made it more optimized.

**The last trick to try is explicitly unrolling the loop warp-based programming**

![](https://lh4.googleusercontent.com/QOfMhY4f6ji0FrgJl6eM9714sPfLMlyDt9hX2fHbzxGp0_auO0UJCkNxaETCwxPt1RYeqndi6_wEUksUDIzo9VR2nI_VLwt26XfMu8MMUpgJHAK0PMSdPj58wro-oueuxZNlF10YyYcMsBb6-0np7Jo)


![](https://lh6.googleusercontent.com/DB-UWAyF-HsKWaET3hbJfhhqhR2pAN5BVvQW___AJVeEqy4T_NaSo2KR36dHkYdnktukfGjCojsGRzfzmW8UQ4zzyPWPOZ5Q90lDHl38qXnlNg8q2E4J0SkAYp4QVFKr8e_p7-3GppZu79Um8Hw_U3o)

These replace the for loop and last line of the previous example. Here we have unrolled the loop on the explicit assumption that the number of threads per block, blockDim.x, is in the range [256,511].

### Opinions and Comments

Always going back to the hardware and checking what is best for the individual piece is best. Yes the use of the common of the suggested could be good but at the end if it's not the best for the hardware you have then there is always an upgrade or a better way to optimize it.

#### 2.7
__ shared __ is used for shared memory in a variable.

Shared memory is a fast-access memory pool on each Streaming Multiprocessor (SM) shared by all the threads in the block. Each thread block on an SM receives a separate allocation from this shared memory pool. 64KB

However, if a kernel requires more than 32 KB of shared memory for its thread blocks, only one thread block can run on the SM at a time, reducing occupancy. Shared memory is local to thread blocks and is not shared between them.

Thus, shared memory use is one of the factors to be considered when optimizing occupancy.

![](https://lh5.googleusercontent.com/P_1SMWob57Jx2PrFR6Xb5yymddYmzaDn6iBWEmh8PLuknvfUs3SpA4hP7ft8cNQlD4mNcHrXCqOGUxoNqPQ4JpGBQW8h9kYpCV7q3NLl4UvbJn4T0uE5R2PKwiPu-uwd8au9-Itg8ibqoSwYzI_BAOs)

Its contents are lost when a thread block exits. Shared memory allocation can be static or dynamic, with dynamic allocation allowing the size to be determined at kernel launch. Multiple shared arrays can be declared in a kernel, but careful handling is required to prevent conflicts in dynamic allocation.

![](https://lh6.googleusercontent.com/nfdQhpjtE9nTJnaHWpIF_xpRQ2w8X3xvYNZVrF0XhMIT5o_OBAESM45AKAhOeFgD74zMWNtIB3uLmd7ERPrepPiqhgxoPInKcDVMhjn0wilPGp36ijCXt5h5Q8TvzfvWPeZb9LNLLxz4dHcq2iBowwo)

The kernel go from the longest variable type (4-byte floats) to the shortest variable type (1-byte chars), natural alignment will be achieved for all three arrays without the compiler needing to introduce gaps.

### Opinions and Comments

Using the shared memory in all the functions could also help to share data between the threads. Maybe you could be able to use a few of the threads as “variables” in order to use it for the rest of the threads. This could help so all the threads don't use the global memory but just the shared memory between them.

#### 2.8

Matrix multiplication.

![](https://lh4.googleusercontent.com/GfkjujT7f0SUd6zUwevutw4yrI8FRGzoyVcraTuGVEGwY7HDgRnq4vrWXJWMiknPWy_C9ZT5IjRzrhvNAy9LPQPTYVhPZbuzJ6iUaIemgSYBn8466qQw2Lg2wgRPnPMdujIIuYu5TuqqfQg9DOBW-r4)

The use of shared memory and contiguous memory blocks helps for efficient implementation in CUDA, with the use for explicit address calculations over complex container classes like C++ vectors or matrices.

![](https://lh4.googleusercontent.com/BA1Eg90ceW69Yd2l1I0AjQ-nkb8hEETjV8MbVuOu7g1ImoSGsGZ3OIJ2_x3BZv41OjuIEOUew5DgHSgMxEZjMJWBbxKWK4K3ArKByLxrPGDnJiVI-0WAdjRAF8ZIuxo5UIyaaloLEO3nRqbgxaWCz14)

This is the 2D linear verizon to call on a matrix.

![](https://lh5.googleusercontent.com/BlOZGm-WStc7Ap8c3JRjwLNznKdnx3EBxzxlL3iIuGOwBt_XVW7K5Y8ad1YD99JMEAcWL5s-lT7npJ7QH1h8q2AYbDLx75GdQnTSfiRT0fUOauNDvEGlW9d8ey_1LjSxks1ZiNyFAxzeD5YCjzDkcug)

On the use of a regular CPU they call the function to multiply a matrix and show that the use of __restrict in the function improved the performance by 44%. Adding the restrict qualifier to a pointer declaration tells the compiler that the pointer is not aliased and aggressive optimisation is safe.

The CUDA NVCC compiler also supports restrict and the performance of many kernels do indeed improve when it is used.

***Always use restrict with pointer arguments.***

C++ compilers may use _restrict or __restrict

*In practice we find using const does not usually give much or any performance gain; its use is actually more important as a safeguard to prevent accidental overwriting of variables and to make a programmer’s intentions clear.*

Try using these for pointer arguments as it adds more optimization to the code.The middle column provides templated wrappers from cx.h for simplifying code. These wrappers are applicable in both host and kernel code.

![](https://lh3.googleusercontent.com/KZlblQsIFRual5IUzSEgBGO743rCpDPQZNiuZRt7QlNbF48TG8jlepIZmjSzUdpaFGGAkkysJzNyQxF66amJaFNYbZzSYyYodmlMtbtIeqYrZbXFp1yig7AdFqgXUXeY6xqrI205LN7ugWnICC7BMPY)

![](https://lh3.googleusercontent.com/NXssJMjS-ApvU3Uv-qiK1bcn4cyJDByEQTqRNu4hhKIH8JYfeqwVWIFmWxCf7cfNT3MKQMdVxCtGoKzgkHutareIAUZ9Fp-2omZ2geFOb1JWsCge8oxXmsaOFms-Di0o3e1f07ledoryrVo1CMXhlHM)

Code used in a GPU to do the multiplication of matrices The the kernel is designed to use one thread to calculate one element of the matrix product.

![](https://lh4.googleusercontent.com/6OrhwOdkp6ngskvmCeiajNLeoSUXdu5IKIECh9PjYSdBmEuGVtXFm29tY0srXePa1bHbKFiQUJOUze20g-rvtO2_A7pf_p2P66u5rFflrwAvauwCfKR1d4wkiEamY05hqIY9tYEKbmz4-HOQdANRdds)

Just adding restrict to the code. First one using regular C++ or the second one using the cx h abbreviation, shows a speed up factor of 2.6 of the code. The effective memory bandwidth is also much greater than the hardware limit of about 400 GBytes/sec separately for read and write, demonstrating that memory caching is playing an important role.

A local function called "idx" is introduced to calculate 2D addresses. This function captures the value "Bx" and uses it by reference, ensuring no copying is needed. This approach modernizes the old macro trick and results in performance similar to the last code.

![](https://lh5.googleusercontent.com/MAZHA1E6JzQBlGrw-msQR3S7zEi05PWSj-CVjlwrteQ1d0NO0QfDZyVThZpy7jzMvuNWY0Es6yraHqj8g7J7_18Fmig3vUrw8bm863r5xuUrJCzNbu_dW9dy5h2Zcp7vNdtv2Uw0I17T3IQtXhUOQM0)

### Opinions and Comments

Also going back to the basics of C++ it has some optimized functions. In reality using everything you can to optimize the code is better than focusing on one thing. Yes, you can use the NVIDIA CUDA cores for parallel programming but also some usage like restrict and const in the code also helps, remembering the GPU will have to go through a lot of data between all the threads.

#### 2.9

We can put values of A and B in the first threads in order to abuse shared memory and have all the other threads use those values to do their process. The idea is we use a set of 3 x 3 tiles to perform the multiplication of 9 x 9 matrices.

*Tiled matrix multiplication can be readily implemented in CUDA kernels by using 16 x 16 or 32 x 32 thread blocks to represent a pair of tiles from A and B. Each thread then first copies one element of its thread block’s allocated A and B tiles into shared memory arrays.*

![](https://lh6.googleusercontent.com/sOJyIBuH2CyDskCybS9_yv_eBTYvoe13M8b08pQi-FHgJSmJpriGBRZ-keXw2wnruhqXLwNW7mUdmB-2VX7xeqKO2YyGG5tO8tI-5Zo1vGnxcYNAYUfCg54oljyj-d69cYvFRr8uBl9NBJGqWKaKRYc)

The kernel launch itself is now changed because the guptiled kernel is written to use the value of tilex as a template parameter. The optimal depth of unrolling can only be found by experiment; on their RTX 2070 the value 16 seems to give the best result. On other GPUs you may find a different optimum.

### Opinions and Comments

Using another parameter to show in the kernel is shown well and really useful for individual hardware, for my 970 I would imagine that it would have a different outcome than the 2070. In order for me to see what would be best is to run the code and check the times. Since it varies on what the hardware has from components, maybe another of the functions will work for me.

#### 2.10

![](https://lh4.googleusercontent.com/Hj1gP0goZkZ_Sk_fkPFbVBAAK-nOG2hF6uGDOVaRoCQq8T0dJJ5OPH-2WmiyKU4BwBv_JSUfNGBtv-wuUgIJWgHeh0xgmKfR8YXGuuOzNOZh2AYsi5VTx3cNvX2P_2G43o0antErXHnH-IAUjFmG8Cc)

cublasSgemm function is much faster than our best custom code for matrix multiplication. If your computer has tensor cores, you can make it even faster. So, if you need fast matrix multiplication, it's better to use NVIDIA's library rather than writing your own code.

This advice also applies to other standard problems like FFT. However, learning to write your own GPU code is still valuable for situations where pre-made solutions don't exist. Your custom code can be faster for smaller matrix sizes and specific tasks, but for large and common operations, NVIDIA's library is the way to go.

### Opinions and Comments

This is true, if something is already optimized then there is no reason to try and make something else. We could try to optimize it more but at the end of the day if there is no way you can think of optimizing it or making it faster then use what they give you already.