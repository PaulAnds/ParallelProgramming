# EXAMEN PARCIAL 2

Programación de paralelismo

**Paul Andres Solis Villanueva**

University of Advanced Technologies

Leonardo Juárez Zucco

07/11/2023


## Table of content
- [EXAMEN PARCIAL 2](#examen-parcial-2)
  - [Table of content](#table-of-content)
  - [simpleMathAoS](#simplemathaos)
    - [Exlpanation](#exlpanation)
  - [simpleMathSoA](#simplemathsoa)
    - [Exlpanation](#exlpanation-1)
  - [sumArrayZerocpy](#sumarrayzerocpy)
    - [Exlpanation](#exlpanation-2)
  - [sumMatrixGPUManaged](#summatrixgpumanaged)
    - [Exlpanation](#exlpanation-3)
  - [sumMatrixGPUManual](#summatrixgpumanual)
    - [Exlpanation](#exlpanation-4)
  - [transpose](#transpose)
    - [Exlpanation](#exlpanation-5)
  - [writeSegment](#writesegment)
    - [Exlpanation](#exlpanation-6)
  - [memTransfer](#memtransfer)
    - [Exlpanation](#exlpanation-7)
  - [pinMemTransfer](#pinmemtransfer)
    - [Exlpanation](#exlpanation-8)
  - [readSegment](#readsegment)
    - [Exlpanation](#exlpanation-9)
  - [readSegmentUnroll](#readsegmentunroll)
    - [Exlpanation](#exlpanation-10)

## simpleMathAoS

==1011== NVPROF is profiling process 1011, command: ./simpleMathAoS

==1011== Profiling application: ./simpleMathAoS

==1011== Profiling result:

|            Type | Time(%) |     Time   |  Calls |      Avg |      Min |      Max | Name|
|      :----:     |  :---:  |  :----:    | :----: |  :----:  |  :----:  |   :---:  |:---:|
| GPU activities: |  77.15% | 28.510ms   |      2 | 14.255ms | 8.5563ms | 19.953ms | [CUDA memcpy DtoH]|
|                 |  20.44% | 7.5520ms   |      1 | 7.5520ms | 7.5520ms | 7.5520ms | [CUDA memcpy HtoD]|
|                 |   1.21% | 447.34us   |      1 | 447.34us | 447.34us | 447.34us | warmup(innerStruct*, innerStruct*, int)|
|                 |   1.21% | 446.70us   |      1 | 446.70us | 446.70us | 446.70us | testInnerStruct(innerStruct*, innerStruct*, int)|
|      API calls: |  89.80% | 740.49ms   |      2 | 370.25ms | 729.44us | 739.76ms | cudaMalloc|
|                 |   4.91% | 40.489ms   |      3 | 13.496ms | 8.7933ms | 20.596ms | cudaMemcpy|
|                 |   4.66% | 38.446ms   |      1 | 38.446ms | 38.446ms | 38.446ms | cudaDeviceReset|
|                 |   0.35% | 2.9273ms   |      1 | 2.9273ms | 2.9273ms | 2.9273ms | cuDeviceGetPCIBusId|
|                 |   0.17% | 1.4021ms   |      2 | 701.05us | 580.99us | 821.11us | cudaDeviceSynchronize|
|                 |   0.09% | 732.33us   |      2 | 366.17us | 250.51us | 481.82us | cudaFree|
|                 |   0.01% | 90.432us   |      2 | 45.216us | 43.833us | 46.599us | cudaLaunchKernel|
|                 |   0.00% | 23.890us   |    101 |    236ns |    160ns | 2.0940us | cuDeviceGetAttribute|
|                 |   0.00% | 12.474us   |      1 | 12.474us | 12.474us | 12.474us | cudaGetDeviceProperties|
|                 |   0.00% | 10.921us   |      1 | 10.921us | 10.921us | 10.921us | cudaSetDevice|
|                 |   0.00% | 5.6000us   |      2 | 2.8000us | 2.6050us | 2.9950us | cudaGetLastError|
|                 |   0.00% | 2.2030us   |      3 |    734ns |    250ns | 1.4830us | cuDeviceGetCount|
|                 |   0.00% | 1.6630us   |      2 |    831ns |    241ns | 1.4220us | cuDeviceGet|
|                 |   0.00% |    972ns   |      1 |    972ns |    972ns |    972ns | cuDeviceGetName|
|                 |   0.00% |    561ns   |      1 |    561ns |    561ns |    561ns | cuDeviceTotalMem|
|                 |   0.00% |    351ns   |      1 |    351ns |    351ns |    351ns | cuDeviceGetUuid|


---

### Exlpanation

**GPU Activities:**

- memcpy DtoH (Device to Host memory copy) consumes 77.15% of the GPU time, with an average time of 14.255ms. This indicates significant time spent transferring data from the GPU to the CPU.
  - Its called in line 130 which sends the back the value d_A to h_A which is the inner structure in the GPU to the data in the device which is the innerstruct of the CPU. It takes the longest out of all of them because the device has a slower clockspeed which means its working at a higher latency rate than a HtoD to recieve data.

- memcpy HtoD (Host to Device memory copy) consumes 20.44% of the GPU time, with an average time of 7.5520ms. This also involves data transfer but in the opposite direction.
  - Its called in line 148 and which sends the back the value d_C to gpuRef which is also an innerStruct variable, fixed with the malloc function and size of the variable nBytes. Its called first in this kernel to warmup the code. Then its called again in the testInnerStruct kernel which does the exact same thing than the warmup and shows a print of both that show the grid and block specified for each of them.

- warmup and testInnerStruct kernels both use 1.21% of the GPU time, with 447.34us average time. These kernels are relatively fast compared to memory transfers.
  - Its a global function called in the processor for it to call the device which is the GPU. It adds value to the array depending if its on the x or the y, and the result makes it into the combination of that innerstruct tmp. After it does that it uses cudamemcpy to add it to the gpuRef from the d_C variable.

**API Calls:**

- cudaMalloc accounts for 89.80% of the API call time, totaling 740.49ms. This suggests that a significant amount of time is spent allocating memory on the GPU.
  - cuda malloc is called in line 126 and line 127 which make it so you allocate the memory of the innerstruct to the variables of the device which are d_A and d_C with the size nBytes. This took the most time becase it was the allocation of memory in to the device so the time the memory was used from the host to be added to the device.

- cudaMemcpy follows with 4.91% of API time, taking 40.489ms. These are likely memory copy operations between the CPU and GPU.
  - This call is used 3 times, 2 times for the DtoH since its dowing two kernels, and once for the HtoD which is copying the memory of the values to the device which then the device uses the memory for both of the kernels.

- cudaDeviceReset takes 4.66% of API time, indicating a relatively small but non-negligible portion of time spent on resetting the CUDA device.
  - This api call releases all allocated memory metioned in the code. It takes 5.61 of the time, second of the most used time, for the reason that it has to free all the memory. Its called in line 171 where its at the end of the code to finilize it before returning an exit success.

- cuDeviceGetPCIBusId, cudaFree, cudaDeviceSynchronize, cudaLaunchKernel, cuDeviceGetAttribute, cudaSetDevice, cudaGetLastError, cudaGetDeviceProperties, cuDeviceGetCount, cuDeviceGet, cuDeviceGetName, and cuDeviceTotalMem are various other CUDA API calls with their respective percentages of the total API time and call counts.

  - These last calls are use to free up the memories of the device and the host and other functions that dont use alot of the time with the API calls. The uses for synchronizing the devices and actually calling the kernels to the GPU.

---

## simpleMathSoA


==1027== NVPROF is profiling process 1027, command: ./simpleMathSoA

==1027== Profiling application: ./simpleMathSoA

==1027== Profiling result:

|            Type | Time(%) |     Time   |  Calls |      Avg |      Min |      Max | Name|
|      :----:     |  :---:  |  :-----:   | :----: |  :----:  |  :----:  |   :---:  |:---:|
| GPU activities: |  74.25% | 26.942ms   |      2 | 13.471ms | 10.969ms | 15.973ms | [CUDA memcpy DtoH]|
|                 |  23.26% | 8.4404ms   |      1 | 8.4404ms | 8.4404ms | 8.4404ms | [CUDA memcpy HtoD]|
|                 |   1.25% | 452.84us   |      1 | 452.84us | 452.84us | 452.84us | warmup2(InnerArray*, InnerArray*, int)|
|                 |   1.23% | 447.98us   |      1 | 447.98us | 447.98us | 447.98us | testInnerArray(InnerArray*, InnerArray*, int)|
|      API calls: |  88.91% | 708.51ms   |      2 | 354.26ms | 645.71us | 707.87ms | cudaMalloc|
|                 |   5.25% | 41.868ms   |      1 | 41.868ms | 41.868ms | 41.868ms | cudaDeviceReset|
|                 |   5.04% | 40.145ms   |      3 | 13.382ms | 11.418ms | 16.745ms | cudaMemcpy|
|                 |   0.49% | 3.9005ms   |      1 | 3.9005ms | 3.9005ms | 3.9005ms | cuDeviceGetPCIBusId|
|                 |   0.19% | 1.5441ms   |      2 | 772.05us | 619.34us | 924.76us | cudaDeviceSynchronize|
|                 |   0.10% | 792.47us   |      2 | 396.24us | 297.70us | 494.78us | cudaFree|
|                 |   0.01% | 93.537us   |      2 | 46.768us | 46.007us | 47.530us | cudaLaunchKernel|
|                 |   0.00% | 20.399us   |    101 |    201ns |    140ns | 1.5030us | cuDeviceGetAttribute|
|                 |   0.00% | 12.133us   |      1 | 12.133us | 12.133us | 12.133us | cudaGetDeviceProperties|
|                 |   0.00% | 8.4660us   |      1 | 8.4660us | 8.4660us | 8.4660us | cudaSetDevice|
|                 |   0.00% | 6.8630us   |      2 | 3.4310us | 3.2560us | 3.6070us | cudaGetLastError|
|                 |   0.00% | 2.3850us   |      3 |    795ns |    251ns | 1.7330us | cuDeviceGetCount|
|                 |   0.00% | 1.3130us   |      1 | 1.3130us | 1.3130us | 1.3130us | cuDeviceGetName|
|                 |   0.00% | 1.2730us   |      2 |    636ns |    191ns | 1.0820us | cuDeviceGet|
|                 |   0.00% |    481ns   |      1 |    481ns |    481ns |    481ns | cuDeviceTotalMem|
|                 |   0.00% |    261ns   |      1 |    261ns |    261ns |    261ns | cuDeviceGetUuid|

---

### Exlpanation

**GPU Activities:**

- [CUDA memcpy DtoH]: This activity shows that 74.25% of the GPU time is spent copying data from the device (GPU) to the host (CPU). The total time spent on this operation is approximately 26.942 milliseconds. This is a relatively slow operation compared to other GPU activities.
  - Its called in line 160 and 170 which sends the back the value d_C to gpuRef which is the InnerArray in the GPU to the data in the device which is the InnerArray of the CPU. It takes the longest out of all of them because the device has a slower clockspeed which means its working at a higher latency rate than a HtoD to recieve data.

- [CUDA memcpy HtoD]: This activity accounts for 23.26% of the GPU time, taking around 8.4404 milliseconds. It represents data copying in the opposite direction, from the host (CPU) to the device (GPU).
  - Its called in line 142 and which sends the back the value h_A to d_A which is also an innerStruct variable, fixed with the malloc function and size of the variable nBytes. Its called first in this kernel to warmup the code. Then its called again in the tetInnerStruct kernel which does the exact same thing than the warmup and shows a print of both that show the grid and block specified for each of them.

- warmup2(InnerArray, InnerArray, int)**: The GPU spends 1.25% of its time on this kernel. It took approximately 452.84 microseconds. This function performs some computations using the InnerArray data structures.
  - In this kernel its sent the data of the InnerArray of the device and a result that will be in the device later to be sent to the host using the DtoH. In lines 105 to lines 110 we modify the values of the array given to us in the parameter and saving it to the result that is giving back the innerarray d_C.

- testInnerArray(InnerArray, InnerArray, int)**: Another kernel function, this accounts for 1.23% of the GPU time, taking approximately 447.98 microseconds. Similar to the "warmup2" function, it works with InnerArray data.
  - A kernel that does the same thing as the warmup2 which is why it took the same amount of time to do compared to that kernel which is 256us, the kernel modifies the array by sending it, changing the x and y of the values and setting it to the result that is sent back.

**API Calls:**

- cudaMalloc: This API call consumes the most time, taking up 88.91% of the time, which is around 708.51 milliseconds. It's called twice in the code.
  - This function is used to allocate the memory to the device, it takes the most out of the API since its the only big call that is made, which is making the allocations.

- cudaDeviceReset: This API call takes 5.25% of the time, around 41.868 milliseconds. It resets the CUDA device.
  - It resets or frees all the allocated memory, since there is memory in both the GPU and CPU it takes a little more time to remove than the cudaMemcpy.

- cudaMemcpy: These calls collectively occupy 5.04% of the time, taking about 40.145 milliseconds. cudaMemcpy is used to transfer data between the host and device.
  - This function is called 3 times, 2 for the HtoD and once for the DtoH. This takes the least of the top time consuming function on the API Calls because it uses the less memory in compared to the last code.

- cuDeviceGetPCIBusId: This API call represents 0.49% of the time, taking 3.9005 milliseconds. 
  - It retrieves the PCI bus identifier of the CUDA device.

- cudaDeviceSynchronize: These calls together account for 0.19% of the time, or 1.5441 milliseconds. 
  - They ensure that the device has completed its tasks and is synchronized with the host.

- cudaFree: This call takes 0.10% of the time, around 792.47 microseconds. 
  - It frees allocated device memory.

- cudaLaunchKernel: These calls represent 0.01% of the time, approximately 93.537 microseconds. 
  - It launches GPU kernels for execution.

---

## sumArrayZerocpy

==1049== NVPROF is profiling process 1049, command: ./sumArrayZerocpy

==1049== Profiling application: ./sumArrayZerocpy

==1049== Profiling result:

|            Type | Time(%) |     Time   |  Calls |      Avg |      Min |      Max | Name|
|      :---:      | :---:   |  :---:     | :---:  |  :---:   |   :---:  |   :---:  |:---:|
| GPU activities: |  34.41% | 5.1200us   |      1 | 5.1200us | 5.1200us | 5.1200us | sumArraysZeroCopy(float*, float*, float*, int)|
|                 |  24.52% | 3.6480us   |      2 | 1.8240us | 1.6960us | 1.9520us | [CUDA memcpy DtoH]|
|                 |  21.51% | 3.2000us   |      1 | 3.2000us | 3.2000us | 3.2000us | sumArrays(float*, float*, float*, int)|
|                 |  19.57% | 2.9120us   |      2 | 1.4560us | 1.4400us | 1.4720us | [CUDA memcpy HtoD]|
|      API calls: |  93.65% | 862.99ms   |      3 | 287.66ms | 3.5360us | 862.98ms | cudaMalloc|
|                 |   5.39% | 49.623ms   |      1 | 49.623ms | 49.623ms | 49.623ms | cudaDeviceReset|
|                 |   0.52% | 4.8311ms   |      1 | 4.8311ms | 4.8311ms | 4.8311ms | cuDeviceGetPCIBusId|
|                 |   0.15% | 1.3551ms   |      2 | 677.53us | 11.131us | 1.3439ms | cudaHostAlloc|
|                 |   0.11% | 1.0459ms   |      4 | 261.48us | 101.90us | 550.08us | cudaMemcpy|
|                 |   0.10% | 912.48us   |      2 | 456.24us | 27.752us | 884.73us | cudaFreeHost|
|                 |   0.06% | 567.78us   |      3 | 189.26us | 4.4680us | 544.50us | cudaFree|
|                 |   0.01% | 119.40us   |      2 | 59.698us | 44.725us | 74.672us | cudaLaunchKernel|
|                 |   0.00% | 23.325us   |    101 |    230ns |    160ns | 1.4930us | cuDeviceGetAttribute|
|                 |   0.00% | 15.279us   |      1 | 15.279us | 15.279us | 15.279us | cudaSetDevice|
|                 |   0.00% | 6.1820us   |      1 | 6.1820us | 6.1820us | 6.1820us | cudaGetDeviceProperties|
|                 |   0.00% | 3.3670us   |      2 | 1.6830us |    682ns | 2.6850us | cudaHostGetDevicePointer|
|                 |   0.00% | 2.1330us   |      3 |    711ns |    270ns | 1.4730us | cuDeviceGetCount|
|                 |   0.00% | 1.6140us   |      2 |    807ns |    251ns | 1.3630us | cuDeviceGet|
|                 |   0.00% | 1.1420us   |      1 | 1.1420us | 1.1420us | 1.1420us | cuDeviceGetName|
|                 |   0.00% |    521ns   |      1 |    521ns |    521ns |    521ns | cuDeviceTotalMem|
|                 |   0.00% |    301ns   |      1 |    301ns |    301ns |    301ns | cuDeviceGetUuid|

---

### Exlpanation

**GPU Activities:**

- sumArraysZeroCopy(float, float, float*, int)**: This GPU activity takes up 34.41% of the GPU time, using 5.1200 microseconds.
  - This kernel last the longest for the reason that the kernel is executed with zero copy memory, which means that the kernel is called without using any copied memory which makes it take longer. In compared with the other kerner to sumArrays which is called the execution to do the sum of the array.

- [CUDA memcpy DtoH]: This activity accounts for 24.52% of the GPU time, with a total time of 3.6480 microseconds. It represents data copying from the device (GPU) to the host (CPU).
  - Recalling that the DtoH takes more data than the HtoD is because of how the clock speeds are for both of the systems. The devices clock is faster than the hardrives so its faster for the drive to process data which makes the time consuming less during the GPU Activities. This process is called in the lines 141 and 176 which copy memory of the result that the kernel made "d_C" to the gpuRef.

- sumArrays(float, float, float*, int)**: This function consumes 21.51% of the GPU time, using 3.2000 microseconds. It is likely another array summation operation similar to the first one.
  - The sumArray is a funtion that takes an int value of the dimentions of the gpu that its going to work with and uses that to compare it to N which isthe number of elements. If that number is less than i which is the dimention, then we make the sum of both of the numbers in the array the result that we sent back.

- [CUDA memcpy HtoD]: This activity uses 19.57% of the GPU time, with a total time of 2.9120 microseconds. 
  - This funtion is called in lines 130 and 131 which copies the data from the host to the device. It doesnt take much time since the clock in the device is faster it can process the data quicker than if i t was DtoH.

**API Calls:**

- cudaMalloc: This API call takes the most significant portion, 93.65% of the time, with a total time of 862.99 milliseconds. It is called three times in the code and is used to allocate memory on the GPU.
  - This function is used to allocate the memory to the device, it takes the most out of the API since its the only big call that is made, which is making the allocations between the device and the host.

- cudaDeviceReset: This API call consumes 5.39% of the time, with a total time of 49.623 milliseconds. It resets the CUDA device.
  - It resets or frees all the allocated memory, since there is memory in both the GPU and CPU it takes a little more time to remove than the cudaMemcpy.

__These next api calls dont use much of the time to be compleated during this execute.__

- cuDeviceGetPCIBusId: This API call accounts for 0.52% of the time, using 4.8311 milliseconds. 
  - It retrieves the PCI bus identifier of the CUDA device.

- cudaHostAlloc: This API call takes up 0.15% of the time, with a total time of 1.3551 milliseconds. 
  - It allocates memory on the host (CPU) that can be directly accessed by the GPU.

- cudaMemcpy: These calls together consume 0.11% of the time, using 1.0459 milliseconds. 
  - cudaMemcpy is used to transfer data between the host and device.

- cudaFreeHost: This API call takes up 0.10% of the time, with a total time of 912.48 microseconds. 
  - It frees memory that was allocated on the host using cudaHostAlloc.

- cudaFree: This call consumes 0.06% of the time, with a total time of 567.78 microseconds. 
  - It frees memory allocated on the device (GPU).

- cudaLaunchKernel: These calls represent 0.01% of the time, using 119.40 microseconds. 
  - They are used to launch GPU kernels for execution.

---

## sumMatrixGPUManaged

==1071== NVPROF is profiling process 1071, command: ./sumMatrixGPUManaged

==1071== Profiling application: ./sumMatrixGPUManaged

==1071== Profiling result:

|            Type | Time(%) |     Time   |  Calls |      Avg |      Min |      Max | Name|
|      :---:      | :---:   |  :---:     |  :---: |  :---:   |   :---:  |   :---:  |:---:|
| GPU activities: | 100.00% | 21.021ms   |      2 | 10.511ms | 509.58us | 20.511ms | sumMatrixGPU(float*, float*, float*, int, int)|
|      API calls: |  89.02% | 784.84ms   |      4 | 196.21ms | 32.232ms | 673.02ms | cudaMallocManaged|
|                 |   4.06% | 35.835ms   |      4 | 8.9587ms | 8.5160ms | 9.3401ms | cudaFree|
|                 |   4.04% | 35.644ms   |      1 | 35.644ms | 35.644ms | 35.644ms | cudaDeviceReset|
|                 |   2.48% | 21.876ms   |      1 | 21.876ms | 21.876ms | 21.876ms | cudaDeviceSynchronize|
|                 |   0.32% | 2.8215ms   |      1 | 2.8215ms | 2.8215ms | 2.8215ms | cuDeviceGetPCIBusId|
|                 |   0.07% | 616.83us   |      2 | 308.41us | 8.5160us | 608.31us | cudaLaunchKernel|
|                 |   0.00% | 20.001us   |    101 |    198ns |    120ns | 1.8830us | cuDeviceGetAttribute|
|                 |   0.00% | 10.911us   |      1 | 10.911us | 10.911us | 10.911us | cudaGetDeviceProperties|
|                 |   0.00% | 9.1170us   |      1 | 9.1170us | 9.1170us | 9.1170us | cudaSetDevice|
|                 |   0.00% | 1.9630us   |      3 |    654ns |    240ns | 1.3520us | cuDeviceGetCount|
|                 |   0.00% | 1.7140us   |      2 |    857ns |    201ns | 1.5130us | cuDeviceGet|
|                 |   0.00% | 1.5230us   |      1 | 1.5230us | 1.5230us | 1.5230us | cudaGetLastError|
|                 |   0.00% | 1.1920us   |      1 | 1.1920us | 1.1920us | 1.1920us | cuDeviceGetName|
|                 |   0.00% |    431ns   |      1 |    431ns |    431ns |    431ns | cuDeviceTotalMem|
|                 |   0.00% |    241ns   |      1 |    241ns |    241ns |    241ns | cuDeviceGetUuid|


---

### Exlpanation

- sumMatrixGPU(float, float, float*, int, int)**: This GPU activity accounts for 100.00% of the GPU time, using a total time of 21.021 milliseconds.
  - This function taking up 100% of the gpu time is taking that much because its the onlything going into the GPU, what its doing is finding the dimentions of the matrix, or what number we are working on and then doing the addition. So adding the [1,1] of matrix A to the [1,1] of matrix B.

**API Calls:**

- cudaMallocManaged: This API call consumes 89.02% of the time, with a total time of 784.84 milliseconds. It's called four times and is used to allocate managed memory on the GPU.
  - cudaMallocManaged is used for unified memory allocation, which allows the GPU and CPU to access the same memory, helping for the values to be the same. It takes the most time due to the size of the memory its using and that is has to share the memory in both the device and in the host. They are called in lines 110 to 113 to use the memory allocation of all the variables used for references and the matrices.

- cudaFree: This API call takes up 4.06% of the time, with a total time of 35.835 milliseconds. It's called four times and is used to free memory on the GPU.
  - This is used to free the memory that is used in the variables in the GPU, Its important so that the memory is not just floating there taking space in the limited memory that we have in a GPU.

- cudaDeviceReset: This API call accounts for 4.04% of the time, using a total time of 35.644 milliseconds. It resets the CUDA device.
  - This is used at the very end to free all the variables and memories used. This takes 3.45% of the time because we are using 2 matrices

- cudaDeviceSynchronize: This API call consumes 2.48% of the time, with a total time of 21.876 milliseconds. It is used to synchronize the CPU with the GPU and ensure all GPU operations are completed.
  - It is used to synchronize the host with the device and ensure that all previously issued commands have completed which is what takes the whole GPU to process as seen in the GPU Activities.

---

## sumMatrixGPUManual

==1089== NVPROF is profiling process 1089, command: ./sumMatrixGPUManual

==1089== Profiling application: ./sumMatrixGPUManual

==1089== Profiling result:

|            Type | Time(%) |     Time   |  Calls |      Avg |      Min |      Max | Name|
|      :---:      | :---:   |  :---:     | :---:  |  :---:   |   :---:  |   :---:  |:---:|
| GPU activities: |  56.34% | 42.887ms   |      1 | 42.887ms | 42.887ms | 42.887ms | [CUDA memcpy DtoH]|
|                 |  40.11% | 30.532ms   |      2 | 15.266ms | 15.164ms | 15.368ms | [CUDA memcpy HtoD]|
|                 |   2.42% | 1.8432ms   |      2 | 921.60us | 508.59us | 1.3346ms | sumMatrixGPU(float*, float*, float*, int, int)|
|                 |   1.13% | 857.14us   |      2 | 428.57us | 428.52us | 428.62us | [CUDA memset]|
|      API calls: |  84.71% | 724.12ms   |      3 | 241.37ms | 1.0040ms | 722.10ms | cudaMalloc|
|                 |  10.32% | 88.257ms   |      3 | 29.419ms | 16.257ms | 43.771ms | cudaMemcpy|
|                 |   4.29% | 36.639ms   |      1 | 36.639ms | 36.639ms | 36.639ms | cudaDeviceReset|
|                 |   0.34% | 2.9157ms   |      1 | 2.9157ms | 2.9157ms | 2.9157ms | cuDeviceGetPCIBusId|
|                 |   0.20% | 1.7193ms   |      1 | 1.7193ms | 1.7193ms | 1.7193ms | cudaDeviceSynchronize|
|                 |   0.12% | 986.04us   |      3 | 328.68us | 228.09us | 509.11us | cudaFree|
|                 |   0.01% | 61.329us   |    101 |    607ns |    130ns | 31.069us | cuDeviceGetAttribute|
|                 |   0.01% | 57.099us   |      2 | 28.549us | 7.4840us | 49.615us | cudaMemset|
|                 |   0.01% | 46.499us   |      2 | 23.249us | 15.780us | 30.719us | cudaLaunchKernel|
|                 |   0.00% | 11.321us   |      1 | 11.321us | 11.321us | 11.321us | cudaSetDevice|
|                 |   0.00% | 9.8890us   |      1 | 9.8890us | 9.8890us | 9.8890us | cudaGetDeviceProperties|
|                 |   0.00% | 2.0340us   |      3 |    678ns |    350ns | 1.3330us | cuDeviceGetCount|
|                 |   0.00% | 1.7020us   |      2 |    851ns |    280ns | 1.4220us | cuDeviceGet|
|                 |   0.00% | 1.2030us   |      1 | 1.2030us | 1.2030us | 1.2030us | cuDeviceTotalMem|
|                 |   0.00% | 1.1720us   |      1 | 1.1720us | 1.1720us | 1.1720us | cuDeviceGetName|
|                 |   0.00% |    762ns   |      1 |    762ns |    762ns |    762ns | cudaGetLastError|
|                 |   0.00% |    330ns   |      1 |    330ns |    330ns |    330ns | cuDeviceGetUuid|


---

### Exlpanation

**GPU Activities:**

- [CUDA memcpy DtoH]: This GPU activity occupies 56.34% of the GPU time, using a total time of 42.887 milliseconds.
  - The operation called in line 158 is copying data from the device product d_MatC to the host array h_C. The source data resides in the GPU memory, and it's being transferred to the CPU memory.

- [CUDA memcpy HtoD]: This GPU activity accounts for 40.11% of the GPU time, with a total time of 30.532 milliseconds. It involves memory copying from the host (CPU) to the device (GPU).
  - The operation called in lines 147 and 148 is copying data from the host arrays h_A and h_B to the device arrays d_MatA and d_MatB. The source data resides in the CPU memory, and it's being transferred to the GPU memory.

- sumMatrixGPU(float, float, float*, int, int)**: This GPU activity consumes 2.42% of the GPU time, with a total time of 1.8432 milliseconds.
  -  This kernel is is responsible for adding two input matrices and storing the result in an output matrix, its called on line 151 and it has as parameters the dimentions of the matrix and the matrices that the device will use and store with 'd_MatA, d_MatB, d_MatC'.

- [CUDA memset]: This GPU activity takes up 1.13% of the GPU time, with a total time of 857.14 microseconds. It's related to memory setting operations.
  - this function takes the pointer ptr as the address of the memory region you want to change. It sets each byte in the specified memory region to the given value. It does this operation for however many number of bytes, which is determined by the '0' parameter, then returns a poniter function like the input ptr.

**API Calls:**

- cudaMalloc: This API call consumes 84.71% of the time, with a total time of 724.12 milliseconds. 
  - It's called three times and is used for allocating memory on the GPU.

- cudaMemcpy: This API call takes up 10.32% of the time, with a total time of 88.257 milliseconds. 
  - It's called three times and is used for copying memory between the host and the device.

- cudaDeviceReset: This API call accounts for 4.29% of the time, using a total time of 36.639 milliseconds. 
  - It resets the CUDA device. 

- cuDeviceGetPCIBusId: This API call consumes 0.34% of the time, with a total time of 2.9157 milliseconds. 
  - It's called once and retrieves the PCI bus ID of the device.

- cudaDeviceSynchronize: This API call takes up 0.20% of the time, with a total time of 1.7193 milliseconds. 
  - It is used to synchronize the CPU with the GPU and ensure all GPU operations are completed.


---

## transpose

==1111== NVPROF is profiling process 1111, command: ./transpose

==1111== Profiling application: ./transpose

==1111== Profiling result:

|            Type | Time(%) |     Time   |  Calls |      Avg |      Min |      Max | Name|
|      :---:      | :---:   |  :---:     | :---:  |  :---:   |   :---:  |   :---:  |:---:|
| GPU activities: |  92.45% | 5.8141ms   |      1 | 5.8141ms | 5.8141ms | 5.8141ms | [CUDA memcpy HtoD]|
|                 |   3.85% | 242.40us   |      1 | 242.40us | 242.40us | 242.40us | warmup(float*, float*, int, int)|
|                 |   3.69% | 232.23us   |      1 | 232.23us | 232.23us | 232.23us | copyRow(float*, float*, int, int|)
|      API calls: |  93.98% | 789.23ms   |      2 | 394.61ms | 506.63us | 788.72ms | cudaMalloc|
|                 |   4.45% | 37.391ms   |      1 | 37.391ms | 37.391ms | 37.391ms | cudaDeviceReset|
|                 |   0.75% | 6.2932ms   |      1 | 6.2932ms | 6.2932ms | 6.2932ms | cudaMemcpy|
|                 |   0.58% | 4.8993ms   |      1 | 4.8993ms | 4.8993ms | 4.8993ms | cuDeviceGetPCIBusId|
|                 |   0.11% | 945.44us   |      2 | 472.72us | 303.27us | 642.17us | cudaDeviceSynchronize|
|                 |   0.11% | 922.80us   |      2 | 461.40us | 320.98us | 601.82us | cudaFree|
|                 |   0.01% | 64.132us   |      2 | 32.066us | 15.239us | 48.893us | cudaLaunchKernel|
|                 |   0.00% | 36.431us   |    101 |    360ns |    190ns | 1.8840us | cuDeviceGetAttribute|
|                 |   0.00% | 11.983us   |      1 | 11.983us | 11.983us | 11.983us | cudaGetDeviceProperties|
|                 |   0.00% | 7.7350us   |      1 | 7.7350us | 7.7350us | 7.7350us | cudaSetDevice|
|                 |   0.00% | 3.5560us   |      2 | 1.7780us |    881ns | 2.6750us | cudaGetLastError|
|                 |   0.00% | 3.1250us   |      3 | 1.0410us |    371ns | 1.4220us | cuDeviceGetCount|
|                 |   0.00% | 1.9530us   |      2 |    976ns |    410ns | 1.5430us | cuDeviceGet|
|                 |   0.00% | 1.4930us   |      1 | 1.4930us | 1.4930us | 1.4930us | cuDeviceGetName|
|                 |   0.00% |    702ns   |      1 |    702ns |    702ns |    702ns | cuDeviceTotalMem|
|                 |   0.00% |    340ns   |      1 |    340ns |    340ns |    340ns | cuDeviceGetUuid|


---

### Exlpanation

**GPU Activities:**

- [CUDA memcpy HtoD]: This GPU activity occupies 92.45% of the GPU time, using a total time of 5.8141 milliseconds. It involves memory copying from the host (CPU) to the device (GPU).
  - This takes most of the gpu activities since the longest thing the gpu can do is copy the variable memories of the host and set it to the device. Usually the DtoH takes longer but since we are not using it here since our ikernel is = 0, the longest one is HtoD.

- copyRow(float, float, int, int)**: This GPU activity consumes 3.69% of the GPU time, with a total time of 232.23 microseconds.
  - In the kernel called in line 321 as a case 0, unsigned int ix and unsigned int iy store the global index of the thread, the if statement check if the values are inside the range, and if it is then the thread copies an element from the in array and puts it in the out array based on its row.

- warmup(float, float, int, int)**: This GPU activity accounts for 3.85% of the GPU time, with a total time of 242.40 microseconds.
  - Does the same thing than the copy row function as it does it before it in order to check warm up the gpu, since the first kernel usually takes the longest and we are using this for the profiler we dont want to see the speed of the first kernel as we want of the rest of them.

**API Calls:**

- cudaMalloc: This API call consumes 93.98% of the time, with a total time of 789.23 milliseconds. 
  - It's called twice and is used for allocating memory on the GPU.

- cudaDeviceReset: This API call accounts for 4.45% of the time, using a total time of 37.391 milliseconds. 
  - It resets the CUDA device.

---

## writeSegment

==1127== NVPROF is profiling process 1127, command: ./writeSegment

==1127== Profiling application: ./writeSegment

==1127== Profiling result:

|            Type | Time(%) |     Time   |  Calls |      Avg |      Min |      Max | Name|
|      :---:      | :---:   |  :---:     | :---:  |  :---:   |   :---:  |   :---:  |:---:|
| GPU activities: |  62.08% | 3.3433ms   |      3 | 1.1144ms | 831.70us | 1.6687ms | [CUDA memcpy DtoH]|
|                 |  33.35% | 1.7961ms   |      2 | 898.05us | 843.06us | 953.04us | [CUDA memcpy HtoD]|
|                 |   1.57% | 84.706us   |      1 | 84.706us | 84.706us | 84.706us | writeOffset(float*, float*, float*, int, int)|
|                 |   1.56% | 84.161us   |      1 | 84.161us | 84.161us | 84.161us | warmup(float*, float*, float*, int, int)|
|                 |   0.86% | 46.369us   |      1 | 46.369us | 46.369us | 46.369us | writeOffsetUnroll2(float*, float*, float*, int, int)|
|                 |   0.57% | 30.529us   |      1 | 30.529us | 30.529us | 30.529us | writeOffsetUnroll4(float*, float*, float*, int, int)|
|      API calls: |  93.31% | 768.78ms   |      3 | 256.26ms | 665.75us | 767.37ms | cudaMalloc|
|                 |   5.11% | 42.084ms   |      1 | 42.084ms | 42.084ms | 42.084ms | cudaDeviceReset|
|                 |   0.86% | 7.0858ms   |      5 | 1.4172ms | 817.79us | 3.0512ms | cudaMemcpy|
|                 |   0.43% | 3.5527ms   |      1 | 3.5527ms | 3.5527ms | 3.5527ms | cuDeviceGetPCIBusId|
|                 |   0.13% | 1.0964ms   |      4 | 274.10us | 156.90us | 524.63us | cudaDeviceSynchronize|
|                 |   0.13% | 1.0760ms   |      3 | 358.67us | 279.10us | 471.99us | cudaFree|
|                 |   0.02% | 197.64us   |      4 | 49.409us | 21.040us | 64.974us | cudaLaunchKernel|
|                 |   0.00% | 23.462us   |    101 |    232ns |    160ns | 2.2640us | cuDeviceGetAttribute|
|                 |   0.00% | 15.469us   |      1 | 15.469us | 15.469us | 15.469us | cudaGetDeviceProperties|
|                 |   0.00% | 9.2970us   |      1 | 9.2970us | 9.2970us | 9.2970us | cudaSetDevice|
|                 |   0.00% | 5.0780us   |      4 | 1.2690us |    961ns | 1.4620us | cudaGetLastError|
|                 |   0.00% | 2.2450us   |      3 |    748ns |    331ns | 1.5030us | cuDeviceGetCount|
|                 |   0.00% | 1.6540us   |      2 |    827ns |    251ns | 1.4030us | cuDeviceGet|
|                 |   0.00% | 1.3530us   |      1 | 1.3530us | 1.3530us | 1.3530us | cuDeviceGetName|
|                 |   0.00% |    511ns   |      1 |    511ns |    511ns |    511ns | cuDeviceTotalMem|
|                 |   0.00% |    291ns   |      1 |    291ns |    291ns |    291ns | cuDeviceGetUuid|

---

### Exlpanation

**GPU Activities:**

- [CUDA memcpy DtoH]: This GPU activity occupies 62.08% of the GPU time, with a total time of 3.3433 milliseconds. It involves memory copying from the device (GPU) to the host (CPU).
  - This operation takes the longest as stated before that the passing of memory between a host and a device, the host has a lower clock speed so it takes more time to process it.

- [CUDA memcpy HtoD]: This GPU activity accounts for 33.35% of the GPU time, with a total time of 1.7961 milliseconds. It involves memory copying from the host (CPU) to the device (GPU).
  - This operation takes less time, as stated before that the passing of memory between a host and a device, the device has a higher clock speed so it takes less time to process it.

- writeOffset(float, float, float*, int, int)**: This GPU activity consumes 1.57% of the GPU time, with a total time of 84.706 microseconds.
  - This kernel called on line 153 after setting the size of the blocks calls the if statement to check if the k (offset index) is in valid range meaning it's less than n. if i t is then a new value for the C array element in value k is made. This value is made of the sum of A[i] plus B[i].

- warmup(float, float, float*, int, int)**: This GPU activity accounts for 1.56% of the GPU time, with a total time of 84.161 microseconds. 
  - Does it before it in order to check warm up the gpu, since the first kernel usually takes the longest and we are using this for the profiler we dont want to see the speed of the first kernel as we want of the rest of them.

- writeOffsetUnroll2(float, float, float*, int, int)**: This GPU activity uses 0.86% of the GPU time, with a total time of 46.369 microseconds.
  - This operation called in line 166 does the same thing as writeoffset but now if the codition is met, instead of processing a single element at a time it does two elements, one at K and another one at K + blockdim, making use of the threads to do more process instead of doing it with multiple threads.

- writeOffsetUnroll4(float, float, float*, int, int)**: This GPU activity consumes 0.57% of the GPU time, with a total time of 30.529 microseconds.
  - This operation called in line 179 does the same thing than in writeoffset but even better than in unroll2. Since it makes it so one thread calculates the time it would take the first function to do 4, it improves the time and shows better way of a code to use parallelism.


**API Calls:**

- cudaMalloc: This API call consumes 93.31% of the time, with a total time of 768.78 milliseconds. 
  - It's called three times and is used for allocating memory on the GPU.

- cudaDeviceReset: This API call accounts for 5.11% of the time, with a total time of 42.084 milliseconds.
  - It resets the CUDA device.

- cudaMemcpy: This API call occupies 0.86% of the time, with a total time of 7.0858 milliseconds. 
  - It's called five times and is used for copying memory data between the host and device.

- cuDeviceGetPCIBusId: This API call accounts for 0.43% of the time, with a total time of 3.5527 milliseconds. 
  - It retrieves the PCI bus ID of the CUDA device.

---

## memTransfer

==935== NVPROF is profiling process 935, command: ./memTransfer

==935== Profiling application: ./memTransfer

==935== Profiling result:

|            Type | Time(%) |     Time  |   Calls |      Avg |      Min |      Max | Name|
|      :---:      | :---:   |  :---:    |  :---:  |  :---:   |   :---:  |   :---:  |:---:|
| GPU activities: |  51.53% | 4.4243ms  |       1 | 4.4243ms | 4.4243ms | 4.4243ms | [CUDA memcpy DtoH]|
|                 |  48.47% | 4.1616ms  |       1 | 4.1616ms | 4.1616ms | 4.1616ms | [CUDA memcpy HtoD]|
|      API calls: |  93.40% | 667.18ms  |       1 | 667.18ms | 667.18ms | 667.18ms | cudaMalloc|
|                 |   4.85% | 34.645ms  |       1 | 34.645ms | 34.645ms | 34.645ms | cudaDeviceReset|
|                 |   1.28% | 9.1722ms  |       2 | 4.5861ms | 4.1593ms | 5.0129ms | cudaMemcpy|
|                 |   0.40% | 2.8733ms  |       1 | 2.8733ms | 2.8733ms | 2.8733ms | cuDeviceGetPCIBusId|
|                 |   0.05% | 365.89us  |       1 | 365.89us | 365.89us | 365.89us | cudaFree|
|                 |   0.00% | 22.965us  |     101 |    227ns |    130ns | 1.1720us | cuDeviceGetAttribute|
|                 |   0.00% | 18.164us  |       1 | 18.164us | 18.164us | 18.164us | cudaSetDevice|
|                 |   0.00% | 4.9690us  |       1 | 4.9690us | 4.9690us | 4.9690us | cudaGetDeviceProperties|
|                 |   0.00% | 1.8930us  |       3 |    631ns |    230ns | 1.3420us | cuDeviceGetCount|
|                 |   0.00% | 1.1820us  |       2 |    591ns |    170ns | 1.0120us | cuDeviceGet|
|                 |   0.00% |    852ns  |       1 |    852ns |    852ns |    852ns | cuDeviceGetName|
|                 |   0.00% |    421ns  |       1 |    421ns |    421ns |    421ns | cuDeviceTotalMem|
|                 |   0.00% |    221ns  |       1 |    221ns |    221ns |    221ns | cuDeviceGetUuid|

---

### Exlpanation

**GPU Activities:**

- [CUDA memcpy DtoH]: This GPU activity accounts for 51.53% of the GPU time, with a total time of 4.4243 milliseconds. It involves memory copying from the device (GPU) to the host (CPU).
  - This is a simple code that just pases variables from host to device and back to device to host. This shows how the main functions, memcpy of DtoH is slower than the HtoD, since it has a slower clock than the device.

- [CUDA memcpy HtoD]: This GPU activity consumes 48.47% of the GPU time, with a total time of 4.1616 milliseconds. It involves memory copying from the host (CPU) to the device (GPU).
  - Same thing as before but here we can see that its faster than the DtoH since the device taking in the variables is faster to process the recieving end.

**API Calls:**

- cudaMalloc: This API call consumes 93.40% of the time, with a total time of 667.18 milliseconds. 
  - It's called once and is used to allocate memory on the GPU.

- cudaDeviceReset: This API call accounts for 4.85% of the time, with a total time of 34.645 milliseconds. 
  - It resets the CUDA device.

- cudaMemcpy: This API call occupies 1.28% of the time, with a total time of 9.1722 milliseconds. 
  - It's called twice and is used for copying memory data between the host and device.

---

## pinMemTransfer

==947== NVPROF is profiling process 947, command: ./pinMemTransfer

==947== Profiling application: ./pinMemTransfer

==947== Profiling result:

|            Type | Time(%) |     Time  |   Calls |      Avg |      Min |      Max | Name|
|      :---:      | :---:   |  :---:    |  :---:  |  :---:   |   :---:  |   :---:  |:---:|
| GPU activities: |  52.30% | 3.2299ms  |       1 | 3.2299ms | 3.2299ms | 3.2299ms | [CUDA memcpy DtoH]|
|                 |  47.70% | 2.9458ms  |       1 | 2.9458ms | 2.9458ms | 2.9458ms | [CUDA memcpy HtoD]|
|      API calls: |  93.45% | 692.70ms  |       1 | 692.70ms | 692.70ms | 692.70ms | cudaHostAlloc|
|                 |   4.81% | 35.636ms  |       1 | 35.636ms | 35.636ms | 35.636ms | cudaDeviceReset|
|                 |   0.88% | 6.5215ms  |       2 | 3.2608ms | 3.1202ms | 3.4013ms | cudaMemcpy|
|                 |   0.41% | 3.0659ms  |       1 | 3.0659ms | 3.0659ms | 3.0659ms | cuDeviceGetPCIBusId|
|                 |   0.30% | 2.2360ms  |       1 | 2.2360ms | 2.2360ms | 2.2360ms | cudaFreeHost|
|                 |   0.09% | 692.67us  |       1 | 692.67us | 692.67us | 692.67us | cudaMalloc|
|                 |   0.05% | 362.56us  |       1 | 362.56us | 362.56us | 362.56us | cudaFree|
|                 |   0.00% | 19.865us  |     101 |    196ns |    130ns | 1.5730us | cuDeviceGetAttribute|
|                 |   0.00% | 13.686us  |       1 | 13.686us | 13.686us | 13.686us | cudaSetDevice|
|                 |   0.00% | 5.2500us  |       1 | 5.2500us | 5.2500us | 5.2500us | cudaGetDeviceProperties|
|                 |   0.00% | 2.4440us  |       3 |    814ns |    280ns | 1.7440us | cuDeviceGetCount|
|                 |   0.00% | 2.2040us  |       2 | 1.1020us |    250ns | 1.9540us | cuDeviceGet|
|                 |   0.00% | 1.0320us  |       1 | 1.0320us | 1.0320us | 1.0320us | cuDeviceGetName|
|                 |   0.00% |    411ns  |       1 |    411ns |    411ns |    411ns | cuDeviceTotalMem|
|                 |   0.00% |    251ns  |       1 |    251ns |    251ns |    251ns | cuDeviceGetUuid|

---

### Exlpanation

**GPU Activities:**

- [CUDA memcpy DtoH]: This GPU activity accounts for 52.30% of the GPU time, with a total time of 3.2299 milliseconds. It involves memory copying from the device (GPU) to the host (CPU).
  - This is a simple code that just pases variables from host to device and back to device to host. This shows how the main functions, memcpy of DtoH is slower than the HtoD, since it has a slower clock than the device.

- [CUDA memcpy HtoD]: This GPU activity consumes 47.70% of the GPU time, with a total time of 2.9458 milliseconds. It involves memory copying from the host (CPU) to the device (GPU).
  - Same thing as before but here we can see that its faster than the DtoH since the device taking in the variables is faster to process the recieving end.

The diference here is that now the API.

**API Calls:**

- cudaHostAlloc: This API call consumes 93.45% of the time, with a total time of 692.70 milliseconds. 
  - It's called once and is used to allocate memory on the host (pinned memory).
  - HostAlloc is the highest instead of the normal cudaMalloc since its which lets the host give the memory allocation to the device to help make the cudamemcpy faster than it would normally do, although it can 'degrade the system performenced if over-used'.

- cudaDeviceReset: This API call accounts for 4.81% of the time, with a total time of 35.636 milliseconds. 
  - It resets the CUDA device.

- cudaMemcpy: This API call occupies 0.88% of the time, with a total time of 6.5215 milliseconds. 
  - It's called twice and is used for copying memory data between the host and device.

- cuDeviceGetPCIBusId: This API call takes up 0.41% of the time, with a total time of 3.0659 milliseconds. 
  - It retrieves the PCI bus ID of the CUDA device.

- cudaFreeHost: This API call accounts for 0.30% of the time, with a total time of 2.2360 milliseconds. 
  - It's called once and is used to free allocated host memory.

---

## readSegment

==963== NVPROF is profiling process 963, command: ./readSegment

==963== Profiling application: ./readSegment

==963== Profiling result:

|            Type | Time(%) |     Time   |  Calls |      Avg |      Min |      Max | Name|
|      :---:      | :---:   |  :---:     |  :---: |  :---:   |   :---:  |   :---:  |:---:|
| GPU activities: |  52.31% | 1.7676ms   |      2 | 883.82us | 861.14us | 906.51us | [CUDA memcpy HtoD]|
|                 |  42.67% | 1.4420ms   |      1 | 1.4420ms | 1.4420ms | 1.4420ms | [CUDA memcpy DtoH]|
|                 |   2.53% | 85.569us   |      1 | 85.569us | 85.569us | 85.569us | warmup(float*, float*, float*, int, int)|
|                 |   2.49% | 84.002us   |      1 | 84.002us | 84.002us | 84.002us | readOffset(float*, float*, float*, int, int)|
|      API calls: |  93.54% | 651.44ms   |      3 | 217.15ms | 730.69us | 649.95ms | cudaMalloc|
|                 |   5.09% | 35.479ms   |      1 | 35.479ms | 35.479ms | 35.479ms | cudaDeviceReset|
|                 |   0.62% | 4.2860ms   |      3 | 1.4287ms | 1.0987ms | 2.0832ms | cudaMemcpy|
|                 |   0.53% | 3.6576ms   |      1 | 3.6576ms | 3.6576ms | 3.6576ms | cuDeviceGetPCIBusId|
|                 |   0.12% | 818.01us   |      3 | 272.67us | 189.53us | 425.11us | cudaFree|
|                 |   0.09% | 623.48us   |      2 | 311.74us | 138.17us | 485.30us | cudaDeviceSynchronize|
|                 |   0.01% | 88.408us   |      2 | 44.204us | 13.075us | 75.333us | cudaLaunchKernel|
|                 |   0.00% | 19.024us   |    101 |    188ns |    130ns | 1.4430us | cuDeviceGetAttribute|
|                 |   0.00% | 9.9490us   |      1 | 9.9490us | 9.9490us | 9.9490us | cudaGetDeviceProperties|
|                 |   0.00% | 6.1410us   |      1 | 6.1410us | 6.1410us | 6.1410us | cudaSetDevice|
|                 |   0.00% | 2.1940us   |      3 |    731ns |    240ns | 1.2630us | cuDeviceGetCount|
|                 |   0.00% | 1.5340us   |      2 |    767ns |    722ns |    812ns | cudaGetLastError|
|                 |   0.00% | 1.2930us   |      2 |    646ns |    241ns | 1.0520us | cuDeviceGet|
|                 |   0.00% | 1.0720us   |      1 | 1.0720us | 1.0720us | 1.0720us | cuDeviceGetName|
|                 |   0.00% |    451ns   |      1 |    451ns |    451ns |    451ns | cuDeviceTotalMem|
|                 |   0.00% |    260ns   |      1 |    260ns |    260ns |    260ns | cuDeviceGetUuid|

---

### Exlpanation

**GPU Activities:**

- [CUDA memcpy HtoD]: This GPU activity accounts for 52.31% of the GPU time, with a total time of 1.7676 milliseconds. It involves memory copying from the host (CPU) to the device (GPU).
  - On this operation the HtoD is the one with the higher time usage. Normally its the other way around, but the reason why its higher here is that the misaligned reads affects on the performace making the device take longer than the host. The clock speed is still the same but now this time the device had more process than the host.

- [CUDA memcpy DtoH]: This GPU activity consumes 42.67% of the GPU time, with a total time of 1.4420 milliseconds. It involves memory copying from the device (GPU) to the host (CPU).
  - As said in the HtoD, this process ended up being faster since the impact of misaligned reads affected the performence of the device making it take more time this time than the host. Since here we are reading the segment, the one reading is the host showing how its slower.

- readOffset(float, float, float*, int, int)**: This GPU activity accounts for 2.49% of the time, with a total time of 84.002 microseconds. 
  - this does a similar function like the writeoffset function in the writeSegment code, which instead on writing an array its focusing on reading them which causes the host to take longer.

**API Calls:**

- cudaMalloc: This API call consumes 93.54% of the time, with a total time of 651.44 milliseconds. 
  - It's called three times and is used to allocate memory on the device (GPU).

- cudaDeviceReset: This API call accounts for 5.09% of the time, with a total time of 35.479 milliseconds. 
  - It resets the CUDA device.

- cudaMemcpy: This API call occupies 0.62% of the time, with a total time of 4.2860 milliseconds. 
  - It's called three times and is used for copying memory data between the host and device.

- cuDeviceGetPCIBusId: This API call takes up 0.53% of the time, with a total time of 3.6576 milliseconds. 
  - It retrieves the PCI bus ID of the CUDA device.

---

## readSegmentUnroll

==985== NVPROF is profiling process 985, command: ./readSegmentUnroll

==985== Profiling application: ./readSegmentUnroll

==985== Profiling result:

|            Type | Time(%) |     Time   |  Calls |      Avg |      Min |      Max | Name|
| GPU activities: |  59.86% | 3.2882ms   |      3 | 1.0961ms | 868.94us | 1.5457ms | [CUDA memcpy DtoH]|
|                 |  31.90% | 1.7523ms   |      2 | 876.14us | 857.20us | 895.09us | [CUDA memcpy HtoD]|
|                 |   2.02% | 110.91us   |      4 | 27.728us | 26.561us | 28.928us | [CUDA memset]|
|                 |   1.58% | 86.657us   |      1 | 86.657us | 86.657us | 86.657us | readOffsetUnroll4(float*, float*, float*, int, int)|
|                 |   1.56% | 85.729us   |      1 | 85.729us | 85.729us | 85.729us | warmup(float*, float*, float*, int, int)|
|                 |   1.54% | 84.801us   |      1 | 84.801us | 84.801us | 84.801us | readOffsetUnroll2(float*, float*, float*, int, int)|
|                 |   1.54% | 84.482us   |      1 | 84.482us | 84.482us | 84.482us | readOffset(float*, float*, float*, int, int)|
|      API calls: |  92.85% | 643.00ms   |      3 | 214.33ms | 511.27us | 641.93ms | cudaMalloc|
|                 |   5.34% | 37.016ms   |      1 | 37.016ms | 37.016ms | 37.016ms | cudaDeviceReset|
|                 |   1.03% | 7.1436ms   |      5 | 1.4287ms | 1.0554ms | 2.7758ms | cudaMemcpy|
|                 |   0.42% | 2.9003ms   |      1 | 2.9003ms | 2.9003ms | 2.9003ms | cuDeviceGetPCIBusId|
|                 |   0.17% | 1.2059ms   |      4 | 301.47us | 214.76us | 492.97us | cudaDeviceSynchronize|
|                 |   0.13% | 892.63us   |      3 | 297.54us | 177.67us | 489.14us | cudaFree|
|                 |   0.03% | 174.78us   |      4 | 43.695us | 32.993us | 48.763us | cudaMemset|
|                 |   0.02% | 164.32us   |      4 | 41.080us | 15.550us | 83.809us | cudaLaunchKernel|
|                 |   0.00% | 19.819us   |    101 |    196ns |    120ns | 1.4330us | cuDeviceGetAttribute|
|                 |   0.00% | 11.221us   |      1 | 11.221us | 11.221us | 11.221us | cudaGetDeviceProperties|
|                 |   0.00% | 6.8230us   |      1 | 6.8230us | 6.8230us | 6.8230us | cudaSetDevice|
|                 |   0.00% | 4.4580us   |      4 | 1.1140us |    942ns | 1.5830us | cudaGetLastError|
|                 |   0.00% | 1.7840us   |      3 |    594ns |    230ns | 1.2130us | cuDeviceGetCount|
|                 |   0.00% | 1.5930us   |      2 |    796ns |    271ns | 1.3220us | cuDeviceGet|
|                 |   0.00% | 1.0920us   |      1 | 1.0920us | 1.0920us | 1.0920us | cuDeviceGetName|
|                 |   0.00% |    541ns   |      1 |    541ns |    541ns |    541ns | cuDeviceTotalMem|
|                 |   0.00% |    251ns   |      1 |    251ns |    251ns |    251ns | cuDeviceGetUuid|

---

### Exlpanation

**GPU Activities:**

- [CUDA memcpy DtoH]: This GPU activity accounts for 59.86% of the GPU time, with a total time of 3.2882 milliseconds. It involves memory copying from the device (GPU) to the host (CPU).
  - This operation called in line 166 and in line 180, which passes a copy of the values of the device which is the result of a kernel to the host so it can show it to the user.

- [CUDA memcpy HtoD]: This GPU activity consumes 31.90% of the GPU time, with a total time of 1.7523 milliseconds. It involves memory copying from the host (CPU) to the device (GPU).
  - This operation called in lines 143 and 144 pass a copy of the initial values to the device in order to do operations on them so it can send it back. This took lower than the DtoH since the device can process things faster because of its clock than the host.

- [CUDA memset]: This GPU activity occupies 2.02% of the time, with a total time of 110.91 microseconds. It involves memory setting operations.
  - this function takes the pointer ptr as the address of the memory region you want to change. It sets each byte in the specified memory region to the given value. It does this operation for however many number of bytes, which is determined by the '0x00' parameter, then returns a poniter function.

- readOffsetUnroll4(float, float, float*, int, int)**: This GPU activity takes up 1.58% of the time, with a total time of 86.657 microseconds. It appears to be a function named "readOffsetUnroll4" involving operations on float arrays.
- readOffsetUnroll2(float, float, float*, int, int)**: This GPU activity occupies 1.54% of the time, with a total time of 84.801 microseconds. It appears to be a function named "readOffsetUnroll2" involving operations on float arrays.
- readOffset(float, float, float*, int, int)**: This GPU activity takes up 1.54% of the time, with a total time of 84.482 microseconds. It appears to be a function named "readOffset" involving operations on float arrays.
  - Here we can see that the offsett not unrolled took the shortest time, as in compared to the other code that it took the longest. The reason is that when used kernels that reduce the performace impact of misaligned reads with unrolling also make it so that the functions that are suppose to make it faster en up being slower. which in this reasons its better to use not try to misaligned reads in order to have a faster performance.

**API Calls:**

- cudaMalloc: This API call consumes 92.85% of the time, with a total time of 643.00 milliseconds. 
  - It's called three times and is used for allocating memory on the device (GPU).

- cudaDeviceReset: This API call accounts for 5.34% of the time, with a total time of 37.016 milliseconds. 
  - It resets the CUDA device.

- cudaMemcpy: This API call occupies 1.03% of the time, with a total time of 7.1436 milliseconds. 
  - It's called five times and is used for copying memory data between the host and device.

---