## Introduction

##### Important differences between CPU and GPU
- GPU has many cores compared to CPU
- But on the other hand, the CPU's frequency is higher than the GPU. That makes the CPU faster in computing compared to GPU
    - [Intel® Core™ i7-10700K Processor](https://ark.intel.com/content/www/us/en/ark/products/199335/intel-core-i710700k-processor-16m-cache-up-to-5-10-ghz.html)
       base frequency is 3.80 GHz, whereas, [Nvidia Ampere has 0.765 GHz](https://www.techpowerup.com/gpu-specs/a100-pcie.c3623)
    - [The higher the frequency, the faster the processor can do the computation](https://www.intel.com/content/www/us/en/gaming/resources/cpu-clock-speed.html)


<figure markdown>
![](/figures/cpu-gpu.png){ align=middle width=700}
<figcaption></figcaption>
</figure>


- However, GPU can handle many threads in parallel, which can process many data in parallel
- In the GPU, cores are grouped into GPU Processing Clusters (GPCs), and each GPC has its own Streaming Multiprocessors (SMs) and Texture Processor Clusters (TPCs)
- Nvidia (microarchitecture): Tesla (2006), Fermi (2010), Kepler (2012), Maxwell (2014),
[Pascal (2016)](https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper.pdf),
[Volta (2017)](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf),
[Turing (2018)](https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf),
[Ampere (2020)](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf)
 - Video Link: [Mythbusters Demo GPU versus CPU](https://www.youtube.com/watch?v=-P28LKWTzrI)


<figure markdown>
![](/figures/memory-spaces-on-cuda-device.png){ align=left width=700}
<figcaption>a</figcaption>
</figure>


##### Serial programming vs parallel programming
- Serial programming
    - An entire problem can be divided into discrete series of instructions
    - All the instructions are executed one by one
    - Executed by single thread or processor
    - Only one instruction can be executed at the same time
  
- Parallel programming 
    - An entire problem can be divided into discrete
    parts in such a way that it can be solved concurrently
    - Each part may have a set of instructions
    - Each part instructions are executed on a different thread/processor
    - Since it is parallel execution, a target problem needs to be controlled/coordinated
    - CPU, GPU, and other parallel processors can perform the parallel computing


<figure markdown>
![](/figures/serial-1.png){ align=right width=350}
![](/figures/parallel-1.svg){ align=left width=350}
<figcaption>b</figcaption>
</figure>

