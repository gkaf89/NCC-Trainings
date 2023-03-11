
Now our first exercise would be to print out the hello world from GPU.
To do that, we need to do the following things:

 - Run a part or entire application on the GPU
 - Call cuda function on device
 - It should be called using function qualifier **`__global__`**
 - Calling the device function on the main program:
 - C/C++ example, **`c_function()`**
 - CUDA example, **`cuda functio<<<1,1>>>()`** (just using 1 thread)
 - **`<<< >>>`**, specify the threads blocks within the bracket
 - Make sure to synchronize the threads
 - **`syncthreads()`** synchronizes all the threads within a thread block
 - **`CudaDeviceSynchronize()`** synchronizes a kernel call in host
 - Most of the CUDA APIs are synchronized calls by default (but sometimes
   it is good to call explicit synchronized calls to avoid errors
   in the computation)

??? Example "Examples: Hello World"

    === "Serial-version"

        ```c
        #include<studio.h>
        #include<cuda.h>
        
        void c_function()
        {
          printf{"Hello World!\n"};
        }
        
        int main()
        {
          c_function();
          return 0;
        }
        ```


    === "CUDA-version"

        ```c
        #include<studio.h>
        #include<cuda.h>
        
        // device function will be executed on device (GPU) 
        __global__
        void cuda_function()
        {
          printf{"Hello World from GPU!\n"};
          
          // synchronize all the threads
          __syncthreads();
        }
   
        int main()
        {
        
          // call the kernel function 
          cuda_function<<<1,1>>>();
          
          // synchronize the device kernel call
          cudaDeviceSynchronize();
          return 0;
        }
        ```


??? Question

    Right now, you are printing just one `Hello World from GPU`,
    but what if you would like to print more `Hello World from GPU`? How can you do that?


    === "Question"

        ```c
        #include<studio.h>
        __global__ void cuda_function()
        {
          printf{"Hello World from GPU!\n"};
          __syncthreads();
        }

        int main()
        {
        // define your thread block here
          cuda_function<<<>>>();
          cudaDeviceSynchronize();
          return 0;
        }
        ```
    
    === "Answer"
  
        ```c
        #include<studio.h>
        __global__ void cuda_function()
        {
          printf{"Hello World from GPU!\n"};
          __syncthreads();
        }

        int main()
        {
          cuda_function<<<10,1>>>();
          cudaDeviceSynchronize();
          return 0;
        }
        ```

    === "Solution Output"

        ```c
        Hello World from GPU!
        Hello World from GPU!
        Hello World from GPU!
        Hello World from GPU!
        Hello World from GPU!
        Hello World from GPU!
        Hello World from GPU!
        Hello World from GPU!
        Hello World from GPU!
        Hello World from GPU!
        ```
