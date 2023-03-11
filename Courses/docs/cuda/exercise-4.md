##### Unified Memory

 - Just one memory allocation is enough [`cudaMallocManaged`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gd228014f19cc0975ebe3e0dd2af6dd1b)

<figure markdown>
![](/figures/unified-memory.svg)
<figcaption>b</figcaption>
</figure>


|Without unified memory|With unified memory|
|----------------------|-------------------|
|Allocate the host memory|~~Allocate the host memory~~|
|Allocate the device memory|Allocate the device memory|
|Initialize the host value|Initialize the host value|
|Transfer the host value to the device memory location|~~Transfer the host value to the device memory location~~|
|Do the computation using the CUDA kernel|Do the computation using the CUDA kernel|
|Transfer the data from the device to host|~~Transfer the data from the device to host~~|
|Free device memory|Free device memory|
|Free host memor|~~Free host memory~~|


??? example "Examples: Unified Memory - Vector Addition"


    === "Without Unified Memory"

        ```c
        //-*-C++-*-
        #include <stdio.h>
        #include <stdlib.h>
        #include <math.h>
        #include <assert.h>
        #include <time.h>
        
        #define N 256
        #define MAX_ERR 1e-6


        // GPU function that adds two vectors 
        __global__ void vector_add(float *a, float *b, 
                float *out, int n) 
        {

          int i = blockIdx.x * blockDim.x * blockDim.y + 
           threadIdx.y * blockDim.x + threadIdx.x;   
          // Allow the   threads only within the size of N
          if(i < n)
            {
              out[i] = a[i] + b[i];
            }

          // Synchronice all the threads 
          __syncthreads();
        }
 
        int main()
        {
          // Initialize the memory on the host
          float *a, *b, *out; 

          // Allocate host memory
          a = (float*)malloc(sizeof(float) * N);
          b = (float*)malloc(sizeof(float) * N);
          c = (float*)malloc(sizeof(float) * N);

          // Initialize the memory on the device
          float *d_a, *d_b, *d_out;

          // Allocate device memory
          cudaMalloc((void**)&d_a, sizeof(float) * N);
          cudaMalloc((void**)&d_b, sizeof(float) * N);
          cudaMalloc((void**)&d_out, sizeof(float) * N); 

          // Initialize host arrays
          for(int i = 0; i < N; i++)
            {
              a[i] = 1.0f;
              b[i] = 2.0f;
            }

          // Transfer data from host to device memory
          cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
          cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

          // Thread organization 
          dim3 dimGrid(1, 1, 1);    
          dim3 dimBlock(16, 16, 1); 

          // execute the CUDA kernel function 
          vector_add<<<dimGrid, dimBlock>>>(d_a, d_b, d_out, N);

          // Transfer data back to host memory
          cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

          // Verification
          for(int i = 0; i < N; i++)
             {
               assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
             }

          printf("out[0] = %f\n", out[0]);
          printf("PASSED\n");

          // Deallocate device memory
          cudaFree(d_a);
          cudaFree(d_b);
          cudaFree(d_out);

          // Deallocate host memory
          free(a); 
          free(b); 
          free(out);
  
          return 0;
        }
        ```

    === "With Unified Memory"
   
        ```c
        //-*-C++-*-
        #include <stdio.h>
        #include <stdlib.h>
        #include <math.h>
        #include <assert.h>
        #include <time.h>

        #define N 256
        #define MAX_ERR 1e-6


        // GPU function that adds two vectors 
        __global__ void vector_add(float *a, float *b, 
                                   float *out, int n) 
        {
          int i = blockIdx.x * blockDim.x * blockDim.y + 
            threadIdx.y * blockDim.x + threadIdx.x;   
          // Allow the   threads only within the size of N
          if(i < n)
            {
              out[i] = a[i] + b[i];
            }

          // Synchronice all the threads 
          __syncthreads();
        }

        int main()
        {
          /*
          // Initialize the memory on the host
          float *a, *b, *out;
    
          // Allocate host memory
          a = (float*)malloc(sizeof(float) * N);
          b = (float*)malloc(sizeof(float) * N);
          c = (float*)malloc(sizeof(float) * N);
          */
   
          // Initialize the memory on the device
          float *d_a, *d_b, *d_out;

          // Allocate device memory
          cudaMallocManaged(&d_a, sizeof(float) * N);
          cudaMallocManaged(&d_b, sizeof(float) * N);
          cudaMallocManaged(&d_out, sizeof(float) * N); 
  
         // Initialize host arrays
         for(int i = 0; i < N; i++)
           {
             d_a[i] = 1.0f;
             d_b[i] = 2.0f;
           }

         /*
         // Transfer data from host to device memory
         cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
         cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);
         */

         // Thread organization 
         dim3 dimGrid(1, 1, 1);    
         dim3 dimBlock(16, 16, 1); 

         // execute the CUDA kernel function 
         vector_add<<<dimGrid, dimBlock>>>(d_a, d_b, d_out, N);
         cudaDeviceSynchronize();
         /*
         // Transfer data back to host memory
         cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
         */
  
         // Verification
         for(int i = 0; i < N; i++)
           {
             assert(fabs(d_out[i] - d_a[i] - d_b[i]) < MAX_ERR);
           }

         printf("out[0] = %f\n", d_out[0]);
         printf("PASSED\n");
    
         // Deallocate device memory
         cudaFree(d_a);
         cudaFree(d_b);
         cudaFree(d_out);

         /*
         // Deallocate host memory
         free(a); 
         free(b); 
         free(out);
         */
  
         return 0;
        }
        ```
