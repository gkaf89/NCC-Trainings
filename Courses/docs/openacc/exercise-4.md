Unified memory simplifies the explicit data movement from host to device by programmers.
OpenACC API will manage the data transfer between CPU and GPU.
In this example, we will look into vector addition in GPU using the unified memory concept.


<figure markdown>
![](../figures/unified-memory-white.png)
<figcaption></figcaption> 
</figure>


 - Just using the compiler flag **`-gpu=managed`** will enable the unified memory in OpenACC.
 
 The below table summarises the required steps needed for the unified memory concept.

!!! Info "Unified Memory"

    === "C/C++"
        ```
        nvc -fast -acc=gpu -gpu=cc80 -gpu=managed -Minfo=accel test.c
        ```
    === "FORTRAN"
        ```
        nvfortran -fast -acc=gpu -gpu=cc80 -gpu=managed -Minfo=accel test.c
        ```


|__Without unified memory__|__With unified memory__|
|----------------------|-------------------|
|Allocate the host memory|Allocate the host memory|
|Initialize the host value|Initialize the host value|
|Use data cluases, e.g,. copy, copyin|~~Use data cluases, e.g,. copy, copyin~~|
|Do the computation using the GPU kernel|Do the computation using the GPU kernel|
|Free host memory|Free host memory|


### <u>Questions and Solutions</u>

??? example "Examples: Vector Addition"

    === "OpenACC-template"
    
        ```c  
        // Vector-addition-template.c
	
        #include <stdio.h>
        #include <stdlib.h>
        #include <math.h>
        #include <assert.h>
        #include <time.h>
        #include <openacc.h>	


        #define N 5120
        #define MAX_ERR 1e-6


        // GPU function that adds two vectors 
        // function that adds two vector 
        void Vector_Addition(float *restrict a, float *restrict b, float *restrict c, int n) 
        {

        // add here either parallel or kernel and do need to add data map clauses
        #pragma acc 
        for(int i = 0; i < n; i ++)
           {
             c[i] = a[i] + b[i];
           }
        }

        int main()
        {
          // Initialize the memory on the host
          float *restrict a, *restrict b, *restrict c;       
  
          // Allocate host memory
          a = (float*)malloc(sizeof(float) * N);
          b = (float*)malloc(sizeof(float) * N);
          c = (float*)malloc(sizeof(float) * N);
  
          // Initialize host arrays
          for(int i = 0; i < N; i++)
            {
              a[i] = 1.0f;
              b[i] = 2.0f;
            }
    
          // Start measuring time
          clock_t start = clock();

          // Executing CPU function 
          Vector_Addition(a, b, c, N);

          // Stop measuring time and calculate the elapsed time
          clock_t end = clock();
          double elapsed = (double)(end - start)/CLOCKS_PER_SEC;
        
          printf("Time measured: %.3f seconds.\n", elapsed);
  
          // Verification
          for(int i = 0; i < N; i++)
            {
              assert(fabs(c[i] - a[i] - b[i]) < MAX_ERR);
            }

          printf("PASSED\n");
    
          // Deallocate host memory
          free(a); 
          free(b); 
          free(c);
   
          return 0;
        }
        ```
	
    === "OpenACC-version"
    
        ```c  
        // Vector-addition-openacc.c
        
        #include <stdio.h>
        #include <stdlib.h>
        #include <math.h>
        #include <assert.h>
        #include <time.h>
        #include <openacc.h>

        #define N 5120
        #define MAX_ERR 1e-6


        // function that adds two vector 
        void Vector_Addition(float *restrict a, float *restrict b, float *restrict c, int n) 
        {
        // or #pragma acc kernels loop copyin(a[0:n], b[0:n]) copyout(c[0:n])
        #pragma acc kernels loop //copyin(a[0:n], b[0:n]) copyout(c[0:n])
        for(int i = 0; i < n; i ++)
           {
            c[i] = a[i] + b[i];
           }
        }

        int main()
        {
          // Initialize the memory on the host
          float *restrict a, *restrict b, *restrict c;       
  
          // Allocate host memory
          a = (float*)malloc(sizeof(float) * N);
          b = (float*)malloc(sizeof(float) * N);
          c = (float*)malloc(sizeof(float) * N);
  
          // Initialize host arrays
          for(int i = 0; i < N; i++)
            {
              a[i] = 1.0f;
              b[i] = 2.0f;
            }
    
          // Start measuring time
          clock_t start = clock();

          // Executing CPU function 
          Vector_Addition(a, b, c, N);

          // Stop measuring time and calculate the elapsed time
          clock_t end = clock();
          double elapsed = (double)(end - start)/CLOCKS_PER_SEC;
          
          printf("Time measured: %.3f seconds.\n", elapsed);
  
          // Verification
          for(int i = 0; i < N; i++)
            {
              assert(fabs(c[i] - a[i] - b[i]) < MAX_ERR);
            }

          printf("PASSED\n");
    
          // Deallocate host memory
          free(a); 
          free(b); 
          free(c);
   
          return 0;
        }
        ```


??? "Compilation and Output"
        
    === "OpenACC-version"
        ```c
        // compilation
        $ nvc -fast -acc=gpu -gpu=cc80 -Minfo=accel -gpu=managed Vector-addition-openacc.c -o Vector-Addition-GPU
        Vector_Addition:
        12, Generating copyin(a[:n]) [if not already present]
            Generating copyout(c[:n]) [if not already present]
            Generating copyin(b[:n]) [if not already present]
        14, Loop is parallelizable
            Generating NVIDIA GPU code
            14, #pragma acc loop gang, vector(128) /* blockIdx.x threadIdx.x */

        // execution
        $ ./Vector-Addition-GPU
        
        // output
        $ ./Vector-addition-GPU
        PASSED
        ```

??? Question "Questions"

    - Do you already see any performance difference? Using unified memory?

