Hello world





### <u>Questions and Solutions</u>


??? example "Examples: Vector Addition"


    === "C/C++ - version"
    
        ```c  
        //-*-C++-*-
        // Vector-addition.c
        
        #include <stdio.h>
        #include <stdlib.h>
        #include <math.h>
        #include <assert.h>
        #include <time.h>
        
        #define N 5120
        #define MAX_ERR 1e-6

        // CPU function that adds two vector 
        float * Vector_Add(float *a, float *b, float *out, int n) 
        {
          for(int i = 0; i < n; i ++)
            {
              out[i] = a[i] + b[i];
            }
          return out;
        }

        int main()
        {
          // Initialize the variables
          float *a, *b, *out;       
  
          // Allocate the memory
          a   = (float*)malloc(sizeof(float) * N);
          b   = (float*)malloc(sizeof(float) * N);
          out = (float*)malloc(sizeof(float) * N);
  
          // Initialize the arrays
          for(int i = 0; i < N; i++)
            {
              a[i] = 1.0f;
              b[i] = 2.0f;
            }
    
          // Start measuring time
          clock_t start = clock();

          // Executing vector addtion function 
          Vector_Add(a, b, out, N);

          // Stop measuring time and calculate the elapsed time
          clock_t end = clock();
          double elapsed = (double)(end - start)/CLOCKS_PER_SEC;
        
          printf("Time measured: %.3f seconds.\n", elapsed);
  
          // Verification
          for(int i = 0; i < N; i++)
            {
              assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
            }

          printf("out[0] = %f\n", out[0]);
          printf("PASSED\n");
    
          // Deallocate the memory
          free(a); 
          free(b); 
          free(out);
   
          return 0;
        }
        ```

    === "FORTRAN - version"
        ```
        module Vector_Addition_Mod  
        implicit none 
          contains
        subroutine Vector_Addition(a, b, c, n)
        ! Input vectors
        real(8), intent(in), dimension(:) :: a
        real(8), intent(in), dimension(:) :: b
        real(8), intent(out), dimension(:) :: c
        integer :: i, n
          do i = 1, n
            c(i) = a(i) + b(i)
          end do
         end subroutine Vector_Addition
        end module Vector_Addition_Mod

        program main
        use Vector_Addition_Mod
        implicit none
        ! Input vectors
        real(8), dimension(:), allocatable :: a
        real(8), dimension(:), allocatable :: b 
        ! Output vector
        real(8), dimension(:), allocatable :: c
        ! real(8) :: sum = 0

        integer :: n, i  
        print *, "This program does the addition of two vectors "
        print *, "Please specify the vector size = "
        read *, n

        ! Allocate memory for vector
        allocate(a(n))
        allocate(b(n))
        allocate(c(n))
  
        ! Initialize content of input vectors, 
        ! vector a[i] = sin(i)^2 vector b[i] = cos(i)^2
        do i = 1, n
          a(i) = sin(i*1D0) * sin(i*1D0)
          b(i) = cos(i*1D0) * cos(i*1D0) 
        enddo
    
        ! Call the vector add subroutine 
        call Vector_Addition(a, b, c, n)

        !!Verification
        do i = 1, n
          if (abs(c(i)-(a(i)+b(i)) == 0.00000)) then 
           else
             print *, "FAIL"
           endif
        enddo
        print *, "PASS"
    
        ! Delete the memory
        deallocate(a)
        deallocate(b)
        deallocate(c)
  
        end program main

        ```



    === "C/C++-template"
    
        ```c
	        //-*-C++-*-
        // Vector-addition.c
        
        #include <stdio.h>
        #include <stdlib.h>
        #include <math.h>
        #include <assert.h>
        #include <time.h>
        
        #define N 5120
        #define MAX_ERR 1e-6

        // CPU function that adds two vector 
        float * Vector_Add(float *a, float *b, float *out, int n) 
        {
        // ADD YOUR PARALLEL REGION FOR THE LOOP
          for(int i = 0; i < n; i ++)
            {
              out[i] = a[i] + b[i];
            }
          return out;
        }

        int main()
        {
          // Initialize the variables
          float *a, *b, *out;       
  
          // Allocate the memory
          a   = (float*)malloc(sizeof(float) * N);
          b   = (float*)malloc(sizeof(float) * N);
          out = (float*)malloc(sizeof(float) * N);
  
          // Initialize the arrays
          for(int i = 0; i < N; i++)
            {
              a[i] = 1.0f;
              b[i] = 2.0f;
            }
    
          // Start measuring time
          clock_t start = clock();


          #pragma omp parallel
          // Executing vector addtion function 
          Vector_Add(a, b, out, N);

          // Stop measuring time and calculate the elapsed time
          clock_t end = clock();
          double elapsed = (double)(end - start)/CLOCKS_PER_SEC;
        
          printf("Time measured: %.3f seconds.\n", elapsed);
  
          // Verification
          for(int i = 0; i < N; i++)
            {
              assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
            }

          printf("out[0] = %f\n", out[0]);
          printf("PASSED\n");
    
          // Deallocate the memory
          free(a); 
          free(b); 
          free(out);
   
          return 0;
        }

        ```
        
    === "CUDA-version"
    
        ```c  
        //-*-C++-*-
        // Vector-addition.cu
        
        #include <stdio.h>
        #include <stdlib.h>
        #include <math.h>
        #include <assert.h>
        #include <time.h>
        #include <cuda.h>

        #define N 5120
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
          a   = (float*)malloc(sizeof(float) * N);
          b   = (float*)malloc(sizeof(float) * N);
          out = (float*)malloc(sizeof(float) * N);
           
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
          dim3 dimGrid(ceil(N/32), ceil(N/32), 1);
          dim3 dimBlock(32, 32, 1); 

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

??? "Compilation and Output"

    === "Serial-version"
        ```c
        // compilation
        $ gcc Vector-addition.c -o Vector-Addition-CPU
        
        // execution 
        $ ./Vector-Addition-CPU
        
        // output
        $ ./Vector-addition-CPU 
        out[0] = 3.000000
        PASSED
        ```
        
    === "CUDA-version"
        ```c
        // compilation
        $ nvcc -arch=compute_70 Vector-addition.cu -o Vector-Addition-GPU
        
        // execution
        $ ./Vector-Addition-GPU
        
        // output
        $ ./Vector-addition-GPU
        out[0] = 3.000000
        PASSED
        ```


??? Question "Questions"

    - What happens if you remove the **`__syncthreads();`** from the **`__global__ void vector_add(float *a, float *b, 
       float *out, int n)`** function.
    - Can you remove the if condition **`if(i < n)`** from the **`__global__ void vector_add(float *a, float *b,
       float *out, int n)`** function. If so how can you do that?
    - Here we do not use the **`cudaDeviceSynchronize()`** in the main application, can you figure out why we
        do not need to use it. 
    - Can you create a different kinds of threads block for larger number of array?
