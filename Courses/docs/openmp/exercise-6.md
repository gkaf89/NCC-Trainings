We will now look into the basic matrix multiplication.
In this example, we will perform the matrix multiplication. Matrix multiplication involves a nested loop. Again, most of the time, we might end up doing computation with a nested loop. Therefore, studying this example would be good practice for solving the nested loop in the future. 

<figure markdown>
![](../figures/mat.png){align=center width=500}
<figcaption>b</figcaption>
</figure>

 - Allocating the CPU memory for A, B, and C matrix.
   Here we notice that the matrix is stored in a
   1D array because we want to consider the same function concept for CPU and GPU.
```c
// Initialize the memory on the host
float *a, *b, *c;

// Allocate host memory
a   = (float*)malloc(sizeof(float) * (N*N));
b   = (float*)malloc(sizeof(float) * (N*N));
c   = (float*)malloc(sizeof(float) * (N*N));
```

 - Allocating the GPU memory for A, B, and C matrix
```c
// Initialize the memory on the device
float *d_a, *d_b, *d_c;

// Allocate device memory
cudaMalloc((void**)&d_a, sizeof(float) * (N*N));
cudaMalloc((void**)&d_b, sizeof(float) * (N*N));
cudaMalloc((void**)&d_c, sizeof(float) * (N*N));
```

 - Now we need to fill the values for the matrix A and B.
```c
// Initialize host matrix
for(int i = 0; i < (N*N); i++)
   {
    a[i] = 2.0f;
    b[i] = 2.0f;
   }
```

- Transfer initialized A and B matrix
from CPU to GPU
```c
cudaMemcpy(d_a, a, sizeof(float) * (N*N), cudaMemcpyHostToDevice);
cudaMemcpy(d_b, b, sizeof(float) * (N*N), cudaMemcpyHostToDevice);
```

 - 2D thread block for indexing x and y
```c
// Thread organization
int blockSize = 32;
dim3 dimBlock(blockSize,blockSize,1);
dim3 dimGrid(ceil(N/float(blockSize)),ceil(N/float(blockSize)),1);
```

 - Calling the kernel function
```c
// Device function call
matrix_mul<<<dimGrid,dimBlock>>>(d_a, d_b, d_c, N);
```
 -  ??? "matrix multiplication function call"

        === "serial"
            ```c
            float * matrix_mul(float *h_a, float *h_b, float *h_c, int width)
            {
              for(int row = 0; row < width ; ++row)
                {
                  for(int col = 0; col < width ; ++col)
                    {
                      float temp = 0;
                      for(int i = 0; i < width ; ++i)
                        {
                          temp += h_a[row*width+i] * h_b[i*width+col];
                        }
                      h_c[row*width+col] = temp;
                    }
                }
              return h_c;
            }
            ```
        === "cuda"
            ```c
            __global__ void matrix_mul(float* d_a, float* d_b, 
            float* d_c, int width)
            {
              int row = blockIdx.x * blockDim.x + threadIdx.x;
              int col = blockIdx.y * blockDim.y + threadIdx.y;
    
              if ((row < width) && (col < width)) 
                {
                  float temp = 0;
                  // each thread computes one 
                  // element of the block sub-matrix
                  for (int i = 0; i < width; ++i) 
                    {
                      temp += d_a[row*width+i]*d_b[i*width+col];
                    }
                  d_c[row*width+col] = temp;
                }
            }
            ```

 - Copy back computed value from GPU to CPU;
   transfer the data back to GPU (from device to host).
   Here is the C matrix that contains the product of the two matrices.
```c
// Transfer data back to host memory
cudaMemcpy(c, d_c, sizeof(float) * (N*N), cudaMemcpyDeviceToHost);
```

 - Deallocate the host and device memory
```c
// Deallocate device memory
cudaFree(d_a);
cudaFree(d_b);
cudaFree(d_c);

// Deallocate host memory
free(a); 
free(b); 
free(c);
```

### <u>Questions and Solutions</u>


??? example "Examples: Matrix Multiplication" 


    === "Serial(C/C++)"
        ```c
        #include<stdio.h>
        #include<stdlib.h>
        #include<omp.h>
        
        void Matrix_Multiplication(float *a, float *b, float *c, int width)   
        { 
          float sum = 0;
          for(int row = 0; row < width ; ++row)                           
            {                                                             
              for(int col = 0; col < width ; ++col)
                {
                  sum=0;
                  for(int i = 0; i < width ; ++i)                         
                    {                                                     
                      sum += a[row*width+i] * b[i*width+col];      
                    }                                                     
                  c[row*width+col] = sum;                           
                }
            }   
        }

        int main()
         {  
           printf("Programme assumes that matrix size is N*N \n");
           printf("Please enter the N size number \n");
           int N =0;
           scanf("%d", &N);

           // Initialize the memory on the host
           float *a, *b, *c;       
    
           // Allocate host memory
           a = (float*)malloc(sizeof(float) * (N*N));
           b = (float*)malloc(sizeof(float) * (N*N));
           c = (float*)malloc(sizeof(float) * (N*N));
  
          // Initialize host arrays
          for(int i = 0; i < (N*N); i++)
             {
               a[i] = 1.0f;
               b[i] = 2.0f;
             }

           // Device fuction call 
           Matrix_Multiplication(a, b, c, N);
  
           // Verification
           for(int i = 0; i < N; i++)
              {
              for(int j = 0; j < N; j++)
                 {
          	  printf("%f ", c[j]);

          	}
              printf("\n");
              }
  
            // Deallocate host memory
            free(a); 
            free(b); 
            free(c);

           return 0;
        }

        ```
        
        

    === "Serial(FORTRAN)"
        ```c
        module Matrix_Multiplication_Mod  
        implicit none 
        contains
         subroutine Matrix_Multiplication(a, b, c, width)
        use omp_lib
        ! Input vectors
        real(8), intent(in), dimension(:) :: a
        real(8), intent(in), dimension(:) :: b
        real(8), intent(out), dimension(:) :: c
        real(8) :: sum = 0
        integer :: i, row, col, width

        do row = 0, width-1
           do col = 0, width-1
              sum=0
               do i = 0, width-1
                 sum = sum + (a((row*width)+i+1) * b((i*width)+col+1))
               enddo
              c(row*width+col+1) = sum
           enddo
        enddo


          end subroutine Matrix_Multiplication
        end module Matrix_Multiplication_Mod

        program main
        use Matrix_Multiplication_Mod
        use omp_lib
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
        allocate(a(n*n))
        allocate(b(n*n))
        allocate(c(n*n))
  
        ! Initialize content of input vectors, 
        ! vector a[i] = sin(i)^2 vector b[i] = cos(i)^2
        do i = 1, n*n
           a(i) = sin(i*1D0) * sin(i*1D0)
           b(i) = cos(i*1D0) * cos(i*1D0) 
        enddo

        ! Call the vector add subroutine 
        call Matrix_Multiplication(a, b, c, n)
  
        !!Verification
        do i=1,n*n
           print *, c(i)
        enddo
  
        ! Delete the memory
        deallocate(a)
        deallocate(b)
        deallocate(c)
  
        end program main
        ```



    === "Template(C/C++)"
        ```c
        #include<stdio.h>
        #include<stdlib.h>
        #include<omp.h>
        
        void Matrix_Multiplication(float *a, float *b, float *c, int width)   
        { 
          float sum = 0;
          for(int row = 0; row < width ; ++row)                           
            {                                                             
              for(int col = 0; col < width ; ++col)
                {
                  sum=0;
                  for(int i = 0; i < width ; ++i)                         
                    {                                                     
                      sum += a[row*width+i] * b[i*width+col];      
                    }                                                     
                  c[row*width+col] = sum;                           
                }
            }   
        }

        int main()
         {  
           printf("Programme assumes that matrix size is N*N \n");
           printf("Please enter the N size number \n");
           int N =0;
           scanf("%d", &N);

           // Initialize the memory on the host
           float *a, *b, *c;       
    
           // Allocate host memory
           a = (float*)malloc(sizeof(float) * (N*N));
           b = (float*)malloc(sizeof(float) * (N*N));
           c = (float*)malloc(sizeof(float) * (N*N));
  
          // Initialize host arrays
          for(int i = 0; i < (N*N); i++)
             {
               a[i] = 1.0f;
               b[i] = 2.0f;
             }

           // Device fuction call 
           Matrix_Multiplication(a, b, c, N);
  
           // Verification
           for(int i = 0; i < N; i++)
              {
              for(int j = 0; j < N; j++)
                 {
          	  printf("%f ", c[j]);

          	}
              printf("\n");
              }
  
            // Deallocate host memory
            free(a); 
            free(b); 
            free(c);

           return 0;
        }
        ```

    === "Template(FORTRAN)"
        ```c
         module Matrix_Multiplication_Mod  
        implicit none 
        contains
         subroutine Matrix_Multiplication(a, b, c, width)
        use omp_lib
        ! Input vectors
        real(8), intent(in), dimension(:) :: a
        real(8), intent(in), dimension(:) :: b
        real(8), intent(out), dimension(:) :: c
        real(8) :: sum = 0
        integer :: i, row, col, width

        do row = 0, width-1
           do col = 0, width-1
              sum=0
               do i = 0, width-1
                 sum = sum + (a((row*width)+i+1) * b((i*width)+col+1))
               enddo
              c(row*width+col+1) = sum
           enddo
        enddo


          end subroutine Matrix_Multiplication
        end module Matrix_Multiplication_Mod

        program main
        use Matrix_Multiplication_Mod
        use omp_lib
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
        allocate(a(n*n))
        allocate(b(n*n))
        allocate(c(n*n))
  
        ! Initialize content of input vectors, 
        ! vector a[i] = sin(i)^2 vector b[i] = cos(i)^2
        do i = 1, n*n
           a(i) = sin(i*1D0) * sin(i*1D0)
           b(i) = cos(i*1D0) * cos(i*1D0) 
        enddo

        ! Call the vector add subroutine 
        call Matrix_Multiplication(a, b, c, n)
  
        !!Verification
        do i=1,n*n
           print *, c(i)
        enddo
  
        ! Delete the memory
        deallocate(a)
        deallocate(b)
        deallocate(c)
  
        end program main

        ```


    === "Solution(C/C++)"
        ```c
               #include<stdio.h>
        #include<stdlib.h>
        #include<omp.h>
        
        void Matrix_Multiplication(float *a, float *b, float *c, int width)   
        { 
          float sum = 0;
          for(int row = 0; row < width ; ++row)                           
            {                                                             
              for(int col = 0; col < width ; ++col)
                {
                  sum=0;
                  for(int i = 0; i < width ; ++i)                         
                    {                                                     
                      sum += a[row*width+i] * b[i*width+col];      
                    }                                                     
                  c[row*width+col] = sum;                           
                }
            }   
        }

        int main()
         {  
           printf("Programme assumes that matrix size is N*N \n");
           printf("Please enter the N size number \n");
           int N =0;
           scanf("%d", &N);

           // Initialize the memory on the host
           float *a, *b, *c;       
    
           // Allocate host memory
           a = (float*)malloc(sizeof(float) * (N*N));
           b = (float*)malloc(sizeof(float) * (N*N));
           c = (float*)malloc(sizeof(float) * (N*N));
  
          // Initialize host arrays
          for(int i = 0; i < (N*N); i++)
             {
               a[i] = 1.0f;
               b[i] = 2.0f;
             }

           // Device fuction call 
           Matrix_Multiplication(a, b, c, N);
  
           // Verification
           for(int i = 0; i < N; i++)
              {
              for(int j = 0; j < N; j++)
                 {
          	  printf("%f ", c[j]);

          	}
              printf("\n");
              }
  
            // Deallocate host memory
            free(a); 
            free(b); 
            free(c);

           return 0;
        }
        ```

    === "Solution(FORTRAN)"
        ```c
        module Matrix_Multiplication_Mod  
        implicit none 
        contains
         subroutine Matrix_Multiplication(a, b, c, width)
        use omp_lib
        ! Input vectors
        real(8), intent(in), dimension(:) :: a
        real(8), intent(in), dimension(:) :: b
        real(8), intent(out), dimension(:) :: c
        real(8) :: sum = 0
        integer :: i, row, col, width

        do row = 0, width-1
           do col = 0, width-1
              sum=0
               do i = 0, width-1
                 sum = sum + (a((row*width)+i+1) * b((i*width)+col+1))
               enddo
              c(row*width+col+1) = sum
           enddo
        enddo


          end subroutine Matrix_Multiplication
        end module Matrix_Multiplication_Mod

        program main
        use Matrix_Multiplication_Mod
        use omp_lib
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
        allocate(a(n*n))
        allocate(b(n*n))
        allocate(c(n*n))
  
        ! Initialize content of input vectors, 
        ! vector a[i] = sin(i)^2 vector b[i] = cos(i)^2
        do i = 1, n*n
           a(i) = sin(i*1D0) * sin(i*1D0)
           b(i) = cos(i*1D0) * cos(i*1D0) 
        enddo

        ! Call the vector add subroutine 
        call Matrix_Multiplication(a, b, c, n)
  
        !!Verification
        do i=1,n*n
           print *, c(i)
        enddo
  
        ! Delete the memory
        deallocate(a)
        deallocate(b)
        deallocate(c)
  
        end program main

        ```




??? "Compilation and Output"

    === "Serial-version"
        ```c
        // compilation
        $ gcc Matrix-multiplication.c -o Matrix-Multiplication-CPU
        
        // execution 
        $ ./Matrix-Multiplication-CPU
        
        // output
        $ g++ Matrix-multiplication.cc -o Matrix-multiplication
        $ ./Matrix-multiplication
        Programme assumes that matrix (square matrix) size is N*N 
        Please enter the N size number 
        4
        16 16 16 16 
        16 16 16 16  
        16 16 16 16  
        16 16 16 16 
        ```
        
    === "CUDA-version"
        ```c
        // compilation
        $ nvcc -arch=compute_70 Matrix-multiplication.cu -o Matrix-Multiplication-GPU
        
        // execution
        $ ./Matrix-Multiplication-GPU
        Programme assumes that matrix (square matrix) size is N*N 
        Please enter the N size number
        $ 256
        
        // output
        $ Two matrices are equal
        ```

??? Question "Questions"

    - Right now, we are using the 1D array to represent the matrix. However, you can also do it with the 2D matrix.
    Can you try with 2D array matrix multiplication with 2D thread block?
    - Can you get the correct soltion if you remove the **`if ((row < width) && (col < width))`**
    condition from the **`__global__ void matrix_mul(float* d_a, float* d_b, float* d_c, int width)`** function?
    - Please try with different thread blocks and different matrix sizes.
    ```
    // Thread organization
    int blockSize = 32;
    dim3 dimBlock(blockSize,blockSize,1);
    dim3 dimGrid(ceil(N/float(blockSize)),ceil(N/float(blockSize)),1);
    ```

