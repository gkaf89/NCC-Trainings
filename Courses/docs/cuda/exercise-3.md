<figure markdown>
![](/figures/mat.png){align=center width=500}
<figcaption>b</figcaption>
</figure>


 - Allocating the CPU memory for A, B, and C matrix
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

 - Now we need to fill the values for the matrix a and b.
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
// Device fuction call
matrix_mul<<<dimGrid,dimBlock>>>(d_a, d_b, d_c, N);
```
 -  ??? "matrix multiplication fuction call"

        === "serial"
            ```c
            float * matrix_mul(float *h_a, float *h_b, float *h_c, int width)
            {
              for(int row = 0; row < width ; ++row)
                {
                  for(int col = 0; col < width ; ++col)
                    {
                      float single_entry = 0;
                      for(int i = 0; i < width ; ++i)
                        {
                          single_entry += h_a[row*width+i] * h_b[i*width+col];
                        }
                      h_c[row*width+col] = single_entry;
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
                  float single_entry = 0;
                  // each thread computes one 
                  // element of the block sub-matrix
                  for (int i = 0; i < width; ++i) 
                    {
                      single_entry += d_a[row*width+i]*d_b[i*width+col];
                    }
                  d_c[row*width+col] = single_entry;
                }
            }
            ```

 - Transfer back (copy back to ) the computed c matrix to host from GPU
```c
// Transfer data back to host memory
cudaMemcpy(c, d_c, sizeof(float) * (N*N), cudaMemcpyDeviceToHost);
```
 - Copy back computed value from GPU to CPU
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


### Solutions and Questions


??? example "Examples: Matrix Multiplication" 


    === "Serial-version"
        ```c
        //-*-C++-*-
        #include<iostream>
        
        using namespace std;
        
        float * matrix_mul(float *h_a, float *h_b, float *h_c, int width)   
        {                                                                 
          for(int row = 0; row < width ; ++row)                           
            {                                                             
              for(int col = 0; col < width ; ++col)                       
                {                                                         
                  float single_entry = 0;                                       
                  for(int i = 0; i < width ; ++i)                         
                    {                                                     
                      single_entry += h_a[row*width+i] * h_b[i*width+col];      
                    }                                                     
                  h_c[row*width+col] = single_entry;                            
                }                                                         
            }   
          return h_c;           
        }

        int main()
        {
  
          cout << "Programme assumes that matrix size is N*N "<<endl;
          cout << "Please enter the N size number "<< endl;
          int N = 0;
          cin >> N;

          // Initialize the memory on the host
          float *a, *b, *c;       
    
          // Allocate host memory
          a   = (float*)malloc(sizeof(float) * (N*N));
          b   = (float*)malloc(sizeof(float) * (N*N));
          c   = (float*)malloc(sizeof(float) * (N*N));
  
          // Initialize host matrix
          for(int i = 0; i < (N*N); i++)
            {
              a[i] = 1.0f;
              b[i] = 2.0f;
            }
   
          // Device fuction call 
          matrix_mul(a, b, c, N);

          // Verification
          for(int i = 0; i < N; i++)
            {
              for(int j = 0; j < N; j++)
                 {
                  cout << c[j] <<" ";
                 }
              cout << " " <<endl;
           }
    
          // Deallocate host memory
         free(a); 
         free(b); 
         free(c);

         return 0;
        }
        ```

    === "CUDA-version"

        ```c
        //-*-C++-*-
        #include<iostream>
	
        using namespace std;
	
        __global__ void matrix_mul(float* d_a, float* d_b, 
        float* d_c, int width)
        {
  
          int row = blockIdx.x * blockDim.x + threadIdx.x;
          int col = blockIdx.y * blockDim.y + threadIdx.y;
    
          if ((row < width) && (col < width)) 
            {
              float single_entry = 0;
              // each thread computes one 
              // element of the block sub-matrix
              for (int i = 0; i < width; ++i) 
                {
                  single_entry += d_a[row*width+i]*d_b[i*width+col];
                }
              d_c[row*width+col] = single_entry;
            }
        }

        int main()
        {
  
          cout << "Programme assumes that matrix size is N*N "<<endl;
          cout << "Please enter the N size number "<< endl;
          int N = 0;
          cin >> N;

          // Initialize the memory on the host
          float *a, *b, *c;       
  
          // Initialize the memory on the device
          float *d_a, *d_b, *d_c; 
  
          // Allocate host memory
          a   = (float*)malloc(sizeof(float) * (N*N));
          b   = (float*)malloc(sizeof(float) * (N*N));
          c   = (float*)malloc(sizeof(float) * (N*N));
  
          // Initialize host matrix
          for(int i = 0; i < (N*N); i++)
            {
              a[i] = 2.0f;
              b[i] = 2.0f;
            }
  
          // Allocate device memory
          cudaMalloc((void**)&d_a, sizeof(float) * (N*N));
          cudaMalloc((void**)&d_b, sizeof(float) * (N*N));
          cudaMalloc((void**)&d_c, sizeof(float) * (N*N));
  
          // Transfer data from host to device memory
          cudaMemcpy(d_a, a, sizeof(float) * (N*N), cudaMemcpyHostToDevice);
          cudaMemcpy(d_b, b, sizeof(float) * (N*N), cudaMemcpyHostToDevice);
  
          // Thread organization
          int blockSize = 32;
          dim3 dimBlock(blockSize,blockSize,1);
          dim3 dimGrid(ceil(N/float(blockSize)),ceil(N/float(blockSize)),1);
  
          // Device fuction call 
          matrix_mul<<<dimGrid,dimBlock>>>(d_a, d_b, d_c, N);

          // Transfer data back to host memory
          cudaMemcpy(c, d_c, sizeof(float) * (N*N), cudaMemcpyDeviceToHost);

          // Verification
          for(int i = 0; i < N; i++)
            {
              for(int j = 0; j < N; j++)
                {
                cout << c[j] <<" ";
                }
              cout << " " <<endl;
            }
  
          // Deallocate device memory
          cudaFree(d_a);
          cudaFree(d_b);
          cudaFree(d_c);
  
          // Deallocate host memory
          free(a); 
          free(b); 
          free(c);

          return 0;
        }
        ```

??? "Compilation and Output"

    === "Serial-version"
        ```
        // compilation
        $ gcc Matrix-multiplication.c -o Matrix-Multiplication-CPU
        
        // execution 
        $ ./Matrix-Multiplication-CPU
        
        // output
        $ Hello World from CPU!
        ```
        
    === "CUDA-version"
        ```c
        // compilation
        $ nvcc -arch=compute_70 Matrix-multiplication.cu -o Matrix-Multiplication-GPU
        
        // execution
        $ ./Matrix-Multiplication-GPU
        
        // output
        $ Hello World from GPU!


??? Question "Questions"

    - What happens if you remove the **`__syncthreads();`** from the **`__global__ void vector_add(float *a, float *b, 
       float *out, int n)`** function.
    - Can you remove the if condition **`if(i < n)`** from the **`__global__ void vector_add(float *a, float *b,
       float *out, int n)`** function. If so how can you do that?
    - Here we do not use the **`cudaDeviceSynchronize()`** in the main application, can you figure out why we
        do not need to use it. 
    - Can you create a different kinds of threads block for larger number of array?
