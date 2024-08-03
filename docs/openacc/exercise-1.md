#### <u>Compute Constructs</u>

In our first exercise, we will look into how to offload the computation to the device (GPU).
Because the main aim of the OpenACC is to facilitate offloading the computation using OpenACC APIs.
OpenACC provides two variants to offload the computations to GPU. They are explained as follows,

   - OpenACC provides two compute constructs to parallelize the computation
   - The first one is `parallel`, and the second is `kernels`
   - Both of these parallel constructs perform more or less the same
   - However, `kernels` will have more control over the parallel region
   - Therefore, as a programmer, if you are very familiar with what you are doing in the parallel region,
     you may use `parallel`; otherwise, it is better to use `kernels`
   - Because the compiler will take care of the safe parallelization under the `kernels` construct 

At the same time, in order to enable OpenACC constructs, clauses, and environment variables. etc., we need to include the OpenACC library as follows:

!!! Info "OpenACC library"

    === "C/C++"
        ```
        #include<openacc.h>
        ```
    === "FORTRAN"
        ```
        use openacc
        ```

<figure markdown>
![](../figures/acc-parallel.png){align=center width=500}
</figure>


To create a parallel region in OpenACC, we use the following compute constructs:

!!! Info "Parallel Constructs"

    === "C/C++"
        ```
        #pragma acc parallel [clause-list] new-line
           structured block
        ```
    === "FORTRAN"
        ```
        !$acc parallel [ clause-list ]
            structured block
        !$acc end parallel
        ```

??? "Available clauses for parallel"

    === "C/C++ and FORTRAN"
	```c
        async [ ( int-expr ) ]
        wait [ ( int-expr-list ) ]
        num_gangs( int-expr )
        num_workers( int-expr )
        vector_length( int-expr )
        device_type( device-type-list )
        if( condition )
        self [ ( condition ) ]
        reduction( operator : var-list )
        copy( var-list )
        copyin( [ readonly: ] var-list )
        copyout( [ zero: ] var-list )
        create( [ zero: ] var-list )
        no_create( var-list )
        present( var-list )
        deviceptr( var-list )
        attach( var-list )
        private( var-list )
        firstprivate( var-list )
        default( none | present )
	```



!!! Info "Kernels Constructs"

    === "C/C++"
        ```
        #pragma acc kernels [ clause-list ] new-line
           structured block
        ```
	
    === "FORTRAN"
        ```
        !$acc kernels [ clause-list ]
           structured block
        !$acc end kernels
        ```


??? "Available clauses for kernels"

    === "C/C++ and FORTRAN"
        ```c
        async [ ( int-expr ) ]
        wait [ ( int-expr-list ) ]
        num_gangs( int-expr )
        num_workers( int-expr )
        vector_length( int-expr )
        device_type( device-type-list )
        if( condition )
        self [ ( condition ) ]
        copy( var-list )
        copyin( [ readonly: ] var-list )
        copyout( [ zero: ] var-list )
        create( [ zero: ] var-list )
        no_create( var-list )
        present( var-list )
        deviceptr( var-list )
        attach( var-list )
        default( none | present )
        ```


#### <u>Compilers</u>

The following compilers would support the [OpenACC programming model](https://www.openacc.org/tools).

 - [GNU](https://gcc.gnu.org/) - It is an open source and can be used for Nvidia and AMD CPUs
 - [Nvidia HPC SDK](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html#gs.zd201n) - It is from Nvidia, and works very well for Nvidia GPUs
 - [HPE](https://buy.hpe.com/us/en/software/high-performance-computing-software/high-performance-computing-software/high-performance-computing-software/hpe-cray-programming-environment/p/1012707351) - Presently it supports the FORTRAN (not C/C++) 

!!! Info "Examples (GNU, Nvidia HPC SDK and HPE): Compilation"

    === "Nvidia HPC SDK"
        ```c
        $ nvc -fast -acc=gpu -gpu=cc80 -Minfo=accel test.c 
        ```

### <u>Questions and Solutions</u>


??? Example "Examples: Hello World"

    === "Serial-version"
        ```c
        //Hello-world-CPU.c	
        #include<stdio.h>
        int main()
        {
          printf("Hello World from CPU!\n");		
          return 0;
        }
        ```


    === "OpenACC-version-parallel"
        ```c
        //Hello-world-parallel.c	
        #include<stdio.h>
        #include<openacc.h>		
        int main()
        { 
        #pragma acc parallel                                                             
          printf("Hello World from GPU!\n");
          return 0;
        }
        ```


    === "OpenACC-version-kernels"
        ```c
        //Hello-world-kernels.c	
        #include<stdio.h>
        #include<openacc.h>		
        int main()
        {
        #pragma acc kernels                            
          printf("Hello World from GPU!\n");
          return 0;
        }
        ```


??? "Compilation and Output"

    === "Serial-version"
        ```c
        // compilation
        $ gcc Hello-world-CPU.c -o Hello-World-CPU
        
        // execution 
        $ ./Hello-World-CPU
        
        // output
        $ Hello World from CPU!
        ```
        
    === "OpenACC-version-parallel"
        ```c
        // compilation
        $ nvc -fast -acc=gpu -gpu=cc80 -Minfo=accel Hello-world-parallel.c -o Hello-World-GPU
        main:
        7, Generating NVIDIA GPU code
        
        // execution
        $ ./Hello-World-GPU
        
        // output
        $ Hello World from GPU!
        ```


    === "OpenACC-version-kernels"
        ```c
        // compilation
        $ nvc -fast -acc=gpu -gpu=cc80 -Minfo=accel Hello-world-kernels.c -o Hello-World-GPU
        main:
        7, Accelerator serial kernel generated
           Generating NVIDIA GPU code
         
        // execution
        $ ./Hello-World-GPU
        
        // output
        $ Hello World from GPU!
        ```


#### <u>Loop</u>

Our second exercise is to work on how to parallelize the loop. 
Most of the time, we would be doing the intense computation under the loop. In situations like that, it would be more efficient to parallelize the loops in the computation. 
To start with a simple example, we will begin with printing **`Hello World from GPU`** multiple times in addition to our previous example. 
Moreover, just adding  **`#pragma acc parallel`** or  **`#pragma acc kernels`** would not parallelize your computation; instead, it would ensure that the computation is executed on the device. 

    
<figure markdown>
![](../figures/acc-loop.png){align=center width=500}
</figure>
    
    
!!! Info "Loop Constructs"

    === "C/C++"
        ```
        #pragma acc loop [clause-list] new-line
           for loop
        ```
	
    === "FORTRAN"
        ```
        !$acc loop [clause-list]
           do loop
        ```
        

??? "Available clauses for loop"

    === "C/C++ and FORTRAN"
        ```c
        collapse( n )
        gang [( gang-arg-list )]
        worker [( [num:]int-expr )]
        vector [( [length:]int-expr )]
        seq
        independent
        auto
        tile( size-expr-list )
        device_type( device-type-list )
        private( var-list )
        reduction( operator:var-list )
        ```



### <u>Questions and Solutions</u>


??? Example "Examples: Loop (Hello World)"

    === "Serial-version-loop"
        ```c
        //Hello-world-CPU-loop.c	
        #include<stdio.h>
        int main()
        {
          for(int i = 0; i < 5; i++)
            {         
              printf("Hello World from CPU!\n");
            }		
          return 0;
        }
        ```


    === "OpenACC-version-parallel-loop"
        ```c
        //Hello-world-parallel-loop.c	
        #include<stdio.h>
        #include<openacc.h>		
        int main()
        {
        #pragma acc parallel loop
          for(int i = 0; i < 5; i++)
            {                                
              printf("Hello World from GPU!\n");
            }
        return 0;
        }
        ```


    === "OpenACC-version-kernels-loop"
        ```c
        //Hello-world-kernels-loop.c	
        #include<stdio.h>
        #include<openacc.h>		
        int main()
        {
        #pragma acc kernels loop
          for(int i = 0; i < 5; i++)
            {                                
              printf("Hello World from GPU!\n");
            }
        return 0;
        }
        ```


??? "Compilation and Output"

    === "Serial-version-loop"
        ```c
        // compilation
        $ gcc Hello-world-CPU-loop.c -o Hello-World-CPU
        
        // execution 
        $ ./Hello-World-CPU
        
        // output
        $ Hello World from CPU!
        $ Hello World from CPU!
        $ Hello World from CPU!
        $ Hello World from CPU!
        $ Hello World from CPU!                                
        ```
        
        
    === "OpenACC-version-parallel-loop"
        ```c
        // compilation
        $ nvc -fast -acc=gpu -gpu=cc80 -Minfo=accel Hello-world-parallel-loop.c -o Hello-World-GPU
        main:
        5, Generating NVIDIA GPU code
          7, #pragma acc loop gang /* blockIdx.x */
        
        // execution
        $ ./Hello-World-GPU
        
        // output
        $ Hello World from GPU!
        $ Hello World from GPU!
        $ Hello World from GPU!
        $ Hello World from GPU!
        $ Hello World from GPU!                                
        ```


    === "OpenACC-version-kernels-loop"
        ```c
        // compilation
        $ nvc -fast -acc=gpu -gpu=cc80 -Minfo=accel Hello-world-kernels-loop.c -o Hello-World-GPU
        main:
        7, Loop is parallelizable
           Generating NVIDIA GPU code
            7, #pragma acc loop gang, vector(32) /* blockIdx.x threadIdx.x */
        
        // execution
        $ ./Hello-World-GPU
        
        // output
        $ Hello World from GPU!
        $ Hello World from GPU!
        $ Hello World from GPU!
        $ Hello World from GPU!
        $ Hello World from GPU!                                
        ```

