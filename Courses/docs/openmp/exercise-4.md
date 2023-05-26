### <u>Loop scheduling</u>

   However, the above example is very simple.
   Because, in most cases, we would end up doing a large list of arrays with complex computations within the loop.
   Therefore, the work loading should be optimally distributed among the threads in those cases.
   To handle those considerations, OpenMP has provided the following loop-sharing clauses. They are:

    - Static
    - Dynamic
    - Guided
    - Auto 
    - Runtime

#### <u>Static</u>

 - The number of iterations are divided by chunksize. 
 - If the chunksize is not provided, a number of iterations will be divided by the size of the team of threads.
    - e.g., n=100, numthreads=5; each thread will execute the 20 iterations in parallel.
 - This is useful when the computational cost is similar to each iteration.

??? example "Examples: Loops"

    === "Serial(C/C++)"
        ```c
        #include <iostream>
        #include <omp.h>
        
        int main()
        {
         omp_set_num_threads(5);
         
        #pragma omp parallel for schedule(static)
        for(int i = 0; i < N; i++)
           {
            cout << " Thread id" << " " << omp_get_thread_num() << endl;    
           }  
          return 0;
        }
        ```

#### <u>Dynamic</u>

 - The number of iterations are divided by chunksize
 - If the chunksize is not provided, it will consider the default value as 1
 - This is useful when the computational cost is different in the iteration
 - This will quickly place the chunk of data in the queue


#### <u>Guided</u>

#### <u>Auto</u>

#### <u>Runtime</u>
