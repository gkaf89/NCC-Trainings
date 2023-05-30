<figure markdown>
![](../figures/simd.png){align=center width=500}
<figcaption></figcaption>
</figure>






### <u>Questions and Solutions</u>


??? Questions

     - How can you identify the thread numbers within the parallel region?
     - What happens if you not set `omp_set_num_threads()`, for example, `omp_set_num_threads(5)|call omp_set_num_threads(5)`, what do you notice? 
     - Alternatively, you can also set a number of threads to be used in the application while the compilation `export OMP_NUM_THREADS`; what do you see?

    === "Question (C/C++)"

        ```c
        #include<iostream>
        #include<omp.h>
        using namespace std;
        int main()
        {
          cout << "Hello world from master thread "<< endl;
          cout << endl;
                    
          // creating the parallel region (with N number of threads)
          #pragma omp parallel
           {
                //cout << "Hello world from thread id "
                << " from the team size of "
                << endl;
            } // parallel region is closed
            
        cout << endl;
        cout << "end of the programme from master thread" << endl;
        return 0;
        }
        ```

    === "Question (FORTRAN)"
        ``` fortran
        program Hello_world_OpenMP
        use omp_lib
                
        !$omp parallel 
        !! print *, 
        !$omp end parallel
        
        end program
        ```
	
    === "Answer (C/C++)"
        ```c
        #include<iostream>
        #include<omp.h>
        using namespace std;
        int main()
        {
          cout << "Hello world from master thread "<< endl;
          cout << endl;
                    
          // creating the parallel region (with N number of threads)
          #pragma omp parallel
           {
                cout << "Hello world from thread id "
                << omp_get_thread_num() << " from the team size of "
                << omp_get_num_threads()
                << endl;
            } // parallel region is closed
            
        cout << endl;
        cout << "end of the programme from master thread" << endl;
        return 0;
        }
        ```

    === "Answer (FORTRAN)"
        ``` fortran
        program Hello_world_OpenMP
        use omp_lib
                
        !$omp parallel 
        print *, 'Hello world from thread id ', omp_get_thread_num(), 'from the team size of', omp_get_num_threads()
        !$omp end parallel
        
        end program
        ```
        
    === "Answer"
        ``` c
        $ export OMP_NUM_THREADS=10
        // or 
        $ setenv OMP_NUM_THREADS 4
        // or
        $ OMP NUM THREADS=4 ./omp code.exe
        ```

    === "Solution Output (C/C++)"
        ```c
        ead id Hello world from thread id Hello world from thread id 3 from the team size of 9 from the team size of 52 from the team size of  from the team size of 10
        0 from the team size of 10
        10
        10
        10
        7 from the team size of 10
        4 from the team size of 10
        8 from the team size of 10
        1 from the team size of 10
        6 from the team size of 10
        ```
        
    === "Solution Output (FORTRAN)"

        ```c
        Hello world from thread id            0 from the team size of          10
        Hello world from thread id            4 from the team size of          10
        Hello world from thread id            5 from the team size of          10
        Hello world from thread id            9 from the team size of          10
        Hello world from thread id            2 from the team size of          10
        Hello world from thread id            3 from the team size of          10
        Hello world from thread id            7 from the team size of          10
        Hello world from thread id            6 from the team size of          10
        Hello world from thread id            8 from the team size of          10
        Hello world from thread id            1 from the team size of          10
        ```


