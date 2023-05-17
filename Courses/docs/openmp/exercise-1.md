Hello world

<figure markdown>
![](../figures/diagram-20230517-1.png){align=center width=500}
</figure>




### <u>Questions and Solutions</u>


??? Example "Examples: Hello World"

    === "Serial-version (C/C++)"
        ``` c
        #include<iostream>
        using namespace std;
        
        int main()
        {
          cout << endl;
          cout << "Hello world from master thread"<< endl;
          cout << endl;
          
          return 0;
        }
        ```

    === "Serial-version (FORTRAN)"
        ``` fortran
        program Hello_world_Serial
        
        print *, 'Hello world from master thread'
        
        end program
        ```
	
    === "OpenMP-version (C/C++)"
        ``` c
        #include<iostream>
        #include<omp.h>
        using namespace std;
        int main()
        {
          cout << "Hello world from master thread "<< endl;
          cout << endl;
          
          // set number of threads to be used here
          omp_set_num_threads(5);
          
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

    === "OpenMP-version (FORTRAN)"
        ``` fortran
        program Hello_world_OpenMP
        use omp_lib
        
        call omp_set_num_threads(4)
        
        !$omp parallel
        print *, 'Hello world from thread id ', omp_get_thread_num(), 'from the team size of', omp_get_num_threads()
        !$omp end parallel
        
        end program
        ```

??? "Compilation and Output"

    === "Serial-version (C/C++)"
        ```c
        // compilation
        $ g++ Hello-world-Serial.cc -o Hello-World-Serial
        
        // execution 
        $ ./Hello-World-Serial
        
        // output
        $ Hello world from master thread
        ```

    === "Serial-version (FORTRAN)"
        ```c
        // compilation
        $ gfortran Hello-world-Serial.f90 -o Hello-World-Serial
        
        // execution 
        $ ./Hello-World-Serial
        
        // output
        $ Hello world from master thread
        ```
        
    === "OpenMP-version (C/C++)"
        ```c
        // compilation
        $ g++ -fopenmp Hello-world-OpenMP.cc -o Hello-World-OpenMP
        
        // execution
        $ ./Hello-World-OpenMP
        
        // output
        $ 
        ```

    === "OpenMP-version (FORTRAN)"
        ```c
        // compilation
        $ gfortran -fopenmp Hello-world-OpenMP.f90 -o Hello-World-OpenMP
        
        // execution
        $ ./Hello-World-OpenMP
        
        // output
        $  Hello world from thread id            0 from the team size of           4
           Hello world from thread id            1 from the team size of           4
           Hello world from thread id            2 from the team size of           4
           Hello world from thread id            3 from the team size of           4
        ```




??? Questions


     - Right now, we are printing the `hello world` from the threads we set in the main program. However, in some situations, we can also set several threads during the compilation time. This can be done by using `export OMP_NUM_THREADS`.



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

    === "Question (FORTRAN)"
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
	
     - What happens if you do not set `omp_set_num_threads()`, for example, `omp_set_num_threads(5)|call omp_set_num_threads(5)`, what do you notice? 
