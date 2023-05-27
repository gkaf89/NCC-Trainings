Profiling is an important task to be considered when a computer code is written. From the programming and programmer’s perspective, we want to know where the code spends most of its time. Plenty of tools are available to profile a scientific code (computer code for doing arithmetic computing using processors). However, we will focus few of the widely used tools here. They are:

 - [AMD uProf](https://www.amd.com/content/dam/amd/en/documents/developer/uprof-v4.0-gaGA-user-guide.pdf)
 - [ARM Forge](https://developer.arm.com/documentation/101136/22-1-3/Performance-Reports?lang=en)
 - [Intel tools](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html)

####<u>[ARM Forge](https://developer.arm.com/documentation/101136/22-1-3/Performance-Reports?lang=en)</u>
Arm Forge [40] is another standard commercial tool for debugging [39], profiling [41],
and analyzing [42] scientific code on the massively parallel computer architecture.
They have a separate toolset for each category with the common environment:
DDT for debugging, MAP for profiling, and performance reports for analysis.
It also supports the MPI, UPC, CUDA, and OpenMP programming models for a different architecture with different variety of compilers.
DDT and MAP will launch the GUI, where we can interactively debug and profile the code.
Whereas perf-report will provide the analysis results in .html and .txt files.
The following command shows how to work on Arm Forge, and the blow figure shows the performance analysis of the hybrid programming (OpenMP+MPI).


####<u>[Intel tools](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html)</u>

#####<u>Intel Application Snapshot</u>
Intel Application Performance Snapchat tool helps to find essential performance factors and the metrics of CPU utilisation, memory access efficiency, and vectorisation.
`aps -help` will list out profiling metrics options in APS
     
<figure markdown>
![](../figures/APS_OpenMP_flow_chart.png){align=center}
<figcaption></figcaption>
</figure>

??? Info "APS"

    === "C/C++"
        ```c
        # compilation
        $ icc -qopenmp test.c
        
        # code execution
        $ aps --collection-mode=all -r report_output ./a.out
        $ aps-report -g report_output                        # create a .html file
        $ firefox report_output_<postfix>.html               # APS GUI in a browser
        $ aps-report report_output                           # command line output
        ```

    === "FORTRAN"
    	```c
        # compilation
        $ ifort -qopenmp test.f90
        
        # code execution
        $ aps --collection-mode=all -r report_output ./a.out
        $ aps-report -g report_output                        # create a .html file
        $ firefox report_output_<postfix>.html               # APS GUI in a browser
        $ aps-report report_output                           # command line output
        ```

#####<u>Intel Inspector</u>

Intel Inspector detects and locates the memory, deadlocks, and data races in the code.
For example, memory access and memory leaks can be found.

??? Info "Intel Inspector"

    === "C/C++"
        ```c
        # compile the code
	$ icc -qopenmp example.c
        # execute and profile the code
        $ inspxe-cl -collect mi1 -result-dir mi1 -- ./a.out
        $ cat inspxe-cl.txt
        # open the file to see if there is any memory leak
        === Start: [2020/12/12 01:19:59] ===
        0 new problem(s) found
        === End: [2020/12/12 01:20:25] ===
        ```

    === "FORTRAN"
    	```c
	# compile the code
	$ ifort -qopenmp test.f90
        # execute and profile the code
        $ inspxe-cl -collect mi1 -result-dir mi1 -- ./a.out
        $ cat inspxe-cl.txt
        # open the file to see if there is any memory leak
        === Start: [2023/05/10 01:19:59] ===
        0 new problem(s) found
        === End: [2020/05/10 01:20:25] ===
        ```


####<u>[AMD uProf](https://www.amd.com/content/dam/amd/en/documents/developer/uprof-v4.0-gaGA-user-guide.pdf)</u>


     - AMDuProfCLI collect --trace openmp --config tbp --output-dir solution ./a.out -d 1
