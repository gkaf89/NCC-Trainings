#### 1. [How to login to MeluXina machine](https://docs.lxp.lu/first-steps/quick_start/)
- 1.1 [Please take a look if you are using Windows](https://docs.lxp.lu/first-steps/connecting/)
- 1.2 [Please take a look if you are using Linux/Mac](https://docs.lxp.lu/first-steps/connecting/)

#### 2. Use your username to connect to MeluXina
- 2.1 For exmaple the below example shows the user of `u100490` 
  ```
  $ ssh u100490@login.lxp.lu -p 8822
  ```
#### 3. Once you have logged in
- 3.1 Once you have logged in, you will be in a default home directory 
  ```
  [u100490@login02 ~]$ pwd
  /home/users/u100490
  ```
- 3.2 After that go to project directory (Nvidia Bootcamp activites).
  ```
  [u100490@login02 ~]$ cd /project/home/p200117
  [u100490@login02 p200117]$ pwd
  /project/home/p200117
  ```
  
#### 4. And please create your own working folder under the project directory.
- 4.1 For example, here it is user with `u100490`:
  ```
  [u100490@login02 p200117]$ mkdir $USER
  ### or 
  [u100490@login02 p200117]$ mkdir u100490  
  ```

#### 5. Untill now you are in in the login node, now it is time to reserve the GPU compute node:
- 5.1 We will do the excercises in interactive mode
  ```
  $ salloc -A p200117 --res gpudev -q dev -N 1 -t 02:00:00
  ```
- ??? "check if your reservation is allocated"
      ```
      [u100490@login03 ~]$ salloc -A p200117 --res gpudev -q dev -N 1 -t 02:00:00
      salloc: Pending job allocation 296848
      salloc: job 296848 queued and waiting for resources
      salloc: job 296848 has been allocated resources
      salloc: Granted job allocation 296848
      salloc: Waiting for resource configuration
      salloc: Nodes mel2131 are ready for job
      ```

#### 6. Finally we need to load the compiler to test the GPU CUDA codes
 - 6.1 We need a Nvidia HPC SDK compiler for compiling and testing CUDA code
 ```
 $ module load OpenMPI/4.1.4-NVHPC-22.7-CUDA-11.7.0
 ```

- ??? "check if the module is loaded properly"
      ```
      [u100490@mel2131 ~]$ module load OpenMPI/4.1.4-NVHPC-22.7-CUDA-11.7.0
      [u100490@mel2131 ~]$ module list
 
      Currently Loaded Modules:
      1) env/release/2022.1           (S)   6) numactl/2.0.14-GCCcore-11.3.0  11) libpciaccess/0.16-GCCcore-11.3.0  16) GDRCopy/2.3-GCCcore-11.3.0                  21) knem/1.1.4.90-GCCcore-11.3.0
      2) lxp-tools/myquota/0.3.1      (S)   7) CUDA/11.7.0                    12) hwloc/2.7.1-GCCcore-11.3.0        17) UCX-CUDA/1.13.1-GCCcore-11.3.0-CUDA-11.7.0  22) OpenMPI/4.1.4-NVHPC-22.7-CUDA-11.7.0
      3) GCCcore/11.3.0                     8) NVHPC/22.7-CUDA-11.7.0         13) OpenSSL/1.1                       18) libfabric/1.15.1-GCCcore-11.3.0
      4) zlib/1.2.12-GCCcore-11.3.0         9) XZ/5.2.5-GCCcore-11.3.0        14) libevent/2.1.12-GCCcore-11.3.0    19) PMIx/4.2.2-GCCcore-11.3.0
      5) binutils/2.38-GCCcore-11.3.0      10) libxml2/2.9.13-GCCcore-11.3.0  15) UCX/1.13.1-GCCcore-11.3.0         20) xpmem/2.6.5-36-GCCcore-11.3.0

      Where:
          S:  Module is Sticky, requires --force to unload or purge
      ```