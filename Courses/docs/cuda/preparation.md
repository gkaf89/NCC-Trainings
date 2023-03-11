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
  
### 4. And please create your own working folder under the project directory.
- 4.1 For example, here it is user with `u100490`:
  ```
  [u100490@login02 p200117]$ mkdir $USER
  ### or 
  [u100490@login02 p200117]$ mkdir u100490  
  ```