# Environment management and containers for HPC systems

Environment management and systems containers emerged as a result of efforts to customize the user environment. The effort has 2 main targets.

- Customization of workflows: the user can swap between different sets of program and files without modifying the underlying system.
- Reproducibility of workflows: the same set of programs and files can be moved to a different system where it will behave in a similar manner; quite often the whole environment can be described in a simple text file which can be shared and used to reconstruct the environment from online resources.

_Environments_ modify the user space to provide access to different sets of executables and libraries without modifying globally installed tools. Environment management tools such as Environment Modules, Conda, and application specific tools such Python Virtual Environments (`venv`) rely on environment variables such as `PATH` and `LD_LIBRARY_PATH`.

_Containers_ on top of user space modifications can also modify aspects of the operating system. Singularity containers for instance, can use a custom GNU C Library (`glibc`), a wrapper around the system calls of the Linux kernel. Containers rely on a variety of tools to control the environment, access to files, and access to resources such as cores and memory.

Anything that requires customization of the kernel and drivers must use _virtualisation_, which is beyond the scope of most HPC applications.


## Overview of the course

The training is an overview of the technologies underlying environments and containers accompanied by demonstrations and examples.


