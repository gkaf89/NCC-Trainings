# Environment management and containers for HPC systems

Environment management in shared computing resources like HPC systems and workstations involves a collection of techniques that allow users to install, configure and run the software they require on top of the base system. These techniques have expended over time to serve a number of needs such as

- creating environments that are isolated from the base system and other environments to manage software with conflicting requirements,
- source software from a variety of sources, and
- create consistent and reproducible environments.

The ability of environment management techniques to create reproducible environments in particular are particularly important in efforts to produce reproducible research results. Workflows for simulation and analysis of results can be encapsulated in a reproducible environment and distributed together with the environment.


## Overview of the course

The training is an overview of the technologies underlying environments and containers accompanied by demonstrations and examples.


## Terms

_Environments_ modify the user space to provide access to different sets of executables and libraries without modifying globally installed tools. Environment management tools such as Environment Modules, Conda, and application specific tools such Python Virtual Environments (`venv`) rely on environment variables such as `PATH` and `LD_LIBRARY_PATH`.

_Containers_ on top of user space modifications can also modify aspects of the operating system. Singularity containers for instance, can use a custom GNU C Library (`glibc`), a wrapper around the system calls of the Linux kernel. Containers rely on a variety of tools to control the environment, access to files, and access to resources such as cores and memory.

Anything that requires customization of the kernel and drivers must use _virtualisation_, which is beyond the scope of most HPC applications.
