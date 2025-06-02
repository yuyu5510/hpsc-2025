10_cavity.cpp        -> Rewrite the 2-D Navier-Stokes code from the Scientific Computing lecture in C++

10_cavity_openmp.cpp -> OpenMP Version

10_cavity_openacc.cpp -> OpenACC Version

10_cavity.cu         -> cuda Version


In 10_cavity.py, I outputted the values of p, u, and v for cases where n % 10 == 0.
I then used diff.py to determine the maximum error between these outputs and the outputs from the respective executable files.
All errors were within 1e-7, so I believe the results are correct.
The commands are written in the Makefile.
