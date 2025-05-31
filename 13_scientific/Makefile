python:
	python 10_cavity.py
cpp:
	g++ 10_cavity.cpp
	./a.out
	python diff.py
openmp:
	g++ -fopenmp 10_cavity_openmp.cpp
	OMP_NUM_THREADS=4 ./a.out
	python diff.py
openacc:
	nvc++ -acc 10_cavity_openacc.cpp
	./a.out
	python diff.py
cuda:
	nvcc 10_cavity.cu
	./a.out
	python diff.py

