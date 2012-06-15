# makefile

CPPSRC=ham.cpp\
       hamsparse.cpp\
       main.cpp\

CUDASRC=hamgpu.cu

OBJ=$(CPPSRC:.cpp=.o) $(CUDASRC:.cu=.o)

EXE=main

CC=gcc
CXX=g++

CFLAGS=-g -Wall -O2 -march=native
CPPFLAGS=$(CFLAGS)
LDFLAGS=-g -O2 -Wall -march=native
NVFLAGS=-g -O2 --ptxas-options=-v -arch=sm_13

INCLUDE=-I/opt/cuda/include
LIBS=-lblas -llapack -lcudart -lcublas

%.o:    %.c
	$(CC) -c $(CFLAGS) $(INCLUDE) $(@:.o=.c) -o $@

%.o:    %.cpp
	$(CXX) -c $(CPPFLAGS) $(INCLUDE) $(@:.o=.cpp) -o $@

%.o:    %.cu
	nvcc -c $(NVFLAGS) $(INCLUDE) $(@:.o=.cu) -o $@
#	nvcc -cuda $(NVFLAGS) $(INCLUDE) $(@:.o=.cu) -o $(@:.o=.cu.ii)
#	$(CXX) -c $(CPPFLAGS) $(INCLUDE) $(@:.o=.cu.ii) -o $@


all: $(OBJ)
	$(CXX) $(LDFLAGS) -o $(EXE) $(OBJ) $(LIBS)

main2: main2.o
	$(CXX) $(LDFLAGS) -o main2 main2.o $(LIBS)

doc: $(CPPSRC) doc-config
	doxygen doc-config

.PHONY: clean
clean:
	rm -f $(OBJ) main2.o
