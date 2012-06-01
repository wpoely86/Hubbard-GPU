# makefile

CPPSRC=ham.cpp\
       hamsparse.cpp\
       main.cpp\

OBJ=$(CPPSRC:.cpp=.o)

EXE=main

CC=gcc
CXX=g++

CFLAGS=-g -Wall -O2 -march=native
CPPFLAGS=$(CFLAGS)
LDFLAGS=-g -O2 -Wall -march=native

INCLUDE=
LIBS=-lblas -llapack

%.o:    %.c
	$(CC) -c $(CFLAGS) $(INCLUDE) $(@:.o=.c) -o $@

%.o:    %.cpp
	$(CXX) -c $(CPPFLAGS) $(INCLUDE) $(@:.o=.cpp) -o $@

all: $(OBJ)
	$(CXX) $(LDFLAGS) -o $(EXE) $(OBJ) $(LIBS)

main2: main2.o
	$(CXX) $(LDFLAGS) -o main2 main2.o $(LIBS)

doc: $(CPPSRC) doc-config
	doxygen doc-config

.PHONY: clean
clean:
	rm -f $(OBJ) main2.o
