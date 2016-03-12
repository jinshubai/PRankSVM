CXX = icpc
CC = icc
CFLAGS = -Wall -Wconversion -O3 -fPIC -std=c++11 -openmp -lrt -g
LIBS = blas/blas.a 
#LIBS = -lblas

# Location of the MKL Toolkit binaries and libraries
COM_PATH       ?= /opt/composerxe
COM_PATH_INC       ?= $(COM_PATH)/include

# Common includes and paths for MKL
INCLUDES      := -I$(COM_PATH_INC) -I. -I.. -I../../common/inc -I$(COM_PATH)/mkl/include -I$(COM_PATH)/tbb/include

all: train predict


train: tron.o selectiontree.o linear.o train.c blas/blas.a
	$(CXX) $(CFLAGS) $(INCLUDES) -o train train.c tron.o selectiontree.o linear.o $(LIBS) -m64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm

predict: tron.o selectiontree.o linear.o predict.c blas/blas.a
	$(CXX) $(CFLAGS) $(INCLUDES) -o predict predict.c tron.o selectiontree.o linear.o $(LIBS) -m64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm

tron.o: tron.cpp tron.h
	$(CXX) $(CFLAGS) -c -o tron.o tron.cpp

selectiontree.o: selectiontree.cpp selectiontree.h
	$(CXX) $(CFLAGS) $(INCLUDES) -c -o selectiontree.o selectiontree.cpp

linear.o: linear.cpp linear.h
	$(CXX) $(CFLAGS) $(INCLUDES) -c -o linear.o linear.cpp -m64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm

blas/blas.a: blas/*.c blas/*.h
	make -C blas OPTFLAGS='$(CFLAGS)' CC='$(CC)';

clean:
	make -C blas clean
	rm -f *~ selectiontree.o tron*.o linear*.o train predict
