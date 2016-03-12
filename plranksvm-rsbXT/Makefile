CXX = icpc
CC = icc
RSBPATH = /home/jin/multicore/librsb/
CFLAGS = -Wall -Wconversion -O3 -fPIC -std=c++11 -fopenmp -lrt -g `$(RSBPATH)librsb-config --I_opts`
LIBS = blas/blas.a `$(RSBPATH)librsb-config --static --ldflags --extra_libs`
#LIBS = -lblas

# Location of the CUDA Toolkit binaries and libraries
COM_PATH       ?= /opt/composerxe
COM_PATH_INC       ?= $(COM_PATH)/include

# Common includes and paths for MKL
INCLUDES      := -I$(COM_PATH_INC) -I. -I.. -I../../common/inc -I$(COM_PATH)/mkl/include -I$(COM_PATH)/tbb/include

all: train predict


train: tron.o selectiontree.o linear.o train.c blas/blas.a
	$(CXX) $(CFLAGS) $(INCLUDES) -o train train.c tron.o selectiontree.o linear.o $(LIBS) 
predict: tron.o selectiontree.o linear.o predict.c blas/blas.a
	$(CXX) $(CFLAGS) $(INCLUDES) -o predict predict.c tron.o selectiontree.o linear.o $(LIBS)
tron.o: tron.cpp tron.h
	$(CXX) $(CFLAGS) -c -o tron.o tron.cpp

selectiontree.o: selectiontree.cpp selectiontree.h
	$(CXX) $(CFLAGS) $(INCLUDES) -c -o selectiontree.o selectiontree.cpp

linear.o: linear.cpp linear.h
	$(CXX) $(CFLAGS) $(INCLUDES) -c -o linear.o linear.cpp

blas/blas.a: blas/*.c blas/*.h
	make -C blas OPTFLAGS='$(CFLAGS)' CC='$(CC)';

clean:
	make -C blas clean
	rm -f *~ selectiontree.o tron*.o linear*.o train predict
