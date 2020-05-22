objects = cuda_carlo.o cuda_rig.o

NVCC = /usr/local/apps/cuda/cuda-10.1/bin/nvcc 

all: $(objects)
	   $(NVCC) $(objects) -o CudaCarlo

%.o: %.cu
	    $(NVCC) -x cu -I. -I/usr/local/apps/cuda/cuda-10.1/samples/common/inc/ -dc $< -o $@ -std=c++11

clean:
	    rm -f *.o app
