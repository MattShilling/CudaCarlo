objects = cuda_carlo.o cuda_rig.o

all: $(objects)
	nvcc $(objects) -o CudaCarlo

%.o: %.cu
    # -x cu = Explicitly specify the language as CU.
    # -dc = Compile each .c, .cc, .cpp, .cxx, and .cu input file into an object file that contains relocatable device code.
	nvcc -x cu \
		-I. \
		-dc \
		-arch=sm_87 \
		-std=c++17 \
		$< -o $@

clean:
	rm -f *.o app
