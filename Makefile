

NVCC        = nvcc
NVCC_FLAGS  = -I/usr/local/cuda/include -gencode=arch=compute_75,code=\"sm_75\"
ifdef dbg
	NVCC_FLAGS  += -g -G
else
	NVCC_FLAGS  += -O3
endif

LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = macrospin
OBJ	        = main.o kernel.o macrospin_cpp.o

default: $(EXE)

main.o: main.cu helper_math.h header.h
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)

kernel.o: kernel.cu helper_math.h header.h
	$(NVCC) -c -o $@ kernel.cu $(NVCC_FLAGS)

macrospin_cpp.o: macrospin_gold.cpp header.h
	$(NVCC) -c -o $@ macrospin_gold.cpp $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS) $(NVCC_FLAGS)

clean:
	rm -rf *.o $(EXE)
