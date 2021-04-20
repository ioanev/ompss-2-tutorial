BENCHMARK=matmul

CXX      =mcxx --ompss-2
CXXFLAGS =-O3 -lnuma -lrt
CXXFLAGS+=-DMATMUL_SIZE=512

# option to enable cluster version
ifdef version
	ifeq "$(version)" "cluster"
	CXXFLAGS+=-DENABLE_CLUSTER
	endif
endif

all   : ${BENCHMARK}

matmul: ${BENCHMARK}.cpp
	$(CXX) $(CXXFLAGS) ${BENCHMARK}.cpp -o ${BENCHMARK}

clean :
	rm -f ${BENCHMARK} *.o
