BENCHMARK=matmul

CXX      =mcxx --ompss-2
CXXSRC   =matmul-ompss-2
CXXFLAGS =-O3 -lnuma -lrt
CXXDEFS +=-DMATMUL_SIZE=1024

# option to enable cluster version
ifdef version
	ifeq "$(version)" "omp"
	CXX   =g++ -fopenmp
	CXXSRC=matmul-omp
	endif
endif
ifdef version
	ifeq "$(version)" "ompss-2-cluster"
	CXXSRC=matmul-ompss-2-cluster
	endif
endif

all         : ${BENCHMARK}

${BENCHMARK}: ${CXXSRC}.cpp
	$(CXX) $(CXXFLAGS) $(CXXDEFS) ${CXXSRC}.cpp -o ${CXXSRC}

clean       :
	rm -f ${CXXSRC} *.o
