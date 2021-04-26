.PHONY : default
default: all;

MATMUL_DIM=512

MCXXFLAGS =--ompss-2
CXXFLAGS  =-O3 -lnuma -lrt
CXXDEFS   =-DMATMUL_SIZE=$(MATMUL_DIM)

TARGET_OMP           = matmul-omp
TARGET_OMPSS         = matmul-ompss-2
TARGET_OMPSS_CLUSTER = matmul-ompss-2-cluster

all             : omp ompss-2 ompss-2-cluster

omp             : $(TARGET_OMP).cpp
	g++ $(CXXFLAGS) $(CXXDEFS) $(TARGET_OMP).cpp -o $(TARGET_OMP)

ompss-2         : $(TARGET_OMPSS).cpp
	mcxx $(MCXXFLAGS) $(CXXFLAGS) $(CXXDEFS) $(TARGET_OMPSS).cpp -o $(TARGET_OMPSS)

ompss-2-cluster : $(TARGET_OMPSS_CLUSTER).cpp
	mcxx $(MCXXFLAGS) $(CXXFLAGS) $(CXXDEFS) $(TARGET_OMPSS_CLUSTER).cpp -o $(TARGET_OMPSS_CLUSTER)

clean           :
	rm -rf $(TARGET_OMP) $(TARGET_OMPSS) $(TARGET_OMPSS_CLUSTER) *.o
