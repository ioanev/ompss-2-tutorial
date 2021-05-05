.PHONY : default
default: all;

MATMUL_DIM=512

OMPFLAGS  =-fopenmp
MCXXFLAGS =--ompss-2
CXXFLAGS  =-O3 -lnuma -lrt
CXXDEFS   =-DMATMUL_SIZE=$(MATMUL_DIM)

TARGET_SER           = matmul-ser
TARGET_OMP           = matmul-omp
TARGET_OMPSS         = matmul-ompss-2
TARGET_OMPSS_CLUSTER = matmul-ompss-2-cluster

all             : ser omp ompss-2 ompss-2-cluster

ser             : $(TARGET_SER).cpp
	g++ $(CXXFLAGS) $(CXXDEFS) $(TARGET_SER).cpp -o $(TARGET_SER)

omp             : $(TARGET_OMP).cpp
	g++ $(OMPFLAGS) $(CXXFLAGS) $(CXXDEFS) $(TARGET_OMP).cpp -o $(TARGET_OMP)

ompss-2         : $(TARGET_OMPSS).cpp
	mcxx $(MCXXFLAGS) $(CXXFLAGS) $(CXXDEFS) $(TARGET_OMPSS).cpp -o $(TARGET_OMPSS)

ompss-2-cluster : $(TARGET_OMPSS_CLUSTER).cpp
	mcxx $(MCXXFLAGS) $(CXXFLAGS) $(CXXDEFS) $(TARGET_OMPSS_CLUSTER).cpp -o $(TARGET_OMPSS_CLUSTER)

clean           :
	rm -rf $(TARGET_SER) $(TARGET_OMP) $(TARGET_OMPSS) $(TARGET_OMPSS_CLUSTER) *.o
