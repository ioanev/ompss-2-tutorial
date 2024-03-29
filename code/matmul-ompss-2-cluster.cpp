/**
 * @file
 * @brief An OmpSs-2@Cluster implementation of matrix multiplication.
 */

#include <string>
#include <cstdlib>
#include <iostream>
#include <unistd.h>
#include <sys/time.h>

#include "memory.hpp"

/* task granularity */
#define BSIZE_UNIT 64
/* size of matrices */
#ifndef MATMUL_SIZE
#define MATMUL_SIZE 512
#endif
#define N MATMUL_SIZE

/* global variables */
int BSIZE;
int NSIZE;

/* working matrices */
double (*mat_a)[N];
double (*mat_b)[N];
double (*mat_c)[N];
double (*mat_r)[N];

void matmul_opt();
void init_matrices();

void info(const int&);
void usage(const char*);
void run_matmul(const int&);

/**
 * @note: Directives are outlined,
 *        and as such, all function
 *        invocations become a task
 * 
 * @warning: The parameter names
 *           need to be included
 * @warning: Cannot use the static
 *           variant N for depende-
 *           ncy specification, so
 *           use NSIZE (NSIZE == N)
 */
#pragma oss task		\
in(mat_c[0;NSIZE][0;NSIZE],	\
   mat_r[0;NSIZE][0;NSIZE])
void verify_results(
		double (*mat_c)[N],
		double (*mat_r)[N]);
/**
 * @note: Directives are outlined,
 *        and as such, all function
 *        invocations become a task
 * 
 * @warning: The parameter names
 *           need to be included
 * @warning: Cannot use the static
 *           variant N for depende-
 *           ncy specification, so
 *           use NSIZE (NSIZE == N)
 */
#pragma oss task		\
in(   mat_a[0;NSIZE][0;NSIZE], 	\
      mat_b[0;NSIZE][0;NSIZE]) 	\
inout(mat_r[0;NSIZE][0;NSIZE])
void matmul_ref(
		double (*mat_a)[N],
		double (*mat_b)[N],
		double (*mat_r)[N]);

void init_block(
		const int&,
		const int&,
		const int&,
		double (*)[N],
		double (*)[N],
		double (*)[N],
		double (*)[N]);

/**
 * @note: Directives are outlined,
 *        and as such, all function
 *        invocations become a task
 * 
 * @warning: The parameter names
 *           need to be included
 */
#pragma oss task		\
in(   mat_a[i;bsize][k;bsize],	\
      mat_b[k;bsize][j;bsize])	\
inout(mat_c[i;bsize][j;bsize])
void multiply_block(
		const int i,
		const int j,
		const int k,
		const int bsize,
		double (*mat_a)[N],
		double (*mat_b)[N],
		double (*mat_c)[N]);

double get_time();

int
main(int argc, char *argv[])
{
        int c;
	int verify{0};
        int errexit{0};
        extern char *optarg;
        extern int optind, optopt, opterr;

	/* block size unit */
	BSIZE = BSIZE_UNIT;
	/* matrix dimension */
	NSIZE = N;

        while ((c = getopt(argc, argv, "b:vh")) != -1) {
                switch (c) {
			case 'b':
				BSIZE = atoi(optarg);
				break;
			case 'v':
				verify = 1;
				break;
			case 'h':
				usage(argv[0]);
				exit(0);
				break;
			case ':':
				std::cerr << argv[0] << ": option -" << (char)optopt << " requires an operand"
					  << std::endl;
				errexit = 1;
				break;
			case '?':
				std::cerr << argv[0] << ": illegal option -- " << (char)optopt
					  << std::endl;
				errexit = 1;
				break;
			default:
				abort();
                }
        }

        if (errexit) {
        	usage(argv[0]);
                exit(errexit);
        } else {
		info(verify);
	}

	if (N % BSIZE) {
		std::cerr << "Error: The block size needs to "
			  << "divide the matrix dimensions."
			  << std::endl;
		exit(EXIT_FAILURE);
	}
 
 	/**
	 * Allocate the matrices using the nanos6 allocator calls
	 * and view the one-dimensional arrays as two-dimensional
	 * for ease of dependency specification and ease of access
	 * 
	 * @note: dmalloc is just a wrapper for nanos6_dmalloc,
	 *        which is used to allocate distributed memory
	 */
	mat_a = reinterpret_cast<double(*)[N]>(dmalloc<double>(N * N));
	mat_b = reinterpret_cast<double(*)[N]>(dmalloc<double>(N * N));
	mat_c = reinterpret_cast<double(*)[N]>(dmalloc<double>(N * N));
	mat_r = reinterpret_cast<double(*)[N]>(dmalloc<double>(N * N));

        init_matrices();
        run_matmul(verify);

	/**
	 * Deallocate the matrices using the nanos6 allocator calls
	 * 
	 * @note: dfree is just a wrapper for nanos6_dfree,
	 *        which is used to deallocate distributed memory
	 */
	dfree<double>(mat_a, N * N);
	dfree<double>(mat_b, N * N);
	dfree<double>(mat_c, N * N);
	dfree<double>(mat_r, N * N);

        return 0;
}

void
init_block(const int &i,
	   const int &j,
	   const int &bsize,
	   double (*mat_a)[N],
	   double (*mat_b)[N],
	   double (*mat_c)[N],
	   double (*mat_r)[N])
{
	for (int ii = i; ii < i+bsize; ii++) {
		for (int jj = j; jj < j+bsize; jj++) {
			mat_c[ii][jj] = 0.0;
			mat_r[ii][jj] = 0.0;
			mat_a[ii][jj] = ((ii + jj) & 0x0F) * 0x1P-4;
			mat_b[ii][jj] = (((ii << 1) + (jj >> 1)) & 0x0F) * 0x1P-4;
		}
	}
}

void
init_matrices()
{
	/**
	 * @note: The initialization of the global matrices can
	 *        also be handled by any thread
	 * 
	 * @warning: The distributed memory cannot be dereferenced
	 *           directly after being allocated, but only insi-
	 *           de sub-tasks
	 */

	/* Parallelize for better cluster node data distribution */
        for (int i = 0; i < N; i += BSIZE) {
		for (int j = 0; j < N; j += BSIZE) {
			/* Spawn a task for each individual block */
			#pragma oss task				\
					out(mat_a[i;BSIZE][j;BSIZE],	\
					    mat_b[i;BSIZE][j;BSIZE],	\
					    mat_c[i;BSIZE][j;BSIZE],	\
					    mat_r[i;BSIZE][j;BSIZE])	\
					firstprivate(i, j, BSIZE)
			init_block(i, j, BSIZE, mat_a, mat_b, mat_c, mat_r);
		}
        }

	/*
	 * We need to wait for the tasks to finish before moving to the
	 * computation, not for correctness reasons, but solely for co-
	 * nsistency in the performance measurements
	 */
	#pragma oss taskwait
}

void
multiply_block(const int i,
	       const int j,
	       const int k,
	       const int bsize,
	       double (*mat_a)[N],
	       double (*mat_b)[N],
	       double (*mat_c)[N])
{
		for (int ii = i; ii < i+bsize; ii++) {
			for (int jj = j; jj < j+bsize; jj++) {
				for (int kk = k; kk < k+bsize; kk++) {
					mat_c[ii][jj] += mat_a[ii][kk] * mat_b[kk][jj];
				}
			}
		}
}

void
matmul_opt()
{
	/* Parallel block-based computation */
	for (int i = 0; i < N; i += BSIZE) {
		for (int j = 0; j < N; j += BSIZE) {
			for (int k = 0; k < N; k += BSIZE) {
				/* Spawn a task for each individual block */
				multiply_block(i, j, k, BSIZE, mat_a, mat_b, mat_c);
			}
		}
	}

	/*
	 * We need to wait for the tasks to finish before deallocating
	 * the global data structures (in the case of verification=OFF),
	 * and for correct measurements
	 */
	#pragma oss taskwait
}

void
matmul_ref(double (*mat_a)[N],
	   double (*mat_b)[N],
	   double (*mat_r)[N])
{
	/**
	 * @note: The serial execution can be handled by any thread
	 *        as in this case
	 */

	/* Serial execution of matrix multiplication for verification */
        for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                        for (int k = 0; k < N; k++) {
                        	mat_r[i][j] += mat_a[i][k] * mat_b[k][j];
                        }
                }
        }
}

void
verify_results(double (*mat_c)[N],
	       double (*mat_r)[N])
{
	/**
	 * @note: The verification can be handled by any thread
	 *        as in this case
	 */

	double e_sum{0.0};

	/* Verify the results by finding the error between mat_c and mat_r */
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			e_sum += (mat_c[i][j] < mat_r[i][j])
				? mat_r[i][j] - mat_c[i][j]
				: mat_c[i][j] - mat_r[i][j];
		}
	}

	if (e_sum < 1E-6) {
		std::cout << "OK" << std::endl;
	} else {
		std::cerr << "MISMATCH" << std::endl;
		exit(EXIT_FAILURE);
	}
}

void
run_matmul(const int& verify)
{
        double time_start, time_stop;

        time_start = get_time();
        matmul_opt();
        time_stop = get_time();
        
        std::cout.precision(4);
        std::cout << "Time: " << time_stop - time_start
		  << std::endl;

	if (verify) {
		std::cout << "Verifying solution... ";
		
		time_start = get_time();
		/**
		 * @note: The distributed matrices need
		 *        to be included both in the de-
		 *        pendency list, as well as in
		 *        the function's parameter list
		 */
		matmul_ref(mat_a, mat_b, mat_r);
		/*
		 * We need to wait for the tasks to fini-
		 * sh for correct measurements
		 */
		#pragma oss taskwait
		time_stop = get_time();

		/**
		 * @note: The distributed matrices need
		 *        to be included both in the de-
		 *        pendency list, as well as in
		 *        the function's parameter list
		 * 
		 * @warning: The compiler and runtime
		 *           might be silent when dire-
		 *           ctly derefercing distribu-
		 *           ted memory for reading
		 *           However, in case this ha-
		 *           ppens, outdated values mi-
		 *           ght be observed, as the la-
		 *           test copy of the values mi-
		 *           ght reside in remote nodes
		 */
		verify_results(mat_c, mat_r);
		/*
		 * We need to wait for the tasks to fini-
		 * sh before deallocating the global data
		 */
		#pragma oss taskwait
		std::cout << "Reference runtime: " << time_stop - time_start
			  << std::endl;
	}
}

double
get_time()
{
        struct timeval tv;

        if (gettimeofday(&tv, NULL)) {
                std::cerr << "gettimeofday failed. Aborting."
			  << std::endl;
                abort();
        }
        
        return tv.tv_sec + tv.tv_usec * 1E-6;
}

void
usage(const char* argv0)
{
	std::cout << "Usage: " << argv0 << " [OPTION]...\n"
		  << "\n"
		  << "Options:\n"
		  << "\t-i\tSelect implementation <0:jik, 1:ijk, 2:ikj>\n"
		  << "\t-s\tSize of matrices <N>\n"
		  << "\t-v\tVerify solution\n"
		  << "\t-h\tDisplay usage"
		  << std::endl;
}

void
info(const int& verify)
{
	const std::string sverif = (verify == 0) ? "OFF" : "ON";

	std::cout << "MatMul: "         << N << "x" << N
		  << ", verification: " << sverif
		  << std::endl;
}
