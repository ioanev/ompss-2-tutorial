/* An OmpSs-2 and OmpSs-2@Cluster implementation of matrix multiplication.
 *
 * It receives as input the dimension N and constructs three NxN matrices
 * (+1 for verification). We can enable verification with the -v argument.
 *
 * We initialize the matrices with prefixed values which we can later
 * check to ensure correctness of the computations.
 */

#include <string>
#include <cstdlib>
#include <iostream>
#include <unistd.h>
#include <sys/time.h>

#include "memory.hpp"

/* task granularity */
static int BSIZE;
#define BSIZE_UNIT 64
/* size of matrices */
#define N MATMUL_SIZE

/* working matrices */
double (*mat_a)[N];
double (*mat_b)[N];
double (*mat_c)[N];
double (*mat_r)[N];

void matmul_opt();
void matmul_ref();
void init_matrices();
void verify_results();

void info(const int&);
void usage(const char*);
void run_matmul(const int&);

void init_i_section(
		const int&,
		const int&,
		const int&,
		double (*)[N],
		double (*)[N],
		double (*)[N],
		double (*)[N]);
void multiply_block(
		const int&,
		const int&,
		const int&,
		const int&,
		double (*)[N],
		double (*)[N],
		double (*)[N]);

double get_time();

int
main(int argc, char *argv[])
{
        int c;
	int verify;
        int errexit{0};
        extern char *optarg;
        extern int optind, optopt, opterr;

	// block size unit
	BSIZE = BSIZE_UNIT;

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
 
	mat_a = reinterpret_cast<double(*)[N]>(dmalloc<double>(N * N));
	mat_b = reinterpret_cast<double(*)[N]>(dmalloc<double>(N * N));
	mat_c = reinterpret_cast<double(*)[N]>(dmalloc<double>(N * N));
	mat_r = reinterpret_cast<double(*)[N]>(dmalloc<double>(N * N));

        init_matrices();
        run_matmul(verify);

	dfree<double>(mat_a, N * N);
	dfree<double>(mat_b, N * N);
	dfree<double>(mat_c, N * N);
	dfree<double>(mat_r, N * N);

        return 0;
}

static void
init_i_section( const int &i,
		const int &n,
		const int &bsize,
		double (*mat_a)[N],
		double (*mat_b)[N],
		double (*mat_c)[N],
		double (*mat_r)[N])
{
	for (int ii = i; ii < i+bsize; ii++) {
		for (int jj = 0; jj < n; jj++) {
			mat_c[ii][jj] = 0.0;
			mat_r[ii][jj] = 0.0;
			mat_a[ii][jj] = ((ii + jj) & 0x0F) * 0x1P-4;
			mat_b[ii][jj] = (((ii << 1) + (jj >> 1)) & 0x0F) * 0x1P-4;
		}
	}
}

static void
init_matrices()
{
	int n{N};

        for (int i = 0; i < n; i += BSIZE) {
		#pragma oss task				\
				out(mat_a[i;BSIZE][0;n],	\
				    mat_b[i;BSIZE][0;n],	\
				    mat_c[i;BSIZE][0;n],	\
				    mat_r[i;BSIZE][0;n])	\
				    firstprivate(i, n, BSIZE)
		init_i_section(i, n, BSIZE, mat_a, mat_b, mat_c, mat_r);
        }
}

static void
multiply_block( const int &i,
		const int &j,
		const int &k,
		const int &bsize,
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

static void
matmul_opt()
{
	int n{N};

	for (int i = 0; i < n; i += BSIZE) {
		for (int j = 0; j < n; j += BSIZE) {
			for (int k = 0; k < n; k += BSIZE) {
				#pragma oss task				\
						in(   mat_a[i;BSIZE][k;BSIZE],	\
						      mat_b[k;BSIZE][j;BSIZE])	\
						inout(mat_c[i;BSIZE][j;BSIZE])	\
						firstprivate(i, j, k, BSIZE)
				multiply_block(i, j, k, BSIZE, mat_a, mat_b, mat_c);
			}
		}
	}
}

static void
matmul_ref()
{
	int n{N};

	#pragma oss task			\
			in( mat_a[0;n][0;n],	\
			    mat_b[0;n][0;n])	\
			out(mat_r[0;n][0;n])	\
			firstprivate(n)
        for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                        for (int k = 0; k < n; k++) {
                        	mat_r[i][j] += mat_a[i][k] * mat_b[k][j];
                        }
                }
        }
}

static void
verify_results()
{
	int n{N};

	#pragma oss task			\
			in( mat_c[0;n][0;n],	\
			    mat_r[0;n][0;n])	\
			firstprivate(n)
	{
		double e_sum = 0.0;

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				e_sum += (mat_c[i][j] < mat_r[i][j])
					? mat_r[i][j] - mat_c[i][j]
					: mat_c[i][j] - mat_r[i][j];
			}
		}

		/* wrong results */
		if (!(e_sum < 1E-6)) {
			std::cerr << "MISMATCH" << std::endl;
			exit(EXIT_FAILURE);
		}
	}
	#pragma oss taskwait

	/* we have correct results */
	std::cout << "OK" << std::endl;
}

static void
run_matmul(const int& verify)
{
        double time_start, time_stop;

        time_start = get_time();
        matmul_opt();
	#pragma oss taskwait
        time_stop = get_time();
        
        std::cout.precision(4);
        std::cout << "Time: " << time_stop - time_start
		  << std::endl;

	if (verify) {
		std::cout << "Verifying solution... ";
		
		time_start = get_time();
		matmul_ref();
		#pragma oss taskwait
		time_stop = get_time();

		verify_results();
		std::cout << "Reference runtime: " << time_stop - time_start
			  << std::endl;
	}
}

static double
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

static void
usage(const char *argv0)
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

static void
info(const int& verify)
{
	const std::string sverif = (verify == 0) ? "OFF" : "ON";

	std::cout << "MatMul: " << N << "x" << N
		  << ", verification: " << sverif
		  << std::endl;
}
