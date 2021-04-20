/* An OmpSs-2 implementation of matrix multiplication.
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

/* task granularity */
int BSIZE;
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

void init_block(
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
	int verify{0};
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

	if (N % BSIZE) {
		std::cerr << "Error: The block size needs to "
			  << "divide the matrix dimensions."
			  << std::endl;
		exit(EXIT_FAILURE);
	}
 
	mat_a = reinterpret_cast<double(*)[N]>(new double[N * N]);
	mat_b = reinterpret_cast<double(*)[N]>(new double[N * N]);
	mat_c = reinterpret_cast<double(*)[N]>(new double[N * N]);
	mat_r = reinterpret_cast<double(*)[N]>(new double[N * N]);

        init_matrices();
        run_matmul(verify);

	delete[] mat_a;
	delete[] mat_b;
	delete[] mat_c;
	delete[] mat_r;

        return 0;
}

void
init_block( const int &i,
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
	int n{N};

        for (int i = 0; i < N; i += BSIZE) {
		for (int j = 0; j < N; j += BSIZE) {
			#pragma oss task				\
					out(mat_a[i;BSIZE][j;BSIZE],	\
					    mat_b[i;BSIZE][j;BSIZE],	\
					    mat_c[i;BSIZE][j;BSIZE],	\
					    mat_r[i;BSIZE][j;BSIZE])	\
					firstprivate(i, j, BSIZE)
			init_block(i, j, BSIZE, mat_a, mat_b, mat_c, mat_r);
		}
        }
}

void
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

void
matmul_opt()
{
	for (int i = 0; i < N; i += BSIZE) {
		for (int j = 0; j < N; j += BSIZE) {
			for (int k = 0; k < N; k += BSIZE) {
				#pragma oss task				\
						in(   mat_a[i;BSIZE][k;BSIZE],	\
						      mat_b[k;BSIZE][j;BSIZE])	\
						inout(mat_c[i;BSIZE][j;BSIZE])	\
						firstprivate(i, j, k, BSIZE)
				multiply_block(i, j, k, BSIZE, mat_a, mat_b, mat_c);
			}
		}
	}

	#pragma oss taskwait
}

void
matmul_ref()
{
        for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                        for (int k = 0; k < N; k++) {
                        	mat_r[i][j] += mat_a[i][k] * mat_b[k][j];
                        }
                }
        }
}

void
verify_results()
{
	double e_sum{0.0};

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
		matmul_ref();
		time_stop = get_time();

		verify_results();
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
