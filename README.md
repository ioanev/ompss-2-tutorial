# EPEEC OmpSs-2 & OmpSs-2@Cluster Tutorial

This is a small tutorial on how to convert OpenMP applications to run on the OmpSs-2 programming model. We also provide guidelines on how to enable programming in a much larger scale than a single SMP, by using the cluster version of the model, utilizing both the MPI and ArgoDSM communication backends.

## System requirements

For this tutorial, we assume that you already have OpenMP available on your local machine, as well as compiled and installed in a user local directory the [Mercurium compiler](https://github.com/bsc-pm/mcxx) and the Nano6 runtime library. Note that if you are interested in running applications solely on the shared memory level, installing the plain [nanos6](https://github.com/bsc-pm/nanos6) version of the runtime is sufficient. However, if you are interested in scaling the application to the cluster level, it is advised to use the [nanos6-cluster](https://github.com/bsc-pm/nanos6-cluster) version.

## The Simple Example

The example concerns a naive matrix multiplication implementation, which is a kernel operation that calculates the dot product of two matrices. It is meant as an introduction to OmpSs-2 and OmpSs-2@Cluster and its differences with non-task-based and non-distributed applications. It is written in C++.

### **OpenMP implementation**

The full source code of the implementation can be found [here](https://github.com/IoanAnev/ompss-tutorial/blob/master/code/matmul-omp.cpp).

**Allocation & Deallocation**

The example starts with the working arrays being allocated using the C++ `new` operator.

```cpp
/* Allocate the matrices using `new`      */
mat_a = new double[N][N]; // input  matrix
mat_b = new double[N][N]; // input  matrix
mat_c = new double[N][N]; // output matrix (optimized)
mat_r = new double[N][N]; // output matrix (reference)
```

At the end of program execution, the structures are deallocated using the C++ `delete` operator.
```cpp
/* Deallocate the matrices using `delete` */
delete[] mat_a;           // -//-
delete[] mat_b;           // -//-
delete[] mat_c;           // -//-
delete[] mat_r;           // -//-
```

**Initialization**

After allocation, the arrays are being initialized collectively using the OpenMP worksharing-loop construct `pragma omp for`, for a correct and balanced NUMA node data distribution. Operation on the arrays is being done in a blocked manner for cache-friendly accesses.

```cpp
void
init_matrices()
{
	/**
	 * @note: The initialization of the global matrices can
	 *        also be handled by the master or by any thread
	 */

	/* Parallelize for better NUMA node data distribution */
	#pragma omp for
	for (int i = 0; i < N; i += BSIZE)
		for (int j = 0; j < N; j += BSIZE)
			/* Init. a block for cache-friendly accesses  */
			init_block(i, j, BSIZE, mat_a, mat_b, mat_c, mat_r);
	/* _implicit barrier_ */
}
```

**Computation**

The multiplication is being done similar to the initialization, exploiting coarse-grained parallelism by parallelizing the outermost loop of the kernel, and using the method of tiling. Note that there is a barrier implied after each `pragma omp for` construct, thus there is no need for one to be exclusively included.

```cpp
void
matmul_opt()
{
	/* Parallel block-based computation */
	#pragma omp for
	for (int i = 0; i < N; i += BSIZE)
		for (int j = 0; j < N; j += BSIZE)
			for (int k = 0; k < N; k += BSIZE)
				/* Compute an individual block */
				multiply_block(i, j, k, BSIZE, mat_a, mat_b, mat_c);
	/* _implicit barrier_ */
}
```

**Verification**

The validity of the results is being checked by serially executing the kernel (no tiling), and then comparing the values of the output matrices of the two different kernels. Both the serial execution and the correctness test are handled by a single thread, chosen at runtime through `pragma omp single`. Similar to `pragma omp for`, this construct also includes an implicit barrier at the end of its body.

```cpp
/**
 * @note: the serial execution can be handled by the master or
 *        by any thread as in this case
 */

/* serial execution of matrix multiplication for verification      */
#pragma omp single
	matmul_ref();
/* _implicit barrier_ */

/**
 * @note: the verification can be handled by the master or
 *        by any thread as in this case
 */

/* verify the results by finding the error between mat_c and mat_r */
#pragma omp single
	verify_results();
/* _implicit barrier_ */
```
