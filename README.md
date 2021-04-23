# EPEEC OmpSs-2 & OmpSs-2@Cluster Tutorial

This is a small tutorial on how to convert OpenMP applications to run on the OmpSs-2 programming model. We also provide guidelines on how to enable programming in a much larger scale than a single SMP, by using the cluster version of the model, utilizing both the MPI and ArgoDSM communication backends.

## System requirements

For this tutorial, we assume that you already have OpenMP available on your local machine, as well as compiled and installed in a user local directory the [Mercurium compiler](https://github.com/bsc-pm/mcxx) and the Nano6 runtime library. Note that if you are interested in running applications solely on the shared memory level, installing the plain [nanos6](https://github.com/bsc-pm/nanos6) version of the runtime is sufficient. However, if you are interested in scaling the application to the cluster level, it is advised to use the [nanos6-cluster](https://github.com/bsc-pm/nanos6-cluster) version.

## The Simple Example

The example concerns a naive matrix multiplication implementation, which is a kernel operation that calculates the dot product of two matrices. It is meant as an introduction to OmpSs-2 and OmpSs-2@Cluster and its differences with non-task-based and non-distributed applications. It is written in C++.

### **OpenMP implementation**
-----------------------------

The full source code of the implementation can be found [here](https://github.com/IoanAnev/ompss-tutorial/blob/master/code/matmul-omp.cpp).

**Allocation & Deallocation**

The example starts with the working arrays being allocated using the C++ `new` operator.

```cpp
/* Allocate the matrices using C++ `new`  */
mat_a = new double[N][N]; // input  matrix
mat_b = new double[N][N]; // input  matrix
mat_c = new double[N][N]; // output matrix (optimized)
mat_r = new double[N][N]; // output matrix (reference)
```

At the end of program execution, the structures are deallocated using the `delete` operator.
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

The validity of the results is being checked by serially executing the kernel (no tiling), and then comparing the values of the output matrices from the two different kernels. Both the serial execution and the correctness test are handled by a single thread chosen at runtime through `pragma omp single`. Similar to `pragma omp for`, this construct also includes an implicit barrier at the end of its body.

```cpp
/**
 * @note: The serial execution can be handled by the master or
 *        by any thread as in this case
 */

/* Serial execution of matrix multiplication for verification      */
#pragma omp single
	matmul_ref();
/* _implicit barrier_ */

/**
 * @note: The verification can be handled by the master or
 *        by any thread as in this case
 */

/* Verify the results by finding the error between mat_c and mat_r */
#pragma omp single
	verify_results();
/* _implicit barrier_ */
```

### **OmpSs-2 implementation**
-----------------------------

The full source code of the implementation can be found [here](https://github.com/IoanAnev/ompss-tutorial/blob/master/code/matmul-ompss-2.cpp).

**Allocation & Deallocation**

Since operation is still being done on the shared memory level, there is no difference in the allocation and deallocation mechanisms between the OpenMP and OmpSs-2 implementations.

**Initialization**

OmpSs being a task-based runtime system, the way we distribute work differs from the OpenMP method of parallelization used in the current example. In this example, we choose to associate a task to each tile to be computed. That said, the main thread iterates over the loops and keeps spawning tasks, which then reside on the task pool, ready to be served by a team of threads. Which thread computes which tile is chosen by the runtime.

The tasks need to also include directionality annotations based on the type of accesses being done to the data structures operated in the task body. As the matrices are only written during initialization, we enclose them in the `out` clause. Aside from expressing data direction, we need to specify which section of data is going to be operated by each task, in order to build proper dependencies between tasks, and thus effectively avoid data races.

In OmpSs, directives can be _inlined_ or _outlined_. When inlined, the `pragma` applies immediatly to the following statement, and the compiler outlines that statement as in OpenMP. The code block below presents an example of inlined directives.

```cpp
void
init_matrices()
{
	/**
	 * @note: The initialization of the global matrices can
	 *        also be handled by the master or by any thread
	 */

	/* Parallelize for better NUMA node data distribution */
	for (int i = 0; i < N; i += BSIZE)
		for (int j = 0; j < N; j += BSIZE)
			/* Spawn a task for each individual block     */
			#pragma oss task	             		\
					out(mat_a[i;BSIZE][j;BSIZE],	\
					    mat_b[i;BSIZE][j;BSIZE],	\
					    mat_c[i;BSIZE][j;BSIZE],	\
					    mat_r[i;BSIZE][j;BSIZE])	\
					firstprivate(i, j, BSIZE)
			init_block(i, j, BSIZE, mat_a, mat_b, mat_c, mat_r);
}
```

> **NOTE:**
> The plain OmpSs version is very relaxed in the way its dependencies are specified, in contrast to the cluster version. It allows the use of sentinels as representatives of a larger section of the data structures. However, despite this amenity that it provides, it is good practice to provide as much information as possible to the runtime regarding the data access pattern, for portability and compatibility reasons.

**Computation**

The execution of the matrix multiplication kernel is similar to the initialization, in the way of how the tasks are dispatched. However, there are differences between the two code sections. The first difference that hits the eye is that the `pragma` directive is not _inlined_ to the function invocation of `multiply_block`, but attached to its declaration instead. This is an example of _outlined_ directives. Outlining the directive for a function means that this function effectively becomes a task upon its invocation. When outlining directives, additional information needs to be included in the function declaration for the dependencies to be computed by the runtime.

```cpp
/**
 * @note   : Directives are outlined, and as such,
 *           all function invocations become a task
 * @warning: The parameter names need to be included
 */
#pragma oss task            	\
in(   mat_a[i;bsize][k;bsize],	\
      mat_b[k;bsize][j;bsize])	\
inout(mat_c[i;bsize][j;bsize])
void
multiply_block(
		const int i, const int j, const int k, const int bsize,
		double (*mat_a)[N], double (*mat_b)[N], double (*mat_c)[N]);
```

Another difference, is that the `pragma` directive now includes two different dependency clauses, namely `in`, and `inout`. The former, encloses the two input matrices as they are only read during computation, while the latter, encloses the output matrix, of which parts are accumulated with the execution of each task. These two, in conjuction with the `out` dependency clause, make up for all the basic strong dependency clauses.

As there is no implicit mechanism in the OmpSs programming model for imposing a "barrier" whenever necessary, the `taskwait` construct needs to be explicitly issued. This construct, blocks the current control flow of the program, until the completion of all the direct descendant tasks. Notice that a `taskwait` is not included between the initialization and computation, due to the implicit synchronization imposed from the task dependencies.

```cpp
void
matmul_opt()
{
	/* Parallel block-based computation */
	for (int i = 0; i < N; i += BSIZE)
		for (int j = 0; j < N; j += BSIZE)
			for (int k = 0; k < N; k += BSIZE)
				/* Spawn a task for each individual block */
				multiply_block(i, j, k, BSIZE, mat_a, mat_b, mat_c);

	/*
	 * We need to wait for the tasks to finish before deallocating
	 * the global data structures (in the case of verification=OFF),
	 * and for correct measurements
	 */
	#pragma oss taskwait
}
```

**Verification**

The verification of the results is left to be taken care of by the main thread, as we are guaranteed to see the updated values of the memory being worked on during the computation, by directly dereferencing. Of course, verification can also be handled within tasks.
