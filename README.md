# EPEEC OmpSs-2 & OmpSs-2@Cluster Tutorial

<div align="justify">

This is a small tutorial on how to convert OpenMP applications to run on the OmpSs-2 programming model. We also provide guidelines on how to enable programming in a much larger scale than a single SMP, by using the cluster version of the model, utilizing both the MPI and ArgoDSM communication backends.

</div>

## System requirements

<div align="justify">

For this tutorial, we assume that you already have OpenMP available on your local machine, as well as compiled and installed in a user local directory the [Mercurium compiler](https://github.com/bsc-pm/mcxx) and the Nano6 runtime library. Note that if you are interested in running applications solely on the shared memory level, installing the plain [nanos6](https://github.com/bsc-pm/nanos6) version of the runtime is sufficient. However, if you are interested in scaling the application to the cluster level, it is advised to use the [nanos6-cluster](https://github.com/bsc-pm/nanos6-cluster) version.

</div>

## The Simple Example

<div align="justify">

The example concerns a naive matrix multiplication implementation, which is a kernel operation that calculates the dot product of two matrices. It is meant as an introduction to OmpSs-2 and OmpSs-2@Cluster and its differences with non-task-based and non-distributed applications. It is written in C++.

</div>

### **OpenMP implementation**
-----------------------------

The full source code of the implementation can be found [here](https://github.com/IoanAnev/ompss-tutorial/blob/master/code/matmul-omp.cpp).

#### **Allocation & Deallocation**

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

#### **Initialization**

<div align="justify">

After allocation, the arrays are being initialized collectively using the OpenMP worksharing-loop construct `pragma omp for`, for a correct and balanced NUMA node data distribution. Operation on the arrays is being done in a blocked manner for cache-friendly accesses.

</div>

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
        /* Init. a block for cache-friendly accesses    */
        init_block(i, j, BSIZE, mat_a, mat_b, mat_c, mat_r);
  /* _implicit barrier_ */
}
```

#### **Computation**

<div align="justify">

The multiplication is being done similar to the initialization, exploiting coarse-grained parallelism by parallelizing the outermost loop of the kernel, and using the method of tiling. Note that there is a barrier implied after each `pragma omp for` construct, thus there is no need for one to be exclusively included.

</div>

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

#### **Verification**

<div align="justify">

The validity of the results is being checked by serially executing the kernel (no tiling), and then comparing the values of the output matrices from the two different kernels. Both the serial execution and the correctness test are handled by a single thread chosen at runtime through `pragma omp single`. Similar to `pragma omp for`, this construct also includes an implicit barrier at the end of its body.

</div>

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

#### **Allocation & Deallocation**

<div align="justify">

Since operation is still being done on the shared memory level, there is no difference in the allocation and deallocation mechanisms between the OpenMP and OmpSs-2 implementations.

</div>

#### **Initialization**

<div align="justify">

OmpSs being a task-based runtime system, the way we distribute work differs from the OpenMP method of parallelization used in the current example. In this example, we choose to associate a task to each tile to be computed. That said, the main thread iterates over the loops and keeps spawning tasks, which then reside on the task pool, ready to be served by a team of threads. Which thread computes which tile is chosen by the runtime.

The tasks need to also include directionality annotations based on the type of accesses being done to the data structures operated in the task body. As the matrices are only written during initialization, we enclose them in the `out` clause. Aside from expressing data direction, we need to specify which section of data is going to be operated by each task, in order to build proper dependencies between tasks, and thus effectively avoid data races.

In OmpSs, directives can be _inlined_ or _outlined_. When inlined, the `pragma` applies immediatly to the following statement, and the compiler outlines that statement as in OpenMP. The code block below presents an example of inlined directives.

</div>

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
      /* Spawn a task for each individual block         */
      #pragma oss task                 \
          out(mat_a[i;BSIZE][j;BSIZE], \
              mat_b[i;BSIZE][j;BSIZE], \
              mat_c[i;BSIZE][j;BSIZE], \
              mat_r[i;BSIZE][j;BSIZE]) \
          firstprivate(i, j, BSIZE)
      init_block(i, j, BSIZE, mat_a, mat_b, mat_c, mat_r);
}
```

<div align="justify">

> **NOTE:**
> The plain OmpSs version is very relaxed in the way its dependencies are specified, in contrast to the cluster version. It allows the use of sentinels as representatives of a larger section of the data structures. However, despite this amenity that it provides, it is good practice to provide as much information as possible to the runtime regarding the data access pattern, for portability and compatibility reasons.

</div>

#### **Computation**

<div align="justify">

The execution of the matrix multiplication kernel is similar to the initialization, in the way of how the tasks are dispatched. However, there are differences between the two code sections. The first difference that hits the eye is that the `pragma` directive is not _inlined_ to the function invocation of `multiply_block`, but attached to its declaration instead. This is an example of _outlined_ directives. Outlining the directive for a function means that this function effectively becomes a task upon its invocation. When outlining directives, additional information needs to be included in the function declaration for the dependencies to be computed by the runtime.

</div>

```cpp
/**
 * @note: Directives are outlined, and as such,
 *        all function invocations become a task
 * @warning: The parameter names need to be included
 */
#pragma oss task               \
in(   mat_a[i;bsize][k;bsize], \
      mat_b[k;bsize][j;bsize]) \
inout(mat_c[i;bsize][j;bsize])
void
multiply_block(
    const int i, const int j, const int k, const int bsize,
    double (*mat_a)[N], double (*mat_b)[N], double (*mat_c)[N]);
```

<div align="justify">

Another difference, is that the `pragma` directive now includes two different dependency clauses, namely `in`, and `inout`. The former, encloses the two input matrices as they are only read during computation, while the latter, encloses the output matrix, of which parts are accumulated with the execution of each task. These two, in conjuction with the `out` dependency clause, make up for all the basic strong dependency clauses.

As there is no implicit mechanism in the OmpSs programming model for imposing a "barrier" whenever necessary, the `taskwait` construct needs to be explicitly issued. This construct, blocks the current control flow of the program, until the completion of all the direct descendant tasks. Notice that a `taskwait` is not included between the initialization and computation, due to the implicit synchronization imposed from the task dependencies.

</div>

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

#### **Verification**

<div align="justify">

The verification of the results is left to be taken care of by the main thread, as we are guaranteed to see the updated values of the memory being worked on during the computation, by directly dereferencing. Of course, verification can also be handled within tasks.

</div>

### **OmpSs-2@Cluster implementation**
-----------------------------

The full source code of the implementation can be found [here](https://github.com/IoanAnev/ompss-tutorial/blob/master/code/matmul-ompss-2-cluster.cpp).

#### **Allocation & Deallocation**

<div align="justify">

OmpSs@Cluster is an extension of the OmpSs programming model to handle disjoint address spaces. As such, the memory model introduces the notion of two distinct memory types in which distributed computations can take place, _local_, and _distributed_ memory.

Local memory is cluster-capable memory, as it can participate in distributed computations, and is allocated on a single node. This type of memory can be directly dereferenced within the context of the task that allocated it. Users can allocate and deallocate local memory through the `nanos6_lmalloc` and `nanos6_lfree` API calls, respectively. As seen below, we introduce the corresponding wrappers for these functions, `lmalloc` and `lfree`, mainly for programming convencience.

</div>

```cpp
/**
 * @brief: Wrapper for the nanos6_lmalloc call
 *
 * @param[in] size: Number of type T elements to allocate
 * @returns       : A pointer to the local memory allocated
 */
template<typename T>
static inline T* lmalloc(size_t size)
{
  T* alloc = (T*)nanos6_lmalloc(sizeof(T) * size);
  return alloc;
}

/**
 * @brief: Wrapper for the nanos6_lfree call
 *
 * @param[in] ptr : Pointer to the locally allocated memory
 * @param[in] size: Size of the locally allocated memory
 */
template<typename T>
static inline void lfree(void* ptr, size_t size)
{
  nanos6_lfree(ptr, sizeof(T) * size);
}
```

<div align="justify">

Distributed memory is cluster-capable memory that is allocated collectively by all the nodes that participate in the execution. This type of memory can be dereferenced only within the sub-tasks of the task that allocated it. Users can allocate and deallocate distributed memory through the `nanos6_dmalloc` and `nanos6_lfree` API calls, respectively. When allocating distributed memory, the user application can define a _distribution policy_, to give hints to the runtime about the data placement of the allocated data across nodes. As there is one distribution currently available, it is passed as a default value to the relevant parameter in the `dmalloc` wrapper. Similarly, we pass magic values to the remaining parameters, due to their missing underlying functionality.

</div>

```cpp
/**
 * @brief: Wrapper for the nanos6_dmalloc call
 *
 * @param[in] size          : Number of type T elements to allocate
 * @param[in] policy        : Data distribution policy
 * @param[in] num_dimensions: Number of dimensions
 * @param[in] dimensions    : Array containing the size of
 *                            every distribution dimension
 */
template<typename T>
static inline T* dmalloc(size_t size,
    nanos6_data_distribution_t policy = nanos6_equpart_distribution,
    size_t num_dimensions = 0,
    size_t *dimensions = NULL)
{
  T* alloc = (T*)nanos6_dmalloc(sizeof(T) * size,
                                policy, num_dimensions, dimensions);
  return alloc;
}

/**
 * @brief: Wrapper for the nanos6_dfree call
 *
 * @param[in] ptr : Pointer to the distributed allocated memory
 * @param[in] size: Size of the distributed allocated memory
 */
template<typename T>
static inline void dfree(void* ptr, size_t size)
{
  nanos6_dfree(ptr, sizeof(T) * size);
}
```

<div align="justify">

We allocate the program matrices on distributed memory, since they are operated collectively, but also to take advantage of the extented memory capabilities of the cluster machine. Notice that we cast the single pointers returned from the `dmalloc` calls to view the arrays as two-dimensional, thing which helps us ease the specification of the data dependencies in the `pragma` directives, as well as accessing the structures.

</div>

```cpp
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
```

#### **Initialization & Computation**

<div align="justify">

One of the characteristics of the cluster version of OmpSs, aside from its memory model, is its more strict data-flow semantics over traditional OmpSs. Having said that, it requires the programmer to specify exactly all the memory accesses that happen inside a task in its dependency list, without allowing the use of defining only a subset of the them. If the program is not annotated correctly, proper ordering will not be enforced between tasks and wrong results will most probably be observed.

From the above statements, it can be deduced that any OmpSs@Cluster program is a correct shared memory OmpSs program as well, but not vice versa. As far as the current application is concerned, the initialization and computation sections are compatible between the traditional and the cluster version of OmpSs, as the relevant code blocks were initially profoundly annotated.

</div>

#### **Verification**

<div align="justify">

Due to the restrictions for accessing distributed memory, the functions related to verification need to be taskified, in order to avoid runtime access errors and reading outdated values. Of course, after this modification, `taskwait` clauses need to be included after the invocation of these functions, for correct performance measurements and early program termination avoidance.

</div>

```cpp
/**
 * @note: Directives are outlined, and as such,
 *        all function invocations become a task
 * @warning: The parameter names need to be included
 */
#pragma oss task               \
in(mat_c[0;NSIZE][0;NSIZE],    \
   mat_r[0;NSIZE][0;NSIZE])
void verify_results(
    double (*mat_c)[N],
    double (*mat_r)[N]);

/**
 * @note: Directives are outlined, and as such,
 *        all function invocations become a task
 * @warning: The parameter names need to be included
 */
#pragma oss task               \
in(   mat_a[0;NSIZE][0;NSIZE], \
      mat_b[0;NSIZE][0;NSIZE]) \
inout(mat_r[0;NSIZE][0;NSIZE])
void matmul_ref(
    double (*mat_a)[N],
    double (*mat_b)[N],
    double (*mat_r)[N]);
```
