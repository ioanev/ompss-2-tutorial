/**
 * @file
 * @brief This file implements wrappers for
 *        the nanos6 memory allocation calls
 */

#ifndef __MATMUL_MEMORY_HPP__
#define __MATMUL_MEMORY_HPP__

#include <iostream>
using namespace std;

/** @brief wrapper for the nanos6_lmalloc call */
template<typename T>
static inline T* lmalloc(size_t size)
{
#ifdef ENABLE_CLUSTER
	T* alloc = (T*)nanos6_lmalloc(sizeof(T) * size);
#else
	T* alloc = new T[size];
#endif
	if (!alloc)
		std::cerr << "Could not allocate "
			  << "local memory"
			  << std::endl;
	return alloc;
}

/** @brief wrapper for the nanos6_dmalloc call */
template<typename T>
static inline T* dmalloc(size_t size,
		nanos6_data_distribution_t policy =
			nanos6_equpart_distribution,
		size_t num_dimensions = 0,
		size_t *dimensions = NULL)
{
#ifdef ENABLE_CLUSTER
	T* alloc = (T*)nanos6_dmalloc(sizeof(T) * size,
			policy, num_dimensions, dimensions);
#else
	T* alloc = new T[size];
#endif
	if (!alloc)
		std::cerr << "Could not allocate "
			  << "distributed memory"
			  << std::endl;
	return alloc;
}

/** @brief wrapper for the nanos6_lfree call */
template<typename T>
static inline void lfree(void* ptr, size_t size)
{
#ifdef ENABLE_CLUSTER
	nanos6_lfree(ptr, sizeof(T) * size);
#else
	delete[] static_cast<T*>(ptr);
#endif
}

/** @brief wrapper for the nanos6_dfree call */
template<typename T>
static inline void dfree(void* ptr, size_t size)
{
#ifdef ENABLE_CLUSTER
	nanos6_dfree(ptr, sizeof(T) * size);
#else
	delete[] static_cast<T*>(ptr);
#endif
}

#endif /* __MATMUL_MEMORY_HPP__ */
