// Host-only BLAS-3 GEMM for the HIP-CPU backend.
//
// Each rocblas_{s,d,c,z}gemm launches a single-thread (1,1,1) HIP kernel on
// the handle's stream. The kernel body calls the matching OpenBLAS
// cblas_{s,d,c,z}gemm through an __asm-labeled declaration, so the matrix
// multiply runs inside the HIP runtime and OpenBLAS supplies the internal
// threading. The cblas symbols are resolved against libopenblas when the
// device image is loaded.

#include "handle.hpp"
#include "rocblas.h"

#include <hip/hip_runtime.h>

namespace
{
    // rocblas is column-major (Fortran order). CblasColMajor == 102.
    constexpr int kCblasColMajor = 102;
}

// rocblas_operation values (111/112/113) equal CBLAS_TRANSPOSE values
// (CblasNoTrans/CblasTrans/CblasConjTrans), so the operation arguments pass
// through as ints with no mapping.

extern "C" __device__ void cblas_sgemm(int          order,
                                       int          transA,
                                       int          transB,
                                       int          m,
                                       int          n,
                                       int          k,
                                       float        alpha,
                                       const float* A,
                                       int          lda,
                                       const float* B,
                                       int          ldb,
                                       float        beta,
                                       float*       C,
                                       int          ldc) __asm("cblas_sgemm");

extern "C" __device__ void cblas_dgemm(int           order,
                                       int           transA,
                                       int           transB,
                                       int           m,
                                       int           n,
                                       int           k,
                                       double        alpha,
                                       const double* A,
                                       int           lda,
                                       const double* B,
                                       int           ldb,
                                       double        beta,
                                       double*       C,
                                       int           ldc) __asm("cblas_dgemm");

extern "C" __device__ void cblas_cgemm(int         order,
                                       int         transA,
                                       int         transB,
                                       int         m,
                                       int         n,
                                       int         k,
                                       const void* alpha,
                                       const void* A,
                                       int         lda,
                                       const void* B,
                                       int         ldb,
                                       const void* beta,
                                       void*       C,
                                       int         ldc) __asm("cblas_cgemm");

extern "C" __device__ void cblas_zgemm(int         order,
                                       int         transA,
                                       int         transB,
                                       int         m,
                                       int         n,
                                       int         k,
                                       const void* alpha,
                                       const void* A,
                                       int         lda,
                                       const void* B,
                                       int         ldb,
                                       const void* beta,
                                       void*       C,
                                       int         ldc) __asm("cblas_zgemm");

__global__ void rocblas_gemm_sgemm_kernel(int          order,
                                          int          transA,
                                          int          transB,
                                          int          m,
                                          int          n,
                                          int          k,
                                          float        alpha,
                                          const float* A,
                                          int          lda,
                                          const float* B,
                                          int          ldb,
                                          float        beta,
                                          float*       C,
                                          int          ldc)
{
    cblas_sgemm(order, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

__global__ void rocblas_gemm_dgemm_kernel(int           order,
                                          int           transA,
                                          int           transB,
                                          int           m,
                                          int           n,
                                          int           k,
                                          double        alpha,
                                          const double* A,
                                          int           lda,
                                          const double* B,
                                          int           ldb,
                                          double        beta,
                                          double*       C,
                                          int           ldc)
{
    cblas_dgemm(order, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

__global__ void rocblas_gemm_cgemm_kernel(int         order,
                                          int         transA,
                                          int         transB,
                                          int         m,
                                          int         n,
                                          int         k,
                                          const void* alpha,
                                          const void* A,
                                          int         lda,
                                          const void* B,
                                          int         ldb,
                                          const void* beta,
                                          void*       C,
                                          int         ldc)
{
    cblas_cgemm(order, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

__global__ void rocblas_gemm_zgemm_kernel(int         order,
                                          int         transA,
                                          int         transB,
                                          int         m,
                                          int         n,
                                          int         k,
                                          const void* alpha,
                                          const void* A,
                                          int         lda,
                                          const void* B,
                                          int         ldb,
                                          const void* beta,
                                          void*       C,
                                          int         ldc)
{
    cblas_zgemm(order, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

extern "C" {

rocblas_status rocblas_sgemm(rocblas_handle    handle,
                             rocblas_operation transA,
                             rocblas_operation transB,
                             rocblas_int       m,
                             rocblas_int       n,
                             rocblas_int       k,
                             const float*      alpha,
                             const float*      A,
                             rocblas_int       lda,
                             const float*      B,
                             rocblas_int       ldb,
                             const float*      beta,
                             float*            C,
                             rocblas_int       ldc)
{
    if(!handle)
        return rocblas_status_invalid_handle;
    if(m <= 0 || n <= 0 || k <= 0)
        return rocblas_status_success;
    if(!alpha || !beta || !A || !B || !C)
        return rocblas_status_invalid_pointer;

    hipLaunchKernelGGL(rocblas_gemm_sgemm_kernel,
                       dim3(1),
                       dim3(1),
                       0,
                       handle->get_stream(),
                       kCblasColMajor,
                       (int)transA,
                       (int)transB,
                       (int)m,
                       (int)n,
                       (int)k,
                       *alpha,
                       A,
                       (int)lda,
                       B,
                       (int)ldb,
                       *beta,
                       C,
                       (int)ldc);
    return rocblas_status_success;
}

rocblas_status rocblas_dgemm(rocblas_handle    handle,
                             rocblas_operation transA,
                             rocblas_operation transB,
                             rocblas_int       m,
                             rocblas_int       n,
                             rocblas_int       k,
                             const double*     alpha,
                             const double*     A,
                             rocblas_int       lda,
                             const double*     B,
                             rocblas_int       ldb,
                             const double*     beta,
                             double*           C,
                             rocblas_int       ldc)
{
    if(!handle)
        return rocblas_status_invalid_handle;
    if(m <= 0 || n <= 0 || k <= 0)
        return rocblas_status_success;
    if(!alpha || !beta || !A || !B || !C)
        return rocblas_status_invalid_pointer;

    hipLaunchKernelGGL(rocblas_gemm_dgemm_kernel,
                       dim3(1),
                       dim3(1),
                       0,
                       handle->get_stream(),
                       kCblasColMajor,
                       (int)transA,
                       (int)transB,
                       (int)m,
                       (int)n,
                       (int)k,
                       *alpha,
                       A,
                       (int)lda,
                       B,
                       (int)ldb,
                       *beta,
                       C,
                       (int)ldc);
    return rocblas_status_success;
}

rocblas_status rocblas_cgemm(rocblas_handle               handle,
                             rocblas_operation            transA,
                             rocblas_operation            transB,
                             rocblas_int                  m,
                             rocblas_int                  n,
                             rocblas_int                  k,
                             const rocblas_float_complex* alpha,
                             const rocblas_float_complex* A,
                             rocblas_int                  lda,
                             const rocblas_float_complex* B,
                             rocblas_int                  ldb,
                             const rocblas_float_complex* beta,
                             rocblas_float_complex*       C,
                             rocblas_int                  ldc)
{
    if(!handle)
        return rocblas_status_invalid_handle;
    if(m <= 0 || n <= 0 || k <= 0)
        return rocblas_status_success;
    if(!alpha || !beta || !A || !B || !C)
        return rocblas_status_invalid_pointer;

    hipLaunchKernelGGL(rocblas_gemm_cgemm_kernel,
                       dim3(1),
                       dim3(1),
                       0,
                       handle->get_stream(),
                       kCblasColMajor,
                       (int)transA,
                       (int)transB,
                       (int)m,
                       (int)n,
                       (int)k,
                       (const void*)alpha,
                       (const void*)A,
                       (int)lda,
                       (const void*)B,
                       (int)ldb,
                       (const void*)beta,
                       (void*)C,
                       (int)ldc);
    return rocblas_status_success;
}

rocblas_status rocblas_zgemm(rocblas_handle                handle,
                             rocblas_operation             transA,
                             rocblas_operation             transB,
                             rocblas_int                   m,
                             rocblas_int                   n,
                             rocblas_int                   k,
                             const rocblas_double_complex* alpha,
                             const rocblas_double_complex* A,
                             rocblas_int                   lda,
                             const rocblas_double_complex* B,
                             rocblas_int                   ldb,
                             const rocblas_double_complex* beta,
                             rocblas_double_complex*       C,
                             rocblas_int                   ldc)
{
    if(!handle)
        return rocblas_status_invalid_handle;
    if(m <= 0 || n <= 0 || k <= 0)
        return rocblas_status_success;
    if(!alpha || !beta || !A || !B || !C)
        return rocblas_status_invalid_pointer;

    hipLaunchKernelGGL(rocblas_gemm_zgemm_kernel,
                       dim3(1),
                       dim3(1),
                       0,
                       handle->get_stream(),
                       kCblasColMajor,
                       (int)transA,
                       (int)transB,
                       (int)m,
                       (int)n,
                       (int)k,
                       (const void*)alpha,
                       (const void*)A,
                       (int)lda,
                       (const void*)B,
                       (int)ldb,
                       (const void*)beta,
                       (void*)C,
                       (int)ldc);
    return rocblas_status_success;
}

} // extern "C"
