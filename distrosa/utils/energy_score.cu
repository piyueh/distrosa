/**
 * @file energy_score.cu
 * @brief CUDA C implementations of Energy Score with Jacobian.
 * @author Pi-Yueh Chuang
 * @version 0.1-alpha
 * @date 2025-04-01
 */
#include <cuda_runtime.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>

// A small constant to avoid division by zero.
#define EPS 1.0e-15
#define NDIM_MAX 12  // assume ndim <= 12


/**
 * @brief Kernel for computing x-y score.
 *
 * @param x Read-only pointer to a nx-by-ndim double array.
 * @param y Read-only pointer to a ny-by-ndim double array.
 * @param jac Pointer to a nx-by-ndim double array.
 * @param score Pointer to a double.
 * @param nx Number of rows in x.
 * @param ny Number of rows in y.
 * @param ndim Number of columns in x and y.
 * @param coeff1 Coefficient for the score.
 * @param coeff2 Coefficient for the jacobian.
 * @return void.
 *
 * @note
 *  - Requires shared memory of size
 *     ((blockDim.x+blockDim.y)*ndim+blockDim.x*blockDim.y*(ndim+1))*sizeof(double).
 *  - Assume CUDA arrange 2D threads into warps by running x dimension first.
 */
extern "C" __global__ void _kernel_xy(
    const double* __restrict__ x,
    const double* __restrict__ y,
    double* __restrict__ jac,
    double* __restrict__ score,
    size_t nx,
    size_t ny,
    size_t ndim,
    double coeff1,
    double coeff2
){

    assert(ndim <= NDIM_MAX);

    // memory space shared by all threads in the block
    extern __shared__ double shared[];
    double* s_x = shared;  // x array starts here in the shared memory
    double* s_y = s_x + blockDim.x * ndim;  // y array starts here in the shared memory
    double* s_score = s_y + blockDim.y * ndim;  // local scores starts here
    double* s_jac = s_score + (blockDim.x * blockDim.y);  // jacobian starts here

    // data belongs to the current thread only
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;  // global index in x
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;  // global index in y
    size_t tid = threadIdx.x + blockDim.x * threadIdx.y;  // flattened ID in a 2D array
    size_t total_threads = blockDim.x * blockDim.y;  // num. of threads in a block
    double rvec[NDIM_MAX];  // distance vector between x[i] and y[j]
    double rnorm = 0.0;  // to store \sqrt{\sum_{d=1}^{ndim}(x[i][d]-y[j][d])^2}

    // load x segment to local mem by thread with local id (:, 0)
    if(threadIdx.y == 0 && i < nx) {
        for (size_t d = 0; d < ndim; d++) {
            s_x[threadIdx.x*ndim+d] = x[i*ndim+d];
        }
    }

    // load y segment to local mem by thread with local id (:, 0)
    if(threadIdx.x == 0 && j < ny) {
        for (int d = 0; d < ndim; d++) {
            s_y[threadIdx.y*ndim+d] = y[j*ndim+d];
        }
    }

    // initialize the local jacobian container
    for (size_t d = 0; d < ndim; d++) {
        s_jac[tid*ndim+d] = 0.0;
    }

    // initialize the local score holder
    s_score[tid] = 0.0;

    // synchronize threads in this block so s_x and s_y are ready
    __syncthreads();

    // get score_{ij} and its gradients w.r.t. x_i
    if(i < nx && j < ny) {
        // get distance vector and its norm
        for (size_t d = 0; d < ndim; d++) {
            rvec[d] = s_x[threadIdx.x*ndim+d] - s_y[threadIdx.y*ndim+d];
            rnorm += (rvec[d] * rvec[d]);
        }
        rnorm = sqrt(rnorm);

        // compute the score and the gradients
        if (rnorm > EPS) {
            s_score[tid] = rnorm;
            for (size_t d = 0; d < ndim; d++) {
                s_jac[tid*ndim+d] = rvec[d] / rnorm;
            }
        }

    }
    __syncthreads();  // synchronize threads so that s_score and s_jac are ready

    // naive tree-reduction; first reduction across warps to the 1st warp
    for (size_t stride = total_threads / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            s_score[tid] += s_score[tid + stride];
        }
        __syncthreads();
    }

    // next, reduction within a warp
    if (tid < 32) {
        s_score[tid] += s_score[tid+32];
        double tmp = s_score[tid];
        for (size_t offset = 16; offset > 0; offset /= 2) {
            tmp += __shfl_down_sync(0xFFFFFFFF, tmp, offset);
        }
        s_score[tid] = tmp;
    }

    // atomic add the block’s score to the global score
    if(tid == 0) atomicAdd(score, s_score[0]/coeff1);

    // naive tree-reduction for local jacobian (i.e., sum(s_jac, axis=1))
    for (size_t stride = blockDim.y / 2; stride > 0; stride >>= 1) {
        if (threadIdx.y < stride) {
            for (size_t d = 0; d < ndim; d++) {
                s_jac[tid*ndim+d] += s_jac[(tid+stride*blockDim.x)*ndim+d];
            }
        }
        __syncthreads();
    }

    // add to the global jac
    if(threadIdx.y == 0) {
        for (size_t d = 0; d < ndim; d++) {
            atomicAdd(&jac[i*ndim+d], s_jac[tid*ndim+d]/coeff2);
        }
    }
}
