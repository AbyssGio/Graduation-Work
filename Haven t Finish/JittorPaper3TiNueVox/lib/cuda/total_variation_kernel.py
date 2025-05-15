import jittor as jt
import os

import math


THREADS = 256

header_path = os.path.dirname(__file__)
proj_options = {f'FLAGS: -I"{header_path}" -L"{os.path.dirname(__file__)}"' : 1}

jt.flags.use_cuda = 1

cudaheader ="""

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <cmath>
#include <iomainp>

using namespace std;
"""

def total_variation_add_grad_cuda(
        param: jt.Var,
        grad: jt.Var,
        wx: float,
        wy: float,
        wz: float,
        densemode: bool
):
    N = 1
    for num in jt.size(param):
        N *= num

    szi = jt.size(param)[2]
    szj = jt.size(param)[3]
    szk = jt.size(param)[4]

    threads = THREADS
    blocks = (N + threads - 1) / threads

    wx /= 6
    wy /= 6
    wz /= 6

    if densemode:
        grad = jt.code(
            grad.shape,
            grad.dtype,
            inputs=[param, grad, wx, wy, wz, szi, szj, szk, N],
            cuda_header=cudaheader,
            cuda_src="""
            template <typename scalar_t, typename bound_t>
            __device__ __forceinline__ scalar_t clamp(const scalar_t v, const bound_t lo, const bound_t hi) {
              return min(max(v, lo), hi);
            }
            
            template <bool dense_mode>
            __global__ void total_variation_add_grad_cuda_kernel(@ARGS_DEF) {
              @PRECALC
              const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
              if(index<@in8(0) && (dense_mode || @in1(index)!=0)) {
                const size_t k = index % @in7(0);
                const size_t j = index / @in7(0) % @in6(0);
                const size_t i = index / @in7(0) / @in6(0) % @in5(0);
            
                float grad_to_add = 0;
                grad_to_add += (k==0      ? 0 : @in4(0) * clamp<float, float>(@in0(index)-@in0(index-1), -1.f, 1.f));
                grad_to_add += (k==@in7(0)-1 ? 0 : @in4(0) * clamp<float, float>(@in0(index)-@in0(index+1), -1.f, 1.f));
                grad_to_add += (j==0      ? 0 : @in3(0) * clamp<float, float>(@in0(index)-@in0(index-@in7(0)), -1.f, 1.f));
                grad_to_add += (j==@in6(0)-1 ? 0 : @in3(0) * clamp<float, float>(@in0(index)-@in0(index+@in7(0)), -1.f, 1.f));
                grad_to_add += (i==0      ? 0 : @in4(0) * clamp<float, float>(@in0(index)-@in0(index-@in7(0)*@in6(0)), -1.f, 1.f));
                grad_to_add += (i==@in5(0)-1 ? 0 : @in4(0) * clamp<float, float>(@in0(index)-@in0(index+@in7(0)*@in6(0)), -1.f, 1.f));
                @in1(index) += grad_to_add;
              }
            }
            
            int threads = 256;
            int blocks = (@in8(0) + threads - 1) / threads;

            total_variation_add_grad_cuda_kernel<true><<<blocks, threads>>>(@ARGS);
            
            @out0 = @in1;
            """
        )

    else:
        grad = jt.code(
            grad.shape,
            grad.dtype,
            inputs=[param, grad, wx, wy, wz, szi, szj, szk, N],
            cuda_header=cudaheader,
            cuda_src="""
            template <typename scalar_t, typename bound_t>
            __device__ __forceinline__ scalar_t clamp(const scalar_t v, const bound_t lo, const bound_t hi) {
              return min(max(v, lo), hi);
            }

            template <bool dense_mode>
            __global__ void total_variation_add_grad_cuda_kernel(@ARGS_DEF) {
              @PRECALC
              const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
              if(index<@in8(0) && (dense_mode || @in1(index)!=0)) {
                const size_t k = index % @in7(0);
                const size_t j = index / @in7(0) % @in6(0);
                const size_t i = index / @in7(0) / @in6(0) % @in5(0);

                float grad_to_add = 0;
                grad_to_add += (k==0      ? 0 : @in4(0) * clamp<float, float>(@in0(index)-@in0(index-1), -1.f, 1.f));
                grad_to_add += (k==@in7(0)-1 ? 0 : @in4(0) * clamp<float, float>(@in0(index)-@in0(index+1), -1.f, 1.f));
                grad_to_add += (j==0      ? 0 : @in3(0) * clamp<float, float>(@in0(index)-@in0(index-@in7(0)), -1.f, 1.f));
                grad_to_add += (j==@in6(0)-1 ? 0 : @in3(0) * clamp<float, float>(@in0(index)-@in0(index+@in7(0)), -1.f, 1.f));
                grad_to_add += (i==0      ? 0 : @in4(0) * clamp<float, float>(@in0(index)-@in0(index-@in7(0)*@in6(0)), -1.f, 1.f));
                grad_to_add += (i==@in5(0)-1 ? 0 : @in4(0) * clamp<float, float>(@in0(index)-@in0(index+@in7(0)*@in6(0)), -1.f, 1.f));
                @in1(index) += grad_to_add;
              }
            }

            int threads = 256;
            int blocks = (@in8(0) + threads - 1) / threads;

            total_variation_add_grad_cuda_kernel<false><<<blocks, threads>>>(@ARGS);

            @out0 = @in1;
            """
        )
    return grad
