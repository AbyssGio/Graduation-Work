import jittor as jt
import os

import math


THREADS = 256

header_path = os.path.dirname(__file__)

jt.flags.use_cuda = 1

cudaheader = """
#include <cuda.h>
#include <cuda_runtime.h>
#include <iomanip.h>
#include <cmath.h>

#include <vector>
using namespace std;
"""

def adam_upd_cuda(
        param: jt.Var,
        grad: jt.Var,
        expavg: jt.Var,
        expavgsq: jt.Var,
        step: int,
        beta1: float,
        beta2: float,
        lr: float,
        eps: float
):
    N = 1
    for num in jt.size(param):
        N *= num

    N = jt.array(N)

    stepsize =  lr * math.sqrt(1 - math.pow(beta2, step)) / (1 - math.pow(beta1, step))

    step = jt.array(step)
    beta1 = jt.array(beta1)
    beta2 = jt.array(beta2)
    lr = jt.array(lr)
    eps = jt.array(eps)
    stepsize = jt.array(stepsize)

    param, expavg, expavgsq = jt.code(
        [param.shape, expavg.shape, expavgsq.shape],
        [param.dtype, expavg.dtype, expavgsq.dtype],
        inputs=[param, grad, expavg, expavgsq, N, stepsize, beta1, beta2, eps],
        cuda_header=cudaheader,
        cuda_src="""
        template <typename scalar_t>
        __global__ void adam_upd_cuda_kernel(@ARGS_DEF) {
          @PRECALC
          const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
          if(index<@in4(0)) {
            @in2(index) = @in6(0) * @in2(index) + (1-@in6(0)) * @in1(index);
            @in3(index) = @in7(0) * @in3(index) + (1-@in7(0)) * @in1(index) * @in1(index);
            @in0(index) -= @in5(0) * @in2(index) / (sqrt(@in3(index)) + @in8(0));
          }
        }
        
        int threads = 256;
        int blocks = (@in4(0) + threads - 1) / threads;
        
        adam_upd_cuda_kernel<<<blocks, threads>>>(@ARGS);
        
        @out0 = @in0;
        @out1 = @in2;
        @out2 = @in3;
        """
    )
    return param, expavg, expavgsq

def masked_adam_upd_cuda(
        param: jt.Var,
        grad: jt.Var,
        expavg: jt.Var,
        expavgsq: jt.Var,
        step: int,
        beta1: float,
        beta2: float,
        lr: float,
        eps: float
):
    N = 1
    for num in jt.size(param):
        N *= num
    N = jt.array(N)

    stepsize = lr * math.sqrt(1 - math.pow(beta2, step)) / (1 - math.pow(beta1, step))

    step = jt.array(step)
    beta1 = jt.array(beta1)
    beta2 = jt.array(beta2)
    lr = jt.array(lr)
    eps = jt.array(eps)
    stepsize = jt.array(stepsize)

    param, expavg, expavgsq = jt.code(
        [param.shape, expavg.shape, expavgsq.shape],
        [param.dtype, expavg.dtype, expavgsq.dtype],
        inputs=[param, grad, expavg, expavgsq, N, stepsize, beta1, beta2, eps],
        cuda_header=cudaheader,
        cuda_src="""
        __global__ void masked_adam_upd_cuda_kernel(@ARGS_DEF) {
          @PRECALC
          const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
          if(index<@in4(0) && @in1(index)!=0) {
            @in2(index) = @in6(0) * @in2(index) + (1-@in6(0)) * @in1(index);
            @in3(index) = @in7(0) * @in3(index) + (1-@in7(0)) * @in1(index) * @in1(index);
            @in0(index) -= @in5(0) * @in2(index) / (sqrt(@in3(index)) + @in8(0));
          }
        }
        
        int threads = 256;
        int blocks = (@in4(0) + threads - 1) / threads;
        
        masked_adam_upd_cuda_kernel<<<blocks, threads>>>(@ARGS);
        
        @out0 = @in0;
        @out1 = @in2;
        @out2 = @in3;
        
        """
    )
    return param, expavg, expavgsq

def adam_upd_with_perlr_cuda(
        param: jt.Var,
        grad: jt.Var,
        expavg: jt.Var,
        expavgsq: jt.Var,
        perlr: jt.Var,
        step: int,
        beta1: float,
        beta2: float,
        lr: float,
        eps: float
):
    N = 1
    for num in jt.size(param):
        N *= num
    N = jt.array(N)

    threads = THREADS
    blocks = (N + threads - 1) / threads
    stepsize = lr * math.sqrt(1 - math.pow(beta2, step)) / (1 - math.pow(beta1, step))

    step = jt.array(step)
    beta1 = jt.array(beta1)
    beta2 = jt.array(beta2)
    lr = jt.array(lr)
    eps = jt.array(eps)
    stepsize = jt.array(stepsize)

    param, expavg, expavgsq = jt.code(
        [param.shape, expavg.shape, expavgsq.shape],
        [param.dtype, expavg.dtype, expavgsq.dtype],
        inputs=[param, grad, expavg, expavgsq, perlr, N, stepsize, beta1, beta2, eps],
        cuda_header=cudaheader,
        cuda_src="""
        __global__ void adam_upd_with_perlr_cuda_kernel(@ARGS_DEF) {
          @PRECALC
          const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
          if(index<@in5(0)) {
            @in2(index) = @in7(0) * @in2(index) + (1-@in7(0)) * @in1(index);
            @in3(index) = @in8(0) * @in3(index) + (1-@in8(0)) * @in1(index) * @in1(index);
            @in0(index) -= @in6(0) * @in4(index) * @in2(index) / (sqrt(@in3(index)) + @in9(0));
          }
        }
        
        int threads = 256;
        int blocks = (@in5(0) + threads - 1) / threads;

        adam_upd_with_perlr_cuda_kernel<<<blocks, threads>>>(@ARGS);
        
        @out0 = @in0;
        @out1 = @in2;
        @out2 = @in3;
        """
    )

    return param, expavg, expavgsq