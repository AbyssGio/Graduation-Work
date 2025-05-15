import jittor as jt
from typing import Tuple

cudaheader = """
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
using namespace std;


"""


def infer_t_minmax_cuda(
        rayso: jt.Var,
        raysd: jt.Var,
        xyzmin: jt.Var,
        xyzmax: jt.Var,
        near: float,
        far: float
) -> Tuple[jt.Var, jt.Var]:
    near1 = jt.array(near).float32()
    far1 = jt.array(far).float32()
    tmin = tmax = jt.zeros(jt.size(rayso)[0], dtype="float")

    tmin, tmax = jt.code(
        [tmin.shape, tmax.shape],
        [tmin.dtype, tmax.dtype],
        inputs=[rayso, raysd, xyzmin, xyzmax, near1, far1, jt.array(rayso.shape[0])],
        cuda_header=cudaheader,
        cuda_src="""

         __global__ void infer_t_minmax_cuda_kernel(@ARGS_DEF){
        @PRECALC
        const int i_ray = blockIdx.x * blockDim.x + threadIdx.x;

        if(i_ray<@in6(0)) {
            const int offset = i_ray;
            float vx = ((@in1(offset,0)==0) ? 1e-6 : @in1(offset,0));
            float vy = ((@in1(offset,1)==0) ? 1e-6 : @in1(offset,1));
            float vz = ((@in1(offset,2)==0) ? 1e-6 : @in1(offset,2));
            float ax = (@in3(0) - @in0(offset,0)) / vx;
            float ay = (@in3(1) - @in0(offset,1)) / vy;
            float az = (@in3(2) - @in0(offset,2)) / vz;
            float bx = (@in2(0) - @in0(offset,0)) / vx;
            float by = (@in2(1) - @in0(offset,1)) / vy;
            float bz = (@in2(2) - @in0(offset,2)) / vz;

            @out0(i_ray) = max(min(max(max(min(ax, bx), min(ay, by)), min(az, bz)), @in5(0)), @in4(0));
            @out1(i_ray) = max(min(min(min(max(ax, bx), max(ay, by)), max(az, bz)), @in5(0)), @in4(0));
          }
        }

        int threads = 256;
        int blocks = (@in6(0) + threads - 1) /  threads;

        infer_t_minmax_cuda_kernel<<<blocks, threads>>>(@ARGS);
        """)

    return tmin, tmax

rayso = jt.randn([25600, 3])
raysd= jt.randn([25600, 3])
xyzmin = jt.randn([3])
xyzmax= jt.randn([3])
near = 1.
far = 1.

tmin , tmax = infer_t_minmax_cuda(rayso, raysd, xyzmin, xyzmax, near, far)
