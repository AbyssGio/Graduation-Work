import jittor as jt
import os

from typing import Tuple


THREADS = jt.array(256)

header_path = os.path.dirname(__file__)
proj_options = {f'FLAGS: -I"{header_path}" -L"{os.path.dirname(__file__)}"' : 1}

jt.flags.use_cuda = 1

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
    ) -> Tuple[jt.Var, jt.Var] :

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
            const int offset = i_ray * 3;
            float vx = ((@in1(offset  )==0) ? 1e-6 : @in1(offset  ));
            float vy = ((@in1(offset+1)==0) ? 1e-6 : @in1(offset+1));
            float vz = ((@in1(offset+2)==0) ? 1e-6 : @in1(offset+2));
            float ax = (@in3(0) - @in0(offset  )) / vx;
            float ay = (@in3(1) - @in0(offset+1)) / vy;
            float az = (@in3(2) - @in0(offset+2)) / vz;
            float bx = (@in2(0) - @in0(offset  )) / vx;
            float by = (@in2(1) - @in0(offset+1)) / vy;
            float bz = (@in2(2) - @in0(offset+2)) / vz;

            @out0(i_ray) = max(min(max(max(min(ax, bx), min(ay, by)), min(az, bz)), @in5(0)), @in4(0));
            @out1(i_ray) = max(min(min(min(max(ax, bx), max(ay, by)), max(az, bz)), @in5(0)), @in4(0));
          }
        }
        
        int threads = 256;
        int blocks = (@in6(0) + threads - 1) /  threads;
        
        infer_t_minmax_cuda_kernel<<<blocks, threads>>>(@ARGS);
        """)

    return tmin, tmax

def infer_n_samples_cuda(
        tmin: jt.Var,
        tmax: jt.Var,
        stepdist: float
    ) -> jt.Var :


    nsamples = jt.zeros(jt.size(tmin)[0])
    stepdist = jt.array(stepdist).float32()

    nsamples =  jt.code(
        nsamples.shape,
        nsamples.dtype,
        inputs=[tmin, tmax, stepdist, jt.array(jt.size(tmin)[0])],
        cuda_header=cudaheader,
        cuda_src="""
        __global__ void infer_n_samples_cuda_kernel(@ARGS_DEF){
          @@PRECALC    
          const int i_ray = blockIdx.x * blockDim.x + threadIdx.x;
          if(i_ray<@in3(0)) {
            // at least 1 point for easier implementation in the later sample_pts_on_rays_cuda
            @out0(i_ray) = max(ceil((@in1(i_ray)-@in0(i_ray)) / @in2(0)), 1.);
          }
        }
        
        int threads = 256;
        int blocks = (in0_shape0 + threads - 1)/ threads;
        
        infer_n_samples_cuda_kernel<<<blocks ,threads>>>(@ARGS);
        """
    )
    return nsamples

def infer_ray_start_dir_cuda(
        rayso: jt.Var,
        raysd: jt.Var,
        tmin: jt.Var
    ) -> Tuple[jt.Var, jt.Var] :


    raysstart = raysdir = jt.zeros(jt.size(rayso)[0]).float32()
    raysstart, raysdir = jt.code(
        [raysstart.shape, raysdir.shape],
        [raysstart.dtype, raysdir.dtype],
        inputs=[rayso, raysd, tmin, jt.array(jt.size(rayso)[0])],
        cuda_header=cudaheader,
        cuda_src="""
        __global__ void infer_ray_start_dir_cuda_kernel(@ARGS_DEF) {      
          @PRECALC      
          const int i_ray = blockIdx.x * blockDim.x + threadIdx.x;
          if(i_ray<@in3(0)) {
            const int offset = i_ray * 3;
            const float rnorm = sqrt(
                    @in1(offset  )*@in1(offset  ) +\
                    @in1(offset+1)*@in1(offset+1) +\
                    @in1(offset+2)*@in1(offset+2));
            @out0(offset  ) = @in0(offset  ) + @in1(offset  ) * @in2(i_ray);
            @out0(offset+1) = @in0(offset+1) + @in1(offset+1) * @in2(i_ray);
            @out0(offset+2) = @in0(offset+2) + @in1(offset+2) * @in2(i_ray);
            @out1(offset  ) = @in1(offset  ) / rnorm;
            @out1(offset+1) = @in1(offset+1) / rnorm;
            @out1(offset+2) = @in1(offset+2) / rnorm;
          }
        }
        
        int threads = 256;
        int blocks = (in0_shape0 + threads - 1)/ threads;
        
        infer_ray_start_dir_cuda_kernel<<<blocks, threads>>>(@ARGS);
        """)

    return raysstart, raysdir

def sample_pts_on_rays_cuda(
        rayso: jt.Var,
        raysd: jt.Var,
        xyzmin: jt.Var,
        xyzmax: jt.Var,
        near: float,
        far: float,
        stepdist: float
    ) -> Tuple[jt.Var,jt.Var,jt.Var,jt.Var,jt.Var,jt.Var,jt.Var]:

    nrays = jt.array(jt.size(rayso)[0])
    tmin, tmax = infer_t_minmax_cuda(rayso, raysd, xyzmin, xyzmax, near, far)
    Nsteps = infer_n_samples_cuda(tmin, tmax, stepdist)
    Nstepscumsum = jt.array(jt.cumsum(Nsteps, 0))
    totallen = Nsteps.sum().int16()

    rayid = jt.zeros_like(totallen, dtype="int64")
    rayid = jt.code(
        rayid.shape,
        rayid.dtype,
        inputs=[rayid, Nstepscumsum, nrays],
        cuda_header=cudaheader,
        cuda_src="""
        __global__ void __set_1_at_ray_seg_start(@ARGS_DEF) {
          @PRECALC           
          const int idx = blockIdx.x * blockDim.x + threadIdx.x;
          if(0<idx && idx<@in2(0)) {
            @in0(@in1(idx-1)) = 1;
          }
        }
        
        int threads = 256;
        
        __set_1_at_ray_seg_start<<<(@in2(0)+threads-1)/threads, threads>>>(@ARGS);
        
        @out0 = @in0;
        """
    )
    rayid = jt.cumsum(rayid, dim=0)
    rayid = jt.array(rayid)
    stepid = jt.zeros_like(totallen, dtype="int64")
    stepid = jt.code(
        stepid.shape,
        stepid.dtype,
        inputs=[stepid, rayid, Nstepscumsum,totallen],
        cuda_header=cudaheader,
        cuda_src="""
            __global__ void __set_step_id(@ARGS_DEF) {     
                @PRECALC
                const int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if(idx<@in3(0)) {
                  const int rid = @in1(idx);
                  @in0(idx) = idx - ((rid!=0) ? @in2(rid-1) : 0);
                }
            }
            
            int threads = 256;

            __set_step_id<<<(@in3(0)+threads-1)/threads, threads>>>(@ARGS);
            
            @out0 = @in0;
            
            """
    )
    raysstartdir = infer_ray_start_dir_cuda(rayso, raysd, tmin)
    raystart = raysstartdir[0]
    raysdir = raysstartdir[1]

    rayspts = jt.zeros([totallen.item(), 3], dtype=rayso.dtype())
    maskoutbbox = jt.zeros_like(totallen, dtype="bool")

    rayspts, maskoutbbox = jt.code(
        [rayspts.shape,maskoutbbox.shape],
        [rayspts.dtype,maskoutbbox.dtype],
        inputs=[raystart, raysdir, xyzmin, xyzmax,  rayid, stepid, stepdist, totallen, rayspts, maskoutbbox],
        cuda_header=cudaheader,
        cuda_src="""
        __global__ void sample_pts_on_rays_cuda_kernel(@ARGS_DEF) {
          @PRECALC
          const int idx = blockIdx.x * blockDim.x + threadIdx.x;
          if(idx<@in7(0)) {
            const int i_ray = @in4(idx);
            const int i_step = @in5(idx);
        
            const int offset_p = idx * 3;
            const int offset_r = i_ray * 3;
            const float dist = @in6(0) * i_step;
            const float px = @in0(offset_r  ) + @in1(offset_r  ) * dist;
            const float py = @in0(offset_r+1) + @in1(offset_r+1) * dist;
            const float pz = @in0(offset_r+2) + @in1(offset_r+2) * dist;
            @in8(offset_p  ) = px;
            @in8(offset_p+1) = py;
            @in8(offset_p+2) = pz;
            @in9(idx) = (@in2(0)>px) | (@in2(1)>py) | (@in2(2)>pz) | \
                                (@in3(0)<px) | (@in3(1)<py) | (@in3(2)<pz);
          }
        }
        
        int threads = 256;
        
        sample_pts_on_rays_cuda_kernel<<<(@in7(0)+threads-1)/threads, threads>>>(@ARGS);
        
        @out0 = @in8;
        @out1 = @in9;
        
        """
    )
    return rayspts, maskoutbbox, rayid, stepid, Nsteps, tmin, tmax

def sample_ndc_pts_on_rays_cuda(
        rayso: jt.Var,
        raysd: jt.Var,
        xyzmin: jt.Var,
        xyzmax: jt.Var,
        Nsamples: float
    ) -> Tuple[jt.Var,jt.Var]:
    nrays = jt.array(jt.size(rayso)[0])
    rayspts = jt.zeros([nrays, Nsamples, 3], dtype=rayso.dtype())
    maskoutbox = jt.zeros([nrays, Nsamples], dtype="bool")

    rayspts, maskoutbox = jt.code(
        [rayspts.shape, maskoutbox.shape],
        [rayspts.dtype, maskoutbox.dtype],
        inputs=[rayso, raysd, xyzmin, xyzmax, Nsamples, nrays, rayspts, maskoutbox],
        cuda_header=cudaheader,
        cuda_src="""
        __global__ void sample_ndc_pts_on_rays_cuda_kernel(@ARGS_DEF) {
          @PRECALC 
          const int idx = blockIdx.x * blockDim.x + threadIdx.x;
          if(idx<@in4(0)*@in5(0)) {
            const int i_ray = idx / @in4(0);
            const int i_step = idx % @in4(0);
        
            const int offset_p = idx * 3;
            const int offset_r = i_ray * 3;
            const float dist = ((float)i_step) / (@in4(0)-1);
            const float px = @in0(offset_r  ) + @in1(offset_r  ) * dist;
            const float py = @in0(offset_r+1) + @in1(offset_r+1) * dist;
            const float pz = @in0(offset_r+2) + @in1(offset_r+2) * dist;
            @in6(offset_p  ) = px;
            @in6(offset_p+1) = py;
            @in6(offset_p+2) = pz;
            @in7(idx) = (@in2(0)>px) | (@in2(1)>py) | (@in2(2)>pz) | \
                                (@in3(0)<px) | (@in3(1)<py) | (@in3(2)<pz);
          }
        }
        
        int threads = 256;
        
        sample_ndc_pts_on_rays_cuda_kernel<<<(@in5(0)*@in4(0)+threads-1)/threads, threads>>>(@ARGS);
        
        @out1 = @in6;
        @out2 = @in7;
        """
    )

    return rayspts, maskoutbox

def maskcache_lookup_cuda(
        world: jt.Var,
        xyz: jt.Var,
        xyz2ijkscale: jt.Var,
        xyz2ijkshift: jt.Var
) -> jt.Var:
    a = jt.array(jt.size(world))
    szi = jt.array(a[0])
    szj = jt.array(a[1])
    szk = jt.array(a[2])
    npts = jt.array(jt.size(xyz)[0])

    out = jt.zeros(npts, dtype="bool")
    if npts == 0:
        return out

    threads = THREADS
    blocks = (npts + threads - 1) / threads

    out = jt.code(
        [out.shape],
        [out.dtype],
        inputs=[world, xyz, out, xyz2ijkscale, xyz2ijkshift, szi, szj, szk, npts],
        cuda_header=cudaheader,
        cuda_src="""
        static __forceinline__ __device__
        bool check_xyz(int i, int j, int k, int sz_i, int sz_j, int sz_k) {
          return (0 <= i) && (i < sz_i) && (0 <= j) && (j < sz_j) && (0 <= k) && (k < sz_k);
        }
        
        __global__ void maskcache_lookup_cuda_kernel(@ARGS_DEF) {
          @PRECALC
          const int i_pt = blockIdx.x * blockDim.x + threadIdx.x;
          if(i_pt<@in8) {
            const int offset = i_pt * 3;
            const int i = round(@in1(offset  ) * @in3(0) + @in4(0));
            const int j = round(@in1(offset+1) * @in3(1) + @in4(1));
            const int k = round(@in1(offset+2) * @in3(2) + @in4(2));
            if(check_xyz(i, j, k, @in5(0), @in6(0), @in7(0))) {
              @in2(i_pt) = @in0(i*@in6(0)* @in7(0) + j* @in7(0) + k);
            }
          }
        }
        
        int threads = 256;
        int blocks = (@in8(0)+ threads - 1) / threads;
                
        maskcache_lookup_cuda_kernel<<<blocks, threads>>>(@ARGS);
        
        @out0 = @in2;
        """
    )

    return out

def raw2alpha_cuda(
        density: jt.Var,
        shift: float,
        intercal: float
) -> Tuple[jt.Var, jt.Var] :
    npts = jt.array(jt.size(density)[0])
    expd = alpha = jt.zeros_like(density)
    if npts == 0:
        return expd, alpha

    shift = jt.array(shift)
    intercal = jt.array(intercal)

    expd, alpha = jt.code(
        [expd.shape, alpha.shape],
        [expd.dtype, alpha.dtype],
        inputs=[density, shift, intercal, npts, expd, alpha],
        cuda_header=cudaheader,
        cuda_src="""
        __global__ void raw2alpha_cuda_kernel(@ARGS_DEF) {
          @PRECALC
          const int i_pt = blockIdx.x * blockDim.x + threadIdx.x;
          if(i_pt<@in3(0)) {
            const scalar_t e = exp(@in0(i_pt) + @in1(0)); // can be inf
            @in4(i_pt) = e;
            @in5(i_pt) = 1 - pow(1 + e, -@in2(0));
          }
        }
        
        int threads = 256;
        int blocks = (@in3(0)+ threads - 1) / threads;
        
        raw2alpha_cuda_kernel<float><<<blocks, threads>>>(@ARGS);
        
        @out0 = @in4;
        @out1 = @in5;
        """
    )

    return expd, alpha

def raw2alpha_backward_cuda(
        expd: jt.Var,
        gradback: jt.Var,
        interval: float
) -> jt.Var :
    npts = jt.array(jt.size(expd)[0])
    grad = jt.zeros_like(expd)
    if npts == 0:
        return grad

    threads = THREADS
    blocks = (npts + threads - 1) / threads

    grad = jt.code(
        inputs=[expd, gradback, interval, npts, grad],
        cuda_header=cudaheader,
        cuda_src="""
        __global__ void raw2alpha_backward_cuda_kernel(@ARGS_DEF) {
          @PRECALC
          const int i_pt = blockIdx.x * blockDim.x + threadIdx.x;
          if(i_pt<@in3(0)) {
            @in4(i_pt) = min(@in0(i_pt), 1e10) * pow(1+@in0(i_pt), -@in2(0)-1) * @in2(0) * @in1(i_pt);
          }
        }

        int threads = 256;
        int blocks = (@in3(0) + threads - 1) / threads;
        
        raw2alpha_backward_cuda_kernel<<<blocks, threads>>>(@ARGS);
        
        @out0 = @in3;
        """
    )

    return grad

def alpha2weight_cuda(
        alpha: jt.Var,
        rayid: jt.Var,
        nrays: int
) -> Tuple[jt.Var, jt.Var, jt.Var, jt.Var, jt.var]:
    npts = jt.size(alpha)[0]
    nrays = jt.array(nrays)


    weight = jt.zeros_like(alpha)
    T = jt.ones_like(alpha)

    alphainvlast = jt.ones(nrays, dtype=alpha.dtype())
    istart = jt.zeros(nrays, dtype="int64")
    iend = jt.zeros(nrays, dtype="int64")
    if npts == 0:
        return weight, T, alphainvlast, istart, iend

    istart, iend = jt.code(
        [istart.shape, iend.shape],
        [istart.dtype, iend.dtype],
        inputs=[rayid, npts, istart, iend],
        cuda_header=cudaheader,
        cuda_src="""
        __global__ void __set_i_for_segment_start_end(@ARGS_DEF) {
          @PRECALC   
          const int index = blockIdx.x * blockDim.x + threadIdx.x;
          if(0<index && index<@in1(0) && @in0(index)!=@in0(index-1)) {
            @in2(@in0(index)) = index;
            @in2(@in0(index-1)) = index;
          }
        }
        
        int threads = 256;
        int blocks = (@in2(0) + threads - 1) / threads;
        
        __set_i_for_segment_start_end<<<blocks, threads>>>(@ARGS);
        
        @out0 = @in2;
        @out1 = @in3;
        """
    )

    iend[rayid[npts-1]] = npts

    weight, T, alphainvlast, iend = jt.code(
        [weight.shape, T.shape, alphainvlast.shape, iend.shape],
        [weight.dtype, T.dtype, alphainvlast.dtype, iend.dtype],
        inputs=[alpha, nrays, weight, T, alphainvlast, istart, iend],
        cuda_header=cudaheader,
        cuda_src= """
        __global__ void alpha2weight_cuda_kernel(@ARGS_DEF) {
          @PRECALC
          const int i_ray = blockIdx.x * blockDim.x + threadIdx.x;
          if(i_ray<@in1(0)) {
            const int i_s = @in5(i_ray);
            const int i_e_max = @in6(i_ray);
        
            float T_cum = 1.;
            int i;
            for(i=i_s; i<i_e_max; ++i) {
              @in3(i) = T_cum;
              @in2(i) = T_cum * @in0(i);
              T_cum *= (1. - @in0(i));
              if(T_cum<1e-3) {
                i+=1;
                break;
              }
            }
            @in6(i_ray) = i;
            @in4(i_ray) = T_cum;
          }
        }
        
        int threads = 256;
        int blocks = (@in1(0)  + threads - 1) / threads;
        
        alpha2weight_cuda_kernel<<<blocks, threads>>>(@ARGS);
        
        @out0 = @in2;
        @out1 = @in3;
        @out2 = @in4;
        @out3 = @in6;
        """
    )

    return weight, T, alphainvlast, istart, iend

def alpha2weight_backward_cuda(
        alpha: jt.Var,
        weight: jt.Var,
        T: jt.Var,
        alphainvlast: jt.Var,
        istart: jt.Var,
        iend: jt.Var,
        nrays: int,
        gradweights: jt.Var,
        gradlast: jt.Var
) -> jt.Var:
    grad = jt.zeros_like(alpha)
    if nrays == 0:
        return grad

    nrays = jt.array(nrays)

    grad = jt.code(
        grad.shape,
        grad.dtype,
        inputs=[alpha, weight, T, alphainvlast, istart, iend, nrays, gradweights, gradlast, grad],
        cuda_header=cudaheader,
        cuda_src="""     
        __global__ void alpha2weight_backward_cuda_kernel(@ARGS_DEF) {
          @PRECALC
        
          const int i_ray = blockIdx.x * blockDim.x + threadIdx.x;
          if(i_ray<@in6(0)) {
            const int i_s = @in4(i_ray);
            const int i_e = @in5(i_ray);
        
            float back_cum = @in8(i_ray) * @in3(i_ray);
            for(int i=i_e-1; i>=i_s; --i) {
              @in9(i) = @in7(i) * @in2(i) - back_cum / (1-@in0(i) + 1e-10);
              back_cum += @in7(i) * @in2(i);
            }
          }
        }
        
        int threads = 256;
        int blocks = (@in6(0) + threads - 1) / threads;
        
        alpha2weight_backward_cuda_kernel<<<blocks, threads>>>(@ARGS);
        
        @out0 = @in9;
        """
    )

    return grad