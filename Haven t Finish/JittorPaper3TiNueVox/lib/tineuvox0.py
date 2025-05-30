import functools
import math
import os
import time
from tkinter import W

import numpy as np
#import torch
import jittor as jt
#import torch.nn as nn
#import torch.nn.functional as F
import jittor.nn as nn
from torch.utils.cpp_extension import load
from outerjittor.jt_scatter import segment_coo
import outerjittor.weight_normal as wn

from lib.cuda import render_utils_kernel as render_utils_cuda
from lib.cuda import total_variation_kernel as total_variation_cuda

#parent_dir = os.path.dirname(os.path.abspath(__file__))
# render_utils_cuda = load(
#         name='render_utils_cuda',
#         sources=[
#             os.path.join(parent_dir, path)
#             for path in ['cuda/render_utils.cpp', 'cuda/render_utils_kernel.cu']],
#         verbose=True)
#
# total_variation_cuda = load(
#         name='total_variation_cuda',
#         sources=[
#             os.path.join(parent_dir, path)
#             for path in ['cuda/total_variation.cpp', 'cuda/total_variation_kernel.cu']],
#         verbose=True)

class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_views=3, input_ch_time=9, skips=[],):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_time = input_ch_time
        self.skips = skips
        self._time, self._time_out = self.create_net()

    def create_net(self):
        layers = [nn.Linear(self.input_ch + self.input_ch_time, self.W)]
        for i in range(self.D - 2):
            layer = nn.Linear
            in_channels = self.W
            if i in self.skips:
                in_channels += self.input_ch
            layers += [layer(in_channels, self.W)]
        return nn.ModuleList(layers), nn.Linear(self.W, 3)

    def query_time(self, new_pts, t, net, net_final):
        h = jt.concat([new_pts, t], dim=-1)
        for i, l in enumerate(net):
            h = net[i](h)
            h = nn.relu(h)
            if i in self.skips:
                h = jt.concat([new_pts, h], -1)
        return net_final(h)

    def execute(self, input_pts, ts):
        dx = self.query_time(input_pts, ts, self._time, self._time_out)
        input_pts_orig = input_pts[:, :3]
        out=input_pts_orig + dx
        return out

# Model
class RGBNet(nn.Module):
    def __init__(self, D=3, W=256, h_ch=256, views_ch=33, pts_ch=27, times_ch=17, output_ch=3):
        """ 
        """
        super(RGBNet, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = h_ch
        self.input_ch_views = views_ch
        self.input_ch_pts = pts_ch
        self.input_ch_times = times_ch
        self.output_ch = output_ch
        self.feature_linears = nn.Linear(self.input_ch, W)
        self.views_linears = nn.Sequential(nn.Linear(W, W//2),nn.ReLU(),nn.Linear(W//2, self.output_ch))
        
    def execute(self, input_h, input_views):
        feature = self.feature_linears(input_h)
        # feature_views = torch.cat([feature, input_views],dim=-1)
        outputs = self.views_linears(feature)
        return outputs
    
# class RGBNet(nn.Module):
#     def __init__(self, D=3, W=256, h_ch=256, views_ch=33, pts_ch=27, times_ch=17, output_ch=3):
#         """ 
#         """
#         super(RGBNet, self).__init__()
#         self.D = D
#         self.W = W
#         self.input_ch = h_ch
#         self.input_ch_views = views_ch
#         self.input_ch_pts = pts_ch
#         self.input_ch_times = times_ch
#         self.output_ch = output_ch
#         self.feature_linears = nn.Linear(self.input_ch, W)
#         self.views_linears = nn.Sequential(nn.Linear(W+self.input_ch_views, W//2),nn.ReLU(),nn.Linear(W//2, self.output_ch))
        
#     def forward(self, input_h, input_views):
#         feature = self.feature_linears(input_h)
#         feature_views = torch.cat([feature, input_views],dim=-1)
#         outputs = self.views_linears(feature_views)
#         return outputs

'''Model'''
class TiNeuVox(nn.Module):
    def __init__(self, xyz_min, xyz_max,
                 num_voxels=0, num_voxels_base=0, add_cam=False,
                 alpha_init=None, fast_color_thres=0,
                 voxel_dim=0, defor_depth=3, net_width=128,
                 posbase_pe=10, viewbase_pe=4, timebase_pe=8, gridbase_pe=2,
                 **kwargs):
        
        super(TiNeuVox, self).__init__()
        self.add_cam = add_cam
        self.voxel_dim = voxel_dim
        self.defor_depth = defor_depth
        self.net_width = net_width
        self.posbase_pe = posbase_pe
        self.viewbase_pe = viewbase_pe
        self.timebase_pe = timebase_pe
        self.gridbase_pe = gridbase_pe
        times_ch = 2*timebase_pe+1
        views_ch = 3+3*viewbase_pe*2
        pts_ch = 3+3*posbase_pe*2,
        self.xyz_min = jt.array(xyz_min)
        self.xyz_max = jt.array(xyz_max)
        self.fast_color_thres = fast_color_thres
        # determine based grid resolution
        self.num_voxels_base = num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1/3)

        # determine the density bias shift
        self.alpha_init = alpha_init
        self.act_shift = np.log(1/(1-alpha_init) - 1)
        print('TiNeuVox: set density bias shift to', self.act_shift)

        timenet_width = net_width
        timenet_depth = 1
        timenet_output = voxel_dim+voxel_dim*2*gridbase_pe
        self.timenet = nn.Sequential(
        nn.Linear(times_ch, timenet_width), nn.ReLU(inplace=True),
        nn.Linear(timenet_width, timenet_output))
        if self.add_cam == True:
            views_ch = 3+3*viewbase_pe*2+timenet_output
            self.camnet = nn.Sequential(
            nn.Linear(times_ch, timenet_width), nn.ReLU(inplace=True),
            nn.Linear(timenet_width, timenet_output))
            print('TiNeuVox: camnet', self.camnet)

        featurenet_width = net_width
        featurenet_depth = 1
        grid_dim = voxel_dim*3+voxel_dim*3*2*gridbase_pe
        input_dim = grid_dim+timenet_output+0+0+3+3*posbase_pe*2
        self.featurenet = nn.Sequential(
            nn.Linear(input_dim, featurenet_width), nn.ReLU(inplace=True),
            *[
                nn.Sequential(nn.Linear(featurenet_width, featurenet_width), nn.ReLU(inplace=True))
                for _ in range(featurenet_depth-1)
            ],
            )
        self.featurenet_width = featurenet_width
        self._set_grid_resolution(num_voxels)
        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=3+3*posbase_pe*2, input_ch_time=timenet_output)
        input_dim = featurenet_width
        self.densitynet = nn.Linear(input_dim, 1)

        self.time_poc =jt.array([(2**i) for i in range(timebase_pe)])
        self.grid_poc = jt.array([(2**i) for i in range(gridbase_pe)])
        self.pos_poc = jt.array([(2**i) for i in range(posbase_pe)])
        self.view_poc = jt.array([(2**i) for i in range(viewbase_pe)])

        self.voxel_dim = voxel_dim
        self.feature= nn.Parameter(jt.zeros([1, self.voxel_dim, *self.world_size],dtype="float"))
        self.rgbnet = RGBNet(W=net_width, h_ch=featurenet_width, views_ch=views_ch, pts_ch=pts_ch, times_ch=times_ch)

        print('TiNeuVox: feature voxel grid', self.feature.shape)
        print('TiNeuVox: timenet mlp', self.timenet)
        print('TiNeuVox: deformation_net mlp', self.deformation_net)
        print('TiNeuVox: densitynet mlp', self.densitynet)
        print('TiNeuVox: featurenet mlp', self.featurenet)
        print('TiNeuVox: rgbnet mlp', self.rgbnet)


    def _set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1/3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        print('TiNeuVox: voxel_size      ', self.voxel_size)
        print('TiNeuVox: world_size      ', self.world_size)
        print('TiNeuVox: voxel_size_base ', self.voxel_size_base)
        print('TiNeuVox: voxel_size_ratio', self.voxel_size_ratio)


    def get_kwargs(self):
        return {
            'xyz_min': self.xyz_min.numpy(),
            'xyz_max': self.xyz_max.numpy(),
            'num_voxels': self.num_voxels,
            'num_voxels_base': self.num_voxels_base,
            'alpha_init': self.alpha_init,
            'act_shift': self.act_shift,
            'voxel_size_ratio': self.voxel_size_ratio,
            'fast_color_thres': self.fast_color_thres,
            'voxel_dim':self.voxel_dim,
            'defor_depth':self.defor_depth,
            'net_width':self.net_width,
            'posbase_pe':self.posbase_pe,
            'viewbase_pe':self.viewbase_pe,
            'timebase_pe':self.timebase_pe,
            'gridbase_pe':self.gridbase_pe,
            'add_cam': self.add_cam,
        }


    @jt.no_grad()
    def scale_volume_grid(self, num_voxels):
        print('TiNeuVox: scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels)
        print('TiNeuVox: scale_volume_grid scale world_size from', ori_world_size, 'to', self.world_size)
        self.feature = nn.Parameter(
            nn.interpolate(self.feature.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
 
    def feature_total_variation_add_grad(self, weight, dense_mode):
        weight = weight * self.world_size.max() / 128
        total_variation_cuda.total_variation_add_grad(
            self.feature.float(), self.feature.grad.float(), weight, weight, weight, dense_mode)

    def grid_sampler(self, xyz, *grids, mode=None, align_corners=True):
        '''Wrapper for the interp operation'''
        mode = 'bilinear'
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        ret_lst = [
            nn.grid_sample(grid, ind_norm, mode=mode, align_corners=align_corners).reshape(grid.shape[1],-1).T.reshape(*shape,grid.shape[1])
            for grid in grids
        ]
        for i in range(len(grids)):
            if ret_lst[i].shape[-1] == 1:
                ret_lst[i] = ret_lst[i].squeeze(-1)
        if len(ret_lst) == 1:
            return ret_lst[0]
        return ret_lst


    def mult_dist_interp(self, ray_pts_delta):

        x_pad = math.ceil((self.feature.shape[2]-1)/4.0)*4-self.feature.shape[2]+1
        y_pad = math.ceil((self.feature.shape[3]-1)/4.0)*4-self.feature.shape[3]+1
        z_pad = math.ceil((self.feature.shape[4]-1)/4.0)*4-self.feature.shape[4]+1
        grid = nn.pad(self.feature.float(),(0,z_pad,0,y_pad,0,x_pad))
        # three 
        vox_l = self.grid_sampler(ray_pts_delta, grid)
        vox_m = self.grid_sampler(ray_pts_delta, grid[:,:,::2,::2,::2])
        vox_s = self.grid_sampler(ray_pts_delta, grid[:,:,::4,::4,::4])
        vox_feature = jt.concat((vox_l,vox_m,vox_s),-1)

        if len(vox_feature.shape)==1:
            vox_feature_flatten = vox_feature.unsqueeze(0)
        else:
            vox_feature_flatten = vox_feature
        
        return vox_feature_flatten

    def activate_density(self, density, interval=None): 
        interval = interval if interval is not None else self.voxel_size_ratio 
        return 1 - jt.exp(-nn.softplus(density + self.act_shift) * interval)

    def get_mask(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        '''Check whether the rays hit the geometry or not'''
        shape = rays_o.shape[:-1]
        rays_o = rays_o.reshape(-1, 3).contiguous()
        rays_d = rays_d.reshape(-1, 3).contiguous()
        stepdist = stepsize * self.voxel_size
        ray_pts, mask_outbbox, ray_id = render_utils_cuda.sample_pts_on_rays(
                rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)[:3]
        mask_inbbox = ~mask_outbbox
        hit = jt.zeros([len(rays_o)], dtype="bool")
        hit[ray_id[mask_inbbox]] = 1
        return hit.reshape(shape)

    def sample_ray(self, rays_o, rays_d, near, far, stepsize, is_train=False, **render_kwargs):
        '''Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            near, far:        the near and far distance of the rays.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        '''
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        stepdist = stepsize * self.voxel_size
        ray_pts, mask_outbbox, ray_id, step_id, N_steps, t_min, t_max = render_utils_cuda.sample_pts_on_rays(
            rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)
        mask_inbbox = ~mask_outbbox
        ray_pts = ray_pts[mask_inbbox]
        ray_id = ray_id[mask_inbbox]
        step_id = step_id[mask_inbbox]
        return ray_pts, ray_id, step_id,mask_inbbox
    
    def density_func(self, ray_pts, times_sel, stepsize):
        times_emb = poc_fre(times_sel, self.time_poc)
        times_feature = self.timenet(times_emb)

        if self.add_cam==True:
            cam_emb= poc_fre(cam_sel, self.time_poc)
            cams_feature=self.camnet(cam_emb)
        # sample points on rays
        interval = stepsize * self.voxel_size_ratio
        self.ray_pts = ray_pts
        # pts deformation 
        rays_pts_emb = poc_fre(ray_pts, self.pos_poc)
        
        ray_pts_delta = self.deformation_net(rays_pts_emb, times_feature)
        self.ray_pts_delta = ray_pts_delta
        
        # voxel query interp
        vox_feature_flatten=self.mult_dist_interp(ray_pts_delta)

        vox_feature_flatten_emb = poc_fre(vox_feature_flatten, self.grid_poc)
        
        h_feature = self.featurenet(jt.concat((vox_feature_flatten_emb, rays_pts_emb, times_feature), -1))
        
        density_result = self.densitynet(h_feature)
        density_result = nn.softplus(density_result + self.act_shift)
        
        return density_result
    
    def extract_density(self, test_times, stepsize, testsavedird):
        N = 64
        resolution = 256
        
        bound_min = self.xyz_min
        bound_max = self.xyz_max
        X = jt.linspace(bound_min[0], bound_max[0], resolution).split(N)
        Y = jt.linspace(bound_min[1], bound_max[1], resolution).split(N)
        Z = jt.linspace(bound_min[2], bound_max[2], resolution).split(N)
        
        with jt.no_grad():
            for idx in range(len(test_times)):
                print(idx)
                u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    
                for xi, xs in enumerate(X):
                    for yi, ys in enumerate(Y):
                        for zi, zs in enumerate(Z):
                            xx, yy, zz = jt.meshgrid(xs, ys, zs)
                            pts = jt.concat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                            time_one = test_times[idx]*jt.ones_like(pts[:,0:1])
                            den = self.density_func(pts,time_one,stepsize).reshape(len(xs), len(ys), len(zs)).float().detach().numpy()
                            u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = den
                            
                np.savez_compressed(testsavedird+'/%04d.npz'%idx, dmap = u)
                
    def extract_flow(self, test_times, stepsize, testsavedird):
        
        resolution = 128
        
        import vtk
        from vtk.numpy_interface import algorithms as algs
        from vtk.numpy_interface import dataset_adapter as dsa
        def point2vtk(vectors,filename):
            xs, ys, zs = np.meshgrid(np.arange(resolution), np.arange(resolution), np.arange(resolution))

            polydata = vtk.vtkPolyData()

            pts = vtk.vtkPoints()
            points = algs.make_vector(xs.ravel(),
                                    ys.ravel(),
                                    zs.ravel())
            pts.SetData(dsa.numpyTovtkDataArray(points, "Points"))

            polydata.SetPoints(pts)

            vectors = algs.make_vector(vectors[:,:,:,0].ravel(),
                                    vectors[:,:,:,1].ravel(),
                                    vectors[:,:,:,2].ravel())
            polydata.GetPointData().SetScalars(dsa.numpyTovtkDataArray(vectors, "Velocity"))
            writer = vtk.vtkPolyDataWriter()
            writer.SetFileName(filename)
            writer.SetInputData(polydata)
            writer.Update()
        
        # import pandas as pd 
        # from pyntcloud as pc
        # def point2ply(pts,vels,filename):
        #     data = {'x':pts[:,0],
        #             'y':pts[:,1],
        #             'z':pts[:,2],
        #             'u':vels[:,0],
        #             'v':vels[:,1],
        #             'w':vels[:,2]
        #            }
        #     cloud = pc(pd.DataFrame(data))
        #     cloud.to_file(filename)          
        N = 64  
        bound_min = self.xyz_min
        bound_max = self.xyz_max
        m_bound = 0.5 * (bound_min + bound_max)
        X = jt.linspace(bound_min[0]*0.5, bound_max[0]*0.5, resolution).split(N)
        Y = jt.linspace(bound_min[1], bound_max[1], resolution).split(N)
        Z = jt.linspace(bound_min[2], bound_max[2], resolution).split(N)
        
        dt = jt.ones([resolution, resolution, resolution, 1])
        dx = (bound_max[0] - bound_min[0]) / (resolution-1)
        dx = dx*dt
        dy = (bound_max[1] - bound_min[1]) / (resolution-1)
        dy = dy*dt
        dz = (bound_max[2] - bound_min[2]) / (resolution-1)
        dz = dz*dt
        ds = jt.concat([dx,dy,dz], dim=-1).float().detach().numpy()
        
        with jt.no_grad():
            for idx in range(len(test_times)):
                if idx < 100:
                    continue
                
                u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
                v = np.zeros([resolution, resolution, resolution, 3], dtype=np.float32)
                for xi, xs in enumerate(X):
                    for yi, ys in enumerate(Y):
                        for zi, zs in enumerate(Z):
                            xx, yy, zz = jt.meshgrid(xs, ys, zs)
                            pts = jt.concat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                            
                            time_one = test_times[idx]*jt.ones_like(pts[:,0:1])
                            den = self.density_func(pts,time_one,stepsize).reshape(len(xs), len(ys), len(zs)).float().detach().numpy()
                            u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = den
                            
                            times_emb = poc_fre(time_one, self.time_poc)
                            times_feature = self.timenet(times_emb)
                            rays_pts_emb = poc_fre(pts, self.pos_poc)
                            
                            ray_pts_delta = self.deformation_net(rays_pts_emb, times_feature)
                            vel = (ray_pts_delta - rays_pts_emb[...,:3]).reshape(len(xs), len(ys), len(zs), 3)
   
                            vel = vel / jt.norm(vel, p=2, dim=-1, keepdim=True)
                            v[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = vel.float().detach().numpy()
                
                print(u.max(),u.min())
                mask = (u>0.15).astype(np.float32)[...,None]
                v = v*mask
                out_file = os.path.join(testsavedird, '{:03d}.vtk'.format(idx))            
                point2vtk(v,out_file)
                exit(0)

    def execute(self, rays_o, rays_d, viewdirs,times_sel, cam_sel=None,bg_points_sel=None,global_step=None, **render_kwargs):
        '''Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        '''
        assert len(rays_o.shape)==2 and rays_o.shape[-1]==3, 'Only suuport point queries in [N, 3] format'

        ret_dict = {}
        N = len(rays_o)
        times_emb = poc_fre(times_sel, self.time_poc)
        viewdirs_emb = poc_fre(viewdirs, self.view_poc)
        times_feature = self.timenet(times_emb)
        if self.add_cam==True:
            cam_emb= poc_fre(cam_sel, self.time_poc)
            cams_feature=self.camnet(cam_emb)
        # sample points on rays
        ray_pts, ray_id, step_id, mask_inbbox= self.sample_ray(
                rays_o=rays_o, rays_d=rays_d, is_train=global_step is not None, **render_kwargs)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio

        # bound_min = self.xyz_min
        # bound_max = self.xyz_max
        # print(bound_min,bound_max)
        # exit(0)
        # pts deformation 
        rays_pts_emb = poc_fre(ray_pts, self.pos_poc)
        ray_pts_delta = self.deformation_net(rays_pts_emb, times_feature[ray_id])
        # computer bg_points_delta
        if bg_points_sel is not None:
            bg_points_sel_emb = poc_fre(bg_points_sel, self.pos_poc)
            bg_points_sel_delta = self.deformation_net(bg_points_sel_emb, times_feature[:(bg_points_sel_emb.shape[0])])
            ret_dict.update({'bg_points_delta': bg_points_sel_delta})
        # voxel query interp
        vox_feature_flatten=self.mult_dist_interp(ray_pts_delta)

        times_feature = times_feature[ray_id]
        vox_feature_flatten_emb = poc_fre(vox_feature_flatten, self.grid_poc)
        h_feature = self.featurenet(jt.concat((vox_feature_flatten_emb, rays_pts_emb, times_feature), -1))
        density_result = self.densitynet(h_feature)

        # alpha = nn.Softplus()(density_result+self.act_shift)
        alpha = self.activate_density(density_result,interval)
        alpha=alpha.squeeze(-1)
        if self.fast_color_thres > 0:
            mask = (alpha > self.fast_color_thres)
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            alpha = alpha[mask]
            h_feature=h_feature[mask]

        # compute accumulated transmittance
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0:
            mask = (weights > self.fast_color_thres)
            weights = weights[mask]
            alpha = alpha[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            h_feature=h_feature[mask]

        viewdirs_emb_reshape = viewdirs_emb[ray_id]
        if self.add_cam == True:
            viewdirs_emb_reshape=jt.concat((viewdirs_emb_reshape, cams_feature[ray_id]), -1)
        rgb_logit = self.rgbnet(h_feature, viewdirs_emb_reshape)
        rgb = jt.sigmoid(rgb_logit)

        # Ray marching
        rgb_marched = segment_coo(
                src=(weights.unsqueeze(-1) * rgb),
                index=ray_id,
                out=jt.zeros([N, 3]),
                reduce='sum')
        rgb_marched += (alphainv_last.unsqueeze(-1) * render_kwargs['bg'])
        acc = segment_coo(
                src=(weights.unsqueeze(-1)),
                index=ray_id,
                out=jt.zeros([N, 1]),
                reduce='sum')
        ret_dict.update({
            'alphainv_last': alphainv_last,
            'weights': weights,
            'acc': acc,
            'rgb_marched': rgb_marched,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            'ray_id': ray_id,
            'pts_o': ray_pts,
            'pts_delta': ray_pts_delta,
        })

        with jt.no_grad():
            depth = segment_coo(
                    src=(weights * step_id),
                    index=ray_id,
                    out=jt.zeros([N]),
                    reduce='sum')
        ret_dict.update({'depth': depth})
        
        return ret_dict


class Alphas2Weights(jt.Function):
    @staticmethod
    def execute(ctx, alpha, ray_id, N):
        weights, T, alphainv_last, i_start, i_end = render_utils_cuda.alpha2weight(alpha, ray_id, N)
        if alpha.requires_grad:
            ctx.save_for_backward(alpha, weights, T, alphainv_last, i_start, i_end)
            ctx.n_rays = N
        return weights, alphainv_last

    @staticmethod
    def grad(ctx, grad_weights, grad_last):
        alpha, weights, T, alphainv_last, i_start, i_end = ctx.saved_tensors
        grad = render_utils_cuda.alpha2weight_backward(
                alpha, weights, T, alphainv_last,
                i_start, i_end, ctx.n_rays, grad_weights, grad_last)
        return grad, None, None


''' Ray and batch
'''
def get_rays(H, W, K, c2w, inverse_y=False, flip_x=False ,flip_y=False, mode='center'):
    i, j = jt.meshgrid(
        jt.linspace(0, W-1, W),
        jt.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()
    if mode == 'lefttop':
        pass
    elif mode == 'center':
        i, j = i+0.5, j+0.5
    elif mode == 'random':
        i = i+jt.rand_like(i)
        j = j+jt.rand_like(j)
    else:
        raise NotImplementedError

    if flip_x:
        i = i.flip((1,))
    if flip_y:
        j = j.flip((0,))
    if inverse_y:
        dirs = jt.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], jt.ones_like(i)], -1)
    else:
        dirs = jt.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -jt.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = jt.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,3].expand(rays_d.shape)
    return rays_o, rays_d

def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,3], np.shape(rays_d))
    return rays_o, rays_d

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    rays_o = jt.stack([o0,o1,o2], -1)
    rays_d = jt.stack([d0,d1,d2], -1)

    return rays_o, rays_d

def get_rays_of_a_view(H, W, K, c2w, ndc, mode='center'):
    rays_o, rays_d = get_rays(H, W, K, c2w, mode=mode)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
    return rays_o, rays_d, viewdirs

@jt.no_grad()
def get_training_rays(rgb_tr, times,train_poses, HW, Ks, ndc):
    print('get_training_rays: start')
    assert len(np.unique(HW, axis=0)) == 1
    assert len(np.unique(Ks.reshape(len(Ks),-1), axis=0)) == 1
    assert len(rgb_tr) == len(train_poses) and len(rgb_tr) == len(Ks) and len(rgb_tr) == len(HW)
    H, W = HW[0]
    K = Ks[0]
    eps_time = time.time()
    rays_o_tr = jt.zeros([len(rgb_tr), H, W, 3])
    rays_d_tr = jt.zeros([len(rgb_tr), H, W, 3])
    viewdirs_tr = jt.zeros([len(rgb_tr), H, W, 3])
    times_tr = jt.ones([len(rgb_tr), H, W, 1])

    imsz = [1] * len(rgb_tr)
    for i, c2w in enumerate(train_poses):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,)
        rays_o_tr[i].copy_(rays_o)
        rays_d_tr[i].copy_(rays_d)
        viewdirs_tr[i].copy_(viewdirs)
        times_tr[i] = times_tr[i]*times[i]
        del rays_o, rays_d, viewdirs
    eps_time = time.time() - eps_time
    print('get_training_rays: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, times_tr,rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@jt.no_grad()
def get_training_rays_flatten(rgb_tr_ori, times,train_poses, HW, Ks, ndc):
    print('get_training_rays_flatten: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    eps_time = time.time()
    #DEVICE = rgb_tr_ori[0].device
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = jt.zeros([N,3])
    rays_o_tr = jt.zeros_like(rgb_tr)
    rays_d_tr = jt.zeros_like(rgb_tr)
    viewdirs_tr = jt.zeros_like(rgb_tr)
    times_tr=jt.ones([N,1])
    times=times.unsqueeze(-1)
    imsz = []
    top = 0
    for c2w, img, (H, W), K ,time_one in zip(train_poses, rgb_tr_ori, HW, Ks,times):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,)
        n = H * W
        times_tr[top:top+n]=times_tr[top:top+n]*time_one
        rgb_tr[top:top+n].copy_(img.flatten(0,1))
        rays_o_tr[top:top+n].copy_(rays_o.flatten(0,1))
        rays_d_tr[top:top+n].copy_(rays_d.flatten(0,1))
        viewdirs_tr[top:top+n].copy_(viewdirs.flatten(0,1))
        imsz.append(n)
        top += n
    assert top == N
    eps_time = time.time() - eps_time
    print('get_training_rays_flatten: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, times_tr,rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@jt.no_grad()
def get_training_rays_in_maskcache_sampling(rgb_tr_ori, times,train_poses, HW, Ks, ndc, model, render_kwargs):
    print('get_training_rays_in_maskcache_sampling: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    CHUNK = 64
    #DEVICE = rgb_tr_ori[0].device
    eps_time = time.time()
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = jt.zeros([N,3])
    rays_o_tr = jt.zeros_like(rgb_tr)
    rays_d_tr = jt.zeros_like(rgb_tr)
    viewdirs_tr = jt.zeros_like(rgb_tr)
    times_tr = jt.ones([N,1])
    times = times.unsqueeze(-1)
    imsz = []
    top = 0
    for c2w, img, (H, W), K ,time_one in zip(train_poses, rgb_tr_ori, HW, Ks,times):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,)
        mask = jt.empty(img.shape[:2], dtype="bool")
        for i in range(0, img.shape[0], CHUNK):
            mask[i:i+CHUNK] = model.get_mask(
                    rays_o=rays_o[i:i+CHUNK], rays_d=rays_d[i:i+CHUNK], **render_kwargs)
        n = mask.sum()
        times_tr[top:top+n]=times_tr[top:top+n]*time_one
        rgb_tr[top:top+n].copy_(img[mask])
        rays_o_tr[top:top+n].copy_(rays_o[mask])
        rays_d_tr[top:top+n].copy_(rays_d[mask])
        viewdirs_tr[top:top+n].copy_(viewdirs[mask])
        imsz.append(n)
        top += n

    print('get_training_rays_in_maskcache_sampling: ratio', top / N)
    rgb_tr = rgb_tr[:top]
    rays_o_tr = rays_o_tr[:top]
    rays_d_tr = rays_d_tr[:top]
    viewdirs_tr = viewdirs_tr[:top]
    eps_time = time.time() - eps_time
    print('get_training_rays_in_maskcache_sampling: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, times_tr,rays_o_tr, rays_d_tr, viewdirs_tr, imsz


def batch_indices_generator(N, BS):
    # torch.randperm on cuda produce incorrect results in my machine
    idx, top = jt.array(np.random.permutation(N), dtype="long"), 0
    while True:
        if top + BS > N:
            idx, top = jt.array(np.random.permutation(N), dtype="long"), 0
        yield idx[top:top+BS]
        top += BS

def poc_fre(input_data,poc_buf):

    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = jt.concat([input_data, input_data_sin,input_data_cos], -1)
    return input_data_emb