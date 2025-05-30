import jittor as jt
import jittor.nn as F
import numpy as np
import logging
import mcubes
from icecream import ic
from outerjittor.sqz import Osqueeze

def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = jt.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = jt.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = jt.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with jt.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = jt.meshgrid(xs, ys, zs)
                    pts = jt.concat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / jt.sum(weights, -1, keepdims=True)
    cdf = jt.cumsum(pdf, -1)
    cdf = jt.concat([jt.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = jt.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = jt.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    u = u.contiguous()
    inds = jt.searchsorted(cdf, u, right=True)
    below = jt.maximum(jt.zeros_like(inds - 1), inds - 1)
    above = jt.minimum((cdf.shape[-1] - 1) * jt.ones_like(inds), inds)
    inds_g = jt.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = jt.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = jt.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = jt.where(denom < 1e-5, jt.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

class NeuSRenderer:
    def __init__(self,
                 nerf,
                 sdf_network,
                 deviation_network,
                 color_network,
                 pts_bias,
                 n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 perturb):
        self.nerf = nerf
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.pts_bias = pts_bias
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb

    def render_core_outside(self, rays_o, rays_d, z_vals, sample_dist, nerf, background_rgb=None):
        """
        Render background
        """
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = jt.concat([dists, jt.array([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # batch_size, n_samples, 3

        dis_to_center = jt.safe_clip(jt.norm(pts, p=2, dim=-1, keepdim=True),1.0, 1e10)
        pts = jt.concat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)       # batch_size, n_samples, 4

        dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)

        pts = pts.reshape(-1, 3 + int(self.n_outside > 0))
        dirs = dirs.reshape(-1, 3)

        density, sampled_color = nerf(pts, dirs)
        alpha = 1.0 - jt.exp(-F.softplus(density.reshape(batch_size, n_samples)) * dists)
        alpha = alpha.reshape(batch_size, n_samples)
        weights = alpha * jt.cumprod(jt.concat([jt.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        color = (weights[:, :, None] * sampled_color).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

        return {
            'color': color,
            'sampled_color': sampled_color,
            'alpha': alpha,
            'weights': weights,
        }

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = jt.norm(pts, p=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = jt.concat([jt.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = jt.stack([prev_cos_val, cos_val], dim=-1)
        cos_val = jt.min(cos_val, dim=-1, keepdims=False)
        cos_val = jt.safe_clip(cos_val,-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = jt.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = jt.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * jt.cumprod(
            jt.concat([jt.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = jt.concat([z_vals, new_z_vals], dim=-1)
        index, z_vals = jt.argsort(z_vals, dim=-1)

        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = jt.concat([sdf, new_sdf], dim=-1)
            xx = jt.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    sdf_network,
                    deviation_network,
                    color_network,
                    pts_bias,
                    background_alpha=None,
                    background_sampled_color=None,
                    background_rgb=None,
                    cos_anneal_ratio=0.0):
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = jt.concat([dists, jt.array([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5
        #print(rays_d[:, None, :])

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)
        c_pts = rays_o[:, None, :].expand(pts.shape)
        #print(pts)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)
        c_pts = c_pts.reshape(-1, 3)
        #print(pts)

        sdf_nn_output = sdf_network(pts)
        #print(sdf_nn_output.attrs())
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]
        #print(sdf_network)
        gradients = Osqueeze(sdf_network.gradient(pts))


        inv_s = deviation_network(jt.zeros([1, 3]))[:, :1]
        inv_s = jt.safe_clip(inv_s, 1e-6, 1e6)# Single parameter
        inv_s = inv_s.expand(batch_size * n_samples, 1)
        
        true_cos = (dirs * gradients).sum(-1, keepdims=True)
        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive
        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5
        prev_cdf = jt.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = jt.sigmoid(estimated_next_sdf * inv_s)
        p = prev_cdf - next_cdf
        c = prev_cdf
        #print(sdf,iter_cos,dists.reshape(-1, 1) * 0.5)
        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples)
        alpha = jt.safe_clip(alpha, 0.0, 1.0)
        #print(alpha)
        
        ptsn = pts.reshape(batch_size, n_samples, 3)

        dirsn = dirs.reshape(batch_size, n_samples, 3)
        dm,nm,density = pts_bias(dirsn)

        nm2 = nm[:,None,:].expand(ptsn.shape).reshape(-1, 3)
        alpham = 1.0 - jt.exp(-F.softplus(density) * dists)
        alpham = jt.safe_clip(alpham.reshape(batch_size, n_samples),0.0, 1.0)

        #print(pts_bias)

        pts_norm = jt.norm(pts, p=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).float().detach()
        relax_inside_sphere = (pts_norm < 1.2).float().detach()
        sampled_color = color_network(pts, gradients, dirs, feature_vector).reshape(batch_size, n_samples, 3)

        A = nm[:,0].unsqueeze(-1)
        B = nm[:,1].unsqueeze(-1)
        C = nm[:,2].unsqueeze(-1)
        C = -jt.sigmoid(C)
        z_vals_p0 = dm[:,0].unsqueeze(-1)
        D = -z_vals_p0*C

        t0 = jt.zeros_like(A)
        t1 = jt.ones_like(A)
        M_r1 = [1.-2*A*A, -2*A*B, -2*A*C, -2*A*D]
        M_r2 = [-2*A*B, 1.-2*B*B, -2*B*C, -2*B*D]
        M_r3 = [-2*A*C, -2*B*C, 1-2*C*C, -2*C*D]
        M_r4 = [t0, t0, t0, t1]

        M_r1 = jt.concat(M_r1,1).unsqueeze(-2)
        M_r2 = jt.concat(M_r2,1).unsqueeze(-2)
        M_r3 = jt.concat(M_r3,1).unsqueeze(-2)
        M_r4 = jt.concat(M_r4,1).unsqueeze(-2)
        M_r = jt.concat([M_r1,M_r2,M_r3,M_r4],-2)

        pts2 = pts - c_pts
        z_vals_p = z_vals_p0.expand(mid_z_vals.shape) 
        maskz = (mid_z_vals>z_vals_p).float().detach()
        mid_z_vals2 = mid_z_vals * maskz
        mid_z_vals2_2 = mid_z_vals * (1.-maskz)
        
        pts2 = rays_d[:, None, :] * mid_z_vals2[..., :, None]
        pts2 = pts2.reshape(-1, 3)
        t1_2 = jt.ones_like(pts2[:,0]).unsqueeze(-1)
        pts2 = jt.concat([pts2,t1_2],-1)
        pts2 = pts2.reshape(batch_size, n_samples, 4)

        ptsm = jt.linalg.inv(M_r).permute(0,2,1) @ pts2.permute(0,2,1)
        ptsm = ptsm.permute(0,2,1) 
        ptsm = ptsm[...,:-1].reshape(-1, 3)
        
        pts2_2 = rays_d[:, None, :] * mid_z_vals2_2[..., :, None]
        pts2_2 = pts2_2.reshape(-1, 3)
        ptsm = ptsm + pts2_2 #optional
        
        gradientsm = nm2
        sampled_colorm = color_network(ptsm, gradientsm, dirs, feature_vector).reshape(batch_size, n_samples, 3)

        # Render with background
        if background_alpha is not None:
            alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
            alpha = jt.concat([alpha, background_alpha[:, n_samples:]], dim=-1)
            alpham = alpham * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
            alpham = jt.concat([alpham, background_alpha[:, n_samples:]], dim=-1)

            sampled_color = sampled_color * inside_sphere[:, :, None] +\
                            background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
            sampled_color = jt.concat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)
            sampled_colorm = sampled_colorm * inside_sphere[:, :, None] +\
                            background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
            sampled_colorm = jt.concat([sampled_colorm, background_sampled_color[:, n_samples:]], dim=1)

        weights = alpha * jt.cumprod(jt.concat([jt.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        #print(weights)
        color = (sampled_color * weights[:, :, None]).sum(dim=1)

        weightsm = alpham * jt.cumprod(jt.concat([jt.ones([batch_size, 1]), 1. - alpham + 1e-7], -1), -1)[:, :-1]
        color2 = (sampled_colorm * weightsm[:, :, None]).sum(dim=1)

        final_weight = 0.3
        colorm = color*final_weight + color2*(1.0-final_weight)
        #print(alpham)
        #print(1. - alpham + 1e-7)
        gradients = final_weight*gradients 

        # Eikonal loss
        gradient_error = (jt.norm(gradients.reshape(batch_size, n_samples, 3), p=2, dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

        gradient_error2 = (jt.norm(gradientsm.reshape(batch_size, n_samples, 3), p=2, dim=-1) - 1.0) ** 2
        gradient_error2 = (relax_inside_sphere * gradient_error2).sum() / (relax_inside_sphere.sum() + 1e-5)

        gradient_error = gradient_error + gradient_error2

        return {
            'color': color,
            'colorm': colorm,
            'gradientsm': gradientsm.reshape(batch_size, n_samples, 3),
            'zvals': D,
            'sdf': sdf,
            'dists': dists,
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'weightsm': weightsm,
            'cdf': c.reshape(batch_size, n_samples),
            'gradient_error': gradient_error,
            'inside_sphere': inside_sphere
        }

    def render(self, rays_o, rays_d, near, far, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0):
        batch_size = len(rays_o)
        sample_dist = 2.0 / self.n_samples   # Assuming the region of interest is a unit sphere
        z_vals = jt.linspace(0.0, 1.0, self.n_samples)
        z_vals = near + (far - near) * z_vals[None, :]

        z_vals_outside = None
        if self.n_outside > 0:
            z_vals_outside = jt.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside)

        n_samples = self.n_samples
        perturb = self.perturb

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (jt.rand([batch_size, 1]) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

            if self.n_outside > 0:
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = jt.concat([mids, z_vals_outside[..., -1:]], -1)
                lower = jt.concat([z_vals_outside[..., :1], mids], -1)
                t_rand = jt.rand([batch_size, z_vals_outside.shape[-1]])
                z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

        if self.n_outside > 0:
            z_vals_outside = far / jt.flip(z_vals_outside, dim=[-1]) + 1.0 / self.n_samples

        background_alpha = None
        background_sampled_color = None

        # Up sample
        if self.n_importance > 0:
            with jt.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, self.n_samples)

                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2**i)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  last=(i + 1 == self.up_sample_steps))

            n_samples = self.n_samples + self.n_importance

        # Background model
        if self.n_outside > 0:
            z_vals_feed = jt.concat([z_vals, z_vals_outside], dim=-1)
            _, z_vals_feed = jt.argsort(z_vals_feed, dim=-1)
            ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf)

            background_sampled_color = ret_outside['sampled_color']
            background_alpha = ret_outside['alpha']

        # Render core
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    self.pts_bias,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    cos_anneal_ratio=cos_anneal_ratio)

        color_fine = ret_fine['color'] 
        colorm_fine = ret_fine['colorm']
        plane_fine = ret_fine['gradientsm'] 
        z_fine = ret_fine['zvals'] 
        weights = ret_fine['weights']
        weightsm = ret_fine['weightsm']
        weights_sum = weights.sum(dim=-1, keepdims=True)
        gradients = ret_fine['gradients']
        s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdims=True)

        return {
            'color_fine': color_fine,
            'colorm_fine': colorm_fine,
            'gradientsm': plane_fine,
            'zvals': z_fine,
            's_val': s_val,
            'cdf_fine': ret_fine['cdf'],
            'sdf': ret_fine['sdf'],
            'weight_sum': weights_sum,
            'weight_max': jt.max(weights, dim=-1, keepdims=True)[0],
            'gradients': gradients,
            'weights': weights,
            'weightsm': weightsm,
            'gradient_error': ret_fine['gradient_error'],
            'inside_sphere': ret_fine['inside_sphere']
        }

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network.sdf(pts))
