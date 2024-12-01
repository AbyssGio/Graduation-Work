import sys

import jittor
from JittorPaper1NERF.outer_jittor import outerjittor as oj
from jittor import nn as jnn

import numpy as np
from JittorPaper1NERF.models.embedder import get_embedder

np.random.seed(0)
jittor.flags.use_cuda = 1


# 这是一种用来拟合物体表面场函数的神经网络，这个场用一个距离矩阵来表示平面内点到物体表面的距离
class SDFNetwork(jnn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(SDFNetwork, self).__init__()  # ? 用jnn.Module类来初始化自身

        # 记录维数矩阵，dims中包括了所有层数的输入维数
        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = jnn.Linear(dims[l], out_dim)

            # 这里用随机正态分布来填充神经网络初始权重，对特殊层做特殊处理（但jittor的正态分布填充的有点问题
            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        lin.weight = jittor.normal(mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001, size=lin.weight.shape)
                        jnn.init.constant_(lin.bias, -bias)
                    else:
                        lin.weight = jittor.normal(mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001, size=lin.weight.shape)
                        jnn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    jnn.init.constant_(lin.bias, 0.0)
                    jnn.init.constant_(lin.weight[:, 3:], 0.0)
                    temp = lin.weight[:, 3:]
                    lin.weight[:, 3:] = jittor.normal(mean=0.0, std=np.sqrt(2) / np.sqrt(out_dim), size=temp.shape)
                elif multires > 0 and l in self.skip_in:
                    jnn.init.constant_(lin.bias, 0.0)
                    lin.weight = jittor.normal(mean=0.0, std=np.sqrt(2) / np.sqrt(out_dim), size=lin.weight.shape)
                    # jnn.init.trunc_normal_(lin.weight, mean=0.0, std=np.sqrt(2) / np.sqrt(out_dim))
                    jnn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    jnn.init.constant_(lin.bias, 0.0)
                    lin.weight = jittor.normal(mean=0.0, std=np.sqrt(2) / np.sqrt(out_dim), size=lin.weight.shape)
                    # jnn.init.trunc_normal_(lin.weight, mean=0.0, std=np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = oj.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = jnn.Softplus(beta=100)

        # SDFNetwork Done Or lin.weight = jittor.normal() ?

    # 定义前向传播函数
    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = jittor.concat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        # return torch.cat([x[:, :1] / self.scale, x[:, 1:2] / self.scale, x[:, 2:]], dim=-1)
        return jittor.concat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x):
        return self.forward(x)[:, :1]

    def sdfM(self, x):
        return self.forward(x)[:, 1:2]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    # 定义求梯度函数
    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        # d_output = jittor.ones_like(y)
        # d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        # TODO: 有些出入,文档里有retain_graph参数，这里却没有
        gradients = jittor.grad(y, x, retain_graph=True)
        '''
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        '''
        return gradients.unsqueeze(1)

    def gradientM(self, x):
        x.requires_grad_(True)
        y = self.sdfM(x)
        # d_output = jittor.ones_like(y)
        # d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        # TODO: 有些出入,文档里有retain_graph参数，这里却没有
        gradients = jittor.grad(y, x, retain_graph=True)
        '''
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        '''
        return gradients.unsqueeze(1)


# 渲染神经网络
class RenderingNetwork(jnn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = jnn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = oj.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = jnn.ReLU()

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None

        if self.mode == 'idr':
            rendering_input = jittor.concat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = jittor.concat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = jittor.concat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = jittor.sigmoid(x)
        return x


# pts_fea是input_ch维输入，三个全连接层，每层256维输入，1维输出的网络架构
# dm是128维输入，1维输出，nm是128维输入，3维输出
class Pts_Bias(jnn.Module):
    def __init__(self, d_hidden=256, multires=4, d_in=3):
        super(Pts_Bias, self).__init__()
        embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
        self.embed_fn_fine = embed_fn
        self.pts_fea = jnn.Sequential(jnn.Linear(input_ch, d_hidden),
                                      jnn.ReLU(inplace=True),
                                      jnn.Linear(d_hidden, d_hidden),
                                      jnn.ReLU(inplace=True),
                                      jnn.Linear(d_hidden, 1),
                                      jnn.ReLU(inplace=True))

        self.dm = jnn.Sequential(jnn.Linear(128, d_hidden),
                                 jnn.ReLU(inplace=True),
                                 jnn.Linear(d_hidden, 1),
                                 jnn.Sigmoid())

        self.nm = jnn.Sequential(jnn.Linear(128, d_hidden),
                                 jnn.ReLU(inplace=True),
                                 jnn.Linear(d_hidden, 3),
                                 jnn.Tanh())

    def forward(self, x):
        x = self.embed_fn_fine(x)
        pts_bias = self.pts_fea(x)
        pts_fea = pts_bias.squeeze(-1)
        dm = self.dm(pts_fea)
        nm = self.nm(pts_fea)
        return dm, nm, pts_fea


class NeRF(jnn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 d_in=3,
                 d_in_view=3,
                 multires=0,
                 multires_view=0,
                 output_ch=4,
                 skips=[4],  # ？？怎么没初值
                 use_viewdirs=False):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = jnn.ModuleList(
            [jnn.Linear(self.input_ch, W)] +
            [jnn.Linear(W, W) if i not in self.skips else jnn.Linear(W + self.input_ch, W) for i in range(D - 1)])

        ### Implementation according to the official code release
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = jnn.ModuleList([jnn.Linear(self.input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = jnn.Linear(W, W)
            self.alpha_linear = jnn.Linear(W, 1)
            self.rgb_linear = jnn.Linear(W // 2, 3)
        else:
            self.output_linear = jnn.Linear(W, output_ch)

    def forward(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = jnn.relu(h)
            if i in self.skips:
                h = jittor.concat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = jittor.concat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = jnn.relu(h)

            rgb = self.rgb_linear(h)
            return alpha, rgb
        else:
            assert False


class SingleVarianceNetwork(jnn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        # 这里对于jittor来说 一个module的成员变量就是一个参数
        self.variance = jittor.float32(init_val)
        # self.register_parameter('variance', jnn.Parameter(jittor.float32(init_val)))

    def forward(self, x):
        return jittor.ones([len(x), 1]) * jittor.exp(self.variance * jittor.array(10.0))
