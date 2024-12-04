import jittor
from jittor import nn as jnn
from JittorPaper1NERF.outer_jittor import outerjittor as oj

# Positional encoding embedding. Code was taken from https://github.com/bmild/nerf.
import math

jittor.flags.use_cuda = 1


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** jittor.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = jittor.linspace(2. ** 0., 2. ** max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return jittor.concat([fn(inputs) for fn in self.embed_fns], -1)


class IntegratedPositionEncoder(jnn.Module):

    def __init__(self, input_dim, N_freqs, max_freq, log_sampling=True, trainable=False):
        super().__init__()
        self.out_dim = 2 * input_dim * N_freqs

        if log_sampling:
            freq_bands = 2. ** jittor.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = jittor.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)
        if trainable:
            self.freq_bands = jnn.Parameter(freq_bands, requires_grad=True)
        else:
            self.register_buffer('freq_bands', freq_bands, persistent=False)

    def expected_sin(self, x, x_var):
        """Estimates mean and variance of sin(z), z ~ N(x, var)."""
        # When the variance is wide, shrink sin towards zero.
        y = jittor.multiply(jittor.exp(-0.5 * x_var), jittor.sin(x))
        y_var = jittor.maximum(
            jittor.zeros_like(x),
            jittor.array(0.5) * (jittor.array(1) - jittor.multiply(jittor.exp(-2 * x_var),
                                                                   jittor.cos(2 * x))) - jittor.multiply(y, y)
        )
        return y, y_var

    def forward(self, x, x_cov, diag=True):
        """Encode `x` with sinusoids scaled by 2^[min_deg:max_deg-1].
        Args:
            x, [N_pts, 3], variables to be encoded. Should be in [-pi, pi].
            x_cov, [N_pts, 3, 3], covariance matrices for `x`.
            diag: bool, if true, expects input covariances to be diagonal (full
            otherwise).
        Returns:
            encoded: [N_pts, 3], encoded variables.
        """
        if not diag:
            x_cov = oj.diagonal(x_cov)

        y = x[..., None, :] * self.freq_bands[:, None]  # [N_pts, 1, 3] x [N_freqs, 1] -> [N_pts, N_freqs, 3]
        y = y.reshape(x.shape[:-1] + (-1,))  # [N_pts, N_freqs * 3]
        y_var = x_cov[..., None, :] * self.freq_bands[:,
                                      None] ** 2  # [N_pts, 1, 3] x [N_freqs, 1] -> [N_pts, N_freqs, 3]
        y_var = y_var.reshape(x.shape[:-1] + (-1,))  # [N_pts, N_freqs * 3]

        return self.expected_sin(
            jittor.concat([y, y + 0.5 * math.pi], -1),
            jittor.concat([y_var, y_var], -1)
        )[0]


def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [jittor.sin, jittor.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj): return eo.embed(x)

    return embed, embedder_obj.out_dim
