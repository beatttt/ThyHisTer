from functools import partial

import torch
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

from components.senet.se_module import SELayer

# helpers

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# helper classes

class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = ChanLayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))

class OverlappingPatchEmbed(nn.Module):
    def __init__(self, dim_in, dim_out, stride = 2):
        super().__init__()
        kernel_size = stride * 2 - 1
        padding = kernel_size // 2
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride = stride, padding = padding)

    def forward(self, x):
        return self.conv(x)

# positional encoding
class PEG(nn.Module):
    def __init__(self, dim, kernel_size = 3):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, kernel_size = kernel_size, padding = kernel_size // 2, groups = dim, stride = 1)

    def forward(self, x):
        return self.proj(x) + x

# feedforward

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Conv2d(dim, inner_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# attention

class DSSA(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 32,
        dropout = 0.,
        window_size = 7
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size
        inner_dim = dim_head * heads

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, 1, bias = False)

        # window tokens
        self.window_tokens = nn.Parameter(torch.randn(dim))

        # prenorm and non-linearity for window tokens
        # then projection to queries and keys for window tokens

        self.window_tokens_to_qk = nn.Sequential(
            nn.LayerNorm(dim_head),
            nn.GELU(),
            Rearrange('b h n c -> b (h c) n'),
            nn.Conv1d(inner_dim, inner_dim * 2, 1),
            Rearrange('b (h c) n -> b h n c', h = heads),
        )

        # window attention

        self.window_attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        einstein notation

        b - batch
        c - channels
        w1 - window size (height)
        w2 - also window size (width)
        i - sequence dimension (source)
        j - sequence dimension (target dimension to be reduced)
        h - heads
        x - height of feature map divided by window size
        y - width of feature map divided by window size
        """


        # height = x.shape[-2], width = x.shape[-1]
        # x: (batch, dim, height, width)
        _, height, width, heads, wsz = x.shape[0], *x.shape[-2:], self.heads, self.window_size
        assert (height % wsz) == 0 and (width % wsz) == 0, f'height {height} and width {width} must be divisible by window size {wsz}'
        num_windows = (height // wsz) * (width // wsz)

        # fold in windows for "depthwise" attention - not sure why it is named depthwise when it is just "windowed" attention

        # x是已经被切割和铺平的小图片
        # x: (batch * num_windows, dim, window_size ** 2)
        x = rearrange(x, 'b c (h w1) (w w2) -> (b h w) c (w1 w2)', w1 = wsz, w2 = wsz)

        # add windowing tokens

        # window_tokens被设置成随机值，这里c的值就是dim的值
        # self.window_tokens: (dim)
        # w: (batch * num_windows, dim, 1)
        w = repeat(self.window_tokens, 'c -> b c 1', b = x.shape[0])
        # 将window_tokens加到特征图中
        # x: (batch * num_windows, dim, window_size ** 2 + 1)
        x = torch.cat((w, x), dim = -1)

        # project for queries, keys, value
        # 将q, k和v从一个大矩阵中分离出来
        # to_qkv: (batch * num_windows, heads * dim_head * 3, window_size ** 2 + 1)
        # q: (batch * num_windows, heads * dim_head, window_size ** 2 + 1)
        q, k, v = self.to_qkv(x).chunk(3, dim = 1)

        # split out heads
        # 调整q, k, v的维度，不改变其值
        # q: (batch * num_windows, heads, window_size ** 2 + 1, dim_head)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h = heads), (q, k, v))

        # scale
        q = q * self.scale

        # similarity
        # 对d所在的维度乘积求和，求出注意力值
        # dots: (batch * num_windows, heads, window_size ** 2 + 1, window_size ** 2 + 1)
        dots = einsum('b h i d, b h j d -> b h i j', q, k)

        # attention
        # attn: (batch * num_windows, heads, window_size ** 2 + 1, window_size ** 2 + 1)
        attn = self.attend(dots)

        # aggregate values
        # 用注意力权重加权v
        # out: (batch * num_windows, heads, window_size ** 2 + 1, dim_head)
        out = torch.matmul(attn, v)

        # split out windowed tokens
        # 将特征图和window_token分离
        # window_tokens: (batch * num_windows, heads, dim_head)
        # windowed_fmaps: (batch * num_windows, heads, window_size ** 2, dim_head)
        window_tokens, windowed_fmaps = out[:, :, 0], out[:, :, 1:]

        # early return if there is only 1 window

        if num_windows == 1:
            fmap = rearrange(windowed_fmaps, '(b x y) h (w1 w2) d -> b (h d) (x w1) (y w2)', x = height // wsz, y = width // wsz, w1 = wsz, w2 = wsz)
            return self.to_out(fmap)

        # carry out the pointwise attention, the main novelty in the paper

        # 将window_tokens和windowed_fmaps重新调整维度
        # window_tokens: (batch, heads, num_windows, dim_head)
        # windowed_fmaps: (batch, heads, num_windows, window_size ** 2, dim_head)
        window_tokens = rearrange(window_tokens, '(b x y) h d -> b h (x y) d', x = height // wsz, y = width // wsz)
        windowed_fmaps = rearrange(windowed_fmaps, '(b x y) h n d -> b h (x y) n d', x = height // wsz, y = width // wsz)

        # windowed queries and keys (preceded by prenorm activation)
        # window_tokens_to_qk: (batch, heads, num_windows, dim_head * 2)
        # w_q: (batch, heads, num_windows, dim_head)
        w_q, w_k = self.window_tokens_to_qk(window_tokens).chunk(2, dim = -1)

        # scale
        w_q = w_q * self.scale

        # similarities
        # 跟上面类似的操作，相乘相加得到注意力值
        # w_dots: (batch, heads, num_windows, num_windows)
        w_dots = einsum('b h i d, b h j d -> b h i j', w_q, w_k)

        # w_attn: (batch, heads, num_windows, num_windows)
        w_attn = self.window_attend(w_dots)

        # aggregate the feature maps from the "depthwise" attention step (the most interesting part of the paper, one i haven't seen before)
        # 用window_tokens计算出来的注意力值加权特征图
        # aggregated_windowed_fmap: (batch, heads, num_windows, window_size ** 2, dim_head)
        aggregated_windowed_fmap = einsum('b h i j, b h j w d -> b h i w d', w_attn, windowed_fmaps)

        # fold back the windows and then combine heads for aggregation
        # fmap: (batch, heads * dim_head, feat_width, feat_height) feat_width和feat_height表示特征图的尺寸
        fmap = rearrange(aggregated_windowed_fmap, 'b h (x y) (w1 w2) d -> b (h d) (x w1) (y w2)', x = height // wsz, y = width // wsz, w1 = wsz, w2 = wsz)
        return self.to_out(fmap)

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        dim_head = 32,
        heads = 8,
        ff_mult = 4,
        dropout = 0.,
        norm_output = True
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, DSSA(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mult = ff_mult, dropout = dropout)),
            ]))

        self.norm = ChanLayerNorm(dim) if norm_output else nn.Identity()
        self.dim = dim

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class SeSepViT(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        dim,
        depth,
        heads,
        window_size = 7,
        dim_head = 32,
        ff_mult = 4,
        channels = 3,
        dropout = 0.
    ):
        super().__init__()
        assert isinstance(depth, tuple), 'depth needs to be tuple if integers indicating number of transformer blocks at that stage'

        #  dim = 12,               
        #  dim_head = 12,          
        #  heads = (1, 2, 2, 4),   
        #  depth = (1, 2, 4, 2),   
        #  window_size = 7,        
        self.num_stages = len(depth)
        self.num_classes = num_classes
        num_stages = self.num_stages

        # self.dims = (12, 24, 48, 96)
        self.dims = tuple(map(lambda i: (2 ** i) * dim, range(num_stages)))
        self.dim = dim
        dims = self.dims
        # dims = (3, 12, 24, 48, 96)
        dims = (channels, *dims)
        # dim_pairs = ((3, 12), (12, 24), (24, 48), (48, 96))
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        # strides = (4, 2, 2, 2)
        strides = (4, *((2,) * (num_stages - 1)))

        # hyperparams_per_stage = [(1, 2, 2, 4), 7]
        hyperparams_per_stage = [heads, window_size]
        # hyperparams_per_stage = [(1, 2, 2, 4), (7, 7, 7, 7)]
        hyperparams_per_stage = list(map(partial(cast_tuple, length = num_stages), hyperparams_per_stage))
        assert all(tuple(map(lambda arr: len(arr) == num_stages, hyperparams_per_stage)))

        self.layers = nn.ModuleList([])

        for ind, ((layer_dim_in, layer_dim), layer_depth, layer_stride, layer_heads, layer_window_size) in enumerate(zip(dim_pairs, depth, strides, *hyperparams_per_stage)):
            # ind               = 0 , 1 , 2 , 3
            # layer_dim_in      = 3 , 12, 24, 48
            # layer_dim         = 12, 24, 48, 96
            # layer_depth       = 1 , 2 , 4 , 2
            # layer_stride      = 4 , 2 , 2 , 2
            # layer_heads       = 1 , 2 , 2 , 4
            # layer_window_size = 7 , 7 , 7 , 7
            is_last = ind == (num_stages - 1)

            self.layers.append(nn.ModuleList([
                OverlappingPatchEmbed(layer_dim_in, layer_dim, stride = layer_stride),
                PEG(layer_dim),
                Transformer(dim = layer_dim, depth = layer_depth, heads = layer_heads, ff_mult = ff_mult, dropout = dropout, norm_output = not is_last),
                SELayer(layer_dim)
            ]))

        self.mlp_head = nn.Sequential(
            Reduce('b d h w -> b d', 'mean'),
            nn.LayerNorm(dims[-1]),
            nn.Linear(dims[-1], num_classes)
        )
        # 添加注意力机制se_block
        #  self.se = SELayer((2 ** (num_stages - 1)) * dim)

    def forward(self, x):
        for ope, peg, transformer, se_block in self.layers:
            x = ope(x)
            x = peg(x)
            x = transformer(x)
            x = se_block(x)

        # 添加注意力机制se_block
        #  x = self.se(x)
        return self.mlp_head(x)
    

class FuSepViT(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        dim,
        depth,
        heads,
        window_size = 7,
        dim_head = 32,
        ff_mult = 4,
        channels = 3,
        dropout = 0.
    ):
        super().__init__()
        assert isinstance(depth, tuple), 'depth needs to be tuple if integers indicating number of transformer blocks at that stage'

        num_stages = len(depth)

        dims = tuple(map(lambda i: (2 ** i) * dim, range(num_stages)))
        dims = (channels, *dims)
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        strides = (4, *((2,) * (num_stages - 1)))

        hyperparams_per_stage = [heads, window_size]
        hyperparams_per_stage = list(map(partial(cast_tuple, length = num_stages), hyperparams_per_stage))
        assert all(tuple(map(lambda arr: len(arr) == num_stages, hyperparams_per_stage)))

        self.layers = nn.ModuleList([])

        for ind, ((layer_dim_in, layer_dim), layer_depth, layer_stride, layer_heads, layer_window_size) in enumerate(zip(dim_pairs, depth, strides, *hyperparams_per_stage)):
            is_last = ind == (num_stages - 1)

            self.layers.append(nn.ModuleList([
                OverlappingPatchEmbed(layer_dim_in, layer_dim, stride = layer_stride),
                PEG(layer_dim),
                Transformer(dim = layer_dim, depth = layer_depth, heads = layer_heads, ff_mult = ff_mult, dropout = dropout, norm_output = not is_last),
            ]))


    def forward(self, x):

        for ope, peg, transformer in self.layers:
            x = ope(x)
            x = peg(x)
            x = transformer(x)

        return x