import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import time
from einops import rearrange, einsum
from einops.layers.torch import Rearrange
from typing import Tuple


class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor):
        '''
        x: (b c h w)
        '''
        x = x.permute(0, 2, 3, 1).contiguous() #(b h w c)
        x = self.norm(x) #(b h w c)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class SCA(nn.Module):

    def __init__(self, embed_dim, num_heads, num_clusters):
        super().__init__()
        self.num_clusters = num_clusters
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.qkv = nn.Conv2d(embed_dim, embed_dim*3, 1, bias=True)
        self.lepe = nn.Conv2d(embed_dim, embed_dim, 5, 1, 2, groups=embed_dim)
        self.out_proj = nn.Conv2d(embed_dim, embed_dim, 1, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.qkv.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight, gain=2 ** -2.5)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def sort_sim(self, q: torch.Tensor, k: torch.Tensor):
        '''
        q: b n (h w) d
        k: b n (h w) d
        '''
        cls_token = q.mean(2, keepdim=True) # (b n 1 d)
        cls_token = F.normalize(cls_token, dim=-1)
        k = F.normalize(k, dim=-1)
        cos_sim = cls_token @ k.transpose(-1, -2) # (b n 1 (h w))
        sort_indices = torch.argsort(cos_sim, dim=-1)
        return sort_indices.squeeze(2) # (b n (h w))

    def sort_tensor(self, indices: torch.Tensor, target: torch.Tensor):
        '''
        indices: (b n l)
        target: (b n l d)
        '''
        sorted_tensor = torch.gather(target, dim=-2, index=indices.unsqueeze(-1).expand_as(target))
        return sorted_tensor

    def restore_tensor(self, indices: torch.Tensor, target: torch.Tensor):
        '''
        indices: (b n l)
        target: (b n l d)
        '''
        restored_tensor = torch.gather(target, dim=-2, index=indices.sort(dim=-1).indices.unsqueeze(-1).expand_as(target))
        return restored_tensor

    def forward(self, x: torch.Tensor):
        '''
        x: (b c h w)
        '''
        bsz, _, h, w = x.size()
        l = h * w
        assert l % self.num_clusters == 0
        qkv = self.qkv(x) # (b 3*c h w)
        lepe = self.lepe(qkv[:, 2*self.embed_dim:, :, :])

        q, k, v = rearrange(qkv, 'b (m n d) h w -> m b n (h w) d', m=3, n=self.num_heads)
        # q, k, v -> b n (h w) d

        k = k * self.scaling

        if self.num_clusters > 1:
            indices = self.sort_sim(k, k) # (b n (h w))
            q = self.sort_tensor(indices, q) # (b n (h w) d)
            k = self.sort_tensor(indices, k) # (b n (h w) d)
            v = self.sort_tensor(indices, v) # (b n (h w) d)

            q_l = rearrange(q, 'b n (c l) d -> b n c l d', c=self.num_clusters)
            k_l = rearrange(k, 'b n (c l) d -> b n c l d', c=self.num_clusters)
            v = rearrange(v, 'b n (c l) d -> b n c l d', c=self.num_clusters)
            attn = torch.softmax(q_l @ k_l.transpose(-1, -2), dim=-1) # (b n c l l)
            v = attn @ v # (b n c l d)

            v = rearrange(v, 'b n c l d -> b n (c l) d')
            v = self.restore_tensor(indices, v) # (b n (h w) d)
            v = rearrange(v, 'b n (h w) d -> b (n d) h w', h=h, w=w)
        else:
            attn = torch.softmax(q @ k.transpose(-1, -2), dim=-1) # (b n (h w) (h w))
            v = attn @ v # (b n (h w) d)
            v = rearrange(v, 'b n (h w) d -> b (n d) h w', h=h, w=w)
        res = v + lepe
        return self.out_proj(res)

class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn=F.gelu,
        dropout=0.0,
        activation_dropout=0.0,
        subconv=False
        ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = activation_fn
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Conv2d(self.embed_dim, ffn_dim, 1)
        self.fc2 = nn.Conv2d(ffn_dim, self.embed_dim, 1)
        self.dwconv = nn.Conv2d(ffn_dim, ffn_dim, 3, 1, 1, groups=ffn_dim) if subconv else None

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.dwconv.reset_parameters()

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        if self.dwconv is not None:
            residual = x
            x = self.dwconv(x)
            x = x + residual
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x

class Block(nn.Module):

    def __init__(self, embed_dim, num_heads, num_cluster, ffn_dim, drop_path=0., layerscale=False, layer_init_values=1e-6):
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim
        self.norm1 = LayerNorm2d(embed_dim, eps=1e-6)
        self.attn = SCA(embed_dim, num_heads, num_cluster)
        self.drop_path = DropPath(drop_path)
        self.norm2 = LayerNorm2d(embed_dim, eps=1e-6)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.pos = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=embed_dim)

        if layerscale:
            self.gamma_1 = nn.Parameter(layer_init_values * torch.ones(1, embed_dim, 1, 1),requires_grad=True)
            self.gamma_2 = nn.Parameter(layer_init_values * torch.ones(1, embed_dim, 1, 1),requires_grad=True)

    def forward(self, x: torch.Tensor):
        x = x + self.pos(x)
        if self.layerscale:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x
    
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, out_dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim, out_dim, 3, 2, 1)
        self.norm = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        '''
        x: B C H W
        '''
        x = self.reduction(x) #(b oc oh ow)
        x = self.norm(x)
        return x
    
class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, embed_dim, out_dim, depth, num_heads, num_clusters,
                 ffn_dim=96., drop_path=0.,
                 downsample: PatchMerging=None,
                 layerscale=False, layer_init_values=1e-6):

        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, num_clusters[i], ffn_dim, 
                     drop_path[i] if isinstance(drop_path, list) else drop_path, layerscale, layer_init_values)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=embed_dim, out_dim=out_dim)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor):
        _, _, h, w = x.size()
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x
    
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, in_chans=3, embed_dim=96):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim//2, 3, 2, 1),
            nn.BatchNorm2d(embed_dim//2),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim//2, 3, 1, 1),
            nn.BatchNorm2d(embed_dim//2),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim//2, 3, 1, 1),
            nn.BatchNorm2d(embed_dim//2),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim, 3, 2, 1),
            nn.BatchNorm2d(embed_dim),
            # nn.GELU(),
            # nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            # nn.BatchNorm2d(embed_dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)#(b c h w)
        return x
    
class SecViT(nn.Module):

    def __init__(self, in_chans=3, num_classes=1000, 
                 embed_dims=[96, 192, 384, 768], depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], 
                 num_clusterss=[[64, 49], [16, 49], [4, 49], [1, 1]], mlp_ratios=[3, 3, 3, 3], drop_path_rate=0.1, 
                 layerscales=[False, False, False, False], layer_init_values=1e-6):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dims[0]
        self.num_features = embed_dims[-1]
        self.mlp_ratios = mlp_ratios

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dims[0])


        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                embed_dim=embed_dims[i_layer],
                out_dim=embed_dims[i_layer+1] if (i_layer < self.num_layers - 1) else None,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                num_clusters=num_clusterss[i_layer],
                ffn_dim=int(mlp_ratios[i_layer]*embed_dims[i_layer]),
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                layerscale=layerscales[i_layer],
                layer_init_values=layer_init_values
            )
            self.layers.append(layer)
            
        self.norm = nn.BatchNorm2d(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Conv2d(self.num_features, num_classes, 1) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            try:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            except:
                pass

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)

        for layer in self.layers:
            x = layer(x)


        x = self.norm(x) #(b c h w)

        x = self.avgpool(x)  # B C 1 1
        return x

    def forward(self, x):
        # x = F.interpolate(x, (384, 384), mode='bicubic')
        x = self.forward_features(x)
        x = self.head(x).flatten(1)
        return x

@register_model
def SecViT_T(args):
    model = SecViT(
        embed_dims=[64, 128, 256, 512],
        depths=[2, 2, 9, 2],
        num_heads=[2, 4, 8, 16],
        num_clusterss=[[32, 32]*10, [8, 8]*10, [2, 2]*20, [-1, -1]*10],
        mlp_ratios=[3, 3, 3, 3],
        drop_path_rate=0.1,
        layerscales=[False, False, False, False],
    )
    model.default_cfg = _cfg()
    return model
    
@register_model
def SecViT_S(args):
    model = SecViT(
        embed_dims=[64, 128, 256, 512],
        depths=[3, 4, 18, 4],
        num_heads=[2, 4, 8, 16],
        num_clusterss=[[32, 32]*10, [8, 8]*10, [2, 2]*20, [-1, -1]*10],
        mlp_ratios=[3, 3, 3, 3],
        drop_path_rate=0.15,
        layerscales=[False, False, False, False],
    )
    model.default_cfg = _cfg()
    return model

@register_model
def SecViT_M(args):
    model = SecViT(
        embed_dims=[80, 160, 320, 512],
        depths=[4, 8, 26, 8],
        num_heads=[5, 5, 10, 16],
        num_clusterss=[[32, 32]*10, [8, 8]*10, [2, 2]*20, [-1, -1]*10],
        mlp_ratios=[3, 3, 3, 3],
        drop_path_rate=0.4,
        layerscales=[False, False, True, True],
    )
    model.default_cfg = _cfg()
    return model

@register_model
def SecViT_L(args):
    model = SecViT(
        embed_dims=[112, 224, 448, 640],
        depths=[4, 8, 26, 8],
        num_heads=[7, 7, 14, 20],
        num_clusterss=[[32, 32]*10, [8, 8]*10, [2, 2]*20, [-1, -1]*10],
        mlp_ratios=[3, 3, 3, 3],
        drop_path_rate=0.5,
        layerscales=[False, False, True, True],
    )
    model.default_cfg = _cfg()
    return model
