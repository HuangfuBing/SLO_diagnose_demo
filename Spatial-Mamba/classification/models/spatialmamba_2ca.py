# -----------------------------------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# -----------------------------------------------------------------------------------
# VMamba: Visual State Space Model
# Copyright (c) 2024 MzeroMiko
# -----------------------------------------------------------------------------------
# Spatial-Mamba: Effective Visual State Space Models via Structure-Aware State Fusion
# Modified by Chaodong Xiao
# -----------------------------------------------------------------------------------

import math
import copy
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import flop_count, parameter_count
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

try:
    from .utils import selective_scan_state_flop_jit, selective_scan_fn, Stem, DownSampling
except:
    from utils import selective_scan_state_flop_jit, selective_scan_fn, Stem, DownSampling

try:
    from Dwconv.dwconv_layer import DepthwiseFunction
except:
    DepthwiseFunction = None


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class StateFusion(nn.Module):
    def __init__(self, dim):
        super(StateFusion, self).__init__()

        self.dim = dim
        self.kernel_3   = nn.Parameter(torch.ones(dim, 1, 3, 3))
        self.kernel_3_1 = nn.Parameter(torch.ones(dim, 1, 3, 3))
        self.kernel_3_2 = nn.Parameter(torch.ones(dim, 1, 3, 3))
        self.alpha = nn.Parameter(torch.ones(3), requires_grad=True)

    @staticmethod
    def padding(input_tensor, padding):
        return torch.nn.functional.pad(input_tensor, padding, mode='replicate')

    def forward(self, h):

        if self.training:
            h1 = F.conv2d(self.padding(h, (1,1,1,1)), self.kernel_3,   padding=0, dilation=1, groups=self.dim)
            h2 = F.conv2d(self.padding(h, (3,3,3,3)), self.kernel_3_1, padding=0, dilation=3, groups=self.dim)
            h3 = F.conv2d(self.padding(h, (5,5,5,5)), self.kernel_3_2, padding=0, dilation=5, groups=self.dim)
            out = self.alpha[0]*h1 + self.alpha[1]*h2 + self.alpha[2]*h3
            return out

        else:
            if not hasattr(self, "_merge_weight"):
                self._merge_weight = torch.zeros((self.dim, 1, 11, 11), device=h.device)
                self._merge_weight[:, :, 4:7, 4:7] = self.alpha[0]*self.kernel_3

                self._merge_weight[:, :, 2:3, 2:3] = self.alpha[1]*self.kernel_3_1[:,:,0:1,0:1]
                self._merge_weight[:, :, 2:3, 5:6] = self.alpha[1]*self.kernel_3_1[:,:,0:1,1:2]
                self._merge_weight[:, :, 2:3, 8:9] = self.alpha[1]*self.kernel_3_1[:,:,0:1,2:3]
                self._merge_weight[:, :, 5:6, 2:3] = self.alpha[1]*self.kernel_3_1[:,:,1:2,0:1]
                self._merge_weight[:, :, 5:6, 5:6] += self.alpha[1]*self.kernel_3_1[:,:,1:2,1:2]
                self._merge_weight[:, :, 5:6, 8:9] = self.alpha[1]*self.kernel_3_1[:,:,1:2,2:3]
                self._merge_weight[:, :, 8:9, 2:3] = self.alpha[1]*self.kernel_3_1[:,:,2:3,0:1]
                self._merge_weight[:, :, 8:9, 5:6] = self.alpha[1]*self.kernel_3_1[:,:,2:3,1:2]
                self._merge_weight[:, :, 8:9, 8:9] = self.alpha[1]*self.kernel_3_1[:,:,2:3,2:3]

                self._merge_weight[:, :, 0:1, 0:1] = self.alpha[2]*self.kernel_3_2[:,:,0:1,0:1]
                self._merge_weight[:, :, 0:1, 5:6] = self.alpha[2]*self.kernel_3_2[:,:,0:1,1:2]
                self._merge_weight[:, :, 0:1, 10:11] = self.alpha[2]*self.kernel_3_2[:,:,0:1,2:3]
                self._merge_weight[:, :, 5:6, 0:1] = self.alpha[2]*self.kernel_3_2[:,:,1:2,0:1]
                self._merge_weight[:, :, 5:6, 5:6] += self.alpha[2]*self.kernel_3_2[:,:,1:2,1:2]
                self._merge_weight[:, :, 5:6, 10:11] = self.alpha[2]*self.kernel_3_2[:,:,1:2,2:3]
                self._merge_weight[:, :, 10:11, 0:1] = self.alpha[2]*self.kernel_3_2[:,:,2:3,0:1]
                self._merge_weight[:, :, 10:11, 5:6] = self.alpha[2]*self.kernel_3_2[:,:,2:3,1:2]
                self._merge_weight[:, :, 10:11, 10:11] = self.alpha[2]*self.kernel_3_2[:,:,2:3,2:3]

            out = DepthwiseFunction.apply(h, self._merge_weight, None, 11//2, 11//2, False)

            return out

class StructureAwareSSM(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state*2), bias=False, **factory_kwargs)
        self.x_proj_weight = nn.Parameter(self.x_proj.weight)
        del self.x_proj

        self.dt_projs = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
        self.dt_projs_weight = nn.Parameter(self.dt_projs.weight)
        self.dt_projs_bias = nn.Parameter(self.dt_projs.bias)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, dt_init)
        self.Ds = self.D_init(self.d_inner, dt_init)

        self.selective_scan = selective_scan_fn

        self.state_fusion = StateFusion(self.d_inner)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None


    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, bias=True,**factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=bias, **factory_kwargs)

        if bias:
            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))

            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            dt_proj.bias._no_reinit = True

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        elif dt_init == "simple":
            with torch.no_grad():
                dt_proj.weight.copy_(0.1 * torch.randn((d_inner, dt_rank)))
                dt_proj.bias.copy_(0.1 * torch.randn((d_inner)))
                dt_proj.bias._no_reinit = True
        elif dt_init == "zero":
            with torch.no_grad():
                dt_proj.weight.copy_(0.1 * torch.rand((d_inner, dt_rank)))
                dt_proj.bias.copy_(0.1 * torch.rand((d_inner)))
                dt_proj.bias._no_reinit = True
        else:
            raise NotImplementedError

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, init, device=None):
        if init=="random" or "constant":
            # S4D real initialization
            A = repeat(
                torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=d_inner,
            ).contiguous()
            A_log = torch.log(A)
            A_log = nn.Parameter(A_log)
            A_log._no_weight_decay = True
        elif init=="simple":
            A_log = nn.Parameter(torch.randn((d_inner, d_state)))
        elif init=="zero":
            A_log = nn.Parameter(torch.zeros((d_inner, d_state)))
        else:
            raise NotImplementedError
        return A_log

    @staticmethod
    def D_init(d_inner, init="random", device=None):
        if init=="random" or "constant":
            # D "skip" parameter
            D = torch.ones(d_inner, device=device)
            D = nn.Parameter(D) 
            D._no_weight_decay = True
        elif init == "simple" or "zero":
            D = nn.Parameter(torch.ones(d_inner))
        else:
            raise NotImplementedError
        return D

    def ssm(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W

        xs = x.view(B, -1, L)
        
        x_dbl = torch.matmul(self.x_proj_weight.view(1, -1, C), xs)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=1)
        dts = torch.matmul(self.dt_projs_weight.view(1, C, -1), dts)
        
        As = -torch.exp(self.A_logs)
        Ds = self.Ds
        dts = dts.contiguous()
        dt_projs_bias = self.dt_projs_bias

        h = self.selective_scan(
            xs, dts, 
            As, Bs, None,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        )

        h = rearrange(h, "b d 1 (h w) -> b (d 1) h w", h=H, w=W)
        h = self.state_fusion(h)
        h = rearrange(h, "b d h w -> b d (h w)")
        
        y = h * Cs
        y = y + xs * Ds.view(-1, 1)

        return y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) 

        x = rearrange(x, 'b h w d -> b d h w').contiguous()
        x = self.act(self.conv2d(x)) 

        y = self.ssm(x) 

        y = rearrange(y, 'b d (h w)-> b h w d', h=H, w=W)

        y = self.out_norm(y)
        y = y * F.silu(z)
        y = self.out_proj(y)
        if self.dropout is not None:
            y = self.dropout(y)
        return y


class SpatialMambaBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        dt_init: str = "random",
        num_heads: int = 8,
        mlp_ratio = 4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        **kwargs,
    ):
        super().__init__()

        self.cpe1 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = StructureAwareSSM(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, dt_init=dt_init, **kwargs)
        self.drop_path = DropPath(drop_path)

        self.cpe2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLP(in_features=hidden_dim, hidden_features=int(hidden_dim*mlp_ratio), act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=False)
    
    def forward(self, x: torch.Tensor):

        x = x + self.cpe1(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1)
        x = x + self.drop_path(self.self_attention(self.ln_1(x)))
        x = x + self.cpe2(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1)
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class SpatialMambaLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        downsample=None, 
        use_checkpoint=False, 
        d_state=16,
        dt_init="random",
        mlp_ratio=4.0,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SpatialMambaBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                dt_init=dt_init,
                mlp_ratio=mlp_ratio,
            )
            for i in range(depth)])

        if True:
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim)
        else:
            self.downsample = None


    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class SpatialMamba(nn.Module):
    def __init__(self, 
                 img_size=224, 
                 patch_size=4, 
                 in_chans=3, 
                 num_classes=1000, 
                 depths=[2, 2, 9, 2], 
                 dims=[96, 192, 384, 768], 
                 d_state=1, 
                 dt_init="random",
                 mlp_ratio=4.0,
                 drop_rate=0., 
                 attn_drop_rate=0., 
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, 
                 patch_norm=True,
                 use_checkpoint=False, 
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed = Stem(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim )

        self.ape = False
        if self.ape:
            self.patches_resolution = self.patch_embed.patches_resolution
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = SpatialMambaLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
                dt_init=dt_init,
                mlp_ratio=mlp_ratio,
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=DownSampling if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        # TODO: check output dim
        print("num_classes-----------------------", num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
    
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def flops(self, shape=(3, 224, 224), verbose=True):
        # shape = self.__input_shape__[1:]
        supported_ops={
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            "prim::PythonOp.SelectiveScanStateFn": selective_scan_state_flop_jit,
        }

        model = copy.deepcopy(self)
        model.cuda()#.eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        return sum(Gflops.values()) * 1e9
        return f"params {params} GFLOPs {sum(Gflops.values())}"

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = torch.flatten(x, 1, 2)
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

class Backbone_SpatialMamba(SpatialMamba):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer="ln", **kwargs):

        super().__init__(**kwargs)
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None) 
        kwargs.update(norm_layer=norm_layer)       

        # add norm ========================
        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        # modify layer ========================
        def layer_forward(self: SpatialMambaLayer, x):
            for blk in self.blocks:
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)

            y = None
            if self.downsample is not None:
                y = self.downsample(x)

            return x, y

        for l in self.layers:
            l.forward = partial(layer_forward, l)

        # del self.classifier ===================
        del self.head
        del self.avgpool
        del self.norm

        self.load_pretrained(pretrained, key=kwargs.get('key','model'))

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return

        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)        
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self, x):
        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                if not self.channel_first:
                    out = out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        if len(self.out_indices) == 0:
            return x

        return outs

class LesionQueryHead(nn.Module):
    def __init__(
        self,
        backbone: SpatialMamba,
        num_lesions: int,
        num_patches: int,
        lesion_ids: Optional[list] = None,
        embed_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_lesions = int(num_lesions)
        self.num_patches = int(num_patches)

        if lesion_ids is None:
            lesion_ids = list(range(num_lesions))
        lesion_ids = list(lesion_ids)
        self.register_buffer(
            "lesion_ids",
            torch.tensor(lesion_ids, dtype=torch.long),
            persistent=False
        )

        feat_dim = getattr(backbone, "num_features", None)
        if feat_dim is None:
            raise ValueError(
                "Check SpatialMamaba attribute: num_features"
            )
        
        self.feat_dim = int(feat_dim)
        self.embed_dim = int(embed_dim)

        self.proj = nn.Linear(self.feat_dim, self.embed_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))

        self.query_embed = nn.Parameter(
            torch.randn(self.num_lesions, self.embed_dim) * 0.02
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.ln_q = nn.LayerNorm(self.embed_dim)
        self.ln_kv = nn.LayerNorm(self.embed_dim)
        
        self.lesion_head = nn.Linear(self.embed_dim, 1)

        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.0)

        nn.init.normal_(self.query_embed, std=0.02)

        trunc_normal_(self.pos_embed, std=0.02)

        nn.init.xavier_uniform_(self.lesion_head.weight)
        if self.lesion_head.bias is not None:
            nn.init.constant_(self.lesion_head.bias, 0.0)

    @torch.no_grad()
    def _extract_patch_features(self, patches: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = patches.shape
        patches = patches.view(B * N, C, H, W)
        feats = self.backbone.forward_features(patches)
        feats = feats.view(B, N, self.feat_dim)
        return feats
    
    def forward(self, patches: torch.Tensor, return_embeds: bool = False, return_attn: bool = False):
        feats = self._extract_patch_features(patches)
        kv = self.proj(feats)
        if kv.shape[1] == self.num_patches:
            kv = kv + self.pos_embed
        else:
            print(f"Warning: Patch num mismatch. Got {kv.shape[1]}, expected {self.num_patches}")
        
        kv = self.ln_kv(kv)

        B = kv.size(0)
        q = self.query_embed.unsqueeze(0).expand(B, -1, -1)
        q = self.ln_q(q)

        lesion_embeds, attn_w = self.attn(q, kv, kv, need_weights=return_attn)

        lesion_logits = self.lesion_head(lesion_embeds).squeeze(-1)
        lesion_probs = torch.sigmoid(lesion_logits)
        # Multilabel classification (42 classes)
        if return_embeds and return_attn:
            return lesion_probs, lesion_embeds, attn_w
        elif return_embeds:
            return lesion_probs, lesion_embeds
        else:
            return lesion_probs

class LesionToDiseaseMapper(nn.Module):

    def __init__(
        self,
        prior_matrix,
        learn_delta: bool = True,
        init_delta_prior: float = 0.0,    
        init_delta_noprior: float = 0.01, 
        embed_dim: int = 256,
        num_heads: int = 4,
        prior_bias_scale: float = 2.0,
        positive_only: bool = True, # only use (s-tau)+, avoid global probs↓
    ):
        super().__init__()
        prior = torch.as_tensor(prior_matrix, dtype=torch.float32)
        self.num_lesions, self.num_classes = prior.shape
        self.register_buffer("M_prior", prior, persistent=True)

        if learn_delta:
            prior_mask = prior
            noprior_mask = 1.0 - prior_mask
            delta_init = torch.zeros_like(prior)
            delta_init = delta_init + prior_mask * float(init_delta_prior)
            delta_init = delta_init + noprior_mask * float(init_delta_noprior)
            self.delta = nn.Parameter(delta_init)
        else:
            self.register_parameter("delta", None)
        
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.prior_bias_scale = float(prior_bias_scale)
        self.positive_only = bool(positive_only)

        # FIXME: 0.02 is ok?
        self.disease_query = nn.Parameter(torch.randn(self.num_classes, self.embed_dim) * 0.02)
        self.ln_q = nn.LayerNorm(self.embed_dim)
        self.ln_kv = nn.LayerNorm(self.embed_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            batch_first=True,
            dropout=0.0,
        )

        # self.out = nn.Linear(self.embed_dim, 1)

        self.last_attn = None # [B, C, L] for visualization

    def get_W(self) -> torch.Tensor:
        if self.delta is None:
            W = self.M_prior
        else:
            W = self.M_prior + self.delta
            # TODO: 20251205-softplus
            W = torch.relu(W)
        return W

    def forward(
        self,
        lesion_embeds: torch.Tensor,
        lesion_probs: torch.Tensor,
        lesion_threshold: torch.Tensor,
        return_attn: bool = False,
    ):
        if lesion_embeds.ndim != 3:
            raise ValueError(f"lesion_embeds 期望 [B,L,D]，但得到 {lesion_embeds.shape}")
        if lesion_probs.ndim != 2:
            raise ValueError(f"lesion_probs 期望 [B,L]，但得到 {lesion_probs.shape}")

        B, L, D = lesion_embeds.shape
        if L != self.num_lesions:
            raise ValueError(f"lesion_embeds L={L} 与 num_lesions={self.num_lesions} 不一致")
        if D != self.embed_dim:
            raise ValueError(f"lesion_embeds D={D} 与 embed_dim={self.embed_dim} 不一致（请对齐 lesion_head.embed_dim）")
        
        u = lesion_probs - lesion_threshold.unsqueeze(0)
        if self.positive_only:
            u = torch.relu(u)
        
        W = self.get_W().to(lesion_embeds.device, dtype=lesion_embeds.dtype)

        kv = self.ln_kv(lesion_embeds)
        q = self.disease_query.unsqueeze(0).expand(B, -1, -1)
        q = self.ln_q(q)

        attn_bias = (self.prior_bias_scale * W.t()).to(dtype=lesion_embeds.dtype)
        attn_bias = attn_bias.clamp(min=-10.0, max=10.0)

        # attn_w -> [B, C, L]
        disease_embeds, attn_w = self.attn(q, kv, kv, attn_mask=attn_bias, need_weights=True)

        self.last_attn = attn_w.detach()
        
        # uW ~ pos_(S - tau) @ (M_prior + delta)
        uW = (u.unsqueeze(-1) * W.unsqueeze(0))
        corr_logits = torch.einsum("bcl,blc->bc", attn_w, uW)

        if return_attn:
            return corr_logits, attn_w
        return corr_logits

class LesionCalibModel2CA(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        lesion_head: "LesionQueryHead",
        mapper: "LesionToDiseaseMapper",
        num_classes: int,
        alpha_init: float = 1.0,
        tau_init: float = 0.3,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.lesion_head = lesion_head
        self.mapper = mapper
        self.num_classes = int(num_classes)
        self.freeze_backbone = bool(freeze_backbone)

        # NOTE: 逐类 alpha 改成了 shape=[C], for seperatable explain.
        self.alpha = nn.Parameter(
            torch.ones(self.num_classes, dtype=torch.float32) * float(alpha_init)
        )
        n_lesions = getattr(lesion_head, "num_lesions", None)
        if n_lesions is None:
            n_lesions = int(getattr(mapper, "num_lesions", mapper.M_prior.shape[0]))
        self.lesion_threshold = nn.Parameter(torch.ones(n_lesions, dtype=torch.float32) * float(tau_init))
    
    def forward(
        self,
        img_hr: torch.Tensor,
        patches: torch.Tensor,
        backbone_img_size: int,
        return_attn: bool=False,
    ):
        img_backbone = F.interpolate(
            img_hr, size=(int(backbone_img_size), int(backbone_img_size)),
            mode="bicubic", align_corners=False
        )

        if self.freeze_backbone:
            with torch.no_grad():
                base_logits = self.backbone(img_backbone)  # [B, C]
        else:
            base_logits = self.backbone(img_backbone)      # [B, C]

        if return_attn:
            lesion_probs, lesion_embeds, attn1 = self.lesion_head(
                patches, return_embeds=True, return_attn=True
            )  # probs:[B,L], embeds:[B,L,D], attn1:[B,L,N]
            corr_logits, attn2 = self.mapper(
                lesion_embeds, lesion_probs, self.lesion_threshold, return_attn=True
            )  # corr:[B,C], attn2:[B,C,L]
        else:
            lesion_probs, lesion_embeds = self.lesion_head(
                patches, return_embeds=True, return_attn=False
            )  # probs:[B,L], embeds:[B,L,D]
            corr_logits = self.mapper(
                lesion_embeds, lesion_probs, self.lesion_threshold, return_attn=False
            )  # corr:[B,C]
            attn1 = attn2 = None

        # ---- fusion ----
        final_logits = base_logits + self.alpha * corr_logits

        if return_attn:
            extra = {
                "attn_lesion_patch": attn1,
                "attn_disease_lesion": attn2,
                "lesion_threshold": self.lesion_threshold.detach(),
            }
            return base_logits, corr_logits, final_logits, lesion_probs, extra

        return base_logits, corr_logits, final_logits, lesion_probs