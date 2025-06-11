import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange

from uws4vad.utils import get_log
log = get_log(__name__)

def exists(val):
    return val is not None

def attention(q, k, v):
    sim = einsum('b i d, b j d -> b i j', q, k)
    attn = sim.softmax(dim=-1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

def FeedForward(dim, repe = 4, dropout = 0.):
    return nn.Sequential(
        LayerNorm(dim),
        nn.Conv1d(dim, dim * repe, 1),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Conv1d(dim * repe, dim, 1)
    )

# MHRAs (multi-head relation aggregators)
class Focus(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head = 64,
        local_aggr_kernel = 5
    ):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads ## stage2: 64*2h, stage3: 64*16h
        self.norm = nn.BatchNorm1d(dim)
        self.to_v = nn.Conv1d(dim, inner_dim, 1, bias = False)
        self.rel_pos = nn.Conv1d(heads, heads, local_aggr_kernel, padding = local_aggr_kernel // 2, groups = heads)
        self.to_out = nn.Conv1d(inner_dim, dim, 1)

    def forward(self, x):
        x = self.norm(x) #(b*crop,c,t)
        b, c, *_, h = *x.shape, self.heads
        v = self.to_v(x) #(b*crop,c,t)
        v = rearrange(v, 'b (c h) ... -> (b c) h ...', h = h) #(b*ten*64,c/64,32)
        out = self.rel_pos(v)
        out = rearrange(out, '(b c) h ... -> b (c h) ...', b = b)
        return self.to_out(out)

class Glance(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads ## stage1: 64*1h
        self.norm = LayerNorm(dim)
        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(inner_dim, dim, 1)
        self.attn =0

    def forward(self, x):
        ## b*nc, dim, t
        x = self.norm(x)
        shape, h = x.shape, self.heads
        x = rearrange(x, 'b c ... -> b c (...)')
        
        q, k, v = self.to_qkv(x).chunk(3, dim = 1) ## b*nc, dim, t
        q, k, v = map(lambda t: rearrange(t, 'b (h d) n -> b h n d', h = h), (q, k, v)) ## b*nc, 1, t, dim  
        q = q * self.scale
        
        sim = einsum('b h i d, b h j d -> b h i j', q, k) ## att scores
        self.attn = sim.softmax(dim = -1) ## b*nc, 1, t, t prob
        out = einsum('b h i j, b h j d -> b h i d', self.attn, v) ## b*nc, 1, t, dim  weighted sum of values
        ## Concatenation of Head Outputs
        out = rearrange(out, 'b h n d -> b (h d) n', h = h) ## b*nc, dim, t
                
        out = self.to_out(out)
        return out.view(*shape)


class _GlanceFocus(nn.Module):
    def __init__(
        self,
        *,
        dim, ## 64, 128, 1024
        depth, ## 3, 3, 2
        heads, ## 1, 2, 16
        mgfn_type = 'gb', ## 'gb', 'fb', 'fb'
        kernel = 5,
        dim_headnumber = 64,
        ff_repe = 4,
        dropout = 0.,
        attention_dropout = 0.
    ):
        super().__init__()

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            if mgfn_type == 'fb':
                attention = Focus(dim, heads = heads, dim_head = dim_headnumber, local_aggr_kernel = kernel)
            elif mgfn_type == 'gb':
                attention = Glance(dim, heads = heads, dim_head = dim_headnumber, dropout = attention_dropout)
            else: raise ValueError('unknown mhsa_type')

            self.layers.append(nn.ModuleList([
                nn.Conv1d(dim, dim, 3, padding = 1),
                attention,
                FeedForward(dim, repe = ff_repe, dropout = dropout),
            ]))

    def forward(self, x):
        for i, (scc, attention, ffn) in enumerate(self.layers):
            log.debug(f"BACKBONE @ depth {i}")
            x = scc(x) + x  ## encode + residual
            log.debug(f"scc [{i}] {x.shape}")
            x = attention(x) + x ## Glance -> Focus -> Focus
            log.debug(f"att [{i}] {x.shape}")
            x = ffn(x) + x
            log.debug(f"ffn [{i}] {x.shape}")
        return x
    

#########################################
## Glance + Focus (MGFN)
## excepts b, f, t
## return b, f, t
class GlanceFocus(nn.Module):
    def __init__( self, 
        dfeat,
        dims = (64, 128, 1024),
        depths = (3, 3, 2),
        mgfn_types = ( 'gb', 'fb', 'fb'),
        lokernel = 5,
        ff_repe = 4,
        dim_head = 64,
        dropout = 0.,
        attention_dropout = 0.
    ):
        super().__init__()
        
        ## 3*Glance -> 3*Focus -> 2*Focus
        mgfn_types = tuple(map(lambda t: t.lower(), mgfn_types))
        self.stages = nn.ModuleList([])
        for ind, (depth, mgfn_types) in enumerate(zip(depths, mgfn_types)):
            is_last = ind == len(depths) - 1
            stage_dim = dims[ind] ## 64, 128, 1024
            heads = stage_dim // dim_head ## 1, 2, 16

            self.stages.append(nn.ModuleList([
                _GlanceFocus(
                    dim = stage_dim,
                    depth = depth,
                    heads = heads,
                    mgfn_type = mgfn_types,
                    ff_repe = ff_repe,
                    dropout = dropout,
                    attention_dropout = attention_dropout
                ),
                nn.Sequential(
                    LayerNorm(stage_dim),
                    nn.Conv1d(stage_dim, dims[ind + 1], 1, stride = 1),
                ) if not is_last else None
            ]))
        log.debug(f"{self.stages} \n\n\n\n")

    def forward(self, x_in):
        x_gf = x_in
        for i, (backbone, conv) in enumerate(self.stages):
            log.debug(f"MGFN @ STAGE {i} ")
            x_gf = backbone(x_gf)
            log.debug(f"MGFN / after G or F {x_gf.shape}")
            if exists(conv):
                x_gf = conv(x_gf)
                log.debug(f"MGFN / conv dim prep {x_gf.shape}")
        log.debug(f"FM {x_in.shape=} -> {x_gf.shape=}")
        
        return x_gf
        