import torch
import torch.nn as nn
import torch.nn.functional as F
import copy, math

from src.model.pstfwd.utils import PstFwdUtils

from omegaconf.dictconfig import DictConfig
from hydra.utils import instantiate as instantiate
from src.utils import get_log
log = get_log(__name__)



############
class SelfAttentionBlock(nn.Module):
    def __init__(self, attention_layer):
        super(SelfAttentionBlock, self).__init__()
        self.layer = attention_layer
        self.size = attention_layer.size
    def forward(self, feature):
        feature_sa = self.layer(feature, feature, feature)
        return feature_sa

class CrossAttentionBlock(nn.Module):
    def __init__(self, attention_layer):
        super(CrossAttentionBlock, self).__init__()
        self.layer = attention_layer
        self.size = attention_layer.size
    def forward(self, video, audio):
        video_cma = self.layer(video, audio, audio)
        audio_cma = self.layer(audio, video, video)
        return video_cma, audio_cma

class TransformerLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
    def forward(self, q, k, v):
        q = self.sublayer[0](q, lambda q: self.self_attn(q, k, v)[0])
        return self.sublayer[1](q, self.feed_forward)

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

def attention(query, key, value, masksize, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if masksize != 1:
        masksize = int(masksize / 2)
        mask = torch.ones(scores.size()).cuda()
        for i in range(mask.shape[2]):
            if i - masksize > 0:
                mask[:, :, i, :i - masksize] = 0
            if i + masksize + 1 < mask.shape[3]:
                mask[:, :, i, masksize + i + 1:] = 0
        # print(mask[0][0])
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, masksize=1, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.masksize = masksize
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                            zip(self.linears, (query, key, value))]
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, self.masksize, dropout=self.dropout)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        out = self.linears[-1](x)
        return out, self.attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = self.w_2(self.dropout(F.relu(self.w_1(x))))
        return output
#############



###############
class Att_MMIL(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Att_MMIL, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    ## topkmean
    def clas(self, logits, seq_len):
        logits = logits.squeeze()
        instance_logits = torch.zeros(0)#.cuda()  # tensor([])
        for i in range(logits.shape[0]):
            if seq_len is None:
                tmp = torch.mean(logits[i]).view(1)
            else:
                tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)
                tmp = torch.mean(tmp).view(1)
            instance_logits = torch.cat((instance_logits, tmp))
        instance_logits = torch.sigmoid(instance_logits)
        return instance_logits

    def forward(self, a_out, v_out, seq_len):
        x = torch.cat([a_out.unsqueeze(-2), v_out.unsqueeze(-2)], dim=-2) ## b, t, 2, 128
        frame_prob = self.fc(x) ## b, t, 2, 1
        
        a_sls = torch.sigmoid(frame_prob[:, :, 0, :]) ## b, t, 1
        v_sls = torch.sigmoid(frame_prob[:, :, 1, :]) ## b, t, 1
        av_sls = frame_prob.sum(dim=2) ## b, t, 1
        mil_vls = self.clas(av_sls, seq_len) ## b
        return mil_vls, a_sls, v_sls, av_sls


class Network_AV(nn.Module):
    def __init__(self):
        super(Network_AV, self).__init__()
        c = copy.deepcopy
        dropout = 0.1 #_cfg.dropout
        nhead = 4 #_cfg.nhead
        hid_dim = 128 #_cfg.hid_dim
        ffn_dim = 128 #_cfg.ffn_dim
        num_classes = 1
        self.multiheadattn = MultiHeadAttention(nhead, hid_dim)
        self.feedforward = PositionwiseFeedForward(hid_dim, ffn_dim)
        self.fc_v = nn.Linear(1024, hid_dim)
        self.fc_a = nn.Linear(128, hid_dim)
        self.cma = CrossAttentionBlock(
            TransformerLayer(hid_dim, 
                            MultiHeadAttention(nhead, hid_dim), 
                            c(self.feedforward), 
                            dropout))
        self.att_mmil = Att_MMIL(hid_dim, num_classes)

    def forward(self, f_a, f_v, seq_len):
        f_v, f_a = self.fc_v(f_v), self.fc_a(f_a) ## b, t, 128
        v_out, a_out = self.cma(f_v, f_a) ## b, t, 128
        mil_vls, a_sls, v_sls, av_sls = self.att_mmil(a_out, v_out, seq_len)
        return {
            'mil_vls': mil_vls, 
            'a_sls': a_sls, 
            'v_sls': v_sls, 
            'av_sls': av_sls, 
            'v_out': v_out, 
            'a_out': a_out
            }


class Network_V(nn.Module):
    def __init__(self, dfeat: int, _cfg: DictConfig, rgs = None ):
        super().__init__()
        dfeat = dfeat[0] #1024 
        dropout = 0.1 #_cfg.do
        nhead = 4 #_cfg.nhead
        hid_dim = 128 #_cfg.hid_dim
        ffn_dim = 128 #_cfg.ffn_dim
        c = copy.deepcopy
        
        self.fc_v = nn.Linear(dfeat, hid_dim)
        self.multiheadattn = MultiHeadAttention(nhead, hid_dim)
        self.feedforward = PositionwiseFeedForward(hid_dim, ffn_dim)
        self.cma = SelfAttentionBlock(
            TransformerLayer(hid_dim, 
                MultiHeadAttention(nhead, hid_dim), 
                c(self.feedforward), 
                dropout)
            )
        self.fc = nn.Linear(hid_dim, 1)
        self.sig = nn.Sigmoid()

    def clas(self, logits, seq_len):
        #logits = logits.squeeze()
        instance_logits = torch.zeros(0) # tensor([])
        for i in range(logits.shape[0]):
            tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)
            tmp = torch.mean(tmp).view(1)
            instance_logits = torch.cat((instance_logits, tmp))
        instance_logits = torch.sigmoid(instance_logits)
        return instance_logits

    def forward(self, x):
        x = self.fc_v(x)  ## b, t, 128
        sa = self.cma(x)  ## b, t, 128  
        sls = self.fc(sa) ## b, t, 1  
        sls = self.sig( sls.squeeze() )
        #mil_vls = self.clas(scores, seq_len) ## b VL -> part of loss process, prefer output at segment level
        return {
            'scores': sls,
            #'mil_vls': mil_vls
            }

class Infer():
    def __init__(self, _cfg, pfu: PstFwdUtils = None): 
        super().__init__()
        self._cfg = _cfg
        self.pfu = pfu
        
    def __call__(self, ndata): 
        scores = self.pfu.uncrop(ndata['scores'], 'mean')
        return scores
    
    
#############
def avce_train(dataloader, model_av, model_v, optimizer_av, optimizer_v, criterion, lamda_a2b, lamda_a2n, logger):
    with torch.set_grad_enabled(True):
        model_av.train()
        model_v.train()
        for i, (f_v, f_a, label) in enumerate(dataloader):
            ## f_v f_a : b, maxseqlen, f
            seq_len = torch.sum(torch.max(torch.abs(f_v), dim=2)[0] > 0, 1)
            ## by doing so t goes back to original shape before pad
            ## or simply reduce amount of pad by consider the max seqlen as TH
            f_v = f_v[:, :torch.max(seq_len), :]
            f_a = f_a[:, :torch.max(seq_len), :]
            f_v, f_a, label = f_v.float().cuda(), f_a.float().cuda(), label.float().cuda()
            
            ## AV
            mil_vls, a_sls, v_sls, _, audio_rep, visual_rep = model_av(f_a, f_v, seq_len)
            a_sls = a_sls.squeeze()
            v_sls = v_sls.squeeze()
            mil_vls = mil_vls.squeeze()
            clsloss = criterion(mil_vls, label) ## bce
            cmaloss_a2v_a2b, cmaloss_a2v_a2n, cmaloss_v2a_a2b, cmaloss_v2a_a2n = CMAL(mil_vls, a_sls,
                                                                                    v_sls, seq_len, audio_rep,
                                                                                    visual_rep)
            loss_av = clsloss + lamda_a2b * cmaloss_a2v_a2b + lamda_a2b * cmaloss_v2a_a2b + lamda_a2n * cmaloss_a2v_a2n + lamda_a2n * cmaloss_v2a_a2n
            unit = dataloader.__len__() // 2
            if i % unit == 0:
                logger.info(f"Current Lambda_a2b: {lamda_a2b:.2f}, Current Lambda_a2n: {lamda_a2n:.2f}")
                logger.info(
                    f"{int(i // unit)}/{2} MIL Loss: {clsloss:.4f}, CMA Loss A2V_A2B: {cmaloss_a2v_a2b:.4f}, CMA Loss A2V_A2N: {cmaloss_a2v_a2n:.4f},"
                    f"CMA Loss V2A_A2B: {cmaloss_v2a_a2b:.4f},  CMA Loss V2A_A2N: {cmaloss_v2a_a2n:.4f}")
            
            ## V
            v_logits = model_v(f_v, seq_len) 
            loss_v = criterion(v_logits, label) ## bce

            optimizer_av.zero_grad()
            optimizer_v.zero_grad()
            model_av.requires_grad = True
            model_v.requires_grad = False
            loss_av.backward()
            optimizer_av.step()

            optimizer_av.zero_grad()
            optimizer_v.zero_grad()
            model_av.requires_grad = False
            model_v.requires_grad = True
            loss_v.backward()
            optimizer_v.step()

        return loss_av, loss_v

