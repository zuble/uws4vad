import torch
from Transformer import *

## https://github.com/shengyangsun/MSBT

class MultiScale_Bottleneck_Transformer(nn.Module):
    def __init__(self, hid_dim, n_head, dropout, n_bottleneck=8, bottleneck_std=0.15):
        super(MultiScale_Bottleneck_Transformer, self).__init__()
        self.n_layers = int(math.log2(n_bottleneck)) + 1
        self.sma = nn.ModuleList([
            TransformerLayer(hid_dim, MultiHeadAttention(h=n_head, d_model=hid_dim), PositionwiseFeedForward(hid_dim, hid_dim), dropout=dropout)
            for _ in range(self.n_layers)])
        self.decoder = TransformerLayer(hid_dim, MultiHeadAttention(h=n_head, d_model=hid_dim), PositionwiseFeedForward(hid_dim, hid_dim), dropout=dropout)
        self.bottleneck_list = nn.ParameterList([
            nn.Parameter(nn.init.normal_(torch.zeros(1, int(n_bottleneck / (2 ** layer_i)), hid_dim).cuda(), std=bottleneck_std))
            for layer_i in range(self.n_layers)])

    def forward(self, m_a, m_b):
        n_batch = m_a.shape[0]
        n_modality = m_a.shape[1]
        bottleneck = self.bottleneck_list[0]
        bottleneck = bottleneck.repeat([n_batch, 1, 1])
        m_a_in, m_b_in = m_a, m_b
        for layer_i in range(self.n_layers):
            m_a_cat = torch.cat([m_a_in, bottleneck], dim=1)
            m_a_cat = self.sma[layer_i](m_a_cat, m_a_cat, m_a_cat)
            m_a_in = m_a_cat[:, :n_modality, :]
            m_a_bottleneck = m_a_cat[:, n_modality:, :]
            if layer_i < self.n_layers - 1:
                next_bottleneck = self.bottleneck_list[layer_i + 1]
                next_bottleneck = next_bottleneck.repeat([n_batch, 1, 1])
                bottleneck = self.decoder(next_bottleneck, m_a_bottleneck, m_a_bottleneck)
            m_b_cat = torch.cat([m_b_in, m_a_bottleneck], dim=1)
            m_b_cat = self.sma[layer_i](m_b_cat, m_b_cat, m_b_cat)
            m_b_in = m_b_cat[:, :n_modality, :]

        return m_b_in, m_a_bottleneck
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
from Transformer import *
from MultiScaleBottleneckTransformer import MultiScale_Bottleneck_Transformer

def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


def infoNCE(f_a, f_b, seq_len, temperature=0.5):
    n_batch = f_a.shape[0]
    total_loss = .0
    for i in range(n_batch):
        exp_mat = torch.exp(torch.mm(f_a[i][:seq_len[i]], f_b[i][:seq_len[i]].t()) / temperature)
        positive_mat = torch.diag(exp_mat)
        exp_mat_transpose = exp_mat.t()
        loss_i = torch.mean(-torch.log(positive_mat / torch.sum(exp_mat, dim=-1)))
        loss_i += torch.mean(-torch.log(positive_mat / torch.sum(exp_mat_transpose, dim=-1)))
        total_loss += loss_i
    return total_loss / n_batch


class MultimodalTransformer(nn.Module):
    def __init__(self, args):
        super(MultimodalTransformer, self).__init__()
        dropout = args.dropout
        nhead = args.nhead
        hid_dim = args.hid_dim
        ffn_dim = args.ffn_dim
        n_transformer_layer = args.n_transformer_layer
        n_bottleneck = args.n_bottleneck
        self.fc_v = nn.Linear(args.v_feature_size, hid_dim)
        self.fc_a = nn.Linear(args.a_feature_size, hid_dim)
        self.fc_f = nn.Linear(args.f_feature_size, hid_dim)
        self.msa = SelfAttentionBlock(TransformerLayer(hid_dim, MultiHeadAttention(nhead, hid_dim), PositionwiseFeedForward(hid_dim, ffn_dim), dropout))
        self.bottle_msa = SelfAttentionBlock(TransformerLayer(hid_dim, MultiHeadAttention(nhead, hid_dim), PositionwiseFeedForward(hid_dim, ffn_dim), dropout))
        self.MST = MultiScale_Bottleneck_Transformer(hid_dim, n_head=nhead, dropout=dropout, n_bottleneck=n_bottleneck, bottleneck_std=args.bottleneck_std)
        d_mmt = hid_dim * 6
        h_mmt = 6
        self.mm_transformer = MultilayerTransformer(TransformerLayer(d_mmt, MultiHeadAttention(h=h_mmt, d_model=d_mmt), PositionwiseFeedForward(d_mmt, d_mmt), dropout), n_transformer_layer)
        self.MIL = MIL(d_mmt)
        self.regressor = nn.Sequential(
            nn.Linear(hid_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.Dropout(0.3),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, f_a, f_v, f_f, seq_len):  # audio, RGB, flow
        f_a, f_v, f_f = self.fc_a(f_a), self.fc_v(f_v), self.fc_f(f_f)
        f_a, f_v, f_f = self.msa(f_a), self.msa(f_v), self.msa(f_f)
        f_av, b_av = self.MST(f_a, f_v)
        f_va, b_va = self.MST(f_v, f_a)
        f_af, b_af = self.MST(f_a, f_f)
        f_fa, b_fa = self.MST(f_f, f_a)
        f_vf, b_vf = self.MST(f_v, f_f)
        f_fv, b_fv = self.MST(f_f, f_v)
        bottle_cat = torch.cat([b_av, b_va, b_af, b_fa, b_vf, b_fv], dim=1)
        bottle_cat = self.bottle_msa(bottle_cat)
        bottle_weight = self.regressor(bottle_cat)

        loss_infoNCE = .0
        if seq_len != None:
            cnt_n = 0
            n_av, n_va, n_af, n_fa, n_vf, n_fv = normalize(f_av, f_va, f_af, f_fa, f_vf, f_fv)
            n_list = [n_av, n_va, n_af, n_fa, n_vf, n_fv]
            for i in range(len(n_list)):
                for j in range(i + 1, len(n_list)):
                    cnt_n += 1
                    loss_infoNCE += infoNCE(n_list[i], n_list[j], seq_len)
            loss_infoNCE = loss_infoNCE / cnt_n

        f_av, f_va, f_af, f_fa, f_vf, f_fv = [
            bottle_weight[:, i, :].view([-1, 1, 1]) * f
            for i, f in enumerate([f_av, f_va, f_af, f_fa, f_vf, f_fv])
        ]
        f_avf = torch.cat([f_av, f_va, f_af, f_fa, f_vf, f_fv], dim=-1)

        f_avf = self.mm_transformer(f_avf)
        MIL_logits = self.MIL(f_avf, seq_len)
        return MIL_logits, loss_infoNCE


class MIL(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.6):
        super(MIL, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(512, 32), nn.Dropout(dropout_rate),
            nn.Linear(32, 1), nn.Sigmoid())

    def filter(self, logits, seq_len):
        instance_logits = torch.zeros(0).cuda()
        for i in range(logits.shape[0]):
            if seq_len is None:
                return logits
            else:
                tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)
                tmp = torch.mean(tmp).view(1)
            instance_logits = torch.cat((instance_logits, tmp))
        return instance_logits

    def forward(self, avf_out, seq_len):
        avf_out = self.regressor(avf_out)
        avf_out = avf_out.squeeze()
        mmil_logits = self.filter(avf_out, seq_len)
        return mmil_logits
    
    
    
def MSBT_train(args, dataloader, model_MT, optimizer_MT, criterion, logger):
    with torch.set_grad_enabled(True):
        model_MT.train()
        for i, (f_v, f_a, f_f, label) in enumerate(dataloader):
            seq_len = torch.sum(torch.max(torch.abs(f_v), dim=2)[0] > 0, 1)
            f_v = f_v[:, :torch.max(seq_len), :]
            f_a = f_a[:, :torch.max(seq_len), :]
            f_f = f_f[:, :torch.max(seq_len), :]
            f_v, f_a, f_f, label = f_v.float().cuda(), f_a.float().cuda(), f_f.float().cuda(), label.float().cuda()
            MIL_logits, loss_TCC = model_MT(f_a, f_v, f_f, seq_len)
            loss_MIL = criterion(MIL_logits, label)
            loss_TCC = args.lambda_infoNCE * loss_TCC
            total_loss = loss_MIL + loss_TCC
            logger.info(f"Current batch: {i}, Loss: {total_loss:.4f}, MIL: {loss_MIL:.4f}, TCC: {loss_TCC:.4f}")
            optimizer_MT.zero_grad()
            total_loss.backward()
            optimizer_MT.step()


def MSBT_test(dataloader, model_MT, gt):
    with torch.no_grad():
        model_MT.eval()
        pred = torch.zeros(0).cuda()
        for i, (f_v, f_a, f_f) in tqdm(enumerate(dataloader)):
            f_v, f_a, f_f = f_v.cuda(), f_a.cuda(), f_f.cuda()
            logits, _ = model_MT(f_a, f_v, f_f, seq_len=None)
            logits = torch.mean(logits, 0)
            pred = torch.cat((pred, logits))
        pred = list(pred.cpu().detach().numpy())
        precision, recall, th = precision_recall_curve(list(gt), np.repeat(pred, 16))
        ap = auc(recall, precision)
        return ap