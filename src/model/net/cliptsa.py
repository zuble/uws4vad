import torch
import torch.nn as nn
import torch.nn.init as torch_init
from utils.hard_attention import HardAttention
import itertools
import numpy as np
from scipy.linalg import pascal

torch.set_default_tensor_type('torch.cuda.FloatTensor')

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

################
class MLP(nn.Module):
    def __init__(self, input_dim=512):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        out = self.sigmoid(self.fc2(x))
        return out


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                        kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                            kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                            kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                        kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z

class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                            inter_channels=inter_channels,
                                            dimension=1, sub_sample=sub_sample,
                                            bn_layer=bn_layer)

class Aggregate(nn.Module):
    def __init__(self, len_feature):
        super(Aggregate, self).__init__()
        bn = nn.BatchNorm1d
        self.len_feature = len_feature
        self.division = 2048 // len_feature
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=512 // self.division, kernel_size=3,
                    stride=1,dilation=1, padding=1),
            nn.ReLU(),
            bn(512 // self.division)
            # nn.dropout(0.7)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=512 // self.division, kernel_size=3,
                    stride=1, dilation=2, padding=2),
            nn.ReLU(),
            bn(512 // self.division)
            # nn.dropout(0.7)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=512 // self.division, kernel_size=3,
                    stride=1, dilation=4, padding=4),
            nn.ReLU(),
            bn(512 // self.division)
            # nn.dropout(0.7),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv1d(in_channels=2048 // self.division, out_channels=512 // self.division, kernel_size=1,
                    stride=1, padding=0, bias = False),
            nn.ReLU(),
            # nn.dropout(0.7),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv1d(in_channels=2048 // self.division, out_channels=2048 // self.division, kernel_size=3,
                    stride=1, padding=1, bias=False), # should we keep the bias?
            nn.ReLU(),
            nn.BatchNorm1d(2048 // self.division),
            # nn.dropout(0.7)
        )

        self.non_local = NONLocalBlock1D(512 // self.division, sub_sample=False, bn_layer=True)


    def forward(self, x):
            # x: (B, T, F) -> torch.Size([10, 28, 2048])
            out = x.permute(0, 2, 1) # -> torch.Size([10, 2048, 28])
            residual = out

            out1 = self.conv_1(out)
            out2 = self.conv_2(out)

            out3 = self.conv_3(out)
            out_d = torch.cat((out1, out2, out3), dim = 1)
            out = self.conv_4(out)
            out = self.non_local(out)
            out = torch.cat((out_d, out), dim=1)
            out = self.conv_5(out)   # fuse all the features together
            out = out + residual
            out = out.permute(0, 2, 1)
            # out: (B, T, 1)  => [10, 28, 2048]

            return out
################


#################
class PerturbedTopK(nn.Module):
    def __init__(self, k: float, num_samples: int=1000, sigma: float=0.05):
        super(PerturbedTopK, self).__init__()
        self.num_samples = num_samples
        self.sigma = sigma
        self.k = k

    def __call__(self, x, train_mode):
        return PerturbedTopKFunction.apply(x, self.k, self.num_samples, self.sigma, train_mode)

class PerturbedTopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k: float, num_samples:int=1000, sigma: float=0.05, train_mode: bool=True): # k = top-k
        b, t = x.shape  ## b*nc, t
        
        k = int(t * k) ## t*0.95, if t=32->k=30
        print(k)
        # for Gaussian: noise and gradient are the same.
        noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples, t)).to(x.device)
        perturbed_x = x[:, None, :] + noise * sigma # b, n_samples , t


        if k > perturbed_x.shape[-1]:
            k = perturbed_x.shape[-1]
        elif k == 0:
            k = 1

        # k = max(3, k)
        if not train_mode:
            k = min(1000, k)

        #topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=False) 
        #indices = topk_results.indices # b, n_samples , k
        #valus = torch.sort(indices, dim=-1).values 
        topk_values = torch.topk(perturbed_x, k=k, dim=-1, sorted=True)[1]  ## b, n_samples , k
        
        perturbed_output = torch.nn.functional.one_hot(topk_values, num_classes=t).float()
        # b, n_samples, k, t
        indicators = perturbed_output.mean(dim=1) # b, k, t

        # constants for backward
        ctx.k = k
        ctx.num_samples = num_samples
        ctx.sigma = sigma

        # tensors for backward
        ctx.perturbed_output = perturbed_output
        ctx.noise = noise

        return indicators

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return tuple([ None ] * 5)

        grad_expected = torch.einsum("bnkd,bne->bkde", ctx.perturbed_output, ctx.noise)
        grad_expected /= (ctx.num_samples * ctx.sigma)
        grad_input = torch.einsum("bkde,bke->bd", grad_expected, grad_output)
        return (grad_input,) + tuple([ None ] * 5)

class MLP2(nn.Module):
    def __init__(self, input_dim=512):
        super(MLP2, self).__init__()

        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        out = self.sigmoid(self.fc2(x))
        return out

class HardAttention(nn.Module):
    def __init__(self, k=1.0, num_samples=100, input_dim=512):
        super(HardAttention, self).__init__()
        self.scorer = MLP2(input_dim)
        self.hard_att = PerturbedTopK(k=k, num_samples=num_samples)

    def forward(self, inputs):
        ## b*nc, t, 512
        scores = self.scorer(inputs)
        b, t, _ = scores.shape ## b*nc, t, 1

        if b > 1:
            train_mode = True
        else:
            train_mode = False

        topk = self.hard_att(scores.squeeze(-1), train_mode) ## b*nc, k, t
        out = topk.unsqueeze(-1) * inputs.unsqueeze(1)         
        ## b*nc, k, t, 1 * b*ns, 1, t, 512 -> b*nc, k, t, 512
        out = torch.sum(out, dim=1) ## b*nc, t, 512

        return out

#################



class Model(nn.Module):
    def __init__(self, n_features, batch_size, k=0.95, num_samples=10, apply_HA=True, args=None):
        super(Model, self).__init__()
        args = {}
        visual = 'I3D'
        gpu = [0]
        
        OG_feat = n_features
        if visual.upper() in ["I3D", "C3D"]: # and args.enable_HA:            
            n_features = 512

        self.hard_attention = HardAttention(k=k, num_samples=num_samples, input_dim=n_features)
        self.apply_HA = apply_HA
        self.batch_size = batch_size
        self.num_segments = 32
        self.k_abn = self.num_segments // 10
        self.k_nor = self.num_segments // 10

        self.division = 2048 // n_features

        self.Aggregate = Aggregate(len_feature=2048 // self.division)
        self.fc1 = nn.Linear(n_features, 512 // self.division)
        self.fc2 = nn.Linear(512 // self.division, 128 // self.division)
        self.fc3 = nn.Linear(128 // self.division, 1)

        self.drop_out = nn.Dropout(0.7)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)

        self.parallel = 0.5 if "," in gpu else 1
        self.visual = visual

        self.mlp = MLP(input_dim=OG_feat)

    def forward(self, inputs):
        k_abn = self.k_abn
        k_nor = self.k_nor

        out = inputs # ^torch.Size([64, 10, 32, 2048]), *[64,1,32,512]
        bs, ncrops, t, f = out.size() # => torch.Size([1, 1, 89, 512]), ^torch.Size([64, 10, 32, 2048])

        out = out.view(-1, t, f) # => ^[640, 32, 2048] bs*nc, t, f 

        if f > 512:
            out = self.mlp(out) ## down_res
            f = 512

        ## b*nc, t, 512
        
        if self.apply_HA:
            if self.visual != "vit" and out.shape[0] > 1 and out.shape[1] != 32:
                concat = []
                for i in out:
                    concat.append(self.hard_attention(i.unsqueeze(0)))
                out = torch.cat(concat, dim=0)
            else:
                out = self.hard_attention(out)
        return 
    
        out = self.Aggregate(out) # ^[640, 32, 2048]
        out = self.drop_out(out) # => torch.Size([1, 89, 512]), ^[640, 32, 2048], *[64,32,512]

        features = out # torch.Size([1, 89, 512]), *[64,32,512]
        scores = self.relu(self.fc1(features))
        scores = self.drop_out(scores)
        scores = self.relu(self.fc2(scores))
        scores = self.drop_out(scores)
        scores = self.sigmoid(self.fc3(scores))
        scores = scores.view(bs, ncrops, -1).mean(1) # => ^torch.Size([64, 32])
        scores = scores.unsqueeze(dim=2)    # => torch.Size([1, 89, 1]), ^[64,32,1], *[64, 32, 1]

        ####################
        
        # Place self.hard_attention here and remove MLP inside hard attention
        adjusted_scoremag_batch_size = int(self.batch_size * self.parallel)
        adjusted_feat_batch_size = int(self.batch_size*ncrops * self.parallel)

        normal_features = features[0:adjusted_feat_batch_size] # torch.Size([1, 89, 512]), ^[320, 32, 2048], *[64, 32, 512]
        normal_scores = scores[0:adjusted_scoremag_batch_size]

        abnormal_features = features[adjusted_feat_batch_size:] # torch.Size([0, 89, 512]), ^[320, 32, 2048], *[0, 32, 512], +[32, 32, 512]
        abnormal_scores = scores[adjusted_scoremag_batch_size:]

        feat_magnitudes = torch.norm(features, p=2, dim=2)
        feat_magnitudes = feat_magnitudes.view(bs, ncrops, -1).mean(1) # [1, 89], ^[64, 32], *+[64,32]
        nfea_magnitudes = feat_magnitudes[0:adjusted_scoremag_batch_size]  # normal feature magnitudes
        afea_magnitudes = feat_magnitudes[adjusted_scoremag_batch_size:]  # abnormal feature magnitudes
        n_size = nfea_magnitudes.shape[0] # 1, ^32, +32

        if nfea_magnitudes.shape[0] == 1:  # this is for inference, the batch size is 1
            afea_magnitudes = nfea_magnitudes
            abnormal_scores = normal_scores
            abnormal_features = normal_features # == torch.Size([1, 89, 512])

        #######  process abnormal videos -> select top3 feature magnitude  #######
        select_idx = torch.ones_like(nfea_magnitudes).cuda()
        select_idx = self.drop_out(select_idx)

        afea_magnitudes_drop = afea_magnitudes * select_idx

        idx_abn = torch.topk(afea_magnitudes_drop, k_abn, dim=1)[1]

        idx_abn_feat = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_features.shape[2]]) # => torch.Size([1, 3, 512]), ^[32, 3, 2048], +[32, 3, 512]
        abnormal_features = abnormal_features.view(n_size, ncrops, t, f) # => torch.Size([1, 1, 197, 512]), ^[32, 10, 32, 2048], +[1,32,32,512]
        abnormal_features = abnormal_features.permute(1, 0, 2, 3) # => ^[10, 32, 32, 2048], +[1, 32, 32, 512]

        total_select_abn_feature = torch.zeros(0)
        for abnormal_feature in abnormal_features:
            feat_select_abn = torch.gather(abnormal_feature, 1, idx_abn_feat)   # top 3 features magnitude in abnormal bag
            total_select_abn_feature = torch.cat((total_select_abn_feature, feat_select_abn))

        idx_abn_score = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_scores.shape[2]])
        score_abnormal = torch.mean(torch.gather(abnormal_scores, 1, idx_abn_score), dim=1)  # top 3 scores in abnormal bag based on the top-3 magnitude

        ####### process normal videos -> select top3 feature magnitude #######
        select_idx_normal = torch.ones_like(nfea_magnitudes).cuda()
        select_idx_normal = self.drop_out(select_idx_normal)
        nfea_magnitudes_drop = nfea_magnitudes * select_idx_normal
        
        idx_normal = torch.topk(nfea_magnitudes_drop, k_nor, dim=1)[1]

        idx_normal_feat = idx_normal.unsqueeze(2).expand([-1, -1, normal_features.shape[2]])

        normal_features = normal_features.view(n_size, ncrops, t, f)
        normal_features = normal_features.permute(1, 0, 2, 3)

        total_select_nor_feature = torch.zeros(0)
        for nor_fea in normal_features:
            feat_select_normal = torch.gather(nor_fea, 1, idx_normal_feat)  # top 3 features magnitude in normal bag (hard negative)
            total_select_nor_feature = torch.cat((total_select_nor_feature, feat_select_normal))

        idx_normal_score = idx_normal.unsqueeze(2).expand([-1, -1, normal_scores.shape[2]])
        score_normal = torch.mean(torch.gather(normal_scores, 1, idx_normal_score), dim=1) # top 3 scores in normal bag

        feat_select_abn = total_select_abn_feature
        feat_select_normal = total_select_nor_feature

        return score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_select_abn, feat_select_abn, scores, feat_select_abn, feat_select_abn, feat_magnitudes

f = torch.randn((4, 5, 32, 512))

net = Model(512, 2)

_ = net(f)



######################################

def sparsity(arr, batch_size, lamda2):
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2*loss


def smooth(arr, lamda1):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]

    loss = torch.sum((arr2-arr)**2)

    return lamda1*loss


def l1_penalty(var):
    return torch.mean(torch.norm(var, dim=0))


class SigmoidMAELoss(torch.nn.Module):
    def __init__(self):
        super(SigmoidMAELoss, self).__init__()
        from torch.nn import Sigmoid
        self.__sigmoid__ = Sigmoid()
        self.__l1_loss__ = MSELoss()

    def forward(self, pred, target):
        return self.__l1_loss__(pred, target)


class SigmoidCrossEntropyLoss(torch.nn.Module):
    # Implementation Reference: http://vast.uccs.edu/~adhamija/blog/Caffe%20Custom%20Layer.html
    def __init__(self):
        super(SigmoidCrossEntropyLoss, self).__init__()

    def forward(self, x, target):
        tmp = 1 + torch.exp(- torch.abs(x))
        return torch.abs(torch.mean(- x * target + torch.clamp(x, min=0) + torch.log(tmp)))


class RTFM_loss(torch.nn.Module):
    def __init__(self, alpha, margin):
        super(RTFM_loss, self).__init__()
        self.alpha = alpha
        self.margin = margin
        self.sigmoid = torch.nn.Sigmoid()
        self.mae_criterion = SigmoidMAELoss()
        self.criterion = torch.nn.BCELoss()

    def forward(self, score_normal, score_abnormal, nlabel, alabel, feat_n, feat_a):
        label = torch.cat((nlabel, alabel), 0)
        score_abnormal = score_abnormal
        score_normal = score_normal

        score = torch.cat((score_normal, score_abnormal), 0)
        score = score.squeeze()

        label = label.cuda()

        loss_cls = self.criterion(score, label)  # BCE loss in the score space

        loss_abn = torch.abs(self.margin - torch.norm(torch.mean(feat_a, dim=1), p=2, dim=1))

        loss_nor = torch.norm(torch.mean(feat_n, dim=1), p=2, dim=1)

        loss_rtfm = torch.mean((loss_abn + loss_nor) ** 2)

        loss_total = loss_cls + self.alpha * loss_rtfm

        return loss_total


def train(nloader, aloader, model, batch_size, optimizer, viz, device, args):
    parallel = 0.5 if "," in args.gpu else 1

    with torch.set_grad_enabled(True):
        model.train()
        ninput, nlabel = next(nloader)
        ainput, alabel = next(aloader)    

        if parallel == 0.5:
            adjusted_batch_size = int(batch_size * parallel)

            first_half = torch.cat((ninput[:adjusted_batch_size], ainput[:adjusted_batch_size]), 0).to(device)
            second_half = torch.cat((ninput[adjusted_batch_size:], ainput[adjusted_batch_size:]), 0).to(device)

            input = torch.cat((first_half, second_half), 0).to(device)
        else:
            input = torch.cat((ninput, ainput), 0).to(device)

        score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, \
        feat_normal_bottom, scores, scores_nor_bottom, scores_nor_abn_bag, _ = model(input)  # b*32  x 2048

        scores = scores.view(batch_size * 32 * 2, -1)

        scores = scores.squeeze()
        abn_scores = scores[batch_size * 32:]

        nlabel = nlabel[0:batch_size]
        alabel = alabel[0:batch_size]

        loss_criterion = RTFM_loss(0.0001, 100)
        loss_sparse = sparsity(abn_scores, batch_size, 8e-3)
        loss_smooth = smooth(abn_scores, 8e-4)
        cost = loss_criterion(score_normal, score_abnormal, nlabel, alabel, feat_select_normal, feat_select_abn) + loss_smooth + loss_sparse

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        return {
            "avg_score_abnormal": torch.mean(score_abnormal),
            "avg_score_normal": torch.mean(score_normal),
            "avg_scores": torch.mean(scores),
            "loss": cost.item(),
            "smooth_loss": loss_smooth.item(),
            "sparsity_loss": loss_sparse.item()
        }