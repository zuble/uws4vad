import torch
import torch.nn as nn
import torch.nn.functional as F

#from .layers import *
#log = None
#def init(l):
#    global log
#    log = l


class TemporalCrossCovariance(nn.Module):
    def __init__(self):
        super(TemporalCrossCovariance, self).__init__()
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, input1, input2):
        # Subtract mean along the temporal dimension (axis=2)
        mean_subtracted1 = input1 - input1.mean(dim=2, keepdim=True)
        mean_subtracted2 = input2 - input2.mean(dim=2, keepdim=True)
        
        # Batch dot product
        # input1 and input2 shapes are assumed to be (batch, channels, time)
        # We transpose the last two dimensions of input2 for batch matrix multiplication
        cov_temp = torch.bmm(mean_subtracted1, mean_subtracted2.transpose(1, 2)) / 7
        
        # Keep diagonal elements only (zero out others)
        batch_size, channels, _ = cov_temp.shape
        eye = torch.eye(channels).expand(batch_size, channels, channels).to(cov_temp.device)
        cov_temp = cov_temp * eye
        
        # Global max pooling over the time dimension
        # Before applying global max pooling, ensure the dimensionality is right.
        # We need (batch, channels, time), hence we might need to transpose
        pooled_output = self.global_max_pool(cov_temp)

        # Remove extra dimension after pooling
        return pooled_output.squeeze(2)

class DAM(nn.Module):
    def __init__(self, ts=7, data_dim=1024):
        super(DAM, self).__init__()
        self.ts = ts
        self.data_dim = data_dim
        self.lstm_encd = nn.LSTM(data_dim, 1024, batch_first=True)
        self.rgb_attention = nn.Linear(1024, 1)

    def forward(self, x, x_shift):
        # Computing temporal cross-covariance
        print("\nTEMPO\n")
        tmp1 = x - x.mean(dim=2, keepdim=True)
        tmp2 = x_shift - x_shift.mean(dim=2, keepdim=True)
        print(f"{tmp1.shape} {tmp2.shape}")
        
        cov_temp = torch.bmm(tmp1, tmp2.transpose(1, 2)) / self.ts
        print(f"{cov_temp.shape=} ")
        
        cov_temp = cov_temp * torch.eye(cov_temp.size(-1)).to(cov_temp.device)
        cov_temp = cov_temp.max(dim=1)[0] ## bs, t
        print(f"{cov_temp.shape=} ")


        ## Computing channel cross-covariance
        print("\nCHANEL\n")
        tmp1 = x - x.mean(dim=1, keepdim=True)
        tmp2 = x_shift - x_shift.mean(dim=1, keepdim=True)
        print(f"{tmp1.shape} {tmp2.shape}")
        
        cov_channel = torch.bmm(tmp1.transpose(1, 2), tmp2) / self.data_dim
        print(f"{cov_channel.shape=} ")
        
        cov_channel = cov_channel * torch.eye(cov_channel.size(-1)).to(cov_channel.device)
        cov_channel = cov_channel.max(dim=1)[0] ## excepted bs, f 
        print(f"{cov_channel.shape=} ")

        ## Preparing for multiplication
        ## bs, 1, t -> bs, f, t -> bs, t, f
        #cov_temp = cov_temp.unsqueeze(1).repeat(1, self.data_dim, 1)
        #cov_temp = cov_temp.permute(0, 2, 1)
        ## bs, t, 1 -> bs, t, f
        cov_temp = cov_temp.unsqueeze(2).repeat(1, 1, self.data_dim)
        
        ## bs, 1, f -> bs, t, f
        cov_channel = cov_channel.unsqueeze(1).repeat(1, self.ts, 1)

        dist = cov_temp * cov_channel
        print(f"{cov_temp.shape} * {cov_channel.shape} -> {dist.shape} ")
        
        # Attention and final output layers
        spatial_attn_output, _ = self.lstm_encd(dist)
        
        ## chanelA
        spatial_attn = F.softmax(spatial_attn_output, dim=1)
        ## clipA
        spatial_attn_output = torch.tanh(spatial_attn_output)
        temporal_attn = torch.sigmoid(self.rgb_attention(spatial_attn_output))

        return spatial_attn, temporal_attn

class MLP(nn.Module):
    def __init__(self, n_neuron=64):
        super(MLP, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.dense1 = nn.Linear(1024, n_neuron)
        self.dense2 = nn.Linear(n_neuron, 1)

    def forward(self, x):
        x = self.dropout(x)
        x = F.relu(self.dense1(x))
        x = torch.sigmoid(self.dense2(x))
        return x

class DAMLSTM(nn.Module):
    def __init__(self, data_dim=1024, n_neuron=64, ts=7):
        super(DAMLSTM, self).__init__()
        self.lstm = nn.LSTM(data_dim, 1024, batch_first=True)
        self.dam = DAM(ts, data_dim)
        self.mlp = MLP(n_neuron)

    def forward(self, x):
        _, t, f = x.shape

        tmp_x = x.view(2, 32, t, f)
        x_shift = torch.cat((tmp_x[:,:-1,:,:],torch.zeros(2, 1, t, f)), dim=1)
        x_shift = x_shift.view(-1, t, f)
        print(f"{tmp_x.shape=} {x_shift.shape=}")
        
        
        out_lstm, _ = self.lstm(x)
        out_lstm = out_lstm[:,-1,:]
        print(f"{out_lstm.shape=}")
        
        spatial_attn, temporal_attn = self.dam(x, x_shift)
        print(f"{spatial_attn.shape=} {temporal_attn.shape=}")
        
        
        spatial_attn_multiply = out_lstm * spatial_attn
        spatial_attn_add = spatial_attn_multiply + out_lstm

        detection_output = self.mlp(spatial_attn_add)

        detection_attn_multiply = detection_output * temporal_attn
        detection_attn_add = detection_output + detection_attn_multiply

        return detection_attn_add


if __name__ == 'main':
    
    net = DAMLSTM(data_dim=1024, n_neuron=64)

    bs = 2
    seglen = 32
    ts = 7
    f = 1024

    x = torch.randn(bs * seglen, ts, f)
    output = net(x)