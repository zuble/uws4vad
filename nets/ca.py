import torch
from torch import nn
import torch.nn.functional as F



class GCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.residual = False
        self.A = False
        self.C = True
        self.CM = False
        self.width = 3
        if self.C:
            self.theta = nn.Linear(in_channels, in_channels)
            self.phi = nn.Linear(in_channels, in_channels)
        self.conv_d = nn.Linear(in_channels, out_channels)
        
        if self.residual:
            self.down = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1))

    def forward(self, x):
        t, c = x.size()
        A, M = self.generate_A(t, self.width)
        M = M.detach()
        if self.A:
            A = A.detach()
        else:
            A = 0.
        if self.C:
            theta = self.theta(x)
            phi = self.phi(x)
            C = torch.mm(theta, phi.permute(1, 0))
            if self.CM:
                tmp = torch.exp(C - torch.max(C*M, dim=-1, keepdim=True)[0]) * M
                A += tmp / tmp.sum(dim=-1, keepdim=True)
            else:
                A += F.softmax(C, dim=-1)
        if self.residual:
            out = self.conv_d(torch.bmm(A, x.permute(0, 2, 1)).permute(0, 2, 1)) + self.down(x)
        else:
            out = self.conv_d(torch.mm(A, x))
        return out

    @staticmethod
    def generate_A(dim, width=3):
        A = torch.zeros(dim, dim, requires_grad=False) #, device='cuda'
        min_value = -(width - 1) // 2
        extent = [min_value+i for i in range(width)]
        for i in range(dim):
            for j in extent:
                if i+j >=0 and i+j <=dim-1:
                    A[i, i+j] = 1.
        M = A
        A = A/A.sum(dim=1, keepdim=True)
        return A, M

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.conv = nn.Conv2d(1024, 256, 1)
        self.gcn = GCN(256, 256)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = x.view(-1, 1024, 1, 1)
        x = self.conv(x) ## seglen, 256, 1, 1
        # x = torch.tanh(x)
        x = F.relu(x)
        x = x.view(-1, 256)
        A = self.gcn(x)
        x = self.fc(A) ## seglen, 1
        
        mask = torch.sigmoid(x) + 1e-5
        ## reverses the AW -> represent the normal possibility of the segments.
        inverse_mask = torch.reciprocal(mask)
        return mask, inverse_mask

class Classification(nn.Module):
    def __init__(self):
        super(Classification, self).__init__()
        self.fc = nn.Linear(1024, 1)

    def forward(self, x):
        return self.fc(x)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.attention = Attention()
        self.classification = Classification()

    def forward(self, x):
        ## x is except to be (seglen, f)
        ## AW -> SL anom 
        ## rev_AW -> SL norm 
        mask, inverse_mask = self.attention(x) ## seglen,1  seglen,1
        
        ## GLOBAL VL FEAT
        video_feature = torch.sum(x * mask, dim=0, keepdim=True) / torch.sum(mask) ## 1,f
        video_score = self.classification(video_feature) ## 1,1
        
        ## GLOBAL VL FEAT -> loss applied only to ANOM inputs 
        ## We input the  xconv into our classifier 
        # and maximize the probability of the classifier making a mistake to refine the attention module.
        inverse_video_feature = torch.sum(x * inverse_mask, dim=0, keepdim=True) / torch.sum(inverse_mask) ## 1,f
        inverse_video_score = self.classification(inverse_video_feature) ## 1,1

        segments_scores = self.classification(x) ## seglen, 1
        return video_score, inverse_video_score, mask, segments_scores


def tower_loss(net, features, labels, dims, args):
    loss = []
    inverse_loss = []
    sum_sal_loss = []
    labels = torch.from_numpy(labels).cuda()
    for i in range(len(features)):
        feature = torch.from_numpy(features[i]).cuda()
        video_score, inverse_video_score, mask, seg_scores = net(feature)
        entropy_loss = F.binary_cross_entropy_with_logits(video_score, labels[i: i+1, :])
        margin = torch.max(torch.tensor(0., device='cuda', requires_grad=False), (torch.sigmoid(seg_scores) - mask) ** 2 - args.sal_ratio ** 2)
        count_nonzero = (margin != 0.).sum().detach().to(torch.float32)
        sal_loss = torch.sum(margin) / (count_nonzero + 1e-6)
        inverse_entropy_loss = labels[i, 0] * F.binary_cross_entropy_with_logits(inverse_video_score, torch.tensor([[0.]], requires_grad=False, device='cuda'))
        loss.append(entropy_loss)
        inverse_loss.append(inverse_entropy_loss + args.sal_coe * sal_loss)
        sum_sal_loss.append(args.sal_coe * sal_loss)
    return sum(loss) / args.batch_size, sum(inverse_loss) / args.batch_size, sum(sum_sal_loss) / args.batch_size

def parse_args():
    parser = argparse.ArgumentParser()
    # input for training
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--iterations', default=67, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', default=0.5e-3, type=float)
    parser.add_argument('--restore', default=False, type=bool)
    parser.add_argument('--sal_coe', default=0.5, type=float)
    parser.add_argument('--weight_decay', default=0.2e-5, type=float)
    parser.add_argument('--sal_ratio', default=0.3, type=float)
    parser.add_argument('--save_path', default='./checkpoints/', type=str)
    parser.add_argument('--gpu_list', default=[0], type=list)
    parser.add_argument('--TEST', default=True, type=bool)

    parser.add_argument('--A', action='store_false')
    parser.add_argument('--C', action='store_true')
    parser.add_argument('--CM', action='store_false')
    parser.add_argument('--residual', action='store_true')
    parser.add_argument('--num_gcn', default=1, type=int)
    parser.add_argument('--width', default=3, type=int)
    args = parser.parse_args()
    return args
def train():
    args = parse_args()
    print('Hyper-parameters:')
    d_args = vars(args)
    for i in d_args:
        print('{}: {}'.format(i, d_args[i]))
    gpu_list = args.gpu_list
    num_gpus = len(gpu_list)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in gpu_list])
    net = model.Network(args)
    net.to('cuda')
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_ass = torch.optim.Adam(net.attention.parameters(), lr=args.lr)
    train_data = input_data.InputData(train_feature_code_path, shuffle=True)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    for i in range(args.epochs):
        print('[*] Current epochs: %d ---' % i)
        sum_loss = 0.
        sum_inverse_loss = 0.
        sum_sum_sal_loss = 0.
        for j in range(args.iterations):
            list_features, numpy_labels, numpy_dims = train_data.next_batch(size=args.batch_size)
            loss, inverse_loss, sum_sal_loss = tower_loss(net, list_features, numpy_labels, numpy_dims, args)
            optimizer.zero_grad()
            optimizer_ass.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            inverse_loss.backward()
            optimizer_ass.step()
            sum_loss += loss.item()
            sum_inverse_loss += inverse_loss.item()
            sum_sum_sal_loss += sum_sal_loss.item()
        print('Loss: {:.3f}, Inverse Loss: {:.3f}, sal_loss: {:.3f}'.format(sum_loss / args.iterations, sum_inverse_loss / args.iterations, sum_sum_sal_loss / args.iterations))
        if i > 50:
            torch.save(net.state_dict(), args.save_path + '{}.param'.format(i))
    if args.TEST:
        test(args)


if __name__ == '__main__':

    net = Network()

    bs = 2
    seglen = 32
    f = 1024

    x = torch.randn(seglen, f)
    output = net(x)