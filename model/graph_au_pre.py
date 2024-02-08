import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .swin_transformer import swin_transformer_base

class Graph_au_net(nn.Module):
    def __init__(self, au_num_classes=12, backbone='swin_transformer_base', neighbor_num=4, metric='dots'):
        super(Graph_au_net, self).__init__()
        if 'transformer' in backbone:
            if backbone == 'swin_transformer_base':
                self.backbone = swin_transformer_base()
            self.in_channels = self.backbone.num_features
            self.out_channels = self.in_channels
            self.backbone.head = None
        else:
            raise Exception("Error: wrong backbone name: ", backbone)

        self.global_linear = LinearBlock(self.in_channels, self.out_channels)
        self.head = Head(self.out_channels, au_num_classes, neighbor_num, metric)
        self.full_connect = Linear_Net(au_num_classes = au_num_classes)

    def forward(self, x):
        x = self.backbone(x) 
        x = self.global_linear(x) 
        x = self.head(x)
        cl = self.full_connect(x)
        return cl
    
class Graph_au_net_NCE(nn.Module):
    def __init__(self, au_num_classes=12, backbone='swin_transformer_base', neighbor_num=4, metric='dots'):
        super(Graph_au_net_NCE, self).__init__()
        if 'transformer' in backbone:
            if backbone == 'swin_transformer_base':
                self.backbone = swin_transformer_base()
            self.in_channels = self.backbone.num_features
            self.out_channels = self.in_channels
            self.backbone.head = None
        else:
            raise Exception("Error: wrong backbone name: ", backbone)

        self.global_linear = LinearBlock(self.in_channels, self.out_channels)
        self.head = Head(self.out_channels, au_num_classes, neighbor_num, metric)

    def forward(self, x):
        x = self.backbone(x) 
        x = self.global_linear(x) 
        cl = self.head(x) 
        return cl

class GNN(nn.Module):
    def __init__(self, in_channels, au_num_classes, neighbor_num=4, metric='dots'):
        super(GNN, self).__init__()
        self.in_channels = in_channels
        self.au_num_classes = au_num_classes
        self.relu = nn.ReLU()
        self.metric = metric
        self.neighbor_num = neighbor_num

        self.U = nn.Linear(self.in_channels,self.in_channels)
        self.V = nn.Linear(self.in_channels,self.in_channels)
        self.bnv = nn.BatchNorm1d(au_num_classes)

        self.U.weight.data.normal_(0, math.sqrt(2. / self.in_channels))
        self.V.weight.data.normal_(0, math.sqrt(2. / self.in_channels))
        self.bnv.weight.data.fill_(1)
        self.bnv.bias.data.zero_()

    def forward(self, x):
        b, n, c = x.shape
        if self.metric == 'dots':
            si = x.detach()
            si = torch.einsum('b i j , b j k -> b i k', si, si.transpose(1, 2)) 
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(b, n, 1) 
            adj = (si >= threshold).float() 

        elif self.metric == 'cosine':
            si = x.detach()
            si = F.normalize(si, p=2, dim=-1)
            si = torch.einsum('b i j , b j k -> b i k', si, si.transpose(1, 2))
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(b, n, 1)
            adj = (si >= threshold).float()

        elif self.metric == 'l1':
            si = x.detach().repeat(1, n, 1).view(b, n, n, c)
            si = torch.abs(si.transpose(1, 2) - si)
            si = si.sum(dim=-1)
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=False)[0][:, :, -1].view(b, n, 1)
            adj = (si <= threshold).float()

        else:
            raise Exception("Error: wrong metric: ", self.metric)

        A = normalize_digraph(adj)
        aggregate = torch.einsum('b i j, b j k->b i k', A, self.V(x))
        x = self.relu(x + self.bnv(aggregate + self.U(x)))
        return x


class Head(nn.Module):
    def __init__(self, in_channels, au_num_classes, neighbor_num=4, metric='dots'):
        super(Head, self).__init__()
        self.in_channels = in_channels
        self.au_num_classes = au_num_classes
        class_linear_layers = []
        for i in range(self.au_num_classes):
            layer = LinearBlock(self.in_channels, self.in_channels)
            class_linear_layers += [layer]
        self.class_linears = nn.ModuleList(class_linear_layers)
        self.gnn = GNN(self.in_channels, self.au_num_classes,neighbor_num=neighbor_num,metric=metric)
        self.sc = nn.Parameter(torch.FloatTensor(torch.zeros(self.au_num_classes, self.in_channels)))
        self.relu = nn.ReLU()

        nn.init.xavier_uniform_(self.sc)

    def forward(self, x):
        # AFG
        f_u = []
        for i, layer in enumerate(self.class_linears):
            f_u.append(layer(x).unsqueeze(1))
        f_u = torch.cat(f_u, dim=1)
        f_v = f_u.mean(dim=-2)
        f_v = self.gnn(f_v)
        b, n, c = f_v.shape
        sc = self.sc
        sc = self.relu(sc)
        sc = F.normalize(sc, p=2, dim=-1)
        cl = F.normalize(f_v, p=2, dim=-1) 
        cl = cl.transpose(1,2)
        return cl

def normalize_digraph(A):
    b, n, _ = A.shape
    node_degrees = A.detach().sum(dim = -1)
    degs_inv_sqrt = node_degrees ** -0.5
    norm_degs_matrix = torch.eye(n)
    dev = A.get_device()
    if dev >= 0:
        norm_degs_matrix = norm_degs_matrix.to(dev)
    norm_degs_matrix = norm_degs_matrix.view(1, n, n) * degs_inv_sqrt.view(b, n, 1)
    norm_A = torch.bmm(torch.bmm(norm_degs_matrix,A),norm_degs_matrix)
    return norm_A


class LinearBlock(nn.Module):
    def __init__(self, in_features,out_features=None,drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop)
        self.fc.weight.data.normal_(0, math.sqrt(2. / out_features))
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, x):
        x = self.drop(x)
        x = self.fc(x).permute(0, 2, 1)
        x = self.relu(self.bn(x)).permute(0, 2, 1)
        return x

class Linear_Net(nn.Module):
    def __init__(self, au_num_classes, unit_dim=4):
        super(Linear_Net, self).__init__()
        self.fc = nn.ModuleList(
            [nn.Sequential(
            nn.Linear(1024, unit_dim * 8),
            nn.Linear(unit_dim * 8, 2),                                
        ) for i in range(au_num_classes)])

    def forward(self, x):
        for i in range(len(self.fc)):
            au_output = self.fc[i](x[:,:,i])            
            au_output = au_output.unsqueeze(2)
            au_output = torch.softmax(au_output, dim=1)
            if i == 0:
                aus_output = au_output
            else:
                aus_output = torch.cat((aus_output, au_output), dim=2)   
        return aus_output

if __name__=="__main__":
    model = Graph_au_net_NCE()
    input = torch.randn(1,3,224,224)
    output = model(input)
    print("output_shape:",output.shape)
