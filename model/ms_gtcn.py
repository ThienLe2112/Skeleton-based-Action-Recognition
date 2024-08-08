import sys
sys.path.insert(0, '')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.ms_tcn import MultiScale_TemporalConv as MS_TCN
from model.mlp import MLP
from model.activation import activation_factory
from graph.tools import k_adjacency, normalize_adjacency_matrix, normalize_dsg_adjacency_matrix


class UnfoldTemporalWindows(nn.Module):
    def __init__(self, window_size, window_stride, window_dilation=1):
        super().__init__()
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_dilation = window_dilation

        self.padding = (window_size + (window_size-1) * (window_dilation-1) - 1) // 2
        self.unfold = nn.Unfold(kernel_size=(self.window_size, 1),
                                dilation=(self.window_dilation, 1),
                                stride=(self.window_stride, 1),
                                padding=(self.padding, 0))

    def forward(self, x):
        # Input shape: (N,C,T,V), out: (N,C,T,V*window_size)
        N, C, T, V = x[0].shape
        
        x[0] = self.unfold(x[0])
        # Permute extra channels from window size to the graph dimension; -1 for number of windows
        x[0] = x[0].view(N, C, self.window_size, -1, V).permute(0,1,3,2,4).contiguous()
        x[0] = x[0].view(N, C, -1, self.window_size * V)
        # print('x[0].shape: ', x[0].shape)
        
        return x


class SpatialTemporal_MS_GCN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_size,
                 disentangled_agg=False,##Original, disentangled_agg=True
                 use_Ares=False, ##Original, use_Ares=True
                 residual=False,
                 dropout=0,
                 activation='relu'):

        super().__init__()
        self.num_scales = num_scales
        self.window_size = window_size
        self.use_Ares = use_Ares
        A = self.build_spatial_temporal_graph(A_binary, window_size)

        #Edit Code Begin 15
        self.conv = torch.nn.Conv3d(1,1,(300,3,3),padding=(0,1,1),stride=(300,1,1))
        #Edit Code End 15
        if disentangled_agg:
            A_scales = [k_adjacency(A, k, with_self=True) for k in range(num_scales)]
            A_scales = np.concatenate([normalize_adjacency_matrix(g) for g in A_scales])
        else:
            # Self-loops have already been included in A
            # print(A)
            A_scales = [normalize_adjacency_matrix(A) for k in range(num_scales)]
            A_scales = [np.linalg.matrix_power(g, k) for k, g in enumerate(A_scales)]
            A_scales = np.concatenate(A_scales)

        self.A_scales = torch.Tensor(A_scales)
        self.V = len(A_binary)

        if use_Ares:
            self.A_res = nn.init.uniform_(nn.Parameter(torch.randn(self.A_scales.shape)), -1e-6, 1e-6)
        else:
            self.A_res = torch.tensor(0)

        self.mlp = MLP(in_channels * num_scales, [out_channels], dropout=dropout, activation='linear')

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels):
            self.residual = lambda x: x
        else:
            self.residual = MLP(in_channels, [out_channels], activation='linear')

        self.act = activation_factory(activation)

    def build_spatial_temporal_graph(self, A_binary, window_size):
        assert isinstance(A_binary, np.ndarray), 'A_binary should be of type `np.ndarray`'
        V = len(A_binary)
        V_large = V * window_size
        A_binary_with_I = A_binary + np.eye(len(A_binary), dtype=A_binary.dtype)
        # Build spatial-temporal graph
        A_large = np.tile(A_binary_with_I, (window_size, window_size)).copy()
        # print('A_large: ',A_large)
        # print('A_large.shape: ',A_large.shape)
        
        return A_large

    def forward(self, x):
        ##Edit Code Begin 5
        x,xx = x

        # print('x.shape: ',x.shape)
        # print('xx.shape: ',xx.shape)
        # print(x[0,0,0,:])
        # print(xx[:3,0,:])

        # print(xx[:10,:,:])
        # print('x.shape: ',x.shape)
        # print('self.window_size: ',self.window_size)
        
        ##Edit Code End 5
        
        N, C, T, V = x.shape    # T = number of windows
        
        # print('xx.shape: ', xx.shape)
        # Build graphs
        
        ##Original Code Begin
        # A = self.A_scales.to(x.dtype).to(x.device) + self.A_res.to(x.dtype).to(x.device)
        # print('A.shape: ',A.shape)
        # print(A)
        ##Original Code End
        
        ##Edit Code Begin 6
        
        #Calculate AlL DSG Matrix in All sample, all frame

        N_xx, C_xx, V_xx = xx.shape
        xx=xx.cuda()

        A_dsg_frame = torch.zeros((25, 25), dtype=torch.float32).to(x.device)
        A_dsg_all = torch.zeros((25, 25), dtype=torch.float32).to(x.device)
        xx_reshaped = xx.permute(0, 2, 1).contiguous()  # Shape: (N_xx, V_xx, C_xx)

        # Compute the pairwise distances in a vectorized manner
        dists = torch.norm(xx_reshaped[:, :, None, :] - xx_reshaped[:, None, :, :], dim=-1)

        A_dsg=dists
        
        lamda=1
        A_dsg_all=torch.exp(-(A_dsg**2)/lamda)
        A_dsg_all=A_dsg_all[None,None,:,:,:]
        A_dsg_all=self.conv.to('cuda')(A_dsg_all)

        A_binary_with_I = A_dsg_all
        print("A_binary_with_I.shape: ",A_binary_with_I.shape)
        
        A_large = torch.stack([i.repeat(self.window_size, self.window_size).clone() for i in A_binary_with_I[0,0]])
        A = A_large

        # A_scales = [normalize_dsg_adjacency_matrix(A) for k in range(self.num_scales)]
        A_scales = [A for k in range(self.num_scales)]

        
        A_scales = torch.stack([A[i] for i in range(len(A)) for k in range(self.num_scales) ]) 

        A_scales = A_scales[:,None,:,:]
        A_scales = torch.stack([torch.matrix_power(g, k) for i in A_scales for k, g in enumerate(i)])
        
        # if torch.isnan(A_scales).any()==True :
        #     print(torch.isnan(A_scales).any())
        #     print("A_scales: ", A_scales)
        #     return 0

        t_a, v_a, v_a2=A_scales.shape
        
        A_scales=A_scales.reshape(1,1,t_a,v_a,v_a2)
        

        self.A_scales = torch.Tensor(A_scales)
        
        A = A_scales
        A = self.A_scales.to(x.dtype).to(x.device)
        
        # print("A.shape: ",A.shape)
        res = self.residual(x)
        _,_,n,_,_=A.shape

        agg = torch.stack([torch.einsum('vu,ctu->ctv', A[0,0,i], x[i]) for i in range(n)])
        ##Edit Code End 6

        # Perform Graph Convolution
        # res = self.residual(x)
        # agg = torch.einsum('nvu,nctu->nctv', A[0,0], x)
        agg = agg.view(N, C, T, self.num_scales, V)
        agg = agg.permute(0,3,1,2,4).contiguous().view(N, self.num_scales*C, T, V)
        out = self.mlp(agg)
        out += res
        return self.act(out)

