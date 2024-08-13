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

import pandas as pd

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
                 disentangled_agg=True,##Original, disentangled_agg=True
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
            # A_scales = np.concatenate([normalize_adjacency_matrix(g) for g in A_scales])
            A_scales = np.concatenate([g for g in A_scales])

            # print('A_scales: ',pd.DataFrame(A_scales))
            # pd.DataFrame(A_scales).to_csv("./csv/A_scale.csv")
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
        ##Edit Code End 5
        
        # N, C, T, V = x.shape    # T = number of windows
        
        # print('xx.shape: ', xx.shape)
        # Build graphs
        
        ##Original Code Begin
        # A = self.A_scales.to(x.dtype).to(x.device) + self.A_res.to(x.dtype).to(x.device)
        # print('A.shape: ',A.shape)
        # print(A)
        ##Original Code End
        
        ##Edit Code Begin 6
        
        ##Calculate DSG Matrix Method 5
        
        N, C, T, V, M = xx.shape
        xx_reshaped = xx.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        # xx_reshaped = torch.nn.BatchNorm1d(2 * 3 * self.num_point)(xx_reshaped)
        xx_reshaped = xx_reshaped.view(N * M, V, C, T).permute(0,2,3,1).contiguous()
        
        
        xx_reshaped = xx_reshaped.permute(0, 2, 3, 1).contiguous()  # Shape: (N_xx, T_xx, V_xx, C_xx)
        
        # print('xx_reshaped.shape: ',xx_reshaped.shape)
        
        dists = torch.norm(xx_reshaped[:, :, :,None, :] - xx_reshaped[:, :, None, :, :], dim=-1).to("cuda")
        
        # A_dsg = torch.zeros(N_xx, T_xx, V_xx, V_xx)
        
        lamda=1
        
        A_dsg = torch.exp(-(dists**2)/lamda).to("cuda") #Shape: N, T, V, V
        A_large = torch.stack([torch.stack([t.repeat(self.window_size, self.window_size).clone()  for t in i]) for i in A_dsg]).to("cuda")
        N_xx, T_xx, V_xx, _= A_large.shape
        
        
        
        
        # A_scales = torch.zeros( N_xx, T_xx, self.num_scales*V_xx, V_xx)
        
        # for k in range(self.num_scales):
        A_scales = torch.cat([A_large for k in range(self.num_scales)], dim = 2).to("cuda")
        # A_scales = A_large.repeat( N_xx, T_xx, self.num_scales, 1)
        
        # A_Mask = torch.stack([torch.stack([self.A_scales for F in range(T_xx)]).to("cuda") for NN in range(N_xx)]).to("cuda")
        
        
        # A_Mask = self.A_scales.repeat( N_xx, T_xx, 1, 1).to("cuda")
        
        
        
        #Multi_scale
        # A_list = torch.mul(A_scales ,A_Mask).to("cuda")
        
        #Non_Scale
        
        A_list = A_scales

        # import seaborn as sns
        # import pandas as pd
        # import matplotlib.pyplot as plt

        # for i in range(100):
        #     lamda=i
        #     A_dsg_all=torch.exp(-(A_dsg**2)/lamda)
        #     px = pd.DataFrame(A_dsg_all.cpu().numpy())
        #     print(px)
        #     # # glue = sns.load_dataset("glue").pivot(index="Model", columns="Task", values="Score")
        #     sns.heatmap(px, vmin = 0, vmax = 1, cmap = sns.color_palette("Blues", as_cmap=True),annot=True)
        #     plt.title(f'L = {i}')
        #     plt.show()
        
        res = self.residual(x)
        agg = torch.einsum('ntvu,ncfu->ncfv', A_list, x)
        
        
        N, C, T, V = x.shape    # T = number of windows
        
        
        
        
        
        
        ##Edit Code End 6

        # Perform Graph Convolution
        #Origin Code Begin
        # res = self.residual(x)
        # agg = torch.einsum('vu,nctu->nctv', A, x)
        
        #Origin Code End
        agg = agg.view(N, C, T, self.num_scales, V)
        agg = agg.permute(0,3,1,2,4).contiguous().view(N, self.num_scales*C, T, V)
        out = self.mlp(agg)
        out += res
        return self.act(out)

