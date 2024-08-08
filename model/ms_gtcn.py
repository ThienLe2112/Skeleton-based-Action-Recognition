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
        
        ##Edit Code Begin 6
        #Calculate AlL DSG Matrix in All sample, all frame
        
        
        ##Calculate DSG Matrix Method 1
        # N_xx, C_xx, V_xx = xx.shape
        # xx=xx.cuda()
        # A_dsg_frame = torch.zeros((25, 25), dtype=torch.float32).to(x.device)
        # A_dsg_all = torch.zeros((25, 25), dtype=torch.float32).to(x.device)
        # #Sample each of people
        # for sample in range(N_xx):
        #     dim3d= torch.tensor(xx[sample, :, :]).T
        #     dists = torch.cdist(dim3d,dim3d)
        #     A_dsg_frame += dists
                
        #     #Temporal Pooling: GAP method
        # A_dsg_frame = A_dsg_frame/sample
        # A_dsg_all+=A_dsg_frame
        #     # # Reset for the next sample
        #     # A_dsg_frame.zero_()  
        # A_dsg=A_dsg_all/N_xx
        
        ##Calculate DSG Matrix Method 2
        # N_xx, C_xx, V_xx = xx.shape
        
        # A_dsg_frame = torch.zeros((25, 25), dtype=torch.float32).to(x.device)
        # A_dsg_all = torch.zeros((25, 25), dtype=torch.float32).to(x.device)
        # xx_reshaped = xx.permute(0, 2, 1).contiguous()  # Shape: (N_xx, V_xx, C_xx)

        # # Compute the pairwise distances in a vectorized manner
        # dists = torch.norm(xx_reshaped[:, :, None, :] - xx_reshaped[:, None, :, :], dim=-1)

        
        
        # # Sum the distances across all samples
        # A_dsg_frame = dists.sum(dim=0) / N_xx

        # # Normalize by the number of samples
        # A_dsg_all += A_dsg_frame
        
        # lamda=1
        # A_dsg_all=torch.exp(-(A_dsg_all**2)/lamda)
        

        # A_binary_with_I = A_dsg_all
        # # Build spatial-temporal graph
        # A_large = A_binary_with_I.repeat(self.window_size, self.window_size).clone()

        # A = A_large

        # # A_scales = [normalize_dsg_adjacency_matrix(A) for k in range(self.num_scales)]
        # A_scales = [A for k in range(self.num_scales)]
        
        # # A_scales= A
        # A_scales = [torch.matrix_power(g, k) for k, g in enumerate(A_scales)]
        # A_scales = torch.cat(A_scales)

        # self.A_scales = torch.Tensor(A_scales)

        # A = self.A_scales.to(x.dtype).to(x.device)
        
        ##Calculate DSG Matrix Method 3
        # N_xx, C_xx, V_xx = xx.shape
        # # print("xx.shape: ",xx.shape)
        # xx=xx.cuda()

        # A_dsg_frame = torch.zeros((25, 25), dtype=torch.float32).to(x.device)
        # A_dsg_all = torch.zeros((25, 25), dtype=torch.float32).to(x.device)
        # xx_reshaped = xx.permute(0, 2, 1).contiguous()  # Shape: (N_xx, V_xx, C_xx)

        # # Compute the pairwise distances in a vectorized manner
        # dists = torch.norm(xx_reshaped[:, :, None, :] - xx_reshaped[:, None, :, :], dim=-1)

        # A_dsg=dists
        
        # lamda=1
        # A_dsg_all=torch.exp(-(A_dsg**2)/lamda)
        # A_dsg_all=A_dsg_all[None,None,:,:,:]
        # # A_dsg_all=self.conv.to('cuda')(A_dsg_all)

        # A_binary_with_I = A_dsg_all
        
        # A_large = torch.stack([i.repeat(self.window_size, self.window_size).clone() for i in A_binary_with_I[0,0]])
        # A = A_large
                
        # n_x,v_x,_=A.shape
        # A_scales=torch.zeros(n_x,self.num_scales,v_x,v_x)
        # for i in range(len(A)):
        #     for k in range(self.num_scales):
        #         A_scales[i,k]=A[i]
                
        # A_list=torch.zeros(n_x,self.num_scales*v_x,v_x)
        # for num in range(len(A_scales)):
        #     A_list[num] = torch.cat([torch.matrix_power(g, k) for k, g in enumerate( A_scales[num] )])
            
        # A_scales=A_list    
        # t_a, v_a, v_a2=A_scales.shape
        
        # A_scales=A_scales.reshape(1,1,t_a,v_a,v_a2)
        

        # self.A_scales = torch.Tensor(A_scales)
        
        # A = A_scales
        # A = self.A_scales.to(x.dtype).to(x.device)
        
        # # print("A.shape: ",A.shape)
        # res = self.residual(x)
        # # _,_,n,_,_=A.shape
        # n,_,_,_=x.shape
                
        # agg = torch.stack([torch.einsum('vu,ctu->ctv', A[0,0,i], x[i]) for i in range(n)])
        
        ##Calculate DSG Matrix Method 4
        # print('xx.shape: ',xx.shape)
        
        N, C, T, V, M = xx.shape
        xx_reshaped = xx.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        # xx_reshaped = torch.nn.BatchNorm1d(2 * 3 * self.num_point)(xx_reshaped)
        xx_reshaped = xx_reshaped.view(N * M, V, C, T).permute(0,2,3,1).contiguous()
        
        
        xx_reshaped = xx_reshaped.permute(0, 2, 3, 1).contiguous()  # Shape: (N_xx, T_xx, V_xx, C_xx)
        
        # print('xx_reshaped.shape: ',xx_reshaped.shape)
        
        dists = torch.norm(xx_reshaped[:, :, :,None, :] - xx_reshaped[:, :, None, :, :], dim=-1).to("cuda")
        
        # A_dsg = torch.zeros(N_xx, T_xx, V_xx, V_xx)
        
        lamda=1
        
        A_dsg = torch.exp(-(dists**2)/lamda).to("cuda")
        # print('A_dsg.shape: ',A_dsg.shape)
        
        # A_large = A_dsg.clone()
        
        A_large = torch.stack([torch.stack([t.repeat(self.window_size, self.window_size).clone()  for t in i]) for i in A_dsg]).to("cuda")
        
        
        N_xx, T_xx, V_xx, _= A_dsg.shape
        
        A_scales = torch.zeros(N_xx,T_xx,self.num_scales,A_large.shape[2],A_large.shape[3]).to("cuda")
        
        for k in range(self.num_scales):
            A_scales[:,:, k] = A_large[:,:]
        # print("A_scales.shape: ", A_scales.shape)
        
        # A_list = torch.zeros(N_xx,T_xx,self.num_scales*A_large.shape[2],A_large.shape[2], device = x.device)
        
        # for N in range(N_xx):
        #     for F in range(T_xx):
        A_list = torch.stack([torch.stack([torch.cat([torch.matrix_power(g, k).to("cuda") for k, g in enumerate( A_scales[N,F] )]) for F in range(T_xx)]).to("cuda") for N in range (N_xx)]).to("cuda")
        
        # print('A_list.shape: ',A_list.shape)
        # print('x.shape: ',x.shape)
        
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

