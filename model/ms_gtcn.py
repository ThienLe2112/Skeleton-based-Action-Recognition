import sys
sys.path.insert(0, '')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.ms_tcn import MultiScale_TemporalConv as MS_TCN
from model.mlp import MLP
from model.activation import activation_factory
from graph.tools import k_adjacency, normalize_adjacency_matrix, normalize_dsg_A_tilta, normalize_adjacency_matrix_cuda, normalize_dsg_adjacency_matrix

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
        # num_scales = num_scales - 1
        self.conv = torch.nn.Conv3d(1,1,(300,1,1),padding=(0,0,0),stride=(300,1,1))
        self.conv2l150 = torch.nn.Conv3d(1,1,(2,1,1),padding=(0,0,0),stride=(2,1,1))
        self.conv2l75 = torch.nn.Conv3d(1,1,(4,1,1),padding=(0,0,0),stride=(4,1,1))
        #Edit Code End 15
        if disentangled_agg:
            #Edit Code Begin 18
            if num_scales == 1:
                
                A_scales = torch.stack([torch.from_numpy(k_adjacency(A, k, with_self=True)) for k in range(num_scales)]).to("cuda")
                # A_scales = [k_adjacency(A, k, with_self=True) for k in range(num_scales)]
            else:
                A_scales = torch.stack([k_adjacency(A, k, with_self=True) for k in range(num_scales - 1)]).to("cuda")
            A_scales = torch.cat([normalize_adjacency_matrix_cuda(g) for g in A_scales]).to("cuda")
            # A_scales = np.concatenate([normalize_adjacency_matrix(g) for g in A_scales])
            #Edit Code End 18
            
            #Original Code Begin
            # A_scales = [k_adjacency(A, k, with_self=True) for k in range(num_scales)]
            # A_scales = np.concatenate([g for g in A_scales])
            #Original Code End
        else:
            # Self-loops have already been included in A
            # print(A)
            A_scales = [normalize_adjacency_matrix(A) for k in range(num_scales)]
            A_scales = [np.linalg.matrix_power(g, k) for k, g in enumerate(A_scales)]
            A_scales = np.concatenate(A_scales)

        self.A_scales = torch.Tensor(A_scales).to("cuda")
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
        xx = xx.permute(0,2,3,1).contiguous()
        
        N, T, V, C = x.shape
        
        xx_reshaped = xx.cuda(non_blocking = True)
        
        dists = torch.norm(xx_reshaped[:, :, :,None, :] - xx_reshaped[:, :, None, :, :], dim=-1).to("cuda")#Shape: N, T, V, V
        lamda=1
        A_dsg = torch.exp(-(dists**2)/lamda).to("cuda") #Shape: N, T, V, V
        
        # A_norm = normalize_dsg_A_tilta(A_dsg)
        A_norm = normalize_dsg_adjacency_matrix(A_dsg)
        
        A_large = torch.stack([torch.stack([t.repeat(self.window_size, self.window_size).to("cuda").clone()  
                                            for t in i]).to("cuda") 
                                                for i in A_norm]).to("cuda")
        if self.num_scales == 1:
            A_scales = torch.cat([A_large for k in range(self.num_scales )], dim = 2).to("cuda")
        else:
            A_scales = torch.cat([A_large for k in range(self.num_scales - 1 )], dim = 2).to("cuda")
        
        # print("A_scales.shape: ", A_scales.shape)
        
        # #Multi_scale
        # # if self.num_scales !=1:
        # #     #Method 1: only dsg K-adjacency
        # #     # N_xx, T_xx, V_xx, _= A_large.shape
        # #     # A_Mask = self.A_scales.repeat( N_xx, T_xx, 1, 1)
        # #     # A_list = torch.mul(A_scales ,A_Mask.to("cuda")).to("cuda")
            
        # #     #Method 2: Adsg + dsg K-adjacency
        # #     N_xx, T_xx, V_xx, _= A_large.shape
        # #     A_Mask = self.A_scales.repeat( N_xx, T_xx, 1, 1).to("cuda")
        # #     A_list = torch.mul(A_scales ,A_Mask.to("cuda")).to("cuda")
        # #     A_comb = torch.cat((A_scales[:,:,0:V_xx], A_list), dim = 2).to("cuda")
        # #     A_list = A_comb
        # # else:
            
        # #Non_Scale
        
        A_list = 0
        
        if x.shape[2] == 150:
            NN, TT, VV, _ = A_scales.shape
            A_list = torch.stack([self.conv2l150(A_scales[i][None,:,:,:]) for i in range(NN)])
            NN,_, TT, VV, _ = A_list.shape
            
            A_list = A_list.reshape(NN,TT,VV,VV)
            
        elif x.shape[2] == 75:
            NN, TT, VV, _ = A_scales.shape
            
            A_list_0 = torch.stack([self.conv2l75(A_scales[i][None,:,:,:]) for i in range(NN)])
            # A_list_0 = torch.stack([self.conv2l150(A_scales[i][None,:,:,:]) for i in range(NN)])
            # NN,_, TT, VV, _ = A_list_0.shape
            # A_list_0 = A_list_0.reshape(NN,TT,VV,VV)
            # A_list = torch.stack([self.conv2l75(A_list_0[i][None,:,:,:]) for i in range(NN)])
            
            NN,_, TT, VV, _ = A_list_0.shape
            A_list = A_list_0.reshape(NN,TT,VV,VV)
        else:
            A_list = A_scales
        # print("A_list.shape: ", A_list.shape)
        NN, TT, VV, _ = A_list.shape
            
        N, C, T, V = x.shape    # T = number of windows
        # print("x.shape: ", x.shape)
        res = self.residual(x.clone())
        y = x.permute(0,2,1,3).contiguous().to("cuda")
        agg = torch.einsum('ntvu,ntcu->ntcv', A_list, y)
        agg = agg.permute(0,2,1,3).contiguous()
        
        ## Adaptive DSG
        # xx=xx.cuda()

        # xx_reshaped = xx.permute(0, 2, 1).contiguous()  # Shape: (N_xx, V_xx, C_xx)

        # # Compute the pairwise distances in a vectorized manner
        # dists = torch.norm(xx_reshaped[:, :, None, :] - xx_reshaped[:, None, :, :], dim=-1)
        # A_dsg=dists
        
        # lamda=1
        # A_dsg_all=torch.exp(-(A_dsg**2)/lamda)
        # A_dsg_all=A_dsg_all[None,None,:,:,:]
        # A_dsg_all=self.conv.to('cuda')(A_dsg_all)
        # # print('A_dsg_all.shape: ', A_dsg_all.shape)
        # _, _, N_xx, _, V_xx = A_dsg_all.shape
        
        # # A_dsg_all = A_dsg_all.reshape(N_xx, 1,V_xx, V_xx)
        # # A_norm = normalize_dsg_A_tilta(A_dsg_all)
        
        # A_norm = A_dsg_all.reshape(N_xx, V_xx, V_xx)
        # # A_norm = A_norm.reshape(N_xx, V_xx, V_xx)
        # # normalize_adjacency_matrix_cuda(A_dsg_all[0,0,0])
        # # print(pd.DataFrame(A_dsg_all[0,0,0].cpu().detach().numpy()))
        # A_binary_with_I = A_norm
        
        # A_large = torch.stack([i.repeat(self.window_size, self.window_size).clone() for i in A_binary_with_I])
        # # A_large = torch.stack([i.repeat(self.window_size, self.window_size).clone() for i in A_binary_with_I[0,0]])
        # A = A_large
        # A_scales = [A for k in range(self.num_scales)]

        # # print('A_large.shape: ', A_large.shape)

        # A_scales = torch.stack([A[i] for i in range(len(A)) for k in range(self.num_scales) ]) 
        
        # A_scales = A_scales[:,None,:,:]
        
        # A_scales = torch.stack([torch.matrix_power(g, k) for i in A_scales for k, g in enumerate(i)])
        
        # t_a, v_a, v_a2=A_scales.shape
        # # print("A_scales.shape: ",A_scales.shape )
        # A_scales=A_scales.reshape(1,1,t_a,v_a,v_a2)
        
        # # print('A_scales.shape: ', A_scales.shape)
        # # print('x.shape[0]: ', x.shape[0])
        # res = self.residual(x)
        # agg = torch.stack([torch.einsum('vu,ctu->ctv', A_scales[0,0,i], x[i]) for i in range(x.shape[0])])
        
        # N, C, T, V = x.shape
        ####################################### PLOT MODEL
        # import seaborn as sns
        # import pandas as pd
        # import matplotlib.pyplot as plt
        # for t in range(self.num_scales):
        #     # t = 6 # t= 0, 3, 6, 9 
        #     px = pd.DataFrame(A_adapt[0,0].cpu().numpy())
        #     # px = pd.DataFrame(A_list[0,0,t*3*25:(t*3+1)*25,0:25].cpu().numpy())
        #     # print(A_dsg.shape)
        #     # # glue = sns.load_dataset("glue").pivot(index="Model", columns="Task", values="Score")
        #     sns.heatmap(px, vmin = 0, vmax = 1, cmap = sns.color_palette("Blues", as_cmap=True),annot=True)
        #     plt.title(f'A_dsg({t})')
        #     plt.show()
        # return
        #######################################



        
        
        
        ##Edit Code End 6

        # Perform Graph Convolution
        #Origin Code Begin
        # res = self.residual(x)
        # agg = torch.einsum('vu,nctu->nctv', A, x)
        
        #Origin Code End
        agg = agg.view(N, C, T, self.num_scales, V).to("cuda")
        agg = agg.permute(0,3,1,2,4).contiguous().view(N, self.num_scales*C, T, V).to("cuda")
        out = self.mlp(agg).to("cuda")
        # print("out.shape: ", out.shape)
        out += res
        return self.act(out)

