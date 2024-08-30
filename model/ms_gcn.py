import sys
sys.path.insert(0, '')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from graph.tools import k_adjacency, normalize_adjacency_matrix, normalize_dsg_adjacency_matrix, normalize_dsg_A_tilta, normalize_adjacency_matrix_cuda
from model.mlp import MLP
from model.activation import activation_factory


class MultiScale_GraphConv(nn.Module):
    def __init__(self,
                 num_scales,
                 in_channels,
                 out_channels,
                 A_binary,
                 disentangled_agg=True, 
                 use_mask=False, #Original True
                 dropout=0,
                 activation='relu'):
        super().__init__()
        self.num_scales = num_scales

        if disentangled_agg:
            #Edit Code Begin 17
            if num_scales == 1:
                
                A_powers = torch.stack([torch.from_numpy(k_adjacency(A_binary, k, with_self=True)) for k in range(num_scales)]).to("cuda")
                # A_powers = [k_adjacency(A, k, with_self=True) for k in range(num_scales)]
            else:
                A_powers = torch.stack([torch.from_numpy(k_adjacency(A_binary, k, with_self=True)) for k in range(num_scales-1)]).to("cuda")
            A_powers = torch.cat([normalize_adjacency_matrix_cuda(g) for g in A_powers]).to("cuda")
             #Edit Code End 17
            
            #Original Code Begin
            # A_powers = [k_adjacency(A_binary, k, with_self=True) for k in range(num_scales)]
            # A_powers = np.concatenate([normalize_adjacency_matrix(g) for g in A_powers])
            #Original Code End 
        else:
            A_powers = [A_binary + np.eye(len(A_binary)) for k in range(num_scales)]
            A_powers = [normalize_adjacency_matrix(g) for g in A_powers]
            A_powers = [np.linalg.matrix_power(g, k) for k, g in enumerate(A_powers)]
            A_powers = np.concatenate(A_powers)

        self.A_powers = torch.Tensor(A_powers)
        self.use_mask = use_mask
        if use_mask:
            # NOTE: the inclusion of residual mask appears to slow down training noticeably
            self.A_res = nn.init.uniform_(nn.Parameter(torch.Tensor(self.A_powers.shape)), -1e-6, 1e-6)

        self.mlp = MLP(in_channels * num_scales, [out_channels], dropout=dropout, activation=activation)

    def forward(self, x):
        x, xx = x
        N, C, T, V = x.shape
        
        #Edit Code Begin 16
        # print("xx.shape: ", xx.shape)
        N, C, T, V, M = xx.shape
        xx_reshaped = xx.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        xx_reshaped = xx_reshaped.view(N * M, V, C, T).permute(0,2,3,1).contiguous()
        
        
        xx_reshaped = xx_reshaped.permute(0, 2, 3, 1).contiguous()  # Shape: (N_xx, T_xx, V_xx, C_xx)
        
        xx_reshaped = xx_reshaped.cuda(non_blocking = True)
        
        dists = torch.norm(xx_reshaped[:, :, :,None, :] - xx_reshaped[:, :, None, :, :], dim=-1).to("cuda")#Shape: N, T, V, V
        
        
        lamda=1
        A_dsg = torch.exp(-(dists**2)/lamda).to("cuda") #Shape: N, T, V, V
        # A_dsg = dists
        
        # normalize_dsg_A_tilta
        # A_norm = normalize_dsg_A_tilta(A_dsg)
        A_norm = normalize_dsg_adjacency_matrix(A_dsg)
        
        
        
        # A_large = torch.stack([torch.stack([t.repeat(self.window_size, self.window_size).clone()  for t in i]) for i in A_norm]).to("cuda")
        # A_large = torch.stack([torch.stack([t.repeat(self.window_size, self.window_size).to("cuda").clone()  
        #                                     for t in i]).to("cuda") 
        #                                         for i in A_norm]).to("cuda")
        A_large = A_norm.to("cuda")
        if self.num_scales == 1:
            A_scales = torch.cat([A_large for k in range(self.num_scales )], dim = 2).to("cuda")
        else:
            A_scales = torch.cat([A_large for k in range(self.num_scales - 1 )], dim = 2).to("cuda")
        
        # print("Self.num_scales: ", self.num_scales)
        
        #Multi_scale
        if self.num_scales !=1:
            #Method 1: only dsg K-adjacency
            # N_xx, T_xx, V_xx, _= A_large.shape
            # A_Mask = self.A_scales.repeat( N_xx, T_xx, 1, 1)
            # A_list = torch.mul(A_scales ,A_Mask.to("cuda")).to("cuda")
            
            #Method 2: Adsg + dsg K-adjacency
            N_xx, T_xx, V_xx, _= A_large.shape
            A_Mask = self.A_powers.repeat( N_xx, T_xx, 1, 1).to("cuda")
            A_list = torch.mul(A_scales ,A_Mask.to("cuda")).to("cuda")
            A_comb = torch.cat((A_scales[:,:,0:V_xx], A_list), dim = 2).to("cuda")
            A_list = A_comb
        else:
            
            #Non_Scale
            
            A_list = A_scales
        ####################################### PLOT MODEL
        # import seaborn as sns
        # import pandas as pd
        # import matplotlib.pyplot as plt
        # for t in range(self.num_scales):
        #     # t = 6 # t= 0, 3, 6, 9 
        #     px = pd.DataFrame(A_list[0,0,t*3*25:(t*3+1)*25,0:25].cpu().numpy())
        #     # print(px)
        #     # # glue = sns.load_dataset("glue").pivot(index="Model", columns="Task", values="Score")
        #     sns.heatmap(px, vmin = 0, vmax = 1, cmap = sns.color_palette("Blues", as_cmap=True),annot=True)
        #     plt.title(f'A_dsg({t})')
        #     plt.show()
        # return
        #######################################



        # res = self.residual(x)
        support = torch.einsum('ntvu,ncfu->ncfv', A_list, x)
        
        
        N, C, T, V = x.shape    # T = number of windows
        #Edit Code End 16
        
        
        #Original Code Begin
        # self.A_powers = self.A_powers.to(x.device)
        # A = self.A_powers.to(x.dtype)
        # if self.use_mask:
        #     A = A + self.A_res.to(x.dtype)
        # support = torch.einsum('vu,nctu->nctv', A, x)
        #Original Code End
        
        support = support.view(N, C, T, self.num_scales, V)
        support = support.permute(0,3,1,2,4).contiguous().view(N, self.num_scales*C, T, V)
        out = self.mlp(support)
        return out


if __name__ == "__main__":
    from graph.ntu_rgb_d import AdjMatrixGraph
    graph = AdjMatrixGraph()
    A_binary = graph.A_binary
    msgcn = MultiScale_GraphConv(num_scales=15, in_channels=3, out_channels=64, A_binary=A_binary)
    msgcn.forward(torch.randn(16,3,30,25))
