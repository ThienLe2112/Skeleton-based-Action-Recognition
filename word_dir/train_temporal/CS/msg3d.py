import sys
sys.path.insert(0, '')

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import import_class, count_params
from model.ms_gcn import MultiScale_GraphConv as MS_GCN
from model.ms_tcn import MultiScale_TemporalConv as MS_TCN
from model.ms_gtcn import SpatialTemporal_MS_GCN, UnfoldTemporalWindows
from model.mlp import MLP
from model.activation import activation_factory

#Transformer
from .temporal_transformer_windowed import tcn_unit_attention_block
from .temporal_transformer import tcn_unit_attention

from .gcn_attention import gcn_unit_attention
from .net import Unit2D, conv_init, import_class
from .unit_gcn import unit_gcn
from .unit_agcn import unit_agcn

default_backbone_all_layers = [(3, 64, 1), (64, 64, 1), (64, 64, 1), (64, 64, 1), (64, 128,
                                                                                   2), (128, 128, 1),
                               (128, 128, 1), (128, 256, 2), (256, 256, 1), (256, 256, 1)]

default_backbone = [(64, 64, 1), (64, 64, 1), (64, 64, 1), (64, 128,
                                                            2), (128, 128, 1),
                    (128, 128, 1), (128, 256, 2), (256, 256, 1), (256, 256, 1)]




class MS_G3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_size,
                 window_stride,
                 window_dilation,
                 embed_factor=1,
                 activation='relu'):
        super().__init__()
        self.window_size = window_size
        self.out_channels = out_channels
        self.embed_channels_in = self.embed_channels_out = out_channels // embed_factor
        if embed_factor == 1:
            self.in1x1 = nn.Identity()
            self.embed_channels_in = self.embed_channels_out = in_channels
            # The first STGC block changes channels right away; others change at collapse
            if in_channels == 3:
                self.embed_channels_out = out_channels
        else:
            self.in1x1 = MLP(in_channels, [self.embed_channels_in])

        self.gcn3d = nn.Sequential(
            UnfoldTemporalWindows(window_size, window_stride, window_dilation),
            SpatialTemporal_MS_GCN(
                in_channels=self.embed_channels_in,
                out_channels=self.embed_channels_out,
                A_binary=A_binary,
                num_scales=num_scales,
                window_size=window_size,
                use_Ares=False ##Original, use_Ares=True
            )
        )

        self.out_conv = nn.Conv3d(self.embed_channels_out, out_channels, kernel_size=(1, self.window_size, 1))
        self.out_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        #Edit Code Begin 8
        x,xx=x
        #Edit Code End 8

        N, _, T, V = x.shape
        x = self.in1x1(x)
        # Construct temporal windows and apply MS-GCN
        x = self.gcn3d([x,xx])

        # Collapse the window dimension
        x = x.view(N, self.embed_channels_out, -1, self.window_size, V)
        x = self.out_conv(x).squeeze(dim=3)
        x = self.out_bn(x)

        # no activation
        return x


class MultiWindow_MS_G3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_sizes=[3,5],
                 window_stride=1,
                 window_dilations=[1,1]):

        super().__init__()
        self.gcn3d = nn.ModuleList([
            MS_G3D(
                in_channels,
                out_channels,
                A_binary,
                num_scales,
                window_size,
                window_stride,
                window_dilation
            )
            for window_size, window_dilation in zip(window_sizes, window_dilations)
        ])

    def forward(self, x):
        # Input shape: (N, C, T, V)
        #Edit Code Begin 9
        x,xx=x
        #Edit Code End 9
        
        
        out_sum = 0
        for gcn3d in self.gcn3d:
            out_sum += gcn3d([x,xx])
        # no activation
        return out_sum


class Model(nn.Module):
    def __init__(self,
                 num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 channel,
                 attention,
                 only_attention,
                 tcn_attention,
                 only_temporal_attention,
                 attention_3,
                 relative,
                 kernel_temporal,
                 double_channel,
                 drop_connect,
                 concat_original,
                 dv,
                 dk,
                 Nh,
                 dim_block1,
                 dim_block2,
                 dim_block3,
                 all_layers,
                 data_normalization,
                 visualization,
                 skip_conn,
                 adjacency,
                 bn_flag,
                 weight_matrix,
                 device,
                 n,
                 more_channels,use_data_bn=False,
                 backbone_config=None,
                 mask_learning=False,
                 use_local_bn=False,
                 multiscale=False,
                 temporal_kernel_size=9,
                 dropout=0.5,
                 agcn = True,
                 in_channels=3):
        super(Model, self).__init__()

        Graph = import_class(graph)
        A_binary = Graph().A_binary
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.conv2l = torch.nn.Conv3d(2,2,(4,1,1),padding=(0,0,0),stride=(4,1,1))

        #Transformer Begin 1
        self.num_class = num_class
        self.use_data_bn = use_data_bn
        self.multiscale = multiscale
        self.attention = attention
        self.tcn_attention = tcn_attention
        self.drop_connect = drop_connect
        self.more_channels = more_channels
        self.concat_original = concat_original
        self.all_layers = all_layers
        self.dv = dv
        self.num = n
        self.Nh = Nh
        self.dk = dk
        self.data_normalization = data_normalization
        self.skip_conn = skip_conn
        self.visualization = visualization
        self.double_channel = double_channel
        self.adjacency = adjacency

               # Different bodies share batchNorm parameters or not
        self.M_dim_bn = True

        # if self.M_dim_bn:
        #     # print("M", channel * num_point * num_person)
        #     self.data_bn = nn.BatchNorm1d(channel * num_point * num_person)
        # else:
        #     # print("Not M", channel * num_point * num_person)
            
        #     self.data_bn = nn.BatchNorm1d(channel * num_point )

        if self.all_layers:
            if not self.double_channel:
                self.starting_ch = 64
            else:
                self.starting_ch = 128
        else:
            if not self.double_channel:
                self.starting_ch = 128
            else:
                self.starting_ch = 256

        kwargs = dict(
            A=torch.tensor(A_binary),
            mask_learning=mask_learning,
            use_local_bn=use_local_bn,
            dropout=dropout,
            kernel_size=temporal_kernel_size,
            attention=attention,
            only_attention=only_attention,
            tcn_attention=tcn_attention,
            only_temporal_attention=only_temporal_attention,
            attention_3=attention_3,
            relative=relative,
            weight_matrix=weight_matrix,
            device=device,
            more_channels=self.more_channels,
            drop_connect=self.drop_connect,
            data_normalization=self.data_normalization,
            skip_conn=self.skip_conn,
            adjacency=self.adjacency,
            starting_ch=self.starting_ch,
            visualization=self.visualization,
            all_layers=self.all_layers,
            dv=self.dv,
            dk=self.dk,
            Nh=self.Nh,
            num=n,
            dim_block1=dim_block1,
            dim_block2=dim_block2,
            dim_block3=dim_block3,
            num_point=num_point,
            agcn = agcn
        )
        window_size= 300
        if self.multiscale:

            ## Transformer trong đây nhưng không dùng MS
            ## Nên không quan tâm thằng này
            # unit = TCN_GCN_unit_multiscale
            pass

        else:
            ## Tk này mới cần quan tâm này
            unit = TCN_GCN_unit

        # backbone
        if backbone_config is None:
            if self.all_layers:
                backbone_config = default_backbone_all_layers
            else:
                backbone_config = default_backbone
        self.backbone = nn.ModuleList([
            unit(in_c, out_c, stride=stride, **kwargs)
            for in_c, out_c, stride in backbone_config
        ])
        if self.double_channel:
            backbone_in_c = backbone_config[0][0] * 2
            backbone_out_c = backbone_config[-1][1] * 2
        else:
            backbone_in_c = backbone_config[0][0]
            backbone_out_c = backbone_config[-1][1]
        backbone_out_t = window_size
        backbone = []
        for i, (in_c, out_c, stride) in enumerate(backbone_config):
            if self.double_channel:
                in_c = in_c * 2
                out_c = out_c * 2
            if i == 3 and concat_original:

                ## trong backbone là tk Unit chứa module transformer
                backbone.append(unit(in_c + channel, out_c, stride=stride, last=i == len(default_backbone) - 1,
                                     last_graph=(i == len(default_backbone) - 1), layer=i, **kwargs))
            else:

                ## trong backbone là tk Unit chứa module transformer

                backbone.append(unit(in_c, out_c, stride=stride, last=i == len(default_backbone) - 1,
                                     last_graph=(i == len(default_backbone) - 1), layer=i, **kwargs))
            if backbone_out_t % stride == 0:
                backbone_out_t = backbone_out_t // stride
            else:
                backbone_out_t = backbone_out_t // stride + 1

        ## Transformer từ thằng backbone
        self.backbone = nn.ModuleList(backbone)
        # print("self.backbone: ", self.backbone)
        for i in range(0, len(backbone)):
            pytorch_total_params = sum(p.numel() for p in self.backbone[i].parameters() if p.requires_grad)
            print(pytorch_total_params)

        # head

        if not all_layers:
            if not agcn:
                self.gcn0 = unit_gcn(
                    channel,
                    backbone_in_c,
                    torch.tensor(A_binary),
                    mask_learning=mask_learning,
                    use_local_bn=use_local_bn)
            else:
                self.gcn0 = unit_agcn(
                    channel,
                    backbone_in_c,
                    self.A,
                    mask_learning=mask_learning,
                    use_local_bn=use_local_bn)

            self.tcn0 = Unit2D(backbone_in_c, backbone_in_c, kernel_size=9)

        # tail
        self.person_bn = nn.BatchNorm1d(backbone_out_c)
        self.gap_size = backbone_out_t
        self.fcn = nn.Conv1d(backbone_out_c, num_class, kernel_size=1)
        conv_init(self.fcn)
        #Transformer End 1 

        # channels
        c1 = 96
        c2 = c1 * 2     # 192
        c3 = c2 * 2     # 384
        #transformer
        self.sgcnT = nn.Sequential(
            MS_GCN(num_gcn_scales, 128, 384, A_binary, disentangled_agg=True),
            MS_TCN(384, 384),
            MS_TCN(384, 384))


        # r=3 STGC blocks
        self.gcn3d1 = MultiWindow_MS_G3D(3, c1, A_binary, num_g3d_scales, window_stride=1)
        self.sgcn1 = nn.Sequential(
            MS_GCN(num_gcn_scales, 3, c1, A_binary, disentangled_agg=True),
            MS_TCN(c1, c1),
            MS_TCN(c1, c1))
        self.sgcn1[-1].act = nn.Identity()
        self.tcn1 = MS_TCN(c1, c1)

        self.gcn3d2 = MultiWindow_MS_G3D(c1, c2, A_binary, num_g3d_scales, window_stride=2)
        self.sgcn2 = nn.Sequential(
            MS_GCN(num_gcn_scales, c1, c1, A_binary, disentangled_agg=True),
            MS_TCN(c1, c2, stride=2),
            MS_TCN(c2, c2))
        self.sgcn2[-1].act = nn.Identity()
        self.tcn2 = MS_TCN(c2, c2)

        self.gcn3d3 = MultiWindow_MS_G3D(c2, c3, A_binary, num_g3d_scales, window_stride=2)
        # self.sgcn3 = nn.Sequential(
        #     MS_GCN(num_gcn_scales, c2, c2, A_binary, disentangled_agg=True),
        #     MS_TCN(c2, c3, stride=2),
        #     MS_TCN(c3, c3))
        self.sgcn3 = nn.Sequential(
            MS_GCN(num_gcn_scales, c2, c2, A_binary, disentangled_agg=True),
            MS_TCN(c2, c3, stride=2),
            MS_TCN(c3, c3))
        self.sgcn3[-1].act = nn.Identity()
        self.tcn3 = MS_TCN(c3, c3)

        self.fc = nn.Linear(c3, num_class)

    def forward(self, x, label, name):
    # def forward(self, x):
        N, C, T, V, M = x.size()
        ##Edit Code Begin 3
        
        # xx= torch.clone(x)
        # print('x.shape: ',x.shape)
        ##Edit Code End 3

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        # print("x.shape: ", x.shape)
        x = self.data_bn(x)
        x = x.view(N * M, V, C, T).permute(0,2,3,1).contiguous()
        x_coord = torch.clone(x)
        ##Edit Code Begin 13
        
        
        
        xx=torch.clone(x)
        xx = xx.permute(0,2,1,3).contiguous().view(N * M * T, C, V)
        ##Edit Code End 13
        


        # Apply activation to the sum of the pathways
        ##Edit Code Begin 4
        x = F.relu(self.sgcn1(x) + self.gcn3d1([x,xx]), inplace=True)
        x = self.tcn1(x)
        # print("x1.shape: ", x.shape)
        x = F.relu(self.sgcn2(x) + self.gcn3d2([x,xx]), inplace=True)
        x = self.tcn2(x)
        # print("x2.shape: ", x.shape)
        x = F.relu(self.sgcn3(x) + self.gcn3d3([x,xx]), inplace=True)
        x = self.tcn3(x)



        # x = self.conv2l(x)
        # x = self.sgcnT(x)


        x_trans = self.gcn0(x_coord, label, name)
        x_trans = self.tcn0(x_trans)

        for i, m in enumerate(self.backbone):
            if i == 3 and self.concat_original:
                x_trans = m(torch.cat((x_trans, x_coord), dim=1), label, name)
            else:
                x_trans = m(x_trans, label, name)

        x_trans = self.conv2l(x_trans)
        x_trans = self.sgcnT(x_trans)
        x = F.relu(x + x_trans.detach(), inplace=True)
        # x_trans = F.avg_pool2d(x_trans, kernel_size=(1, V))
        # print("x_trans.shape: ", x_trans.shape)
        # print("STGC x.shape: ", x.shape)
        # print(x)
        # x_trans = F.relu(x_trans)
        # x = x + x_trans
        # print(x_trans)
        # x = x + x_trans
        # x = F.relu(x + x_trans, inplace=True)
        ##Edit Code End 4

        ##Original Code Begin
        # x = F.relu(self.sgcn1(x) + self.gcn3d1(x), inplace=True)
        # x = self.tcn1(x)
        #
        # x = F.relu(self.sgcn2(x) + self.gcn3d2(x), inplace=True)
        # x = self.tcn2(x)
        #
        # x = F.relu(self.sgcn3(x) + self.gcn3d3(x), inplace=True)
        # x = self.tcn3(x)
        
        ##Original Code End


        out = x
        out_channels = out.size(1)
        out = out.view(N, M, out_channels, -1)
        out = out.mean(3)   # Global Average Pooling (Spatial+Temporal)
        out = out.mean(1)   # Average pool number of bodies in the sequence
        out = self.fc(out)
        

        
        return out

class TCN_GCN_unit(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 A,
                 attention,
                 only_attention,
                 tcn_attention,
                 only_temporal_attention,
                 relative,
                 device,
                 attention_3,
                 dv,
                 dk,
                 Nh,
                 num,
                 dim_block1,
                 dim_block2,
                 dim_block3,
                 num_point,
                 weight_matrix,
                 more_channels,
                 drop_connect,
                 starting_ch,
                 all_layers,
                 adjacency,
                 data_normalization,
                 visualization,
                 skip_conn,
                 layer=0,
                 kernel_size=9,
                 stride=1,
                 dropout=0.5,
                 use_local_bn=False,
                 mask_learning=False,
                 last=False,
                 last_graph=False,
                 agcn = False
                 ):
        super(TCN_GCN_unit, self).__init__()
        half_out_channel = out_channel / 2
        self.A = A

        self.V = A.shape[-1]
        self.C = in_channel
        self.last = last
        self.data_normalization = data_normalization
        self.skip_conn = skip_conn
        self.num_point = num_point
        self.adjacency = adjacency
        self.last_graph = last_graph
        self.layer = layer
        self.stride = stride
        self.drop_connect = drop_connect
        self.visualization = visualization
        self.device = device
        self.all_layers = all_layers
        self.more_channels = more_channels

        ## set attention = 0 để không dùng spatial transformer
        if (out_channel >= starting_ch and attention or (self.all_layers and attention)):

            self.gcn1 = gcn_unit_attention(in_channel, out_channel, dv_factor=dv, dk_factor=dk, Nh=Nh,
                                           complete=True,
                                           relative=relative, only_attention=only_attention, layer=layer, incidence=A,
                                           bn_flag=True, last_graph=self.last_graph, more_channels=self.more_channels,
                                           drop_connect=self.drop_connect, adjacency=self.adjacency, num=num,
                                           data_normalization=self.data_normalization, skip_conn=self.skip_conn,
                                           visualization=self.visualization, num_point=self.num_point)
        else:
            # print(A)
            if not agcn:
                self.gcn1 = unit_gcn(
                    in_channel,
                    out_channel,
                    A,
                    use_local_bn=use_local_bn,
                    mask_learning=mask_learning)
            else:
                self.gcn1 = unit_agcn(
                    in_channel,
                    out_channel,
                    A,
                    use_local_bn=use_local_bn,
                    mask_learning=mask_learning)

        if (out_channel >= starting_ch and tcn_attention or (self.all_layers and tcn_attention)):

            if out_channel <= starting_ch and self.all_layers:
                self.tcn1 = tcn_unit_attention_block(out_channel, out_channel, dv_factor=dv,
                                                     dk_factor=dk, Nh=Nh,
                                                     relative=relative, only_temporal_attention=only_temporal_attention,
                                                     dropout=dropout,
                                                     kernel_size_temporal=9, stride=stride,
                                                     weight_matrix=weight_matrix, bn_flag=True, last=self.last,
                                                     layer=layer,
                                                     device=self.device, more_channels=self.more_channels,
                                                     drop_connect=self.drop_connect, n=num,
                                                     data_normalization=self.data_normalization,
                                                     skip_conn=self.skip_conn,
                                                     visualization=self.visualization, dim_block1=dim_block1,
                                                     dim_block2=dim_block2, dim_block3=dim_block3, num_point=self.num_point)
            else:
                self.tcn1 = tcn_unit_attention(out_channel, out_channel, dv_factor=dv,
                                               dk_factor=dk, Nh=Nh,
                                               relative=relative, only_temporal_attention=only_temporal_attention,
                                               dropout=dropout,
                                               kernel_size_temporal=9, stride=stride,
                                               weight_matrix=weight_matrix, bn_flag=True, last=self.last,
                                               layer=layer,
                                               device=self.device, more_channels=self.more_channels,
                                               drop_connect=self.drop_connect, n=num,
                                               data_normalization=self.data_normalization, skip_conn=self.skip_conn,
                                               visualization=self.visualization, num_point=self.num_point)



        else:
            self.tcn1 = Unit2D(
                out_channel,
                out_channel,
                kernel_size=kernel_size,
                dropout=dropout,
                stride=stride)
        if ((in_channel != out_channel) or (stride != 1)):
            self.down1 = Unit2D(
                in_channel, out_channel, kernel_size=1, stride=stride)
        else:
            self.down1 = None

    def forward(self, x, label, name):
        # N, C, T, V = x.size()
        x = self.tcn1(self.gcn1(x, label, name)) + (x if
                                                    (self.down1 is None) else self.down1(x))

        return x


class TCN_GCN_unit_multiscale(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 kernel_size=9,
                 stride=1,
                 **kwargs):
        super(TCN_GCN_unit_multiscale, self).__init__()
        self.unit_1 = TCN_GCN_unit(
            in_channels,
            out_channels / 2,
            A,
            kernel_size=kernel_size,
            stride=stride,
            **kwargs)
        self.unit_2 = TCN_GCN_unit(
            in_channels,
            out_channels - out_channels / 2,
            A,
            kernel_size=kernel_size * 2 - 1,
            stride=stride,
            **kwargs)

    def forward(self, x):
        return torch.cat((self.unit_1(x), self.unit_2(x)), dim=1)


if __name__ == "__main__":
    # For debugging purposes
    import sys
    sys.path.append('..')

    model = Model(
        num_class=60,
        num_point=25,
        num_person=2,
        num_gcn_scales=13,
        num_g3d_scales=6,
        graph='graph.ntu_rgb_d.AdjMatrixGraph'
    )

    N, C, T, V, M = 6, 3, 50, 25, 2
    x = torch.randn(N,C,T,V,M)
    model.forward(x)

    print('Model total # params:', count_params(model))
