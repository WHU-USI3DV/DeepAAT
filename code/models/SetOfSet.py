import torch
from torch import nn
from models.baseNet import BaseNet
from models.layers import *
from utils import geo_utils


class SetOfSetBlock(nn.Module):
    def __init__(self, d_in, d_out, conf):
        super(SetOfSetBlock, self).__init__()
        self.block_size = conf.get_int("model.block_size")
        self.use_skip = conf.get_bool("model.use_skip")

        modules = []
        # modules.extend([SetOfSetLayer(d_in, 16), 
        #                 NormalizationLayer(),
        #                 ActivationLayer(),
        #                 SetOfSetLayer(16, 64), 
        #                 NormalizationLayer(),
        #                 ActivationLayer(),
        #                 SetOfSetLayer(64, 256), 
        #                 NormalizationLayer(),
        #                 ])
        modules.extend([SetOfSetLayer(d_in, d_out), NormalizationLayer()])    
        for i in range(1, self.block_size):
            modules.extend([ActivationLayer(), SetOfSetLayer(d_out, d_out), NormalizationLayer()])
        self.layers = nn.Sequential(*modules)    

        self.final_act = ActivationLayer()    

        if self.use_skip:    
            if d_in == d_out:
                self.skip = IdentityLayer()
            else:
                self.skip = nn.Sequential(ProjLayer(d_in, d_out), NormalizationLayer())

    def forward(self, x):
        # x is [m,n,d] sparse matrix
        xl = self.layers(x)
        if self.use_skip:
            xl = self.skip(x) + xl

        out = self.final_act(xl)
        return out
    
    # def __init__(self, d_in, d_out, conf):
    #     super(SetOfSetBlock, self).__init__()
    #     self.block_size = conf.get_int("model.block_size")
        
    #     self.firstlayer = SetOfSetLayer(d_in, d_out)
    #     self.submodules1 = nn.Sequential(NormalizationLayer(), ActivationLayer(), SetOfSetLayer(d_out, d_out))
    #     self.submodules2 = nn.Sequential(NormalizationLayer(), ActivationLayer(), SetOfSetLayer(d_out, d_out))
    #     self.lastlayers = nn.Sequential(NormalizationLayer(), ActivationLayer())

    # def forward(self, x):
    #     # x is [m,n,d] sparse matrix
    #     x = self.firstlayer(x)
    #     x = self.submodules1(x) + x
    #     x = self.submodules2(x) + x
    #     out = self.lastlayers(x)
    #     return out


class SetOfSetNet(BaseNet):
    def __init__(self, conf):
        super(SetOfSetNet, self).__init__(conf)
        # n is the number of points and m is the number of cameras
        num_blocks = conf.get_int('model.num_blocks')
        num_feats = conf.get_int('model.num_features')
        self.train_trans = conf.get_bool('train.train_trans')
        self.use_spatial_encoder = conf.get_bool('dataset.use_spatial_encoder')
        self.x_embed_rank = conf.get_int('dataset.x_embed_rank')
        self.egps_embed_rank = conf.get_int('dataset.egps_embed_rank')
        self.dsc_egps_embed_width = conf.get_int('dataset.dsc_egps_embed_width')
        self.gps_embed_width = conf.get_int('dataset.gps_embed_width')

        m_d_out_rot = 4
        m_d_out_trans = 3
        d_in = 2

        self.embed_x = EmbeddingLayer(self.x_embed_rank, d_in)
        self.embed_egps = EmbeddingLayer(self.egps_embed_rank, 3)
        self.embed_dsc_egps = ProjLayer(128+self.embed_egps.d_out, self.dsc_egps_embed_width)

        if self.use_spatial_encoder:    
            self.equivariant_blocks = torch.nn.ModuleList([SetOfSetBlock(self.embed_x.d_out + self.dsc_egps_embed_width, num_feats, conf)])
        else:
            self.equivariant_blocks = torch.nn.ModuleList([SetOfSetBlock(2, num_feats, conf)])
        
        for i in range(num_blocks - 1):
            self.equivariant_blocks.append(SetOfSetBlock(num_feats, num_feats, conf))
        
        self.pt_net = nn.Sequential(ProjLayer(num_feats, num_feats//2), 
                                    RCNormLayer(), 
                                    ActivationLayer(),
                                    ProjLayer(num_feats, num_feats//2), 
                                    RCNormLayer(), 
                                    ActivationLayer(),
                                    ProjLayer(num_feats, 1))

        # self.embed_gps = PosiEmbedding(self.egps_embed_rank)    #nn.Linear(3, 128)
        self.embed_gps = nn.Linear(3, self.gps_embed_width)

        # self.m_net = get_linear_layers([num_feats] * 2 + [m_d_out], final_layer=True, batchnorm=False)    
        # self.n_net = get_linear_layers([num_feats] * 2 + [n_d_out], final_layer=True, batchnorm=False)    
        self.m_net_rot = get_linear_layers([num_feats] * 2 + [m_d_out_rot], final_layer=True, batchnorm=False)
        self.m_net_tran = get_linear_layers([num_feats+self.gps_embed_width, 256, m_d_out_trans], final_layer=True, batchnorm=False)
        # self.m_net_tran = get_linear_layers([num_feats+6*self.egps_embed_rank+3, 256, m_d_out_trans], final_layer=True, batchnorm=False)

    def forward(self, data):
        x = data.x  # x is [m,n,d] sparse matrix
        # x = self.embed_x(x)                                    
        if self.use_spatial_encoder:
            x = self.embed_x(x)
            egps = self.embed_egps(data.egps_sparse)
            dsc_egps = data.dsc.last_dim_cat(egps)
            embed_de = self.embed_dsc_egps(dsc_egps)
            x = x.last_dim_cat(embed_de)
            # x = x.last_dim_cat(self.embed_dsc_egps(data.dsc.last_dim_cat(self.embed_egps(data.egps_sparse))))

        for eq_block in self.equivariant_blocks:             
            x = eq_block(x)  # [m,n,d_in] -> [m,n,d_out]
        
        # pt class predictions
        pt_out = self.pt_net(x)    # [m,n,d_out] -> [m,n,1]

        # Cameras predictions
        # x = x*pt_out             # [m,n,d_out] -> [m,n,d_out]
        m_input = x.mean(dim=1)    # [m,d_out]
        # m_out = self.m_net(m_input)  # [m, d_m]
        # pred_cam = self.extract_camera_outputs(m_out)
        rot = self.m_net_rot(m_input)
        # normed_gps = geo_utils.scale_ts_norm(data.gpss)
        if self.train_trans:
            tran = self.m_net_tran(torch.cat([m_input, self.embed_gps(data.gpss)], -1))
            # tran = self.m_net_tran(m_input)
            pred_cam = {"quats": rot, "ts": tran}
        else:
            pred_cam = {"quats": rot}
        # tran = self.m_net_tran(m_input)
        # pred_cam = {"quats": rot, "ts": data.gpss}

        return pt_out, pred_cam