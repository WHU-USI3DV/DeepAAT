from turtle import forward
from cv2 import norm
import torch
from torch.nn import Linear, ReLU, BatchNorm1d, Sequential, Module, Identity, Sigmoid, Tanh
from utils.sparse_utils import SparseMat
from utils.pos_enc_utils import get_embedder


def get_linear_layers(feats, final_layer=False, batchnorm=True):
    layers = []

    # feats = 256*2 + out_dim
    # Add layers
    for i in range(len(feats) - 2):
        layers.append(Linear(feats[i], feats[i + 1]))

        if batchnorm:
            layers.append(BatchNorm1d(feats[i + 1], track_running_stats=False))

        layers.append(ReLU())

    # Add final layer
    layers.append(Linear(feats[-2], feats[-1]))
    if not final_layer:
        if batchnorm:
            layers.append(BatchNorm1d(feats[-1], track_running_stats=False))

        layers.append(ReLU())

    return Sequential(*layers)


class Parameter3DPts(torch.nn.Module):
    def __init__(self, n_pts):
        super().__init__()

        # Init points randomly
        pts_3d = torch.normal(mean=0, std=0.1, size=(3, n_pts), requires_grad=True)

        self.pts_3d = torch.nn.Parameter(pts_3d)    

    def forward(self):
        return self.pts_3d


class SetOfSetLayer(Module):
    def __init__(self, d_in, d_out):
        super(SetOfSetLayer, self).__init__()
        # n is the number of points and m is the number of cameras
        self.lin_all = Linear(d_in, d_out)    # w1
        self.lin_n = Linear(d_in, d_out)      # w2
        self.lin_m = Linear(d_in, d_out)      # w3
        self.lin_both = Linear(d_in, d_out)   # w4

    def forward(self, x):
        # x is [m,n,d] sparse matrix
        out_all = self.lin_all(x.values)  # [nnz,d_in] -> [nnz,d_out]

        mean_rows = x.mean(dim=0) # [m,n,d_in] -> [n,d_in]
        out_rows = self.lin_n(mean_rows)  # [n,d_in] -> [n,d_out]

        mean_cols = x.mean(dim=1) # [m,n,d_in] -> [m,d_in]
        out_cols = self.lin_m(mean_cols)  # [m,d_in] -> [m,d_out]

        out_both = self.lin_both(x.values.mean(dim=0, keepdim=True))  # [1,d_in] -> [1,d_out]

        new_features = (out_all + out_rows[x.indices[1], :] + out_cols[x.indices[0], :] + out_both) / 4  # [nnz,d_out]
        new_shape = (x.shape[0], x.shape[1], new_features.shape[1])

        return SparseMat(new_features, x.indices, x.cam_per_pts, x.pts_per_cam, new_shape)


class ProjLayer(Module):
    def __init__(self, d_in, d_out):
        super(ProjLayer, self).__init__()
        # n is the number of points and m is the number of cameras
        self.lin_all = Linear(d_in, d_out)

    def forward(self, x):
        # x is [m,n,d] sparse matrix
        new_features = self.lin_all(x.values)  # [nnz,d_in] -> [nnz,d_out]
        new_shape = (x.shape[0], x.shape[1], new_features.shape[1])
        return SparseMat(new_features, x.indices, x.cam_per_pts, x.pts_per_cam, new_shape)


class NormalizationLayer(Module):
    def forward(self, x):
        features = x.values
        norm_features = features - features.mean(dim=0, keepdim=True)
        norm_features = norm_features / norm_features.std(dim=0, keepdim=True)        
        return SparseMat(norm_features, x.indices, x.cam_per_pts, x.pts_per_cam, x.shape)


class CenterNorm(Module):
    def forward(self, x):
        features = x.values
        norm_features = features - features.mean(dim=0, keepdim=True)
        return SparseMat(norm_features, x.indices, x.cam_per_pts, x.pts_per_cam, x.shape)
    

class LastNorm(Module):
    def forward(self, x):
        features = x.values
        norm_features = features - features.mean(dim=0, keepdim=True)
        maxnum = torch.max(norm_features.max(), -norm_features.min())
        norm_features = norm_features*5/maxnum
        return SparseMat(norm_features, x.indices, x.cam_per_pts, x.pts_per_cam, x.shape)


class RCNormLayer(Module):
    def forward(self, x):    # x.shape = m,n,d
        # dim0
        mean0 = x.mean(dim=0)                                          # n,d
        meandif0 = x.values - mean0[x.indices[1], :]                   # nnz,d
        # dim1
        mean1 = x.mean(dim=1)                                          # m,d
        meandif1 = x.values - mean1[x.indices[0], :]                   # nnz,d
        # output
        new_shape = (x.shape[0], x.shape[1], x.shape[2]*2)             # m,n,2d
        new_vals = torch.cat([meandif0, meandif1], -1)                 # m,n,2d
        return SparseMat(new_vals, x.indices, x.cam_per_pts, x.pts_per_cam, new_shape)



class InstanceNormLayer(Module):    
    # def forward(self, x):    # x.shape = m,n,d
    #     n_features = x.shape[2]                                        # d
    #     # dim0
    #     mean0 = x.mean(dim=0)                                          # n,d
    #     meandif0 = x.values - mean0[x.indices[1], :]                   # nnz,d
    #     sum0 = torch.zeros(x.shape[1], n_features, device=x.device)    # n,d
    #     sum0.index_add(0, x.indices[1], meandif0.pow(2.0))
    #     norm0 = (sum0 / x.cam_per_pts).pow(0.5)                        # n,d
    #     normx0 = meandif0 / norm0[x.indices[1], :]                     # nnz,d  
    #     # dim1
    #     mean1 = x.mean(dim=1)                                          # m,d
    #     meandif1 = x.values - mean1[x.indices[0], :]                   # nnz,d
    #     sum1 = torch.zeros(x.shape[0], n_features, device=x.device)    # m,d
    #     sum1.index_add(0, x.indices[0], meandif1.pow(2.0))
    #     norm1 = (sum1 / x.pts_per_cam).pow(0.5)                        # m,d
    #     normx1 = meandif1 / norm1[x.indices[0], :]                     # nnz,d
    #     # output
    #     new_shape = (x.shape[0], x.shape[1], x.shape[2]*2)             # m,n,2d
    #     new_vals = torch.cat([normx0, normx1], -1)                     # m,n,2d
    #     return SparseMat(new_vals, x.indices, x.cam_per_pts, x.pts_per_cam, new_shape)
    
    def forward(self, x):    # x.shape = m,n,d
        # dim0
        mean0 = x.mean(dim=0)                                                    # n,d
        std0 = x.std(dim=0)                                                      # n,d
        norm0 = (x.values - mean0[x.indices[1], :]) / std0[x.indices[1], :]      # nnz,d
        # dim1
        mean1 = x.mean(dim=1)                                                    # n,d
        std1 = x.std(dim=1)                                                      # n,d
        norm1 = (x.values - mean1[x.indices[0], :]) / std1[x.indices[0], :]      # nnz,d
        # output
        new_shape = (x.shape[0], x.shape[1], x.shape[2]*2)                       # m,n,2d
        new_vals = torch.cat([norm0, norm1], -1)                                 # m,n,2d
        return SparseMat(new_vals, x.indices, x.cam_per_pts, x.pts_per_cam, new_shape)
        

class ActivationLayer(Module):
    def __init__(self):
        super(ActivationLayer, self).__init__()
        self.relu = ReLU()

    def forward(self, x):
        new_features = self.relu(x.values)
        return SparseMat(new_features, x.indices, x.cam_per_pts, x.pts_per_cam, x.shape)


class IdentityLayer(Module):
    def forward(self, x):
        return x


class EmbeddingLayer(Module):
    def __init__(self, multires, in_dim):
        super(EmbeddingLayer, self).__init__()
        if multires > 0:
            self.embed, self.d_out = get_embedder(multires, in_dim)
        else:
            self.embed, self.d_out = (Identity(), in_dim)

    def forward(self, x):
        embeded_features = self.embed(x.values)
        new_shape = (x.shape[0], x.shape[1], embeded_features.shape[1])
        return SparseMat(embeded_features, x.indices, x.cam_per_pts, x.pts_per_cam, new_shape)


class PosiEmbedding(Module):
    def __init__(self, num_freqs: int, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        """
        super(PosiEmbedding, self).__init__()

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, num_freqs - 1, num_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (num_freqs - 1), num_freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = [x]
        for freq in self.freq_bands:
            out += [torch.sin(freq * x), torch.cos(freq * x)]
        return torch.cat(out, -1)


class SigmoidScoreLayer(Module):
    def __init__(self):
        super(SigmoidScoreLayer, self).__init__()
        self.sigmoid = Sigmoid()
        
    def forward(self, x):
        new_features = self.sigmoid(x.values)
        return SparseMat(new_features, x.indices, x.cam_per_pts, x.pts_per_cam, x.shape)


class TanhScoreLayer(Module):
    def __init__(self):
        super(TanhScoreLayer, self).__init__()
        self.relu = ReLU()
        self.tanh = Tanh()
        
    def forward(self, x):
        new_features = self.relu(x.values)
        new_features = self.tanh(new_features)
        return SparseMat(new_features, x.indices, x.cam_per_pts, x.pts_per_cam, x.shape)