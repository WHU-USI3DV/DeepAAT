import numpy
import torch
from utils import geo_utils
from torch import dtype, nn
from torch.nn import functional as F
from pytorch3d import transforms as py3d_trans


class ESFMLoss(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.infinity_pts_margin = conf.get_float("loss.infinity_pts_margin")    
        self.normalize_grad = conf.get_bool("loss.normalize_grad")

        self.hinge_loss = conf.get_bool("loss.hinge_loss")
        if self.hinge_loss:
            self.hinge_loss_weight = conf.get_float("loss.hinge_loss_weight")
        else:
            self.hinge_loss_weight = 0

    def forward(self, pred_cam, data, epoch=None):
        Ps = pred_cam["Ps_norm"]
        pts_2d = Ps @ pred_cam["pts3D"]  # [m, 3, n]

        if self.normalize_grad:
            pts_2d.register_hook(lambda grad: F.normalize(grad, dim=1) / data.valid_pts.sum())

            projected_points = geo_utils.get_positive_projected_pts_mask(pts_2d, self.infinity_pts_margin)
        else:
            projected_points = geo_utils.get_projected_pts_mask(pts_2d, self.infinity_pts_margin)

        # Calculate hinge Loss
        hinge_loss = (self.infinity_pts_margin - pts_2d[:, 2, :]) * self.hinge_loss_weight

        # Calculate reprojection error
        pts_2d = (pts_2d / torch.where(projected_points, pts_2d[:, 2, :], torch.ones_like(projected_points).float()).unsqueeze(dim=1))
        reproj_err = (pts_2d[:, 0:2, :] - data.norm_M.reshape(Ps.shape[0], 2, -1)).norm(dim=1)

        return torch.where(projected_points, reproj_err, hinge_loss)[data.valid_pts].mean()


class GTLoss(nn.Module):
    def __init__(self, conf):
        super().__init__()
        # self.calibrated = conf.get_bool('dataset.calibrated')
        self.train_trans = conf.get_bool('train.train_trans')
        self.alpha =  conf.get_float("dataset.alpha")
        self.beta =  conf.get_float("dataset.beta")

    def forward(self, pred_cam, data, epoch=None):

        if self.train_trans:
            ts_gt = data.ts - data.gpss
            orient_err = (data.quats - pred_cam["quats"]).norm(2, dim=1)    
            translation_err = (ts_gt - pred_cam["ts"]).norm(2, dim=1)   
            orient_loss = orient_err.mean() * self.alpha
            trans_loss = translation_err.mean() * self.beta
            loss = orient_loss + trans_loss

            if epoch is not None and epoch % 1000 == 0:
                # Print loss
                print("GTloss = {}, orient err = {}, trans err = {}".format(loss, orient_loss, trans_loss))
            return loss, orient_loss, trans_loss
        
        else:
            orient_err = (data.quats - pred_cam["quats"]).norm(2, dim=1)    
            orient_loss = orient_err.mean()

            if epoch is not None and epoch % 1000 == 0:
                # Print loss
                print("orient err = {}".format(orient_loss))
            return orient_loss
        

class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred_mask, data, epoch=None):
        # pred_mask_ = pred_mask.reshape(-1, 1)                     # m,n,1 -> m*n,1
        # mask_ = data.mask                                         # 2m,n
        # mask_ = mask_[0:mask_.shape[0]:2,:].reshape(-1, 1)        # 2m,n -> m*n,1
        pred_mask_ = torch.as_tensor(pred_mask.values).reshape(len(pred_mask.values), 1)
        gt_mask_ = torch.as_tensor(data.mask_sparse.values).reshape(len(data.mask_sparse.values), 1)
        
        loss = self.bceloss(pred_mask_, gt_mask_).mean()
        if epoch is not None and epoch % 1000 == 0:
            # Print loss
            print("BCEloss = {}".format(loss))
        return loss

class BCEWithLogitLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.lbceloss = nn.BCEWithLogitsLoss()

    def forward(self, pred_mask, data, epoch=None):
        
        pred_mask_ = torch.as_tensor(pred_mask.values).reshape(len(pred_mask.values), 1)
        gt_mask_ = torch.as_tensor(data.mask_sparse.values).reshape(len(data.mask_sparse.values), 1)
        
        loss = self.lbceloss(pred_mask_, gt_mask_).mean()
        if epoch is not None and epoch % 1000 == 0:
            # Print loss
            print("BCEloss = {}".format(loss))
        return loss