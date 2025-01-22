import cv2  # Do not remove
import torch

import sys
sys.path.append('/path/to/deepaat/code')
import utils.path_utils
from utils import geo_utils, general_utils, dataset_utils
import scipy.io as sio
import numpy as np
import os.path
from scipy.sparse import coo_matrix


def get_raw_data(conf, scan, flag):
    """
    :param conf:
    :return:
    M - Points Matrix (2mxn)
    Ns - Inversed Calibration matrix (Ks-1) (mx3x3)
    Ps_gt - GT projection matrices (mx3x4)
    NBs - Normzlize Bifocal Tensor (En) (3mx3m)
    mask - Inlier points mask (2mxn)
    triplets
    """

    # Init
    # dataset_path_format = os.path.join(utils.path_utils.path_to_datasets(), 'Euclidean', '{}.npz')
    # dataset_path_format = os.path.join(conf.get_string('dataset.dataset_path'), '{}.npz')
    if flag==0:
        dataset_path_format = os.path.join(conf.get_string('dataset.trainset_path'), '{}.npz')
    elif flag==1:
        dataset_path_format = os.path.join(conf.get_string('dataset.valset_path'), '{}.npz')
    else:
        dataset_path_format = os.path.join(conf.get_string('dataset.testset_path'), '{}.npz')

    # Get conf parameters
    if scan is None:
        scan = conf.get_string('dataset.scan')

    # Get raw data
    dataset = np.load(dataset_path_format.format(scan))

    # Get bifocal tensors and 2D points
    # M = dataset['M']
    # mask = dataset['mask']
    Rs = dataset['Rs']
    ts = dataset['ts']
    # ts = dataset['enu']
    quats = dataset['quats']
    Ns = dataset['Ns']
    M_col = dataset['M_col']
    M_row = dataset['M_row']       
    M_data = dataset['M_data']
    M_shape = dataset['M_shape']
    M = coo_matrix((M_data, (M_row, M_col)), shape=M_shape).todense().A
    mask_col = dataset['mask_col']
    mask_row = dataset['mask_row']
    mask_data = dataset['mask_data']
    mask_shape = dataset['mask_shape']
    mask = coo_matrix((mask_data, (mask_row, mask_col)), shape=mask_shape).todense().A
    # M_gt = dataset_utils.correct_matches_global(M, Ps_gt, Ns)
    # mask = dataset_utils.get_mask_by_reproj(M, M_gt, 2)
    gpss = dataset['enu_noisy']
    # gpss = dataset['enu']

    scale = 1.0
    if conf.get_bool('train.train_trans'):
        t0 = ts[0]
        nts = ts-t0
        scale = np.max(np.linalg.norm(nts, axis=1))
        ts = nts/scale
        gpss = (gpss-t0)/scale

    # use_gt = conf.get_bool('dataset.use_gt')
    # if use_gt:
    #     M = torch.from_numpy(dataset_utils.correct_matches_global(M, Ps_gt, Ns)).float()

    use_spatial_encoder = conf.get_bool('dataset.use_spatial_encoder')
    
    if flag==0:    # shuffle row and col of train set
        indices = torch.randperm(Ns.shape[0])
        M_indices = torch.zeros(len(indices)*2, dtype=torch.int64)
        M_indices[::2] = 2 * indices
        M_indices[1::2] = 2 * indices + 1
        M = torch.from_numpy(M).float()[M_indices]
        Rs = torch.from_numpy(Rs).float()[indices]
        ts = torch.from_numpy(ts).float()[indices]
        quats = torch.from_numpy(quats).float()[indices]
        Ns = torch.from_numpy(Ns).float()[indices]
        mask = torch.from_numpy(mask).float()[M_indices]
        gpss = torch.from_numpy(gpss).float()[indices]
        # shuffle column
        idx = torch.randperm(M.shape[1])
        M = M[:,idx]
        mask = mask[:,idx]
        color = None

        if use_spatial_encoder:
            dsc_idx_np = dataset['D_idxs']
            dsc_data_np = dataset['D_data']
            dsc_shape_np = dataset['D_shape']
            ord_data = np.arange(dsc_idx_np.shape[1]) + 1                   
            ord_shape = (dsc_shape_np[0], dsc_shape_np[1])
            ord_dense = coo_matrix((ord_data, (dsc_idx_np[0], dsc_idx_np[1])), shape=ord_shape).todense().A
            ord_dense = ord_dense[indices,:]
            ord_dense = ord_dense[:,idx]
            ord_sparse = coo_matrix(ord_dense)
            ord_new = ord_sparse.data - 1                                  
            dsc_idx = torch.from_numpy(np.vstack((ord_sparse.row, ord_sparse.col))).int()
            dsc_data = torch.from_numpy(dsc_data_np[ord_new, :]/128.0).float()    # nnz,128
            dsc_shape = torch.from_numpy(dsc_shape_np).int()
        else:
            dsc_idx = None
            dsc_data = None
            dsc_shape = None

    else:
        M = torch.from_numpy(M).float()
        Rs = torch.from_numpy(Rs).float()
        ts = torch.from_numpy(ts).float()
        quats = torch.from_numpy(quats).float()
        Ns = torch.from_numpy(Ns).float()
        mask = torch.from_numpy(mask).float()
        gpss = torch.from_numpy(gpss).float()
        color = torch.from_numpy(dataset['rgbs']).int()

        if use_spatial_encoder:
            dsc_idx = torch.from_numpy(dataset['D_idxs']).int()
            dsc_data = torch.from_numpy(dataset['D_data']/128.0).float()
            dsc_shape = torch.from_numpy(dataset['D_shape']).int()
        else:
            dsc_idx = None
            dsc_data = None
            dsc_shape = None

    # Add Noise
    if conf.get_bool("dataset.addNoise"):
        noise_mean = conf.get_float("dataset.noise_mean")
        noise_std = conf.get_float("dataset.noise_std")
        noise_radio = conf.get_float("dataset.noise_radio")
        M = geo_utils.addNoise(M, noise_mean, noise_std, noise_radio)    
        # dsc_data = geo_utils.addNoise(dsc_data, noise_mean, noise_std, noise_radio)

    return M, Ns, Rs, ts, quats, mask, gpss, color, scale, use_spatial_encoder, dsc_idx, dsc_data, dsc_shape
  

def test_Ps_M(Ps, M, Ns):
    global_rep_err = geo_utils.calc_global_reprojection_error(Ps.numpy(), M.numpy(), Ns.numpy())
    print("Reprojection Error: Mean = {}, Max = {}".format(np.nanmean(global_rep_err), np.nanmax(global_rep_err)))


def test_euclidean_dataset(scan):
    # dataset_path_format = os.path.join(utils.path_utils.path_to_datasets(), 'Euclidean', '{}.npz')

    # # Get raw data
    # dataset = np.load(dataset_path_format.format(scan))
    dataset = np.load("/path/to/data.npz")

    # Get bifocal tensors and 2D points
    M = dataset['M']
    # M_col = dataset['M_col']
    # M_row = dataset['M_row']
    # M_data = dataset['M_data']
    # M_shape = dataset['M_shape']
    # M = coo_matrix((M_data, (M_row, M_col)), shape=M_shape).todense().A
    # mask_col = dataset['mask_col']
    # mask_row = dataset['mask_row']
    # mask_data = dataset['mask_data']
    # mask_shape = dataset['mask_shape']
    # mask = coo_matrix((mask_data, (mask_row, mask_col)), shape=mask_shape).todense().A
    Ps_gt = dataset['Ps_gt']
    Ns = dataset['Ns']

    print(M.shape)
    M_gt = torch.from_numpy(dataset_utils.correct_matches_global(M, Ps_gt, Ns)).float()
    print(M_gt.shape)

    M = torch.from_numpy(M).float()
    Ps_gt = torch.from_numpy(Ps_gt).float()
    Ns = torch.from_numpy(Ns).float()

    print("Test Ps and M")
    test_Ps_M(Ps_gt, M, Ns)

    print("Test Ps and M_gt")
    test_Ps_M(Ps_gt, M_gt, Ns)


if __name__ == "__main__":
    scan = "Alcatraz Courtyard"
    test_euclidean_dataset(scan)