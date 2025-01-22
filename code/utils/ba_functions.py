import numpy as np
from utils import ceres_utils
from utils import geo_utils


def euc_ba(xs, raw_xs, Rs, ts, Ks, Xs, Ps=None, Ns=None, repeat=True, max_iter=100, ba_times=2, repro_thre=5, refined=False):
    """
    Computes bundle adjustment with ceres solver
    :param xs: 2d points [m,n,2]
    :param Rs: rotations [m,3,3]
    :param ts: translations [m,3]
    :param Ks: inner parameters, calibration matrices [m,3,3]
    :param Xs_our: initial 3d points [n,3] or None if triangulation needed
    :param Ps: cameras [m,3,4]. Ps[i] = Ks[i] @ Rs[i].T @ [I, -ts[i]]
    :param Ns: normalization matrices. If Ks are known, Ns = inv(Ks)
    :param repeat: run ba twice. default: True
    :param triangulation: For initial point run triangulation. default: False
    :param return_repro: compute and return the reprojection errors before and after.
    :return: results. The new camera parameters, 3d points, and if requested the reprojection errors.
    """
    results = {}

    visible_points = xs[:, :, 0] != 0                     # m,3dptnum
    point_indices = np.stack(np.where(visible_points))    # 2,2dptnum
    visible_xs = xs[visible_points]                       # 2dptnum,2

    # if Ps is None:
    #     Ps = geo_utils.batch_get_camera_matrix_from_rtk(Rs, ts, Ks)

    # if triangulation:
    #     if Ns is None:
    #         Ns = np.linalg.inv(Ks)
    #     norm_P, norm_x = geo_utils.normalize_points_cams(Ps, xs, Ns)
    #     Xs = geo_utils.dlt_triangulation(norm_P, norm_x, visible_points)
    # else:
    #     Xs = Xs_our

    if refined:
        new_Rs, new_ts, new_Ps, _, _, _, _, _ = ceres_utils.run_euc_ceres_iter(
            Ps, Xs, xs, Rs, ts, Ks, point_indices, visible_points, max_iter, ba_times, repro_thre)
        # new_Xs = np.concatenate([new_Xs, np.ones([new_Xs.shape[0],1])], axis=1)
        if repeat:
            visible_points = raw_xs[:, :, 0] != 0                     # m,3dptnum
            point_indices = np.stack(np.where(visible_points))        # 2,2dptnum
            norm_P, norm_x = geo_utils.normalize_points_cams(new_Ps, raw_xs, Ns)
            new_Xs = geo_utils.dlt_triangulation(norm_P, norm_x, visible_points)
            proj_first = True    
            # print(f"new_Xs1: {new_Xs.shape}")
            new_Rs, new_ts, new_Ps, new_Xs, new_xs, visible_points, point_indices, colidx = ceres_utils.run_euc_ceres_iter(
                new_Ps, new_Xs, raw_xs, new_Rs, new_ts, Ks, point_indices, visible_points, max_iter, ba_times, repro_thre, proj_first)
    else:
        new_Rs, new_ts, new_Ps, new_Xs = ceres_utils.run_euclidean_python_ceres(
            Xs, visible_xs, Rs, ts, Ks, point_indices)
        if repeat:
            norm_P, norm_x = geo_utils.normalize_points_cams(new_Ps, xs, Ns)
            new_Xs = geo_utils.dlt_triangulation(norm_P, norm_x, visible_points)
            new_Rs, new_ts, new_Ps, new_Xs = ceres_utils.run_euclidean_python_ceres(
                new_Xs, visible_xs, new_Rs, new_ts, Ks, point_indices)
    new_Xs = np.concatenate([new_Xs, np.ones([new_Xs.shape[0],1])], axis=1)
    
    results['Rs'] = new_Rs
    results['ts'] = new_ts
    results['Ps'] = new_Ps
    results['Xs'] = new_Xs
    if refined: 
        results['valid_points'] = visible_points
        results['new_xs'] = new_xs
        results['colidx'] = colidx

    return results


def merged_ba(xs, raw_xs, Rs, ts, Ks, Xs, Ps=None, Ns=None, repeat=True, max_iter=100, ba_times=2, repro_thre=5, refined=False,
              proj_first=False, proj_second=True):
    """
    Computes bundle adjustment with ceres solver
    :param xs: 2d points [m,n,2]
    :param Rs: rotations [m,3,3]
    :param ts: translations [m,3]
    :param Ks: inner parameters, calibration matrices [m,3,3]
    :param Xs_our: initial 3d points [n,3] or None if triangulation needed
    :param Ps: cameras [m,3,4]. Ps[i] = Ks[i] @ Rs[i].T @ [I, -ts[i]]
    :param Ns: normalization matrices. If Ks are known, Ns = inv(Ks)
    :param repeat: run ba twice. default: True
    :param triangulation: For initial point run triangulation. default: False
    :param return_repro: compute and return the reprojection errors before and after.
    :return: results. The new camera parameters, 3d points, and if requested the reprojection errors.
    """
    results = {}

    visible_points = xs[:, :, 0] != 0                     # m,3dptnum
    point_indices = np.stack(np.where(visible_points))    # 2,2dptnum

    # if Ps is None:
    #     Ps = geo_utils.batch_get_camera_matrix_from_rtk(Rs, ts, Ks)

    # if triangulation:
    #     if Ns is None:
    #         Ns = np.linalg.inv(Ks)
    #     norm_P, norm_x = geo_utils.normalize_points_cams(Ps, xs, Ns)
    #     Xs = geo_utils.dlt_triangulation(norm_P, norm_x, visible_points)
    # else:
    #     Xs = Xs_our

    if refined:    
        new_Rs, new_ts, new_Ps, new_Xs, new_xs, visible_points, point_indices, colidx = ceres_utils.run_euc_ceres_iter(
            Ps, Xs, xs, Rs, ts, Ks, point_indices, visible_points, max_iter, ba_times, repro_thre, proj_first)
        # new_Xs = np.concatenate([new_Xs, np.ones([new_Xs.shape[0],1])], axis=1)
        if repeat:
            visible_points = raw_xs[:, :, 0] != 0                     # m,3dptnum
            point_indices = np.stack(np.where(visible_points))        # 2,2dptnum
            # new_Ps = geo_utils.batch_get_camera_matrix_from_rtk(new_Rs, new_ts, Ks)
            norm_P, norm_x = geo_utils.normalize_points_cams(new_Ps, raw_xs, Ns)
            new_Xs = geo_utils.dlt_triangulation(norm_P, norm_x, visible_points)
            # print(f"new_Xs1: {new_Xs.shape}")
            new_Rs, new_ts, new_Ps, new_Xs, new_xs, visible_points, point_indices, colidx = ceres_utils.run_euc_ceres_iter(
                new_Ps, new_Xs, raw_xs, new_Rs, new_ts, Ks, point_indices, visible_points, max_iter, ba_times, repro_thre, proj_second)
    else:
        visible_xs = xs[visible_points]                       # 2dptnum,2
        new_Rs, new_ts, new_Ps, new_Xs = ceres_utils.run_euclidean_python_ceres(
            Xs, visible_xs, Rs, ts, Ks, point_indices)
        if repeat:
            norm_P, norm_x = geo_utils.normalize_points_cams(new_Ps, xs, Ns)
            new_Xs = geo_utils.dlt_triangulation(norm_P, norm_x, visible_points)
            new_Rs, new_ts, new_Ps, new_Xs = ceres_utils.run_euclidean_python_ceres(
                new_Xs, visible_xs, new_Rs, new_ts, Ks, point_indices)
    new_Xs = np.concatenate([new_Xs, np.ones([new_Xs.shape[0],1])], axis=1)
    
    results['Rs'] = new_Rs
    results['ts'] = new_ts
    results['Ps'] = new_Ps
    results['Xs'] = new_Xs
    if refined: 
        results['valid_points'] = visible_points
        results['new_xs'] = new_xs
        results['colidx'] = colidx

    return results


def proj_ba(Ps, xs, Xs_our=None, Ns=None, repeat=True, triangulation=False, return_repro=True,normalize_in_tri=True):
    """
    Computes bundle adjustment with ceres solve
    :param Ps: cameras [m,3,4]. Ps[i] = Ks[i] @ Rs[i].T @ [I, -ts[i]]
    :param xs: 2d points [m,n,2]
    :param Xs_our: initial 3d points [n,3] or None if triangulation needed
    :param Ns: normalization matrices.
    :param repeat: run ba twice. default: True
    :param triangulation: For initial point run triangulation. default: False
    :param return_repro: compute and return the reprojection errors before and after.
    :param normalize_in_tri: Normalize the points and the cameras when computing triangulation. default: True
    :return: results. The new camera parameters, 3d points, and if requested the reprojection errors.
    """
    results = {}

    visible_points = xs[:, :, 0] != 0
    point_indices = np.stack(np.where(visible_points))
    visible_xs = xs[visible_points]

    if triangulation:
        if normalize_in_tri:
            if Ns is None:
                Ns = geo_utils.batch_get_normalization_matrices(xs)
            norm_P, norm_x = geo_utils.normalize_points_cams(Ps, xs, Ns)
            Xs = geo_utils.dlt_triangulation(norm_P, norm_x, visible_points)
        else:
            Xs = geo_utils.dlt_triangulation(Ps, xs, visible_points)
    else:
        Xs = Xs_our

    if return_repro:
        results['repro_before'] = np.nanmean(geo_utils.reprojection_error_with_points(Ps, Xs, xs, visible_points))

    new_Ps, new_Xs = ceres_utils.run_projective_python_ceres(Ps, Xs, visible_xs, point_indices)

    if repeat:
        if return_repro:
            results['repro_middle'] = np.nanmean(geo_utils.reprojection_error_with_points(new_Ps, new_Xs, xs, visible_points))

        if normalize_in_tri:
            if Ns is None:
                Ns = geo_utils.batch_get_normalization_matrices(xs)
            norm_P, norm_x = geo_utils.normalize_points_cams(new_Ps, xs, Ns)
            new_Xs = geo_utils.dlt_triangulation(norm_P, norm_x, visible_points)
        else:
            new_Xs = geo_utils.dlt_triangulation(new_Ps, xs, visible_points)

        new_Ps, new_Xs = ceres_utils.run_projective_python_ceres(new_Ps, new_Xs, visible_xs, point_indices)

    if return_repro:
        results['repro_after'] = np.nanmean(geo_utils.reprojection_error_with_points(new_Ps, new_Xs, xs, visible_points))

    new_Xs = np.concatenate([new_Xs, np.ones([new_Xs.shape[0],1])], axis=1)

    results['Ps'] = new_Ps
    results['Xs'] = new_Xs
    return results

