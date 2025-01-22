import torch
import os
from utils import general_utils, geo_utils, ba_functions
from pytorch3d import transforms as py3d_trans
import numpy as np
# from sklearn.metrics import precision_recall_curve
# import matplotlib.pyplot as plt


def prepare_predictions(data, pred_cam, conf, bundle_adjustment, refined=False):
    # Take the inputs from pred cam and turn to ndarray
    outputs = {}
    outputs['scan_name'] = data.scan_name
    calibrated = conf.get_bool('dataset.calibrated')

    Ns = data.Ns.cpu().numpy()
    Ks = data.Ks.cpu().numpy()  # Ks for calibrated, a normalization matrix for uncalibrated
    M = data.M.cpu().numpy()
    xs = geo_utils.M_to_xs(M)

    Ps_norm = pred_cam["Ps_norm"].cpu().numpy()  # Normalized camera!!
    Ps = Ks @ Ps_norm  # unnormalized cameras
    pts3D_pred = geo_utils.pflat(pred_cam["pts3D"]).cpu().numpy()

    pts3D_triangulated = geo_utils.n_view_triangulation(Ps, M=M, Ns=Ns)

    outputs['xs'] = xs  # to compute reprojection error later
    outputs['Ps'] = Ps  # Ps = K@(R|t)
    outputs['Ps_norm'] = Ps_norm  # Ps_norm = R|t
    outputs['pts3D_pred'] = pts3D_pred  # 4,m
    outputs['pts3D_triangulated'] = pts3D_triangulated  # 4,n

    if calibrated:
        outputs['Ks'] = Ks
        Rs_gt, ts_gt = geo_utils.decompose_camera_matrix(data.y.cpu().numpy(), Ks)  # For alignment and R,t errors
        outputs['Rs_gt'] = Rs_gt
        outputs['ts_gt'] = ts_gt

        Rs_pred, ts_pred = geo_utils.decompose_camera_matrix(Ps_norm)
        outputs['Rs'] = Rs_pred
        outputs['ts'] = ts_pred

        Rs_fixed, ts_fixed, similarity_mat = geo_utils.align_cameras(Rs_pred, Rs_gt, ts_pred, ts_gt, return_alignment=True) # Align  Rs_fixed, tx_fixed
        outputs['Rs_fixed'] = Rs_fixed
        outputs['ts_fixed'] = ts_fixed
        outputs['pts3D_pred_fixed'] = (similarity_mat @ pts3D_pred)  # 4,n
        outputs['pts3D_triangulated_fixed'] = (similarity_mat @ pts3D_triangulated)

        if bundle_adjustment:
            repeat = conf.get_bool('ba.repeat')
            triangulation = conf.get_bool('ba.triangulation')
            ba_res = ba_functions.euc_ba(xs, Rs=Rs_pred, ts=ts_pred, Ks=np.linalg.inv(Ns),
                                         Xs_our=pts3D_pred.T, Ps=None,
                                         Ns=Ns, repeat=repeat, triangulation=triangulation, 
                                         return_repro=True, refined=refined) #    Rs, ts, Ps, Xs
            outputs['Rs_ba'] = ba_res['Rs']
            outputs['ts_ba'] = ba_res['ts']
            outputs['Xs_ba'] = ba_res['Xs'].T  # 4,n
            outputs['Ps_ba'] = ba_res['Ps']
            if refined: outputs['valid_points'] = ba_res['valid_points']

            R_ba_fixed, t_ba_fixed, similarity_mat = geo_utils.align_cameras(ba_res['Rs'], Rs_gt, ba_res['ts'], ts_gt,
                                                                       return_alignment=True)  # Align  Rs_fixed, tx_fixed
            outputs['Rs_ba_fixed'] = R_ba_fixed
            outputs['ts_ba_fixed'] = t_ba_fixed
            outputs['Xs_ba_fixed'] = (similarity_mat @ outputs['Xs_ba'])

    else:
        if bundle_adjustment:
            repeat = conf.get_bool('ba.repeat')
            triangulation = conf.get_bool('ba.triangulation')
            ba_res = ba_functions.proj_ba(Ps=Ps, xs=xs, Xs_our=pts3D_pred.T, Ns=Ns, repeat=repeat,
                                          triangulation=triangulation, return_repro=True, normalize_in_tri=True)   # Ps, Xs
            outputs['Xs_ba'] = ba_res['Xs'].T  # 4,n
            outputs['Ps_ba'] = ba_res['Ps']

    return outputs


def compute_errors(outputs, conf, bundle_adjustment, refined=False):
    model_errors = {}
    
    # evaluate mask
    premask = outputs['premask']
    gtmask = outputs['gtmask']
    
    tp, fp, tn, fn = general_utils.compute_confusion_matrix(premask, gtmask)
    accuracy, precision, recall, F1 = general_utils.compute_indexes(tp, fp, tn, fn)
    model_errors["TP"] = tp
    model_errors["FP"] = fp
    model_errors["TN"] = tn
    model_errors["FN"] = fn
    model_errors["Accuracy"] = accuracy
    model_errors["Precision"] = precision
    model_errors["Recall"] = recall
    model_errors["F1"] = F1
    
    calibrated = conf.get_bool('dataset.calibrated')
    Ps = outputs['Ps']
    pts3D_pred = outputs['pts3D_pred']
    xs = outputs['xs']
    #pts3D_triangulated = outputs['pts3D_triangulated']

    our_repro_error = geo_utils.reprojection_error_with_points(Ps, pts3D_pred.T, xs)
    if not our_repro_error.shape: return model_errors
    model_errors["our_repro"] = np.nanmean(our_repro_error)
    model_errors["our_repro_max"] = np.nanmax(our_repro_error)
    if calibrated:
        # Rs_fixed = outputs['Rs_fixed']
        # ts_fixed = outputs['ts_fixed']
        Rs_fixed = outputs['Rs']
        ts_fixed = outputs['ts']
        Rs_gt = outputs['Rs_gt']
        ts_gt = outputs['ts_gt']
        Rs_error, ts_error = geo_utils.tranlsation_rotation_errors(Rs_fixed, ts_fixed, Rs_gt, ts_gt)
        model_errors["ts_mean"] = np.mean(ts_error)
        model_errors["ts_med"] = np.median(ts_error)
        model_errors["ts_max"] = np.max(ts_error)
        model_errors["Rs_mean"] = np.mean(Rs_error)
        model_errors["Rs_med"] = np.median(Rs_error)
        model_errors["Rs_max"] = np.max(Rs_error)

    if bundle_adjustment:
        Xs_ba = outputs['Xs_ba']
        Ps_ba = outputs['Ps_ba']
        if refined: 
            valid_points = outputs['valid_points']
            new_xs = outputs['new_xs']
            repro_ba_error = geo_utils.reprojection_error_with_points(Ps_ba, Xs_ba.T, new_xs, visible_points=valid_points)
        else:
            repro_ba_error = geo_utils.reprojection_error_with_points(Ps_ba, Xs_ba.T, xs)
        model_errors['repro_ba'] = np.nanmean(repro_ba_error)
        model_errors['repro_ba_max'] = np.nanmax(repro_ba_error)
        if calibrated:
            Rs_fixed = outputs['Rs_ba_fixed']
            ts_fixed = outputs['ts_ba_fixed']
            Rs_gt = outputs['Rs_gt']
            ts_gt = outputs['ts_gt']
            Rs_ba_error, ts_ba_error = geo_utils.tranlsation_rotation_errors(Rs_fixed, ts_fixed, Rs_gt, ts_gt)
            model_errors["ts_ba_mean"] = np.mean(ts_ba_error)
            model_errors["ts_ba_med"] = np.median(ts_ba_error)
            model_errors["ts_ba_max"] = np.max(ts_ba_error)
            model_errors["Rs_ba_mean"] = np.mean(Rs_ba_error)
            model_errors["Rs_ba_med"] = np.median(Rs_ba_error)
            model_errors["Rs_ba_max"] = np.max(Rs_ba_error)
    # Rs errors mean, ts errors mean, ba repro, rs ba mean, ts ba mean

    # projected_pts = geo_utils.get_positive_projected_pts_mask(Ps @ pts3D_pred, conf.get_float('loss.infinity_pts_margin'))
    # #projected_pts = geo_utils.get_positive_projected_pts_mask(Ps @ pts3D_triangulated, conf.get_float('loss.infinity_pts_margin'))
    # valid_pts = geo_utils.xs_valid_points(xs)
    # unprojected_pts = np.logical_and(~projected_pts, valid_pts)
    # part_unprojected = unprojected_pts.sum() / valid_pts.sum()

    # model_errors['unprojected'] = part_unprojected

    return model_errors


def prepare_predictions_loop(data, pred_cam, conf, bundle_adjustment):
    # Take the inputs from pred cam and turn to ndarray
    outputs = {}
    outputs['scan_name'] = data.scan_name
    calibrated = conf.get_bool('dataset.calibrated')

    Ns = data.Ns.cpu().numpy()
    Ks = data.Ks.cpu().numpy()  # Ks for calibrated, a normalization matrix for uncalibrated
    M = data.M.cpu().numpy()
    xs = geo_utils.M_to_xs(M)

    Ps_norm = pred_cam["Ps_norm"].cpu().numpy()  # Normalized camera!!
    Ps = Ks @ Ps_norm  # unnormalized cameras
    pts3D_pred = geo_utils.pflat(pred_cam["pts3D"]).cpu().numpy()

    pts3D_triangulated = geo_utils.n_view_triangulation(Ps, M=M, Ns=Ns)

    outputs['xs'] = xs  # to compute reprojection error later
    outputs['Ps'] = Ps  # Ps = K@(R|t)
    outputs['Ps_norm'] = Ps_norm  # Ps_norm = R|t
    outputs['pts3D_pred'] = pts3D_pred  # 4,m
    outputs['pts3D_triangulated'] = pts3D_triangulated  # 4,n

    if calibrated:
        outputs['Ks'] = Ks
        Rs_gt, ts_gt = geo_utils.decompose_camera_matrix(data.y.cpu().numpy(), Ks)  # For alignment and R,t errors
        outputs['Rs_gt'] = Rs_gt
        outputs['ts_gt'] = ts_gt

        Rs_pred, ts_pred = geo_utils.decompose_camera_matrix(Ps_norm)
        outputs['Rs'] = Rs_pred
        outputs['ts'] = ts_pred

        Rs_fixed, ts_fixed, similarity_mat = geo_utils.align_cameras(Rs_pred, Rs_gt, ts_pred, ts_gt, return_alignment=True) # Align  Rs_fixed, tx_fixed
        outputs['Rs_fixed'] = Rs_fixed
        outputs['ts_fixed'] = ts_fixed
        outputs['pts3D_pred_fixed'] = (similarity_mat @ pts3D_pred)  # 4,n
        outputs['pts3D_triangulated_fixed'] = (similarity_mat @ pts3D_triangulated)

        if bundle_adjustment:
            repeat = conf.get_bool('ba.repeat')
            triangulation = conf.get_bool('ba.triangulation')
            ba_res = ba_functions.euc_ba(xs, Rs=Rs_pred, ts=ts_pred, Ks=np.linalg.inv(Ns),
                                         Xs_our=pts3D_pred.T, Ps=None,
                                         Ns=Ns, repeat=repeat, triangulation=triangulation, return_repro=True) #    Rs, ts, Ps, Xs
            
            repro_ba_error = geo_utils.reprojection_error_with_points(ba_res['Ps'], ba_res['Xs'], xs)
            
            
            outputs['Rs_ba'] = ba_res['Rs']
            outputs['ts_ba'] = ba_res['ts']
            outputs['Xs_ba'] = ba_res['Xs'].T  # 4,n
            outputs['Ps_ba'] = ba_res['Ps']

            R_ba_fixed, t_ba_fixed, similarity_mat = geo_utils.align_cameras(ba_res['Rs'], Rs_gt, ba_res['ts'], ts_gt,
                                                                       return_alignment=True)  # Align  Rs_fixed, tx_fixed
            outputs['Rs_ba_fixed'] = R_ba_fixed
            outputs['ts_ba_fixed'] = t_ba_fixed
            outputs['Xs_ba_fixed'] = (similarity_mat @ outputs['Xs_ba'])

    else:
        if bundle_adjustment:
            repeat = conf.get_bool('ba.repeat')
            triangulation = conf.get_bool('ba.triangulation')
            ba_res = ba_functions.proj_ba(Ps=Ps, xs=xs, Xs_our=pts3D_pred.T, Ns=Ns, repeat=repeat,
                                          triangulation=triangulation, return_repro=True, normalize_in_tri=True)   # Ps, Xs
            outputs['Xs_ba'] = ba_res['Xs'].T  # 4,n
            outputs['Ps_ba'] = ba_res['Ps']

    return outputs


def sigmoid_numpy(x):
    return 1/(1+np.exp(-x))


def prepare_predictions_2(data, pred_mask, pred_cam, conf, epoch, bundle_adjustment, refined):
    # Take the inputs from pred cam and turn to ndarray
    outputs = {}
    outputs['scan_name'] = data.scan_name
    train_trans = conf.get_bool('train.train_trans')

    Ns = data.Ns.cpu().numpy()
    color = data.color.cpu().numpy()
    
    # processing mask
    thred = conf.get_float('loss.mask_thred')   
    premask = pred_mask.to('cpu')
    premask.values = sigmoid_numpy(premask.values)
    predensemask = torch.sparse_coo_tensor(premask.indices, premask.values, premask.shape).to_dense().numpy().squeeze()>thred   
    prediction = premask.values.numpy().squeeze()
    outputs['prediction'] = prediction
    premask = prediction>thred   
    gtmask = data.mask_sparse.values.cpu().numpy().squeeze()
    outputs['premask'] = premask+0
    outputs['gtmask'] = gtmask.astype(int)

    # precisions, recalls, thresholds = precision_recall_curve(gtmask.reshape(-1), prediction.reshape(-1))
    # plt.plot(precisions, recalls)
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # print(f"pr thresholds: {thresholds}")
    # prpath = os.path.join(conf.get_string('dataset.results_path'), "pr_img")
    # if not os.path.exists(prpath): os.mkdir(prpath)
    # plt.savefig(prpath+"/"+str(epoch)+".jpg")
    
    # processing M
    M = data.M.cpu().numpy()
    mask = np.zeros(M.shape)
    mask[0:mask.shape[0]:2]=predensemask
    mask[1:mask.shape[0]:2]=predensemask
    preM = M*mask
    colidx = (mask!=0).sum(axis=0)>2
    preM = preM[:, colidx]    
    precolor = color[:, colidx]
    outputs['precolor'] = precolor
    outputs['raw_color'] = color
    # M = M[(mask!=0).sum(axis=1)!=0, :]   
    # print(M.shape)
    # input()
    raw_xs = geo_utils.M_to_xs(M)
    xs = geo_utils.M_to_xs(preM)
    
    Rs_pred = py3d_trans.quaternion_to_matrix(geo_utils.norm_quats(pred_cam["quats"])).cpu().numpy()
    if train_trans:
        ts_pred = (data.gpss.cpu().numpy() + pred_cam["ts"].cpu().numpy()) * data.scale
        # ts_pred = pred_cam["ts"].cpu().numpy() * data.scale
    else:
        ts_pred = data.gpss.cpu().numpy() * data.scale
    
    #Rs_gt, ts_gt = geo_utils.decompose_camera_matrix(data.y.cpu().numpy(), Ks)  # For alignment and R,t errors
    Rs_gt, ts_gt = data.Rs.cpu().numpy(), data.ts.cpu().numpy()
    outputs['Rs_gt'] = Rs_gt
    outputs['Rs'] = Rs_pred
    outputs['ts_gt'] = ts_gt * data.scale
    outputs['ts'] = ts_pred

    Ks = data.Ks.cpu().numpy()  # data.Ns.inverse().cpu().numpy()
    outputs['Ks'] = Ks 
    Ps = geo_utils.batch_get_camera_matrix_from_rtk(Rs_pred, ts_pred, Ks)
    pts3D_triangulated = geo_utils.n_view_triangulation(Ps, M=preM, Ns=Ns)
    outputs['xs'] = xs  # to compute reprojection error later
    outputs['raw_xs'] = raw_xs
    outputs['Ps'] = Ps  # Ps = K@(R|t)
    outputs['pts3D_pred'] = pts3D_triangulated  # 4,n

    #Rs_pred, ts_pred = geo_utils.decompose_camera_matrix(Ps_norm)

    # Rs_fixed, ts_fixed, similarity_mat = geo_utils.align_cameras(Rs_pred, Rs_gt, ts_pred, ts_gt, return_alignment=True) # Align  Rs_fixed, tx_fixed
    # outputs['Rs_fixed'] = Rs_fixed
    # outputs['ts_fixed'] = ts_fixed
    # outputs['pts3D_pred_fixed'] = (similarity_mat @ pts3D_triangulated)
    # outputs['Rs_fixed'] = Rs_pred
    # outputs['ts_fixed'] = ts_pred
    # outputs['pts3D_pred_fixed'] = pts3D_triangulated

    if bundle_adjustment:
        repeat = conf.get_bool('ba.repeat')
        max_iter = conf.get_int('ba.max_iter')
        ba_times = conf.get_int('ba.ba_times')
        repro_thre = conf.get_float('ba.repro_thre')
        ba_res = ba_functions.euc_ba(xs, raw_xs, Rs=Rs_pred, ts=ts_pred, Ks=np.linalg.inv(Ns),
                                    Xs=pts3D_triangulated.T, Ps=Ps, Ns=Ns, 
                                    repeat=repeat, max_iter=max_iter, ba_times=ba_times,
                                    repro_thre=repro_thre, refined=refined) #    Rs, ts, Ps, Xs
        outputs['Rs_ba'] = ba_res['Rs']
        outputs['ts_ba'] = ba_res['ts']
        outputs['Xs_ba'] = ba_res['Xs'].T  # 4,n
        outputs['Ps_ba'] = ba_res['Ps']
        if refined: 
            outputs['valid_points'] = ba_res['valid_points']
            outputs['new_xs'] = ba_res['new_xs']

        # R_ba_fixed, t_ba_fixed, similarity_mat = geo_utils.align_cameras(ba_res['Rs'], Rs_gt, ba_res['ts'], ts_gt,
        #                                                             return_alignment=True)  # Align  Rs_fixed, tx_fixed
        # outputs['Rs_ba_fixed'] = R_ba_fixed
        # outputs['ts_ba_fixed'] = t_ba_fixed
        # outputs['Xs_ba_fixed'] = (similarity_mat @ outputs['Xs_ba'])
        outputs['Rs_ba_fixed'] = ba_res['Rs']
        outputs['ts_ba_fixed'] = ba_res['ts']
        outputs['Xs_ba_fixed'] = ba_res['Xs'].T

    # else:
    #     if bundle_adjustment:
    #         repeat = conf.get_bool('ba.repeat')
    #         triangulation = conf.get_bool('ba.triangulation')
    #         ba_res = ba_functions.proj_ba(Ps=Ps, xs=xs, Xs_our=pts3D_triangulated.T, Ns=Ns, repeat=repeat,
    #                                       triangulation=triangulation, return_repro=True, normalize_in_tri=True)   # Ps, Xs
    #         outputs['Xs_ba'] = ba_res['Xs'].T  # 4,n
    #         outputs['Ps_ba'] = ba_res['Ps']

    return outputs


def prepare_ptpredictions(data, pred_mask):
    # Take the inputs from pred cam and turn to ndarray
    outputs = {}
    outputs['scan_name'] = data.scan_name

    premask = pred_mask.to('cpu')
    prediction = premask.values.numpy().squeeze()
    outputs['prediction'] = prediction
    premask = prediction>0.8   
    gtmask = data.mask_sparse.values.cpu().numpy().squeeze()
    outputs['premask'] = premask+0
    outputs['gtmask'] = gtmask.astype(int)

    return outputs