import numpy as np
from utils import geo_utils, ba_functions, general_utils
import os
import pandas as pd


def _runba(npzpath, repeat, max_iter, ba_times, repro_thre, refined):
    outputs = {}

    file = dict(np.load(npzpath))
    scan_name = file['scan_name']
    xs = file['xs']
    Rs_pred = file['Rs']
    ts_pred = file['ts']
    Rs_gt = file['Rs_gt']
    ts_gt = file['ts_gt']
    Ks = file['Ks']
    Ns = np.linalg.inv(Ks)
    Xs = file['pts3D_pred']
    Ps = file['Ps']
    raw_xs = file['raw_xs']

    outputs['xs'] = xs
    outputs['Rs_gt'] = Rs_gt
    outputs['ts_gt'] = ts_gt

    ba_res = ba_functions.euc_ba(xs, raw_xs, Rs=Rs_pred, ts=ts_pred, Ks=Ks, Xs=Xs.T, Ps=Ps, Ns=Ns, 
                                repeat=repeat, max_iter=max_iter, ba_times=ba_times,
                                repro_thre=repro_thre, refined=refined) #    Rs, ts, Ps, Xs
    outputs['Rs_ba'] = ba_res['Rs']
    outputs['ts_ba'] = ba_res['ts']
    outputs['Xs_ba'] = ba_res['Xs'].T  # 4,n
    outputs['Ps_ba'] = ba_res['Ps']
    if refined: 
        outputs['valid_points'] = ba_res['valid_points']
        outputs['new_xs'] = ba_res['new_xs']
        outputs['colidx'] = ba_res['colidx']    

    R_ba_fixed, t_ba_fixed, similarity_mat = geo_utils.align_cameras(ba_res['Rs'], Rs_gt, ba_res['ts'], ts_gt,
                                                                return_alignment=True)  # Align  Rs_fixed, tx_fixed
    outputs['Rs_ba_fixed'] = R_ba_fixed
    outputs['ts_ba_fixed'] = t_ba_fixed
    outputs['Xs_ba_fixed'] = (similarity_mat @ outputs['Xs_ba'])
    # outputs['Rs_ba_fixed'] = ba_res['Rs']
    # outputs['ts_ba_fixed'] = ba_res['ts']
    # outputs['Xs_ba_fixed'] = ba_res['Xs'].T
    file.update(outputs)
    np.savez(npzpath, **file)

    return file, scan_name


def _compute_errors(outputs, refined=False):
    model_errors = {}

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

    pts3D_pred = outputs['pts3D_pred']
    Ps = outputs['Ps']
    Rs_fixed = outputs['Rs']
    ts_fixed = outputs['ts']
    Rs_gt = outputs['Rs_gt']
    ts_gt = outputs['ts_gt']
    xs = outputs['xs']
    Xs_ba = outputs['Xs_ba']
    Ps_ba = outputs['Ps_ba']
    our_repro_error = geo_utils.reprojection_error_with_points(Ps, pts3D_pred.T, xs)
    if not our_repro_error.shape: return model_errors
    model_errors["our_repro"] = np.nanmean(our_repro_error)
    model_errors["our_repro_max"] = np.nanmax(our_repro_error)

    Rs_error, ts_error = geo_utils.tranlsation_rotation_errors(Rs_fixed, ts_fixed, Rs_gt, ts_gt)
    model_errors["ts_mean"] = np.mean(ts_error)
    model_errors["ts_med"] = np.median(ts_error)
    model_errors["ts_max"] = np.max(ts_error)
    model_errors["Rs_mean"] = np.mean(Rs_error)
    model_errors["Rs_med"] = np.median(Rs_error)
    model_errors["Rs_max"] = np.max(Rs_error)

    if refined: 
        valid_points = outputs['valid_points']
        new_xs = outputs['new_xs']
        repro_ba_error = geo_utils.reprojection_error_with_points(Ps_ba, Xs_ba.T, new_xs, visible_points=valid_points)
    else:
        repro_ba_error = geo_utils.reprojection_error_with_points(Ps_ba, Xs_ba.T, xs)
    model_errors['repro_ba'] = np.nanmean(repro_ba_error)
    model_errors['repro_ba_max'] = np.nanmax(repro_ba_error)
    
    Rs_ba_fixed = outputs['Rs_ba_fixed']
    ts_ba_fixed = outputs['ts_ba_fixed']
    Rs_ba_error, ts_ba_error = geo_utils.tranlsation_rotation_errors(Rs_ba_fixed, ts_ba_fixed, Rs_gt, ts_gt)
    model_errors["ts_ba_mean"] = np.mean(ts_ba_error)
    model_errors["ts_ba_med"] = np.median(ts_ba_error)
    model_errors["ts_ba_max"] = np.max(ts_ba_error)
    model_errors["Rs_ba_mean"] = np.mean(Rs_ba_error)
    model_errors["Rs_ba_med"] = np.median(Rs_ba_error)
    model_errors["Rs_ba_max"] = np.max(Rs_ba_error)

    return model_errors


def runba():
    basedir = "/home/zeeq/results/test10"
    refined = True    
    repeat = True    
    max_iter = 50    
    ba_times = 1    
    repro_thre = 5    

    errors_list = []
    for _,_,files in os.walk(basedir):
        for f in files:
            print(f'processing {f} ...')
            npzpath = os.path.join(basedir, f)
            outputs, scan_name = _runba(npzpath, repeat, max_iter, ba_times, repro_thre, refined)
            errors = _compute_errors(outputs, refined)
            errors['Scene'] = scan_name
            errors_list.append(errors)

    df_errors = pd.DataFrame(errors_list)
    mean_errors = df_errors.mean(numeric_only=True)
    # df_errors = df_errors.append(mean_errors, ignore_index=True)
    df_errors = pd.concat([df_errors, mean_errors], ignore_index=True)
    df_errors.at[df_errors.last_valid_index(), "Scene"] = "Mean"    
    df_errors.set_index("Scene", inplace=True)    
    df_errors = df_errors.round(3)
    print(df_errors.to_string(), flush=True)    

if __name__=="__main__":
    runba()
