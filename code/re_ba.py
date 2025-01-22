import numpy as np
from utils import geo_utils, ba_functions, general_utils
import os
import pandas as pd
from scipy.sparse import coo_matrix


# get Ps_gt
# m,3,3  m,3,3  m,3
def get_Ps_gt(K_gt, R_gt, T_gt):
    m = R_gt.shape[0]
    K_RT = np.matmul(K_gt, R_gt.transpose((0, 2, 1)))                    # K@R.T
    Is = np.expand_dims(np.identity(3), 0).repeat(m, axis = 0)
    tmp = np.concatenate((Is, -np.expand_dims(T_gt, 2)), axis = 2)       # I|-t   
    Ps_gt = np.matmul(K_RT, tmp)
    return Ps_gt


def _runba(npzpath, repeat, max_iter, ba_times, repro_thre, refined):
    outputs = {}

    scan_name = npzpath.split('/')[-1].split('.')[0]
    file = dict(np.load(npzpath))
    M_row = file['M_row']
    M_col = file['M_col']
    M_data = file['M_data']
    M_shape = file['M_shape']
    mask_row = file['mask_row']
    mask_col = file['mask_col']
    mask_data = file['mask_data']
    mask_shape = file['mask_shape']
    M = coo_matrix((M_data, (M_row, M_col)), shape=M_shape).todense().A
    mask = coo_matrix((mask_data, (mask_row, mask_col)), shape=mask_shape).todense().A

    enu = file['enu']
    Ns = file['Ns']
    Rs = file['Rs']
    ts = file['ts']
    quats = file['quats']
    color = file['rgbs']
    Ks = np.linalg.inv(Ns)
    Ps = get_Ps_gt(Ks, Rs, enu)

    M_gt = M*mask
    colidx = (mask!=0).sum(axis=0)>2
    M_gt = M_gt[:, colidx]    
    color_gt = color[:, colidx]
    xs_gt = geo_utils.M_to_xs(M_gt)

    xs_raw = geo_utils.M_to_xs(M)
    Xs_gt = geo_utils.n_view_triangulation(Ps, M=M_gt, Ns=Ns)
    Xs_raw = geo_utils.n_view_triangulation(Ps, M=M, Ns=Ns)

    Rs_error_b, ts_error_b = geo_utils.tranlsation_rotation_errors(Rs, ts, Rs, enu)
    print(f"before BA, Rs_error={Rs_error_b}, ts_error={ts_error_b}")

    ba_res = ba_functions.euc_ba(xs_gt, xs_raw, Rs=Rs, ts=enu, Ks=Ks, Xs=Xs_gt.T, Ps=Ps, Ns=Ns, 
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
    # M_new = geo_utils.xs_to_M(outputs['new_xs'])
    # mask_new = np.array((M_new!=0), dtype=np.int8)
    # colidx2 = (M_new!=0).sum(axis=0)>2
    # M_new = M_new[:,colidx2]

    R_ba_fixed, t_ba_fixed, similarity_mat = geo_utils.align_cameras(ba_res['Rs'], Rs, ba_res['ts'], enu,
                                                                return_alignment=True)  # Align  Rs_fixed, tx_fixed
    outputs['Rs_ba_fixed'] = R_ba_fixed
    outputs['ts_ba_fixed'] = t_ba_fixed
    outputs['Xs_ba_fixed'] = (similarity_mat @ outputs['Xs_ba'])
    
    Rs_error_a, ts_error_a = geo_utils.tranlsation_rotation_errors(R_ba_fixed, t_ba_fixed, Rs, enu)
    print(f"after BA, Rs_error={Rs_error_a}, ts_error={ts_error_a}")
    np.savetxt(ts_path, t_ba_fixed, delimiter=',')
    np.savetxt(enu_path, enu, delimiter=',')
    input()
    file.update(outputs)
    # np.savez(npzpath, **file)

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
    basedir = ""
    refined = True    
    repeat = True    
    max_iter = 50    
    ba_times = 1    
    repro_thre = 2   

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
    df_errors = df_errors.append(mean_errors, ignore_index=True)
    df_errors.at[df_errors.last_valid_index(), "Scene"] = "Mean"    
    df_errors.set_index("Scene", inplace=True)    
    df_errors = df_errors.round(3)
    print(df_errors.to_string(), flush=True)    

if __name__=="__main__":
    runba()