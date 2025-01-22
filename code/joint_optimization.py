import numpy as np
from scipy.sparse import coo_matrix
from utils import geo_utils, ceres_utils, ba_functions
import pandas as pd

# m,3,3  m,3,3  m,3
def get_Ps_gt(K_gt, R_gt, T_gt):
    m = R_gt.shape[0]
    K_RT = np.matmul(K_gt, R_gt.transpose((0, 2, 1)))                    # K@R.T
    Is = np.expand_dims(np.identity(3), 0).repeat(m, axis = 0)
    tmp = np.concatenate((Is, -np.expand_dims(T_gt, 2)), axis = 2)       # I|-t   
    Ps_gt = np.matmul(K_RT, tmp)
    return Ps_gt


def loadtxt(classed_txt):
    ids_list = []
    with open(classed_txt, 'r') as f:
        for line in f:
            ids0 = list(map(int, line[:-1].strip().split(' ')))
            ids = [x-1 for x in ids0]
            ids.sort()
            ids_list.append(ids)
    return ids_list


def get_col_ids(cond, minval):
    ids = np.arange(cond.shape[1])
    return ids[np.sum(cond!=0, axis=0)>=minval]


def make_full_scene_by_predRt_rawM(pred_npzs, raw_npzs, full_npz, classed_txt, outpath):
    
    full_f = np.load(full_npz)
    Rs_gt = full_f['newRs']
    ts_gt = full_f['newts']
    Ns = full_f['Ns']
    Ks = np.linalg.inv(Ns)
    M_data = full_f['M_data']
    M_row = full_f['M_row']
    M_col = full_f['M_col']
    M_shape = full_f['M_shape']
    M = coo_matrix((M_data, (M_row, M_col)), shape=M_shape).todense().A
    xs = geo_utils.M_to_xs(M)
    
    ids_list = loadtxt(classed_txt)
    Rs = np.zeros(Rs_gt.shape)
    ts = np.zeros(ts_gt.shape)
    
    scan_name = "ortho1_full"
    for i in range(len(pred_npzs)):
        fpred = np.load(pred_npzs[i])
        fraw = np.load(raw_npzs[i])
        ids = ids_list[i]
        Rs[ids,:] = fpred['Rs']
        ts[ids,:] = fpred['ts'] + fraw['ts'][0]
    
    Ps = get_Ps_gt(Ks, Rs, ts)
    pts3D_triangulated = geo_utils.n_view_triangulation(Ps, M=M, Ns=Ns)

    results = {}
    results['scan_name'] = scan_name
    results['xs'] = xs
    results['Rs'] = Rs
    results['ts'] = ts
    results['Rs_gt'] = Rs_gt
    results['ts_gt'] = ts_gt
    results['Ks'] = Ks
    results['pts3D_pred'] = pts3D_triangulated.T
    results['Ps'] = Ps
    results['raw_xs'] = xs
    results['raw_color'] = full_f['rgbs']
    np.savez(outpath, **results)


def make_full_scene_by_abaRtM(aba_npzs, raw_npzs, full_npz, classed_txt, outpath):
    print("in make_full_scene_by_abaRtM() now ...")
    print("1. get full raw data ...")
    full_f = np.load(full_npz)
    Rs_gt = full_f['newRs']
    ts_gt = full_f['newts']
    Ns = full_f['Ns']
    Ks = np.linalg.inv(Ns)
    M_data = full_f['M_data']
    M_row = full_f['M_row']
    M_col = full_f['M_col']
    M_shape = full_f['M_shape']
    raw_M = coo_matrix((M_data, (M_row, M_col)), shape=M_shape).todense().A
    raw_xs = geo_utils.M_to_xs(raw_M)
    xs = np.zeros((M_shape[0]//2, M_shape[1], 2))
    
    ids_list = loadtxt(classed_txt)
    Rs = np.zeros(Rs_gt.shape)
    ts = np.zeros(ts_gt.shape)
    
    print("2. process splited data ...")
    scan_name = "ortho1_full"
    for i in range(len(aba_npzs)):
        faba = np.load(aba_npzs[i])
        fraw = np.load(raw_npzs[i])
        camids = ids_list[i]                         
        colids = fraw['pt3d_idx'][faba['colidx']]    
        xstmp = np.zeros((len(camids), M_shape[1], 2))
        xstmp[:,colids,:] = faba['new_xs']
        xs[camids, :, :] = xstmp
        Rs[camids,:] = faba['Rs_ba_fixed']
        ts[camids,:] = faba['ts_ba_fixed'] + fraw['ts'][0]
    
    M=geo_utils.xs_to_M(xs)
    valid_colidx = get_col_ids(M,4)

    Ps = get_Ps_gt(Ks, Rs, ts)
    pts3D_triangulated = geo_utils.n_view_triangulation(Ps, M=M[:,valid_colidx], Ns=Ns)

    print("3. write results ...")
    results = {}
    results['precolor'] = full_f['rgbs'][:, valid_colidx]
    results['scan_name'] = scan_name
    results['xs'] = xs[:,valid_colidx,:]
    results['Rs'] = Rs
    results['ts'] = ts
    results['Rs_gt'] = Rs_gt
    results['ts_gt'] = ts_gt
    results['Ks'] = Ks
    results['pts3D_pred'] = pts3D_triangulated
    results['Ps'] = Ps
    results['raw_xs'] = raw_xs
    results['raw_color'] = full_f['rgbs']
    np.savez(outpath, **results)


def _runba(npzpath, repeat, max_iter, ba_times, repro_thre, refined, proj_first, proj_second):
    print("in _runba() now ...")
    outputs = {}

    print("1. load data ...")
    file = dict(np.load(npzpath))
    scan_name = file['scan_name']
    xs = file['xs']
    Rs_pred = file['Rs']
    ts_pred = file['ts']
    Rs_gt = file['Rs_gt']
    ts_gt = file['ts_gt']
    Ks = file['Ks']
    Ns = np.linalg.inv(Ks)
    # file['pts3D_pred'] = file['pts3D_pred'].T
    Xs = file['pts3D_pred']
    Ps = file['Ps']
    raw_xs = file['raw_xs']

    outputs['xs'] = xs
    outputs['Rs_gt'] = Rs_gt
    outputs['ts_gt'] = ts_gt

    print("2. ba now ...")
    ba_res = ba_functions.merged_ba(xs, raw_xs, Rs=Rs_pred, ts=ts_pred, Ks=Ks, Xs=Xs.T, Ps=Ps, Ns=Ns, 
                                repeat=repeat, max_iter=max_iter, ba_times=ba_times,
                                repro_thre=repro_thre, refined=refined, proj_first=proj_first, proj_second=proj_second) #    Rs, ts, Ps, Xs
    outputs['Rs_ba'] = ba_res['Rs']
    outputs['ts_ba'] = ba_res['ts']
    outputs['Xs_ba'] = ba_res['Xs'].T  # 4,n
    outputs['Ps_ba'] = ba_res['Ps']
    if refined: 
        outputs['valid_points'] = ba_res['valid_points']
        outputs['new_xs'] = ba_res['new_xs']
        outputs['colidx'] = ba_res['colidx']    

    print("3. align cams ...")
    R_ba_fixed, t_ba_fixed, similarity_mat = geo_utils.align_cameras(ba_res['Rs'], Rs_gt, ba_res['ts'], ts_gt,
                                                                return_alignment=True)  # Align  Rs_fixed, tx_fixed
    outputs['Rs_ba_fixed'] = R_ba_fixed
    outputs['ts_ba_fixed'] = t_ba_fixed
    outputs['Xs_ba_fixed'] = (similarity_mat @ outputs['Xs_ba'])
    # outputs['Rs_ba_fixed'] = ba_res['Rs']
    # outputs['ts_ba_fixed'] = ba_res['ts']
    # outputs['Xs_ba_fixed'] = ba_res['Xs'].T

    print("4. save results ...")
    file.update(outputs)
    np.savez(npzpath, **file)

    return file, scan_name


def _compute_errors(outputs, results_file_path, refined=False):
    print("in _compute_errors() now ...")
    model_errors = {}

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

    errors_list = []
    errors_list.append(model_errors)
    df_errors = pd.DataFrame(errors_list)
    mean_errors = df_errors.mean(numeric_only=True)
    # df_errors = pd.concat([df_errors,mean_errors], axis=0, ignore_index=True)
    df_errors = df_errors.append(mean_errors, ignore_index=True)
    df_errors.at[df_errors.last_valid_index(), "Scene"] = "Mean"    
    df_errors.set_index("Scene", inplace=True)    
    # df_errors = df_errors.round(3)
    print(df_errors.to_string(), flush=True)    
    df_errors.to_excel(results_file_path)    


if __name__=="__main__":

    refined = True    
    repeat = True    
    max_iter = 50
    ba_times = 2
    repro_thre = 2
    proj_first = True
    proj_second = True
    outputs, scan_name = _runba(out_npz_path, repeat, max_iter, ba_times, repro_thre, refined, proj_first, proj_second)
    _compute_errors(outputs, out_xlsx_path, refined=refined)    
