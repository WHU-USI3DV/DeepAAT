import cv2  # DO NOT REMOVE
from time import time
from utils import general_utils, dataset_utils, plot_utils
from utils.Phases import Phases
from datasets import SceneData, ScenesDataSet
from datasets.ScenesDataSet import DataLoader, myDataSet
from single_scene_optimization import train_single_model
import evaluation
import torch
import pandas as pd

def inference(conf, device, phase):
    
    # Get conf
    flag = conf.get_int("dataset.flag")             # 1 means val, 2 means test
    scans_list = SceneData.get_data_list(conf, flag)    
    bundle_adjustment = conf.get_bool("ba.run_ba")
    refined = conf.get_bool("ba.refined")
    
    # Create model
    model_path = conf.get_string('model.model_path')
    model = general_utils.get_class("models." + conf.get_string("model.type"))(conf).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    errors_list = []
    with torch.no_grad():
        data_loader = myDataSet(conf, flag, scans_list).to(device)
        for batch_data in data_loader:
            for scene_data in batch_data:
                print(f"processing {scene_data.scan_name} ...")

                # Optimize Scene
                begin_time = time()
                pred_mask, pred_cam = model(scene_data)
                pred_time = time() - begin_time
                outputs = evaluation.prepare_predictions_2(scene_data, pred_mask, pred_cam, conf, 0, bundle_adjustment, refined=refined)
                errors = evaluation.compute_errors(outputs, conf, bundle_adjustment, refined=refined)
                
                errors['Inference time'] = pred_time
                errors['Scene'] = scene_data.scan_name
                errors['all_pts'] = scene_data.M.shape[-1]
                errors['pred_pts'] = outputs['pts3D_pred'].shape[1]
                # errors['after_ba_pts'] = outputs['Xs_ba'].shape[1]
                errors['gt_pts'] = scene_data.mask[:,scene_data.mask.sum(axis=0)!=0].shape[1]
                
                errors_list.append(errors)
                dataset_utils.save_cameras(outputs, conf, curr_epoch=None, phase=phase)
                plot_utils.plot_cameras_before_and_after_ba(outputs, errors, conf, phase, scan=scene_data.scan_name, epoch=None, bundle_adjustment=bundle_adjustment)
    
    # Write results
    df_errors = pd.DataFrame(errors_list)
    mean_errors = df_errors.mean(numeric_only=True)
    # df_errors = pd.concat([df_errors,mean_errors], axis=0, ignore_index=True)
    df_errors = df_errors.append(mean_errors, ignore_index=True)
    df_errors.at[df_errors.last_valid_index(), "Scene"] = "Mean"    
    df_errors.set_index("Scene", inplace=True)    
    df_errors = df_errors.round(3)
    print(df_errors.to_string(), flush=True)    
    general_utils.write_results(conf, df_errors, file_name="Inference")    
    

if __name__ == "__main__":
    conf, device, phase = general_utils.init_exp(Phases.INFERENCE.name)
    inference(conf, device, phase)