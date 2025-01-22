import cv2  # DO NOT REMOVE
from utils import general_utils, dataset_utils
from utils.Phases import Phases
from datasets.ScenesDataSet import ScenesDataSet, DataLoader, myDataSet
from datasets import SceneData
from single_scene_optimization import train_single_model
import train
import copy
import time


def main():
    # Init Experiment
    conf, device, phase = general_utils.init_exp(Phases.TRAINING.name)    
    general_utils.log_code(conf)                                          

    # Get configuration
    sample = conf.get_bool('dataset.sample')
    batch_size = conf.get_int('dataset.batch_size')

    train_list = SceneData.get_data_list(conf, 0)
    val_list = SceneData.get_data_list(conf, 1)
    test_list = SceneData.get_data_list(conf, 2)

    if sample:
        min_sample_size = conf.get_int('dataset.min_sample_size')
        max_sample_size = conf.get_int('dataset.max_sample_size')
        # optimization_num_of_epochs = conf.get_int("train.optimization_num_of_epochs")
        # optimization_eval_intervals = conf.get_int('train.optimization_eval_intervals')
        # optimization_lr = conf.get_int('train.optimization_lr')

        # Create train, test and validation sets
        train_scenes = SceneData.create_scene_data_from_list(train_list, conf, 0)
        validation_scenes = SceneData.create_scene_data_from_list(val_list, conf, 1)
        test_scenes = SceneData.create_scene_data_from_list(test_list, conf, 2)

        train_set = ScenesDataSet(train_scenes, return_all=False, min_sample_size=min_sample_size, max_sample_size=max_sample_size)
        validation_set = ScenesDataSet(validation_scenes, return_all=True)
        test_set = ScenesDataSet(test_scenes, return_all=True)

        # Create dataloaders
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True).to(device)
        validation_loader = DataLoader(validation_set, batch_size=1, shuffle=False).to(device)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False).to(device)
    else:
        train_loader = myDataSet(conf, 0, train_list, batch_size, True).to(device)
        validation_loader = myDataSet(conf, 1, val_list).to(device)
        test_loader = myDataSet(conf, 2, test_list).to(device)

    # Train model
    model = general_utils.get_class("models." + conf.get_string("model.type"))(conf).to(device)
    train_stat, train_errors, validation_errors, test_errors = train.train(conf, train_loader, model, phase, validation_loader, test_loader)
    # Write results
    general_utils.write_results(conf, train_stat, file_name="Train_Stats")
    general_utils.write_results(conf, train_errors, file_name="Train")
    general_utils.write_results(conf, validation_errors, file_name="Validation")
    general_utils.write_results(conf, test_errors, file_name="Test")


def optimization_all_sets(conf, device, phase):
    # Get logs directories
    scans_list = conf.get_list('dataset.scans_list')
    for i, scan in enumerate(scans_list):
        conf["dataset"]["scan"] = scan
        train_single_model(conf, device, phase)


if __name__ == "__main__":
    main()