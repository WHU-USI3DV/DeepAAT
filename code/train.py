import torch
import math

import loss_functions
import evaluation
import copy
from utils import path_utils, dataset_utils, plot_utils, geo_utils
from time import time
import pandas as pd
from utils.Phases import Phases
from utils.path_utils import path_to_exp
import os
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

# import fitlog

# for each epoch
def epoch_train(train_data, model, loss_func1, loss_func2, optimizer, scheduler, epoch, min_valid_pts):
    model.train()
    train_losses = []
    bce_losses = []
    orient_losses = []
    trans_losses = []
    
    # for each batch
    for train_batch in train_data:  # Loop over all sets - 30
        batch_loss = torch.tensor([0.0], device=train_data.device)
        optimizer.zero_grad()
        
        # for each data
        for curr_data in train_batch:
            
            pred_mask, pred_cam = model(curr_data)
            bceloss = loss_func1(pred_mask, curr_data, epoch)
            gtloss, orient_loss, trans_loss = loss_func2(pred_cam, curr_data, epoch)
            loss = bceloss + gtloss
            batch_loss += loss
            train_losses.append(loss.item())
            bce_losses.append(bceloss.item())
            orient_losses.append(orient_loss.item())
            trans_losses.append(trans_loss.item())
        
        if batch_loss.item()>0:
            batch_loss.backward()
            optimizer.step()    
    
    scheduler.step()    
    mean_loss = torch.tensor(train_losses).mean()
    mean_bceloss = torch.tensor(bce_losses).mean()
    mean_oriloss = torch.tensor(orient_losses).mean()
    mean_trloss = torch.tensor(trans_losses).mean()
    return mean_loss, mean_bceloss, mean_oriloss, mean_trloss


def sim_epoch_train(conf, tbwriter, train_data, model, loss_func1, loss_func2, optimizer, scheduler, epoch, min_valid_pts, phase, validation_data, best_validation_metric):
    model.train()
    train_losses = []
    bce_losses = []
    orient_losses = []
    trans_losses = []
    
    i = 0
    epoch_size = len(train_data)
    print(f"there are {epoch_size} data in each epoch")
    eval_intervals = conf.get_int('train.eval_intervals', default=500)
    validation_metric = conf.get_list('train.validation_metric', default=["our_repro"])
    no_ba_during_training = not conf.get_bool('ba.only_last_eval')
    train_trans = conf.get_bool('train.train_trans')
    save_predictions = conf.get_bool('train.save_predictions')

    best_epoch = 0
    best_model = torch.empty(0)

    for train_batch in train_data:  # Loop over all sets
        batch_loss = torch.tensor([0.0], device=train_data.device)
        optimizer.zero_grad()
        
        for curr_data in train_batch:
            i+=1
            epoch0 = epoch*epoch_size+i
            # if not dataset_utils.is_valid_sample(curr_data, min_valid_pts):
            #     print('{} {} has a camera with not enough points'.format(epoch0, curr_data.scan_name))
            #     continue
            
            pred_mask, pred_cam = model(curr_data)
            bceloss = loss_func1(pred_mask, curr_data, epoch0)
            if train_trans:
                gtloss, orient_loss, trans_loss = loss_func2(pred_cam, curr_data, epoch0)
                loss = bceloss + gtloss
                batch_loss += loss
                train_losses.append(loss.item())
                bce_losses.append(bceloss.item())
                orient_losses.append(orient_loss.item())
                trans_losses.append(trans_loss.item())
            else:
                orient_loss = loss_func2(pred_cam, curr_data, epoch0)
                loss = bceloss + orient_loss
                batch_loss += loss
                train_losses.append(loss.item())
                bce_losses.append(bceloss.item())
                orient_losses.append(orient_loss.item())
            # if torch.isnan(gtloss):
            #     print(epoch0)
            #     continue
            # print(bceloss.item())
            # print(gtloss.item())
        
        # print(batch_loss.item())
        if batch_loss.item()>0:
            batch_loss.backward()
            optimizer.step()    
    
        mean_loss = torch.tensor(train_losses).mean()
        mean_bceloss = torch.tensor(bce_losses).mean()
        mean_oriloss = torch.tensor(orient_losses).mean()
        if train_trans: mean_trloss = torch.tensor(trans_losses).mean()

        epoch0 = epoch*epoch_size+i
        if epoch0 % 100 == 0:
            print('{} Train Loss: {}'.format(epoch0, mean_loss))
            tbwriter.add_scalar("Total_Loss", mean_loss, epoch0)
            tbwriter.add_scalar("BCE_Loss", mean_bceloss, epoch0)
            tbwriter.add_scalar("Orient_Loss", mean_oriloss, epoch0)
            if train_trans: tbwriter.add_scalar("Trans_Loss", mean_trloss, epoch0)
        # if epoch0 % 20 == 0:
        #     scheduler.step()    
        
        if epoch0!=0 and (epoch0 % eval_intervals == 0):  # Eval current results
            if phase is Phases.TRAINING:
                validation_errors = epoch_evaluation(validation_data, model, conf, epoch0, Phases.VALIDATION, save_predictions=save_predictions,bundle_adjustment=no_ba_during_training)
            else:
                validation_errors = epoch_evaluation(train_data, model, conf, epoch0, phase, save_predictions=save_predictions,bundle_adjustment=no_ba_during_training)

            metric = validation_errors.loc[["Mean"], validation_metric].sum(axis=1).values.item()
            # fitlog.add_metric({"dev":{"Acc":metric}}, step=epoch)
            
            if metric < best_validation_metric:
                best_validation_metric = metric
                best_epoch = epoch0
                best_model = copy.deepcopy(model)
                print('Updated best validation metric: {}'.format(best_validation_metric))
                path = path_utils.path_to_model(conf, phase, epoch=epoch0)
                torch.save(best_model.state_dict(), path)
    
    scheduler.step()    
    return best_epoch, best_validation_metric, best_model


def epoch_evaluation(data_loader, model, conf, epoch, phase, save_predictions=False, bundle_adjustment=True):
    refined = conf.get_bool("ba.refined")
    errors_list = []
    model.eval()    
    with torch.no_grad():
        for batch_data in data_loader:
            for curr_data in batch_data:
                # Get predictions
                begin_time = time()
                pred_mask, pred_cam = model(curr_data)    
                # pred_mask = model(curr_data)    
                pred_time = time() - begin_time

                # Eval results
                # outputs = evaluation.prepare_ptpredictions(curr_data, pred_mask)
                outputs = evaluation.prepare_predictions_2(curr_data, pred_mask, pred_cam, conf, epoch, bundle_adjustment, refined=refined)
                errors = evaluation.compute_errors(outputs, conf, bundle_adjustment, refined=refined)

                errors['Inference time'] = pred_time
                errors['Scene'] = curr_data.scan_name

                # Get scene statistics on final evaluation
                if epoch is None:
                    # stats = dataset_utils.get_data_statistics(curr_data)
                    stats = dataset_utils.get_data_statistics2(curr_data, outputs)
                    errors.update(stats)

                errors_list.append(errors)

                if save_predictions:
                    dataset_utils.save_cameras(outputs, conf, curr_epoch=epoch, phase=phase)
                    if conf.get_bool('dataset.calibrated'):
                        plot_utils.plot_cameras_before_and_after_ba(outputs, errors, conf, phase, scan=curr_data.scan_name, epoch=epoch, bundle_adjustment=bundle_adjustment)

    df_errors = pd.DataFrame(errors_list)
    mean_errors = df_errors.mean(numeric_only=True)
    df_errors = df_errors.append(mean_errors, ignore_index=True)
    df_errors.at[df_errors.last_valid_index(), "Scene"] = "Mean"    
    df_errors.set_index("Scene", inplace=True)    
    df_errors = df_errors.round(3)
    print(df_errors.to_string(), flush=True)    
    model.train()    

    return df_errors


def train(conf, train_data, model, phase, validation_data=None, test_data=None):
    # fitlog.set_log_dir(os.path.join(conf.get_string('dataset.results_path'), 'logs'))
    tbwriter = SummaryWriter(log_dir=os.path.join(path_to_exp(conf), 'tb'), flush_secs=60)

    num_of_epochs = conf.get_int('train.num_of_epochs')
    eval_intervals = conf.get_int('train.eval_intervals', default=500)
    validation_metric = conf.get_list('train.validation_metric', default=["our_repro"])

    # Loss functions
    loss_func2 = getattr(loss_functions, conf.get_string('loss.func'))(conf)
    loss_func1 = loss_functions.BCEWithLogitLoss()

    # Optimizer params
    lr = conf.get_float('train.lr')
    scheduler_milestone = conf.get_list('train.scheduler_milestone')
    gamma = conf.get_float('train.gamma', default=0.1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_milestone, gamma=gamma)   
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    best_validation_metric = math.inf
    best_epoch = 0
    best_model = torch.empty(0)
    converge_time = -1
    begin_time = time()

    no_ba_during_training = not conf.get_bool('ba.only_last_eval')
    min_valid_pts = conf.get_int('train.min_valid_pts', default=20)

    if not conf.get_bool("dataset.sample"):    # using preprocessed dataset
        for epoch in range(num_of_epochs):
            print(f"epoch {epoch} now ...")
            bepoch, bmetric, bmodel = sim_epoch_train(conf, tbwriter, train_data, model, loss_func1, loss_func2, optimizer, scheduler, epoch, min_valid_pts, phase, validation_data, best_validation_metric)
            if bepoch>best_epoch: 
                best_epoch=bepoch
                best_model=bmodel
            if bmetric<best_validation_metric: best_validation_metric=bmetric
            cur_model = copy.deepcopy(model)
            path = path_utils.path_to_model(conf, phase, epoch=(epoch+1)*10000)
            torch.save(cur_model.state_dict(), path)
    else:                                      # sample every time
        for epoch in range(num_of_epochs):
            print(f"epoch {epoch} now ...")
            mean_loss, mean_bceloss, mean_oriloss, mean_trloss = epoch_train(train_data, model, loss_func1, loss_func2, optimizer, scheduler, epoch, min_valid_pts)
            if epoch % 100 == 0:
                print('{} Train Loss: {}'.format(epoch, mean_loss))
                tbwriter.add_scalar("Total_Loss", mean_loss, epoch)
                tbwriter.add_scalar("BCE_Loss", mean_bceloss, epoch)
                tbwriter.add_scalar("Orient_Loss", mean_oriloss, epoch)
                tbwriter.add_scalar("Trans_Loss", mean_trloss, epoch)
            
            # mean_bceloss = epoch_train(train_data, model, loss_func1, loss_func2, optimizer, scheduler, epoch)
            # if epoch % 100 == 0:
            #     print('{} Train Loss: {}'.format(epoch, mean_bceloss))
            #     fitlog.add_loss(mean_bceloss, name="BCE_Loss", step=epoch)
            
            if epoch!=0 and (epoch % eval_intervals == 0 or epoch == num_of_epochs - 1):  # Eval current results
                if phase is Phases.TRAINING:
                    validation_errors = epoch_evaluation(validation_data, model, conf, epoch, Phases.VALIDATION, save_predictions=True,bundle_adjustment=no_ba_during_training)
                else:
                    validation_errors = epoch_evaluation(train_data, model, conf, epoch, phase, save_predictions=True,bundle_adjustment=no_ba_during_training)

                metric = validation_errors.loc[["Mean"], validation_metric].sum(axis=1).values.item()
                # fitlog.add_metric({"dev":{"Acc":metric}}, step=epoch)
                
                if metric < best_validation_metric:
                    converge_time = time()-begin_time
                    best_validation_metric = metric
                    best_epoch = epoch
                    best_model = copy.deepcopy(model)
                    print('Updated best validation metric: {} time so far: {}'.format(best_validation_metric, converge_time))
                    path = path_utils.path_to_model(conf, phase, epoch=epoch)
                    torch.save(best_model.state_dict(), path)
                
                if epoch % 10001 ==0:
                    cur_model = copy.deepcopy(model)
                    path = path_utils.path_to_model(conf, phase, epoch=epoch)
                    torch.save(cur_model.state_dict(), path)
    
    converge_time = time()-begin_time
    print(f"converge_time: {converge_time}")
    # fitlog.finish()    # finish the logging
    tbwriter.close()

    # Eval final model
    train_stat = {}
    print("Evaluate training set")
    run_ba = conf.get_bool('ba.run_ba', default=True)
    train_errors = epoch_evaluation(train_data, best_model, conf, None, phase, save_predictions=True, bundle_adjustment=run_ba)

    if phase is Phases.TRAINING:
        print("Evaluate validation set")
        validation_errors = epoch_evaluation(validation_data, best_model, conf, None, Phases.VALIDATION, save_predictions=True,bundle_adjustment=run_ba)
        print("Evaluate test set")
        test_errors = epoch_evaluation(test_data, best_model, conf, None, Phases.TEST, save_predictions=True,bundle_adjustment=run_ba)
    else:
        validation_errors = None
        test_errors = None

    # Saving the best model
    path = path_utils.path_to_model(conf, phase, epoch=None)
    torch.save(best_model.state_dict(), path)

    train_stat['Convergence time'] = converge_time
    train_stat['best_epoch'] = best_epoch
    train_stat['best_validation_metric'] = best_validation_metric
    train_stat = pd.DataFrame([train_stat])

    return train_stat, train_errors, validation_errors, test_errors
