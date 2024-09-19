import sys
import os
import shutil
import numpy as np
import pandas as pd
import random
import time
import torch
import torch.nn as nn

import datetime
import yaml
import json
import argparse
import logging
from utils import StandardScaler, DataLoader, masked_mae_loss, masked_mape_loss, \
    masked_rmse_loss, print_log, CustomJSONEncoder, quadruplet_loss, steps_output
from STMAGRN import STMAGRN
import time

def prepare_x_y(x, y, cfg):
    """
    :param x: shape (batch_size, seq_len, num_sensor, input_dim)
    :param y: shape (batch_size, horizon, num_sensor, input_dim)
    :return1: x shape (seq_len, batch_size, num_sensor, input_dim)
              y shape (horizon, batch_size, num_sensor, input_dim)
    :return2: x: shape (seq_len, batch_size, num_sensor * input_dim)
              y: shape (horizon, batch_size, num_sensor * output_dim)
    :return : x , y ,y_cov 时间信息
    """
    input_dim = cfg['model_args']['input_dim']
    output_dim = cfg['model_args']['output_dim']
    x_input_dim = x[..., :input_dim]
    x_time = x[..., input_dim:]
    y_input_dim = y[..., :output_dim]
    y_time = y[..., output_dim:]
    x_input_dim = torch.from_numpy(x_input_dim).float()
    x_time = torch.from_numpy(x_time).float()
    y_input_dim = torch.from_numpy(y_input_dim).float()
    y_time = torch.from_numpy(y_time).float()
    return x_input_dim.to(DEVICE), y_input_dim.to(DEVICE), x_time.to(DEVICE), y_time.to(DEVICE)


def print_model(model):
    param_count = 0
    logger.info('Trainable parameter list:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print_log(name, param.shape, param.numel(), log=log)
            param_count += param.numel()
    print_log(f'In total: {param_count} trainable parameters.', log=log)
    return


def load_dataset(args, cfg, log=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(f'../{args.dataset}', category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
        data['y_' + category][..., 0] = scaler.transform(data['y_' + category][..., 0])

    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], cfg['batch_size'], shuffle=True)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], cfg['batch_size'], shuffle=False)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], cfg['batch_size'], shuffle=False)

    print_log('train xs.shape, ys.shape', data['x_train'].shape, data['y_train'].shape, log=log)
    print_log('val xs.shape, ys.shape', data['x_val'].shape, data['y_val'].shape, log=log)
    print_log('test xs.shape, ys.shape', data['x_test'].shape, data['y_test'].shape, log=log)

    return data, scaler


logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def evaluate(model, data, dataset, mode=None):
    with torch.no_grad():
        model = model.eval()
        data_iter = data[f'{mode}_loader'].get_iterator()
        losses = []
        ys_true, ys_pred = [], []
        start_time = time.time()
        for x, y in data_iter:
            x, y, x_cov, ycov = prepare_x_y(x, y, cfg)
            output = model(x, x_cov, ycov)
            y_pred = scaler.inverse_transform(output)
            y_true = scaler.inverse_transform(y)
            
            loss = masked_mae_loss(y_pred, y_true)  # masked_mae_loss(y_pred, y_true)
            losses.append(loss.item())
            ys_true.append(y_true)
            ys_pred.append(y_pred)
        mean_loss = np.mean(losses)
        y_size = data[f'y_{mode}'].shape[0]
        ys_true, ys_pred = torch.cat(ys_true, dim=0)[:y_size], torch.cat(ys_pred, dim=0)[:y_size]
        true = ys_true.detach().cpu().numpy()
        pred = ys_pred.detach().cpu().numpy()
        np.save(f'./temp/{dataset}_true.npy',true)
        np.save(f'/temp/{dataset}_pred.npy',pred)
        if mode == 'test':
            ys_true, ys_pred = ys_true.permute(1, 0, 2, 3), ys_pred.permute(1, 0, 2, 3)
            results = {
                'overall': {
                    'mae': masked_mae_loss(ys_pred, ys_true).item(),
                    'mape': masked_mape_loss(ys_pred, ys_true).item(),
                    'rmse': masked_rmse_loss(ys_pred, ys_true).item()
                }
            }
            # 计算每个步长的MAE、MAPE、RMSE
            for i in range(1, 13):
                horizon = 'Horizon {}mins'.format(i * 5)
                results[i] = {
                    'mae': masked_mae_loss(ys_pred[i-1:i], ys_true[i-1:i]).item(),
                    'mape': masked_mape_loss(ys_pred[i-1:i], ys_true[i-1:i]).item(),
                    'rmse': masked_rmse_loss(ys_pred[i-1:i], ys_true[i-1:i]).item()
                }
                print_log('{}: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(horizon,
                                                                               results[i]['mae'], results[i]['mape'], results[i]['rmse']), log=log)
            ys_true, ys_pred = ys_true.permute(1, 0, 2, 3), ys_pred.permute(1, 0, 2, 3)
            with open(f'/temp/{dataset}_results.json', 'w') as f:
                json.dump(results, f, indent=4)

        return mean_loss, ys_true, ys_pred

def traintest_model(model, optimizer, lr_scheduler, data, scaler, cfg, model_path, dataset, log=None):
    model = model.to(DEVICE)
    print_log('=' * 35 + 'Best model performance' + '=' * 35, log=log)
    print_log(log=log)
    model = model
    model.load_state_dict(torch.load(model_path))
    test_loss, _, _ = evaluate(model, data, dataset, 'test')


def seed_torch(seed=0):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.benchmark = False

        torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['METRLA', 'PEMSBAY', 'PEMS08', 'PEMS04', 'PEMS03', 'PEMS07'],
                        default='METRLA')
    parser.add_argument('--g', type=int, default=0)
    parser.add_argument('--path', type=str, default="../saved_models/PEMS08_STMAGRN_2024-09-04-07-25-12/STMAGRN-2024-09-04-07-25-12.pt")
    args = parser.parse_args()

    GPU_ID = args.g
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_ID}"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = args.dataset
    dataset = dataset.upper()

    model_name = STMAGRN.__name__
    with open(f'{model_name}.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = cfg[dataset]
    # -------------------------- seed ------------------------- #
    seed_torch(cfg['seed'])
    # np.random.seed(cfg['seed'])
    # torch.manual_seed(cfg['seed'])
    # if torch.cuda.is_available(): torch.cuda.manual_seed(cfg['seed'])
    # -------------------------- load model ------------------------- #
    model = STMAGRN(**cfg['model_args'])
    # ----------------------------make log file----------------------------- #
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_path = f'../logs/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = os.path.join(log_path, f'{model_name}-{args.dataset}-{now}.log')
    log = open(log, 'a')
    log.seek(0)
    log.truncate()

    # ----------------------------load dataset ---------------------------- #
    print_log(dataset, log=log)
    data, scaler = load_dataset(args, cfg, log=log)
    print_log(log=log)
    
    model_path = args.path

    # ----------------------------set model opt, scheduler ---------------------------- #
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], eps=cfg['epsilon'],
                                  weight_decay=cfg['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'],
                                                        gamma=cfg['lr_decay_ratio'])

    # ----------------------------print model structure---------------------------- #
    print_log("-----------", model_name, "-----------", log=log)
    print_log(
        json.dumps(cfg, ensure_ascii=False, indent=4, cls=CustomJSONEncoder), log=log
    )
    print_log("----------------------------------------\n", log=log)
    print_model(model)
    # ----------------------------train and test model ---------------------------- #
    traintest_model(model, optimizer, lr_scheduler, data, scaler, cfg, model_path, dataset, log=log)



