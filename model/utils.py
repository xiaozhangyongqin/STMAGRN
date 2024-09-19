import pandas as pd
import torch
import pickle
import numpy as np
import json

class DataLoader(object):
    def __init__(self, xs, ys, bs, pad_with_last_sample=True, shuffle=False):
        """
        :param xs:
        :param ys:
        :param bs: batch_size
        :param pad_with_last_sample:
        """
        self.bs = bs
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (bs - (len(xs) % bs)) % bs
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.bs)
        if shuffle:
            permutation = np.random.permutation(self.size)
            xs, ys = xs[permutation], ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.bs * self.current_ind
                end_ind = min(self.size, self.bs * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()

class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true) * mask
    loss[loss != loss] = 0
    return loss.mean()
# def masked_mae_loss(preds, labels, null_val=0.0):
#     if np.isnan(null_val):
#         mask = ~torch.isnan(labels)
#     else:
#         mask = labels != null_val
#     mask = mask.float()
#     mask /= torch.mean((mask))
#     mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
#     loss = torch.abs(preds - labels)
#     loss = loss * mask
#     loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
#     return torch.mean(loss)
    
def masked_mape_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(torch.div(y_true - y_pred, y_true + 1e-3))
    loss1 = loss.cpu().numpy()
    condition = loss1 > 40.0
    loss1 = np.where(condition, 0, loss1)
    loss = torch.from_numpy(loss1)
    loss = loss.to('cuda:0') * mask
    loss[loss != loss] = 0
    return loss.mean()

def masked_rmse_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.pow(y_pred - y_true, 2) * mask
    loss[loss != loss] = 0
    return torch.sqrt(loss.mean())

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def print_log(*values, log=None, end='\n'):
    print(*values, end=end)
    if log:
        if isinstance(log, str):
            log = open(log, 'a')
        print(*values, file=log, end=end)
        log.flush()

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return f"Shape: {obj.shape}"
        elif isinstance(obj, torch.device):
            return str(obj)
        else:
            return super(CustomJSONEncoder, self).default(obj)
def quadruplet_loss(query, pos, neg1, neg2):
    loss1 = torch.pow(query - pos, 2) - torch.pow(query - neg1, 2) + 1
    zero = torch.zeros_like(loss1)
    loss1 = torch.where(loss1 < 0, zero, loss1)
    loss2 = torch.pow(query - pos, 2) - torch.pow(neg1 - neg2, 2) + 0.5
    loss2 = torch.where(loss2 < 0, zero, loss2)
    loss = loss1 + loss2
    return loss.mean()

def steps_output(y_true, y_pred, l_3, m_3, r_3, l_6, m_6, r_6, l_12, m_12, r_12):
    l_3.append(masked_mae_loss(y_pred[2:3], y_true[2:3]).item())
    m_3.append(masked_mape_loss(y_pred[2:3], y_true[2:3]).item())
    r_3.append(masked_rmse_loss(y_pred[2:3], y_true[2:3]).item())
    l_6.append(masked_mae_loss(y_pred[5:6], y_true[5:6]).item())
    m_6.append(masked_mape_loss(y_pred[5:6], y_true[5:6]).item())
    r_6.append(masked_rmse_loss(y_pred[5:6], y_true[5:6]).item())
    l_12.append(masked_mae_loss(y_pred[11:12], y_true[11:12]).item())
    m_12.append(masked_mape_loss(y_pred[11:12], y_true[11:12]).item())
    r_12.append(masked_rmse_loss(y_pred[11:12], y_true[11:12]).item())
    return l_3, m_3, r_3, l_6, m_6, r_6, l_12, m_12, r_12


