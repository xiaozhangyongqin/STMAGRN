from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=True, scaler=None
):
    """
    Generate samples from
    :param df: data
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]
    if add_time_in_day:
        # df.index.values为数据的日期每隔五分钟采样一次，df.index.values.astype("datetime64[D]")保留日期的天数
        # np.timedelta64(1, "D")表示一天的时间
        # time_ind 索引值中的时间在一天内的偏移量占一天的比例。例如，中午12点的偏移量为0.5，因为它是一天中的一半。
        # 5 / (24 * 60) = 0.0034722222222222222
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        # day_in_week = np.zeros(shape=(num_samples, num_nodes, 1)) # (17856, 170, 1)
        #
        # day_in_week[np.arange(num_samples), :, -1] = df.index.dayofweek
        # data_list.append(day_in_week)
        shape = (num_samples, num_nodes, 1)
        period = 288
        pattern_len = 7

        arr = np.empty(shape, dtype=float)

        for i in range(shape[0]):
            period_num = i // period % pattern_len
            arr[i, :, :] = period_num
        data_list.append(arr)

    data = np.concatenate(data_list, axis=-1)
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y = [], []
    # t is the index of the last observation.
    # 添加窗口12 预测12 一个小时数据来预测下一个小时数据
    min_t = abs(min(x_offsets))  # 11
    max_t = abs(num_samples - abs(max(y_offsets)))  # 34272 - 12 = 34270 # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)  # 34260 - 12 = 34248
    y = np.stack(y, axis=0)
    return x, y


def generate_train_val_test(args):
    #  Metrla数据集 34272 207
    df = pd.read_hdf(args.traffic_df_filename)
    # 0 is the latest observed sample.
    x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        np.concatenate((np.arange(-11, 1, 1),))
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 13, 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=True
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    # num_train = round(num_samples * 0.7) # PEMS-BAY METR-LA
    num_train = round(num_samples * 0.6)
    num_val = num_samples - num_test - num_train

    # train
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        # locasl函数将x_train值赋值给_x，y_train值赋值给_y
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


def main(args):
    print("Generating training data")
    generate_train_val_test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['METRLA', 'PEMSBAY', 'PEMS04', 'PEMS07', 'PEMS08', 'PEMS03'], default='PEMS04', help='which dataset to run')
    parser.add_argument("--output_dir", type=str, default="METRLA/", help="Output directory.")
    parser.add_argument("--traffic_df_filename", type=str, default="METRLA/metr-la.h5", help="Raw traffic readings.")
    args = parser.parse_args()
    args.output_dir = f'{args.dataset}/'
    if args.dataset == 'METRLA':
        args.traffic_df_filename = f'{args.dataset}/metr-la.h5'
    elif args.dataset == 'PEMSBAY':
        args.traffic_df_filename = f'{args.dataset}/pems-bay.h5'
    elif args.dataset == 'PEMS04':
        args.traffic_df_filename = f'{args.dataset}/pemsd04.h5'
    elif args.dataset == 'PEMS07':
        args.traffic_df_filename = f'{args.dataset}/pemsd07.h5'
    elif args.dataset == 'PEMS08':
        args.traffic_df_filename = f'{args.dataset}/pemsd08.h5'
    elif args.dataset == 'PEMS03':
        args.traffic_df_filename = f'{args.dataset}/pemsd03.h5'
    main(args)

