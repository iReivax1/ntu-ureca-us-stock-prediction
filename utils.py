import sys
import numpy as np


def write_and_restart_line(text_to_write):
    sys.stdout.write(text_to_write)
    sys.stdout.write('\b')
    sys.stdout.flush()


def normalize(data, axis=[0], norm=0):
    '''

    :param data: numpy array
    :param axis: a list, and the first value < the second value, and so on.
    :return:
    '''

    data_x = data
    if norm == 0:
        data_x -= np.mean(data_x, axis=tuple(axis))
        sd = np.std(data_x, axis=tuple(axis))
        if type(sd).__module__ == np.__name__:
            sd[sd == 0] = 1
            data_x /= sd
        elif sd != 0:
            # data_x为一维array
            data_x /= sd
    elif norm == 1:
        mx = np.max(data_x, axis=tuple(axis))
        if type(mx).__module__ == np.__name__:
            mx[mx == 0] = 1
            data_x /= mx
        elif mx != 0:
            # data_x为一维array
            data_x /= mx
    elif norm == 2:
        data_x = np.tanh(data_x)
    elif norm == 3:
        data_x = np.log(1 + data_x)
    elif norm == 4:
        data_x = np.log10(1 + data_x)
    elif norm == 5:
        data_x = np.log2(1 + data_x)
    return data_x


def calculate_error(y, y_hat):
    '''
    :param y: 观测值
    :param y_hat: 预测值
    :return: 返回计算的三种error（%）
    '''
    mae = np.mean(np.abs((y - y_hat)))
    mape = np.mean(np.abs((y - y_hat) / y))
    rmse = np.sqrt(np.mean(np.square(y - y_hat)))
    return mae*100, mape*100, rmse*100
