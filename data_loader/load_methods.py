import os
import numpy as np
import pandas as pd
from scipy.io import loadmat


def CWRU(item_path):
    axis = ["_DE_time", "_FE_time", "_BA_time"]
    datanumber = os.path.basename(item_path).split(".")[0]
    if eval(datanumber) < 100:
        realaxis = "X0" + datanumber + axis[0]
    else:
        realaxis = "X" + datanumber + axis[0]
    signal = loadmat(item_path)[realaxis]

    return signal


def MFPT(item_path):
    f = item_path.split("/")[-2]
    if f == 'normal':
        signal = (loadmat(item_path)["bearing"][0][0][1])
    else:
        signal = (loadmat(item_path)["bearing"][0][0][2])

    return signal


def PU(item_path):
    name = os.path.basename(item_path).split(".")[0]
    fl = loadmat(item_path)[name]
    signal = fl[0][0][2][0][6][2]  #Take out the data
    signal = signal.reshape(-1,1)

    return signal


def XJTU(item_path):
    fl = pd.read_csv(item_path)
    signal = fl["Horizontal_vibration_signals"]
    signal = signal.values.reshape(-1,1)

    return signal


def IMS(item_path):
    channel = {'normal': 0,
               'inner': 4,
               'outer': 0,
               'ball': 6}
    f = item_path.split("/")[-2]
    signal = np.loadtxt(item_path)[:, channel[f]]

    return signal


def JNU(item_path):
    fl = pd.read_csv(item_path)
    signal = fl.values
    
    return signal

def GearFace(item_path):
    try:
        signal = pd.read_pickle(item_path)  # 假设 signal.shape = (970, 128)
        print(f"Loaded {item_path}, shape: {signal.shape}")
        return signal  # 返回完整数组，后续由调用方拆分
    except Exception as e:
        print(f"Failed to load {item_path}: {e}")
        return None

def GT(item_path):
    try:
        signal = pd.read_pickle(item_path)  # 假设 signal.shape = (970, 128)
        print(f"Loaded {item_path}, shape: {signal.shape}")
        return signal  # 返回完整数组，后续由调用方拆分
    except Exception as e:
        print(f"Failed to load {item_path}: {e}")
        return None

def GearFace2(item_path):
    try:
        signal = pd.read_pickle(item_path)  # 假设 signal.shape = (970, 128)
        print(f"Loaded {item_path}, shape: {signal.shape}")
        return signal  # 返回完整数组，后续由调用方拆分
    except Exception as e:
        print(f"Failed to load {item_path}: {e}")
        return None

def GearFace3(item_path):
    try:
        signal = pd.read_pickle(item_path)  # 假设 signal.shape = (970, 128)
        print(f"Loaded {item_path}, shape: {signal.shape}")
        return signal  # 返回完整数组，后续由调用方拆分
    except Exception as e:
        print(f"Failed to load {item_path}: {e}")
        return None
def GearFace20(item_path):
    try:
        signal = pd.read_pickle(item_path)  # 假设 signal.shape = (970, 128)
        print(f"Loaded {item_path}, shape: {signal.shape}")
        return signal  # 返回完整数组，后续由调用方拆分
    except Exception as e:
        print(f"Failed to load {item_path}: {e}")
        return None
