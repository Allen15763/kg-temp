import numpy as np
import pandas as pd


# -----------------------------------
# 使用相似的自定義目標函數最佳化MAE
# -----------------------------------

# Fair 函數
def fair(preds, dtrain):
    x = preds - dtrain.get_labels()  # 求得殘差
    c = 1.0  # Fair 函數的參數
    den = abs(x) + c  # 計算斜率公式的分母
    grad = c * x / den  # 斜率
    hess = c * c / den ** 2  # 二階微分値
    return grad, hess


# Pseudo-Huber 函數
def psuedo_huber(preds, dtrain):
    d = preds - dtrain.get_labels()  # 求得殘差
    delta = 1.0  # Pseudo-Huber 函數的參數
    scale = 1 + (d / delta) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = d / scale_sqrt  # 斜率
    hess = 1 / scale / scale_sqrt  # 二階微分値
    return grad, hess
