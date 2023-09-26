import numpy as np
import pandas as pd

# -----------------------------------
# 最佳化閾値
# -----------------------------------
from sklearn.metrics import f1_score
from scipy.optimize import minimize

# 生成 10000 筆樣本資料 (機率值)
rand = np.random.RandomState(seed=71)
train_y_prob = np.linspace(0, 1.0, 10000)

# 隨機產生 10000 筆資料，每一個都和 train_y_prob 的機率值比較，小於就是負例、大於則為正例，以此做為實際值的標籤
train_y = pd.Series(rand.uniform(0.0, 1.0, train_y_prob.size) < train_y_prob)

# 從 train_y_prob 的機率值生成常態分佈的隨機數列，並控制數列範圍在 0 和 1 之間，以此做為輸出的預測機率值
train_pred_prob = np.clip(train_y_prob * np.exp(rand.standard_normal(train_y_prob.shape) * 0.3), 0.0, 1.0)

# 當初始閾值為 0.5 時，F1 為 0.722
init_threshold = 0.5
init_score = f1_score(train_y, train_pred_prob >= init_threshold)
print(init_threshold, init_score)


# 建立想要進行最佳化的目標函數
def f1_opt(x):
    return -f1_score(train_y, train_pred_prob >= x)


# 在 scipy.optimize 套件提供的 minimize() 中指定 'Nelder-Mead' 演算法來求得最佳閾值
# 在最佳閾值下計算 F1、求得 0.756
result = minimize(f1_opt, x0=np.array([0.5]), method='Nelder-Mead')
best_threshold = result['x'].item()
best_score = f1_score(train_y, train_pred_prob >= best_threshold)
print(best_threshold, best_score)
