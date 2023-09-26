# ---------------------------------
# 準備資料
# ----------------------------------
import numpy as np
import pandas as pd

# 建立隨機資料
rand = np.random.RandomState(71)
train_x = pd.DataFrame(rand.uniform(0.0, 1.0, (10000, 2)), columns=['model1', 'model2'])
adv_train = pd.Series(rand.uniform(0.0, 1.0, 10000))
w = np.array([0.3, 0.7]).reshape(1, -1)
train_y = pd.Series((train_x.values * w).sum(axis=1) > 0.5)

# ---------------------------------
# adversarial stochastic blending
# ----------------------------------
# 使用 adversarial validation 求得模型預測值的加權平均權重
# train_x: 各模型預測機率的預測值 (實際上因為評價指標是 AUC，可以使用機率大小的排序結果)
# train_y: 標籤
# adv_train: 表示訓練資料與測試資料的相似程度的機率值

from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

n_sampling = 50  # 抽樣次數
frac_sampling = 0.5  # 抽樣中從訓練資料取出的比例


def score(x, data_x, data_y):
    # 評價指標為 AUC
    y_prob = data_x['model1'] * x + data_x['model2'] * (1 - x)
    return -roc_auc_score(data_y, y_prob)


# 重覆在抽樣中求得加權平均的權重
results = []
for i in range(n_sampling):
    # 進行抽樣
    seed = i
    idx = pd.Series(np.arange(len(train_y))).sample(frac=frac_sampling, replace=False,
                                                    random_state=seed, weights=adv_train)
    x_sample = train_x.iloc[idx]
    y_sample = train_y.iloc[idx]

    # 計算抽樣資料的最佳化加權平均權重值
    # 為了使其具有制約式，選擇使用 COBYLA 演算法
    init_x = np.array(0.5)
    constraints = (
        {'type': 'ineq', 'fun': lambda x: x},
        {'type': 'ineq', 'fun': lambda x: 1.0 - x},
    )
    result = minimize(score, x0=init_x,
                      args=(x_sample, y_sample),
                      constraints=constraints,
                      method='COBYLA')
    results.append((result.x, 1.0 - result.x))

# model1, model2 加權平均的權重
results = np.array(results)
w_model1, w_model2 = results.mean(axis=0)
