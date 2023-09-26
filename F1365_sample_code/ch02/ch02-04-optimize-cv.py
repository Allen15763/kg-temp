import numpy as np
import pandas as pd

# -----------------------------------
# 使用 out-of-fold 來最佳化閾值
# -----------------------------------
from scipy.optimize import minimize
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

# 產生樣本資料
rand = np.random.RandomState(seed=71)
train_y_prob = np.linspace(0, 1.0, 10000)

# 實際值和預測值分別為 train_y, train_pred_prob
train_y = pd.Series(rand.uniform(0.0, 1.0, train_y_prob.size) < train_y_prob)
train_pred_prob = np.clip(train_y_prob * np.exp(rand.standard_normal(train_y_prob.shape) * 0.3), 0.0, 1.0)

# 在交叉驗證範圍內求得閾值
thresholds = []
scores_tr = []
scores_va = []

kf = KFold(n_splits=4, random_state=71, shuffle=True)
for i, (tr_idx, va_idx) in enumerate(kf.split(train_pred_prob)):
    tr_pred_prob, va_pred_prob = train_pred_prob[tr_idx], train_pred_prob[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

    # 設定最佳化目標函數
    def f1_opt(x):
        return -f1_score(tr_y, tr_pred_prob >= x)

    # 在訓練資料中進行閾值的最佳化，使用驗證資料來進行評價
    result = minimize(f1_opt, x0=np.array([0.5]), method='Nelder-Mead')
    threshold = result['x'].item()
    score_tr = f1_score(tr_y, tr_pred_prob >= threshold)
    score_va = f1_score(va_y, va_pred_prob >= threshold)
    print(threshold, score_tr, score_va)

    thresholds.append(threshold)
    scores_tr.append(score_tr)
    scores_va.append(score_va)

# 將每個 fold 的最佳化閾值平均，再使用於測試資料
threshold_test = np.mean(thresholds)
print(threshold_test)

test_y_prob = np.linspace(0, 1.0, 10000)
test_pred_prob = np.clip(test_y_prob * np.exp(rand.standard_normal(test_y_prob.shape) * 0.3), 0.0, 1.0)
test_y = pd.Series(rand.uniform(0.0, 1.0, test_y_prob.size) < test_y_prob)

test_pred = test_pred_prob >= threshold_test
# test_y也是一個numpy.ndarray包含True/False的二分類標籤。也就是把實際標籤與預測標籤放進F1做比較
test_score = f1_score(test_y, test_pred)
print(test_score)
