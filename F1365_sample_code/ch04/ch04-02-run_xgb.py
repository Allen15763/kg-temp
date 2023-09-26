# ---------------------------------
# 準備資料
# ----------------------------------
import numpy as np
import pandas as pd

# 讀取資料
# 用 Pandas 的 DataFrame 存資料

train = pd.read_csv('../input/sample-data/train_preprocessed.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
test_x = pd.read_csv('../input/sample-data/test_preprocessed.csv')
print(test_x.shape, train_y.shape, train_x.shape)

# 將具有標籤的資料分為訓練資料以及驗證資料
from sklearn.model_selection import KFold

kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# from sklearn.model_selection import train_test_split
# # Split dataset into training and test sets
# tr_x, va_x, tr_y, va_y = train_test_split(train_x, train_y, test_size=0.3, random_state=42)


# -----------------------------------
# 使用 xgboost
# -----------------------------------
import xgboost as xgb
from sklearn.metrics import log_loss

# 將特徵和標籤轉換為 xgboost 的資料結構
dtrain = xgb.DMatrix(tr_x, label=tr_y) # dataframe to sparse matrix
dvalid = xgb.DMatrix(va_x, label=va_y)
dtest = xgb.DMatrix(test_x)

# 設定超參數
params = {'objective': 'binary:logistic', 'silent': 1, 'random_state': 71}
num_round = 50 # 最大迭代數

# 在 watchlist 中組合訓練資料與驗證資料
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

# 進行訓練，將驗證資料代入模型中，一面訓練模型，一面監控分數的變化
model = xgb.train(params, dtrain, num_round, evals=watchlist)

# 計算驗證資料的 logloss 分數
va_pred = model.predict(dvalid)
score = log_loss(va_y, va_pred)
print(f'logloss: {score:.4f}') # 0.2223

# 對測試資料進行預測 (輸出的預測值為資料是正例的機率，而非輸出正例或負例)
pred = model.predict(dtest)

# -----------------------------------
# 提前中止
# -----------------------------------
# 以 logloss 來進行監控，early_stopping_rounds 執行提前中止的 round 設定為 20，發現連續n次訓練結果都沒有或的更好的驗證資料分數
params = {'objective': 'binary:logistic', 'silent': 1, 'random_state': 71,
          'eval_metric': 'logloss'}
num_round = 500 # 最大迭代數
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
model = xgb.train(params, dtrain, num_round, evals=watchlist,
                  early_stopping_rounds=20)

# 以最佳決策樹的數量來進行預測；best_ntree_limit回傳最小損失的索引[73]	train-logloss:0.04405	eval-logloss:0.21397
pred = model.predict(dtest, ntree_limit=model.best_ntree_limit)
va_pred = model.predict(dvalid)
score_limit = log_loss(va_y, va_pred)
from sklearn.metrics import accuracy_score
print(f'logloss: {score_limit:.4f}\nAccuracy: {accuracy_score(va_y, np.where(va_pred > 0.5,1,0))}') # 0.2196, 0.9132


"""
此段代碼為使用 XGBoost 套件建立二元分類模型的範例，主要步驟如下：
1.  使用 pandas 套件讀取訓練資料集 `train_preprocessed.csv` 和測試資料集 `test_preprocessed.csv`。
2.  將訓練資料集中的特徵和標籤分別存儲在 `train_x` 和 `train_y` 變數中，將測試資料集存儲在 `test_x` 變數中。
3.  使用 `KFold` 函數將具有標籤的訓練資料分為訓練資料和驗證資料。
4.  使用 `xgb.DMatrix` 函數將訓練資料和驗證資料轉換為 XGBoost 的資料結構。
5.  設定 XGBoost 的超參數，例如迭代次數、目標函數、靜音模式等。
6.  透過 `train` 函數對訓練資料進行訓練，其中也將驗證資料加入 watchlist 以監控分數的變化，得到訓練好的模型。
7.  計算驗證資料的 logloss 分數。
8.  使用 `early_stopping_rounds` 參數，提前中止訓練過程，避免過度擬合，並得到最佳決策樹的數量。
9.  以最佳決策樹的數量進行預測。
最後會印出最佳決策樹的 logloss 分數。

model.get_fscore() = feature_importances_
通過計算特徵對於損失函數的貢獻度
通常是通過計算特徵對於每棵樹的增益值（Gain）或分裂值（Split）來得到的。最後特徵的增益值或分裂值平均值越高，則該特徵對於損失函數的貢獻度就越大。

特徵增益值可以通過以下步驟計算：
1.首先計算每個節點的損失函數值，並計算特徵在該節點上的梯度值和黑塞矩陣值。
2.然後根據特徵的梯度值和黑塞矩陣值，計算該特徵在該節點上的增益值。
3.最後，將所有節點上的特徵增益值相加，就可以得到該特徵對於損失函數的貢獻度。
特徵分裂值的計算方式與特徵增益值類似，只不過是使用分裂前後的損失函數值之差作為分裂的評估標準。
"""