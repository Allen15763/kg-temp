# ---------------------------------
# 準備資料
# ----------------------------------
import numpy as np
import pandas as pd

# 讀取資料
# 用 Pandas 的 DataFrame 存資料

train = pd.read_csv('../input/sample-data/train_preprocessed_onehot.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
test_x = pd.read_csv('../input/sample-data/test_preprocessed_onehot.csv')

# ---------------------------------
# numpy 的 argsort 函式
# ---------------------------------
# 透過使用 argsort，在 index 中排列值的大小順序
ary = np.array([10, 20, 30, 0])
idx = ary.argsort()
print(idx)  # 降冪的索引 - [3 0 1 2]
print(idx[::-1])  # 升冪的索引 - [2 1 0 3]

print(ary[idx[::-1][:3]])  # 由大到小的前三個輸出 - [30, 20, 10]

# ---------------------------------
# 計算相關係數
# ---------------------------------
import scipy.stats as st

# 相關係數
corrs = []
for c in train_x.columns:
    corr = np.corrcoef(train_x[c], train_y)[0, 1]
    corrs.append(corr)
corrs = np.array(corrs)

# 斯皮爾曼等級相關係數
corrs_sp = []
for c in train_x.columns:
    corr_sp = st.spearmanr(train_x[c], train_y).correlation
    corrs_sp.append(corr_sp)
corrs_sp = np.array(corrs_sp)

# 輸出重要性高的前 5 順位
# 使用 np.argsort 來取得依數值大小排序的索引
idx = np.argsort(np.abs(corrs))[::-1]
top_cols, top_importances = train_x.columns.values[idx][:5], corrs[idx][:5]
print(top_cols, top_importances)

idx2 = np.argsort(np.abs(corrs_sp))[::-1]
top_cols2, top_importances2 = train_x.columns.values[idx][:5], corrs_sp[idx][:5]
print(top_cols2, top_importances2)

# ---------------------------------
# 計算卡方統計量
# ---------------------------------
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler

# 卡方統計量
x = MinMaxScaler().fit_transform(train_x)
c2, _ = chi2(x, train_y)

# 輸出前 5 重要 (性) 的特徵
idx = np.argsort(c2)[::-1]
top_cols, top_importances = train_x.columns.values[idx][:5], corrs[idx][:5]
print(top_cols, top_importances)

# ---------------------------------
# 計算相互資訊量
# ---------------------------------
from sklearn.feature_selection import mutual_info_classif

# 相互資訊量
mi = mutual_info_classif(train_x, train_y)

# 輸出前 5 重要(性) 的特徵
idx = np.argsort(mi)[::-1]
top_cols, top_importances = train_x.columns.values[idx][:5], corrs[idx][:5]
print(top_cols, top_importances)
