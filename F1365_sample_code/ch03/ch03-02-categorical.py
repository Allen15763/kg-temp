# ---------------------------------
# 準備資料
# ----------------------------------
import numpy as np
import pandas as pd

# 讀取資料
# 用Pandas的DataFrame存資料

train = pd.read_csv('../input/sample-data/train.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
test_x = pd.read_csv('../input/sample-data/test.csv')

# 備份資料，以便之後再利用
train_x_saved = train_x.copy()
test_x_saved = test_x.copy()

print(f'Raw {train_x.shape}\n {train_x.head()}')
# 讀取資料的函數
def load_data():
    train_x, test_x = train_x_saved.copy(), test_x_saved.copy()
    return train_x, test_x


# 將需要轉換的變數放在list
cat_cols = ['sex', 'product', 'medical_info_b2', 'medical_info_b3']

# -----------------------------------
# one-hot encoding
# -----------------------------------
# 讀取資料
train_x, test_x = load_data()
# -----------------------------------

# 整合訓練與測試資料，執行 One-hot encoding
all_x = pd.concat([train_x, test_x])
all_x = pd.get_dummies(all_x, columns=cat_cols)

# 重新分割訓練、測試資料
train_x = all_x.iloc[:train_x.shape[0], :].reset_index(drop=True)
test_x = all_x.iloc[train_x.shape[0]:, :].reset_index(drop=True)

# -----------------------------------
# 讀取資料
train_x, test_x = load_data()
# -----------------------------------
from sklearn.preprocessing import OneHotEncoder

# 建立 OneHotEncoder 物件
ohe = OneHotEncoder(sparse=False, categories='auto')
ohe.fit(train_x[cat_cols])

# 建立虛擬變數的欄位名稱
columns = []
for i, c in enumerate(cat_cols):
    columns += [f'{c}_{v}' for v in ohe.categories_[i]]

# 將建立好的虛擬變數轉換成 dataframe
dummy_vals_train = pd.DataFrame(ohe.transform(train_x[cat_cols]), columns=columns)
dummy_vals_test = pd.DataFrame(ohe.transform(test_x[cat_cols]), columns=columns)

# 將轉換後的 dataframe 跟其他特徵結合
train_x = pd.concat([train_x.drop(cat_cols, axis=1), dummy_vals_train], axis=1)
test_x = pd.concat([test_x.drop(cat_cols, axis=1), dummy_vals_test], axis=1)
print(f'OneHot {train_x.shape}\n{train_x.head()}')
# -----------------------------------
# label encoding
# -----------------------------------
# 讀取資料
train_x, test_x = load_data()
# -----------------------------------
from sklearn.preprocessing import LabelEncoder

# 將類別變數進行 lopp (迴圈 並進行 Label encoding
for c in cat_cols:
    # 建立 LabelEncoder 物件
    le = LabelEncoder()
    # 以訓練資料來對 LabelEncoder 物件進行定義
    le.fit(train_x[c])
    # 以 LabelEncoder 物件對訓練資料進行 Label encoding
    train_x[c] = le.transform(train_x[c])
    # 以 LabelEncoder 物件對測試資料進行 Label encoding
    test_x[c] = le.transform(test_x[c])

# -----------------------------------
# feature hashing
# -----------------------------------
# 讀取資料
train_x, test_x = load_data()
# -----------------------------------
from sklearn.feature_extraction import FeatureHasher

# 對每個類別變數進行 feature hasing
for c in cat_cols:
    # FeatureHasher 的用法和其他的 encoder 有些不同；No fit

    fh = FeatureHasher(n_features=5, input_type='string') # n_features:將原始特徵映射到的新特徵的維度數量
    # 將變數轉換成文字列後則可使用 FeatureHasher；Return sparse metrix
    """
    使用 FeatureHasher 的 transform() 方法將原始資料集的類別變數進行特徵哈希化，轉換成稀疏矩陣 (sparse matrix) 的格式。
    其中 train_x[[c]] 表示選擇 train_x 資料集中的 c 列作為特徵，astype(str) 則表示將其轉換為字符串類型，
    最後使用 .values 方法將其轉換為 numpy.ndarray 的格式。
    """
    hash_train = fh.transform(train_x[[c]].astype(str).values)
    hash_test = fh.transform(test_x[[c]].astype(str).values)
    # print(type(hash_train), hash_train.shape) # <class 'scipy.sparse.csr.csr_matrix'> (10000, 5) # 5-->n_features

    # 轉換成 dataframe
    hash_train = pd.DataFrame(hash_train.todense(), columns=[f'{c}_{i}' for i in range(5)])
    hash_test = pd.DataFrame(hash_test.todense(), columns=[f'{c}_{i}' for i in range(5)])
    # print(type(hash_train), hash_train.shape) # <class 'pandas.core.frame.DataFrame'> (10000, 5)
    # print(hash_train.head())
    """
           product_0  product_1  product_2  product_3  product_4
    0        0.0        0.0       -1.0        0.0        0.0
    1        0.0        0.0       -1.0        0.0        0.0
    """

    # 與原資料的 dataframe 進行水平結合；原25+5*4=45
    # print(f'Before concat: {train_x.shape}')
    train_x = pd.concat([train_x, hash_train], axis=1)
    test_x = pd.concat([test_x, hash_test], axis=1)
    # print(f'After concat: {train_x.shape}')
# 刪除原資料的類別變數
train_x.drop(cat_cols, axis=1, inplace=True) # (10000, 41)
test_x.drop(cat_cols, axis=1, inplace=True)
print(f'feature {train_x.shape}\n{train_x.head()}')
# -----------------------------------
# frequency encoding
# -----------------------------------
# 讀取資料
train_x, test_x = load_data()
# -----------------------------------
# 對每個類別變數進行 Frequency encoding
for c in cat_cols:
    freq = train_x[c].value_counts()
    # 將變數代換為類別變數的出現次數
    train_x[c] = train_x[c].map(freq)
    test_x[c] = test_x[c].map(freq)

# -----------------------------------
# target encoding
# -----------------------------------
# 讀取資料
train_x, test_x = load_data()
# -----------------------------------
from sklearn.model_selection import KFold

# 對每個類別變數進行 Target encoding
for c in cat_cols:
    # 以訓練資料來計算各個類別的標籤平均
    data_tmp = pd.DataFrame({c: train_x[c], 'target': train_y})
    """
    用每個變量的共同變量(標籤)取平均
         index:  sex     target
            0    Male       0
            1  Female       0
            2    Male       1
            3    Male       0
            4  Female       1
    """
    target_mean = data_tmp.groupby(c)['target'].mean()
    """
    Female    0.210963
    Male      0.179977
    """
    # 轉換測試資料的類別
    test_x[c] = test_x[c].map(target_mean)

    # 設定訓練資料轉換後格式
    """
    創建一個 NaN 數組，並指定其大小為 train_x 數據集的行數。
    這個數組用於暫時保存每個交叉驗證折疊後的訓練數據 Target Encoding 轉換後的結果，以便後續的合併操作。
    最終的 Target Encoding 轉換結果會通過該數組來暫存，並且再將其寫回到 train_x 數據集的對應類別特徵列中。
    """
    tmp = np.repeat(np.nan, train_x.shape[0])

    # 分割訓練資料
    kf = KFold(n_splits=4, shuffle=True, random_state=72)
    for idx_1, idx_2 in kf.split(train_x): # idx_1, idx_2: tr_idx, va_idx
        # 以 out-of-fold 方法來計算各類別變數的標籤平均值
        target_mean = data_tmp.iloc[idx_1].groupby(c)['target'].mean()
        # 在暫訂格式中置入轉換後的值
        tmp[idx_2] = train_x[c].iloc[idx_2].map(target_mean)

    # 以轉換後的資料代換原本的變數
    train_x[c] = tmp
print(f'Target {train_x.shape}\n{train_x.head()}')

# -----------------------------------
# 交叉驗證下的 Target encoding
# -----------------------------------
# 讀取資料
train_x, test_x = load_data()
# -----------------------------------
from sklearn.model_selection import KFold

# 對交叉驗證的每個 fold 重新執行 Target encoding
"""
在每次 fold 訓練前重新進行 Target Encoding。將訓練資料分成 4 個 fold，
每次使用其中 3 個 fold 來計算類別變數的平均值，再將結果轉換至剩餘的 fold 中，然後進行模型訓練。
可以減少過度擬合的情況，但是需要多次執行模型訓練，因此較為耗時。
"""
kf = KFold(n_splits=4, shuffle=True, random_state=71)
for i, (tr_idx, va_idx) in enumerate(kf.split(train_x)):

    # 將驗證資料從訓練資料中分離
    tr_x, va_x = train_x.iloc[tr_idx].copy(), train_x.iloc[va_idx].copy()
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

    # 進行每個類別變數的 Target encoding
    for c in cat_cols:
        # 計算所有訓練資料中各個項目的標籤平均值
        data_tmp = pd.DataFrame({c: tr_x[c], 'target': tr_y})
        target_mean = data_tmp.groupby(c)['target'].mean()
        # 代換驗證資料的類別
        va_x.loc[:, c] = va_x[c].map(target_mean)

        # 設定訓練資料轉換後的排列方式
        tmp = np.repeat(np.nan, tr_x.shape[0])
        kf_encoding = KFold(n_splits=4, shuffle=True, random_state=72)
        for idx_1, idx_2 in kf_encoding.split(tr_x):
            # 以 out-of-fold 方法來計算各類別變數的標籤平均值
            target_mean = data_tmp.iloc[idx_1].groupby(c)['target'].mean()
            # 將轉換後的值傳回至暫訂排列中
            tmp[idx_2] = tr_x[c].iloc[idx_2].map(target_mean)

        tr_x.loc[:, c] = tmp

    # 若有需要可以保存encode的特徵，以便隨時讀取。(指把內層的tr_x.loc[:, c]存成另一個tmp_埔回該層的tr_x)
    # print(f'Cross Target {tr_x.shape}\n{tr_x.head()}') # 4折
# -----------------------------------
# 交叉驗證的 fold 及 Target encoding 的 fold 合併
# -----------------------------------
# 讀取資料
train_x, test_x = load_data()
# -----------------------------------
from sklearn.model_selection import KFold

# 定義交叉驗證的fold
kf = KFold(n_splits=4, shuffle=True, random_state=71)

# 進行每個類別變數的 Target encoding 
for c in cat_cols:

    # 加上 target 
    data_tmp = pd.DataFrame({c: train_x[c], 'target': train_y})
    # 設定轉換後置入數值的格式
    tmp = np.repeat(np.nan, train_x.shape[0])

    # 將驗證資料從訓練資料中分離
    for i, (tr_idx, va_idx) in enumerate(kf.split(train_x)):
        # 計算訓練資料中各類別的變數平均
        target_mean = data_tmp.iloc[tr_idx].groupby(c)['target'].mean()
        # 將轉換後的驗證資料數值置入暫訂格式中
        tmp[va_idx] = train_x[c].iloc[va_idx].map(target_mean)

    # 以轉換後的資料代換原資料
    train_x[c] = tmp
print(f'3 Target {train_x.shape}\n{train_x.head()}')