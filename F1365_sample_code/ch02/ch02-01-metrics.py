import numpy as np
import pandas as pd

# -----------------------------------
# 迴歸
# -----------------------------------
# rmse

from sklearn.metrics import mean_squared_error

# y_true為真實值、y_pred為預測值
y_true = [1.0, 1.5, 2.0, 1.2, 1.8]
y_pred = [0.8, 1.5, 1.8, 1.3, 3.0]

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(rmse)
# 0.5532

# -----------------------------------
# 二元分類
# -----------------------------------
# 混淆矩陣

from sklearn.metrics import confusion_matrix

# 以 0,1 來表示二元分類的負例與正例
y_true = [1, 0, 1, 1, 0, 1, 1, 0]
y_pred = [0, 0, 1, 1, 0, 0, 1, 1]

tp = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 1))
tn = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 0))
fp = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 1))
fn = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 0))

confusion_matrix1 = np.array([[tp, fp],
                              [fn, tn]])
print(confusion_matrix1)
# array([[3, 1],
#        [2, 2]])

# 也可以使用 scikit-learn 的 metrics 套件所提供的 confusion_matrix() 函式來製作，
# 但要注意兩種方法在混淆矩陣元素的配置有所不同。
confusion_matrix2 = confusion_matrix(y_true, y_pred)
print(confusion_matrix2)
# array([[2, 1],
#        [2, 3]])

# -----------------------------------
# accuracy

from sklearn.metrics import accuracy_score

# 使用 0 和 1 來表示二元分類的負例和正例
y_true = [1, 0, 1, 1, 0, 1, 1, 0]
y_pred = [0, 0, 1, 1, 0, 0, 1, 1]
accuracy = accuracy_score(y_true, y_pred)
print(accuracy)
# 0.625

# -----------------------------------
# logloss

from sklearn.metrics import log_loss

# 以 0 和 1 表示二元分類的負例和正例
y_true = [1, 0, 1, 1, 0, 1]
y_prob = [0.1, 0.2, 0.8, 0.8, 0.1, 0.3]

logloss = log_loss(y_true, y_prob)
print(logloss)
# 0.7136

# -----------------------------------
# 多元分類
# -----------------------------------
# multi-class logloss

from sklearn.metrics import log_loss

# 3 類別分類的實際值與預測值
y_true = np.array([0, 2, 1, 2, 2])
y_pred = np.array([[0.68, 0.32, 0.00],
                   [0.00, 0.00, 1.00],
                   [0.60, 0.40, 0.00],
                   [0.00, 0.00, 1.00],
                   [0.28, 0.12, 0.60]])
logloss = log_loss(y_true, y_pred)
print(logloss)
# 0.3626

# -----------------------------------
# 多標籤分類
# -----------------------------------
# mean_f1, macro_f1, micro_f1

from sklearn.metrics import f1_score

# 在計算多標籤分類的評價指標時，將實際值與預測值以 k-hot 編碼的形式來表示會比較好計算
# 實際值：[[1,2], [1], [1,2,3], [2,3], [3]]
y_true = np.array([[1, 1, 0],
                   [1, 0, 0],
                   [1, 1, 1],
                   [0, 1, 1],
                   [0, 0, 1]])

# 預測值：[[1,3], [2], [1,3], [3], [3]]
y_pred = np.array([[1, 0, 1],
                   [0, 1, 0],
                   [1, 0, 1],
                   [0, 0, 1],
                   [0, 0, 1]])

# 計算 mean-f1 評價指標時，先以資料為單位計算 F1-score，再取其平均
mean_f1 = np.mean([f1_score(y_true[i, :], y_pred[i, :]) for i in range(len(y_true))])

# 計算 macro-f 評價指標時，先以分類為單位計算 F1-score，再取其平均
n_class = 3
macro_f1 = np.mean([f1_score(y_true[:, c], y_pred[:, c]) for c in range(n_class)])

# 計算 micro-f1 評價指標時，以資料×分類為一組，計算各組別的 TP/TN/FP/FN 並求得 F1-score
micro_f1 = f1_score(y_true.reshape(-1), y_pred.reshape(-1))

print(mean_f1, macro_f1, micro_f1)
# 0.5933, 0.5524, 0.6250

# 也可以直接在 f1_score 函式中加上 average 參數來計算
mean_f1 = f1_score(y_true, y_pred, average='samples')
macro_f1 = f1_score(y_true, y_pred, average='macro')
micro_f1 = f1_score(y_true, y_pred, average='micro')

# -----------------------------------
# 類別之間有順序關係的多元分類
# -----------------------------------
# quadratic weighted kappa

from sklearn.metrics import confusion_matrix, cohen_kappa_score


# 建立用來計算 quadratic weighted kappa 的函式
def quadratic_weighted_kappa(c_matrix):
    numer = 0.0
    denom = 0.0

    for i in range(c_matrix.shape[0]):
        for j in range(c_matrix.shape[1]):
            n = c_matrix.shape[0]
            wij = ((i - j) ** 2.0)
            oij = c_matrix[i, j]
            eij = c_matrix[i, :].sum() * c_matrix[:, j].sum() / c_matrix.sum()
            numer += wij * oij
            denom += wij * eij

    return 1.0 - numer / denom


# y_true 為實際值類別的 list、y_pred 為預測值 list
y_true = [1, 2, 3, 4, 3]
y_pred = [2, 2, 4, 4, 5]

# 計算混淆矩陣
c_matrix = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])

# 計算 quadratic weighted kappa
kappa = quadratic_weighted_kappa(c_matrix)
print(kappa)
# 0.6153

# 也能夠直接計算 quadratic weighted kappa，不用先算混淆矩陣
kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')

# -----------------------------------
# 推薦任務
# -----------------------------------
# MAP@K

# K=3，資料筆數為 5 筆，類別有 4 類
K = 3

# 每筆資料的實際值
y_true = [[1, 2], [1, 2], [4], [1, 2, 3, 4], [3, 4]]

# 每筆資料的預測值。因為 K=3，因此從每筆資料預測出最有可能的 3 筆數據，並將其排名
y_pred = [[1, 2, 4], [4, 1, 2], [1, 4, 3], [1, 2, 3], [1, 2, 4]]


# 建立以下函式來計算每筆資料的平均精確率
def apk(y_i_true, y_i_pred):
    # y_pred 的長度必須在 K 以下，且元素不能重覆
    assert (len(y_i_pred) <= K)
    assert (len(np.unique(y_i_pred)) == len(y_i_pred))

    sum_precision = 0.0
    num_hits = 0.0

    for i, p in enumerate(y_i_pred):
        if p in y_i_true:
            num_hits += 1
            precision = num_hits / (i + 1)
            sum_precision += precision

    return sum_precision / min(len(y_i_true), K)


# 建立計算 MAP@K 的函式
def mapk(y_true, y_pred):
    return np.mean([apk(y_i_true, y_i_pred) for y_i_true, y_i_pred in zip(y_true, y_pred)])


# 求得 MAP@K
print(mapk(y_true, y_pred))
# 0.65

# 即便預測值內的正解與正解的數量相同，只要預測值的排名不同分數就會不同
print(apk(y_true[0], y_pred[0]))
print(apk(y_true[1], y_pred[1]))
# 1.0, 0.5833
