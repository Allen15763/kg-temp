# ---------------------------------
# 準備資料
# ----------------------------------
import numpy as np
import pandas as pd

# 讀取資料
# 用Pandas的DataFrame存資料

train = pd.read_csv('../input/sample-data/train_preprocessed_onehot.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
test_x = pd.read_csv('../input/sample-data/test_preprocessed_onehot.csv')

# 備份資料，以便之後再利用
train_x_saved = train_x.copy()
test_x_saved = test_x.copy()

from sklearn.preprocessing import StandardScaler, MinMaxScaler


# 標準化訓練資料與測試資料，並傳回 dataframe
def load_standarized_data():
    train_x, test_x = train_x_saved.copy(), test_x_saved.copy()

    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)
    return pd.DataFrame(train_x), pd.DataFrame(test_x)


# 使用 Min-Max 縮放訓練資料與測試資料，並傳回 dataframe
def load_minmax_scaled_data():
    train_x, test_x = train_x_saved.copy(), test_x_saved.copy()

    # Min-Max 縮放
    scaler = MinMaxScaler()
    scaler.fit(pd.concat([train_x, test_x], axis=0))
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)

    return pd.DataFrame(train_x), pd.DataFrame(test_x)


# -----------------------------------
# 主成分分析
# -----------------------------------
# 讀取標準化數據
train_x, test_x = load_standarized_data()
# -----------------------------------
from sklearn.decomposition import PCA

# 定義以訓練資料來進行PCA轉換
pca = PCA(n_components=5)
pca.fit(train_x)

# 進行轉換
train_x = pca.transform(train_x)
test_x = pca.transform(test_x)

# -----------------------------------
# 讀取標準化數據
train_x, test_x = load_standarized_data()
# -----------------------------------
# TruncatedSVD
from sklearn.decomposition import TruncatedSVD

# 定義以訓練資料來進行PCA轉換
svd = TruncatedSVD(n_components=5, random_state=71)
svd.fit(train_x)

# 進行轉換
train_x = svd.transform(train_x)
test_x = svd.transform(test_x)

# -----------------------------------
# 非負矩陣分解
# -----------------------------------
# 用 Min-Max 縮放資料
train_x, test_x = load_minmax_scaled_data()
# -----------------------------------
from sklearn.decomposition import NMF

# 定義訓練資料的 NMF 轉換
model = NMF(n_components=5, init='random', random_state=71)
model.fit(train_x)

# 進行轉換
train_x = model.transform(train_x)
test_x = model.transform(test_x)

# -----------------------------------
# LatentDirichletAllocation
# -----------------------------------
# 用 Min-Max 縮放資料
train_x, test_x = load_minmax_scaled_data()
# -----------------------------------
from sklearn.decomposition import LatentDirichletAllocation

# 假設資料為單字語句的計算陣列

# 定義訓練資料的 LDA 轉換
model = LatentDirichletAllocation(n_components=5, random_state=71)
model.fit(train_x)

# 進行轉換
train_x = model.transform(train_x)
test_x = model.transform(test_x)

# -----------------------------------
# 線性判別分析 
# -----------------------------------
# 使用標準化數據
train_x, test_x = load_standarized_data()
# -----------------------------------
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# 定義訓練資料的 LDA 轉換
lda = LDA(n_components=1)
lda.fit(train_x, train_y)

# 進行轉換
train_x = lda.transform(train_x)
test_x = lda.transform(test_x)

# -----------------------------------
# t-sne
# -----------------------------------
# 使用標準化數據
train_x, test_x = load_standarized_data()
# -----------------------------------
import bhtsne

# 進行 t-sne 的轉換
data = pd.concat([train_x, test_x])
embedded = bhtsne.tsne(data.astype(np.float64), dimensions=2, rand_seed=71)

# -----------------------------------
# UMAP
# -----------------------------------
# 使用標準化數據
train_x, test_x = load_standarized_data()
# -----------------------------------
import umap

# 定義訓練資料的 UMAP 轉換
um = umap.UMAP()
um.fit(train_x)

# 執行轉換
train_x = um.transform(train_x)
test_x = um.transform(test_x)

# -----------------------------------
# 群聚分析
# -----------------------------------
# 使用標準化數據
train_x, test_x = load_standarized_data()
# -----------------------------------
from sklearn.cluster import MiniBatchKMeans

# 定義訓練資料的 Mini-Batch K-Means 轉換
kmeans = MiniBatchKMeans(n_clusters=10, random_state=71)
kmeans.fit(train_x)

# 輸出所屬的組別
train_clusters = kmeans.predict(train_x)
test_clusters = kmeans.predict(test_x)

# 輸出到各組別中心的距離
train_distances = kmeans.transform(train_x)
test_distances = kmeans.transform(test_x)
