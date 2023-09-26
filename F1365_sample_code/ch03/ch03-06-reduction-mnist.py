# ---------------------------------
# 準備資料
# ----------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# MNIST 手寫數字資料視覺化

# 讀取 keras.datasets 裡面的 MNIST 手寫數字資料
from keras.datasets import mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# 將 28 x 28 的影像轉換成 784 x 1 的影像
train_x = train_x.reshape(train_x.shape[0], -1)

# 此範例程式只分析前 1000 張影像
train_x = pd.DataFrame(train_x[:1000, :])
train_y = train_y[:1000]

# -----------------------------------
# PCA
# -----------------------------------
from sklearn.decomposition import PCA

# 將訓練資料做 PCA 轉換
pca = PCA()
x_pca = pca.fit_transform(train_x)

# 將每一個項目(數字)指定一個顏色，並且顯示出來
f, ax = plt.subplots(1)
for i in range(10):
    mask = train_y == i
    plt.scatter(x_pca[mask, 0], x_pca[mask, 1], label=i, s=10, alpha=0.5)
ax.legend(bbox_to_anchor=(1.00, 1), loc='upper left')

plt.show()

# -----------------------------------
# LDA (Linear Discriminant Analysis)
# -----------------------------------
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# 透過線性判別分析找出最能分辨資料的兩個軸
lda = LDA(n_components=2)
x_lda = lda.fit_transform(train_x, train_y)

# 將每一個項目(數字)指定一個顏色，並且顯示出來
# 這個方法的劃分效果還滿好，但請注意這個方法使用了訓練資料的標籤
f, ax = plt.subplots(1)
for i in range(10):
    mask = train_y == i
    plt.scatter(x_lda[mask, 0], x_lda[mask, 1], label=i, s=10, alpha=0.5)
ax.legend(bbox_to_anchor=(1.00, 1), loc='upper left')

plt.show()

# -----------------------------------
# t-sne
# -----------------------------------
from sklearn.manifold import TSNE

# 使用 t-sne 轉換
tsne = TSNE(n_components=2)
x_tsne = tsne.fit_transform(train_x)

# 將每一個項目(數字)指定一個顏色，並且顯示出來
f, ax = plt.subplots(1)
for i in range(10):
    mask = train_y == i
    plt.scatter(x_tsne[mask, 0], x_tsne[mask, 1], label=i, s=10, alpha=0.5)
ax.legend(bbox_to_anchor=(1.00, 1), loc='upper left')

plt.show()

# -----------------------------------
# UMAP
# -----------------------------------
import umap

# 使用 UMAP 轉換
um = umap.UMAP()
x_umap = um.fit_transform(train_x)

# 將每一個項目(數字)指定一個顏色，並且顯示出來
f, ax = plt.subplots(1)
for i in range(10):
    mask = train_y == i
    plt.scatter(x_umap[mask, 0], x_umap[mask, 1], label=i, s=10, alpha=0.5)
ax.legend(bbox_to_anchor=(1.00, 1), loc='upper left')

plt.show()
