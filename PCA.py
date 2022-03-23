from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def draw3D(X,y=None):
    # 降维结果绘图
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker="o", c=y, s=50, edgecolors="k")
    plt.show()

def draw2D(X,y=None):
    # 聚类结果绘图
    plt.figure(figsize=(12,8))
    plt.scatter(X[:, 0], X[:, 1], marker="o", c=y, s=50, edgecolors="k")
    plt.show()

def decomposition_PCA(X):
    # 先不降维，观察每个维度的方差贡献
    pca = PCA()
    pca.fit(X)
    print('降维前方差：',pca.explained_variance_, '\n', '降维前方差占比：',pca.explained_variance_ratio_)

    # 通过观察方差占比接着进行降维，有三种方式：
    # 1.人为决定降维的维度数 pca = PCA(n_components=2)
    # 2.保留主成分的方差占比 pca = PCA(n_components=0.95)
    # 3.MLE算法           pca = PCA(n_components='mle')
    pca = PCA(n_components='mle')
    pca.fit(X)
    print('降维后方差：',pca.explained_variance_, '\n', '降维后方差占比：',pca.explained_variance_ratio_)

    embedding = pca.transform(X)  # 将原始数据投影到保留的维度上形成维度更低的新数据
    return embedding
    # X_ = pca.inverse_transform(X)  # 将降维后的数据恢复到原始数据的维度上

def decomposition_TSNE(X,n_components=2):
    # TSNE降维不能观察方差占比，而且需要指定维度
    return TSNE(n_components=n_components).fit_transform(X)


# 数据集
X,y = make_blobs(n_samples=200, n_features=3,centers=[[-1,-1,-1],[1,1,1],[1,-1,-1],[-1,1,-1]], cluster_std=0.3)
draw3D(X,y)

# PCA
X_PCA = decomposition_PCA(X)
draw2D(X_PCA,y)

# TSNE
# 一般而言TSNE降维后的数据维度较少，但是方差占比较高，数据更好分开，更有利于作为特征训练模型
X_TSNE = decomposition_TSNE(X)
draw2D(X_TSNE,y)