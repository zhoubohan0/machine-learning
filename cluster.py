from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import calinski_harabasz_score, silhouette_score
import matplotlib.pyplot as plt


def draw3D(X,y=None):
    # 聚类结果绘图
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker="o", c=y, s=50, edgecolors="k")
    plt.show()

def draw2D(X,y=None):
    # 聚类结果绘图
    plt.figure(figsize=(12,8))
    plt.scatter(X[:, 0], X[:, 1], marker="o", c=y, s=50, edgecolors="k")
    plt.show()

# 数据集
X, y = make_blobs(n_samples=1000, n_features=3, centers=[[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]], cluster_std=[0.3, 0.2, 0.2, 0.2])
draw3D(X,y)

# 聚类
clusterer = KMeans(n_clusters=4)        # 定义聚类器
clusterer.fit(X)                        # 训练，fit之后才能调用各种方法
predict = clusterer.predict(X)          # 聚类结果
distance = clusterer.transform(X)       # 聚类后每个样本到簇心距离
centroids = clusterer.cluster_centers_  # 聚类簇心
draw3D(X,predict)

# 聚类评价
'''聚类后每个样本到簇心距离的平方总和,看作损失'''
inertia = clusterer.inertia_
'''平均轮廓系数,越接近1越好'''
s_score = silhouette_score(X, predict)
'''CH分数，簇间的协方差矩阵与簇内的协方差矩阵相除，类内数据协方差越小，类间数据协方差越大，Calinski-Harabasz分数越高'''
CH_score = calinski_harabasz_score(X, predict)
