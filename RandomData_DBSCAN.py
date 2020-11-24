import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

from sklearn.datasets import make_moons

from sklearn import metrics

#X为数据集，包括数据特征，y为标签。生成1000个月牙型样本，noise=0.1
X , y = make_moons(n_samples=1000, noise=0.1)
print("\n生成训练数据样本数量", "1000")                      
#展示所有样本的分布
plt.scatter(X[:, 0], X[:, 1], c = "red", marker = '+', label = 'all samples')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=0)
plt.show()

#设定训练的簇中心数量为2，初始化KMeans对象并完成训练，用labels_kmeans存储样本标签
kmeans_2centers = KMeans(n_clusters=2).fit(X)
print("\n簇数量", "2" ,",KMeans训练完成")     
labels_kmeans = kmeans_2centers.labels_
#设定为3个簇时，输出聚类效果
plt.scatter(X[:, 0], X[:, 1], c=labels_kmeans, marker = '+', label = 'KMeans n_clusters = 2')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=0)
plt.show()

#设定邻域半径和邻域样本数量阈值，初始化DBSCAN对象并完成训练，用labels_dbscan存储样本标签
dbscan_ncenters = DBSCAN(eps = 0.1, min_samples = 10).fit(X)
print("\nDBSCAN训练完成\n")   
labels_dbscan = dbscan_ncenters.labels_   
#输出聚类效果
plt.scatter(X[:, 0], X[:, 1], c=labels_dbscan, marker = '+', label = 'DBSCAN eps = 0.1, min_samples = 10')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=0)
plt.show()
