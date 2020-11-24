import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from sklearn.datasets import make_blobs

from sklearn import metrics

#X为数据集，包括数据特征，y为标签。生成1000个样本，包括2个属性，4个簇中心分别为
#[0,1], [0.5,1.5], [1,2], [2,3]，每个簇的方差分别是0.2, 0.2, 0.2, 0.2
X, y = make_blobs(n_samples=1000, n_features=2, centers=[[0,1], [0.5,1.5], [1,2], [2,3]], cluster_std=[0.2, 0.2, 0.2, 0.2])
print("\n生成训练数据簇数量", "4")
print("簇中心分别是","[0,1], [0.5,1.5], [1,2], [2,3]")
print("生成训练数据样本数量", "1000")                        
print("生成训练数据属性数量", "2")
#展示所有样本的分布
plt.scatter(X[:, 0], X[:, 1], c = "red", marker = '+')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#设定训练的簇中心数量为3，初始化KMeans对象并完成训练，用labels_3centers存储样本标签
kmeans_3centers = KMeans(n_clusters=3).fit(X)
print("\n簇数量", "3" ,",训练完成")     
print("簇中心\n", kmeans_3centers.cluster_centers_)
labels_3centers = kmeans_3centers.labels_
#用Calinski-Harabasz Index评估方式为聚类打分
score_3centers = metrics.calinski_harabasz_score(X, labels_3centers)
print("kmeans把训练数据聚为3个簇，打分是",score_3centers)
#设定为3个簇时，输出聚类效果
plt.scatter(X[:, 0], X[:, 1], c=labels_3centers, marker = '+')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#设定训练的簇中心数量为4，初始化KMeans对象并完成训练，用labels_4centers存储样本标签
kmeans_4centers = KMeans(n_clusters=4).fit(X)
print("\n簇数量", "4" ,",训练完成")     
print("簇中心\n", kmeans_4centers.cluster_centers_)
labels_4centers = kmeans_4centers.labels_
#用Calinski-Harabasz Index评估方式为聚类打分
score_4centers = metrics.calinski_harabasz_score(X, labels_4centers)
print("kmeans把训练数据聚为4个簇，打分是",score_4centers)
#设定为4个簇时，输出聚类效果
plt.scatter(X[:, 0], X[:, 1], c=labels_4centers, marker = '+')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#设定训练的簇中心数量为5，初始化KMeans对象并完成训练，用labels_5centers存储样本标签
kmeans_5centers = KMeans(n_clusters=5).fit(X)
print("\n簇数量", "5" ,",训练完成")     
print("簇中心\n", kmeans_5centers.cluster_centers_)
labels_5centers = kmeans_5centers.labels_
#用Calinski-Harabasz Index评估方式为聚类打分
score_5centers = metrics.calinski_harabasz_score(X, labels_5centers)
print("kmeans把训练数据聚为5个簇，打分是",score_5centers)
#设定为5个簇时，输出聚类效果
plt.scatter(X[:, 0], X[:, 1], c=labels_5centers, marker = '+')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
print("\n")
