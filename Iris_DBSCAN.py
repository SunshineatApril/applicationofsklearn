import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from sklearn.datasets import load_iris

iris = load_iris()                      #导入sklearn包中自带的鸢尾花数据集(包括数据特征)
X = iris['data']                        #取训练数据
n_samples, n_features = X.shape         #取训练数据样本数量、属性数量
feature_names = iris['feature_names']   #取训练数据属性名称
print("\n训练数据样本数量", n_samples)                        
print("训练数据属性数量", n_features)
print("训练数据属性名称", feature_names)

X = X[0:n_samples,0:2]                  #为更好地展示聚类效果，只用训练数据的前两个属性聚类，即将训练数据降为2维
plt.scatter(X[0:n_samples, 0], X[0:n_samples, 1], c = "red", marker='o', label='all samples')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.legend(loc=0)
plt.show()                              #用matplotlib包展示所有点的位置

iris_dbscan = DBSCAN(eps = 0.3, min_samples = 5)        #设定邻域半径和邻域样本数量阈值，初始化DBSCAN对象
iris_dbscan.fit(X)                      #用训练数据完成训练
print("\n训练完成")  
print("核心对象索引\n", iris_dbscan.core_sample_indices_)     
print("核心对象坐标\n", iris_dbscan.components_)
label = iris_dbscan.labels_             #训练数据的簇标签，-1表示噪音点
print("训练数据簇标签\n", label)
print("\n")

plt.scatter(X[:, 0], X[:, 1], c=label, marker = 'o', label = 'eps = 0.3, min_samples = 5')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.legend(loc=0)
plt.show()                              #用matplotlib包展示聚类效果
