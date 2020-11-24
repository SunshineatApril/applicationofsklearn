import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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

iris_kmeans = KMeans(n_clusters=3)      #设定簇中心数量，初始化KMeans对象
iris_kmeans.fit(X)                      #用训练数据完成训练
print("\n簇数量", "3" ,",训练完成")     
print("簇中心\n", iris_kmeans.cluster_centers_)
label = iris_kmeans.labels_             #训练数据的簇标签(长度为n_samples)
print("训练数据簇标签\n", label)

l0 = X[label == 0]
l1 = X[label == 1]
l2 = X[label == 2]
plt.scatter(l0[:, 0], l0[:, 1], c = "red", marker='o', label='samples-label0')
plt.scatter(l1[:, 0], l1[:, 1], c = "green", marker='*', label='samples-label1')
plt.scatter(l2[:, 0], l2[:, 1], c = "blue", marker='+', label='samples-label2')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.legend(loc=0)
plt.show()                              #用matplotlib包展示聚类效果

print("\n预测点[1, 2], [5, 6]的簇标签")     
print(iris_kmeans.predict([[1, 2], [5, 6]]))
print("\n")
