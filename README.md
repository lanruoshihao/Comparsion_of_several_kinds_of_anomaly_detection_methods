# Comparsion_of_several_kinds_of_anomaly_detection_methods

python开发环境：pycharm+Anaconda3

几种非监督学习的异常检测算法：

1、孤立森林(Isolation Forest)

2、一分类支持向量机(one-class svm)

3、自编码算法(Autoencoder)

4、基于距离的嵌套循环检测算法

5、局部离群因子(LOF)

6、基于平均密度的增强局部稀疏系数(Enhanced Local Sparsity Coefficient)

ELSC作为LOF算法的改进，属于基于平均密度的离群点检测算法，

缺点：计算复杂度较高，不适合于大规模数据集和高维数据集。

 
算法原理参考链接：http://www.doc88.com/p-9807408443179.html

7、kmeans算法

8、DBSCAN算法

优点：DBSCAN在聚类时不需要事先指定簇类的数量；可以处理任意形状的簇类；可以检测数据集的噪声；并且对数据集中的异常点不敏感；聚类结果对样本集的随机抽样顺序不敏感。

缺点：DBSCAN进行聚类时一般使用欧氏距离，对于很高维的数据集，这会带来维数灾难，导致选择合适的值很困难；

DBSCAN在聚类时如果在不同簇之间的样本集密度相差很大，导致对数据集在聚类时选择合适的簇类最小样本数minPts比较困难，这样选择的minPts很难适合所有的簇，

最终会导致DBSCAN聚类效果很差。


python中可以用来进行异常检测的算法库：sklearn、pyod和PyNomaly

sklearn库的学习参考链接：https://scikit-learn.org/

pyod库的学习参考链接：https://www.biaodianfu.com/pyod.html

PyNomaly库的学习参考链接：https://python.ctolib.com/vc1492a-PyNomaly.html
