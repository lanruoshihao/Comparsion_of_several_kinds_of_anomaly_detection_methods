#!/usr/bin/env python
# coding: utf-8

import codecs
import numpy as np
import pandas as pd
from sklearn import metrics
from pylab import *
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib.ticker import MultipleLocator


def read_file_data(file_path):
    f_obj = open(file_path,  'rb', encoding='utf-8')
    file_data = f_obj.read()
    file_data = file_data.strip()
    # print("file_data=", file_data)
    data_list = file_data.split("\n")
    final_data_list = []
    for line in data_list:
        if len(line) == 0:
            continue
        else:
            line = line.strip()
            # print("line=", line)
            final_data_list.append(line.split(','))
    # print(final_data)
    return final_data_list


def max_min_standard(cluster_data):
    for js in range(len(cluster_data)):
        max_data = np.max(cluster_data[js])
        min_data = np.min(cluster_data[js])
        for num in range(len(cluster_data[js])):
            cluster_data[js][num] = (cluster_data[js][num] - min_data)/(max_data - min_data)
    return cluster_data


def kmeans_cluster(file_path):
    # 通过比较不同k值对应的轮廓系数来获得最优的k值以及聚类的组别标签
    final_data_list = read_file_data(file_path)
    final_data_list = np.array(final_data_list).astype(np.float64)
    # final_data_list = max_min_standard(final_data_list)
    score_list = []  # 存放每次结果的轮廓系数silhouette_score
    cluster_num_list = []
    for k in range(2, 30):
        km_cluster = KMeans(n_clusters=k, max_iter=1000, tol=1e-5)  # 构造kmeans聚类算法模型
        km_cluster.fit(final_data_list)
        km_labels = km_cluster.labels_
        scores = metrics.silhouette_score(final_data_list, km_labels)
        score_list.append(scores)
        cluster_num_list.append(k)
    print("score_list=", score_list)
    return final_data_list, score_list, cluster_num_list


def construct_silhouette_score(score_list, cluster_num_list):
    ###
    ax = subplot(1, 1, 1)
    ymajorLocator = MultipleLocator(0.02)  # 将y轴主刻度标签设置为0.5的倍数
    ax.yaxis.set_major_locator(ymajorLocator)
    ##
    xmajorLocator = MultipleLocator(1)  # 将x轴主刻度标签设置为1的倍数
    ax.xaxis.set_major_locator(xmajorLocator)
    ###
    plt.plot(cluster_num_list, score_list)
    plt.xlabel("聚类的组别数目")
    plt.ylabel("轮廓系数silhouette_score")
    plt.show()


def get_nodes_distance(list1, list2):
    # 计算任意两点之间的距离
    data_distances = np.sqrt(sum([(x - y) ** 2 for x, y in zip(list1, list2)]))
    return data_distances


def Outlier_test_based_on_distance(test_df, dist_r=10, p_n=0.01):
    '''
    # 基于距离的嵌套循环检测算法
    :param dist_r表示距离阈值, 可根据需要自由调整
    :param 为分数阈值, 可根据需要自由调整
    :param test_df是根据组编号和组内点离中心点距离排好序的数据框
    '''
    data_df = test_df
    data_df.drop(['dist', 'cluster_code'], axis=1, inplace=True)
    Outlier_set_list = []
    for js in range(len(test_df)):
        js_cnt = 0
        flag = 0
        for num in range(len(test_df)):
            dist_obj = get_nodes_distance(list(data_df.ix[js]), list(data_df.ix[num]))
            if js != num and dist_obj <= dist_r:
                js_cnt = js_cnt + 1
                if js_cnt >= p_n*len(test_df):
                    flag = 1
                    break
        if flag == 1:
            continue
        else:
            print("排序好的第%d个数据对象是离群点"%(js))
            Outlier_set_list.append(js)
    return Outlier_set_list


file_path = "Ionosphere.txt"
final_data_list, score_list, cluster_num_list = kmeans_cluster(file_path)
print("len(final_data_list)=", len(final_data_list))
print(np.argmax(score_list))
print(cluster_num_list[np.argmax(score_list)])
###
n_clusters = cluster_num_list[np.argmax(score_list)]
km_cluster = KMeans(n_clusters=n_clusters, max_iter=10000, tol=2e-6).fit(final_data_list)
km_labels = km_cluster.labels_
km_cluster_centers = km_cluster.cluster_centers_
print("km_cluster_centers=", km_cluster_centers)
print("km_labels=", km_labels)
print("len(km_labels)=", len(km_labels))
###
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
###
construct_silhouette_score(score_list, cluster_num_list)
###
data_df = pd.DataFrame(final_data_list)
test_data_df = data_df
###
###
pca = PCA(n_components=2)
new_pca = pd.DataFrame(pca.fit_transform(data_df))
print("type(new_pca)=", type(new_pca))
##
data_df['cluster_code'] = km_labels
###
##
dist_list = []
for js in range(len(test_data_df)):
    if data_df.loc[js, 'cluster_code'] == 0:
        dist_list.append(get_nodes_distance(list(test_data_df.iloc[js]), list(km_cluster_centers[0])))
    elif data_df.loc[js, 'cluster_code'] == 1:
        dist_list.append(get_nodes_distance(list(test_data_df.iloc[js]), list(km_cluster_centers[1])))
    elif data_df.loc[js, 'cluster_code'] == 2:
        dist_list.append(get_nodes_distance(list(test_data_df.iloc[js]), list(km_cluster_centers[2])))
    else:
        dist_list.append(get_nodes_distance(list(test_data_df.iloc[js]), list(km_cluster_centers[3])))
###
data_df['dist'] = dist_list
###
# 根据聚类组别和每组按照点离该组均值点距离由近到远进行排序
sort_data_df = data_df.sort_values(by=['cluster_code', 'dist'], ascending=[True, True])
sort_data_df.reset_index(drop=True, inplace=True)
print("sort_data_df=", sort_data_df)
###
Outlier_set_list = Outlier_test_based_on_distance(sort_data_df, dist_r=1, p_n=0.01)
print("sort_data_df中为离群点的数据对象索引号列表Outlier_set_list=", Outlier_set_list)
print("sort_data_df中为离群点的数据对象个数len(Outlier_set_list)=", len(Outlier_set_list))
###
data_df.drop('dist', axis=1, inplace=True)
print("data_df.columns=", data_df.columns)
###
print("km_labels=", km_labels)
df_count_type = data_df.groupby('cluster_code').apply(np.size)
print("df_count_type=", df_count_type)
print(data_df)
###
# print("data_df=", data_df)
print("data_df['cluster_code']=", list(set(data_df['cluster_code'])))
data_0 = new_pca[data_df['cluster_code'] == 0]
plt.plot(data_0[0], data_0[1], 'r*')
###
data_1 = new_pca[data_df['cluster_code'] == 1]
plt.plot(data_1[0], data_1[1], 'g*')
###
data_2 = new_pca[data_df['cluster_code'] == 2]
plt.plot(data_2[0], data_2[1], 'b*')
###
data_3 = new_pca[data_df['cluster_code'] == 3]
plt.plot(data_3[0], data_3[1], 'k*')
plt.title('kmeans_clustering_algorithm')
###
###
plt.gcf().savefig('kmeans.png')
plt.show()




