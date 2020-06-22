#!/usr/bin/env python
# coding: utf-8

import codecs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


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


def get_nodes_distance(list1, list2):
    # 计算任意两点之间的距离
    data_distances = np.sqrt(sum([(float(x)-float(y))**2 for x, y in zip(list1, list2)]))
    return data_distances


def get_min_k_distances(code_p, data_df, k=100):
    '''
    :param code_p: 数据对象的编号
    :param data_df: 待分析数据对象组成的数据框data_df的长度
    :param k: 离数据对象code_p距离最近的个数k
    :return:返回跟code_p最近的k个点中的距离最大值
    '''
    dist_list = []
    k_neighbourhood_dict = {}
    if code_p >= len(data_df):
        print("数据对象的编号code_p要小于数据框data_df的长度")
    for js in range(len(data_df)):
        if code_p == js:
            continue
        else:
            # print("data_df.iloc[code_p]=", data_df.iloc[code_p])
            # print("data_df.iloc[js]=", data_df.iloc[js])
            dist = get_nodes_distance(list(data_df.iloc[code_p]), list(data_df.iloc[js]))
            k_neighbourhood_dict[js] = dist
            dist_list.append(dist)
    k_neighbourhood_dict = dict(sorted(k_neighbourhood_dict.items(), key=lambda x: x[1], reverse=False))
    return sorted(dist_list)[k-1], list(k_neighbourhood_dict.keys())[:k], k_neighbourhood_dict


def get_p_lsr(k_neighbourhood_dict, k):
    # 计算数据对象p的局部稀疏率
    k_neighbourhood_dict = dict(sorted(k_neighbourhood_dict.items(), key=lambda x: x[1], reverse=False))
    dist_list = list(k_neighbourhood_dict.values())[:k]  # 计算p点k邻域内所有点离p点的距离之和
    total_dist = np.sum(dist_list)
    p_nums = len(dist_list)
    p_lsr = p_nums/total_dist
    return p_lsr


def get_p_obj_pf(code_p, k_neighbourhood_list, data_df):
    # 计算数据对象p的裁剪系数，裁剪系数表示数据集中数据的平均密度
    dist_sum = 0
    k_neighbourhood_nodes = k_neighbourhood_list[code_p]
    for node_num in range(len(k_neighbourhood_nodes)):
        for node_js in range(node_num+1, len(k_neighbourhood_nodes)):
            neighbour_1 = int(k_neighbourhood_nodes[node_num])
            neighbour_2 = int(k_neighbourhood_nodes[node_js])
            dist = get_nodes_distance(list(data_df.iloc[neighbour_1]), list(data_df.iloc[neighbour_2]))
            # print("dist=", dist)
            dist_sum = dist_sum + np.sqrt(dist)
    Nk_p_mean = float(dist_sum)/len(k_neighbourhood_nodes)
    p_pf = 1.0/Nk_p_mean  # 计算数据对象p的裁剪系数
    return p_pf


file_path = "Ionosphere.txt"
data_list = read_file_data(file_path)
data_df = pd.DataFrame(data_list)
###
###
k_max_dist_list = []  # 记录距离指定数据对象最近的k个点中距离的最大值
k_neighbourhood_list = []
outlier_node_list = []  # 可能是离群点的候选集合
k = 5
data_js_cnt = 0
for nums in range(len(data_df)):
    data_js_cnt += 1
    print("data_js_cnt=", data_js_cnt)
    k_max_dist, k_neighbourhood_code, k_neighbourhood_dict = get_min_k_distances(nums, data_df, k)
    k_max_dist_list.append(k_max_dist)
    k_neighbourhood_list.append(k_neighbourhood_code)
    p_pf = get_p_obj_pf(nums, k_neighbourhood_list, data_df)
    print("p_pf=", p_pf)
    p_lsr = get_p_lsr(k_neighbourhood_dict, k)
    print("p_lsr=", p_lsr)
    if p_lsr < p_pf:
        outlier_node_list.append(nums)
    else:
        continue
###
wait_test_df = data_df.iloc[outlier_node_list]
wait_test_df.reset_index(drop=True, inplace=True)
print("outlier_node_list=", outlier_node_list)
print("type(outlier_node_list)=", type(outlier_node_list))
print("wait_test_df=", wait_test_df)
print("type(wait_test_df)=", type(wait_test_df))
###
LSC_dict = {}
data_js_num = 0
for js_cnt in range(len(wait_test_df)):
    data_js_num += 1
    lsr_sum = 0
    print("data_js_num=", data_js_num)
    cnt_k_max_dist, cnt_k_neighbourhood_code, cnt_k_neighbourhood_dict = get_min_k_distances(js_cnt, wait_test_df, k)
    p_lsr = get_p_lsr(cnt_k_neighbourhood_dict, k)
    for js_num in range(len(wait_test_df)):
        if js_cnt!=js_num:
            num_k_max_dist, num_k_neighbourhood_code, num_k_neighbourhood_dict = get_min_k_distances(js_num, wait_test_df, k)
            p_lsr_num = get_p_lsr(num_k_neighbourhood_dict, k)
            lsr_sum += p_lsr_num
    LSC_dict[str(js_cnt)] = lsr_sum/(p_lsr*len(cnt_k_neighbourhood_code))
###
n = 40
LSC_dict = dict(sorted(LSC_dict.items(), key=lambda x: x[1], reverse=True))
# print("LSC_dict=", LSC_dict)
code_list = [int(node_code) for node_code in list(LSC_dict.keys())]
outlier_df = wait_test_df.iloc[code_list]
outlier_df.reset_index(drop=True, inplace=True)
###
pca = PCA(n_components=2)
new_pca = pd.DataFrame(pca.fit_transform(data_df))
# print("type(new_pca)=", type(new_pca))
###
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.scatter(np.array(new_pca)[:, 0], np.array(new_pca)[:, 1], s=10, c='b', alpha=0.5)
###
print("outlier_df=", outlier_df)
print("type(outlier_df)=", type(outlier_df))
###
# outlier_df.to_excel("avg_density_outliers_test.xlsx", sheet_name="avg_density_outliers", \
#                         encoding='utf-8')  # 将采用离群因子而检测到的异常点保存到数据表中
###
pca = PCA(n_components=2)
pca_outlier_df = pd.DataFrame(pca.fit_transform(outlier_df))
plt.scatter(np.array(pca_outlier_df)[:, 0], np.array(pca_outlier_df)[:, 1], s=10, c='r', alpha=0.5)
plt.title("基于平均密度的ELSC算法异常检测")
plt.show()





