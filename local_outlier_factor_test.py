#!/usr/bin/env python
# coding: utf-8

import codecs
import pandas as pd
from pylab import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor


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


def localoutlierfactor(data, predict, k):
    lof_clf = LocalOutlierFactor(n_neighbors=k+1, contamination=0.2, n_jobs=-1)
    lof_clf.fit(data)
    # 记录 k 邻域距离
    predict['k_distances'] = lof_clf .kneighbors(predict)[0].max(axis=1)
    # 记录 LOF 离群因子，做相反数处理
    predict['local_outlier_factor'] = -lof_clf._decision_function(predict.iloc[:, :-1])
    return predict


def plot_lof(result, method):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.figure(figsize=(8, 4)).add_subplot(111)
    plt.scatter(result[result['local_outlier_factor'] > method].index,
                result[result['local_outlier_factor'] > method]['local_outlier_factor'], c='red', s=50,
                marker='.', alpha=None,
                label='离群点')
    plt.scatter(result[result['local_outlier_factor'] <= method].index,
                result[result['local_outlier_factor'] <= method]['local_outlier_factor'], c='black', s=50,
                marker='.', alpha=None, label='正常点')
    plt.hlines(method, -2, 2 + max(result.index), linestyles='--')
    plt.xlim(-2, 2 + max(result.index))
    plt.title('LOF局部离群点检测', fontsize=13)
    plt.ylabel('局部离群因子', fontsize=15)
    plt.legend()
    plt.show()


def lof(data, predict=None, k=5, method=1, plot=False):
    import pandas as pd
    # 判断是否传入测试数据，若没有传入则测试数据赋值为训练数据
    try:
        if predict == None:
            predict = data.copy()
    except Exception:
        pass
    predict = pd.DataFrame(predict)
    # 计算 LOF 离群因子
    predict = localoutlierfactor(data, predict, k)
    if plot == True:
        plot_lof(predict, method)
    # 根据阈值划分离群点与正常点
    outliers = predict[predict['local_outlier_factor'] > method].sort_values(by='local_outlier_factor')
    inliers = predict[predict['local_outlier_factor'] <= method].sort_values(by='local_outlier_factor')
    return outliers, inliers


if __name__ == '__main__':
    file_path = "Ionosphere.txt"
    final_data_list = read_file_data(file_path)
    final_data_df = pd.DataFrame(final_data_list)
    ###
    # 获取任务密度，取第5邻域，阈值为4（LOF大于4认为是离群值）
    outliers1, inliers1 = lof(final_data_df, k=5, method=0.1)  # 这里的第k邻域可根据需要自由调整
    print("outliers1=", outliers1)
    # print("outliers1.iloc[[1, 3]]=", outliers1.iloc[[1, 3]])
    pca = PCA(n_components=2)
    new_pca = pd.DataFrame(pca.fit_transform(final_data_df))
    # print("type(new_pca)=", type(new_pca))
    ###
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.scatter(np.array(new_pca)[:, 0], np.array(new_pca)[:, 1], s=10, c='b', alpha=0.5)
    ###
    # new_pca['k_distances'] = list(final_data_df['k_distances'])
    data_outliers1 = outliers1.drop(['k_distances', 'local_outlier_factor'], axis=1)
    # print("outliers1=", outliers1)
    # data_outliers1.to_excel("lof_outliers_node.xlsx", sheet_name="lof_outliers", \
    #                         encoding='utf-8')    # 将采用离群因子而检测到的异常点保存到数据表中
    # print("type(data_outliers1)=", type(data_outliers1))
    data_outliers1.reset_index(drop=True, inplace=True)
    pca = PCA(n_components=2)
    data_outliers1_pca = pd.DataFrame(pca.fit_transform(data_outliers1))
    data_outliers1_pca['k_distances'] = list(outliers1['k_distances'])
    data_outliers1_pca['local_outlier_factor'] = list(outliers1['local_outlier_factor'])
    ###
    # 用红色圆圈绘制异常点
    plt.scatter(data_outliers1_pca[0], data_outliers1_pca[1], s=10+outliers1['local_outlier_factor']*100, c='r', alpha=0.2)
    plt.title("离群因子异常点检测")
    plt.show()
    ##
    print("outliers1=", outliers1)


