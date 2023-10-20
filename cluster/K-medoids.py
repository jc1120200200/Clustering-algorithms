#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/19
# @Author  : jc
# @File    : K-means.py
import matplotlib.pyplot as plt
import numpy as np
import random


def ou_distance(x, y):
    # 定义欧式距离的计算
    return np.sqrt(sum(np.square(x - y)))

def run_k_center(raw_data, k, func_of_dis):
    indexs = list(range(len(raw_data)))
    random.shuffle(indexs)  # 随机选择质心
    init_centroids_index = indexs[:k]
    centroids = raw_data[init_centroids_index, :]  # 初始中心点
    # 确定种类编号
    levels = list(range(k))
    sample_target = []
    if_stop = False
    while (not if_stop):
        if_stop = True
        classify_points = [[centroid] for centroid in centroids]
        sample_target = []
        # 遍历数据
        for sample in raw_data:
            # 计算距离，由距离该数据最近的核心，确定该点所属类别
            distances = [func_of_dis(sample, centroid) for centroid in centroids]
            cur_level = np.argmin(distances)
            sample_target.append(cur_level)
            # 统计，方便迭代完成后重新计算中间点
            classify_points[cur_level].append(sample)
        # 重新划分质心
        for i in range(k):  # 几类中分别寻找一个最优点
            distances = [func_of_dis(point_1, centroids[i]) for point_1 in classify_points[i]]
            now_distances = sum(distances)  # 首先计算出现在中心点和其他所有点的距离总和
            for point in classify_points[i]:
                distances = [func_of_dis(point_1, point) for point_1 in classify_points[i]]
                new_distance = sum(distances)
                # 计算出该聚簇中各个点与其他所有点的总和，若是有小于当前中心点的距离总和的，中心点去掉
                if new_distance < now_distances:
                    now_distances = new_distance
                    centroids[i] = point  # 换成该点
                    if_stop = False
    return sample_target, centroids

def main(raw_data, k, name):
    colors = [
        '#FF0000', '#FFA500', '#FFFF00', '#00FF00', '#228B22',
        '#0000FF', '#FF1493', '#EE82EE', '#000000', '#FFA500',
        '#00FF00', '#006400', '#00FFFF', '#0000FF', '#FFFACD',
    ]
    predict, center_coordinate = run_k_center(raw_data, k, ou_distance)
    for index, i in enumerate(predict):
        plt.scatter(x=raw_data[index, 0], y=raw_data[index, 1], c=colors[i], s=50, marker='o', alpha=0.9)
    plt.scatter(x=center_coordinate[:, 0], y=center_coordinate[:, 1], c='b', s=200, marker='+', alpha=0.8)
    plt.title('The Result Of Cluster')
    # plt.show()
    plt.savefig('../result/K-medoids_Result_' + name + '.jpg')

if __name__ == '__main__':
    k = 2
    name = 'Jain'
    data_path = r'../data/Jain.txt'
    raw_data = np.loadtxt(data_path, delimiter='	', usecols=[0, 1])
    main(raw_data, k, name)
