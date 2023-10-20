#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/19
# @Author  : jc
# @File    : K-means.py
import numpy as np
import matplotlib.pyplot as plt
import collections


def classification(k, data_length):
    class_list = np.array([-1 for _ in range(data_length)])
    class_list_temp = np.array([-2 for _ in range(data_length)])
    if isinstance(k, int):
        classification_center = np.random.choice(a=np.arange(data_length), size=k)
    else:
        classification_center = k
    center_coordinate = raw_data[classification_center]
    while not (class_list == class_list_temp).all():
        class_list[:] = class_list_temp[:]
        for point in range(data_length):
            distance_with_center = np.sqrt(np.sum(np.power(center_coordinate - raw_data[point], 2), axis=1))
            class_list_temp[point] = int(np.argwhere(distance_with_center == np.min(distance_with_center)))
        for i in range(k):
            center_coordinate[i] = np.mean(raw_data[np.where(class_list_temp == i)], axis=0)
    return class_list, center_coordinate


# 画最终聚类效果图
def show_result(class_list, raw_data, center_coordinate, name):
    colors = [
              '#FF0000', '#FFA500', '#FFFF00', '#00FF00', '#228B22',
              '#0000FF', '#FF1493', '#EE82EE', '#000000', '#FFA500',
              '#00FF00', '#006400', '#00FFFF', '#0000FF', '#FFFACD',
              ]
    use_color = {}
    total_color = list(dict(collections.Counter(class_list)).keys())
    for index, i in enumerate(total_color):
        use_color[i] = index
    plt.figure(num=1, figsize=(15, 10))
    for index, point in enumerate(class_list):
        plt.scatter(x=raw_data[index, 0], y=raw_data[index, 1], c=colors[use_color[point]], s=50, marker='o', alpha=0.9)
    plt.scatter(x=center_coordinate[:, 0], y=center_coordinate[:, 1], c='b', s=200, marker='+', alpha=0.8)
    plt.title('The Result Of Cluster')
    # plt.show()
    plt.savefig('../result/K-means_Result_' + name + '.jpg' )


def main(raw_data, name, k):
    data_length = len(raw_data)
    class_list, center_coordinate = classification(k, data_length)
    show_result(class_list, raw_data, center_coordinate, name)

if __name__ == '__main__':
    k = 2
    name = 'Jain'
    data_path = r'../data/Jain.txt'
    raw_data = np.loadtxt(data_path, delimiter='	', usecols=[0, 1])
    main(raw_data, name, k)

