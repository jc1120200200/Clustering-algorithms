#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/19
# @Author  : jc
# @File    : K-means++.py
import numpy as np
import collections
import matplotlib.pyplot as plt


def alias_setup(probs):
    """
    probs： 某个概率分布
    返回: Alias数组与Prob数组
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int64)
    # Sort the data into the outcomes with probabilities
    # that are larger and smaller than 1/K.
    smaller = []
    larger = []
    for i, prob in enumerate(probs):
        q[i] = K * prob  # 概率
        if q[i] < 1.0:
            smaller.append(i)
        else:
            larger.append(i)

    # Loop though and create little binary mixtures that
    # appropriately allocate the larger outcomes over the
    # overall uniform mixture.
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] - (1.0 - q[small])

        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    '''
    输入: Prob数组和Alias数组
    输出: 一次采样结果
    '''
    K = len(J)
    # Draw from the overall uniform mixture.
    k = int(np.floor(np.random.rand() * K))  # 随机取一列

    # Draw from the binary mixture, either keeping the
    # small one, or choosing the associated larger one.
    if np.random.rand() < q[k]:
        return k
    else:
        return J[k]


def alias_sample(probs, samples):
    assert isinstance(samples, int), 'Samples must be a integer.'
    sample_result = []
    J, p = alias_setup(probs)
    for i in range(samples):
        sample_result.append(alias_draw(J, p))
    return sample_result


def choose_centers(raw_data, k):
    center_ids = [np.random.choice(len(raw_data), size=1)]
    dist_mat = np.empty(shape=[len(raw_data), len(raw_data)])
    for i in range(len(raw_data)):
        for j in range(len(raw_data)):
            if i == j:
                dist_mat[i, j] = 0.
            elif i < j:
                dist_mat[i, j] = np.mean(np.square(raw_data[i] - raw_data[j]))
            else:
                dist_mat[i, j] = dist_mat[j, i]
    while len(center_ids) < k:
        nodes_min_dist = np.min(dist_mat[:, center_ids], axis=1)
        probs = nodes_min_dist / np.sum(nodes_min_dist)
        center_ids.append(alias_sample(probs.reshape(-1), 1))
    center_ids = np.array(center_ids).reshape(-1)
    return center_ids


def do_cluster(raw_data, centers):
    dist = []
    for center in centers:
        dist.append(np.mean(np.square(raw_data - center), axis=1))
    dist = np.vstack(dist)
    classes = np.argmin(dist, axis=0)
    return classes

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
    plt.figure(num=1, figsize=(16, 9))
    for index, point in enumerate(class_list):
        plt.scatter(x=raw_data[index, 0], y=raw_data[index, 1], c=colors[use_color[point]], s=50, marker='o', alpha=0.9)
    plt.scatter(x=center_coordinate[:, 0], y=center_coordinate[:, 1], c='b', s=200, marker='+', alpha=0.8)
    plt.title('The Result Of Cluster')
    # plt.show()
    plt.savefig('../result/K-means++_Result_' + name + '.jpg' )


def main(raw_data, name, k):
    if not isinstance(raw_data, np.ndarray):
        raw_data = np.array(raw_data)
    center_ids = choose_centers(raw_data, k)
    centers = raw_data[center_ids]
    classes_before = np.arange(len(raw_data))
    while True:
        classes_after = do_cluster(raw_data, centers)
        if (classes_before == classes_after).all():
            break

        classes_before = classes_after
        for c in range(k):
            data_c = raw_data[np.argwhere(classes_after == c)]
            center_c = np.mean(data_c, axis=0)
            centers[c] = center_c
    show_result(classes_after, raw_data, centers, name)


if __name__ == '__main__':
    k = 2
    name = 'Jain'
    data_path = r'../data/Jain.txt'
    raw_data = np.loadtxt(data_path, delimiter='	', usecols=[0, 1])
    main(raw_data, name, k)
