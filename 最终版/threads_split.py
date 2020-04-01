# -*- encoding: utf-8 -*-
"""
@File    : split_img.py
@Time    : 2020/3/26 14:28
@Author  : XXX
@Email   : XXXX@fitow-alpha.com
@Software: PyCharm
# Copyright 2020 The Fitow Authors. All Rights Reserved.

在有txt的情况下，使用多线程切图，目前线程为数为10
"""

import cv2
import json
import numpy as np
import os
import threading
from queue import Queue
import time

class SplitImg():

    def split_single_img_train(self, img_array, threshold=None):
        """
        :param img_path:图片路径
        :param threshold: 切割阈值，默认为sum_max_value/3
        :param save_split_dir: 是否保存
        :return:
            coorlist：切割区域的左右边界,[[x_1,y_1],[x_2,y_2]];
            shsplit_img.shape：切割出的区域尺寸，即切割图像的shape，[w,h]
        """

        coorlist = []
        height,width = img_array.shape[:2]
        if height<width:
            img_array = np.transpose(img_array)

        img_sum = np.sum(img_array,axis=0)
        sum_max_value = max(img_sum)
        if not threshold:
            threshold = sum_max_value/3

        index = np.where(img_sum>threshold)
        xmin = min(index[0])
        xmax = max(index[0])
        a = index[0][1:] - index[0][0:-1]
        center_left_index = np.where(a > 10)[0]    # 判断是否存在两组区域，以10作为阈值
        len_center = len(center_left_index)
        new_center_left_index = center_left_index
        new_index = index
        """
        当存在大于2个区域时,说明threshold对图片分割达到2个区域以上，应减小阈值，已达到最多两个区域
        tips:
            jygz图片中，最多只能分出两个区域，当大于2时说明滚子切面部分亮度起伏较大
            在这种情况下，阈值使用不恰当就会导致分出额外的区域，故，降低阈值，每次降低80%，直到小于2才停止
        """
        while len_center > 1:
            new_threshold = threshold * 0.8
            new_index = np.where(img_sum > (new_threshold))
            new_a = new_index[0][1:] - new_index[0][0:-1]
            new_center_left_index = np.where(new_a > 10)[0]
            len_center = len(new_center_left_index)

        xmin = min(new_index[0])
        xmax = max(new_index[0])
        if new_center_left_index > 1:
            center_right_index = center_left_index + 1
            xmax_left = index[0][center_left_index]
            xmin_right = index[0][center_right_index]
            coorlist.append([xmin,xmax_left[0]])
            coorlist.append([xmin_right[0],xmax])
        else:
            coorlist.append([xmin,xmax])

        split_img = img_array[:, index[0]]

        return coorlist,split_img.shape

    def split_img_with_txt_label(self,img_dir,txt_dir,save_split_dir,extend=40):
        """
        :param img_dir: 图片的完整路径
        :param txt_dir: 图片对应的txt文本的完整路径
        :param save_split_dir: 保存路径
        :param extend: 扩展值
        :return: 将裁后的图片以及对应的txt保存在save_split_dir路径下
        """

        new_txt_dir = []
        ima_name = os.path.basename(img_dir)
        img = cv2.imread(img_dir,0)
        coor,img_size = SplitImg().split_single_img_train(img)
        sort_img_box_idx = np.sort(coor, axis=0)  # 按0轴进行排序
        with open(txt_dir) as f:
            result_list = json.load(f)

        save_img_name = None
        if len(coor) > 1:
            mid_bound = (sort_img_box_idx[1][0] - sort_img_box_idx[0][0]) / 2 + sort_img_box_idx[0][1] # 切割区域的中线
            num_one_area = [sort_img_box_idx[0][0], sort_img_box_idx[0][1]]
            num_two_area = [sort_img_box_idx[1][0], sort_img_box_idx[1][1]]
            for i in result_list:
                if int(i['x']) < mid_bound:
                    num_one_area[0] = min(i['x'],num_one_area[0])
                    num_one_area[1] = max(i['x'] + i['w'],num_one_area[1])
                else:
                    num_two_area[0] = min(i['x'],num_two_area[0])
                    num_two_area[1] = max(i['x'] + i['w'],num_two_area[1])
            num_one_area = [num_one_area[0] - extend,num_one_area[1] + extend]
            num_two_area = [num_two_area[0] - extend,num_two_area[1] + extend]

            save_img_name = '%s_%s_%s.jpg'%(ima_name.split('.')[0],num_one_area[0],num_two_area[0])
            save_txt_name = '%s.txt' % save_img_name
            splid_img = img[:,[i for i in range(num_one_area[0],num_one_area[1])] +
                             [i for i in range(num_two_area[0],num_two_area[1])]]

            for ii in result_list:
                new_ii = ii
                if ii['x'] < num_two_area[0]:
                    new_ii['x'] = int(ii['x'] - num_one_area[0])
                    new_txt_dir.append(new_ii)
                else:
                    new_ii['x'] = int(ii['x'] - num_two_area[1] + num_one_area[1] - num_one_area[0])
                    new_txt_dir.append(new_ii)
        else:
            num_area = [coor[0][0],coor[0][1]]
            for ii in result_list:
                num_area[0] = min(ii['x'],num_area[0])
                num_area[1] = max(ii['x'] + ii['w'],num_area[1])

            num_area=[num_area[0] - extend, num_area[1] + extend]
            for ii in result_list:
                new_ii = ii
                new_ii['x'] = int(ii['x'] - num_area[0])
                new_txt_dir.append(new_ii)

            save_img_name = '%s_%s.jpg'%(ima_name.split('.')[0],coor[0][0])
            save_txt_name = '%s.txt'%save_img_name
            splid_img = img[:,[i for i in range(num_area[0],num_area[1])]]


        with open(os.path.join(save_split_dir,save_txt_name),'w') as f:
            f.write(json.dumps(new_txt_dir))
        cv2.imwrite(os.path.join(save_split_dir,save_img_name),splid_img)       # 保存图

if __name__=="__main__":

    img_dir =r"D:\jygz\txt_split\img"          # image path
    save_split_dir = r"D:\jygz\txt_split/cuted_img"     # save path
    num_threads = 10        # threahs number
    therads = []
    splider = SplitImg()
    print('start')
    start_time = time.time()
    img_list = os.listdir(img_dir)

    queue = Queue()
    def worker(img_dir,save_dir):       # 创建子线程
        while True:
            item = queue.get()
            if item is None:
                break
            txt_name = '%s.txt'%item
            splider.split_img_with_txt_label(os.path.join(img_dir,item),
                                             os.path.join(img_dir,txt_name),
                                             save_dir)
            queue.task_done()

    for i in img_list:      # 创建队列
        if i.endswith('.jpg'):
            queue.put(i)


    for i in range(num_threads):
        t = threading.Thread(target=worker,args=(img_dir,save_split_dir,))
        t.start()
        therads.append(t)

    queue.join()        # 等待queue中的item被调用和执行完
    # 线程停止，给队列中添加num_threads个None，
    for i in range(num_threads):
        queue.put(None)
    for t in therads:
        t.join()

    print(time.time() - start_time)


    # for i in img_list:
    #     if i.endswith('.jpg'):
    #         # print(i)
    #         txt_name = '%s.txt'%i
    #         img_file_path = os.path.join(img_dir,i)
    #         txt_file_path = os.path.join(img_dir,txt_name)
    #         splider.split_img_with_txt_label(img_file_path,txt_file_path,save_split_dir)
    #
    # print(time.time() - start_time)
