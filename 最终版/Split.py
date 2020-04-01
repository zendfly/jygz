import os
import json
import cv2
import numpy as np
import sys
from optparse import OptionParser

"""
    def split_img_with_via_label()，在有json文件下情况下对图片进行裁图，并对json进行修改
    并保存图片和jeson文件
    
    def split_single_img_detection()，没有json文件，直接使用threshold进行裁图
    返回裁图区域和图片
    
    def decode_coor()，对检测出来的box坐标，还原到原图坐标
"""
class Split():

    def __init__(self):
        pass


    def decode_coor(self,cuted_area,detection_box):
        """
        :param cuted_area:切图的区域，[[xmin,xmax],[xmin,xmax]]
        :param detection_box: 检测的结果，[[],[],...]
        :return: 原图上的坐标
        """
        area_num = len(cuted_area)

        if area_num > 1:
            fist_area_width = cuted_area[0][1] - cuted_area[0][0]       # 第一个区域的宽
            for i in range(len(detection_box)):
                if detection_box[i][0] < fist_area_width:
                    detection_box[i][0] = detection_box[i][0] + cuted_area[0]
                else:
                    detection_box[i][0] = detection_box[i][0] + cuted_area[1]
        else:
            for i in range(len(detection_box)):
                detection_box[i][0] = detection_box[i][0] + cuted_area[0]

        return detection_box


    def split_single_img_detection(self, img_path, save_path, threshold=None, extend=40):
        """
        :param img_path:
        :param save_path:
        :param threshold: 分割的阈值
        :param extend: 在计算出的边界上再扩展的宽度
        :return:
            coorlist：切割区域的左右边界,[[x_1,y_1],[x_2,y_2]];
            shsplit_img：图片
        """

        coorlist = []
        img_name = os.path.basename(img_path)
        # print("{} spliting ".format(img_name))
        img_array = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        height, width = img_array.shape[:2]
        if height < width:
            img_array = np.transpose(img_array)

        img_sum = np.sum(img_array, axis=0)
        sum_max_value = max(img_sum)
        if not threshold:
            threshold = sum_max_value / 3

        index = np.where(img_sum > threshold)
        a = index[0][1:] - index[0][0:-1]
        center_left_index = np.where(a > 10)[0]  # 判断是否存在两组区域，以10作为阈值
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

        # c = [index[0][i] for i in center_left_index]
        xmin = min(new_index[0])
        xmax = max(new_index[0])
        # split_img = None

        if new_center_left_index > 1:
            center_right_index = new_center_left_index + 1
            xmax_left = index[0][new_center_left_index]
            xmin_right = index[0][center_right_index]
            coorlist.append([xmin - extend, xmax_left[0] + extend])
            coorlist.append([xmin_right[0] - extend, xmax + extend])
            split_img = img_array[:, [i for i in range(coorlist[0][0], coorlist[0][1], 1)] +
                                     [i for i in range(coorlist[1][0], coorlist[1][1], 1)]]  # save img

            cv2.imwrite(os.path.join(save_path, img_name), split_img)
            print(' %s success save in %s'%(img_name,os.path.join(save_path, img_name)))
            return coorlist,split_img
        else:
            coorlist.append([xmin - extend, xmax + extend])
            split_img = img_array[:, coorlist[0][0] - extend:coorlist[0][1] + extend]

            cv2.imwrite(os.path.join(save_path, img_name), split_img)
            print(' %s success save in %s' % (img_name, os.path.join(save_path, img_name)))
            return coorlist,split_img


    def split_single_img_train(self, img_array, threshold=None):
        """
        :param img_array:
        :param threshold:
        :return:
            coorlist：切割区域的左右边界,[[x_1,y_1],[x_2,y_2]];
            shsplit_img.shape：切割出的区域尺寸，即切割图像的shape，[w,h]
        """
        coorlist = []
        height, width = img_array.shape[:2]
        if height < width:
            img_array = np.transpose(img_array)

        img_sum = np.sum(img_array, axis=0)
        sum_max_value = max(img_sum)
        if not threshold:
            threshold = sum_max_value / 3

        index = np.where(img_sum > threshold)
        xmin = min(index[0])
        xmax = max(index[0])
        a = index[0][1:] - index[0][0:-1]
        center_left_index = np.where(a > 10)[0]  # 判断是否存在两组区域，以10作为阈值
        if center_left_index > 1:
            center_right_index = center_left_index + 1
            xmax_left = index[0][center_left_index]
            xmin_right = index[0][center_right_index]
            coorlist.append([xmin, xmax_left[0]])
            coorlist.append([xmin_right[0], xmax])
        else:
            coorlist.append([xmin, xmax])

        split_img = img_array[:, index[0]]

        return coorlist, split_img.shape


    def split_img_with_via_label(self,img_dir,via_label,save_split_dir,extend=40):
        """
        :param img_dir:图片地址
        :param via_label: label地址
        :param save_split_dir: save 地址
        :param extend: 扩展值
        :return:

        step_1:利用灰度值判断切图区域数量
        step_2:计算bbox中xmin和xmax
        step_3:修改json中的相关参数
        """

        if via_label:
            with open(via_label, "r", encoding="utf-8") as f:
                via_content = json.load(f)
            print("{} read success".format(via_label))
            images = via_content["images"]
            imgs = os.listdir(img_dir)

            for img in imgs:
                img_box_idx = []            # 存bbox坐标
                if img.endswith("jpg"):
                    # coor 裁图的区域
                    print("{} spliting ".format(os.path.join(img_dir, img)))
                    img_array = cv2.imdecode(np.fromfile(os.path.join(img_dir, img),dtype=np.uint8),-1)
                    coor,image_size = Split().split_single_img_train(img_array)
                    annotations = via_content["annotations"]

                    for ann in annotations:
                        img_name = images[int(ann["image_id"])]["file_name"]
                        if img_name == img:
                            bbox = ann["bbox"]  # box矩形坐标框[xmin,ymin,w,h]
                            img_box_idx.append(bbox)

                    if not img_box_idx:
                        raise ValueError('%s not find in jeson'%img)
                    sort_img_box_idx = np.sort(coor,axis=0)         # 按0轴进行排序

                    # 根据box坐标进行划分，并找到两个区域中各的min_x,max_x
                    mid_bound = 0
                    one_area_xmax = mid_bound
                    two_area_xmin = mid_bound
                    num_one_area = [sort_img_box_idx[0][0], one_area_xmax]
                    num_two_area = [two_area_xmin, sort_img_box_idx[-1][0]]

                    if len(coor) > 1:
                        mid_bound = (sort_img_box_idx[1][0] - sort_img_box_idx[0][0]) / 2 + sort_img_box_idx[0][1] # 切割区域的中线
                        num_one_area = [sort_img_box_idx[0][0], sort_img_box_idx[0][1]]
                        num_two_area = [sort_img_box_idx[1][0], sort_img_box_idx[1][1]]
                        for ann in annotations:
                            img_name = images[int(ann["image_id"])]["file_name"]
                            if img_name == img:
                                if ann['bbox'][0] < mid_bound:
                                    num_one_area[0] = sort_img_box_idx[0][0]
                                    num_one_area[1] = sort_img_box_idx[0][1]
                                    num_one_area[0] = min(ann['bbox'][0], num_one_area[0])   # 左边界
                                    num_one_area[1] = max(ann['bbox'][0] + ann['bbox'][2], num_one_area[1])     # 找第一个区域的右边界
                                else:
                                    num_two_area[0] = sort_img_box_idx[1][0]
                                    num_two_area[1] = sort_img_box_idx[1][1]
                                    num_two_area[0] = min(ann['bbox'][0], num_one_area[0])  # 左边界
                                    num_two_area[1] = max(ann['bbox'][1] + ann['bbox'][2], num_one_area[1])     # 找第二个区域的右边界

                    images_valu = None
                    cuted_file_name = None
                    # 重写bbox、segmentation、images_filename、width
                    for ann in annotations:
                        img_name = images[int(ann["image_id"])]["file_name"]
                        if img_name == img:
                            images_valu = images[int(ann["image_id"])]
                            if len(coor) > 1:
                                # 左右区域进行扩展
                                new_num_one_area = [num_one_area[0] - extend,num_one_area[1] + extend]
                                new_num_two_area = [num_two_area[0] - extend,num_two_area[1] + extend]
                                if ann['bbox'][0] < mid_bound:      # 以两个区域间隔的中线进行区分box的归属
                                    ann['bbox'][0] = int(ann['bbox'][0] - new_num_one_area[0])
                                else:
                                    ann['bbox'][0] = int(ann['bbox'][0] - new_num_two_area[0])

                                ann['segmentation'] = [ann['bbox'][0], ann['bbox'][1],
                                                       int(ann['bbox'][0] + ann['bbox'][2]), ann['bbox'][1],
                                                       int(ann['bbox'][0] + ann['bbox'][2]),int(ann['bbox'][1] + ann['bbox'][3]),
                                                       ann['bbox'][0], int(ann['bbox'][1] + ann['bbox'][3])]
                                images[int(ann["image_id"])]['width'] = int(new_num_one_area[1] - new_num_one_area[0] +
                                                                              new_num_two_area[1] - new_num_two_area[0])
                                cuted_file_name = '%s.jpg' % (
                                            img_name.split('.')[0] + '_' + str(new_num_one_area[0]) +
                                            '_' + str(new_num_two_area[0]))

                                # 切图
                                split_img = img_array[:, [i for i in range(new_num_one_area[0], new_num_one_area[1])] +
                                                         [i for i in
                                                          range(new_num_two_area[0], new_num_two_area[1])]]  # save img
                                cv2.imwrite(os.path.join(save_split_dir, cuted_file_name), split_img)
                            else:
                                new_area = [min(num_one_area[0],coor[0][0]) - extend, max(num_one_area[1],coor[0][1]) + extend]
                                ann['bbox'][0] = int(ann['bbox'][0] - new_area[0])
                                ann['segmentation'] = [ann['bbox'][0], ann['bbox'][1],
                                                       int(ann['bbox'][0] + ann['bbox'][2]), ann['bbox'][1],
                                                       int(ann['bbox'][0] + ann['bbox'][2]),int(ann['bbox'][1] + ann['bbox'][3]),
                                                       ann['bbox'][0], int(ann['bbox'][1] + ann['bbox'][3])]
                                images[int(ann["image_id"])]['width'] = int(new_area[1] - new_area[0])

                                cuted_file_name = '%s.jpg'%(img_name.split('.')[0] + '_' + str(new_area[0]))

                                # 切图
                                split_img = img_array[:, [i for i in range(new_area[0], new_area[1])]]  # save img
                                cv2.imwrite(os.path.join(save_split_dir, cuted_file_name), split_img)
                    images_valu['file_name'] = cuted_file_name
            write_json = json.dumps(via_content)
            with open(os.path.join(save_split_dir,('%s'%os.path.basename(via_label))),'w') as op:
                op.write(write_json)


if __name__ == '__main__':

    # eg python py_file.py 'ima_path' 'json_path' 'save_path'
    # img_path = sys.argv[1]
    # save_path = sys.argv[3]
    # json_path = sys.argv[2]
    #
    # start Split images
    # Split().split_img_with_via_label(img_path,json_path,save_path)

    img_path = ''
    save_path = ''
    json_path = ''

    Split().split_img_with_via_label(img_path,json_path,save_path)
