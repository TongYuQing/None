#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 16:23:53 2018
'''
#
#
将文件夹下的子文件夹中的图片路径及标签写入 txt文本
#
#
'''
@author: tong
"""
import os
import numpy as np
data_dir = '/home/tong/图片/Wallpapers/'
label_dir = '/home/tong/图片/Wallpapers/1.txt'
folders = os.listdir(data_dir)
subject_lst = [x for x in folders if os.path.isdir(os.path.join(data_dir, x))]
file_lst = [x for x in folders if os.path.isfile(os.path.join(data_dir, x))]
print('一共有{}个文件夹'.format(len(subject_lst)))
# print(os.path.basename('/home/kesci/input/CASIA-WebFace-Align-96/0001515/054.jpg'))
# print(os.path.join(data_dir,'4456'))
label = 0
with open(label_dir, 'w') as f:
    for i in subject_lst:
        picture_dir = data_dir+i
        picture_folder = os.listdir(picture_dir)
        picture_list = [x for x in picture_folder if os.path.isfile(os.path.join(picture_dir, x))]
        for j in picture_list:
            f.write('{}/{} {}\n'.format(i, j, label))
        label = label + 1
f.close()
