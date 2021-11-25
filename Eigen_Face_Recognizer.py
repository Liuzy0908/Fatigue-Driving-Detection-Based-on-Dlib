# -*- coding: utf-8 -*-
####################################################
#   作者: 刘朝阳
#   时间: 2020.05.01
#   更新时间: 2021.11.25
#   功能: 利用采集好的驾驶人的人脸图像, 进行特征脸模型的训练.
#   使用说明: 自动调用, 无需操作.
####################################################
import os
import cv2
import numpy as np

face_path = './face_path'

def LoadImages(data):  # data:训练数据所在的目录，要求图片尺寸一样。需要自己创建好后指定
    images = []
    names = []
    labels = []
    label = 0

    # 遍历所有文件夹
    for subdir in os.listdir(data):                         # os.listdir()输出该目录下的所有文件名字
        subpath = os.path.join(data, subdir)                # 拼接路径，定位到子文件夹路径
        if os.path.isdir(subpath):                          # 如果子文件夹路径存在
            # 在每一个文件夹中存放着一个人的许多照片
            names.append(subdir)                            # 把每个文件夹的名字 当成每个驾驶人名字
            # 遍历文件夹中的图片文件
            for filename in os.listdir(subpath):            # os.listdir()输出该目录下的所有文件名字
                imgpath = os.path.join(subpath, filename)   # 连接路径，定位到子文件夹路径
                img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                images.append(gray_img)                     # 把该文件夹下人的灰度图像全放在一起，形成列表
                labels.append(label)                        # 给该文件夹下的人打上标签
            label += 1                                      # label 为计数该人采集了多少张脸的数据
        else:
            raise Exception("还未正确采集某个驾驶人图像数据")
    images = np.asarray(images)  # 将数据列表矩阵化，形成一张张单独的图片
    names = np.asarray(names)
    labels = np.asarray(labels)  # 将数据列表矩阵化
    # 返回值：images:[m,height,width]  m为样本数，height为高，width为宽；
    # names：名字的集合；
    # labels：标签集合.
    return images, labels, names

print('/*/*/*/*/*/*/* 特征脸识别器正在训练 /*/*/*/*/*/*/*')
X, y, names = LoadImages(face_path)                      # 加载图像数据
Eigen_Face_Model = cv2.face.EigenFaceRecognizer_create()
Eigen_Face_Model.train(X, y)
print('-*-*-*-*-*-*-* 特征脸识别器训练完成 -*-*-*-*-*-*-*')