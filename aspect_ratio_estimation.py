# -*- coding: utf-8 -*-
####################################################
#   时间: 2020.05.01
#   功能: 计算EAR和MAR
#   使用说明: 自动调用, 无序操作.
####################################################
import numpy as np
from scipy.spatial import distance as dist

class aspect_ratio_estimation:
    def __init__(self):
        pass

    def eye_aspect_ratio(self,eye):
        # 垂直眼标志（X，Y）坐标
        A = dist.euclidean(eye[1], eye[5])  # 计算两个集合之间的欧式距离
        B = dist.euclidean(eye[2], eye[4])
        # 计算水平之间的欧几里得距离
        # 水平眼标志（X，Y）坐标
        C = dist.euclidean(eye[0], eye[3])
        # 眼睛长宽比的计算
        EAR = (A + B) / (2.0 * C)  # 椭圆面积公式为（A+B）/2 * C
        # 返回眼睛的长宽比
        return EAR

    def mouth_aspect_ratio(self,mouth):  # 嘴部
        A = np.linalg.norm(mouth[2] - mouth[9])  # 51, 59
        B = np.linalg.norm(mouth[4] - mouth[7])  # 53, 57
        C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
        MAR = (A + B) / (2.0 * C)
        return MAR