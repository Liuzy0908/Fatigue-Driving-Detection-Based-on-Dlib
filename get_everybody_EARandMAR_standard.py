# -*- coding: utf-8 -*-
####################################################
#   作者: 刘朝阳
#   时间: 2020.05.01
#   更新时间: 2021.11.25
#   功能: 在计算PERCLOS时, 需要知道驾驶在正常情况下的眼睛开度, 来作为基准计算
#   使用说明: 自动调用, 无需操作
####################################################
import os
import numpy as np
import cv2
import dlib

from imutils import face_utils
from head_posture_estimation import head_posture_estimation
from aspect_ratio_estimation import aspect_ratio_estimation

HPE = head_posture_estimation()
ARE = aspect_ratio_estimation()

# 使用dlib.get_frontal_face_detector() 获得脸部位置检测器
detector = dlib.get_frontal_face_detector()
# 使用dlib.shape_predictor获得脸部特征位置检测器
predictor = dlib.shape_predictor('shape_predictor_68_face_landMARks.dat')

# 分别获取左右眼面部标志的索引
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

EAR = everybody_EAR_mean =[]
EAR_all_per_person = []
EAR_all_per_person_open = []
pitch_all_per_person = []
pitch_mean_per_person = []
everybody_pitch_mean = []
everybody_EAR_min = []

def get_everybody_EARandMAR_standard(face_path):
    # 遍历每个人的所有图片，提取出眼睛的平均高度
    for subdir in os.listdir(face_path):                            # os.listdir()输出该目录下的所有文件名字 到了lzy文件夹的面前（未进去）
        EAR_all_per_person_open = EAR_all_per_person = []
        subpath = os.path.join(face_path, subdir)                   # 连接路径，定位到子文件夹路径 到了lzy文件夹的面前（未进去）
        if os.path.isdir(subpath):                              # 如果子文件夹路径存在
            for filename in os.listdir(subpath):                # os.listdir(subpath)输出该目录下的所有文件名字 lzy进入，到了1、2、3.png了，然后对每一张进行处理
                EAR_mean_per_person = EAR_min_per_person = []
                imgpath = os.path.join(subpath, filename)       # 连接路径，定位到子文件夹路径
                img = cv2.imread(imgpath, cv2.IMREAD_COLOR)     # 读1.png
                grayimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                faces = detector(grayimg, 0)
                for k, d in enumerate(faces):                   # 找出每张图片上的人脸 #一个图片上人脸数就1，所以看作没有这句就行
                    shape = predictor(grayimg, d)
                    shape_array = face_utils.shape_to_np(shape)
                    leftEye = shape_array[lStart:lEnd]
                    rightEye = shape_array[rStart:rEnd]

                    reprojectdst, euler_angle, pitch, roll, yaw = HPE.get_head_pose(shape_array)  # 重新投影，欧拉角
                    pitch_all_per_person.append(pitch)

                    leftEAR = ARE.eye_aspect_ratio(leftEye)
                    rightEAR = ARE.eye_aspect_ratio(rightEye)
                    EAR = (leftEAR + rightEAR) / 2.0
                    EAR_all_per_person.append(EAR)

                    # for完全进行完毕后，把文件下的所有眼睛高度存入了
                    if EAR > 0.13 and EAR < 0.23:               # 防止闭眼时为0而拉低整体睁眼值 阈值由经验尝试得出
                        EAR_all_per_person_open.append(EAR)     # 把每张图片的高度值放在一起，形成该人所有图片的高度值集合

            pitch_mean_per_person = np.mean(pitch_all_per_person)
            EAR_mean_per_person = np.mean(EAR_all_per_person_open)  # 算lzy眼睛高度的平均值
            EAR_min_per_person = np.min(EAR_all_per_person)
        everybody_pitch_mean.append(pitch_mean_per_person)
        everybody_EAR_mean.append(EAR_mean_per_person)          # 把每个人眼睛的平均值记录
        everybody_EAR_min.append(EAR_min_per_person)
    return everybody_EAR_mean, everybody_EAR_min, everybody_pitch_mean

