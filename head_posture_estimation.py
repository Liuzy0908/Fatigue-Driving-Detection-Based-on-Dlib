# -*- coding: utf-8 -*-
####################################################
#   时间: 2020.05.01
#   功能: 利用Dlib中的68人脸特征关键点, 相机内外参数, 和其内置的3D头部模型, 计算出头部的三种姿态角度
#   使用说明:
####################################################
import numpy as np
import math
import cv2

class head_posture_estimation:
    def __init__(self):
        # 世界坐标系(UVW)：填写3D参考点，该模型参考http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp
        self.object_pts = np.float32([[6.825897, 6.760612, 4.402142],  # 33左眉左上角
                                 [1.330353, 7.122144, 6.903745],  # 29左眉右角
                                 [-1.330353, 7.122144, 6.903745],  # 34右眉左角
                                 [-6.825897, 6.760612, 4.402142],  # 38右眉右上角
                                 [5.311432, 5.485328, 3.987654],  # 13左眼左上角
                                 [1.789930, 5.393625, 4.413414],  # 17左眼右上角
                                 [-1.789930, 5.393625, 4.413414],  # 25右眼左上角
                                 [-5.311432, 5.485328, 3.987654],  # 21右眼右上角
                                 [2.005628, 1.409845, 6.165652],  # 55鼻子左上角
                                 [-2.005628, 1.409845, 6.165652],  # 49鼻子右上角
                                 [2.774015, -2.080775, 5.048531],  # 43嘴左上角
                                 [-2.774015, -2.080775, 5.048531],  # 39嘴右上角
                                 [0.000000, -3.116408, 6.097667],  # 45嘴中央下角
                                 [0.000000, -7.415691, 4.070434]])  # 6下巴角

        # 相机坐标系(XYZ)：添加相机内参
        self.K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
             0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
             0.0, 0.0, 1.0]  # 等价于矩阵[fx, 0, cx; 0, fy, cy; 0, 0, 1]
        # 图像中心坐标系(uv)：相机畸变参数[k1, k2, p1, p2, k3] 径向畸变参数：k1,k2,k3；切向畸变系数p1,p2
        self.D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

        # 像素坐标系(xy)：填写凸轮的本征和畸变系数
        self.cam_matrix = np.array(self.K).reshape(3, 3).astype(np.float32)
        self.dist_coeffs = np.array(self.D).reshape(5, 1).astype(np.float32)

        # 重新投影3D点的世界坐标轴以验证结果姿势
        self.reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                                   [10.0, 10.0, -10.0],
                                   [10.0, -10.0, -10.0],
                                   [10.0, -10.0, 10.0],
                                   [-10.0, 10.0, 10.0],
                                   [-10.0, 10.0, -10.0],
                                   [-10.0, -10.0, -10.0],
                                   [-10.0, -10.0, 10.0]])

    def get_head_pose(self,shape):  # 头部姿态估计
        # （像素坐标集合）填写2D参考点，注释遵循https://ibug.doc.ic.ac.uk/resources/300-W/
        # 17左眉左上角/21左眉右角/22右眉左上角/26右眉右上角/36左眼左上角/39左眼右上角/42右眼左上角/
        # 45右眼右上角/31鼻子左上角/35鼻子右上角/48左上角/54嘴右上角/57嘴中央下角/8下巴角
        image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                                shape[39], shape[42], shape[45], shape[31], shape[35],
                                shape[48], shape[54], shape[57], shape[8]])
        # solvePnP计算姿势——求解旋转和平移矩阵：
        # rotation_vec表示旋转矩阵，translation_vec表示平移矩阵，cam_matrix与K矩阵对应，dist_coeffs与D矩阵对应。
        _, rotation_vec, translation_vec = cv2.solvePnP(self.object_pts, image_pts, self.cam_matrix, self.dist_coeffs)
        # projectPoints重新投影误差：原2d点和重投影2d点的距离（输入3d点、相机内参、相机畸变、r、t，输出重投影2d点）
        reprojectdst, _ = cv2.projectPoints(self.reprojectsrc, rotation_vec, translation_vec, self.cam_matrix, self.dist_coeffs)
        reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))  # 以8行2列显示

        # 计算欧拉角calc euler angle
        # 参考https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#decomposeprojectionmatrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)  # 罗德里格斯公式（将旋转矩阵转换为旋转向量）
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))  # 水平拼接，vconcat垂直拼接
        # decomposeProjectionMatrix将投影矩阵分解为旋转矩阵和相机矩阵
        _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

        pitch, yaw, roll = [math.radians(_) for _ in euler_angle]

        pitch = math.degrees(pitch)
        roll =  -math.degrees(roll)
        yaw =   math.degrees(yaw)

        return reprojectdst, euler_angle, pitch, roll, yaw  # 投影误差，欧拉角