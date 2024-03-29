# Fatigue-Driving-Detection-Based-on-Dlib
# 项目概述
### 项目版本V1.0
> V1.0版本仅是PC端的驾驶人(实时)视频检测, 暂不涉及硬件控制. 硬件控制见V2.0版本(暂未发布)

本项目为本科毕业设计的相关工作, 于2020.05.01完工, 于2021.11.25整理发布

本项目仅是作为理论分析的简单验证工具和毕业设计的实践训练, 在运行过程中的误差很大,其效果远远达不到实际应用的要求.

由于当时刚接触相关内容, 并且再次整理时已经间隔了一年半, 因此肯定存在设计和代码上的缺陷和漏洞, 欢迎大家积极交流

>演示视频(00:20s - 00:60s):\
https://www.bilibili.com/video/BV1By4y1k7PY?spm_id_from=333.999.0.0

>博客地址: CSDN(https://blog.csdn.net/weixin_43667730/article/details/117585078)

###项目版本V2.0更新预告: 
由树莓派进行人脸位置的检测, 获取位置后控制两个舵机形成二自由度云台追踪人脸\
加入了PID舵机控制 + socket通信 + 视频流输出 + 视频堆栈防延迟
>演示视频(00:00s - 00:20s):\
https://www.bilibili.com/video/BV1By4y1k7PY?spm_id_from=333.999.0.0

(由于手边没有设备, 暂时放弃维护V2.0, 代码文件为: Raspberry Pi 4B + Stack + Servo + socket.py)

#项目文件说明
> **capture_path**:   所有驾驶人的全景图像 (仅采集, 未使用, 不可删) \
> **face_path**:    所有驾驶人的人脸区域图像, 用于身份识别的训练\
> **test_video**:   测试视频所存放的文件夹\
> **aspect_ratio_estimation.py**:   计算EAR 和 MAR的程序\
> **dlib-19.7.0-cp36-cp36m-win_amd64.whl**: Dlib的安装whl文件
> **drivers_img_acquire.py**:       获取驾驶人全景图像和人脸区域的程序\
> **Eigen_Face_Recognizer.py**:     特征脸识别器文件, 用特征脸识别不同驾驶人身份(效果并不好, 仅作为理论分析)\
> **get_everybody_EARandMAR_standard.py**: 得到每个驾驶人的EAR和MAR基准\
> **haarcascade_eye.xml**:          用于检测人眼睛位置的Haar级联分类器文件\
> **haarcascade_frontalface_alt.xml**: 用于检测人脸部位置的Harr级联分类器文件\
> **head_posture_estimation.py**:   头部姿态估计文件\
> **main.py:      主函数, 用于处理拍摄好的视频图像**\
> **shape_predictor_68_face_landmarks.dat**: 特征点数据库文件

# 如何安装必要的环境和依赖库
> + **1.cd 进入该项目文件地址中**
> + 2.创建新的虚拟环境: conda create -n Fatigue-Driving-Detection_py36 python=3.6
> + 3.切换到该虚拟环境: activate Fatigue-Driving-Detection_py36
> + 4.安装必要的依赖库: pip3 install -r requirement.txt
> + 5.使用whl安装Dlib: pip install dlib-19.7.0-cp36-cp36m-win_amd64.whl
> + 6.重启编译器, 运行程序

# 如何运行该项目
在运行项目之前, 应确保你有用于测试的视频文件. 本项目中提供了一个视频例程(/test_video/driving.mp4)
> **必须执行**: 首先运行 drivers_img_acquire.py 文件, 输入当前驾驶人的名字英文缩写, 并创建相应的文件夹. 程序会自动获取不同驾驶人的两类图像
> + 获取的第一类图像为 摄像头全景图像, 默认存放于 './capture_path/{your name}'. 
> + 获取的第二类图像为 驾驶人人脸区域图像, 默认存放于 './face_path/{your name}'. 
 
> 然后直接运行main.py程序即可.






