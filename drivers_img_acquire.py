# -*- coding: utf-8 -*-
####################################################
#   作者: 刘朝阳
#   时间: 2020.05.01
#   更新时间: 2021.11.25
#   功能: 采集当前驾驶人的两种图像 ①摄像头全景图 ②人脸区域图
#   使用说明: 运行程序, 在下方输入各个驾驶人的英文名字缩写(不重名的情况下1-3个小写字母即可);
#            程序自动打开摄像头来采集80张当前驾驶人的图像信息, 要求驾驶人以平静情绪正视摄像头, 在程序结束前完成5次左右眨眼动作
#   文件保存说明: 将采集的两种图像①、②分别存放于capture_path、face_path
####################################################
import cv2
import os
import shutil

# 采集驾驶人的人脸数据
def drivers_img_acquire(capture_path, face_path, video_path):
    '''
    打开摄像头，检测该帧图像中的人脸，并进行剪切、缩放
    '''
    name=input('当前驾驶人的名字(英文缩写即可):')

    capture_path=os.path.join(capture_path,name)    # 将路径和驾驶人名字进行拼接
    face_path=os.path.join(face_path,name)

    if os.path.isdir(capture_path):         # 如果已经存在这个路径
        shutil.rmtree(capture_path)         # 删除这个路径所指的文件夹

    if os.path.isdir(face_path):
        shutil.rmtree(face_path)
    #创建文件夹
    os.mkdir(capture_path)
    os.mkdir(face_path)

    #创建一个级联分类器
    face_casecade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    #打开摄像头
    camera=cv2.VideoCapture(video_path)
    cv2.namedWindow('Dynamic')

    count=1                                 # 采集图像数的计数器
    
    while(True):
        #读取一帧图像
        ret,frame=camera.read()                         # 生成图像帧，ret为0或1
        frame = cv2.flip(frame,1,dst=None)              # 图像水平反转
        if video_path != 0:
            frame = cv2.flip(frame,0,dst=None)          # 图像竖直反转
        original_img = frame.copy()                     # 拷贝原图
        if ret:
            #转换为灰度图
            gray_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #灰度化
            #人脸检测
            face=face_casecade.detectMultiScale(gray_img,1.3,5) #检测人脸，其中后面数字1.3、5为精度参数
            for (x,y,w,h) in face:                      # 在每一个face中提取 图像左上角坐标（xy）、宽度width，高度height
                #在原图上绘制人脸区域的矩形
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                #调整图像大小
                new_frame=cv2.resize(frame[y:y+h,x:x+w],(92,112))
                #保存人脸图像
                cv2.imwrite('%s/%s.png'%(capture_path,str(count)),original_img)     # 保存全景图像
                cv2.imwrite('%s/%s.png'%(face_path,str(count)),new_frame)           # 保存人脸图像
                print('正在记录当前驾驶人的第{}张图像'.format(count + 1))
                count += 1

            cv2.imshow('Dynamic',frame)
            #按下q键或者采集够80张图片后退出
            if (cv2.waitKey(100) & 0xff==27) or count >= 80:
                break
    camera.release()
    cv2.destroyAllWindows()

if __name__=='__main__':                    # 从主函数开始执行

    video_path = 0                          # 选择本地摄像头
    capture_path = './capture_path'         # data='./face'  打开当前目录下创建好的face_data文件夹
    face_path = './face_path'
    drivers_img_acquire(capture_path,face_path,video_path)