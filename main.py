# -*- coding: utf-8 -*-
####################################################
#   作者: 刘朝阳
#   时间: 2020.05.01
#   更新时间: 2021.11.25
#   程序参考: https://blog.csdn.net/cungudafa/article/details/103499230
#   功能: 处理拍摄好的驾驶视频. 驾驶视频应包括 驾驶人的眨眼, 哈欠, 瞌睡点头
#   使用说明: 检测 blink, yawn 和 nod 时, 均是满足持续一定时间阈值后才能计入一次
#   待改进之处: ①-③
#   ①不要用列表存储数据, 改用h5py格式
#   ②使用不同摄像头时, 需要重新标定相机内外参, 才能准确预测出头部姿态
#   ③在采集驾驶人图像于处理驾驶视频时, 最好保持距离摄像头的距离不变, 光照强度不变
####################################################
import imutils
import datetime

from Eigen_Face_Recognizer import *
from get_everybody_EARandMAR_standard import *

face_path =     './face_path'                       # 所有驾驶人脸部图像的存放路径
# path =          "./test_video/driving.mp4"          # 待检测的本地视频的存放路径
path = 0                                            # 切换此处使用本地摄像头实时检测

# 绘制正方体12轴所需的索引
line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]

everybody_EAR_mean, everybody_EAR_min, _ = get_everybody_EARandMAR_standard(face_path)
print('所有人睁眼EAR平均值:', everybody_EAR_mean)
print('所有人闭眼EAR最小值:', everybody_EAR_min)

# 初始化各类参数
# 设定阈值. 阈值可以参考论文, 或者根据经验. 本项目中的阈值设定并不标准.
EAR_threshold = 0.13
MAR_threshold = 0.6
pitch_threshold = 6.5

cap = cv2.VideoCapture(path)
Driving_Time_Threshold = 65
driving_time = P80_time1 = P80_time2 = nod_time = yawn_time = 0
P80_sum_time1 = []
P80_sum_time2 = []
nod_sum_time = []
yawn_sum_time = []
nod_flag = yawn_flag = 0
P80_flag1 = P80_counter1 = 0
P80_flag2 = P80_counter2 = 0
yawn_counter = yawns = blink_counter = blinks = nod_counter = nods = 0  # 初始化各种计数器
alarm_flag = '0'                                            # 初始化疲劳驾驶警报标志位
f = 0                                                       # 初始化 perclos值
last_params = []
EAR_plt = []
MAR_plt = []

nod_starttime = P80_starttime1 = datetime.datetime.now()
P80_starttime2 = datetime.datetime.now()
nod_endtime = datetime.datetime.now()
starttime = datetime.datetime.now()                         # 获取程序开机时刻的时间

print('------------------- 开始执行主函数 -------------------')

while True:
    ret, im_rd = cap.read()
    if ret == 1:                                            # 如果成功获取图像
        im_rd = imutils.resize(im_rd, height=540, width=720)
        # im_rd = cv2.resize(im_rd, (720, 540))
        original_img = im_rd.copy()                         # 分析时用original_img，画图时用im_rd

        # 每帧数据延时1ms，延时为0读取的是静态帧
        k = cv2.waitKey(1)
        # 灰度化图像
        img_gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
        # 使用人脸检测器检测每一帧图像中的人脸。并返回人脸数rects
        faces = detector(img_gray, 0)
        # 如果检测到人脸
        if (len(faces) == 1):                               # 如果有多个人脸就不检测了
            # 对每个人脸都标出68个特征点
            for i in range(len(faces)):                     # 因为人脸数只有1，所以该句可删除，但为以后增加功能所保留
                # enumerate方法同时返回数据对象的索引和数据，k为索引，d为faces中的对象
                for k, d in enumerate(faces):

                    try:
                        roi_gray = img_gray[d.top():d.bottom(), d.left():d.right()]
                        roi_gray = cv2.resize(roi_gray, (92, 112))
                        # 此处为人脸识别器的模型预测值，其返回两个元素的数组
                        # 第一个是识别个体的标签，第二个是置信度参数
                        params = Eigen_Face_Model.predict(roi_gray)
                    except:
                        continue

                    # 使用dlib预测器得到68点数据的坐标
                    shape = predictor(original_img, d)
                    shape_array = face_utils.shape_to_np(shape)  # 将脸部特征信息转换为数组array的格式

                    # 用矩形框出人脸
                    # cv2.rectangle(im_rd, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 255))

                    if params[0] != last_params:  # 如果换了驾驶人，清空持续驾驶时间.
                        driving_time = 0

                    last_params = params[0]  # 记录上一时刻驾驶人的身份

                    # 用圆圈标出每个特征点
                    for i in range(68):
                        cv2.circle(im_rd, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), -1, 8)

                    reprojectdst, _, pitch, roll, yaw = HPE.get_head_pose(shape_array)  # 重新投影，欧拉角

                    # 绘制正方体12轴
                    for start, end in line_pairs:
                        cv2.line(im_rd, (int(reprojectdst[start][0]),int(reprojectdst[start][1])),
                                 (int(reprojectdst[end][0]),int(reprojectdst[end][1])), (0, 0, 255))

                    # 提取左眼、右眼和嘴巴的所有坐标
                    leftEye = shape_array[lStart:lEnd]
                    rightEye = shape_array[rStart:rEnd]
                    mouth = shape_array[mStart:mEnd]

                    # 使用cv2.convexHull获得凸包位置，使用drawContours画出轮廓位置进行画图操作 #把嘴巴、眼睛特征点连起来
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    cv2.drawContours(im_rd, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(im_rd, [rightEyeHull], -1, (0, 255, 0), 1)
                    mouthHull = cv2.convexHull(mouth)
                    cv2.drawContours(im_rd, [mouthHull], -1, (0, 255, 0), 1)

                    '''######################################## 计算眨眼计数 #############################'''
                    leftEAR = ARE.eye_aspect_ratio(leftEye)
                    rightEAR = ARE.eye_aspect_ratio(rightEye)
                    EAR = (leftEAR + rightEAR) / 2.0

                    if EAR < EAR_threshold:
                        blink_counter += 1
                    else:
                        if blink_counter >= 5:
                            blinks += 1
                        blink_counter = 0

                    # 如果在60s内眨眼超过20次
                    if round(driving_time)%60 == 0:
                        if blinks >= 20:
                            alarm_flag = 'blinks_waring'
                        blinks = 0

                    '''######################################## 计算 M A R #################################'''
                    # xxx_flag 都是开关标志位, 作用为: 打开或者关闭判断语句 的判断标志
                    MAR = ARE.mouth_aspect_ratio(mouth)

                    if MAR > MAR_threshold and yawn_flag == 0:  # 点头阈值  #har和x坐标有关  x即为pitch
                        yawn_counter += 1
                        if MAR > MAR_threshold and yawn_counter >= 3:  # 如果连续x次检测到低头，且现在还在低头
                            yawn_starttime = datetime.datetime.now()
                            yawn_flag = 1

                    if MAR <= MAR_threshold:
                        if (sum(yawn_sum_time)) >= 2:
                            if yawns <= 1:
                                yawns += 1

                        yawn_sum_time = [0]
                        yawn_counter = 0

                    if yawn_flag == 1:  # 如果pitch>=3.5且flag=1时
                        yawn_endtime = datetime.datetime.now()
                        if (yawn_endtime - yawn_starttime).seconds > 0 and yawn_flag == 1:
                            yawn_time = (yawn_endtime - yawn_starttime).seconds
                            yawn_sum_time.append(yawn_time)
                            yawn_flag = 0

                    # 如果在60s内哈欠数超过15次
                    if round(driving_time)%60 == 0:
                        if yawns > 15:
                            alarm_flag = 'yawns_waring'
                        yawns = 0

                    '''######################################## 计算点头次数 #############################'''  ###########
                    if pitch > pitch_threshold and nod_flag == 0:
                        nod_counter += 1
                        if pitch > pitch_threshold and nod_counter >= 2:
                            nod_starttime = datetime.datetime.now()
                            nod_flag = 1

                    if pitch <= pitch_threshold:
                        if (sum(nod_sum_time)) >= 2:
                            nods += 1

                        nod_sum_time = []
                        nod_counter = 0

                    if nod_flag == 1:
                        nod_endtime = datetime.datetime.now()
                        if (nod_endtime - nod_starttime).seconds > 0 and nod_flag == 1:
                            nod_time = (nod_endtime - nod_starttime).seconds
                            nod_sum_time.append(nod_time)
                            nod_flag = 0

                    # 如果在60s内瞌睡点头数超过10次
                    if round(driving_time)%60 == 0:
                        if nods > 10:
                            alarm_flag = 'nods_waring'
                        nods = 0

                    '''######################################## Perclos_80标准: #############################'''
                    # 定义两个常数
                    T1 = everybody_EAR_min[params[0]] + 0.2 * (
                                everybody_EAR_mean[params[0]] - everybody_EAR_min[params[0]])  # 睁眼程度20%
                    T2 = everybody_EAR_min[params[0]] + 0.8 * (
                                everybody_EAR_mean[params[0]] - everybody_EAR_min[params[0]])  # 睁眼程度80%

                    # 先算t3-t2
                    if EAR < T1 and abs(pitch) < 10 and abs(yaw) < 20 and abs(roll) < 20 and P80_flag1 == 0:
                        P80_counter1 += 1
                        if EAR < T1 and P80_counter1 >= 1:
                            P80_starttime1 = datetime.datetime.now()
                            P80_counter1 = 0
                            P80_flag1 = 1

                    elif P80_flag1 == 1:
                        P80_endtime1 = datetime.datetime.now()

                        if (P80_endtime1 - P80_starttime1).seconds > 0 and P80_flag1 == 1:
                            P80_time1 = (P80_endtime1 - P80_starttime1).seconds
                            P80_sum_time1.append(P80_time1)
                            P80_flag1 = 0
                    else:
                        P80_counter1 = 0

                        # 再算t4-t1
                    if EAR < T2 and abs(pitch) < 10 and abs(yaw) < 20 and abs(roll) < 20 and P80_flag2 == 0:  # 只目视前方时计时
                        P80_counter2 += 1
                        if EAR < T2 and P80_counter2 >= 1:
                            P80_starttime2 = datetime.datetime.now()
                            P80_counter2 = 0
                            P80_flag2 = 1

                    elif P80_flag2 == 1:
                        P80_endtime2 = datetime.datetime.now()

                        if (P80_endtime2 - P80_starttime2).seconds > 0 and P80_flag2 == 1:
                            P80_time2 = (P80_endtime2 - P80_starttime2).seconds
                            P80_sum_time2.append(P80_time2)
                            P80_flag2 = 0
                    else:
                        P80_counter2 = 0

                    # 计算PERCOLS的值f
                    try:  # 防止除以0
                        f = round(sum(P80_sum_time1) / (sum(P80_sum_time2)), 2)
                    except:
                        pass

                    if round(driving_time) % 60 == 0:

                        P80_sum_time1 = []
                        P80_sum_time2 = []
                        f = 0

                    print('小于20%的时间: ', sum(P80_sum_time1), ' s')
                    print('小于80%的时间: ', sum(P80_sum_time2), ' s')
                    print('当前驾驶人的EAR: ', EAR, '\n')
                    print('T1:', T1, ';  ', 't2:', T2, '\n')

                    '''######################################## 显示警告 ###########################'''
                    cv2.putText(im_rd, "Warning : {}".format(alarm_flag), (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)  # 直接在Haar返回的人脸上画

                    '''######################################## 驾驶时长标准 ###########################'''
                    # 如果超过了规定的持续驾驶时间
                    if driving_time > Driving_Time_Threshold:
                        alarm_flag = 'Long_Time_For_Driving'

                    '''######################################## 显示驾驶人身份和驾驶时长 ###########################'''
                    cv2.putText(im_rd, "Driver identity: {}".format(names[params[0]]), (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)  # 直接在Haar返回的人脸上画
                    cv2.putText(im_rd, "Driving time:{} s".format(int(driving_time)), (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)  # 直接在Haar返回的人脸上画

                    '''######################################## 显示头部状态角度 ###########################'''
                    cv2.putText(im_rd, "Pitch:{}".format(round(pitch, 2)), (int(40 + im_rd.shape[1] * 0.25), 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), thickness=2)  # GREEN
                    cv2.putText(im_rd, "Yaw:{}".format(round(yaw, 2)), (int(40 + im_rd.shape[1] * 0.25), 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), thickness=2)  # BLUE
                    cv2.putText(im_rd, "Roll:{}".format(round(roll, 2)), (int(40 + im_rd.shape[1] * 0.25), 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), thickness=2)  # RED

                    '''######################################## 显示疲劳特征 ###########################'''
                    cv2.putText(im_rd, "EAR:{}".format(round(EAR, 2)), (int(20 + im_rd.shape[1] * 0.5), 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv2.putText(im_rd, "MAR:{}".format(round(MAR, 2)), (int(20 + im_rd.shape[1] * 0.5), 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                    if EAR > 0.2:
                        cv2.putText(im_rd, "eyes_state: {}".format('Open'), (int(20 + im_rd.shape[1] * 0.5), 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    else:
                        cv2.putText(im_rd, "eyes_state: {}".format('Close'), (int(20 + im_rd.shape[1] * 0.5), 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                    cv2.putText(im_rd, "Nod duration: {} s".format(int(sum(nod_sum_time))),
                                (int(20 + im_rd.shape[1] * 0.5), 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv2.putText(im_rd, "Yawn duration: {} s".format(int(sum(yawn_sum_time))),
                                (int(20 + im_rd.shape[1] * 0.5), 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                    '''######################################## 显示疲劳判断值 ###########################'''
                    cv2.putText(im_rd, "Blinks:{}".format(blinks), (int(30 + im_rd.shape[1] * 0.75), 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(im_rd, "Yawns:{}".format(yawns), (int(30 + im_rd.shape[1] * 0.75), 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(im_rd, "Nods:{}".format(nods), (int(30 + im_rd.shape[1] * 0.75), 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(im_rd, "PERCOLS:{}".format(f), (int(30 + im_rd.shape[1] * 0.75), 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # 只检测到一个人脸-正常运行中
                cv2.putText(im_rd, 'Working', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        elif (len(faces) == 0):
            # 如果没有检测到人脸
            cv2.putText(im_rd, "No Face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            # 如果检测到有多个人脸
            cv2.putText(im_rd, "Too Many Faces!", (10, im_rd.shape[0] * 0.5), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0,), 2)

        endtime = datetime.datetime.now()

        driving_time = (endtime - starttime).seconds        # 计算程序持续运行的时间

        # 窗口显示视频帧
        cv2.imshow("camera", im_rd)

        Key = cv2.waitKey(10)
        # 按下Esc键退出
        if (Key == 27):
            break
    else:                                                   # 如果没有获取到图像
        driving_time = P80_time1 = P80_time2 = nod_time = yawn_time = 0
        P80_sum_time1 = []
        P80_sum_time2 = []
        nod_sum_time = yawn_sum_time = []
        nod_flag = yawn_flag = 0
        P80_flag1 = P80_counter1 = 0
        P80_flag2 = P80_counter2 = 0
        yawn_counter = yawns = blink_counter = blinks = nod_counter = nods = 0  # 初始化各种计数器
        alarm_flag = '0'                                    # 初始化疲劳驾驶警报标志
        f = 0
        cap = cv2.VideoCapture(path)                        # 循环视频

    # 删除建立的窗口
cv2.destroyAllWindows()


