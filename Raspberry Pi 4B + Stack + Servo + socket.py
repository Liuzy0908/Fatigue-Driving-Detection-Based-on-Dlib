# -*- coding: utf-8 -*-             
from __future__ import division     #对未来版本兼容  只能放第一句
import Adafruit_PCA9685             #舵机控制库 pwm、频率等
import time                         #time库，用于延时
import cv2
import threading
import socket
import RPi.GPIO as GPIO             #树莓派的gpio库

GPIO.setmode(GPIO.BCM)              #gpio的排序定义方式
GPIO.setup(16, GPIO.OUT)            #16为蜂鸣器io口
GPIO.output(16,True)                #ture为不响，详细见电路原理图


#----------------------------通信程序分割线-------------------------------------
#设置socket的模式是tcp/ip
s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
#这里我们就用到了静态ip
#请根据自身情况自行修改port ，根据电脑ip不同而改
address='192.168.137.102'        
port =8888
#绑定地址和端口号
s.bind((address,port))
#设置允许接入的服务器的个数
s.listen(2)
print("请运行电脑端的通信程序，确保通信已建立后程序才会运行!")
sock,addr=s.accept()

#初始化PCA9685和舵机
pwm = Adafruit_PCA9685.PCA9685()
pwm.set_pwm_freq(60)    #设置pwm频率
time.sleep(0.5)         #延时0.5s
pwm.set_pwm(1,0,90)     # 参数：（通道（哪一个舵机）、（0）、（脉冲数））1为下面的舵机
time.sleep(0.5)
pwm.set_pwm(2,0,325)    #此处为控制Y轴（俯仰），脉冲数越大，舵机越向下  设置以在实验室调好的为准，这个325是以前调的
time.sleep(1)

#初始化、引入分类器
face_cascade = cv2.CascadeClassifier( '123.xml' )
eye_cascade = cv2.CascadeClassifier('eye.xml')

#初始化各个参数，之后用到处详细介绍
x=0
y=0
w=0
h=0
thisError_x=0
lastError_x=0
thisError_y=0
lastError_y=0

Y_P = 425           #舵机开始初试位置设置
X_P = 425           #舵机开始初试位置设置


#控制舵机的详细函数
def xx():
    while True:
        CON=0
        if CON==0: #CON=0 代表是一开机时的初试中位
            pwm.set_pwm(1,0,650-X_P+200)
            pwm.set_pwm(2,0,650-Y_P+200)
            CON=1
        else:
            pwm.set_pwm(1,0,650-X_P) #正常工作时的PWM获取输出 X为下面的舵机轴
            pwm.set_pwm(2,0,650-Y_P) #正常工作时的PWM获取输出 Y为上面的舵机轴
    



class Stack:  #设置视频堆栈
 
    def __init__(self, stack_size):
        self.items = [] 
        self.stack_size = stack_size            #设置想要的堆栈大小=3
 
    def is_empty(self):
        return len(self.items) == 0
 
    def pop(self):
        return self.items.pop()                 #弹出（最新的）项目
 
    def peek(self):
        if not self.isEmpty():
            return self.items[len(self.items) - 1] #返回去掉最后一帧的项目
 
    def size(self):
        return len(self.items)                  #返回项目的长度大小
 
    def push(self, item):
        if self.size() >= self.stack_size:      #如果项目的长度大小 >= 堆栈大小，即表示即将溢出时
            for i in range(self.size() - self.stack_size + 1):
                #(self.size() - self.stack_size + 1) 表示看看要存入的项目比堆栈大多少，既会溢出多少个
                self.items.remove(self.items[0])#依次删除堆栈底部（一开始）的值
        self.items.append(item)#把最新值加入
        
       


def capture_thread(video_path, frame_buffer, lock): #存入视频帧函数
    print("capture_thread start")
    #cap = cv2.VideoCapture(0)              #选择开启哪个摄像头
    cap = cv2.VideoCapture('http://192.168.137.102:8080/?action=stream') #选择开启哪个摄像头
    cap.set(3, 640)                         #设置图像规格
    cap.set(4, 480)
    if not cap.isOpened():
        raise IOError("摄像头不能被调用")
    while True:
        return_value, frame = cap.read()    #获取布尔值和视频帧
        if return_value is not True:
            break
        lock.acquire()                      #上锁保护
        frame_buffer.push(frame)            #添加最新视频帧
        lock.release()                      #解锁
        if cv2.waitKey(1)==27:              #如果按esc则退出
                break
    cap.release()
    cv2.destroyAllWindows()

def play_thread(frame_buffer, lock):            #显示视频帧函数，对最新弹出的视频帧处理函数
    
    print("detect_thread start")
    print("detect_thread frame_buffer size is", frame_buffer.size())
    global thisError_x,lastError_x,thisError_y,lastError_y,Y_P,X_P
    while True:
        try:
            t=sock.recv(1024).decode('utf8')    #接收socket函数，接收的值存入 t （alarm_flag）
            if t =='1':
                print('请勿疲劳驾驶')
                GPIO.output(16,False)           #Flase为响
                cv2.waitKey(1)
                
            elif t=='0':
                GPIO.output(16,True)            #True为不响
                
            else:
                GPIO.output(16,True)            #True为不响 
                
        except Exception:
            continue
        
        if frame_buffer.size() > 0:#确保设置了自定义堆栈有大小
            #print("detect_thread frame_buffer size is", frame_buffer.size())
            lock.acquire()                      #上锁保护
            frame = frame_buffer.pop()          #弹出最新的视频帧
            lock.release()#                     解锁
            #cv2.waitKey(100)
            # TODO 算法
            frame = cv2.flip(frame,0,dst=None)  #图像上下反转
            frame = cv2.flip(frame,1,dst=None)  #图像左右反转
            gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#灰度化图像
            #对灰度图进行.detectMultiScale()
            faces=face_cascade.detectMultiScale( #创建了一个 faces 的向量元组，已经找到了人脸位置
                gray,                           #具体参数可以搜索 “python中face_cascade.detectMultiScale 参数调节”
                scaleFactor=1.3,                #越小越精确且计算量越大
                minNeighbors=2,                 #连接几次检测出人脸 才认为是真人脸
                minSize=(300, 300)              #你认为图像中人脸最小的大小，调大有利于减少误判，但可能把小于此值的人脸排除
            )
        
            
            
            if len(faces)>0: #如果人脸数大于0
                #print('face found!')
                #temp = (x,y,w,h)
                for(x,y,w,h) in faces: #（x，y）为人脸区域左上角的坐标，w、h为人脸区域的宽、高
                    cv2.rectangle(frame,(x,y),(x+h,y+w),(0,255,0),2) #画矩形
                    eyeh=int(h/2)               #仅保留上半脸
                    fac_gray = gray[y: (y+eyeh), x: (x+w)] #将脸部全部灰度化 
                    eye_result = []             #清空眼睛位置坐标
                    eye = eye_cascade.detectMultiScale(fac_gray, 1.1, 7) #眼睛检测器，具体参数可以搜索 “python中eye_cascade.detectMultiScale 参数调节”
                    for (ex, ey, ew, eh) in eye:
                        eye_result.append((x+ex, y+ey, ew, eh)) #（x，y）为眼睛区域左上角的坐标，w、h为眼睛区域的宽、高
                        for (ex, ey, ew, eh) in eye_result:
                            cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                    result=(x,y,w,h)        #提取出xywh
                    x=result[0]+w/2         #即 x=x+w/2 定位到人脸图像的正中央，把x当作人脸中央
                    y=result[1]+h/2         #即 y=y+h/2 定位到人脸图像的正中央，把y当作人脸中央
            
   
                thisError_x=x-320           # 计算人脸中央距离整体图像中央的差距（x方向）
                if thisError_x <10 and thisError_x >-10: #设置死区，小于此值认为现在就在中间
                    thisError_x = 0
                thisError_y=y-240
                if thisError_y <10 and thisError_y >-10:
                    thisError_y = 0
                #if thisError_x > -20 and thisError_x < 20 and thisError_y > -20 and thisError_y < 20:
                #    facebool = False
                    
                #自行对P和D两个值进行调整
                pwm_x = thisError_x*7+7*(thisError_x-lastError_x) #PD计算
                pwm_y = thisError_y*7+7*(thisError_y-lastError_y)
                lastError_x = thisError_x   #把现在的误差记录下来，当作下次程序中上次误差
                lastError_y = thisError_y
                XP=pwm_x/100                #缩小
                YP=pwm_y/100
                X_P=X_P+int(XP)             #注意有没有下划线“_” 
                Y_P=Y_P+int(YP)
                if X_P>670:                 #限位，防止转太多
                    X_P=650
                if X_P<0:
                    X_P=0
                if Y_P>650:
                    Y_P=650
                if X_P<0:
                    Y_P=0
           
            cv2.imshow("capture", frame)
        
    
    s.close()

if __name__ == '__main__':
    
    
    path = 0 #这个path没用了，在capture_thread函数的cap中从新选择
    frame_buffer = Stack(3)             #设置堆栈大小
    lock = threading.RLock()
    t1 = threading.Thread(target=capture_thread, args=(path, frame_buffer, lock)) #capture_thread-存入视频帧函数的线程
    t1.start()
    
    t2 = threading.Thread(target=play_thread, args=(frame_buffer, lock))#play_thread-处理最新帧函数的线程
    t2.start()
    
    tid=threading.Thread(target=xx)     # xx - 舵机pwm输出函数的线程
    tid.setDaemon(True) 
    tid.start()




















