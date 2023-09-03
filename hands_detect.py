# 导入必要的库
import cv2 as cv  # 使用OpenCV库进行图像处理
import mediapipe as mp  # 使用MediaPipe库进行姿势估计任务
import numpy as np
import math
import pulsectl

# 创建一个VideoCapture对象，用于打开摄像头设备
camera = cv.VideoCapture(0)

# 初始化 PulseAudio
pulse = pulsectl.Pulse('set-volume')

# 获取默认的音频输出 (sink)
sink = pulse.sink_list()[0]  # 这里假设使用第一个音频输出

# 初始化两个变量，用于保存左手食指指尖和左手拇指指尖之间的最小距离和最大距离
min_distance = 100
max_distance = 0

# 定义0到100的范围
target_min = 0
target_max = 100


class HandDetector():
    def __init__(self):
        # 创建MediaPipe手部检测器对象
        self.hand_detector = mp.solutions.hands.Hands()
    
    def process(self, img, display_point=True):
        # 将图像从BGR颜色空间转换为RGB颜色空间（MediaPipe需要RGB输入）
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # 使用MediaPipe手部检测器来检测手部
        self.hands_data = self.hand_detector.process(img_rgb)
        """
        result包含MediaPipe手部检测器处理结果的对象。里面含有手部位置的信息
        """

        if display_point:
            # 如果检测到手部
            if self.hands_data.multi_hand_landmarks:
                # 遍历每个检测到的手部
                for handlms in self.hands_data.multi_hand_landmarks:
                    # 在图像上绘制手部关键点和连接线
                    mp.solutions.drawing_utils.draw_landmarks(img, handlms, mp.solutions.hands.HAND_CONNECTIONS)
        """
        self.hands_data.multi_hand_landmarks返回的是一个列表,里面的数据格式为:
        [landmark:{x: ,y: ,z:}, landmark:{x: ,y: ,z:}, ......]
        里面的landmark为手部每个关键点的坐标，一个列表里面有21个元素，代表一共有21个手部关键点
        x 表示关键点在图像上的水平位置（x轴坐标）。
        y 表示关键点在图像上的垂直位置（y轴坐标）。
        z 表示关键点的深度信息(由于这里只有一个摄像头，故这个参数无用)
        """
    
    def find_position(self, img):
        # 获取输入图像的高度、宽度和通道数
        h, w, c = img.shape

        # 创建一个空字典用于存储手部位置信息，包括左手和右手
        position = {'Left': {}, 'Right': {}}

        # 检查是否检测到了多个手部
        if self.hands_data.multi_hand_landmarks:
            i = 0

            # 遍历每个检测到的节点
            for point in self.hands_data.multi_handedness:
                # 获取节点的分类分数
                score = point.classification[0].score

                # 如果分类分数高于或等于0.8，将其视为有效的手部检测
                if score >= 0.8:
                    # 获取手的标签（左手或右手）
                    label = point.classification[0].label

                    # 获取手部关键点的坐标
                    hand_lms = self.hands_data.multi_hand_landmarks[i].landmark

                    # 遍历每个关键点，并将其坐标映射到图像的实际像素坐标
                    for id, lm in enumerate(hand_lms):
                        x, y = int(lm.x * w), int(lm.y * h)

                        # 将关键点的坐标添加到相应手的位置字典中
                        position[label][id] = (x, y)

                i = i + 1

        # 返回包含手部位置信息的字典
        return position


# 距离映射公式(把一段范围内的数映射到另一段范围内)
def map_range(value, from_min, from_max, to_min, to_max):
    try:
        # 使用线性映射公式进行映射
        return (value - from_min) * (to_max - to_min) / (from_max - from_min) + to_min
    except:
        return 0


def volume_control(img, position):

    # 声明要用到的全局变量
    global target_max, target_min, min_distance, max_distance
    
    # 找到左手食指指尖
    left_finger_tip = position['Left'].get(8, None)

    # 找到左手拇指指尖
    left_thumb_tip = position["Left"].get(4, None)
    
    # 如果左手食指指尖和左手拇指指尖都可以被找到
    if left_thumb_tip and left_finger_tip:

        # 在左手食指指尖画一个圆
        cv.circle(img, (left_finger_tip[0], left_finger_tip[1]), 5, (0, 0, 255), cv.FILLED)
        
        # 在左手拇指指尖画一个圆
        cv.circle(img, (left_thumb_tip[0], left_thumb_tip[1]), 5, (0, 0, 255), cv.FILLED)
        
        # 将左手食指指尖和左手拇指指尖用直线连起来
        cv.line(img, left_thumb_tip, left_finger_tip, (0,0,255), 5)
        
        # 计算左手食指指尖和左手拇指指尖的距离(欧式距离)
        distance = np.sqrt((left_finger_tip[0] - left_thumb_tip[0])**2 + (left_finger_tip[1] - left_thumb_tip[1])**2)**0.5
        
        # 四舍五入为整数
        distance = round(distance)


        # 如果当前距离小于最小距离，就更新最小距离
        if min_distance > distance:
            min_distance = distance
        
        # 如果当前距离大于最大距离，就更新最大距离
        elif max_distance < distance:
            max_distance = distance

        
        # 调用距离映射函数把左手食指指尖和左手拇指指尖之间的最小距离和最大距离映射到0~100范围内的数，用来表示音量
        mapped_value = map_range(distance, min_distance, max_distance, target_min, target_max)
        
        # 四舍五入为整数
        mapped_value = round(mapped_value)

        # 根据mapped_value值来设置音量
        pulse.volume_set_all_chans(sink, mapped_value / 100.0)

        # 获取当前的音量的百分比
        current_volume = pulse.volume_get_all_chans(sink)
        
        # 将当前的音量百分比转换为真实值
        current_volume = int(current_volume * 100)

        print(f"当前音量:{mapped_value}")

        


if __name__ == "__main__":
    hand_detector = HandDetector()
    # 进入无限循环以捕捉视频帧
    while True:
        # 读取摄像头帧
        success, img = camera.read()

        if success:

            # 将图像进行镜像翻转
            img = cv.flip(img, 1)

            # 将图像传入手势检测器进行处理
            hand_detector.process(img, display_point=False)

            # 找出所有的关键点坐标
            position = hand_detector.find_position(img)

            volume_control(img, position)
            # 在窗口中显示捕捉到的视频帧
            cv.imshow('Video', img)

        # 等待按键输入，等待时间为1毫秒
        k = cv.waitKey(1)

        # 如果按下 'q' 键，退出循环
        if k == ord('q'):
            break

    # 释放摄像头资源
    camera.release()

    # 关闭所有打开的OpenCV窗口
    cv.destroyAllWindows()
