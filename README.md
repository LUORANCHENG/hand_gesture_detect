# 基于Mediapipe实现的手势识别控制电脑音量

## 目录：

- [前言](#section1)
- [为什么选择MediaPipe?](#section2)
- [MediaPipe手势识别的一般实现原理](#section3)
- [具体的实现细节](#section4)
    - [1.图像预处理过程中的具体细节](#subsection1)
    - [2.通过CNN（卷积神经网络）来提取手部边缘、纹理、形状等信息](#subsection2)
    - [3.进一步检测手部关键点](#subsection3)
- [使用训练好的模型进行手势检测](#section5)
- [前景与展望](#section6)




## 前言：<a name="section1"></a>

本项目主要通过opencv和MediaPipe实现了通过手势检测来控制电脑的音量。

opencv这个库相信大家都很熟悉，所以这里不做过多的介绍

MediaPipe是由Google开发的一个开源框架，用于构建基于机器学习的多媒体处理应用程序。它提供了一系列预训练的机器学习模型和工具，可以用于实现各种计算机视觉和媒体处理任务，例如人脸检测、手势识别、姿势估计、实时物体跟踪、手部追踪等等。

效果演示：

https://github.com/LUORANCHENG/hand_gesture_detect/assets/80999506/34e62cdc-c442-4986-bac3-d453f6096dae



## 为什么选择MediaPipe? <a name="section2"></a>

⭐1. 实时性能：MediaPipe专注于实时性能，可以在低延迟的情况下处理图像和视频数据。这使其适用于需要快速响应的应用程序，如虚拟现实、增强现实和实时手势识别。

2. 跨平台支持：MediaPipe支持多种平台，包括移动设备（如Android和iOS）、桌面计算机和嵌入式系统。这使开发者能够在不同设备上部署他们的应用程序。

3. 预训练模型：MediaPipe提供了大量预训练的机器学习模型，用于执行各种视觉和感知任务。这些模型可以快速集成到应用程序中，无需从头开始训练。

4. 多媒体输入支持：MediaPipe能够处理图像、视频、音频等多媒体输入，使其适用于各种多媒体应用。

5. 灵活性：MediaPipe提供了丰富的工具和库，允许开发者自定义和扩展现有的模型和组件，以满足其特定需求。



## MediaPipe手势识别的一般实现原理 <a name="section3"></a>\

### 1.模型架构

我们的手部识别方案是由两个模型共同工作的
- 手掌检测模型：将完整的图像输入到模型中，输出的是定位手掌的边界框(bounding box)。
- 手部关键点检测的模型：输入的是经过裁切的手掌的边界框的区域图像，输出的是高质量的2.5D的21个手部关键点

⭐如何提升模型的性能
- 在实时跟踪场景中，我们从前一帧的关键点预测中得出一个边界框，作为下一帧的输入，从而避免在每一帧上都使用手掌检测的模型。而是能够用上一帧关键点检测的结果预测出下一帧手掌的区域；我们只有当是第一帧图像或者没有检测到手部关键点的时候，才重新启动手掌检测的模型。这样就可以大大节省时间和节省性能。

手掌检测模型的模型架构
插入图片：

从figure2中我们可以看出手掌检测模型是一个编码解码器，中间还有一些跨层连接。

为了检测出初始的手掌位置，我们使用了SSD模型，这个模型是专门针对移动端实时检测优化的。

检测手是一个非常复杂的任务，首先，手是一个非常复杂的物体，有非常多的尺寸，还会存在遮挡和自遮挡，手的检测不同于人脸检测，人脸是一个具有高对比度的物体，比如眼睛，嘴巴......这些都可以辅助来检测人脸，但是手是一个动态的，非常的复杂，手并没有像眼睛和嘴那样有高对比度和置信度的特征。

但是我们这个方案就解决了上述的问题：

1.首先我们训练的是一个手掌的检测器，而不是整个手部的检测器。因为手掌是一个正方形的物体，是一个类似于刚体的东西，并不会像手指那样灵活乱动，所以手掌是一个比较稳定的特征，而且手掌是一个更小的物体，所以在使用非极大值抑制的算法剔除重复预测的边界框的时候，在两只手握手和自遮挡的情况下能够表现的更好。并且手掌是一个正方形的物体，不需要考虑其他的长宽比，所以可以只保留正方形的锚框，而不需要考虑其他长宽比的锚框，从而使得锚框的数量减少了3~5倍



## 使用训练好的模型进行手势检测 <a name="section5"></a>

我们可以通过Mediapipe提供的预训练好的模型和一系列封装好的方法快速部署一个手势检测的应用程序：

MediaPipe中的手势识别模块允许您识别手部姿势和手势，这对于许多应用程序，如手势控制、虚拟手势交互、手部追踪等非常有用。以下是一些关于如何使用MediaPipe进行手势识别的基本信息：

1. **导入MediaPipe库**：首先，您需要导入MediaPipe库。您可以使用Python进行导入：

   ```python
   import mediapipe as mp
   ```

2. **初始化Hand模型**：接下来，您需要初始化Hand模型，这可以通过`mp.solutions.hands`来完成：

   ```python
   hand_detector = mp.solutions.hands.Hands()
   ```

3. **获取摄像头输入**：您需要设置摄像头输入或者图像输入，以便模型能够处理它。通常，您可以使用OpenCV等库来捕获摄像头帧。

    ```python
    # 进入无限循环以捕捉视频帧
    while True:
        # 读取摄像头帧
        success, img = camera.read()
    
        if success:
    
            # 将图像进行镜像翻转
            img = cv.flip(img, 1)
    
            # 将img传入模型进行手势检测
            ......
    
            # 在窗口中显示捕捉到的视频帧
            cv.imshow('Video', img)
    
        # 等待按键输入，等待时间为1毫秒
        k = cv.waitKey(1)
    
        # 如果按下 'q' 键，退出循环
        if k == ord('q'):
            break
    ```

4. **处理帧**：对于每一帧图像，您需要将其传递给Hand模型进行处理：

   ```python
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

  
   ```

   self.hands_data.multi_hand_landmarks返回的是一个列表,里面的数据格式为:
   
   [landmark:{x: ,y: ,z:}, landmark:{x: ,y: ,z:}, ......]
   
   里面的landmark为手部每个关键点的坐标，一个列表里面有21个元素，代表一共有21个手部关键点
   
   x 表示关键点在图像上的水平位置（x轴坐标）。
   
   y 表示关键点在图像上的垂直位置（y轴坐标）。
   
   z 表示关键点的深度信息(由于这里只有一个摄像头，故这个参数无用)

   <div align="center">
   <img src="https://github.com/LUORANCHENG/hand_gesture_detect/blob/main/picture/hand.png" width="700" >
   </div>

   

6. **提取手部信息**：一旦处理完成，您可以从`results`中提取手部信息，包括手部的关键点坐标和手势识别结果。

   ```python
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


   ```

7. **通过分析手部关键点的坐标实现控制音量功能**：通过分析关键点坐标，可以实现控制音量等各种功能。

   ```python
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

   ```

## 前景与展望 <a name="section6"></a>

由于MediaPipe提供的手势识别模型具有极佳的实时性能，可以在低延迟的情况下处理图像和视频数据，以及优秀的跨平台支持，使得这个模型具有非常广阔的应用情景，本项目所实现的控制音量功能仅仅是一个非常小的应用场景，你还可以通过手势识别来控制机器人，控制各种物联网设备等，应用领域可以横跨工业，医疗，日常生活等各种你可以想到的领域，实现如霍格沃兹魔法般的隔空控制物体，从而给人们的生产生活带来极大的便利。











