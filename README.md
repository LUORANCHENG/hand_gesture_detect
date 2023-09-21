# 一种基于MediaPipe为底层框架的针对移动应用进行优化的实时手势识别模型的实现和基于这个模型实现的音量控制方案

## 目录：

- [前言](#section1)
- [为什么需要开发一个轻量化的手势识别模型？](#section2)
- [为什么选择MediaPipe作为底层框架?](#section3)
- [手势识别模型的实现原理](#section4)
    - [1.总体模型架构](#subsection1)
    - [2.手掌检测模型的模型架构](#subsection2)
    - [3.关键点检测模型的模型架构](#subsection3)
- [数据集](#section5)
- [数据预处理](#section6)
- [训练结果](#section7)
- [整体实现流程图](#section8)
- [使用训练好的模型进行手势检测](#section9)
- [前景与展望](#section10)




## 前言：<a name="section1"></a>

⭐**本项目所用到的计算机视觉知识：图像归一化，边缘检测，图像平移，颜色直方图，图像梯度，卷积操作，池化操作**

本项目主要通过MediaPipe作为底层框架开发的手势识别模型实现了通过手势检测来控制电脑的音量。

MediaPipe是一种用于构建基于机器学习的视觉和感知应用程序的开源框架。它由Google开发和维护，旨在简化利用机器学习模型的过程，特别是在计算机视觉、音频处理和其他感知任务中。

MediaPipe提供了一套工具和库，使开发人员能够构建和部署各种感知应用程序，例如姿态检测、手势识别、人脸检测、物体追踪等。


## 为什么需要开发一个轻量化的手势识别模型？<a name="section2"></a>
因为当前市面上的手势识别模型存在以下的主要缺陷：

1. **依赖特殊硬件**：许多现有手部检测模型依赖专用硬件，如深度传感器，以实现准确的手部姿态估计。这限制了这些模型的通用性和可普及性，使其无法在普通移动设备上实时运行。

2. **性能要求较高**：一些解决方案在实时运行时对处理器性能要求较高，这导致它们只能在配备强大处理器的平台上运行，限制了其适用范围，尤其无法在普通移动设备上实现实时性能，限制了在AR/VR应用中提供实时、流畅的手部交互和沟通的能力。。

所以，我们迫切需要一个依赖特殊硬件、能够实时运行在普通移动设备上，并且能够进行多手追踪和2.5D手部姿态估计的手部检测解决方案。



## 为什么选择MediaPipe作为底层框架? <a name="section3"></a>

1. **模块化组件构建**: MediaPipe使用模块化组件（Calculators）构建手部跟踪模型，这种模块化结构有助于分解复杂任务为更小的、可管理的部分，使其更容易进行优化和扩展。

2. **优化计算任务**: MediaPipe使用称为Calculators的模块化组件来执行模型推断、媒体处理和数据转换等任务。这些Calculators可以定制以解决各种任务，包括模型推断。

3. **GPU加速优化**: MediaPipe针对特定计算任务（如裁剪、渲染和神经网络计算）进行了优化，以利用GPU加速。这意味着在支持GPU的设备上，特定组件能够更高效地利用GPU资源，从而加速处理速度。

4. **TFLite GPU推理**: MediaPipe利用TFLite（TensorFlow Lite）进行GPU推理，尤其在现代手机上，以进一步提高模型推理的速度和效率。

5. **跨平台支持**：MediaPipe支持多种平台，包括移动设备（如Android和iOS）、桌面计算机和嵌入式系统。这使开发者能够在不同设备上部署他们的应用程序，可以轻松部署在终端，云端，桌面端，web端，物联网端，Android端，iOS端。

总的来说，选择MediaPipe的目的是为了提升模型的实时性能以及优秀的跨平台支持



## 手势识别模型的实现原理 <a name="section4"></a>

### 1.总体模型架构<a name="subsection1"></a>

我们的手势识别方案是由两个模型共同工作的
- 手掌检测模型：将完整的图像输入到模型中，输出的是定位手掌的边界框(bounding box)。
- 手部关键点检测的模型：输入的是经过裁切的手掌的边界框的区域图像，输出的是高质量的2.5D的21个手部关键点

⭐如何提升模型的性能
- 在实时跟踪场景中，我们从前一帧的关键点预测中得出一个边界框，作为下一帧的输入，从而避免在每一帧上都使用手掌检测的模型。而是能够用上一帧关键点检测的结果预测出下一帧手掌的区域；我们只有当是第一帧图像或者没有检测到手部关键点的时候，才重新启动手掌检测的模型。**这样就可以大大节省时间和节省性能**。

### 2.手掌检测模型的模型架构<a name="subsection2"></a>
<div align="center">
<img src="https://github.com/LUORANCHENG/hand_gesture_detect/blob/main/picture/%E6%89%8B%E9%83%A8%E6%A3%80%E6%B5%8B%E6%A8%A1%E5%9E%8B.png" width="500" >
</div>

从figure2中我们可以看出手掌检测模型是一个编码器-解码器特征提取器，中间还有一些跨层连接。

这种编码器-解码器是一种用于处理多尺度物体检测任务的深度神经网络架构。它旨在解决在单一尺度下难以同时实现高分辨率和高级语义信息的问题。

首先编码器阶段通过卷积神经网络（CNN）层逐步减少特征图的分辨率，同时提取更高级的语义信息。解码器阶段则采用上采样等操作来逐步恢复特征图的分辨率，以实现更精细的特征表示。这样，在多个不同尺度上生成具有高级语义信息和适应不同目标尺寸的特征图。

**存在的问题以及我们的解决方案**

检测手是一个非常复杂的任务，首先，手是一个非常复杂的物体，有非常多的尺寸，还会存在遮挡和自遮挡，手的检测不同于人脸检测，人脸是一个具有高对比度的物体，比如眼睛，嘴巴......这些都可以辅助来检测人脸，但是手是一个动态的，非常的复杂，手并没有像眼睛和嘴那样有高对比度和置信度的特征。

但是我们这个方案就解决了上述的问题：

1.只检测手掌：首先我们训练的是一个手掌的检测器，而不是整个手部的检测器。因为手掌是一个正方形的物体，是一个类似于刚体的东西，并不会像手指那样灵活乱动，所以手掌是一个比较稳定的特征，而且手掌是一个更小的物体，所以在使用非极大值抑制的算法剔除重复预测的边界框的时候，在两只手握手和自遮挡的情况下能够表现的更好。

2.编码器-解码器特征提取器：我们使用类似于FPN的编码器-解码器特征提取器来进行大场景的上下文感知，即使是对小物体也会有很好的检测效果。

3.我们最小化的函数是一个名为focal-loss的损失函数。

<div align="center">
<img src="https://github.com/LUORANCHENG/hand_gesture_detect/blob/main/picture/focal%20loss.png" width="500" >
</div>

### 3.关键点检测模型的模型架构<a name="subsection3"></a>

<div align="center">
<img src="https://github.com/LUORANCHENG/hand_gesture_detect/blob/main/picture/%E6%89%8B%E9%83%A8%E5%85%B3%E9%94%AE%E7%82%B9%E6%A3%80%E6%B5%8B.png" width="500" >
</div>

在检测出手掌后，就可以对手部的关键点进行检测

把经过裁切的手掌的边界框的区域图像输入到手部关键点检测的模型当中来预测出21个2.5D的手部关键点坐标，由于预测出的坐标是一个连续的数值，所以这是一个回归问题，也就是通过学习一个函数，将手部区域内的每个关键点的坐标（x、y和相对深度）映射到实际的坐标值。**我们这个模型可以学习到手部姿态的本质内在的特征表示，也就是即使在被遮挡的情况下也可以准确识别出手部的关键点信息**。这个模型有三个输出，如figure3所示：
- 第一个输出是关于手部关键点的x,y,z三个坐标，其中z是相对于手腕关键点的深度信息。
- 第二个输出是手的置信度，也是当前检测的物体有多大的概率是一只手。当置信度低于设定的阈值，手掌检测模型将会被重新启用。
- 第三个输出是区别左手还是右手的标签。


**注意**：
- x和y的2D坐标是从真实世界采集的数据集和计算机生成的手部图像获取，而相对于手腕的深度坐标通过真实世界采集的数据是没有办法标注的，所以相对于手腕的深度坐标是仅通过电脑生成的图像获取。
- 我们通过训练了一个二分类的网络来区别是左手还是右手。

## 数据集<a name="section5"></a>
我们的数据集包含了三个部分，用于解决问题的不同方面：

- 真实世界采集的数据集:这个数据集包含6K张各种各样的图像，比如：各种地方，各种照明条件和手的外观。同时这个数据集它不包含复杂的手部姿势。

- 室内采集的手部数据集:这个数据集包含有10k张图像，在特定条件下拍摄，包含了所有物理上可能的手势的不同角度。这个数据集的局限性在于它只收集了30个人，背景变化有限。真实世界采集的数据集和室内采集的数据集是相互补充的，用于提升模型的鲁棒性。

- 电脑生成的手部数据集：这个数据集主要是为了获取真实的相对于手腕的深度信息的。这个数据集包含了10k张图片，并且背景的光照变化非常大，拍摄角度也是非常多变的。

<div align="center">
<img src="https://github.com/LUORANCHENG/hand_gesture_detect/blob/main/picture/%E6%95%B0%E6%8D%AE%E9%9B%86%E6%A0%B7%E4%BE%8B.png" width="500" >
</div>

对于训练手掌检测模型，我们只使用真实世界采集的数据集，这已经足够定位手的位置并且提供非常高的鲁棒性。

对于训练手部关键点模型，我们使用的是所有的数据集。

对于训练输出手置信度的模型，我们使用的是真实世界采集的数据集中的一些包含手的数据作为正样本，再用一些不包含手的区域的图作为负样本，然后用这两种样本训练一个二分类模型来判断手出现的置信度。

对于训练分类左右手的模型：我们对真实世界采集的数据集中的一小部分子集进行了标注左手和右手，然后用这些数据去训练二分类的模型，最后输出是左手还是右手。

## 数据预处理<a name="section6"></a>
- **大小调整**：我们需要将采集到的数据调整到256*256的大小，以适应于模型的输入。
- **旋转变换**：我们可以对图像进行旋转、翻转、平移操作，以增加模型的泛化性能。
- **数据归一化**：将数据缩放到0到1之间的范围，以避免特征之间的差异过大。
- **特征工程**：通过计算图像的颜色直方图，边缘检测来提取图像的特征。
- **批处理和输入格式设置**：将数据组织成适合模型输入的批处理格式，通常是一个张量（tensor）的集合，以便有效地输入到CNN中进行训练和特征提取。

## 训练结果<a name="section7"></a>
在训练手掌检测模型不同模型配置对准确率的影响：
模型配置 | 准确率
--- | ---
没有使用编码器+交叉熵损失函数 | 86.22%
使用编码器+交叉熵损失函数 | 94.07%
使用编码器+focal loss | 95.7%


手部关键点检测的模型表明：集合真实图片和电脑生成图片可以获得最佳的性能：
数据集 | MSE
--- | ---
只有真实世界数据集 | 16.1%
只有电脑合成的数据集 | 25.7%
两者结合 | 13.4%



注：MSE称为均方误差，是一种常用于回归问题的损失函数，用于衡量预测值与真实值之间的差异程度。值越小，代表模型性能越好。其公式如下：

<div align="center">
<img src="https://github.com/LUORANCHENG/hand_gesture_detect/blob/main/picture/MSE.png" width="500" >
</div>


MSE计算方法是将每个样本的预测值与真实值之差的平方求和，然后再取平均值。这样做可以使较大的误差对损失的贡献更显著，因为误差取平方后会放大。

## 整体实现流程图<a name="section8"></a>
<div align="center">
<img src="https://github.com/LUORANCHENG/hand_gesture_detect/blob/main/picture/%E6%B5%81%E7%A8%8B%E5%9B%BE.png" width="1000" >
</div>



## 使用训练好的模型进行手势检测 <a name="section9"></a>

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

## 前景与展望 <a name="section10"></a>

由于MediaPipe提供的手势识别模型具有极佳的实时性能，可以在低延迟的情况下处理图像和视频数据，以及优秀的跨平台支持，使得这个模型具有非常广阔的应用情景，本项目所实现的控制音量功能仅仅是一个非常小的应用场景，下面列举了一些潜在的应用：
- 手势控制：隔空截屏翻页，无人机控制，智能家居控制
- 动作捕捉：电影拍摄，短视频特效，增强现实
- 手部穴位按摩机器人，手部针灸机器人，康复精准医疗
- 手语翻译，手势翻译











