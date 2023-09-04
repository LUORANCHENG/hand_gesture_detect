# 基于Mediapipe实现的手势识别控制电脑音量

目录：

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

## 为什么选择MediaPipe? <a name="section2"></a>

⭐1. 实时性能：MediaPipe专注于实时性能，可以在低延迟的情况下处理图像和视频数据。这使其适用于需要快速响应的应用程序，如虚拟现实、增强现实和实时手势识别。

2. 跨平台支持：MediaPipe支持多种平台，包括移动设备（如Android和iOS）、桌面计算机和嵌入式系统。这使开发者能够在不同设备上部署他们的应用程序。

3. 预训练模型：MediaPipe提供了大量预训练的机器学习模型，用于执行各种视觉和感知任务。这些模型可以快速集成到应用程序中，无需从头开始训练。

4. 多媒体输入支持：MediaPipe能够处理图像、视频、音频等多媒体输入，使其适用于各种多媒体应用。

5. 灵活性：MediaPipe提供了丰富的工具和库，允许开发者自定义和扩展现有的模型和组件，以满足其特定需求。



## MediaPipe手势识别的一般实现原理 <a name="section3"></a>

MediaPipe的手势识别模型使用深度学习技术：卷积神经网络（Convolutional Neural Networks，CNNs），来检测和识别手部的关键点和手势。下面是它的一般工作流程：

1. **图像输入**：首先，需要提供一个图像或者视频帧，这通常是通过摄像头捕获的图像。模型会将这个图像作为输入。

2. **图像预处理**：输入图像会经过一些预处理步骤，以确保模型可以正确处理它。这可能包括图像尺寸的调整、归一化、去除噪声等处理。

3. **卷积神经网络**：接下来，输入图像会经过卷积神经网络（CNN）的层次结构。CNN通过多层卷积和池化层来提取图像中的特征。这些特征可能包括边缘、纹理、形状等信息，有助于模型识别手部。在CNN的最后一层，模型通过一些后续层来检测手部区域。这些后续层可能包括卷积层、全连接层和激活函数。模型的输出是一个热图（heatmap），它显示了图像上可能包含手部的区域。

4. **阈值处理**：根据热图，模型使用阈值处理来确定手部的位置。通常，热图中的像素值高于某个阈值被认为是手部区域。 

5. **手部关键点检测**：一旦确定手部区域，模型可以进一步检测手部关键点，如手指关节的位置。这些关键点的位置可以用来表示手部的姿势和手势。

6. **手势识别**：使用检测到的手部关键点，可以实现手势识别逻辑，以识别特定的手势。

## 具体的实现细节 <a name="section4"></a>

### 1.图像预处理过程中的具体细节 <a name="subsection1"></a>

1. **图像调整大小**：首先，输入图像通常会被调整为模型期望的大小。这通常是为了确保模型能够以一致的输入尺寸进行处理。通常情况下，图像大小会被调整为网络所需的大小，例如300x300像素。

2. **归一化**：图像通常需要被归一化，以确保像素值在一定的范围内，通常是0到1之间或-1到1之间。这可以帮助训练模型更快地收敛并提高模型的稳定性。

3. **数据增强**：通过对输入图像进行随机变换，如旋转、翻转、缩放和亮度调整，来增加模型的鲁棒性。这可以帮助模型在不同条件下更好地泛化。

4. **颜色空间转换**：有时候，图像可能需要从RGB颜色空间转换到其他颜色空间，例如灰度图像或者HSV颜色空间。这可以有助于突出特定的图像特征。

5. **均值减法**：减去图像的均值是一种常见的预处理步骤，以减小图像中的亮度和颜色差异。这可以帮助模型更好地捕获图像的结构。

6. **像素值缩放**：有时候，图像的像素值会被缩放到特定的范围内，以匹配模型的输入要求。

7. **其他预处理**：根据模型的具体要求和应用场景，还可以执行其他预处理步骤，如图像剪裁、去噪等。

这些预处理步骤有助于确保输入图像的质量和一致性，以便模型能够更好地识别手势。不同的应用可能需要不同的预处理步骤，具体的预处理步骤通常会在模型训练和部署过程中进行优化和确定。

### 2.通过CNN（卷积神经网络）来提取手部边缘、纹理、形状等信息⭐  <a name="subsection2"></a>

1. **卷积层提取特征**：CNN的卷积层用于提取图像的局部特征，包括边缘、纹理和形状等信息。这是通过将一系列卷积核应用于输入图像来实现的。每个卷积核在输入图像上滑动，计算与卷积核匹配的图像区域和卷积核之间的点积，从而生成特征图。这些特征图包含了在不同位置检测到的局部特征。

<div align="center">
<img src="https://github.com/LUORANCHENG/hand_gesture_detect/blob/main/picture/convoltion.gif" width="500" >
</div>

2. **池化层减小维度**：池化层通常紧接在卷积层之后，用于减小特征图的维度，同时保留重要的信息。最常见的池化操作是最大池化，它选择每个池化窗口中的最大值。这有助于减小模型的计算复杂性，并提高模型的稳定性。

<div align="center">
<img src="https://github.com/LUORANCHENG/hand_gesture_detect/blob/main/picture/pooling.gif" width="500" >
</div>

3. **多层堆叠**：通常，CNN会堆叠多个卷积层和池化层，以逐渐提取更高级别的特征。低层次的卷积层可能会提取边缘和纹理等低级特征，而高层次的卷积层可能会捕获更抽象的形状和模式。

4. **全连接层进行分类**：在CNN的最后几层通常会添加全连接层，用于将卷积层的输出映射到最终的输出类别。对于手部边缘、纹理、形状等特征的提取，这些全连接层可能用于执行分类任务，将提取的特征与手势类别相关联。

5. **激活函数引入非线性性质**：在卷积层和全连接层之后，通常会应用非线性激活函数，如ReLU（Rectified Linear Unit），以引入非线性性质，从而增强模型的表达能力。

6. **正则化和批量归一化**：为了提高模型的泛化能力和稳定性，可以添加正则化层和批量归一化层。

7. **特征图可视化**：有时，可以通过可视化CNN中间层的特征图来理解模型如何提取不同级别的特征。这有助于了解模型在图像中检测到的信息。

### 3.进一步检测手部关键点 <a name="subsection3"></a>

在CNN模型之后添加一个用于检测手部关键点的回归层。（通常使用卷积神经网络（CNN）来提取图像特征，然后添加一个用于检测手部关键点的回归层）

回归层接收CNN提取的特征并将其映射到手部关键点的坐标。最后一层输出的大小将与手部关键点的数量相匹配，以便在训练期间预测每个关键点的坐标。

示例代码：

```
# 关键点检测部分
self.keypoints = nn.Sequential(
    nn.Linear(in_features=32 * 64 * 64, out_features=128),
    nn.ReLU(),
    nn.Linear(in_features=128, out_features=num_keypoints),  # num_keypoints为手部关键点的数量
)

```

`nn.Linear`：`nn.Linear`是一个全连接层，用于执行线性变换。它有两个参数，`in_features`表示输入特征的数量，`out_features`表示输出特征的数量。在这里，第一个线性层将输入特征的数量从 `32 * 64 * 64`（CNN输出的扁平化特征）降低到 `128`，第二个线性层将其进一步降低到 `num_keypoints`，这是手部关键点的数量。

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











