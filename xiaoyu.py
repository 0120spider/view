import cv2
import numpy as np

# 打开摄像头
camera = cv2.VideoCapture(0)

# 判断摄像头是否打开成功
if not camera.isOpened():
    print('摄像头未打开')
else:
    print('摄像头已打开')

# 获取视频帧尺寸
frame_size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print('视频帧尺寸：' + repr(frame_size))

# 创建椭圆形结构元素，用于后续形态学操作
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))

# 初始化背景帧
background = None

while True:
    # 读取视频流中的一帧
    grabbed, frame = camera.read()

    # 如果读取失败，退出循环
    if not grabbed:
        break

    # 复制当前帧
    frame_copy = frame.copy()

    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 进行高斯模糊，去除噪点Q
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # 如果背景帧为空，将当前帧作为背景帧
    if background is None:
        background = gray
        continue

    # 计算当前帧与背景帧的差异
    diff = cv2.absdiff(background, gray)

    # 对差异图像进行二值化处理
    diff = cv2.threshold(diff, 5, 255, cv2.THRESH_BINARY)[1]

    # 进行形态学膨胀操作，填充目标区域
    diff = cv2.dilate(diff, kernel, iterations=3)

    # 计算轮廓
    image, contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历所有轮廓
    for i, c in enumerate(contours):
        # 如果轮廓面积过小，忽略该轮廓
        if cv2.contourArea(c) < 350:
            continue

        # 计算轮廓的外接矩形框
        (x, y, w, h) = cv2.boundingRect(c)

        # 绘制矩形框
        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 计算矩形框中心点坐标
        cx, cy = x + w // 2, y + h // 2

        # 绘制文本信息
        cv2.putText(frame_copy, str(i+1), (cx, cy), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.5, (0, 0, 255), 1)

    # 显示结果图像
    cv2.imshow('Fish Detection', frame_copy)
    cv2.imshow('Diff', diff)

    # 更新背景帧
    background = gray

    # 等待按键，按下q键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源
camera.release()

# 关闭所有窗口
cv2.destroyAllWindows()
