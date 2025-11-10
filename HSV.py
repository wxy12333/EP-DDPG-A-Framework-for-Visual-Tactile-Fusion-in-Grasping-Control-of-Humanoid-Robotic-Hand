import numpy as np
import cv2
import json
from os import path
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.require('sim')

transform_params_path = 'transform_params.json'
if path.exists(transform_params_path):
    with open(transform_params_path, 'r') as openfile:
        transform_params = json.load(openfile)

visionSensorHandle_rgb = sim.getObject('/kinect/rgb')
resolution_rgb, rawimage_rgb = sim.getVisionSensorImg(visionSensorHandle_rgb, 0)

visionSensorHandle_dep = sim.getObject('/kinect/depth')
resolution_dep, rawimage_dep = sim.getVisionSensorImg(visionSensorHandle_dep, 0)

def readVisionSensor():
    img, [resX, resY] = sim.getVisionSensorImg(visionSensorHandle_rgb)
    img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)

    img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)

    return img

def readDepthSensor():
    # 获取 Depth Info
    depth_buffer, [resX, resY] = sim.getVisionSensorDepth(visionSensorHandle_dep)
    depth_img = np.frombuffer(depth_buffer, dtype=np.float32)
    depth_img.shape = (resY, resX)
    zNear = 0.01
    zFar = 3.5
    depth_img = depth_img * (zFar - zNear) + zNear
    depth_img = cv2.flip(depth_img, 0)

    # 归一化深度值到 [0, 1] 范围
    depth_img_normalized = cv2.normalize(depth_img, None, 0, 1, cv2.NORM_MINMAX)

    # 调整对比度（使图像更暗）
    contrast = 1.8  # 对比度系数，小于 1 会使图像更暗
    depth_img_adjusted = depth_img_normalized * contrast

    # 将图像转换为 8 位无符号整数格式
    depth_img_8bit = np.uint8(depth_img_adjusted * 255)
    return depth_img_8bit

def canny_detect(img, LowerRegion, upperRegion):
    # if roi:
    #     x1, y1, x2, y2 = roi
    #     img = img[y1:y2, x1:x2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    object_kernel = np.ones((transform_params['object_morph_kernel'], transform_params['object_morph_kernel']), "uint8")

    # 橙色
    # orange_LowerRegion = np.array([78, 43, 46], np.uint8)
    # orange_upperRegion = np.array([99, 255, 255], np.uint8)

    # 腐蚀、膨胀、获得掩膜、中值滤波
    orange_mask = cv2.morphologyEx(hsv, cv2.MORPH_OPEN, object_kernel)
    orange_mask = cv2.morphologyEx(hsv, cv2.MORPH_CLOSE, object_kernel)
    orange_mask = cv2.inRange(orange_mask, LowerRegion, upperRegion)
    orange_mask = cv2.medianBlur(orange_mask, 7)


    # canny边缘检测
    edges = cv2.Canny(orange_mask, transform_params['object_canny_param1'],
                             transform_params['object_canny_param2'])
    _, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # output_orange = img.copy()
    return contours

def get_color_angle(img, contours, color, min, max):
    for contour in contours:
        # 计算外接矩形
        x, y, w, h = cv2.boundingRect(contour)

        min_area_threshold = 0  # 根据实际情况调整阈值
        max_area_threshold = 200  # 根据实际情况调整阈值
        # 计算外接矩形的面积
        area = w * h

        # 如果面积超过阈值，则绘制检测框和添加文本标注
        if min < area < max:
            # 在图像上绘制矩形
            cv2.rectangle(img, (x - 2, y - 2), (x + w + 2, y + h + 2), color, 1)
            # cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def get_data():
    color_raw = readVisionSensor()
    depth_raw = readDepthSensor()
    # 青色
    contours_cyan = canny_detect(color_raw, np.array([78, 43, 46], np.uint8), np.array([99, 255, 255], np.uint8))
    get_color_angle(color_raw, contours_cyan, [255, 255, 0], 0, 500)
    # 黄色
    contours_yellow = canny_detect(color_raw, np.array([26, 43, 46], np.uint8), np.array([34, 255, 255], np.uint8))
    get_color_angle(color_raw, contours_yellow, [0, 255, 255], 0, 500)
    # 绿色
    contours_green = canny_detect(color_raw, np.array([35, 43, 46], np.uint8), np.array([77, 255, 255], np.uint8))
    get_color_angle(color_raw, contours_green, [0, 255, 0], 0, 500)
    # 蓝色
    contours_blue = canny_detect(color_raw, np.array([100, 43, 46], np.uint8), np.array([124, 255, 255], np.uint8))
    get_color_angle(color_raw, contours_blue, [255, 0, 0], 0, 500)
    # 紫色
    contours_purple = canny_detect(color_raw, np.array([125, 43, 46], np.uint8), np.array([155, 255, 255], np.uint8))
    get_color_angle(color_raw, contours_purple, [128, 0, 128], 0, 500)

    # 使用opencv显示图片
    cv2.imshow('rgb', color_raw)
    cv2.imshow('depth', depth_raw)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # filename = f"rgb_snapshot1.png"
        # # 保存 RGB 图像
        # cv2.imwrite(filename, color_raw)
        return False
    return True


if __name__ == '__main__':

    try:
        i = 0
        while True:
            # 获取并显示数据
            if not get_data():
                break
            i += 1
    finally:
        # 释放资源
        cv2.destroyAllWindows()