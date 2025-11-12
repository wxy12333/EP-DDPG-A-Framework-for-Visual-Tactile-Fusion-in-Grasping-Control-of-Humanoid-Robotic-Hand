from os import path
import numpy as np
import cv2
import time
import json
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import random

client = RemoteAPIClient()
sim = client.require('sim')


CAMERA = "kinect"
JOINT_NAME = "joint"
SCENE_CONTROLLER_NAME = "SceneController"

JOINT_RANGE_FUNC_NAME = "getJointRange"

RESET_FUNC_NAME = "resetItems"

JOINT_COUNT = 7

class Robot():

    def __init__(self):
        transform_params_path = 'transform_params.json'
        if path.exists(transform_params_path):
            with open(transform_params_path, 'r') as openfile:
                self.transform_params = json.load(openfile)
        self.synchronous = False
        self.current_step = 0
        self.joint_names = [
            './Thumb_MCP_Spin', './Thumb_MCP', './Thumb_PIP', './Index_MCP', './Index_PIP',
            './Middle_MCP', './Middle_PIP', './Ring_MCP', './Ring_PIP', './Little_MCP', './Little_PIP'
        ]
        self.object_names = [
            './object1', './object2', './object3', './object4', './object5', './object6', './object7'
            , './object8', './object9', './object10', './object11'
            # './test1', './test2', './test3'
            # , './test4', './test5', './test6', './test7', './test8', './test9', './test10', './tes11'
        ]
        self.Angle_thumb = [0, 0]
        self.Angle_index = [0, 0]
        self.Angle_middle = [0, 0]
        self.Angle_ring = [0, 0]
        self.Angle_little = [0, 0]
        self.objects = []
        self.cameras = []
        self.hand_joints = []
        self.arm_joints = []
        self.joint_angle = [0, 0, 0, 0, 0]
        self.joint_ranges = np.zeros((JOINT_COUNT, 2), dtype=np.float32)
        # self.default_arm_pos = np.asarray([1.328, -26.197, -2.626, -75.971, -1.517, -49.807, 2.008])
        self.default_arm_pos = [0.073, 0.093, 0.083, 0.11, 0.063, 0.065, 0.063, 0.063, 0.063, 0.113, 0.063,
                                0.074, 0.063, 0.105, 0.06, 0.063, 0.064, 0.093, 0.069, 0.06, 0.09, 0.063]
        self.scene_controller = 0
        self.stereo_matcher = cv2.StereoBM_create(numDisparities=48, blockSize=11)
        self.default_hand_pos = np.asarray([50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.initial_expected_angle = [0, 0, 0, 0, 0]
        self.initial_w = [0, 0, 0, 0, 0]
        self.z = 0
        self.default_obj_pos = [
            [-0.3, -6.675, 0.115], [-0.173, -6.675, 0.083], [-0.085, -6.675, 0.033], [0.005, -6.675, 0.051],
            [0.110, -6.675, 0.08], [0.220, -6.675, 0.0175], [0.323, -6.675, 0.02767], [0.418, -6.675, 0.032],
            [0.518, -6.675, 0.076], [0.638, -6.675, 0.114], [0.765, -6.675, 0.078],
            [-0.250, -6.975, 0.03], [-0.120, -6.975, 0.07], [0.01, -6.975, 0.0525], [0.1, -6.975, 0.1],
            [0.2, -6.975, 0.0465], [0.32, -6.975, 0.12], [0.46, -6.975, 0.05], [0.6, -6.975, 0.017],
            [0.8, -6.975, 0.023], [0.725, -6.38, 0.05], [0.525, -6.355, 0.09]
        ]  # 替换为物体1的初始位置 4:50,6:51,9:-23,10:115,11:60
        self.grasp_obj_pos = [
            [0.256, -5.064, 0.1], [0.263, -5.049, 0.083], [0.263, -5.064, 0.11], [0.263, -5.064, 0.13],
            [0.263, -5.073, 0.08], [0.258, -5.067, 0.095], [0.256, -5.064, 0.102], [0.26, -5.064, 0.115],
            [0.256, -5.064, 0.076], [0.245, -5.064, 0.112], [0.2552, -5.066, 0.078],
            [0.26, -5.074, 0.113], [0.263, -5.064, 0.07], [0.256, -5.064, 0.119], [0.263, -5.072, 0.1],
            [0.257, -5.064, 0.103], [0.259, -5.067, 0.12], [0.255, -5.060, 0.116], [0.269, -5.072, 0.1],
            [0.2556, -5.074, 0.1033], [0.2746, -5.0675, 0.12], [0.263, -5.073, 0.09]
        ]
        self.grasp_obj_ori = [
            [3.14, 0, -1.57], [0, 1.57, 1.57], [0.094, -0.112, -2.421], [0, 0, 0],
            [0, 0, 0.872], [-1.57, 0.89, -1.57], [-1.5708, 0, -1.5708], [-2.632, -0.8375, 1.5035],
            [0, 0, -0.401], [3.14, 0, 2.006], [0, 0, 1.099],
            [1.57, 0.5, 0], [0, 0, -1.57], [0, 0, -1.57], [0, 0, 0],
            [0, 0, 0], [0, -0.1634, 1.465], [0, -1.57, 0], [1.941, 1.06, -0.3506],
            [-3.0946, 1.2, -1.558], [1.57, 0.872, 1.57], [1.57, -0.523, 1.57]
        ] #-1.57, -0.926, 1.57
        self.w = np.array([-1, -0.01, 0.1, 0.2, 0.5, 1])

        self.force_threshold = [
            [0.1, 0.8, 1.1, 1.6], [0.1, 2, 4, 7.5], [0.05, 0.25, 0.4, 0.7], [0.1, 0.45, 0.7, 1.1],
            [0.1, 0.45, 0.7, 1.2], [0.1, 5, 8, 12], [0.1, 1.5, 2.5, 4.5], [0.1, 1.5, 2.5, 4.5],
            [0.01, 0.45, 0.8, 1.2], [0.1, 1, 1.3, 2.1], [1, 35, 45, 75],
            [0.05, 0.15, 0.35, 0.7], [0.01, 0.1, 0.2, 0.5], [0.1, 0.8, 1.5, 2.5], [0.1, 0.8, 1.1, 2.5],
            [0.1, 0.5, 1.1, 2.2], [0.01, 0.3, 0.6, 0.9], [0.1, 1.2, 2, 3], [0.1, 1, 2.4, 3.5],
            [0.05, 0.15, 0.25, 0.7], [0.05, 0.15, 0.4, 3], [0.1, 0.45, 0.7, 1.2]
        ]

        self.Theta = 27
        self.MCP = [0, 0, 0]
        self.PIP = [0, 0, 0]
        self.K = np.array([
            [589.32, 0, 320],
            [0, 589.32, 240],
            [0, 0, 1]
        ])
        self.res = 0

        # if self.is_connected:
        print('Connected to remote API server')

        self.visionSensorHandle_rgb = sim.getObject('/kinect/rgb')
        resolution_rgb, rawimage_rgb = sim.getVisionSensorImg(self.visionSensorHandle_rgb, 0)

        self.visionSensorHandle_dep = sim.getObject('/kinect/depth')
        resolution_dep, rawimage_dep = sim.getVisionSensorImg(self.visionSensorHandle_dep, 0)

        self.camera_handle = sim.getObject('./kinect/depth')
        self.end_effector_handle = sim.getObject('./link8_resp/connection')
        self.force_t = sim.getObject('./Force_t')
        self.force_i = sim.getObject('./Force_i')
        self.force_m = sim.getObject('./Force_m')
        self.force_r = sim.getObject('./Force_r')
        self.force_l = sim.getObject('./Force_l')
        self.bottom = sim.getObject('./bottom')
        # print(self.force_l)
        self.target_pos = sim.getObject('./Target')
        # res, MCP_t0 = sim.simxGetObjectHandle(self.client, './Thumb_MCP_Spin', sim.simx_opmode_blocking)
        # print(MCP_t0)
        for i in range(1, JOINT_COUNT + 1):
            id = sim.getObject('./' + JOINT_NAME + str(i))
            self.arm_joints.append(id)
        for joint_name in self.joint_names:
            joint_handle = sim.getObject(joint_name)
            if not (joint_handle == -1):
                self.hand_joints.append(joint_handle)
            else:
                print(f"Failed to get handle for {joint_name}")
        for object_name in self.object_names:
            object_handle = sim.getObject(object_name)
            if not (object_handle == -1):
                self.objects.append(object_handle)
            else:
                print(f"Failed to get handle for {object_name}")
        # self.reset()
        # res, object11 = sim.getObject('./object11')
        # res, object12 = sim.getObject('./object12')
        # self.objects.append(object11)
        # self.objects.append(object12)
        # else:
        #     print('Failed connecting to remote API server')


    def readVisionSensor(self):
        img, [resX, resY] = sim.getVisionSensorImg(self.visionSensorHandle_rgb)
        img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)

        img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)

        return img

    def readDepthSensor(self):
        # 获取 Depth Info
        depth_buffer, [resX, resY] = sim.getVisionSensorDepth(self.visionSensorHandle_dep)
        depth_img = np.frombuffer(depth_buffer, dtype=np.float32)
        depth_img.shape = (resY, resX)
        zNear = 0.01
        zFar = 3.5
        depth_img = depth_img * (zFar - zNear) + zNear
        depth_img = cv2.flip(depth_img, 0)
        return depth_img

    def get_transform_matrix(self, object_handle):
        # 获取位置和方向
        position = sim.getObjectPosition(object_handle, sim.handle_world)
        orientation = sim.getObjectOrientation(object_handle, sim.handle_world)

        # 创建齐次变换矩阵
        T = np.eye(4)
        T[0:3, 3] = position

        # 旋转矩阵
        alpha, beta, gamma = orientation
        Rz = np.array([
            [np.cos(0), -np.sin(0), 0],
            [np.sin(0), np.cos(0), 0],
            [0, 0, 1]
        ])
        Ry = np.array([
            [np.cos(beta), 0, np.sin(beta)],
            [0, 1, 0],
            [-np.sin(beta), 0, np.cos(beta)]
        ])
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha), np.cos(alpha)]
        ])
        R = Rx @ Ry @ Rz

        T[0:3, 0:3] = R
        return T

    def transform_point(self, camera_point, transformation_matrix):
        # 将相机坐标转换为齐次坐标
        camera_point_homogeneous = np.append(camera_point, 1)

        # 进行坐标变换
        end_effector_point_homogeneous = np.dot(transformation_matrix, camera_point_homogeneous)

        # 提取末端执行器坐标
        end_effector_point = end_effector_point_homogeneous[:3]

        return end_effector_point

    def Hand_IK_Cauculate(self, object_handle1, object_handle2, end_position):
        position_end = sim.getObjectPosition(self.end_effector_handle, sim.handle_world)
        position_M = sim.getObjectPosition(object_handle1, sim.handle_world)
        position_P = sim.getObjectPosition(object_handle2, sim.handle_world)

        position_end = np.array(position_end)
        position_M = np.array(position_M)
        position_P = np.array(position_P)

        fingertip = np.array(end_position)
        delta = np.around((position_M - position_end), 4)
        self.MCP[0] = fingertip[0]
        self.MCP[1] = delta[1]
        self.MCP[2] = delta[0]
        delta1 = np.around((position_P - position_end), 4)
        self.PIP[0] = fingertip[0]
        self.PIP[1] = delta1[1]
        self.PIP[2] = delta1[0]

        L1 = np.sqrt(np.square(self.PIP[1] - self.MCP[1]) + np.square(self.PIP[2] - self.MCP[2]))
        L1 = np.around(L1 * 100, 2)
        L2 = np.sqrt(np.square(self.PIP[1] - fingertip[1]) + np.square(self.PIP[2] - fingertip[2]))
        L2 = np.around(L2 * 100, 2)
        t = np.arctan((self.PIP[1] - fingertip[1]) / (fingertip[2] - self.PIP[2]))
        t = np.degrees(t)

        x = np.around((fingertip[2] - self.MCP[2]) * 100, 2)
        y = np.around((np.abs(fingertip[1] - self.MCP[1])) * 100, 2)

        cos_theta2 = (np.square(x) + np.square(y) - np.square(L1) - np.square(L2)) / (2 * L1 * L2)
        theta2 = np.arccos(cos_theta2)
        theta2 = np.around(np.degrees(theta2) - self.Theta, 2)
        theta1 = theta2 * (75 / 68)

        return theta1, theta2

    def Hand_IK_Cauculate_Thumb(self, object_handle1, object_handle2, end_position):
        position_end = sim.getObjectPosition(self.end_effector_handle, sim.handle_world)
        position_M = sim.getObjectPosition(object_handle1, sim.handle_world)
        position_P = sim.getObjectPosition(object_handle2, sim.handle_world)
        position_S = sim.getObjectPosition(self.hand_joints[0], sim.handle_world)

        # 计算mcp旋转矩阵
        T_wm = self.get_transform_matrix(object_handle1)
        T_we = self.get_transform_matrix(self.end_effector_handle)
        T_me = np.dot(np.linalg.inv(T_wm), T_we)

        position_Pend = self.transform_point(position_P, T_we)
        position_Mend = self.transform_point(position_M, T_we)
        position_Send = self.transform_point(position_S, T_we)

        v1 = np.array(position_Mend) - np.array(position_Send)
        v2 = np.array(position_Pend) - np.array(position_Mend)
        v3 = np.array(end_position) - np.array(position_Pend)

        # 计算点积
        dot_product = np.dot(v1, v2)
        # 计算向量的范数
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        # 计算余弦值
        cos_theta = dot_product / (norm_v1 * norm_v2)
        # 使用 arccos 计算弧度制的夹角
        angle_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        # 将弧度制的夹角转换为角度制
        angle_degrees1 = np.degrees(angle_radians) - 28
        angle_degrees2 = angle_degrees1 * (16 / 15)

        return angle_degrees1, angle_degrees2

    def canny_detect(self, img, LowerRegion, upperRegion):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        object_kernel = np.ones((self.transform_params['object_morph_kernel'], self.transform_params['object_morph_kernel']),
                                "uint8")

        # 橙色
        # orange_LowerRegion = np.array([78, 43, 46], np.uint8)
        # orange_upperRegion = np.array([99, 255, 255], np.uint8)

        # 腐蚀、膨胀、获得掩膜、中值滤波
        orange_mask = cv2.morphologyEx(hsv, cv2.MORPH_OPEN, object_kernel)
        orange_mask = cv2.morphologyEx(hsv, cv2.MORPH_CLOSE, object_kernel)
        orange_mask = cv2.inRange(orange_mask, LowerRegion, upperRegion)
        orange_mask = cv2.medianBlur(orange_mask, 7)

        # canny边缘检测
        edges = cv2.Canny(orange_mask, self.transform_params['object_canny_param1'],
                          self.transform_params['object_canny_param2'])
        _, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # output_orange = img.copy()
        return contours

    def get_color_angle(self, img, depth, contours, T_ec, file_path, MCP, PIP, text, delta):
        for contour in contours:
            # 计算外接矩形
            x, y, w, h = cv2.boundingRect(contour)

            min_area_threshold = 300  # 根据实际情况调整阈值
            # 计算外接矩形的面积
            area = w * h

            # 如果面积超过阈值，则绘制检测框和添加文本标注
            if area < min_area_threshold:
                # 在图像上绘制矩形
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                depth_value = depth[y + delta[0], x + delta[1]]
                depth_value = round(depth_value, 4)
                X = round((x - self.K[0][2]) * depth_value / self.K[0][0], 4)
                Y = round((y - self.K[1][2]) * depth_value / self.K[0][0], 4)

                camera_point = np.array([X, Y, depth_value])
                end_effector_point = self.transform_point(camera_point, T_ec)
                end_effector_point = np.around(end_effector_point, 4)
                theta1, theta2 = self.Hand_IK_Cauculate(MCP, PIP, end_effector_point)
                Angle = np.around(np.array([theta1, theta2]), 2)
                # with open(file_path, 'w') as file:
                #     file.truncate()  # 清除文件中之前的内容
                #     file.seek(0)  # 回到文件开头
                #     file.write(f"{Angle}\n")
                #     file.flush()  # 确保立即写入文件
                #     # time.sleep(1)  # 每秒写入一个参数
                cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                return Angle

    def image(self):
        self.img = self.readVisionSensor()
        self.depth = self.readDepthSensor()

    def get_all_handles(self):
        self.hand_joints = []
        for joint_name in self.joint_names:
            joint_handle = sim.getObject(joint_name)
            if joint_handle == -1 or joint_handle is None:
                print(f"Failed to get handle for {joint_name}, retrying...")
                # 这里可以尝试等待1秒再试
                time.sleep(5)
                joint_handle = sim.getObject(joint_name)
                if joint_handle == -1 or joint_handle is None:
                    raise Exception(f"Critical: Failed to get handle for {joint_name} after retry.")
            self.hand_joints.append(joint_handle)
    def finger(self):
        T_wc = self.get_transform_matrix(self.camera_handle)
        T_we = self.get_transform_matrix(self.end_effector_handle)
        # 计算相机坐标系到末端执行器坐标系的变换矩阵
        T_ec = np.dot(np.linalg.inv(T_we), T_wc)

        # print('Transform matrix from camera to end-effector:')
        # depth = readDepthSensor()
        # print(depth)
        # cv2.imshow("image", img)

        # 橙色
        contours_orange = self.canny_detect(self.img, np.array([78, 43, 46], np.uint8), np.array([99, 255, 255], np.uint8))
        # 黄色
        contours_yellow = self.canny_detect(self.img, np.array([26, 43, 46], np.uint8), np.array([34, 255, 255], np.uint8))
        # 绿色
        contours_green = self.canny_detect(self.img, np.array([35, 43, 46], np.uint8), np.array([77, 255, 255], np.uint8))
        # 蓝色
        contours_blue = self.canny_detect(self.img, np.array([100, 43, 46], np.uint8), np.array([124, 255, 255], np.uint8))
        # 紫色
        contours_purple = self.canny_detect(self.img, np.array([125, 43, 46], np.uint8), np.array([155, 255, 255], np.uint8))

        # 拇指角度
        for contour in contours_yellow:
            # 计算外接矩形
            x_y, y_y, w_y, h_y = cv2.boundingRect(contour)
            min_area_threshold = 2000  # 根据实际情况调整阈值
            # 计算外接矩形的面积
            area = w_y * h_y
            # 如果面积超过阈值，则绘制检测框和添加文本标注
            if area < min_area_threshold:
                # 在图像上绘制矩形
                cv2.rectangle(self.img, (x_y, y_y), (x_y + w_y, y_y + h_y), (0, 255, 0), 2)
                depth_value = self.depth[y_y, x_y]
                depth_value = round(depth_value, 4)
                X_y = round((x_y - self.K[0][2]) * depth_value / self.K[0][0], 4)
                Y_y = round((y_y - self.K[1][2]) * depth_value / self.K[0][0], 4)

                camera_point = np.array([X_y, Y_y, depth_value])
                end_effector_point_y = self.transform_point(camera_point, T_ec)
                end_effector_point_y = np.around(end_effector_point_y, 4)
                theta1, theta2 = self.Hand_IK_Cauculate_Thumb(self.hand_joints[1], self.hand_joints[2], end_effector_point_y)
                self.Angle_thumb = np.around(np.array([theta1, theta2]), 2)

                # 添加文本标注
                cv2.putText(self.img, 'yellow', (x_y, y_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 食指角度
        Angle_index = self.get_color_angle(self.img, self.depth, contours_purple, T_ec, file_path_i, self.hand_joints[3],
                                           self.hand_joints[4], 'purple', [0, 0])
        # 中指角度
        Angle_middle = self.get_color_angle(self.img, self.depth, contours_orange, T_ec, file_path_m, self.hand_joints[5],
                                            self.hand_joints[6], 'cyan', [10, 10])
        # 无名指角度
        Angle_ring = self.get_color_angle(self.img, self.depth, contours_green, T_ec, file_path_r, self.hand_joints[7],
                                          self.hand_joints[8], 'green', [0, 0])
        # 小指角度
        Angle_little = self.get_color_angle(self.img, self.depth, contours_blue, T_ec, file_path_l, self.hand_joints[9],
                                            self.hand_joints[10], 'blue', [0, 0])

        return self.Angle_thumb, Angle_index, Angle_middle, Angle_ring, Angle_little

    def get_tactile_feedback(self):
        desired_force_dir = [0, 0, 1]
        cos = [0, 0, 0, 0, 0]
        result, force_vector_t, torque = sim.readForceSensor(self.force_t)
        result, force_vector_i, torque = sim.readForceSensor(self.force_i)
        result, force_vector_m, torque = sim.readForceSensor(self.force_m)
        result, force_vector_r, torque = sim.readForceSensor(self.force_r)
        result, force_vector_l, torque = sim.readForceSensor(self.force_l)
        force_t = np.linalg.norm(force_vector_t)
        force_i = np.linalg.norm(force_vector_i)
        force_m = np.linalg.norm(force_vector_m)
        force_r = np.linalg.norm(force_vector_r)
        force_l = np.linalg.norm(force_vector_l)
        cos[0] = np.dot(force_vector_t, desired_force_dir) / (force_t * np.linalg.norm(desired_force_dir) + 1e-6)
        cos[1] = np.dot(force_vector_i, desired_force_dir) / (force_i * np.linalg.norm(desired_force_dir) + 1e-6)
        cos[2] = np.dot(force_vector_m, desired_force_dir) / (force_m * np.linalg.norm(desired_force_dir) + 1e-6)
        cos[3] = np.dot(force_vector_r, desired_force_dir) / (force_r * np.linalg.norm(desired_force_dir) + 1e-6)
        cos[4] = np.dot(force_vector_l, desired_force_dir) / (force_l * np.linalg.norm(desired_force_dir) + 1e-6)
        # print(force_t, force_i)
        return force_t, force_i, force_m, force_r, force_l, cos

    def get_finger_angel(self):
        finger_pos = []
        T, I, M, R, L = self.finger()
        for angle in [T, I, M, R, L]:
            if angle is None:
                finger_pos.append(0)
            else:
                finger_pos.append(angle[0])

        finger_pos = np.array(finger_pos)
        # print(finger_pos)
        return finger_pos

    def hand_finger_step(self, handle, step):
        angle = 0
        # sim.simxPauseCommunication(self.client, 1)
        for angle in range(0, 76):
            sim.setJointTargetPosition(handle, angle)
            angle = angle + step
        # sim.simxPauseCommunication(self.client, 0)

    def set_position(self, handle, angle):

        # sim.simxPauseCommunication(self.client, 1)
        sim.setJointTargetPosition(handle, angle)
        # sim.simxPauseCommunication(self.client, 0)


    def arm_move(self, dis):
        initial_pos = sim.getObjectPosition(self.target_pos, sim.handle_world)
        initial_pos = np.array(initial_pos)

        final_pos = initial_pos
        final_pos[2] += dis

        sim.setObjectPosition(self.target_pos, final_pos.tolist(), sim.handle_world)


    def reset(self):
        # self.z = random.randint(0, 10)
        # self.z = 10
        # print(self.z)
        pos = self.default_obj_pos[self.z]
        pos_1 = pos
        pos_1[2] = -1000
        time.sleep(1)
        sim.setObjectPosition(self.objects[self.res], pos_1, sim.handle_world)
        sim.setObjectPosition(self.bottom, [0.453, -5.090, 0.02], sim.handle_world)
        self.get_all_handles()
        # 机械手关节复位
        for i, joint_handle in enumerate(self.hand_joints):
            self.set_position(joint_handle, self.default_hand_pos[i] * np.pi / 180)

        sim.setObjectPosition(self.target_pos, [0.106, -5.015, self.default_arm_pos[self.z]],
                              sim.handle_world)  # robotic arm reset
        time.sleep(2)

        if self.z == 2 or self.z == 3 or self.z == 5 or self.z == 6 or self.z == 7\
                or self.z == 11 or self.z == 13 or self.z == 15 or self.z == 17 or self.z == 18 or self.z == 19 or self.z == 20:
            sim.setObjectPosition(self.bottom, [0.283, -5.090, 0.015], sim.handle_world)
        time.sleep(4)
        sim.setObjectPosition(self.objects[self.z], self.grasp_obj_pos[self.z], sim.handle_world)
        sim.setObjectOrientation(self.objects[self.z], self.grasp_obj_ori[self.z], sim.handle_world)

        time.sleep(2)

        self.joint_angle = [0, 0, 0, 0, 0]
        self.initial_w = [0, 0, 0, 0, 0]
        self.res = self.z

        return np.hstack((self.get_finger_angel(), self.initial_expected_angle, self.z))

    def object_reset(self):
        pos = self.default_obj_pos[self.z]
        sim.setObjectPosition(self.objects[self.z], pos, sim.handle_world)  # [0.362, -5.066, 0.076]
        sim.setObjectOrientation(self.objects[self.z], self.grasp_obj_ori[self.z], sim.handle_world)
        time.sleep(2)

    def apply_action(self, joint_handle_m, joint_handle_p, i, action, k, lower, upper):
        # res, current_position = sim.simxGetJointPosition(self.client, joint_handle_m, sim.simx_opmode_buffer)
        self.joint_angle[i] = self.joint_angle[i] + action * (3.14159 / 180)
        new_position_m = self.joint_angle[i]
        new_position_p = new_position_m * k
        # print(new_position_m)
        if not np.all((lower <= new_position_m) & (new_position_m <= upper)):
            new_position_m = new_position_m - action * (3.14159 / 180)
            self.joint_angle[i] = new_position_m
            new_position_p = new_position_m * k
        sim.setJointTargetPosition(joint_handle_m, float(new_position_m))
        sim.setJointTargetPosition(joint_handle_p, float(new_position_p))

        return new_position_m * (180 / 3.14)



    def step(self, action):
        # next_state, reward, terminate, _ = self.step()
        done = False
        done_episode = False
        thresh_bool_t = 0

        if not (self.initial_w[1] == 1):
            new_angle_i = self.apply_action(self.hand_joints[3], self.hand_joints[4], 1, action, 68 / 75, 0, 1.308)
        else:
            new_angle_i = self.apply_action(self.hand_joints[3], self.hand_joints[4], 1, 0, 68 / 75, 0, 1.308)
        if not (self.initial_w[2] == 1):
            new_angle_m = self.apply_action(self.hand_joints[5], self.hand_joints[6], 2, action, 68 / 75, 0, 1.308)
        else:
            new_angle_m = self.apply_action(self.hand_joints[5], self.hand_joints[6], 2, 0, 68 / 75, 0, 1.308)
        if (not self.initial_w[3] == 1) and (not self.z == 2) and (not self.z == 5) and (not self.z == 6) and (not self.z == 7) \
                and (not self.z == 11) and (not self.z == 15) and (not self.z == 18) and (not self.z == 19):
            new_angle_r = self.apply_action(self.hand_joints[7], self.hand_joints[8], 3, action, 68 / 75, 0, 1.308)
        else:
            new_angle_r = self.apply_action(self.hand_joints[7], self.hand_joints[8], 3, 0, 68 / 75, 0, 1.308)
        if not (self.initial_w[4] == 1) and (not self.z == 2) and (not self.z == 5) and (not self.z == 6) and (not self.z == 7) \
                and (not self.z == 11) and (not self.z == 15) and (not self.z == 18) and (not self.z == 19) and (not self.z == 13)\
                and (not self.z == 17):
            new_angle_l = self.apply_action(self.hand_joints[9], self.hand_joints[10], 4, action, 68 / 75, 0, 1.308)
        else:
            new_angle_l = self.apply_action(self.hand_joints[9], self.hand_joints[10], 4, 0, 68 / 75, 0, 1.308)
        if not (self.initial_w[0] == 1):
            new_angle_t = self.apply_action(self.hand_joints[1], self.hand_joints[2], 0, action, 32 / 30, 0, 0.523)
        else:
            new_angle_t = self.apply_action(self.hand_joints[1], self.hand_joints[2], 0, 0, 32 / 30, 0, 0.523)

        force_t, force_i, force_m, force_r, force_l, theta = self.get_tactile_feedback()
        if 0 <= force_t <= self.force_threshold[self.z][0]:
            w_t = self.w[1]
        elif self.force_threshold[self.z][1] < force_t < self.force_threshold[self.z][2]:
            # w_t = self.w[3]
            w_t = force_t /((self.force_threshold[self.z][2] + self.force_threshold[self.z][3])/2)
            cos = np.cos(theta[0])
            w_t += 0.2 * cos  # 当对齐时为0.2，反向为-0.2，梯度清晰
        elif self.force_threshold[self.z][2] <= force_t <= self.force_threshold[self.z][3]:
            w_t = force_t / ((self.force_threshold[self.z][2] + self.force_threshold[self.z][3]) / 2)
            cos = np.cos(theta[0])
            w_t += 0.2 * cos  # 当对齐时为0.2，反向为-0.2，梯度清晰
            thresh_bool_t = 1
        elif force_t > (self.force_threshold[self.z][3]):
            w_t = -force_t / ((self.force_threshold[self.z][2] + self.force_threshold[self.z][3]) / 2)
        else:
            # w_t = self.w[2]
            w_t = force_t / ((self.force_threshold[self.z][2] + self.force_threshold[self.z][3]) / 2)
        # print(force_t, reward_t)
        w_i, thresh_bool_i = self.set_reward(self.force_i, self.objects[self.z], force_i, theta[1])
        w_m, thresh_bool_m = self.set_reward(self.force_m, self.objects[self.z], force_m, theta[2])
        w_r, thresh_bool_r = self.set_reward(self.force_r, self.objects[self.z], force_r, theta[3])
        w_l, thresh_bool_l = self.set_reward(self.force_l, self.objects[self.z], force_l, theta[4])
        # print(force_l, reward_l)
        self.initial_w = [w_t, w_i, w_m, w_r, w_l]

        reward_t = w_t
        reward_i = w_i
        reward_m = w_m
        reward_r = w_r
        reward_l = w_l

        if (not self.z == 2) and (not self.z == 5) and (not self.z == 6) and (not self.z == 7):
            reward = reward_t + reward_i + reward_m + reward_r + reward_l
        else:
            reward = (reward_t + reward_i + reward_m) * 1.3
        target_angle = np.hstack((new_angle_t, new_angle_i, new_angle_m, new_angle_r, new_angle_l))
        current_angle = self.get_finger_angel()
        next_state = np.hstack((current_angle, target_angle, self.z))
        obj = sim.getObjectPosition(self.objects[self.z], sim.handle_world)
        obj = np.array(obj)
        high_obj = obj[2]
        obj1 = obj[0]
        obj2 = obj[1]
        if w_t < -0.2 or w_i < -0.2 or w_m < -0.2 or w_r < -0.2 or w_l < -0.2:
            # reward = reward * self.w[0]
            terminate = True
            done_episode = True
        # elif done_out:
        #     terminate = True
        elif high_obj < 0.05 or np.abs(obj1 - self.grasp_obj_pos[self.z][0]) > 0.1 or np.abs(obj2 - self.grasp_obj_pos[self.z][1]) > 0.1:
            terminate = True
            done_episode = True
        else:
            terminate = False
        count = sum([thresh_bool_t == 1, thresh_bool_i == 1, thresh_bool_m == 1, thresh_bool_r == 1, thresh_bool_l == 1])
        if (count >= 1) and (terminate is False):
            self.arm_move(0.2)
            time.sleep(6)
            obj = sim.getObjectPosition(self.objects[self.z], sim.handle_world)
            obj = np.array(obj)
            high_obj = obj[2]
            print(high_obj)
            if high_obj > 0.20:
                done = True
                terminate = True
                done_episode = True
                reward = reward + 2
            else:
                reward = reward + self.w[0]
                done = True
                done_episode = True
            # self.arm_move(-0.2)
            # time.sleep(4)

        return next_state, reward, done, terminate, done_episode

    def set_reward(self, object_A, object_B, force, theta):
        thresh_bool = 0

        if 0 <= force <= self.force_threshold[self.z][0]:
            omega = self.w[1]
        elif self.force_threshold[self.z][1] < force < self.force_threshold[self.z][2]:
            # omega = self.w[3]
            omega = force / ((self.force_threshold[self.z][2] + self.force_threshold[self.z][3])/2)
            cos = np.cos(theta)
            omega += 0.2 * cos  # 当对齐时为0.2，反向为-0.2，梯度清晰
        elif self.force_threshold[self.z][2] <= force <= self.force_threshold[self.z][3]:
            omega = force / ((self.force_threshold[self.z][2] + self.force_threshold[self.z][3]) / 2)
            cos = np.cos(theta)
            omega += 0.2 * cos  # 当对齐时为0.2，反向为-0.2，梯度清晰
            thresh_bool = 1
        elif force > self.force_threshold[self.z][3]:
            omega = -force / ((self.force_threshold[self.z][2] + self.force_threshold[self.z][3])/2)
        else:
            # omega = self.w[2]
            omega = force / ((self.force_threshold[self.z][2] + self.force_threshold[self.z][3]) / 2)

        return omega, thresh_bool



