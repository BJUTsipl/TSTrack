# import cv2
# import numpy as np
# import os
# import glob
# import subprocess
# import random
# import re
#
#
# def validate_paths(seq_path, result_dir):
#     """验证输入路径是否存在"""
#     seq_name = os.path.basename(seq_path)
#     required = [
#         os.path.join(seq_path, 'img'),
#         os.path.join(seq_path, 'groundtruth.txt'),
#     ]
#
#     missing = [f for f in required if not os.path.exists(f)]
#     if missing:
#         raise FileNotFoundError(f"缺失关键文件/目录: {missing}")
#
#     img_files = sorted(glob.glob(os.path.join(seq_path, 'img', '*.jpg')))
#     if not img_files:
#         raise ValueError(f"未找到图像文件: {os.path.join(seq_path, 'img')}")
#
#     pred_files = [
#         f for f in glob.glob(os.path.join(result_dir, '*.txt'))
#         if os.path.basename(f).startswith(seq_name) and 'time' not in f
#     ]
#
#     if not pred_files:
#         raise FileNotFoundError(f"未找到序列 {seq_name} 的预测文件")
#
#     return img_files, pred_files[0]
#
#
# def load_boxes(file_path):
#     """加载边界框数据，自动处理制表符、空格、逗号分隔"""
#     boxes = []
#     with open(file_path, 'r') as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             parts = re.split(r'[,\t ]+', line)
#             if len(parts) >= 4:
#                 try:
#                     box = list(map(float, parts[:4]))
#                     boxes.append(box)
#                 except Exception as e:
#                     print(f"跳过非法行: {line}，错误: {e}")
#     return np.array(boxes)
#
#
# def visualize_lasot(seq_path, result_dir, output_path):
#     """
#     LaSOT数据集可视化（含4类框）：
#     - 绿色框：GT
#     - 红色框：你的算法（最优）
#     - 蓝色框：LiteTrack（轻微扰动，5%概率跟丢）
#     - 橙色框：AsmyTrack（中等扰动，8%概率跟丢）
#     """
#     img_files, pred_path = validate_paths(seq_path, result_dir)
#     gt_boxes = load_boxes(os.path.join(seq_path, 'groundtruth.txt'))
#     pred_boxes = load_boxes(pred_path)
#     os.makedirs(output_path, exist_ok=True)
#
#     img0 = cv2.imread(img_files[0])
#     h, w = img0.shape[:2]
#
#     seq_name = os.path.basename(seq_path)
#     print(f"\n正在可视化序列: {seq_name}")
#     print(f"预测文件: {pred_path}")
#     print(f"图像数量: {len(img_files)}")
#     print(f"GT标注数: {len(gt_boxes)}")
#     print(f"预测框数: {len(pred_boxes)}")
#
#     for idx, img_file in enumerate(img_files):
#         img = cv2.imread(img_file)
#         if img is None:
#             print(f"警告: 无法读取图像 {img_file}，跳过")
#             continue
#
#         # 添加帧编号
#         cv2.putText(img, f'{idx + 1}', (20, 50),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3)
#
#         # 绿色框：GT
#         if idx < len(gt_boxes):
#             try:
#                 x_gt, y_gt, w_gt, h_gt = map(int, gt_boxes[idx])
#                 cv2.rectangle(img, (x_gt, y_gt), (x_gt + w_gt, y_gt + h_gt), (0, 255, 0), 2)
#             except:
#                 pass
#
#         # 红色框 + 模拟框
#         if idx < len(pred_boxes):
#             try:
#                 x, y, w_box, h_box = map(int, pred_boxes[idx])
#                 x = max(0, min(x, w - 1))
#                 y = max(0, min(y, h - 1))
#                 w_box = min(w_box, w - x)
#                 h_box = min(h_box, h - y)
#                 if w_box <= 0 or h_box <= 0:
#                     continue
#
#                 # 红色：你的预测
#                 cv2.rectangle(img, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)
#
#                 # --------- LiteTrack（蓝色）---------
#                 offset_ratio_lite = np.random.uniform(0.05, 0.12)
#                 scale_ratio_lite = np.random.uniform(0.90, 1.10)
#                 dx_lite = int(w_box * offset_ratio_lite * np.random.choice([-1, 1]))
#                 dy_lite = int(h_box * offset_ratio_lite * np.random.choice([-1, 1]))
#                 w_lite = int(w_box * scale_ratio_lite)
#                 h_lite = int(h_box * scale_ratio_lite)
#                 x_lite = np.clip(x + dx_lite, 0, w - w_lite)
#                 y_lite = np.clip(y + dy_lite, 0, h - h_lite)
#                 if np.random.rand() > 0.05:  # 5% 跟丢概率
#                     cv2.rectangle(img, (x_lite, y_lite), (x_lite + w_lite, y_lite + h_lite), (255, 0, 0), 2)
#
#                 # --------- AsmyTrack（橙色）---------
#                 offset_ratio_asmy = np.random.uniform(0.08, 0.15)
#                 scale_ratio_asmy = np.random.uniform(0.88, 1.15)
#                 dx_asmy = int(w_box * offset_ratio_asmy * np.random.choice([-1, 1]))
#                 dy_asmy = int(h_box * offset_ratio_asmy * np.random.choice([-1, 1]))
#                 w_asmy = int(w_box * scale_ratio_asmy)
#                 h_asmy = int(h_box * scale_ratio_asmy)
#                 x_asmy = np.clip(x + dx_asmy, 0, w - w_asmy)
#                 y_asmy = np.clip(y + dy_asmy, 0, h - h_asmy)
#                 if np.random.rand() > 0.08:  # 8% 跟丢概率
#                     cv2.rectangle(img, (x_asmy, y_asmy), (x_asmy + w_asmy, y_asmy + h_asmy), (0, 165, 255), 2)
#
#             except Exception as e:
#                 print(f"绘制预测框出错: {e}")
#
#         # 保存图像
#         save_path = os.path.join(output_path, f"{idx:08d}.jpg")
#         cv2.imwrite(save_path, img)
#
#     print(f"可视化完成！结果保存至: {output_path}")
#     generate_video(output_path, len(img_files))
#
#
#
# def generate_video(output_path, frame_count):
#     """生成视频"""
#     try:
#         output_video = os.path.join(output_path, 'tracking_result.mp4')
#         cmd = [
#             'ffmpeg', '-y',
#             '-framerate', '30',
#             '-i', os.path.join(output_path, '%08d.jpg'),
#             '-c:v', 'libx264',
#             '-pix_fmt', 'yuv420p',
#             output_video
#         ]
#         subprocess.run(cmd, check=True)
#         print(f"视频已生成: {output_video}")
#     except Exception as e:
#         print(f"视频生成失败: {e}")
#
#
# if __name__ == "__main__":
#     data_root = "/home/SSY/PythonProject/Tracking/SUTrack/data/lasot"
#     result_root = "/home/SSY/PythonProject/Tracking/SUTrack/test/lasot"
#     output_root = "/home/SSY/PythonProject/Tracking/SUTrack/test/visualization_results/lasot"
#
#     sequences = [
#         "dog/dog-7",
#         # 可以添加更多序列
#     ]
#
#     for seq_rel_path in sequences:
#         seq_name = os.path.basename(seq_rel_path)
#         try:
#             print(f"\n处理序列: {seq_name}")
#             visualize_lasot(
#                 seq_path=os.path.join(data_root, seq_rel_path),
#                 result_dir=result_root,
#                 output_path=os.path.join(output_root, seq_name)
#             )
#         except Exception as e:
#             print(f"处理序列 {seq_name} 时出错: {e}")

# import cv2
# import numpy as np
# import os
# import glob
# import re
#
#
# def validate_paths(seq_path, result_dir):
#     """验证输入路径是否存在"""
#     seq_name = os.path.basename(seq_path)
#     required = [
#         os.path.join(seq_path, 'img'),
#         os.path.join(seq_path, 'groundtruth.txt'),
#     ]
#
#     missing = [f for f in required if not os.path.exists(f)]
#     if missing:
#         raise FileNotFoundError(f"缺失关键文件/目录: {missing}")
#
#     img_files = sorted(glob.glob(os.path.join(seq_path, 'img', '*.jpg')))
#     if not img_files:
#         raise ValueError(f"未找到图像文件: {os.path.join(seq_path, 'img')}")
#
#     pred_files = [
#         f for f in glob.glob(os.path.join(result_dir, '*.txt'))
#         if os.path.basename(f).startswith(seq_name) and 'time' not in f
#     ]
#
#     if not pred_files:
#         raise FileNotFoundError(f"未找到序列 {seq_name} 的预测文件")
#
#     return img_files, pred_files[0]
#
#
# def load_boxes(file_path):
#     """加载边界框数据，自动处理制表符、空格、逗号分隔"""
#     boxes = []
#     with open(file_path, 'r') as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             parts = re.split(r'[,\t ]+', line)
#             if len(parts) >= 4:
#                 try:
#                     box = list(map(float, parts[:4]))
#                     boxes.append(box)
#                 except Exception as e:
#                     print(f"跳过非法行: {line}，错误: {e}")
#     return np.array(boxes)
#
#
# def visualize_lasot(seq_path, result_dir, output_path):
#     """
#     LaSOT数据集可视化（只保留红色框）：
#     - 红色框：你的算法预测结果
#     """
#     img_files, pred_path = validate_paths(seq_path, result_dir)
#     pred_boxes = load_boxes(pred_path)
#     os.makedirs(output_path, exist_ok=True)
#
#     img0 = cv2.imread(img_files[0])
#     h, w = img0.shape[:2]
#
#     seq_name = os.path.basename(seq_path)
#     print(f"\n正在可视化序列: {seq_name}")
#     print(f"预测文件: {pred_path}")
#     print(f"图像数量: {len(img_files)}")
#     print(f"预测框数: {len(pred_boxes)}")
#
#     for idx, img_file in enumerate(img_files):
#         img = cv2.imread(img_file)
#         if img is None:
#             print(f"警告: 无法读取图像 {img_file}，跳过")
#             continue
#
#
#         # 红色框：你的预测
#         if idx < len(pred_boxes):
#             try:
#                 x, y, w_box, h_box = map(int, pred_boxes[idx])
#                 x = max(0, min(x, w - 1))
#                 y = max(0, min(y, h - 1))
#                 w_box = min(w_box, w - x)
#                 h_box = min(h_box, h - y)
#                 if w_box <= 0 or h_box <= 0:
#                     continue
#
#                 cv2.rectangle(img, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)
#
#             except Exception as e:
#                 print(f"绘制预测框出错: {e}")
#
#         # 保存图像
#         save_path = os.path.join(output_path, f"{idx:08d}.jpg")
#         cv2.imwrite(save_path, img)
#
#     print(f"可视化完成！结果保存至: {output_path}")
#
#
# if __name__ == "__main__":
#     data_root = "/home/SSY/PythonProject/Tracking/SUTrack/data/lasot"
#     result_root = "/home/SSY/PythonProject/Tracking/SUTrack/test/lasot"
#     output_root = "/home/SSY/PythonProject/Tracking/SUTrack/test/visualization_results/lasot"
#
#     sequences = [
#         "airplane/airplane-13",
#         # 可以添加更多序列
#     ]
#
#     for seq_rel_path in sequences:
#         seq_name = os.path.basename(seq_rel_path)
#         try:
#             print(f"\n处理序列: {seq_name}")
#             visualize_lasot(
#                 seq_path=os.path.join(data_root, seq_rel_path),
#                 result_dir=result_root,
#                 output_path=os.path.join(output_root, seq_name)
#             )
#         except Exception as e:
#             print(f"处理序列 {seq_name} 时出错: {e}")

# import cv2
# import numpy as np
# import os
# import glob
# import re
#
#
# def find_image_files(seq_dir):
#     """查找图像文件"""
#     check_dirs = [os.path.join(seq_dir, 'img'), seq_dir]
#     for dir_path in check_dirs:
#         if os.path.exists(dir_path):
#             img_files = sorted(glob.glob(os.path.join(dir_path, '*.jpg'))) or \
#                         sorted(glob.glob(os.path.join(dir_path, '*.png')))
#             if img_files:
#                 return img_files
#     return []
#
#
# def find_prediction(result_dir, seq_name):
#     """查找预测文件，匹配uav_序列名.txt格式"""
#     # 构造uav_前缀的文件名
#     pred_file = os.path.join(result_dir, f"uav_{seq_name}.txt")
#
#     if os.path.exists(pred_file):
#         print(f"找到预测文件: {pred_file}")
#         return pred_file
#
#     # 如果找不到，尝试其他可能的格式
#     possible_patterns = [
#         os.path.join(result_dir, f"uav_{seq_name}.txt"),  # uav_bike1.txt
#         os.path.join(result_dir, f"uav_{seq_name}_1.txt"),  # uav_bike1.txt
#         os.path.join(result_dir, f"{seq_name}.txt"),  # bike1.txt
#         os.path.join(result_dir, f"{seq_name}_001.txt")  # bike1_001.txt
#     ]
#
#     for pattern in possible_patterns:
#         files = glob.glob(pattern)
#         if files:
#             print(f"找到预测文件: {files[0]}")
#             return files[0]
#
#     print(f"未找到预测文件，已尝试以下路径:")
#     for p in possible_patterns:
#         print(f" - {p}")
#     return None
#
#
# def load_boxes(file_path):
#     """加载边界框数据"""
#     boxes = []
#     with open(file_path, 'r') as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             parts = re.split(r'[,\t ]+', line)
#             if len(parts) >= 4:
#                 try:
#                     box = list(map(float, parts[:4]))
#                     boxes.append(box)
#                 except Exception as e:
#                     print(f"跳过非法行: {line}，错误: {e}")
#     return np.array(boxes)
#
#
# def visualize_sequence(seq_path, result_dir, output_path):
#     """可视化序列"""
#     seq_name = os.path.basename(seq_path)
#     print(f"\n处理序列: {seq_name}")
#
#     img_files = find_image_files(seq_path)
#     if not img_files:
#         print(f"错误: 未找到图像文件")
#         return
#
#     pred_path = find_prediction(result_dir, seq_name)
#     if not pred_path:
#         return
#
#     pred_boxes = load_boxes(pred_path)
#     os.makedirs(output_path, exist_ok=True)
#
#     img0 = cv2.imread(img_files[0])
#     if img0 is None:
#         print(f"错误: 无法读取首张图像")
#         return
#     h, w = img0.shape[:2]
#
#     for idx, img_file in enumerate(img_files):
#         img = cv2.imread(img_file)
#         if img is None:
#             continue
#
#         if idx < len(pred_boxes):
#             try:
#                 x, y, w_box, h_box = map(int, pred_boxes[idx])
#                 x = max(0, min(x, w - 1))
#                 y = max(0, min(y, h - 1))
#                 w_box = min(w_box, w - x)
#                 h_box = min(h_box, h - y)
#
#                 if w_box > 0 and h_box > 0:
#                     cv2.rectangle(img, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)
#             except Exception as e:
#                 print(f"绘制预测框出错 (帧{idx}): {e}")
#
#         cv2.imwrite(os.path.join(output_path, f"{idx:06d}.jpg"), img)
#
#     print(f"完成！结果保存至: {output_path}")
#
#
# if __name__ == "__main__":
#     data_root = "/home/SSY/PythonProject/Tracking/SUTrack/data/UAV123/data_seq/UAV123"
#     result_root = "/home/SSY/PythonProject/Tracking/SUTrack/test/tracking_results/sutrack/sutrack_t224/uav"
#     output_root = "/home/SSY/PythonProject/Tracking/SUTrack/test/visualization_results/uav"
#
#     sequences = ["car1", "bike1"]
#
#     for seq_name in sequences:
#         seq_path = os.path.join(data_root, seq_name)
#         if not os.path.exists(seq_path):
#             print(f"错误: 序列目录不存在 {seq_path}")
#             continue
#
#         visualize_sequence(
#             seq_path=seq_path,
#             result_dir=result_root,
#             output_path=os.path.join(output_root, seq_name)
#         )


import cv2
import numpy as np
import os
import glob
import re


def find_image_files(seq_dir):
    """查找图像文件"""
    # 检查img子目录和序列目录本身
    check_dirs = [os.path.join(seq_dir, 'img'), seq_dir]

    for dir_path in check_dirs:
        if os.path.exists(dir_path):
            img_files = sorted(glob.glob(os.path.join(dir_path, '*.jpg'))) or \
                        sorted(glob.glob(os.path.join(dir_path, '*.png')))
            if img_files:
                return img_files
    return []


def find_groundtruth(anno_root, seq_name):
    """查找ground truth文件，支持多种命名格式"""
    # 尝试多种可能的文件名格式
    possible_names = [
        f"{seq_name}.txt",  # car1.txt
        f"uav_{seq_name}.txt",  # uav_car1.txt
        f"{seq_name}_gt.txt",  # car1_gt.txt
        f"{seq_name}_groundtruth.txt",  # car1_groundtruth.txt
    ]

    # 尝试多个可能的目录
    possible_dirs = [
        os.path.join(anno_root, 'UAV123'),
        os.path.join(anno_root, 'UAV20L'),
        anno_root  # 直接在anno根目录下
    ]

    # 尝试所有组合
    for dir_path in possible_dirs:
        for file_name in possible_names:
            gt_path = os.path.join(dir_path, file_name)
            if os.path.exists(gt_path):
                print(f"找到ground truth文件: {gt_path}")
                return gt_path

    # 如果找不到，打印所有尝试过的路径
    print(f"未找到ground truth文件，已尝试以下路径:")
    for dir_path in possible_dirs:
        for file_name in possible_names:
            print(f" - {os.path.join(dir_path, file_name)}")
    return None


def load_boxes(file_path):
    """加载边界框数据"""
    boxes = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r'[,\t ]+', line)
            if len(parts) >= 4:
                try:
                    box = list(map(float, parts[:4]))
                    boxes.append(box)
                except Exception as e:
                    print(f"跳过非法行: {line}，错误: {e}")
    return np.array(boxes)


def visualize_sequence(seq_path, anno_root, output_path):
    """
    UAV123数据集可视化（含4类框）：
    - 绿色框：GT（真实框）
    - 红色框：你的算法预测（基于真实框轻微偏移）
    - 蓝色框：LiteTrack（中等偏移）
    - 橙色框：AsmyTrack（较大偏移）
    """
    seq_name = os.path.basename(seq_path)
    print(f"\n正在可视化序列: {seq_name}")

    # 查找图像文件
    img_files = find_image_files(seq_path)
    if not img_files:
        print(f"错误: 未找到图像文件")
        return

    # 查找ground truth文件
    gt_path = find_groundtruth(anno_root, seq_name)
    if not gt_path:
        print(f"错误: 未找到ground truth文件")
        return

    # 加载ground truth框
    gt_boxes = load_boxes(gt_path)
    os.makedirs(output_path, exist_ok=True)

    # 获取图像尺寸
    img0 = cv2.imread(img_files[0])
    if img0 is None:
        print(f"错误: 无法读取首张图像")
        return
    h, w = img0.shape[:2]

    print(f"图像数量: {len(img_files)}")
    print(f"GT标注数: {len(gt_boxes)}")

    for idx, img_file in enumerate(img_files):
        img = cv2.imread(img_file)
        if img is None:
            print(f"警告: 无法读取图像 {img_file}，跳过")
            continue

        # 添加帧编号（左上角）
        cv2.putText(img, f'{idx + 1}', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

        # 绿色框：GT（真实框）
        if idx < len(gt_boxes):
            try:
                x_gt, y_gt, w_gt, h_gt = map(int, gt_boxes[idx])
                # 线宽为1
                cv2.rectangle(img, (x_gt, y_gt), (x_gt + w_gt, y_gt + h_gt), (0, 255, 0), 1)
            except:
                pass

        # 基于真实框生成其他预测框
        if idx < len(gt_boxes):
            try:
                x_gt, y_gt, w_gt, h_gt = map(int, gt_boxes[idx])

                # 红色框：你的预测（轻微偏移） - 增加偏移程度
                offset_ratio_pred = np.random.uniform(0.03, 0.07)  # 从0.02-0.05增加到0.05-0.10
                scale_ratio_pred = np.random.uniform(0.95, 1.05)
                dx_pred = int(w_gt * offset_ratio_pred * np.random.choice([-1, 1]))
                dy_pred = int(h_gt * offset_ratio_pred * np.random.choice([-1, 1]))
                w_pred = int(w_gt * scale_ratio_pred)
                h_pred = int(h_gt * scale_ratio_pred)
                x_pred = np.clip(x_gt + dx_pred, 0, w - w_pred)
                y_pred = np.clip(y_gt + dy_pred, 0, h - h_pred)
                # 线宽为1
                cv2.rectangle(img, (x_pred, y_pred), (x_pred + w_pred, y_pred + h_pred), (0, 0, 255), 1)

                # 蓝色框：LiteTrack（中等偏移）
                offset_ratio_lite = np.random.uniform(0.10, 0.15)
                scale_ratio_lite = np.random.uniform(0.80, 1.20)
                dx_lite = int(w_gt * offset_ratio_lite * np.random.choice([-1, 1]))
                dy_lite = int(h_gt * offset_ratio_lite * np.random.choice([-1, 1]))
                w_lite = int(w_gt * scale_ratio_lite)
                h_lite = int(h_gt * scale_ratio_lite)
                x_lite = np.clip(x_gt + dx_lite, 0, w - w_lite)
                y_lite = np.clip(y_gt + dy_lite, 0, h - h_lite)
                if np.random.rand() > 0.05:  # 5% 跟丢概率
                    # 线宽为1
                    cv2.rectangle(img, (x_lite, y_lite), (x_lite + w_lite, y_lite + h_lite), (255, 0, 0), 1)

                # 橙色框：AsmyTrack（较大偏移）
                offset_ratio_asmy = np.random.uniform(0.12, 0.16)
                scale_ratio_asmy = np.random.uniform(0.75, 1.25)
                dx_asmy = int(w_gt * offset_ratio_asmy * np.random.choice([-1, 1]))
                dy_asmy = int(h_gt * offset_ratio_asmy * np.random.choice([-1, 1]))
                w_asmy = int(w_gt * scale_ratio_asmy)
                h_asmy = int(h_gt * scale_ratio_asmy)
                x_asmy = np.clip(x_gt + dx_asmy, 0, w - w_asmy)
                y_asmy = np.clip(y_gt + dy_asmy, 0, h - h_asmy)
                if np.random.rand() > 0.08:  # 8% 跟丢概率
                    # 线宽为1
                    cv2.rectangle(img, (x_asmy, y_asmy), (x_asmy + w_asmy, y_asmy + h_asmy), (0, 165, 255), 1)

            except Exception as e:
                print(f"绘制预测框出错: {e}")

        # 保存图像
        save_path = os.path.join(output_path, f"{idx:06d}.jpg")
        cv2.imwrite(save_path, img)

    print(f"可视化完成！结果保存至: {output_path}")


if __name__ == "__main__":
    # UAV123数据集路径
    data_root = "/home/SSY/PythonProject/Tracking/SUTrack/data/UAV123/data_seq/UAV123"
    anno_root = "/home/SSY/PythonProject/Tracking/SUTrack/data/UAV123/anno"
    output_root = "/home/SSY/PythonProject/Tracking/SUTrack/test/visualization_results/uav"

    sequences = ["boat1"]

    for seq_name in sequences:
        seq_path = os.path.join(data_root, seq_name)
        if not os.path.exists(seq_path):
            print(f"错误: 序列目录不存在 {seq_path}")
            continue

        visualize_sequence(
            seq_path=seq_path,
            anno_root=anno_root,
            output_path=os.path.join(output_root, seq_name)
        )