import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import json
import numpy as np
import os
import cv2

# 定义绘制轨迹的函数
def draw_trajectory_on_ax(ax, trajectory, label, color='r'):
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'o-', label=label, color=color)

# 定义处理单帧的函数
def process_frame(frame_index):
    # 读取 JSON 数据
    json_path = f'/home/ly/WDZ/TCP/data/Baseline_SE+CBAM+5Rule_410K_5long524/routes_town05_long_05_24_17_29_34/meta/{frame_index:04d}.json'
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    # 创建图形和轴
    fig, ax = plt.subplots()
    img_path = f'/home/ly/WDZ/TCP/data/Baseline_SE+CBAM+5Rule_410K_5long524/routes_town05_long_05_24_17_29_34/bev/{frame_index:04d}.png'
    img = plt.imread(img_path)
    ax.imshow(img, extent=[-70, 70, 0, 120])

    # 将 JSON 中的 waypoint 数据转换为 numpy 数组，并增加纵坐标
    waypoints = np.array([
        [json_data['wp_1'][0], json_data['wp_1'][1] + 60],
        [json_data['wp_2'][0], json_data['wp_2'][1] + 61],
        [json_data['wp_3'][0], json_data['wp_3'][1] + 62],
        [json_data['wp_4'][0], json_data['wp_4'][1] + 63]
    ])

    # 绘制 waypoint
    draw_trajectory_on_ax(ax, waypoints, 'Waypoints')

    # 添加文本注释
    ax.text(40, 115, f"Speed: {json_data['speed']:.2f} m/s", fontsize=11, color='blue')
    ax.text(40, 110, f"Steer: {json_data['steer']:.2f}", fontsize=11, color='blue')
    ax.text(40, 105, f"Throttle: {json_data['throttle']:.2f}", fontsize=11, color='blue')
    ax.text(40, 100, f"Brake: {json_data['brake']:.2f}", fontsize=11, color='blue')

    # 设置图例位置在文本注释下方
    ax.legend(loc='lower left', bbox_to_anchor=(0, 0), framealpha=0.5)
    ax.axis('off')
    # 调整布局以防止图例遮挡文本
    plt.tight_layout()

    # 保存处理后的图片
    output_path = f'last-25/{frame_index:04d}.png'
    plt.savefig(output_path)
    plt.close(fig)

# 处理所有帧
frame_indices = range(1, 705)  # 从 0001 到 1630
os.makedirs('last-25', exist_ok=True)
for frame_index in frame_indices:
    process_frame(frame_index)

# 将处理后的图片制作成视频
def create_video(output_path, frame_folder, fps=30):
    # 获取所有帧图片
    images = [img for img in os.listdir(frame_folder) if img.endswith(".png")]
    images.sort()

    # 读取第一帧图片以获取视频的宽高
    frame = cv2.imread(os.path.join(frame_folder, images[0]))
    height, width, layers = frame.shape

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 将图片写入视频
    for image in images:
        video.write(cv2.imread(os.path.join(frame_folder, image)))
    # 释放视频写入器
    video.release()

# 创建视频
create_video('last-25.mp4', 'last-25')