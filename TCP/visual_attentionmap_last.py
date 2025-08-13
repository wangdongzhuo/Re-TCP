import matplotlib
matplotlib.use('Agg')  # 使用 Agg 后端
import torch
import matplotlib.pyplot as plt
import numpy as np
from model import ResNet34WithHybridAttention
from PIL import Image
import torchvision.transforms as transforms
import os

# # 定义叠加注意力图到原图上的函数
# def overlay_attention_on_image(image, attention_map, alpha=0.5, title="Attention Map", save_path=None):
#     if attention_map.ndim != 2:  # 确保是二维数组
#         raise ValueError(f"Attention map must be 2D, but got shape {attention_map.shape}")
#
#     if np.max(attention_map) == np.min(attention_map):
#         attention_map = np.zeros_like(attention_map)  # 如果全为相同值，生成全零矩阵
#     else:
#         attention_map = (attention_map - np.min(attention_map)) / (np.max(attention_map) - np.min(attention_map))
#
#     attention_map_resized = np.array(Image.fromarray((attention_map * 255).astype(np.uint8)).resize(image.size, Image.BICUBIC))
#
#     # 将注意力图转换为热力图并叠加到图像上
#     attention_map_colored = plt.cm.jet(attention_map_resized)[:, :, :3]
#     attention_map_colored = (attention_map_colored * 255).astype(np.uint8)
#
#     # 转换图像为数组并叠加
#     image_array = np.array(image)
#     overlay = (image_array * (1 - alpha) + attention_map_colored * alpha).astype(np.uint8)
#
#     plt.figure(figsize=(6, 4))
#     plt.imshow(overlay)
#     plt.title(title)
#     plt.axis('off')
#     if save_path:
#         plt.savefig(save_path)
#     plt.show()
def overlay_attention_on_image(image, attention_map, alpha=0.5, title="Attention Map", save_path=None):
    if attention_map.ndim != 2:
        raise ValueError(f"Attention map must be 2D, but got shape {attention_map.shape}")
    if np.max(attention_map) == np.min(attention_map):
        attention_map = np.zeros_like(attention_map)
    else:
        attention_map = (attention_map - np.min(attention_map)) / (np.max(attention_map) - np.min(attention_map))

    attention_map_resized = np.array(
        Image.fromarray((attention_map * 255).astype(np.uint8)).resize(image.size, Image.BICUBIC))
    attention_map_colored = plt.cm.jet(attention_map_resized)[:, :, :3]
    attention_map_colored = (attention_map_colored * 255).astype(np.uint8)
    image_array = np.array(image)
    overlay = (image_array * (1 - alpha) + attention_map_colored * alpha).astype(np.uint8)

    plt.figure(figsize=(6, 4))
    plt.imshow(overlay)
    plt.title(title)
    plt.axis('off')
    plt.gca().set_frame_on(False)
    plt.gca().set_navigate(False)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# 加载模型
model = ResNet34WithHybridAttention()
# state_dict = torch.load('/home/liyang/WDZ/TCP/log/TCP_17_all_data/best_epoch=41-val_loss=0.479.ckpt')['state_dict']
# state_dict = torch.load('/home/ly/WDZ/TCP/log/TCP_Baseline+SE+CBAM_198K/best_epoch=32-val_loss=0.757.ckpt')['state_dict']
state_dict = torch.load('/home/ly/WDZ/TCP/log/TCP_Baseline+SE+CBAM_410K/best_epoch=56-val_loss=0.463.ckpt')['state_dict']
filtered_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
model.load_state_dict(filtered_state_dict, strict=False)
model.eval()

image_path = "/home/ly/WDZ/TCP/data/Baseline_SE+CBAM+5Rule_198K_5long513/routes_town05_long_05_14_01_09_26/rgb/0408.png"
transform = transforms.Compose([
    transforms.Resize((256, 900)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
image = Image.open(image_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0)

save_dir = "attention_visualizations_0408"
os.makedirs(save_dir, exist_ok=True)

with torch.no_grad():
    x = input_tensor
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    feature_layers = [model.layer1, model.layer2, model.layer3, model.layer4]
    attention_layers = [
        (model.hybrid_attention_layer1.channel_attention, model.hybrid_attention_layer1.spatial_attention, model.hybrid_attention_layer1),
        (model.hybrid_attention_layer2.channel_attention, model.hybrid_attention_layer2.spatial_attention, model.hybrid_attention_layer2),
        (model.hybrid_attention_layer3.channel_attention, model.hybrid_attention_layer3.spatial_attention, model.hybrid_attention_layer3),
        (model.hybrid_attention_layer4.channel_attention, model.hybrid_attention_layer4.spatial_attention, model.hybrid_attention_layer4)
    ]

    for i, (feature_layer, (channel_attention, spatial_attention, hybrid_attention)) in enumerate(zip(feature_layers, attention_layers)):
        x = feature_layer(x)  # 应用卷积层

        # 修复通道注意力可视化逻辑
        channel_attention_weights = channel_attention(x)  # [batch_size, num_channels, 1, 1]
        print(f"Layer {i + 1} Channel Attention Shape: {channel_attention_weights.shape}")
        # 将通道注意力权重广播到特征图大小
        channel_attention_map = (x * channel_attention_weights).mean(dim=1).squeeze().cpu().numpy()
        channel_attention_path = os.path.join(save_dir, f"layer_{i + 1}_channel_attention.png")
        overlay_attention_on_image(image, channel_attention_map, alpha=0.5, title=f"Layer {i + 1} Channel Attention", save_path=channel_attention_path)

        # 原有空间注意力可视化
        spatial_attention_weights = spatial_attention(x)
        print(f"Layer {i + 1} Spatial Attention Shape: {spatial_attention_weights.shape}")
        spatial_attention_map = spatial_attention_weights.squeeze().cpu().numpy()  # 确保是二维
        spatial_attention_path = os.path.join(save_dir, f"layer_{i + 1}_spatial_attention.png")
        overlay_attention_on_image(image, spatial_attention_map, alpha=0.5, title=f"Layer {i + 1} Spatial Attention", save_path=spatial_attention_path)

        # 原有混合注意力可视化
        hybrid_attention_weights = hybrid_attention(x)
        print(f"Layer {i + 1} Hybrid Attention Shape: {hybrid_attention_weights.shape}")
        hybrid_attention_map = hybrid_attention_weights.mean(dim=1).squeeze().cpu().numpy()  # 确保是二维
        hybrid_attention_path = os.path.join(save_dir, f"layer_{i + 1}_hybrid_attention.png")
        overlay_attention_on_image(image, hybrid_attention_map, alpha=0.5, title=f"Layer {i + 1} Hybrid Attention", save_path=hybrid_attention_path)

