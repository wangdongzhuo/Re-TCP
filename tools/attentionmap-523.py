import torch
import matplotlib.pyplot as plt
import numpy as np
from TCP.model import ResNet34WithHybridAttention
from PIL import Image
import torchvision.transforms as transforms


# 定义函数将注意力图绘制到车辆前视图上（这里假设车辆前视图是一个固定大小的图像）
def draw_attention_map(attention_map, figsize=(8, 6), title="Attention Map"):
    plt.figure(figsize=figsize)
    # 对注意力图进行归一化处理
    attention_map = (attention_map - np.min(attention_map)) / (np.max(attention_map) - np.min(attention_map))
    plt.imshow(attention_map, cmap='hot')
    plt.title(title)
    plt.axis('off')
    plt.show()

# 加载模型
model = ResNet34WithHybridAttention()
# 这里需要根据实际情况加载权重文件，假设权重文件名为 'your_model.ckpt'
try:
    model.load_state_dict(torch.load('/home/ly/WDZ/TCP/log/TCP_Baseline+SE+CBAM_410K/best_epoch=56-val_loss=0.463.ckpt'))
except FileNotFoundError:
    print("权重文件不存在，请检查路径！")
except RuntimeError as e:
    print(f"加载权重时出现错误: {e}")
model.eval()

# 加载图像并进行预处理
image_path = "/home/ly/WDZ/TCP/data/Baseline_SE+CBAM+5Rule_198K_5long513/routes_town05_long_05_13_23_07_21/rgb/0474.png"
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 根据实际训练设置调整
])
image = Image.open(image_path).convert('RGB')
image = transform(image).unsqueeze(0)

# 获取所有注意力层的注意力图
attention_maps = []
for layer in [model.hybrid_attention_layer1.channel_attention, model.hybrid_attention_layer2.channel_attention,
              model.hybrid_attention_layer3.channel_attention, model.hybrid_attention_layer4.channel_attention]:
    with torch.no_grad():
        attn = layer(image)
        attention_maps.append(attn.squeeze(0).squeeze(0).cpu().numpy())

# 分别可视化这些注意力图
for i, attn_map in enumerate(attention_maps):
    draw_attention_map(attn_map, figsize=(8, 6), title=f"Attention Map Layer {i + 1}")