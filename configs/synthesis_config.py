import os

# ============ 路径配置 ============
BASE_DIR = '/home/unitree/my_opencv'
DATA_DIR = os.path.join(BASE_DIR, 'data')

# 背景图片来源（关键修改）
USE_CUSTOM_BACKGROUNDS = True  # 改为 True
CUSTOM_BG_DIR = os.path.join(DATA_DIR, 'backgrounds')  # AI 生成的背景

# 如果没有自定义背景，使用自动生成
AUTO_GENERATE_BG = not USE_CUSTOM_BACKGROUNDS

# 目标物体路径
OBJECTS_DIR = os.path.join(DATA_DIR, 'objects')

# 输出路径
OUTPUT_TRAIN_DIR = os.path.join(DATA_DIR, 'synthetic_train')
OUTPUT_VAL_DIR = os.path.join(DATA_DIR, 'synthetic_val')

# ============ 数据生成参数 ============
# 训练集和验证集数量
NUM_TRAIN_IMAGES = 8000  # 训练集（可以增加到 6000-8000）
NUM_VAL_IMAGES = 2000    # 验证集

# 每张图片的目标数量（针对动画风格复杂场景）
MIN_OBJECTS_PER_IMAGE = 10  # 最少 10 个
MAX_OBJECTS_PER_IMAGE = 16  # 最多 16 个（模拟密集场景）

# ============ 目标尺寸配置 ============
# 小/中/大目标的具体尺寸范围（像素）
SMALL_SIZE_RANGE = (35, 65)      # 小目标：35-65px
MEDIUM_SIZE_RANGE = (65, 100)    # 中目标：65-100px
LARGE_SIZE_RANGE = (100, 130)    # 大目标：100-130px

# 小/中/大目标的比例
SMALL_OBJECT_RATIO = 0.7   # 70%
MEDIUM_OBJECT_RATIO = 0.2  # 20%
LARGE_OBJECT_RATIO = 0.1   # 10%

# object_3 比例增强（针对 object_3 性能差）
CLASS_WEIGHTS = {
    'object_1': 1.10,
    'object_2': 1.10,
    'object_3': 1.15,  # object_3 增加 30% 样本
    'object_4': 1.2,
    'object_5': 1.15,
    'object_6': 0.85,
}

# ============ 合成参数（针对动画风格）============
# 背景处理
BG_RESIZE_MODE = 'random_crop'  # 随机裁剪背景
TARGET_SIZE = (640, 640)        # 目标图像尺寸

# 目标放置策略
ALLOW_OVERLAP = True           # 允许部分重叠（模拟密集场景）
MAX_OVERLAP_RATIO = 0.3        # 最大重叠 30%
MIN_EDGE_DISTANCE = 5         # 距离边缘至少 5px

# 区域偏好（针对你的场景：左1/4简单，右3/4复杂）
LEFT_ZONE_OBJECT_RATIO = 0.35   # 30% 目标在左侧
RIGHT_ZONE_OBJECT_RATIO = 0.65  # 70% 目标在右侧

# ============ 边缘处理（动画风格关键）============
# 卡通风格：清晰边缘，不要羽化
USE_EDGE_FEATHER = False       # 不使用边缘羽化
ADD_OUTLINE = True             # 添加轮廓线
OUTLINE_WIDTH = 1              # 轮廓线宽度
OUTLINE_COLOR = (0, 0, 0)      # 黑色轮廓

# ============ 颜色调整（动画风格）============
# 提高饱和度和亮度
SATURATION_RANGE = (1.1, 1.2)  # 饱和度 +10% 到 +20%
BRIGHTNESS_RANGE = (1.0, 1.1)  # 亮度 +0% 到 +10%
CONTRAST_RANGE = (0.9, 1.1)    # 对比度微调

# ============ 阴影处理（动画风格）============
# 简化阴影（扁平投影）
ADD_SHADOW = False
SHADOW_TYPE = 'simple'         # 简单阴影（vs 'realistic'）
SHADOW_OPACITY = 0.3           # 阴影透明度 30%
SHADOW_BLUR = 5                # 轻微模糊
SHADOW_OFFSET = (5, 5)         # 阴影偏移

# ============ 数据增强 ============
AUGMENTATION = {
    'horizontal_flip': 0.5,      # 50% 水平翻转
    'rotation': (-15, 15),       # 旋转 ±15度
    'scale': (0.8, 1.2),         # 缩放 80%-120%
    
    # 颜色增强（动画风格更激进）
    'hue_shift': (-20, 20),      # 色调偏移 ±20度
    'saturation': (0.8, 1.4),    # 饱和度 80%-140%
    'brightness': (0.8, 1.2),    # 亮度 80%-120%
    
    # 不使用模糊（保持清晰）
    'blur': False,
    
    # 锐化增强（强化边缘）
    'sharpen': 0.3,              # 30% 概率锐化
}

# ============ 质量控制 ============
# 最小目标面积（避免过小目标）
MIN_OBJECT_AREA = 400          # 至少 20×20 像素

# 边界检查
ENSURE_VALID_BBOX = True       # 确保边界框有效

# ============ 多进程配置 ============
NUM_WORKERS = 8                # 并行进程数

# ============ 随机种子 ============
RANDOM_SEED = 42

print("=" * 60)
print("数据合成配置加载完成")
print("=" * 60)
print(f"使用自定义背景: {USE_CUSTOM_BACKGROUNDS}")
if USE_CUSTOM_BACKGROUNDS:
    print(f"背景目录: {CUSTOM_BG_DIR}")
print(f"训练图像: {NUM_TRAIN_IMAGES} 张")
print(f"验证图像: {NUM_VAL_IMAGES} 张")
print(f"每图目标数: {MIN_OBJECTS_PER_IMAGE}-{MAX_OBJECTS_PER_IMAGE} 个")
print(f"小目标比例: {SMALL_OBJECT_RATIO * 100:.0f}%")
print(f"object_3 权重: {CLASS_WEIGHTS['object_3']}x")
print("=" * 60)
