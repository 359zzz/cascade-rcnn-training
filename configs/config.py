
import os

class Config:
    # ============ 路径配置 ============
    BASE_DIR = "/home/unitree/my_opencv"
    
    # 数据路径
    RAW_TEMPLATES_DIR = os.path.join(BASE_DIR, "data/raw_templates")
    PROCESSED_TEMPLATES_DIR = os.path.join(BASE_DIR, "data/processed_templates")
    BACKGROUNDS_DIR = os.path.join(BASE_DIR, "data/backgrounds")
    SYNTHETIC_TRAIN_DIR = os.path.join(BASE_DIR, "data/synthetic_train")
    SYNTHETIC_VAL_DIR = os.path.join(BASE_DIR, "data/synthetic_val")
    ANNOTATIONS_DIR = os.path.join(BASE_DIR, "data/annotations")
    
    # 日志路径
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    
    # ============ 模板配置 ============
    # 模板类别（根据你的6张模板命名）
    CLASSES = [
        "object_1",  # 例如：耙子
        "object_2",  # 第二个目标
        "object_3",
        "object_4",
        "object_5",
        "object_6"
    ]
    
    NUM_CLASSES = len(CLASSES)
    
    # ============ 数据生成配置 ============
    # 多尺度配置
    SCALES = [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0]
    
    # 旋转角度（度）
    ROTATION_ANGLES = [-15, -10, -5, 0, 5, 10, 15]
    
    # 合成数据数量
    NUM_TRAIN_SAMPLES = 4000  # 训练集样本数
    NUM_VAL_SAMPLES = 1000    # 验证集样本数
    
    # 每张图片最多放置的目标数量
    MIN_OBJECTS_PER_IMAGE = 1
    MAX_OBJECTS_PER_IMAGE = 5
    
    # ============ 图像尺寸配置 ============
    TARGET_IMAGE_SIZE = (640, 640)  # 生成的训练图像尺寸
    
    # ============ 数据增强配置 ============
    # 光照变化范围
    BRIGHTNESS_RANGE = (0.7, 1.3)
    CONTRAST_RANGE = (0.8, 1.2)
    SATURATION_RANGE = (0.8, 1.2)
    HUE_RANGE = (-10, 10)
    
    # 模糊程度（小目标使用）
    BLUR_KERNEL_RANGE = (3, 7)
    BLUR_PROBABILITY = 0.3
    
    # 噪声
    NOISE_VAR_RANGE = (0, 10)
    
    # ============ 背景去除配置 ============
    # U2-Net模型
    REMBG_MODEL = "u2net"  # 可选: u2net, u2netp, u2net_human_seg
    
    # 边缘平滑参数
    EDGE_BLUR_KERNEL = (5, 5)
    EDGE_DILATE_KERNEL = (3, 3)
    
    # ============ 数据合成配置 ============
    # 泊松融合模式
    BLEND_MODE = "normal"  # normal, mixed, monochrome
    
    # 小目标阈值（像素）
    SMALL_OBJECT_THRESHOLD = 50 * 50  # 小于50x50认为是小目标
    
    # ============ COCO标注配置 ============
    ANNOTATION_FORMAT = "coco"  # 输出格式
    
    # ============ 调试配置 ============
    DEBUG = True  # 是否保存中间结果用于调试
    VISUALIZE_SAMPLES = 10  # 可视化前N个样本

# 实例化配置
cfg = Config()
