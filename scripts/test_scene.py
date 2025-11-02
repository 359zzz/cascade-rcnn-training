#!/usr/bin/env python3
"""
测试新训练模型 - 单张图片
"""
import cv2
import numpy as np
from mmdet.apis import init_detector, inference_detector
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# 配置
# ============================================================
CONFIG = 'configs/cascade_rcnn_r50_fpn_8gpu.py'
CHECKPOINT = 'work_dirs/cascade_rcnn_r50_fpn_8gpu/epoch_24.pth'
IMAGE_PATH = '/home/unitree/my_opencv/scene.png'
OUTPUT_PATH = 'output/scene_result.png'
CONF_THRESHOLD = 0.3

# 类别和颜色
CLASSES = ['object_1', 'object_2', 'object_3', 
           'object_4', 'object_5', 'object_6']

COLORS = [
    (220, 20, 60),   # object_1: 红色
    (119, 11, 32),   # object_2: 深红
    (0, 0, 142),     # object_3: 蓝色
    (0, 0, 230),     # object_4: 亮蓝
    (106, 0, 228),   # object_5: 紫色
    (0, 60, 100),    # object_6: 深青
]

# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 70)
    print("🎯 测试新训练的 Cascade R-CNN 模型")
    print("=" * 70)
    print(f"📷 输入图片: {IMAGE_PATH}")
    print(f"🤖 模型配置: {CONFIG}")
    print(f"💾 模型权重: {CHECKPOINT}")
    print(f"🎨 输出图片: {OUTPUT_PATH}")
    print(f"🎚️  置信度阈值: {CONF_THRESHOLD}")
    print("=" * 70)
    
    # 1. 加载模型
    print("\n⏳ 正在加载模型...")
    model = init_detector(CONFIG, CHECKPOINT, device='cuda:0')
    print("✅ 模型加载成功！")
    
    # 2. 读取图片
    print(f"\n📖 正在读取图片: {IMAGE_PATH}")
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"❌ 错误：无法读取图片 {IMAGE_PATH}")
        return
    
    h, w = image.shape[:2]
    print(f"✅ 图片尺寸: {w} x {h}")
    
    # 3. 执行推理
    print("\n🔍 正在进行目标检测...")
    result = inference_detector(model, image)
    print("✅ 检测完成！")
    
    # 4. 处理检测结果
    print("\n" + "=" * 70)
    print("📊 检测结果统计")
    print("=" * 70)
    
    detection_stats = {cls: [] for cls in CLASSES}
    total_detections = 0
    
    # 创建可视化图像
    vis_image = image.copy()
    
    for class_id, (class_name, color) in enumerate(zip(CLASSES, COLORS)):
        # 获取该类别的检测结果
        bboxes = result.pred_instances.bboxes[
            result.pred_instances.labels == class_id
        ].cpu().numpy()
        scores = result.pred_instances.scores[
            result.pred_instances.labels == class_id
        ].cpu().numpy()
        
        # 过滤低置信度
        mask = scores >= CONF_THRESHOLD
        bboxes = bboxes[mask]
        scores = scores[mask]
        
        num_detections = len(bboxes)
        total_detections += num_detections
        
        if num_detections > 0:
            avg_conf = scores.mean()
            max_conf = scores.max()
            min_conf = scores.min()
            detection_stats[class_name] = {
                'count': num_detections,
                'avg_conf': avg_conf,
                'max_conf': max_conf,
                'min_conf': min_conf
            }
            
            print(f"\n🔹 {class_name}:")
            print(f"   数量: {num_detections}")
            print(f"   平均置信度: {avg_conf:.3f}")
            print(f"   最高置信度: {max_conf:.3f}")
            print(f"   最低置信度: {min_conf:.3f}")
            
            # 绘制边界框
            for bbox, score in zip(bboxes, scores):
                x1, y1, x2, y2 = bbox.astype(int)
                
                # 绘制矩形（使用 BGR 颜色）
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                
                # 绘制标签背景
                label = f'{class_name}: {score:.2f}'
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(vis_image, (x1, y1 - text_h - baseline - 5), 
                            (x1 + text_w, y1), color, -1)
                
                # 绘制标签文字
                cv2.putText(vis_image, label, (x1, y1 - baseline - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            print(f"\n🔹 {class_name}: 未检测到")
    
    # 5. 总结
    print("\n" + "=" * 70)
    print(f"✅ 总检测数: {total_detections} 个目标")
    print("=" * 70)
    
    if total_detections == 0:
        print("\n⚠️  警告：未检测到任何目标！")
        print("   可能原因：")
        print("   1. 图片中没有训练的目标类别")
        print("   2. 置信度阈值设置过高")
        print("   3. 目标太小或遮挡严重")
    
    # 6. 保存结果
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(OUTPUT_PATH, vis_image)
    print(f"\n💾 结果已保存到: {OUTPUT_PATH}")
    
    # 7. 显示图片信息
    print("\n" + "=" * 70)
    print("📈 类别分布:")
    print("=" * 70)
    if total_detections > 0:
        for class_name in CLASSES:
            if detection_stats[class_name]:
                count = detection_stats[class_name]['count']
                percentage = (count / total_detections) * 100
                bar = '█' * int(percentage / 5)
                print(f"{class_name:<12} {count:>3} ({percentage:>5.1f}%) {bar}")
    
    print("\n" + "=" * 70)
    print("✅ 测试完成！")
    print("=" * 70)
    print(f"\n💡 提示：")
    print(f"   - 查看结果图片: {OUTPUT_PATH}")
    print(f"   - 修改置信度阈值: 编辑脚本中的 CONF_THRESHOLD = {CONF_THRESHOLD}")
    print(f"   - 使用其他图片: 编辑脚本中的 IMAGE_PATH")
    print("=" * 70)

if __name__ == '__main__':
    main()
